// 头文件保护:保证本头文件只被包含一次
#pragma once

// 编译期通用常量(如 kNumTMAAlignmentBytes 等)
#include <deep_ep/common/compiled.cuh>
// 编译期数学工具(ceil_div、advance_ptr 等)
#include <deep_ep/common/math.cuh>
// PTX 内联汇编封装(TMA、mbarrier、st_bulk 等)
#include <deep_ep/common/ptx.cuh>
// EP token 布局(TokenLayout/BufferLayout)
#include <deep_ep/layout/ep/token.cuh>
// EP 工作区布局(链表尾部指针等)
#include <deep_ep/layout/ep/workspace.cuh>


// DeepEP 的 EP(expert parallel)实现命名空间
namespace deep_ep::ep {

// dispatch 的拷贝尾随(copy epilogue)内核:在 dispatch 主内核完成后(经 PDL 依赖同步),
// 把接收缓冲中的 token 整理拷贝到最终输出张量(含展开 expand / 不展开两种模式),
// 同时搬运 SF、top-k 权重与源元数据,并维护链表与专家对齐的零填充。
// 模板参数:
template <bool kDoExpand, bool kCachedMode, bool kDoZeroPadding,
          // 注意:这里的 channel 概念只针对 scale-out rank
          int kNumSMs, int kNumChannels, int kNumWarps,   // SM 数、channel 数(scale-out 用)、每 SM warp 数
          int kNumScaleoutRanks, int kNumScaleupRanks,    // scale-out(跨节点)rank 数与 scale-up(节点内)rank 数
          int kNumHiddenBytes, int kNumSFPacks,           // 每 token 的 hidden 字节数;SF pack 数
          int kNumMaxTokensPerRank,                       // 每个 rank 最多容纳的 token 数
          int kNumExperts, int kNumTopk, int kExpertAlignment,  // 专家总数、top-k 数、专家计数对齐粒度
          // 以下为派生模板参数
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,          // 全局 rank 总数
          int kNumThreads = kNumWarps * 32,                              // 每 block 线程数
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),  // 每 channel 最大 token 数
          // 仅当存在多个 scale-out rank 且非缓存模式时,才需要构建链表
          bool kDoCreateLinkedList = (kNumScaleoutRanks > 1 and not kCachedMode)>
// 启动边界:每 SM 固定 kNumThreads 线程,且每 SM 至多驻留 1 个 block
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_copy_epilogue_impl(void* buffer, void* workspace,
                            int* psum_num_recv_tokens_per_scaleup_rank,   // 各源 rank 接收 token 数的前缀和
                            int* psum_num_recv_tokens_per_expert,         // 各专家接收 token 数的(原子累加用)计数
                            void* recv_x, sf_pack_t* recv_sf,             // 输出:hidden 数据与 SF 张量
                            topk_idx_t* recv_topk_idx, float* recv_topk_weights,  // 输出:top-k 专家索引与权重
                            int* recv_src_metadata,                       // 输出:源元数据(源 token 下标等)
                            int* channel_linked_list,                     // 链表数组(记录接收顺序)
                            int* num_unaligned_recv_tokens_per_expert,    // 各专家未对齐接收 token 数
                            int num_recv_tokens,                          // 接收 token 总数(特殊值表示需从 GPU 读取)
                            const int recv_sf_token_stride, const int recv_sf_hidden_stride,  // 输出 SF 的步长
                            const int scaleout_rank_idx, const int scaleup_rank_idx) {  // 本 rank 的两级编号
    // 通用工具
    // 取当前 block 的 SM 编号与块内线程编号
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    // 取当前 warp 在 block 内的编号以及 lane 编号
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    // 全局 warp 编号(warp × SM 展开),用于跨步处理 token
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

    // 用于 top-k 索引变换
    // 每个 rank 分摊的专家数
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    // 本 rank 的全局编号
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;
    // 本 rank 负责的专家编号区间 [expert_start_idx, expert_end_idx)
    const auto expert_start_idx = kNumExpertsPerRank * rank_idx, expert_end_idx = kNumExpertsPerRank * (rank_idx + 1);

    // 缓冲区布局
    // 声明动态共享内存首指针,按 TMA 对齐要求对齐
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // TokenLayout 计算 hidden/SF/topk 等各段在共享内存中的排布与总字节数
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    // 为每个 warp 分配一份共享内存 TMA 缓冲,取本 warp 的第 0 个 token 缓冲
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    // 全局接收缓冲:每个 scale-up rank 一份,总容量为 scale-out rank 数 × 每 rank 最大 token 数
    const auto scaleup_buffer = layout::BufferLayout<false>(token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);

    // 初始化 TMA
    // mbarrier 相位计数器,等待 TMA 完成时翻转
    ptx::arrival_phase phase = 0;
    // 取本 warp TMA 缓冲的 mbarrier 地址(共享内存)
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    // 由 elected 线程把 mbarrier 初始化为期望到达次数 1
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // warp 内同步,确保初始化可见
    __syncwarp();

    // 阻塞直到主 dispatch 内核已完成且所有数据可见
    // 注意:这里使用了 PDL(programmatic dependent launch),请勿使用 `__ldg`
    // PDL 依赖同步:等待上游 dispatch 内核的 cudaTriggerProgrammaticLaunchCompletion 及数据可见
    cudaGridDependencySynchronize();

    // 无 CPU 同步的场景:接收 token 总数应从 GPU 张量中读取
    // 特殊哨兵值:表示 num_recv_tokens 未在主机侧确定,取前缀和的最后一项
    if (num_recv_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1];

    // 需要维护"当前 rank"的相关状态
    // 当前处理的源 rank 编号及其接收 token 前缀和缓存
    int current_rank_idx = -1, stored_psum_num_recv_tokens;
    // 当前源 rank 的 token 区间 [current_rank_start, current_rank_end)
    int current_rank_start = 0, current_rank_end = 0;
    #pragma unroll
    // 所有 warp 以固定步长遍历每个接收 token
    for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {
        // 计算 token 在缓冲区中的下标
        // 若 i 超出当前源 rank 区间,则推进到下一个(些)rank
        while (i >= current_rank_end) {
            // 进入下一个源 rank
            current_rank_idx += 1;
            // 设备断言:rank 编号不能越界
            EP_DEVICE_ASSERT(current_rank_idx < kNumScaleupRanks);
            // 该 rank 的前缀和由哪个 lane 负责缓存
            const auto stored_lane_idx = current_rank_idx % 32;
            // 到达新的一组 32 个 rank 时,一次性预取 32 个 rank 的前缀和到各 lane
            if (stored_lane_idx == 0 and current_rank_idx + lane_idx < kNumScaleupRanks)
                stored_psum_num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[current_rank_idx + lane_idx];
            // 更新当前 rank 的区间起点与终点(从负责 lane 交换读取)
            current_rank_start = current_rank_end;
            current_rank_end = ptx::exchange(stored_psum_num_recv_tokens, stored_lane_idx);
        }
        // 取该 token 在接收缓冲中的地址(按源 rank 与 rank 内偏移定位)
        const auto buffer_token = scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);

        // 等待缓冲释放
        // 确保上一轮对共享内存 TMA 缓冲的复用安全
        ptx::tma_store_wait();
        // warp 内同步
        __syncwarp();

        // 发起 TMA 加载
        // 包含全部内容:数据、SF、top-k 元数据
        // elected 线程发起:接收缓冲 token → 共享内存,并设置期望字节数
        if (ptx::elect_one_sync()) {
            ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                             mbarrier_ptr, tma_buffer.get_num_bytes<false>());
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, tma_buffer.get_num_bytes<false>());
        }
        // warp 内同步
        __syncwarp();

        // 单独加载目标专家索引,以容忍 TMA 加载延迟
        // 静态断言:top-k 数不超过 lane 数
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        // 本 lane 对应的 top-k 专家编号(-1 表示无效)
        int dst_expert_idx = -1;
        // 直接从接收缓冲(全局内存)读 top-k 索引,先于 TMA 数据就绪
        if (lane_idx < kNumTopk)
            dst_expert_idx = buffer_token.get_topk_idx_ptr()[lane_idx];
        // warp 内同步
        __syncwarp();

        // 校验目标专家索引;非展开模式下写出
        // 判断该专家是否属于本 rank 负责的区间
        const auto in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;
        // 汇总各 lane 的命中情况,找出首个命中 lane(master)
        const auto master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));
        // 命中则换算成本 rank 内的专家编号,否则置 -1
        dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;
        // 设备断言:warp 内该专家最多命中一次(去重后要么唯一要么全 -1)
        EP_DEVICE_ASSERT(ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1);
        // 非展开模式:写出调整后的局部专家索引(不在本 rank 的写 -1)
        if (not kDoExpand and lane_idx < kNumTopk)
            recv_topk_idx[i * kNumTopk + lane_idx] = static_cast<topk_idx_t>(dst_expert_idx);
        // warp 内同步
        __syncwarp();

        // 计算输出张量中的目标下标
        // 每个接收 token 的源元数据步长:源下标 + 辅助下标 + topk 个目标下标
        constexpr int kMetadataStride = 2 + kNumTopk;
        // 输出张量中的目标下标(-1 表示本 token 不落在本 rank)
        int dst_tensor_idx = -1;
        if (not kDoExpand and ptx::elect_one_sync()) {
            // 非展开模式:输出与接收一一对应,目标下标即 i
            dst_tensor_idx = i;
        } else if (kDoExpand and kCachedMode and lane_idx < kNumTopk) {
            // 缓存展开模式:从元数据中读取预先计算好的目标下标
            dst_tensor_idx = recv_src_metadata[i * kMetadataStride + 2 + lane_idx];
        } else if (kDoExpand and not kCachedMode and dst_expert_idx >= 0) {
            // 普通展开模式:原子递增专家计数,领取该专家内的落位下标
            dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
        }
        // warp 内同步
        __syncwarp();

        // 等待 TMA 到达
        // elected 线程以相位翻转方式等待整 token 加载完成
        if (ptx::elect_one_sync())
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        // warp 内同步
        __syncwarp();

        // 维护链表
        // 需要构建链表时:把本 token 的接收序号写入链表对应位置
        if constexpr (kDoCreateLinkedList) {
            // 由 elected 线程按 master lane 记录的链表下标写入
            if (ptx::elect_one_sync())
                channel_linked_list[tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx]] = i;
            // warp 内同步
            __syncwarp();
        }

        // 为数据(hidden)发起 TMA store
        // 展开模式:有落位的 lane 各自写出;非展开模式:仅 elected 线程写一次
        if (kDoExpand ? (dst_tensor_idx >= 0) : ptx::elect_one_sync()) {
            // TMA store:hidden 段从共享内存写入输出张量对应 token 位置
            ptx::tma_store_1d(math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                              tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
            // 提交 TMA store 组
            ptx::tma_store_commit();
        }
        // warp 内同步
        __syncwarp();

        // 存储 SF
        // 仅当存在 SF pack 时处理
        if constexpr (kNumSFPacks > 0) {
            // 完整轮数:每轮 32 个 lane 各处理一个 pack
            constexpr auto kNumFullIters = kNumSFPacks / 32;
            // 尾部不足一轮的 pack 是否由本 lane 处理
            const bool do_last_iter = (kNumSFPacks % 32 != 0) and (kNumFullIters * 32 + lane_idx < kNumSFPacks);
            // 静态断言:SF 元素类型必须 4 字节对齐
            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");

            // 加载到寄存器
            // 共享内存中的 SF 段源地址
            const auto smem_src_ptr = tma_buffer.get_sf_ptr();
            // 每 lane 暂存其负责的 SF pack
            sf_pack_t reg_src[kNumFullIters + 1];
            #pragma unroll
            for (int k = 0; k < kNumFullIters; ++ k)
                reg_src[k] = smem_src_ptr[k * 32 + lane_idx];
            // 尾部 pack 同样读入寄存器
            if (do_last_iter)
                reg_src[kNumFullIters] = smem_src_ptr[kNumFullIters * 32 + lane_idx];

            // 准备步长
            // 输出 SF 的 token 维步长(64 位,防溢出)
            const auto recv_sf_token_stride_i64 = static_cast<int64_t>(recv_sf_token_stride);
            // 输出 SF 的 hidden 维步长
            const auto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);

            // 遍历所有有效下标并写入输出缓冲
            // 展开模式:各 lane 有效落位的掩码;非展开模式:仅 lane 0(掩码为 1)
            auto mask = kDoExpand ? ptx::gather(dst_tensor_idx >= 0) : 1;
            // 逐个有效 lane 处理其落位
            while (mask) {
                // 最低有效位的 lane 编号
                const int valid_lane_idx = __ffs(mask) - 1;
                // 从负责 lane 交换读取其目标下标,计算该 token 的 SF 输出基址
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(recv_sf,
                    ptx::exchange(dst_tensor_idx, valid_lane_idx) * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k)
                    // 按 hidden 维步长逐 pack 写出
                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[k];
                // 尾部 pack 写出
                if (do_last_iter)
                    gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[kNumFullIters];
                // 清除已处理的 lane 位
                mask ^= 1 << valid_lane_idx;
            }
        }

        // 存储 top-k 权重
        // 展开模式:每个落位 lane 写自己的权重
        if (kDoExpand and recv_topk_weights != nullptr and dst_tensor_idx >= 0) {
            recv_topk_weights[dst_tensor_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        } else if (not kDoExpand and recv_topk_weights != nullptr and lane_idx < kNumTopk) {
            // 反向计算时权重是可选的
            // 非展开模式:按接收序号逐 top-k 写出
            recv_topk_weights[i * kNumTopk + lane_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        }
        // warp 内同步
        __syncwarp();

        // 写源 token 下标(缓存模式下复用元数据,跳过)
        // 并且:
        //   - 非混合(hybrid)模式:写入源 scale-up 对端 rank 下标与 master top-k lane 下标
        //   - 混合模式:写入槽位下标与 master top-k lane 下标
        if constexpr (not kCachedMode) {
            // 由 elected 线程写入公共元数据
            if (ptx::elect_one_sync()) {
                // 源 token 全局下标
                recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();
                if constexpr (kNumScaleoutRanks == 1) {
                    // 单 scale-out rank:辅助下标 = 源 rank 编号 × top-k + master lane
                    recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;
                } else {
                    // 多 scale-out rank:辅助下标 = rank 内槽位 × top-k + master lane
                    recv_src_metadata[i * kMetadataStride + 1] = (i - current_rank_start) * kNumTopk + master_src_topk_idx;
                }
            }
            // warp 内同步
            __syncwarp();

            // 写归约所需的源下标
            // 展开模式:每个落位 lane 记录其输出下标(供 combine 归还原位)
            if (kDoExpand and lane_idx < kNumTopk)
                recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;
            // warp 内同步
            __syncwarp();
        }
    }

    // 维护链表的结尾
    // 也可以理解为一次性写入尾部哨兵
    if constexpr (kDoCreateLinkedList) {
        // 每个 lane 分摊的 scale-up rank 数
        constexpr int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32);
        // 解析工作区布局(读取各 channel 链表尾指针)
        const auto workspace_layout = layout::EPWorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
        // 各 warp 分摊遍历所有 channel
        for (int i = global_warp_idx; i < kNumChannels; i += kNumSMs * kNumWarps) {
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                // 该 lane 负责的 scale-up rank 编号(最后一组须防越界)
                if (const auto k = j * 32 + lane_idx; j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks) {
                    // 在该 channel/rank 链表的尾部写入 -1 作为结束标记
                    channel_linked_list[
                        *workspace_layout.get_channel_scaleup_tail_ptr(i, k)
                    ] = -1;

                    // 为 combine 阶段的使用做清理
                    // 尾指针清零,供下次 dispatch 使用
                    *workspace_layout.get_channel_scaleup_tail_ptr(i, k) = 0;
                }
            }
            // warp 内同步
            __syncwarp();
        }
    }

    // 零填充:清除展开输出中专家之间的对齐空隙
    // 仅在需要零填充且为展开模式时执行
    if constexpr (kDoZeroPadding and kDoExpand) {
        // 等待主循环最后一次 TMA store 完成
        ptx::tma_store_wait();
        // warp 内同步
        __syncwarp();

        // 把共享内存中 TMA 缓冲的 hidden 区域清零
        // 批量(store)写 0,作为零填充的模板数据
        ptx::st_bulk<kNumHiddenBytes>(tma_buffer.get_hidden_ptr());
        // TMA store 前的内存栅栏
        ptx::tma_store_fence();
        // warp 内同步
        __syncwarp();

        // 并行读取所有专家的未对齐计数
        // 每个 lane 分摊的专家数
        constexpr int kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32);
        // 每 lane 缓存其负责专家的未对齐接收 token 数
        int num_experts_per_lane[kNumExpertsPerLane];
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerLane; ++ i) {
            // 该 lane 第 i 份负责的专家编号
            const int expert_idx = i * 32 + lane_idx;
            // 越界 lane 记 0
            num_experts_per_lane[i] = expert_idx < kNumExpertsPerRank ?
                num_unaligned_recv_tokens_per_expert[expert_idx] : 0;
        }

        // 用单个 while 循环遍历所有填充 token
        // 当前波次(wave)、波次前累计填充数、波次前累计张量下标、本 warp 的填充序号
        int wave_idx = 0, wave_pad_psum = 0, wave_tensor_psum = 0, pad_idx = global_warp_idx;
        while (true) {
            // 跳转到正确的位置
            // 本 warp 填充 token 的目标下标
            int dst_tensor_idx;
            // 逐波次(wave,每波 32 个专家)寻找本填充 token 落在哪一波
            while (wave_idx < kNumExpertsPerLane) {
                // 确定当前波次各 lane 的专家计数
                int wave_num_experts_per_lane;
                #pragma unroll
                // 从缓存的各份计数中选出当前波次对应的一份
                for (int i = 0; i < kNumExpertsPerLane; ++ i)
                    wave_num_experts_per_lane = i == wave_idx ? num_experts_per_lane[i] : wave_num_experts_per_lane;
                // 当前波次各 lane 需要的填充数
                int wave_num_pads_per_lane = 0;
                if (wave_idx * 32 + lane_idx < kNumExpertsPerRank) {
                    // 填充数 = 对齐粒度 - (计数 mod 对齐粒度);恰好整除时为 0
                    wave_num_pads_per_lane = kExpertAlignment - (wave_num_experts_per_lane % kExpertAlignment);
                    wave_num_pads_per_lane = wave_num_pads_per_lane == kExpertAlignment ? 0 : wave_num_pads_per_lane;
                }
                // warp 内求和:当前波次的填充总数
                int wave_num_pads = ptx::reduce_add(wave_num_pads_per_lane);

                // 检查是否命中当前波次
                if (pad_idx < wave_pad_psum + wave_num_pads) {
                    // 本 warp 填充 token 在当前波次内的局部序号
                    const int local_pad_idx = pad_idx - wave_pad_psum;
                    // 各 lane 填充数的包含式前缀和,用于定位归属 lane
                    const int pad_psum = ptx::warp_inclusive_sum(wave_num_pads_per_lane, lane_idx);
                    // 首个前缀和覆盖该序号的 lane 即"拥有者"
                    const int owner_lane_idx = __ffs(__ballot_sync(0xffffffff, pad_psum > local_pad_idx)) - 1;
                    // 目标下标 = 波次前张量累计 + (该 lane 的专家数+填充数的排他和 + 专家数 + 局部序号 - 归属 lane 之前的填充)
                    dst_tensor_idx = wave_tensor_psum + ptx::exchange(
                        ptx::warp_exclusive_sum(wave_num_experts_per_lane + wave_num_pads_per_lane, lane_idx) +
                        wave_num_experts_per_lane + local_pad_idx - (pad_psum - wave_num_pads_per_lane),
                        owner_lane_idx);
                    break;
                }

                // 移动到下一个波次
                wave_idx ++;
                // 累计当前波次之前的填充总数
                wave_pad_psum += wave_num_pads;
                // 累计当前波次之前的张量下标总数(专家 token + 填充)
                wave_tensor_psum += ptx::reduce_add(wave_num_experts_per_lane + wave_num_pads_per_lane);
            }
            // 所有波次都已越过:填充完成,退出
            if (wave_idx >= kNumExpertsPerLane)
                break;

            // 经 TMA store 写入零数据
            // elected 线程:把清零的共享内存 hidden 模板写到填充位置
            if (ptx::elect_one_sync()) {
                ptx::tma_store_1d(math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                                  tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
                ptx::tma_store_commit();
            }
            // warp 内同步
            __syncwarp();

            // 权重清零
            // 有权重张量时,填充位置的权重写 0
            if (recv_topk_weights != nullptr and ptx::elect_one_sync())
                recv_topk_weights[dst_tensor_idx] = 0.0f;
            // warp 内同步
            __syncwarp();

            // SF 清零
            // 存在 SF pack 时,填充位置的 SF 写 0
            if constexpr (kNumSFPacks > 0) {
                // 输出 SF 的 token 维步长
                const auto recv_sf_token_stride_i64 = static_cast<int64_t>(recv_sf_token_stride);
                // 输出 SF 的 hidden 维步长
                const auto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);
                // 全零 SF pack 常量
                constexpr sf_pack_t zero_sf_pack = {0};
                // 该填充 token 的 SF 输出基址
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(recv_sf,
                    dst_tensor_idx * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));
                // 完整轮数:每轮 32 个 lane 各写一个 pack
                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k)
                    // 按 hidden 维步长逐 pack 写零
                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = zero_sf_pack;
                // 尾部不足整轮的 pack 由低编号 lane 补零
                if constexpr (kNumSFPacks % 32 != 0) {
                    if (kNumFullIters * 32 + lane_idx < kNumSFPacks)
                        gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] = zero_sf_pack;
                }
            }
            // warp 内同步
            __syncwarp();

            // 尝试填充下一个
            // 跨步到本 warp 的下一个填充 token
            pad_idx += kNumSMs * kNumWarps;
        }
    }
}

}  // namespace deep_ep::ep
