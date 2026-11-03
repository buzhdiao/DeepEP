// 头文件保护:保证本头文件只被包含一次
#pragma once

// NCCL 主库头文件
#include <nccl.h>
// NCCL 设备端 API 头文件(设备通信句柄、对称内存窗口、GIN 原语等)
#include <nccl_device.h>

// GPU 全局屏障(跨 rank、跨 SM 的内核级同步)实现
#include <deep_ep/comm/barrier.cuh>
// 编译期通用常量(如 kNumTMAAlignmentBytes 等)
#include <deep_ep/common/compiled.cuh>
// 静态断言与设备端断言工具
#include <deep_ep/common/exception.cuh>
// 编译期数学工具(对齐、encode_decode_positive、advance_ptr 等)
#include <deep_ep/common/math.cuh>
// PTX 内联汇编封装(TMA、mbarrier、cp.async、red_add、deduplicate 等)
#include <deep_ep/common/ptx.cuh>
// EP token 布局(TokenLayout/BufferLayout,定义 token 在缓冲区中的排布)
#include <deep_ep/layout/ep/token.cuh>
// EP 工作区布局(各计数器/信号在工作区中的偏移)
#include <deep_ep/layout/ep/workspace.cuh>


// DeepEP 的 EP(expert parallel)实现命名空间
namespace deep_ep::ep {

// 单层 dispatch 内核:按 top-k 专家把本地 token 分发到目标 rank。
// 节点内走 NVLink(TMA 直接写远端对称内存),跨节点走 RDMA(GIN put,先落到发送缓冲)。
// 模板参数:
template <bool kIsScaleupNVLink,          // 本 rank 的所有对端是否都可通过 NVLink 直达(全 LSA 视角)
          bool kDoCPUSync,                // 是否把计数结果同步写到映射的主机内存(供 CPU 读取)
          bool kReuseSlotIndices,         // 是否复用上一次 dispatch 记录的槽位索引
          int kNumSMs,                    // GPU 的 SM 数量(用于按 SM 划分 channel)
          int kNumNotifyWarps, int kNumDispatchWarps,  // 每个 SM 中两种角色的 warp 数:统计通知 / 数据分发
          int kNumRanks,                  // 全局 rank 总数
          int kNumHiddenBytes, int kNumSFPacks,        // 每 token 的 hidden 数据字节数;SF(scale factor)pack 数
          int kNumMaxTokensPerRank,       // 每个 rank 最多容纳的 token 数
          int kNumExperts, int kNumTopk, int kExpertAlignment,  // 专家总数、每 token top-k 数、专家计数对齐粒度
          int kNumQPs, int64_t kNumTimeoutCycles,       // NCCL GIN 队列对(QP)数量、轮询超时周期数
          // 以下为派生模板参数(由上面的参数推导,一般无需显式指定)
          int kNumNotifyThreads = kNumNotifyWarps * 32,    // notify 部分线程总数
          int kNumDispatchThreads = kNumDispatchWarps * 32,  // dispatch 部分线程总数
          int kNumThreads = kNumNotifyThreads + kNumDispatchThreads,  // 每 block 总线程数
          // 团队类型:NVLink 场景用 LSA(节点内),否则用 World(全局,可跨 RDMA)
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
// 启动边界:每 SM 固定 kNumThreads 线程,且每 SM 至多驻留 1 个 block
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,  // 输入:hidden 数据、SF、top-k 专家索引、top-k 权重
    int* cumulative_local_expert_recv_stats,   // 累计统计:本地各专家收到的 token 数(可空)
    int* psum_num_recv_tokens_per_scaleup_rank,  // 输出:各源 rank 接收 token 数的前缀和
    int* psum_num_recv_tokens_per_expert,      // 输出:各专家接收 token 数的前缀和(排他)
    int* num_unaligned_recv_tokens_per_expert, // 输出:各专家未对齐的接收 token 数(可空)
    int* dst_buffer_slot_idx,                  // 槽位索引表:token × topk → 接收缓冲槽位(复用/记录用)
    const int num_tokens,                      // 本地 token 数
    const int sf_token_stride, const int sf_hidden_stride,  // SF 张量的 token 维与 hidden 维步长
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* buffer,  // NCCL 设备通信句柄、对称窗口与数据缓冲
    void* workspace, void* mapped_host_workspace,  // GPU 工作区与映射主机工作区
    const int rank_idx                         // 本 rank 的全局编号
) {
    // 每个 rank 分摊的专家数
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    // 静态断言:专家数必须能被 rank 数整除
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    // 静态断言:notify warp 数必须是 4 的倍数(warpgroup 粒度)
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");

    // 通用工具
    // 取当前 block 的 SM 编号与块内线程编号
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    // 取当前 warp 在 block 内的编号以及 lane 编号(0~31)
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();

    // 工作区(workspace)
    // 按 GPU 全局工作区布局解析 workspace 指针(1 个 scale-out 组、kNumRanks 个 rank)
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    // 按同样布局解析映射到设备可访问的主机工作区(用于 CPU 同步读取计数)
    const auto host_workspace_layout = layout::EPWorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);

    // 本内核使用固定大小的动态共享内存(不使用静态共享内存)
    // 声明动态共享内存首指针,按 TMA 对齐要求对齐
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // notify 部分共享内存字节数:kNumRanks 个 rank 计数 + kNumExperts 个 expert 计数,向上对齐到 notify 线程数
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    // 静态断言:notify 共享内存大小必须满足 TMA 对齐
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % kNumTMAAlignmentBytes == 0, "Invalid TMA alignment");

    // 具名屏障(named barrier)的索引
    constexpr int kNotifyBarrierIndex = 1;

    // GIN 句柄
    // 我们把每个 warp 视为一个 "channel"
    // 根据 SM 编号与 warp 编号计算该 warp 使用的 QP 编号与共享模式(notify warp 不参与)
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(
        sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
    // 构造 NCCL GIN 上下文:封装对称内存窗口上的 put/get/red 等远程内存访问原语
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // 屏障:无需 TMA store 冲刷,也无需前导的 grid 级同步
    // 内核入口处做一次全 rank、全 SM 的同步(使用 kDispatchTag0 标记)
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);

    // 按 warp 分工:两种角色
    // 角色 1:notify warp —— 统计本 rank 发出的 token 分布(rank/expert 计数)并广播给所有对端
    if (warp_idx < kNumNotifyWarps) {
        // 分配共享内存
        // notify 共享内存按 int 元素对齐后的元素数
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        // 共享内存首段用于存放 rank 计数与 expert 计数
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // 清零初始计数
        // 注意:若要改变各 warp 角色的顺序,请小心 `thread_idx` 的使用
        // rank 计数在前,expert 计数紧随其后
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        // 以跨步方式由全部 notify 线程并行清零共享内存计数
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        // 具名屏障:等待所有 notify 线程完成清零
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // 在共享内存上做原子加
        // 静态断言:top-k 数不能超过 warp 的 lane 数
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        // 计算本 warp 覆盖全局 token 的起始编号(warp × SM 展开的跨步划分)
        const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;
        // 所有 notify warp 以固定步长遍历本地每个 token
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // 专家选择不允许重复
            // 注意:这里不加断言,因为断言开销太大
            // lane i 读取该 token 的第 i 个专家编号(超出 top-k 的 lane 置 -1)
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            // 有效专家编号:在共享内存的 expert 计数上块内原子加 1
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // rank 选择需要在此处去重(同一 token 可能发给同一 rank 的多个专家)
            // 由专家编号换算目标 rank 编号
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            // warp 内按值去重,只有首个命中的 lane 才原子加 1
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        // 具名屏障:等待所有计数完成
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // 做全网格(grid)级归约
        #pragma unroll
        // 每个 notify 线程负责若干 rank/expert 计数:高位 32 位编码为 1,低位为本 SM 的计数
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            // 打包计数:(1<<32) | 本 SM 计数,作为对全局工作区的一次原子加
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            // 全局内存 red.add:各 SM 累加,高 32 位自然成为已到达的 SM 数
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // 剩余工作由 SM 0 完成
        if (sm_idx == 0) {
            // 归约所有 SM 的计数
            // 等待所有 SM 到达
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                // 带超时的轮询,直到该计数器的高 32 位等于 SM 总数(无条件进入轮询)
                comm::timeout_while<kNumTimeoutCycles>(true, [=](const bool& is_last_check) {
                    // 易失性读取归约工作区中的 64 位计数值
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    // 高 32 位 == kNumSMs 表示所有 SM 的加法都已到达
                    if ((status >> 32) == kNumSMs) {
                        // 写入共享内存
                        // 若走 RDMA,还需写入发送缓冲(工作区)
                        // 低 32 位即全部 SM 的计数总和,做正数编码
                        const auto encoded =
                            math::encode_decode_positive(static_cast<int>(status & 0xffffffffll));
                        // 编码后的总数写回共享内存
                        rank_expert_count[i] = encoded;
                        // 非 NVLink 场景(RDMA),再写入全局工作区的发送缓冲,供后续 bulk 发送
                        if constexpr (not kIsScaleupNVLink)
                            workspace_layout.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;

                        // 为下次使用清零
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    // 最后一轮检查时打印超时诊断信息
                    if (is_last_check) {
                        printf("DeepEP notify (GPU reduction) timeout, rank: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               rank_idx, kNumRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            // 具名屏障:等待归约结果全部就绪
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // TODO: 为进一步优化,可以把 rank 计数与 expert 计数融合
            // 向各对端发出 scale-up rank 计数写操作
            // notify 线程分工遍历所有目标 rank
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                // rank 计数
                // 本 rank 在 rank 计数区中的槽位地址
                const auto dst_rank_counter =
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;
                // GIN put_value:把本 rank 发出的 token 数写到目标 rank i 的对应槽位(聚合请求降低开销)
                gin.put_value<team_t>(dst_rank_counter, static_cast<int64_t>(rank_count[i]), i,
                                      ncclGinOptFlagsAggregateRequests);
            }
            // warp 内同步
            __syncwarp();

            // 向各对端发出 scale-up expert 计数写操作
            if constexpr (kIsScaleupNVLink) {
                // NVLink 逐元素拷贝
                // 不使用 TMA,因为共享内存与全局内存的元素类型不同(此处计数为 int64)
                for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads) {
                    // 把全局专家编号换算成目标 rank 上对应槽位的下标
                    const auto idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank);
                    // GIN put_value:把本 rank 各专家的 token 计数写到目标专家所属的 rank
                    gin.put_value<team_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), i / kNumExpertsPerRank);
                }
            } else {
                // RDMA 整块(bulk)拷贝
                // 逐 rank 发送本 rank 全部专家计数的连续段(已先写入工作区发送缓冲)
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                    // 源:工作区中本 rank 准备好的专家计数段
                    const auto src_ptr = workspace_layout.get_scaleup_expert_count_ptr<true>() + kNumExpertsPerRank * i;
                    // 目的:目标 rank 上本 rank 对应的专家计数段
                    const auto dst_ptr = workspace_layout.get_scaleup_expert_count_ptr<false>() + kNumExpertsPerRank * rank_idx;
                    // GIN put:一次性整块发送 kNumExpertsPerRank 个 8 字节计数
                    gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);
                }
            }

            // 这一步是必要的,因为等待的结果会重写共享内存
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // 等待各对端的 rank 计数与 expert 计数到达
            // 记录起始时钟,作为本轮等待的超时基准
            const auto start_clock = clock64();
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                // 带超时的轮询:线程 i 对应第 i 个 rank 计数(或 expert 计数)槽位
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    // 注意:全局内存的类型是 64 位
                    // 易失性读取对端写来的 64 位计数值
                    const auto count = static_cast<int>(
                        ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + i));
                    // 对正数编码进行解码
                    const auto decoded = math::encode_decode_positive(count);
                    // 已就绪(有效计数)则清零槽位并写入共享内存
                    if (math::is_decoded_positive_ready(decoded)) {
                        workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] = 0;
                        rank_expert_count[i] = decoded;
                        return true;
                    }

                    // 最后一轮检查时打印超时诊断信息
                    if (is_last_check)
                        printf("DeepEP notify timeout, rank: %d, thread: %d, count: %d\n", rank_idx, i, decoded);
                    return false;
                }, start_clock);
            }
            // 具名屏障:等待所有计数到达完成
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // 归约专家计数并更新统计
            // 逐个本地专家,把来自所有源 rank 的计数求和
            for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {
                // 本地专家 i 的接收总数
                int sum = 0;
                #pragma unroll
                // 遍历所有源 rank 的 expert 计数矩阵列
                for (int j = 0; j < kNumRanks; ++ j)
                    sum += expert_count[j * kNumExpertsPerRank + i];

                // 在对齐之前先写入未对齐计数
                if (num_unaligned_recv_tokens_per_expert != nullptr)
                    num_unaligned_recv_tokens_per_expert[i] = sum;

                // 对齐到 kExpertAlignment 后写回共享内存(供前缀和使用)
                expert_count[i] = math::align(sum, kExpertAlignment);

                // 更新统计计数器
                // 累加本专家累计收到的 token 数
                if (cumulative_local_expert_recv_stats != nullptr)
                    atomicAdd(cumulative_local_expert_recv_stats + i, sum);
            }
            // 具名屏障:等待归约完成
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // 写主机工作区
            // 若需要 CPU 同步:把 rank/expert 计数编码后写入映射主机内存
            if constexpr (kDoCPUSync) {
                for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads) {
                    host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] =
                        math::encode_decode_positive(rank_expert_count[i]);
                }
                // warp 内同步
                __syncwarp();
            }

            // 由各 warp 完成前缀和
            // 注意:可以用 `cub::BlockScan` 更快实现,但它太重了
            // 单 warp 前缀和:count 为输入,out 为输出,n 为元素数,is_exclusive=1 表示排他(输出先补 0)
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                // 上一轮 32 个元素的累计和
                int psum = 0;
                #pragma unroll
                // 每 32 个元素(一个 warp 一轮)迭代
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    // 本 lane 处理的全局下标
                    const auto idx = i * 32 + lane_idx;
                    // 对应输入的下标(排他时整体左移一位)
                    const auto mem_idx = idx - is_exclusive;
                    // 越界 lane 读 0
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    // warp 内包含式前缀和再加上上一轮累计
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // 写入全局内存
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // 用最后一个 lane 的值更新 `psum`
                    // lane 31 的包含和即本轮总和
                    psum = ptx::exchange(sum, 31);
                }
            };
            // warp 0:对 rank 计数做包含式前缀和
            if (warp_idx == 0) {
                // 包含式前缀和
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);
            } else if (warp_idx == 1) {
                // 为后续展开(expanding)做排他式前缀和
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    } else {
        // 角色 2:dispatch warp —— 负责 token 数据本身的搬运(TMA 加载 + NVLink/RDMA 发送)
        // 本 warp 在 SM 内的 dispatch 通道编号
        const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;

        // 缓冲区布局
        // TokenLayout 计算 hidden/SF/topk 索引等各段在共享内存中的排布与总字节数
        const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
        // 为每个 dispatch warp 分配一份共享内存 TMA 缓冲(位于 notify 共享内存之后)
        const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumDispatchWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(dispatch_warp_idx).get_token_buffer(0);
        // 接收缓冲:每个 rank 一份,容量为每 rank 最大 token 数(位于全局 buffer)
        auto recv_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
        // 发送缓冲:仅一份(RDMA 场景的本地暂存),紧跟在接收缓冲之后
        auto send_buffer = layout::BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());
        // 接收缓冲限定到本 rank 的部分
        recv_buffer = recv_buffer.get_rank_buffer(rank_idx);

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

        // 遍历所有 token
        // 本 warp 处理的 token 起始编号(warp × SM 展开的跨步划分)
        const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
        // 跨步 = 全部 dispatch warp × SM 数
        const auto token_stride = kNumDispatchWarps * kNumSMs;
        for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
            // 以 64 位计算 token 下标,避免乘法溢出
            const auto token_i64_idx = static_cast<int64_t>(token_idx);

            // 等待 TMA store 到达
            // 确保上一轮对共享内存 TMA 缓冲的复用安全
            ptx::tma_store_wait();
            // warp 内同步
            __syncwarp();

            // 发起数据(hidden)TMA
            // elected 线程发起 1D TMA:全局 x 中该 token 的 hidden 数据 → 共享内存
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            // warp 内同步
            __syncwarp();

            // 发起 SF 的 TMA 或 cp.async
            // 仅当存在 scale factor pack 时才拷贝
            if constexpr (kNumSFPacks > 0) {
                // 静态断言:SF 元素类型必须 4 字节对齐
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                // 全局源地址:该 token 的 SF 起始(按 token 步长偏移)
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                // 共享内存目的地址:TMA 缓冲中的 SF 段
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                // 完整轮数:每轮由 32 个 lane 各拷贝一个 pack
                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    // cp.async.ca:按 hidden 维步长逐 pack 拷入共享内存
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                // 不足整轮的尾部 pack 由低编号 lane 补齐
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                // cp.async 完成后向 mbarrier 发信号(ExpectTx/arrive 组合)
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                // warp 内同步
                __syncwarp();
            }

            // 加载 top-k 索引与权重
            // 静态断言:lane 数足够加载 top-k 索引
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            // 本 lane 对应专家的目标 rank(-1 表示无效)
            int stored_dst_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                // 经只读缓存读取该 token 的第 i 个专家编号
                const auto dst_expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
                // 由专家编号换算目标 rank;无效则置 -1
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
                // 把专家索引写入共享内存缓冲(随 TMA 一起发出)
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                // 若有权重,同样缓存 top-k 权重
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
            }
            // warp 内同步
            __syncwarp();

            // 添加源元数据(rank 编号与 token 编号)
            // 请确保在此之后不要再写共享内存中的 TMA 缓冲
            // elected 线程写入源 token 全局编号(接收端借此还原位置)
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            // TMA store 前的内存栅栏,保证共享内存写入对 TMA 可见
            ptx::tma_store_fence();
            // warp 内同步
            __syncwarp();

            // 去重 rank 并分配槽位
            // 本 lane 的目标发送槽位(-1 表示无需发送)
            int stored_dst_slot_idx = -1;
            if constexpr (kReuseSlotIndices) {
                // 复用模式:读取上次记录的全局槽位并换算成本地接收缓冲槽位
                if (lane_idx < kNumTopk)
                    stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);
                stored_dst_slot_idx = stored_dst_slot_idx >= 0 ?
                    (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank) : -1;
            } else {
                // warp 内按目标 rank 去重,代表 lane 原子递增全局发送计数器分配槽位
                if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
                // 把"全局槽位"(rank 基址 + 槽号)记录到索引表,供下次复用与 combine 阶段还原
                if (lane_idx < kNumTopk) {
                    const auto value = stored_dst_slot_idx >= 0 ?
                        rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;
                }
            }
            // warp 内同步
            __syncwarp();

            // 等待 TMA 加载到达
            // 注意:此 arrive 必须在 `ptx::cp_async_mbarrier_arrive` 之后
            // elected 线程设置期望字节数并等待相位翻转(hidden + SF + 元数据全部就绪)
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            // warp 内同步
            __syncwarp();

            // TMA store 到发送缓冲
            // 本 token 在发送缓冲(RDMA 暂存)中的目的地址
            auto send_buffer_ptr = send_buffer.get_token_buffer(token_idx).get_base_ptr();
            if constexpr (not kIsScaleupNVLink) {
                // 仅 RDMA 场景需要:先把整 token 暂存到本地发送缓冲
                if (ptx::elect_one_sync())
                    ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                // 提交 TMA store 组
                ptx::tma_store_commit();
                // warp 内同步
                __syncwarp();
            }

            // 发起 NVLink 的 TMA store
            // 静态断言:top-k 数不超过 lane 数
            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");
            // 计算目标 rank 上接收缓冲槽位的对称内存地址;无目标槽位时为空指针
            const auto dst_ptr = stored_dst_slot_idx >= 0 ?
                gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
                nullptr;
            // 有目标时:从共享内存直接 TMA 写入远端(节点内)对称内存
            if (dst_ptr != nullptr)
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            // 提交 TMA store 组
            ptx::tma_store_commit();
            // warp 内同步
            __syncwarp();

            // 发起 RDMA put
            if constexpr (not kIsScaleupNVLink) {
                // 等待发送缓冲的 store 到达
                // 只等第 1 组(发送缓冲)完成,保证 RDMA 读到的数据完整
                ptx::tma_store_wait<1>();
                // warp 内同步
                __syncwarp();

                // 注意:我们应跳过 NVLink 可直达的 rank
                // 目标不在本节点(对称指针为空)时,经 GIN put 从发送缓冲发出
                if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
                    gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                    send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
                }
                // warp 内同步
                __syncwarp();
            }
        }
    }

    // 收尾屏障:确保所有数据都已到达对端
    // 全 rank、全 SM 同步(使用 kDispatchTag1 标记,含 TMA 冲刷)
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag1, true, true, false>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);

    // 触发(copy epilogue)后续内核(programmatic dependent launch)
    cudaTriggerProgrammaticLaunchCompletion();

    // 清理原子计数器
    // 静态断言:线程数足够覆盖 rank 数
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    // 不复用槽位时,由 SM 0 的前若干线程清零各 rank 的原子发送计数器
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::ep
