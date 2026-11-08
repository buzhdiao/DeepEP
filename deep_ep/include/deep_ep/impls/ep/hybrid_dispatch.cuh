// 头文件保护:保证本头文件只被包含一次
#pragma once

// GPU 全局屏障(跨 rank、跨 SM 的内核级同步)实现
#include <deep_ep/comm/barrier.cuh>
// 编译期通用常量(如 kNumTMAAlignmentBytes 等)
#include <deep_ep/common/compiled.cuh>
// 静态断言与设备端断言工具
#include <deep_ep/common/exception.cuh>
// 编译期数学工具(对齐、pack2/unpack2、encode_decode_positive、advance_ptr 等)
#include <deep_ep/common/math.cuh>
// PTX 内联汇编封装(TMA、mbarrier、cp.async、red_add、deduplicate、gather 等)
#include <deep_ep/common/ptx.cuh>
// EP token 布局(TokenLayout/BufferLayout)
#include <deep_ep/layout/ep/token.cuh>
// EP 工作区布局(各计数器/信号在工作区中的偏移)
#include <deep_ep/layout/ep/workspace.cuh>


// DeepEP 的 EP(expert parallel)实现命名空间
namespace deep_ep::ep {

// 混合式(hybrid)dispatch 内核:两级架构 = scale-out(跨节点,RDMA/NCCL GIN)+ scale-up(节点内,NVLink 对称内存)。
// 三种 warp 角色协作:notify(统计并广播计数)、scale-out(经 RDMA 发往远端节点)、forward(从远端接收后经 NVLink 转发到本节点内目标 rank)。
// 模板参数:
template <bool kDoCPUSync,                 // 是否把计数结果同步写到映射的主机内存(供 CPU 读取)
          bool kReuseSlotIndices,          // 是否复用上一次记录的槽位索引(免重复去重分配)
          int kNumSMs,                     // GPU 的 SM 数量(用于按 SM 划分 channel)
          int kNumNotifyWarps, int kNumScaleoutWarps, int kNumForwardWarps,  // 每个 SM 中三种角色的 warp 数:统计通知 / scale-out 发送 / 转发
          int kNumScaleoutRanks, int kNumScaleupRanks,   // scale-out(跨节点)rank 数与 scale-up(节点内)rank 数
          int kNumHiddenBytes, int kNumSFPacks,          // 每 token 的 hidden 数据字节数;SF(scale factor)pack 数
          int kNumMaxTokensPerRank,        // 每个 rank 最多容纳的 token 数
          int kNumExperts, int kNumTopk, int kExpertAlignment,  // 专家总数、每 token top-k 数、专家计数对齐粒度
          int kNumQPs, int64_t kNumTimeoutCycles,        // NCCL GIN 队列对(QP)数量、轮询超时周期数
          // 以下为派生模板参数(由上面的参数推导,一般无需显式指定)
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),  // 每个 lane 分摊的 scale-up rank 数
          int kNumChannelsPerSM = kNumScaleoutWarps,     // 每 SM 的 channel 数(一个 scale-out/forward warp 即一个 channel)
          int kNumChannels = kNumScaleoutWarps * kNumSMs,  // 全局 channel 总数
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),  // 每 channel 最大 token 数
          int kScaleoutUpdateInterval = 6,               // scale-out 尾指针的批量发布间隔(减少 RDMA 原子操作)
          int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,  // forward 阶段每次处理的槽位块大小
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,    // 全局 rank 总数
          int kNumNotifyThreads = kNumNotifyWarps * 32,  // notify 部分线程总数
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,  // scale-out 发送线程总数
          int kNumForwardThreads = kNumForwardWarps * 32,  // forward 线程总数
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>  // 每 block 总线程数
// 启动边界:每 SM 固定 kNumThreads 线程,且每 SM 至多驻留 1 个 block
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,  // 输入:hidden 数据、SF、top-k 专家索引、top-k 权重
    int* cumulative_local_expert_recv_stats,   // 累计统计:本地各专家收到的 token 数(可空)
    int* psum_num_recv_tokens_per_scaleup_rank,  // 输出:各 scale-up rank 接收 token 数的前缀和
    int* psum_num_recv_tokens_per_expert,      // 输出:各本地专家接收 token 数的(排他)前缀和
    int* num_unaligned_recv_tokens_per_expert, // 输出:各专家未对齐的接收 token 数(可空)
    int* dst_buffer_slot_idx,                  // 槽位索引表(channel × rank × 槽位 × topk),供槽位复用
    int* token_metadata_at_forward,            // forward 阶段记录的每 token 元数据(供 copy epilogue 还原)
    const int num_tokens,                      // 本地 token 数
    const int sf_token_stride, const int sf_hidden_stride,  // SF 张量的 token 维与 hidden 维步长
    // TODO(NCCL): 参数太多,计划优化?
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,  // NCCL 设备通信句柄与对称内存窗口
    void* buffer,                              // token 数据缓冲(scale-up 接收 / scale-out 收发)
    void* workspace, void* mapped_host_workspace,  // GPU 工作区与映射主机工作区
    const int scaleout_rank_idx, const int scaleup_rank_idx) {  // 本 rank 的两级编号
    // 每个(全局)rank 分摊的专家数
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    // 每个 scale-out 节点(组)分摊的专家数
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    // 静态断言:专家数必须能被 scale-up rank 数整除
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");
    // 静态断言:notify warp 数必须是 4 的倍数(warpgroup 粒度)
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    // 静态断言:scale-out warp 数必须等于 forward warp 数(channel 一一对应)
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp size");

    // 通用工具
    // 注意:一个 warp 即一个 channel(不同 channel 可能共享 QP)
    // 取当前 block 的 SM 编号与块内线程编号
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    // 取当前 warp 在 block 内的编号以及 lane 编号(0~31)
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    // 计算全局 rank 编号 = scale-out 编号 × 节点内 rank 数 + scale-up 编号
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;

    // 工作区(workspace)
    // 按 GPU 全局工作区布局解析 workspace 指针(屏障信号、各 rank/expert 计数器等)
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
    // 按同样布局解析映射到设备可访问的主机工作区(用于 CPU 同步读取计数)
    const auto host_workspace_layout = layout::EPWorkspaceLayout(mapped_host_workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

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

    // NCCL GIN 句柄
    // 每个 warp 即一个 channel
    // 根据 SM 编号、warp 编号计算该 warp 使用的 QP 编号与共享模式(notify warp 不参与)
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, (kNumNotifyWarps > 0)>(
        sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, warp_idx < kNumNotifyWarps);
    // 构造 NCCL GIN 上下文:封装对称内存窗口上的 put/get/red 等远程内存访问原语
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // scale-out 子团队与 scale-up 子团队的全局并行屏障(内核入口同步,tag0)
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, false, true>(
        gin, workspace_layout.get_barrier_signals(), scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // scale-out warp 与 forward warp 在整个过程中共用的黄金布局(golden layout)
    // TokenLayout 计算 hidden/SF/topk 等各段在共享内存中的排布与总字节数
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    // 为 (scale-out + forward) 个 warp 各分配一份共享内存 TMA 缓冲(位于 notify 共享内存之后),
    // 取本 warp 的第 0 个 token 缓冲
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumScaleoutWarps + kNumForwardWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);

    // 各类缓冲区
    // scale-up 接收缓冲:节点内每个 scale-up rank 一份,容量为 scale-out rank 数 × 每 rank 最大 token 数
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);
    // scale-out 发送缓冲:仅一份(发往远端节点前的暂存),紧跟在 scale-up 缓冲之后
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, 1, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());
    // scale-out 接收缓冲:每个 scale-out rank 一份,按 channel 划分容量,紧跟其后
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks, kNumChannels * kNumMaxTokensPerChannel, scaleout_send_buffer.get_buffer_end_ptr());

    // 为 scale-out 与 forward warp 初始化 TMA
    // mbarrier 相位计数器,等待 TMA 完成时翻转
    ptx::arrival_phase phase = 0;
    // 取本 warp TMA 缓冲的 mbarrier 地址(共享内存)
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    // notify 之外的每个 warp 由 elected 线程把 mbarrier 初始化为期望到达次数 1
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // warp 内同步,确保 mbarrier 初始化可见
    __syncwarp();

    // 不同的 warp 角色
    // 角色 1:notify warp —— 统计本 rank 收到的 token 分布(rank/expert 计数)并逐级通知所有对端
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
        // 计算本 warp 覆盖全局 token 的起始编号(跨 SM 展开)
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;
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
            // 由专家编号换算目标全局 rank 编号
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
                // 带超时的轮询,直到该计数器的高 32 位等于 SM 总数
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    // 易失性读取归约工作区中的 64 位计数值
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    // 高 32 位 == kNumSMs 表示所有 SM 的加法都已到达
                    if ((status >> 32) == kNumSMs) {
                        // 编码并写入发送缓冲(工作区)
                        // 低 32 位即全部 SM 的计数总和,做正数编码后写入 rank/expert 计数区
                        workspace_layout.get_scaleout_rank_expert_count_ptr<true>()[i] =
                            math::encode_decode_positive<int>(status & 0xffffffffll);

                        // 为下次使用清零
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    // 最后一轮检查时打印超时诊断信息
                    if (is_last_check) {
                        printf("DeepEP hybrid notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            // 具名屏障:等待归约结果全部就绪
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // 向 scale-out 对端发起写操作
            // 静态断言:复用槽位,或 scale-out rank 数不超过 notify 线程数
            EP_STATIC_ASSERT(kReuseSlotIndices or kNumScaleoutRanks <= kNumNotifyThreads,
                             "kNumScaleoutRanks must be less than kNumNotifyThreads");
            // 线程 i 负责把本 rank 的计数发给第 i 个 scale-out 对端
            if (thread_idx < kNumScaleoutRanks) {
                // 目标 scale-out rank 编号即线程编号
                const auto dst_scaleout_rank_idx = thread_idx;
                // GIN put(Rail 团队视角):把本 rank 的各 scale-up rank 计数写到对端(聚合请求降低开销)
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,
                    ncclGinOptFlagsAggregateRequests);
                // 同样把本节点承担的专家计数写到对端
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx);
            }
            // warp 内同步
            __syncwarp();

            // 从 scale-out 对端读取并归约元数据的工具函数
            // 注意:这是正确的,因为 RDMA 操作的最小写粒度为 1024 字节(单个整数的写是原子的)
            // get_ptr_func(j) 给出第 j 个 scale-out 对端上的待读地址;is_expert_reduction 仅用于日志
            const auto recv_and_reduce = [=](const auto& get_ptr_func, const bool& is_expert_reduction = false) -> int {
                // 累加和
                int count = 0;
                #pragma unroll
                // 遍历所有 scale-out 对端
                for (int j = 0; j < kNumScaleoutRanks; ++ j) {
                    // 取第 j 个对端上的计数指针
                    const auto ptr = get_ptr_func(j);
                    // 解码后的计数值
                    int decoded;
                    // 带超时地等待对端写入有效(正数编码)计数
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){
                        // 以 acquire(系统级)语义读取并做正数编码解码
                        decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));
                        // 已就绪(有效计数)则返回
                        if (math::is_decoded_positive_ready(decoded))
                            return true;

                        // 最后一轮检查打印超时诊断
                        if (is_last_check) {
                            printf("DeepEP hybrid notify (scale-out %s reduction) timeout, "
                                   "scale-out: %d, scale-up: %d, "
                                   "thread: %d, wait scale-out: %d, decoded: %d\n",
                                   is_expert_reduction ? "expert" : "rank",
                                   scaleout_rank_idx, scaleup_rank_idx, thread_idx, j,
                                   decoded);
                        }
                        return false;
                    });

                    // 累加并为下次使用清零
                    count += decoded, *ptr = 0;
                }
                // 返回从所有 scale-out 对端归约出的计数
                return count;
            };

            // 把 rank 级计数写入所有 scale-up 对端
            #pragma unroll
            // notify 线程分工遍历所有 scale-up rank
            for (int i = thread_idx; i < kNumScaleupRanks; i += kNumNotifyThreads) {
                // 等待 scale-out 对端到达并归约
                // 归约所有 scale-out 对端发给 scale-up rank i 的 token 数
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
                });

                // 写入远端 scale-up 对端
                // 打包:(kNumScaleupRanks << 32) | 计数,高位表示还需到达的 scale-up 对端数
                const int64_t counter = (static_cast<int64_t>(kNumScaleupRanks) << 32ll) | count;
                // GIN put_value(LSA 团队,NVLink 节点内):写到 scale-up rank i 的计数槽
                gin.put_value<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
                    counter, i);
            }
            // warp 内同步
            __syncwarp();

            // 对所有 scale-up 对端的专家级计数做原子加
            #pragma unroll
            // 遍历本节点承担的每个专家
            for (int i = thread_idx; i < kNumExpertsPerScaleout; i += kNumNotifyThreads) {
                // 等待 scale-out 对端到达并归约
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                }, true);

                // 写入远端 scale-up 对端
                // 打包:(1 << 32) | 计数
                const int64_t counter = (1ll << 32ll) | count;
                // 由专家编号计算其所属的 scale-up rank 及 rank 内专家编号
                const auto dst_scaleup_rank_idx = i / kNumExpertsPerRank;
                const auto expert_idx_in_dst_rank = i % kNumExpertsPerRank;
                // GIN red.add(带 release,LSA 视角):原子累加到对端专家计数
                gin.red_add_rel<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
                    counter, dst_scaleup_rank_idx);
            }
            // 上面有共享内存读取,必须加屏障
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // 注意:从现在起,`rank` 与 `expert` 的规模变为本地规模
            // expert 计数指针改指向局部 expert 计数段(kNumExpertsPerRank 个)
            expert_count = rank_expert_count + kNumScaleupRanks;

            // 等待本地计数就绪
            // 注意:这里只关心按 scale-up 对端的前缀和(供后续 epilogue 使用),而不是所有 rank
            // 静态断言:notify 线程必须能覆盖 scale-up rank 数 + 局部专家数
            EP_STATIC_ASSERT(kNumNotifyWarps == 0 or kNumScaleupRanks + kNumExpertsPerRank <= kNumNotifyWarps * 32,
                             "Insufficient notify threads");
            // 线程 i 等待 scale-up rank/expert 计数(i < kNumScaleupRanks 为 rank 计数,否则为 expert 计数)
            comm::timeout_while<kNumTimeoutCycles>(thread_idx < kNumScaleupRanks + kNumExpertsPerRank,
                [&](const bool& is_last_check) {
                // 易失性读取本线程对应的计数槽(高位为已到达的 scale-up 对端数)
                const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + thread_idx);
                // 所有 scale-up 对端均已到达
                if ((status >> 32ull) == kNumScaleupRanks) {
                    // 清理 GPU 工作区并写入主机工作区
                    // 低 32 位即汇总计数
                    const auto count = static_cast<int>(status & 0xffffffffll);
                    // 按需对齐:rank 计数对齐到 1,专家计数对齐到 kExpertAlignment
                    const auto aligned_count = math::align<int>(
                        count, thread_idx < kNumScaleupRanks ? 1 : kExpertAlignment);

                    // 清零 GPU 工作区计数槽,供下次使用
                    workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] = 0;
                    // 若需要 CPU 同步:把对齐后的计数做正数编码写入映射主机内存
                    if constexpr (kDoCPUSync) {
                        host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] =
                            math::encode_decode_positive(aligned_count);
                    }

                    // 更新统计计数器
                    // 累加本专家累计收到的 token 数(不含 rank 计数槽)
                    if (cumulative_local_expert_recv_stats != nullptr and thread_idx >= kNumScaleupRanks)
                        atomicAdd(cumulative_local_expert_recv_stats + (thread_idx - kNumScaleupRanks), count);

                    // 在对齐之前先写入未对齐计数
                    if (num_unaligned_recv_tokens_per_expert != nullptr and thread_idx >= kNumScaleupRanks)
                        num_unaligned_recv_tokens_per_expert[thread_idx - kNumScaleupRanks] = count;

                    // 保存对齐后计数,供后续前缀和计算使用
                    rank_expert_count[thread_idx] = aligned_count;
                    return true;
                }

                // 最后一轮检查打印超时诊断
                if (is_last_check) {
                    printf("DeepEP hybrid notify (scale-up reduction) timeout,"
                           "scale-out: %d/%d, scale-up: %d/%d, "
                           "thread: %d, status: %d | %d, expected: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                           static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumScaleupRanks);
                }
                return false;
            });
            // 具名屏障:等待本地计数全部就绪
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // 由第一个 SM 的 warp 完成前缀和
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
            // warp 0:对 scale-up rank 计数做包含式前缀和
            if (warp_idx == 0) {
                // 包含式前缀和
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumScaleupRanks, 0);
            } else if (warp_idx == 1) {
                // 为后续展开(expanding)做排他式前缀和
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        // 角色 2:scale-out warp —— 把 token 经 RDMA/NCCL 发往远端节点的 scale-out rank
        // 本 warp 在 SM 内的 scale-out 通道编号
        const int scaleout_warp_idx = warp_idx - kNumNotifyWarps;
        // 全局 channel 编号(SM 编号 × 每 SM 通道数 + SM 内通道编号)
        const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;
        // scale-out 接收缓冲限定为本 rank 对应的部分
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        // 再限定为本 channel 对应的部分(各 channel 独立槽位)
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);

        // channel 元数据维护
        // 静态断言:scale-out rank 数不能超过一个 warp 的 lane 数
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid number of scale-out ranks");
        // 每个 lane 维护一个 scale-out rank 的已发送尾指针与上次已通知尾指针
        int stored_scaleout_tail = 0, stored_old_scaleout_tail = 0;
        // 批量更新 signaled tail:仅当落后超过 kScaleoutUpdateInterval 或收尾时才发布,减少 RDMA 原子操作次数
        const auto update_scaleout_tail = [&](const bool& finish_flag = false) {
            // lane i 负责第 i 个 scale-out rank;达到更新间隔或带结束标志时才更新
            if (lane_idx < kNumScaleoutRanks and
                (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag)) {
                // 打包(结束标志, 当前尾指针)为一个 64 位值
                const auto signaled_tail = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
                // 对端上本 channel 的 signaled tail 槽位地址
                const auto ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx);
                // 打包上次已通知的旧值(结束标志为 0),用于求差量
                const auto old_signaled_tail = math::pack2<int, int64_t>(0, stored_old_scaleout_tail);

                // 注意:对本地 rank 而言 "release" 作用域需为 `sys`(可能涉及 NVLink,所以不能是 `gpu`)
                // 对 RDMA 请求,"release" 语义由 "atomic" 保证
                // GIN red_add_rel(Rail 视角):把(新尾 - 旧尾)作为 64 位差量原子加到对端
                gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
                // 记录本次已通知的尾指针
                stored_old_scaleout_tail = stored_scaleout_tail;
            }
            // warp 内同步
            __syncwarp();
        };

        // 预取下一个 token
        // 把 token_idx 指定的 token 提前加载进共享内存 TMA 缓冲(hidden 用 TMA,SF 用 cp.async)
        const auto preload_next_token = [&](const int& token_idx) {
            // 越界则直接返回
            if (token_idx >= num_tokens)
                return;

            // 发起 TMA 加载
            // 以 64 位计算 token 下标,避免乘法溢出
            const auto token_i64_idx = static_cast<int64_t>(token_idx);
            // 由 elected 线程发起 1D TMA:全局 x 中该 token 的 hidden 数据 → 共享内存
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            // warp 内同步
            __syncwarp();

            // 发起 SF 的 `cp.async`
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
        };

        // 遍历所有 token
        // 预取本 channel 的第一个 token
        preload_next_token(channel_idx);
        // 以 channel 总数为跨步,每个 channel 处理自己的 token 子集
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {
            // 加载 top-k 索引与权重
            // 静态断言:lane 数足够加载 top-k 索引
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            // 本 lane 对应专家的目标 scale-out rank(-1 表示无效)
            int stored_dst_scaleout_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                // 经只读缓存读取该 token 的第 i 个专家编号
                const auto dst_expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
                // 由专家编号换算目标 scale-out rank;无效则置 -1
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerScaleout : -1;
                // 把专家索引写入共享内存缓冲(随 TMA 一起发出)
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                // 若有权重,同样缓存 top-k 权重
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
            }
            // warp 内同步
            __syncwarp();

            // 添加源元数据(rank 编号与 token 编号)
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
            // 从目标 rank 对应 lane 读取其当前尾指针(同一 rank 的多个专家共享同一槽,即整 token 只发一次)
            const auto stored_old_slot_idx = ptx::exchange(
                stored_scaleout_tail, stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            // warp 内按目标 rank 去重,仅代表 lane 分配槽位
            if (ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and stored_dst_scaleout_rank_idx >= 0)
                stored_dst_slot_idx = stored_old_slot_idx;

            // 更新 scale-out 尾指针
            // 汇总每个目标 scale-out rank 是否被本 token 命中的位掩码
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);
            // 本 lane 若命中自己负责的 rank,则该 rank 的尾指针加 1
            stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // 等待 TMA 到达并向发送缓冲发起 TMA store
            if (ptx::elect_one_sync()) {
                // 设置期望字节数并到达 mbarrier,等待 hidden 数据加载完成
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                // 以相位翻转方式等待 mbarrier 完成
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                // 因此如果没有任何 rank 需要走 RDMA,就跳过发送缓冲的 store
                // 若目标掩码中除本节点外还有其他 scale-out rank,才需要写入发送缓冲
                if (scaleout_rank_mask ^ (1 << scaleout_rank_idx)) {
                    // TMA store:整 token(hidden+SF+topk+元数据)从共享内存写入发送缓冲
                    ptx::tma_store_1d(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                                      tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                }
            }
            // warp 内同步
            __syncwarp();

            // 本地 rank 可以被旁路(直接写入本 rank 接收缓冲,不走网络)
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx) {
                // TMA store:直接写入本 rank 的 scale-out 接收缓冲对应槽位
                ptx::tma_store_1d(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            }
            // 提交并等待本 token 的所有 TMA store 完成
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            // warp 内同步
            __syncwarp();

            // 预取下一个 token(与 IBGDA 请求的发出重叠)
            preload_next_token(token_idx + kNumChannels);

            // 发出 IBGDA(RDMA)请求
            // 有槽位且目标不是本 scale-out rank:经 GIN put 发往远端节点
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
                // 源:scale-out 发送缓冲;目的:远端节点上本 rank 的接收缓冲槽位;聚合请求以降低开销
                gin.put<ncclTeamTagRail>(
                        scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                        tma_buffer.get_num_bytes<false>(),
                        stored_dst_scaleout_rank_idx,
                        ncclGinOptFlagsAggregateRequests);
            }
            // warp 内同步
            __syncwarp();

            // 发布 scale-out 尾指针更新
            update_scaleout_tail();
        }

        // 冲刷(发布)尚未发布的尾指针
        update_scaleout_tail(true);
    } else {
        // 角色 3:forward warp —— 从 scale-out 接收缓冲取出远端 token,再经节点内 NVLink 转发到对应 scale-up rank
        // 本 warp 在 SM 内的 forward 通道编号
        const int forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);
        // 全局 channel 编号
        const int channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;
        // scale-out 接收缓冲限定到本 channel
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
        // scale-up 缓冲限定到本 rank(最终接收落位)
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // `token_metadata_at_forward` 的形状:`[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        // 每 token 的元数据维度:源索引 + 结束标志 + topk 个(scale-up rank)+ topk 个(目标槽位)
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        // 把元数据指针推进到本 channel 的起始位置
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // `dst_buffer_slot_idx` 的形状:`[kNumChannels, kNumScaleoutRanks, kNumMaxTokensPerChannel, kNumTopk]`
        // 把槽位索引指针推进到本 channel 的起始位置
        dst_buffer_slot_idx += channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel * kNumTopk);

        // 变换链表(linked list)下标
        // 把 channel 内的槽位下标映射为全局链表数组下标(按 channel 与 scale-up rank 分段)
        const auto transform_linked_list_idx = [=](const int& idx) {
            // 每个 channel 的链表长度:该 channel 承载的全部槽位 + 1(尾部哨兵)
            constexpr int kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1;
            return channel_idx * (kNumTokensInLinkedList * kNumScaleupRanks) +
                idx * kNumScaleupRanks + scaleup_rank_idx;
        };

        // 从各 scale-out rank 转发 token
        // 静态断言:scale-out rank 数不超过 32(单 warp lane 数)
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Too many scale-out ranks");
        // 本 channel 已处理的 token 数
        int num_tokens_processed = 0;
        // 各 scale-out rank 已消费(旧)尾指针
        int stored_scaleout_old_tail_idx = 0;
        // 每个 lane 分摊的 scale-up 发送计数(kNumScaleupRanksPerLane ≥ 上取整(kNumScaleupRanks/32))
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};
        // lane i 记录第 i 个 scale-out rank 是否已发完(-1 表示本 lane 无对应 rank,视为已结束)
        int stored_finish_flag = lane_idx >= kNumScaleoutRanks;
        // 各 scale-out rank 的最新(已发布)尾指针
        int stored_scaleout_tail_idx = 0;
        // 本轮轮询的起始 scale-out rank(轮转调度用)
        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;
        // warp 内"仍有工作"的 lane 位掩码
        uint32_t wip_mask;
        // 只要还有 rank 有未消费数据或尚未结束,就继续轮询
        while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {
            // 以轮转(round-robin)方式选择下一个 rank
            // 从上次位置顺时针寻找第一个活跃 rank
            const auto offset = (recv_scaleout_rank_idx + 1) % kNumScaleoutRanks;
            // 取偏移之上的高位掩码,优先在其中找活跃 rank
            const auto hi_mask = (wip_mask >> offset) << offset;
            // 高位有活跃 rank 则取其中最小者,否则从头取
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);

            // 等待该 rank 有数据(或已结束)
            // 带超时的轮询
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                // 检查各 lane 对应 rank 是否"有新数据或已结束"
                const uint32_t arrived_or_finished =
                    stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                // 通过 butterfly 交换读取所选 rank 对应 lane 的状态;已就绪则返回
                if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))
                    return true;

                // 超时
                // 最后一轮检查由持有有效 rank 的 lane 打印诊断信息
                if (is_last_check) {
                    if (lane_idx < kNumScaleoutRanks) {
                        printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                               "channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d)\n",
                               scaleout_rank_idx, scaleup_rank_idx,
                               channel_idx, lane_idx, stored_scaleout_old_tail_idx,
                               stored_finish_flag, stored_scaleout_tail_idx);
                    }
                    return false;
                }

                // 读取新的 signaled tail
                // lane i 读取第 i 个 scale-out rank 在本 channel 上发布的(结束标志,尾指针)打包值
                if (lane_idx < kNumScaleoutRanks) {
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                        workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx));
                    // 解包为结束标志与新尾指针
                    math::unpack2<int, int64_t>(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
                }
                // warp 内同步
                __syncwarp();
                return false;
            });

            // 处理当前 rank 的一个数据块(chunk)
            // 交换读取所选 rank 的旧尾指针作为本 chunk 起点
            const auto start_slot_idx = ptx::exchange(stored_scaleout_old_tail_idx, recv_scaleout_rank_idx);
            // 终点取最新尾指针与(起点 + chunk 大小)的较小者
            const auto end_slot_idx = std::min(
                ptx::exchange(stored_scaleout_tail_idx, recv_scaleout_rank_idx),
                start_slot_idx + kNumSlotsPerForwardChunk
            );
            // 由负责该 rank 的 lane 推进其已消费尾指针
            if (lane_idx == recv_scaleout_rank_idx)
                stored_scaleout_old_tail_idx = end_slot_idx;
            // 取所选 rank 在本 channel 的接收缓冲
            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);
            // 逐槽位处理本 chunk 的 token
            for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {
                // 取该槽位的 token 缓冲
                const auto token_buffer = recv_buffer.get_token_buffer(slot_idx);

                // 等待 TMA(store)完成
                // 确保上一轮对共享内存 TMA 缓冲的复用安全
                ptx::tma_store_wait();
                // warp 内同步
                __syncwarp();

                // TMA 加载到共享内存
                // elected 线程发起:接收缓冲 token → 共享内存,并等待加载完成
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), token_buffer.get_base_ptr(),
                                     mbarrier_ptr, token_layout.get_num_bytes<false>());
                    // 设置期望字节数并以相位翻转等待 TMA 加载完成
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                }
                // warp 内同步
                __syncwarp();

                // 读取 top-k 索引
                // 静态断言:top-k 数不超过 lane 数
                EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
                // 本 lane 对应专家的目标 scale-up rank(-1 表示本节点不负责该专家)
                int stored_dst_scaleup_rank_idx = -1;
                // lane i 读取共享内存中缓存的第 i 个专家编号(无效 lane 置 -1)
                auto dst_expert_idx = lane_idx < kNumTopk ? tma_buffer.get_topk_idx_ptr()[lane_idx] : -1;
                // 减去本 scale-out 节点的专家基址,得到节点内相对编号
                dst_expert_idx -= scaleout_rank_idx * kNumExpertsPerScaleout;
                // 相对编号落在本节点范围内时,换算成目标 scale-up rank
                stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < kNumExpertsPerScaleout ?
                    dst_expert_idx / kNumExpertsPerRank : -1;

                // 写入该 token 的每 scale-up 通道(channel)下标
                // 链表下标(-1 表示无效)
                int linked_list_idx = -1;
                #pragma unroll
                // 每个 lane 分摊若干 scale-up rank:从负责 lane 的发送计数中分配链表插入位置
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    // 该 scale-up rank 是否由本 warp 的某个 lane 负责(编号落在该分摊段内)
                    const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;
                    // 仅当落在 0~31 内有效
                    const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;
                    // 从负责 lane 交换读取其当前发送计数(即链表插入位置)
                    const auto exchanged = ptx::exchange(
                        stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                    // 有效则记录链表下标
                    linked_list_idx = valid ? exchanged : linked_list_idx;
                }
                // 不复用槽位时,把链表下标写入共享内存缓冲(随 token 一起发送)
                if (not kReuseSlotIndices and lane_idx < kNumTopk) {
                    tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                    // TMA store 前的内存栅栏
                    ptx::tma_store_fence();
                }
                // warp 内同步
                __syncwarp();

                // 对 scale-up rank 去重
                // 本 lane 的目标槽位(-1 表示无需发送)
                int stored_dst_slot_idx = -1;
                // 该槽位在 dst_buffer_slot_idx 表中的记录地址
                const auto dst_slot_idx_ptr = dst_buffer_slot_idx +
                    recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) + slot_idx * kNumTopk;
                if constexpr (kReuseSlotIndices) {
                    // 复用模式:直接读取上次记录的槽位
                    if (lane_idx < kNumTopk)
                        stored_dst_slot_idx = __ldg(dst_slot_idx_ptr + lane_idx);
                } else {
                    // 对 NVLink rank(scale-up)去重
                    // warp 内按目标 rank 去重,代表 lane 原子递增全局发送计数器分配槽位
                    if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)
                        stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                }
                // warp 内同步
                __syncwarp();

                // 发起 TMA
                // 有目标槽位时才发送
                if (stored_dst_slot_idx >= 0) {
                    // 计算目标 scale-up rank 上 token 槽位的对称内存地址(LSA/NVLink 窗口)
                    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        stored_dst_scaleup_rank_idx);
                    // TMA store:从共享内存直接写入远端(节点内)对称内存
                    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    // 提交 TMA store 组
                    ptx::tma_store_commit();
                }
                // warp 内同步
                __syncwarp();

                // 累加每 scale-up 计数
                // 静态断言:scale-up 对端数不超过 64
                EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Invalid number of scale-up peers");
                // 掩码类型:不超过 32 个对端用 32 位,否则用 64 位
                using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
                // 汇总各 lane 目标 rank 的命中掩码
                const auto scaleup_send_mask = ptx::reduce_or(
                    stored_dst_scaleup_rank_idx >= 0 ?
                    (mask_t(1) << stored_dst_scaleup_rank_idx) : mask_t(0));
                #pragma unroll
                // 各 lane 对自己分摊负责的 scale-up rank 计数加 1
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_scaleup_send_counters[j] += (scaleup_send_mask >> (j * 32 + lane_idx)) & 1;

                // 在 forward 阶段记录元数据
                // 不复用槽位时,为 copy epilogue 内核记录还原所需的元数据
                if constexpr (not kReuseSlotIndices) {
                    // 静态断言:top-k 数不超过 lane 数
                    EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of selections");
                    // 本 token 元数据的写入地址(按处理顺序紧凑排列)
                    const auto metadata_ptr = token_metadata_at_forward +
                        num_tokens_processed * kNumForwardMetadataDims;

                    // 源 token 下标与"最后一个 token"标志
                    if (ptx::elect_one_sync()) {
                        // 写入源 token 全局下标
                        metadata_ptr[0] = tma_buffer.get_src_token_global_idx_ptr()[0];
                        // 本槽位是否为本 chunk 的最后一个(供 epilogue 判断终止)
                        metadata_ptr[1] = slot_idx == (end_slot_idx - 1);
                    }

                    // 其后依次为:原始 top-k 索引与目标槽位
                    if (lane_idx < kNumTopk) {
                        // 记录该 top-k 项的目标 scale-up rank
                        metadata_ptr[2 + lane_idx] = stored_dst_scaleup_rank_idx;
                        // 记录该 top-k 项的目标槽位
                        metadata_ptr[2 + kNumTopk + lane_idx] = stored_dst_slot_idx;
                        // 同时写入槽位索引表,供下次复用
                        dst_slot_idx_ptr[lane_idx] = stored_dst_slot_idx;
                    }
                }
                // 已处理 token 数加 1
                num_tokens_processed += 1;
                // warp 内同步
                __syncwarp();
            }
        }

        // 把元数据的"源 token 下标"部分置为 -1,作为结束标记
        if (not kReuseSlotIndices and ptx::elect_one_sync())
            token_metadata_at_forward[num_tokens_processed * kNumForwardMetadataDims] = -1;
        // warp 内同步
        __syncwarp();

        // 更新链表的结束位置
        // 不复用槽位时,把各 lane 记录的发送计数(即链表尾)写到各 scale-up 对端的尾部槽
        if constexpr (not kReuseSlotIndices) {
            // 本 channel 在本 scale-up rank 视角的链表尾指针地址
            const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
            #pragma unroll
            // 遍历本 lane 分摊的所有 scale-up rank(最后一组须防越界)
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                // 全局 scale-up rank 编号;仅处理有效范围
                if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                    // 松弛 release 的系统级写:把链表尾部下标写到对端对称内存
                    ptx::st_relaxed_sys(
                        gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j),
                        transform_linked_list_idx(stored_scaleup_send_counters[i]));
                }
            }
        }
        // warp 内同步
        __syncwarp();

        // 清理尾部计数,供下次使用
        if (lane_idx < kNumScaleoutRanks)
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
        // warp 内同步
        __syncwarp();
    }

    // scale-up 屏障:确保数据已到达
    // 由于 scale-out 的 token 已被 forward warp 消费,无需再做 scale-out 屏障
    // 全 rank、全 SM 的收尾屏障(tag1,仅 scale-up 部分生效)
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag1, true, true, false>(
        gin, workspace_layout.get_barrier_signals(), scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, /* do not scale-out */ false, true);

    // 触发(copy epilogue)后续内核(programmatic dependent launch)
    cudaTriggerProgrammaticLaunchCompletion();

    // 清理 scale-up 计数器
    // 所有 scale-out 计数器在此之前应已清理
    // 静态断言:线程数足够覆盖 scale-up rank 数
    EP_STATIC_ASSERT(kNumScaleupRanks <= kNumThreads, "Insufficient threads");
    // 不复用槽位时,由 SM 0 的前若干线程清零各 scale-up rank 的原子发送计数器
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumScaleupRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::ep
