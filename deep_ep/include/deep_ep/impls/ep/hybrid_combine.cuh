#pragma once
// 头文件保护：防止重复包含

#include <deep_ep/comm/barrier.cuh>
// 引入 GPU 间全局 barrier（GIN 信号、超时控制）
#include <deep_ep/common/math.cuh>
// 引入数学工具：constexpr_ceil_div、advance_ptr、pack2 等
#include <deep_ep/common/ptx.cuh>
// 引入 PTX 封装：TMA load/store、mbarrier、warp 原语、系统级原子/访存等
#include <deep_ep/layout/ep/token.cuh>
// 引入 token/buffer 布局抽象
#include <deep_ep/layout/ep/workspace.cuh>
// 引入 EP workspace 布局（barrier 信号、channel tail 等）
#include <deep_ep/impls/ep/combine_utils.cuh>
// 引入 combine 归约核心工具（combine_reduce、compute_topk_slots、CombineVecTraits）

namespace deep_ep::ep {
// 进入 deep_ep::ep 命名空间

// 混合（hybrid）combine kernel：同时处理节点内 scale-up（NVLink）与跨节点
// scale-out（RDMA）两级通信。前 kNumScaleupWarps 个 warp 负责 scale-up 段
// （把本地激活写进各 scale-up 对端的接收缓冲），其余 warp 作为 forward 通道
// 负责转发/归约并发起 scale-out 的 RDMA 发送。
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs,
          // scale-up 子团队的 warp 数与 forward（scale-out）通道的 warp 数
          int kNumScaleupWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          // QP 数量与超时周期（RDMA/等待循环用）
          int kNumQPs, int64_t kNumTimeoutCycles,
          // 每个 lane 负责的 scale-up rank 数（rank 数摊到 32 个 lane）
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          // tail（队尾计数）发布间隔：攒够若干次更新再发布，减少慢速的 release 写
          int kNumScaleupUpdateInterval = 3,
          // 每个 SM 的 forward 通道数 = forward warp 数；总通道数 = 每 SM 通道数 * SM 数
          int kNumChannelsPerSM = kNumForwardWarps,
          int kNumChannels = kNumChannelsPerSM * kNumSMs,
          // 每条 channel 分摊的最大 token 数
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          // 总 rank 数 = scale-out * scale-up
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          // 总 warp 数 = scale-up warp + forward warp
          int kNumWarps = kNumScaleupWarps + kNumForwardWarps,
          // 线程块大小
          int kNumThreads = kNumWarps * 32,
          // hidden 字节数（BF16）
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          // scale-out / scale-up 两级各自的布局开关与布局槽位数
          bool kUseScaleoutRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          bool kUseScaleupRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>(),
          int kNumTokensInScaleoutLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          int kNumTokensInScaleupLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>()>
// launch_bounds：固定块大小，每 SM 至少驻留 1 个块
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_combine_impl(nv_bfloat16* x,
                    float* topk_weights,
                    int* src_metadata,
                    int* psum_num_recv_tokens_per_scaleup_rank,
                    int* token_metadata_at_forward,
                    int* channel_linked_list,
                    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                    void* buffer, void* workspace,
                    const int scaleout_rank_idx, const int scaleup_rank_idx,
                    int num_reduced_tokens) {
    // Utils
    // 本线程块所在的 SM 编号
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 块内线程编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内 warp 编号
    const auto warp_idx = ptx::get_warp_idx();
    // warp 内 lane 编号
    const auto lane_idx = ptx::get_lane_idx();
    // 是否为“展开发送”模式：展开布局且不允许多重归约
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;

    // Combine vector type selection
    // 依据架构与 hidden 对齐选择 combine 向量类型（int4 或 longlong4_t）
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    // hidden 维度的向量总数
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);

    // Workspaces
    // workspace 布局：按 scale-out / scale-up 两级 rank 与 expert 数组织屏障信号与 channel 状态
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // We should assign the real number of received tokens if without CPU sync
    // 不经 CPU 同步时，从 GPU 端前缀和读取真实的 token 总数
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        // 最后一个 scale-up rank 的累计值即全局 token 数
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumScaleupRanks - 1);

    // Token layouts
    // token 布局：hidden 字节、无 scale、top-k 个权重槽位、不用 FP8
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

    // TMA buffers
    // 动态共享内存：按 TMA 对齐要求声明
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // 每个 warp 独占一块带 mbarrier 的共享内存缓冲（kNumWarps 份、每份 1 token）
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);

    // All the buffer layouts
    // scale-up 接收缓冲区：位于 buffer 起始处，槽位数按 scale-up 布局，容量为 scale-out rank 数 * 每 rank token 上限
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleupLayout, kNumScaleoutRanks * kNumMaxTokensPerRank,
        buffer);
    // scale-out 接收缓冲区：紧跟 scale-up 缓冲区之后，按 scale-out 布局组织
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank,
        scaleup_buffer.get_buffer_end_ptr());
    // scale-out 发送缓冲区（RDMA 中转）：允许多重归约时只需 1 个槽位，展开模式需 kNumTopk 个
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, kAllowMultipleReduction ? 1 : kNumTopk, kNumChannels * (kNumScaleoutRanks * kNumMaxTokensPerChannel),
        scaleout_recv_buffer.get_buffer_end_ptr());

    // Init TMA for scale-up and forward warps
    // mbarrier 当前相位（翻转等待用）
    ptx::arrival_phase phase = 0;
    // 取本 warp TMA 缓冲的 mbarrier 地址
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        // warp 中唯一的 lane 初始化 mbarrier（到达计数 1）并 fence
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // warp 同步，保证 mbarrier 初始化全员可见
    __syncwarp();

    // NCCL Gin handle
    // Each warp is a channel
    // NCCL GIN 句柄：每个 warp 视作一条 channel，按此分配 QP 与共享模式
    const auto [qp_idx, sharing_mode] =
        // 注意：scale-up warp 与 forward warp 共用每 SM 的通道编号（warp_idx 对通道数取模）
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, warp_idx % kNumChannelsPerSM);
    // 用设备通信句柄、对称窗口和 QP 配置构造 GIN
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Global parallel barriers for scale-out subteam and scale-up subteam
    // NOTES: this barrier needs a grid sync, as there are channel scale-up tail cleaning before
    // scale-out 与 scale-up 两个子团队并行参与的全局屏障
    // NOTES: 因为之前有 channel scale-up tail 的清理，此屏障需要 grid 级同步
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout.get_barrier_signals(), scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    // 特定情况下调整各 warp 组的寄存器配额
    // TODO: 支持更多情况，或让通道数更对齐
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;
    // scale-up warp 用的寄存器数（较少，它们主要做发射）
    constexpr int kNumRegistersForScaleupWarps = 40;
    // forward warp 用剩下的寄存器（归约需要大量寄存器）
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;

    // Different warp roles
    // 按 warp 编号分派角色：前 kNumScaleupWarps 个做 scale-up 发送，其余做 forward 转发/归约
    if (warp_idx < kNumScaleupWarps) {
        // 本 warp 对应的全局 channel 编号
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;

        // Adjust registers
        // 需要时把本 warp 组（scale-up warp）的寄存器重新分配为较小配额
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_realloc<kNumRegistersForScaleupWarps, kNumThreads>();

        // Shift into the right buffer if using rank layout
        // rank 布局下把 scale-up 缓冲区视图移到本 rank 的子缓冲上
        if constexpr (kUseScaleupRankLayout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Expanding send mode must not be backward
        // 展开发送模式不能用于反向传播
        if constexpr (kDoExpandedSend)
            // 运行时断言：不应有权重指针
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // Tail issuer
        // `st.release.sys` is pretty slow, so do it by an interval
        // tail（已发送 token 数）发布器：`st.release.sys` 非常慢，故按间隔批量发布
        int update_counter = 0;
        // 本 lane 记录的、已向各 scale-up rank 发送的 token 数（当前值与上次已发布值）
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        // 本 channel 在 workspace 中对应各 scale-up 对端的 tail 指针
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        // 批量更新 tail 的 lambda：finish=true 表示最终强制发布一次
        const auto update_tails = [&](const bool& finish = false) {
            // 累计更新次数
            ++ update_counter;
            // 达到发布间隔或最终收尾时才真正发布
            if (finish or update_counter == kNumScaleupUpdateInterval) {
                // Wait all TMA stores to finish
                // 先等所有 TMA store 落地，保证发布出去的 tail 与数据一致
                ptx::tma_store_wait();
                // warp 同步
                __syncwarp();

                // Issue
                // 逐个 scale-up rank 发布 tail
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    // lane 对应的全局 scale-up rank 号；最后一组需检查是否越界
                    if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        // NOTES: 用旧值比对省掉无变化的写流量；对端清理期间也不能覆写已完成的槽位
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                            // 带系统级 release 语义写入对端对称内存中的 tail
                            ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        // 记录本次已发布的值
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                // 重置计数器
                update_counter = 0;
            }
            // warp 同步
            __syncwarp();
        };

        // Shape of `channel_linked_list`: `[kNumChannels, kNumMaxTokensPerChannel + 1, kNumScaleupRanks]`
        // Iterate until all scale-up peers finish
        // `channel_linked_list` 形状：[通道数, 每 channel token 上限 + 1, scale-up rank 数]
        // 迭代直到所有 scale-up 对端都处理完
        // 轮询游标：从本 channel 编号开始轮转选择下一个要处理的 scale-up rank
        int dst_scaleup_rank_idx = channel_idx;
        // 各 lane 记录链表当前位置与当前 token 下标
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};
        #pragma unroll
        // 初始化所有 token 下标为 -1（表示尚未加载）
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_token_idx[i] = -1;
        // 主循环：不断从链表取 token 并处理
        while (true) {
            // Load token indices in the list
            // 从链表加载各 lane 负责的 scale-up rank 的当前 token 下标
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                // lane 对应的全局 rank 号
                const auto j = i * 32 + lane_idx;
                // 最后一组越界或链表结束（-1）时不读，置 -1
                stored_token_idx[i] = i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks ?
                    __ldg(channel_linked_list +
                          // 按 channel 定位链表首地址，再按当前节点与 lane 定位
                          channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumScaleupRanks +
                          stored_ll_idx[i] * kNumScaleupRanks + j) : -1;
            }
            // warp 同步
            __syncwarp();

            // Check whether all ranks are finished
            // 检查是否所有 rank 都已结束
            bool exited = true;
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                // ptx::all 对 warp 内做全称归约：该组内所有 lane 的下标都 < 0 才算结束
                exited &= ptx::all(stored_token_idx[i] < 0);
            if (exited)
                // 全部结束则退出主循环
                break;

            // Process tokens for all ranks together using bitmask to skip inactive ranks
            // 用位掩码统一处理所有 rank 的 token，跳过不活跃的 rank
            // scale-up rank 数不能超过 64（64 位掩码上限）
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up ranks for 64-bit mask");
            // 依据 rank 数选择 32 位或 64 位掩码类型
            using mask_t = std::conditional_t<(kNumScaleupRanks <= 32), uint32_t, uint64_t>;
            // 工作中（work-in-progress）掩码
            mask_t wip_mask = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                // 把各组的“有 token”标志汇总成按 rank 编号的位掩码
                wip_mask |= static_cast<mask_t>(ptx::gather(stored_token_idx[j] >= 0)) << (j * 32);
            // 只要还有活跃 rank 就继续处理
            while (wip_mask) {
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)
                // 轮转寻找 dst_scaleup_rank_idx 之后下一个活跃的 rank
                const auto start = (dst_scaleup_rank_idx + 1) % kNumScaleupRanks;
                // 取掩码中 >= start 的高位部分；若为空则回绕到掩码最低位的活跃 rank
                const auto hi_mask = (wip_mask >> start) << start;
                dst_scaleup_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);
                // 从工作掩码中摘除该 rank（已选中）
                wip_mask ^= static_cast<mask_t>(1) << dst_scaleup_rank_idx;

                // Exchange token index from the owning lane using static partition iteration
                // 通过静态分区的交换，把拥有该 rank 的 lane 上的 token 下标广播给全 warp
                int token_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    // 该 rank 所属 lane 的编号（按 32 分组）
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;
                    // 若本 lane 即持有者则取出暂存值
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;
                }
                // warp 交换：从目标 lane 拿到 token 下标
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);

                // Get source metadata and decide the destination buffer
                // 读取源元数据并决定目标缓冲区
                // 元数据步长：2 个整数字段 + kNumTopk 个槽位下标
                constexpr int kMetadataStride = 2 + kNumTopk;
                // 源 token 的全局编号（含 scale-out 与 rank 内偏移）
                const auto src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride + 0);
                // 源 token 在源 rank 内的下标
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
                // 反推源 token 所属的 scale-out rank
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * kNumScaleupRanks);
                // 按 scale-up 布局方式定位目标 token 缓冲
                auto token_buffer = [&]() {
                    if constexpr (kUseScaleupRankLayout) {
                        // rank 布局：直接用元数据中的槽位号（除以 top-k 得到源槽位）
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        // top-k 布局：用 top-k 位置选 rank 分块，再按 scale-out 与 rank 内偏移定位 token
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                    }
                }();
                // 把本地指针换算成目标 scale-up 对端对称内存窗口中的远端指针
                token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));

                // Some checks
                // hidden 维度约束：能被一个 warp 单轮 int4 搬运的 BF16 数量整除
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                // 展开布局下读取各 lane 的源 top-k 槽位下标
                int stored_topk_slot_idx = -1;
                if constexpr (kUseExpandedLayout) {
                    if (lane_idx < kNumTopk)
                        // lane 对应 top-k 位置的槽位下标
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    // warp 同步
                    __syncwarp();
                }

                // 3 cases:
                //  - no-expand, expand + no-reduce
                //  - expand + reduce
                //  - expand + send all
                // 三种情况：不展开或展开但不归约 / 展开且本地归约 / 展开且全量发送
                // warp 内“槽位有效”的掩码
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                // 无需本地归约：非展开布局，或（允许多重归约且仅 1 个有效槽位）
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    // 本地张量中的 token 下标默认即 token_idx
                    int token_idx_in_tensor = token_idx;
                    if constexpr (kUseExpandedLayout)
                        // 展开布局下由主 lane 的槽位经 warp 交换换算真实下标
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // Directly load
                    // 直接从本地 x 加载到共享内存
                    if (ptx::elect_one_sync()) {
                        // 由唯一 lane 执行 TMA load
                        // 计算本地 x 中该 token 的地址
                        const auto load_ptr =
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        // 先等上一次 TMA store 排空以复用缓冲
                        ptx::tma_store_wait();
                        // 发起 1D TMA load：全局 x -> 共享内存
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    // warp 同步
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {
                    // Do local reduction
                    // 本地归约路径
                    // Sort valid top-k indices to front
                    // 把有效槽位下标整理到数组前部
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            // warp 交换换取各有效槽位在本地 x 中的下标
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    // 编译期求展开因子
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
                    // 调用通用归约：从本地 x 的多个槽位累加到共享内存
                    combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ /* 源基址：本地 x 中该槽位 token 的地址 */
                        /* Get source base */ [=](const int& slot_idx) {
                            // 向量类型指针前进 slot_idx 个 token
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ /* 复用共享内存前等待上次 TMA store 完成 */
                        /* Wait buffer release */ [=]() {
                            // 等待 TMA store 排空
                            ptx::tma_store_wait();
                            // warp 同步
                            __syncwarp();
                        }
                    );
                    // fence：共享内存写入对 TMA 异步代理可见
                    ptx::tma_store_fence();
                    // warp 同步
                    __syncwarp();
                } else {
                    // No local reduction, send all data (expanded send)
                    // 无本地归约的全量展开发送
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        // warp 交换得到第 k 个槽位的下标
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (topk_slot_idx < 0)
                            // 无效槽位直接跳过
                            continue;

                        if (ptx::elect_one_sync()) {
                            // 唯一 lane 执行 TMA 流水
                            // Load
                            // 展开发送按槽位下标取源，否则按 token 下标
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);
                            // 等待上次 store 完成
                            ptx::tma_store_wait();
                            // TMA load：本地 x -> 共享内存
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            // 设定 mbarrier 的传输字节计数
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            // 等待 load 完成（相位翻转）
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode
                            // NOTES: 展开模式下无需关心 top-k 权重

                            // Store
                            // 计算目标 rank k 的接收缓冲中该源 token 的位置
                            const auto dst_token_buffer = scaleup_buffer
                                .get_rank_buffer(k)
                                .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                            // TMA store：共享内存 -> 远端对称窗口中的目标位置
                            ptx::tma_store_1d(
                                gin.get_sym_ptr<ncclTeamTagLsa>(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx),
                                tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            // 提交 TMA store 批次
                            ptx::tma_store_commit();
                        }
                        // warp 同步
                        __syncwarp();
                    }
                }

                // Write top-k weights (expanded send handled inside the loop above)
                // 写 top-k 权重（展开发送已在上面循环内处理）
                if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
                    // 无效槽位默认权重 0
                    float value = 0;
                    if constexpr (kUseExpandedLayout) {
                        if (stored_topk_slot_idx >= 0)
                            // 展开布局：读该槽位的权重
                            value = __ldg(topk_weights + stored_topk_slot_idx);
                    } else {
                        // 非展开布局：按 token 与 lane 定位
                        value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));
                    }
                    // 写入本 warp TMA 缓冲的权重区（随后随数据一起 TMA store 出去）
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    // fence：确保权重写入对 TMA 可见
                    ptx::tma_store_fence();
                }
                // warp 同步
                __syncwarp();

                // Issue TMA stores into remote scale-up buffer
                // NOTES: `kDoExpandedSend` mode has already issued
                // 发射 TMA store 到远端 scale-up 缓冲
                // NOTES: `kDoExpandedSend` 模式已在上面发射过
                if (not kDoExpandedSend and ptx::elect_one_sync()) {
                    // Wait TMA arrival (only for non-reduced cases)
                    // 等待 TMA load 到达（仅非归约情形：归约路径已在 combine_reduce 内同步）
                    if (no_local_reduce) {
                        // 设定传输字节计数并等待相位翻转
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    // TMA store：共享内存 -> 目标 token 缓冲（远端或本 rank）
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    // 提交批次
                    ptx::tma_store_commit();
                }
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    // 各 lane 统计本 lane 对应 rank 是否刚发送了一个 token（用于 tail 计数）
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                // warp 同步
                __syncwarp();
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            // 统一更新各对端的 tail
            // NOTES: 内部已含 TMA wait
            update_tails();

            // Move linked list
            // 推进各 lane 在链表中的位置（有 token 的组才前进）
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                stored_ll_idx[i] += (stored_token_idx[i] >= 0);
        }

        // Update for the unissued ones
        // 收尾：把尚未发布的 tail 全部发布出去
        update_tails(true);
    } else {
        // forward warp 的组内编号（去掉前面的 scale-up warp）
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;
        // 本 forward warp 的全局 channel 编号
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

        // Adjust registers
        // 需要时把本 warp 组（forward warp）的寄存器配额放大（归约需要）
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_realloc<kNumRegistersForForwardWarps, kNumThreads>();

        // Shift into the right buffer
        // 把 scale-out 发送缓冲视图移到本 channel 独占的分块上
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer<kNumScaleoutRanks * kNumMaxTokensPerChannel>(channel_idx);

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        // `token_metadata_at_forward` 形状：[通道数, 每 channel token 上限 + 1, 元数据维度数]
        // 每个 token 的 forward 元数据维度：全局下标、是否 chunk 最后一个、每组 top-k 两个整数
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        // 把元数据指针推进到本 channel 的起始处
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // Overlap TMA stores and reduction
        // 让 TMA store 与归约重叠：暂存上一条尚未发射 RDMA 的信息
        int last_src_scaleout_rank_idx = -1;
        int last_is_token_last_in_chunk = 0;
        void* last_recv_token_buffer_ptr = nullptr;
        void* last_send_token_buffer_ptr = nullptr;
        // 冲刷上一条 TMA 并发射其 RDMA put 的 lambda
        const auto flush_last_tma_and_issue_rdma = [&]() {
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {
                // 等待上一条 TMA store 完成后数据才可被 RDMA 读取
                ptx::tma_store_wait();

                // Issue only if not local rank
                // 目标是本 rank 时无需 RDMA（数据已在本地接收缓冲）
                if (last_src_scaleout_rank_idx != scaleout_rank_idx) {
                    // GIN RDMA put（Rail team）：发送缓冲 -> 远端接收缓冲；
                    // 若该 token 不是 chunk 最后一个，则聚合多个请求以减少开销
                    gin.put<ncclTeamTagRail>(
                        last_recv_token_buffer_ptr,
                        last_send_token_buffer_ptr,
                        token_layout.get_num_bytes<false>(),
                        last_src_scaleout_rank_idx,
                        last_is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                    );
                }
            }
            // warp 同步
            __syncwarp();
        };

        // Replay the dispatch
        // 重放 dispatch 的 token 流：本 lane 记录的各 scale-up rank 已收计数与缓存的 tail 值
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        // 主循环：逐 token 读取 forward 元数据（i 自增，遇负数下标退出）
        for (int i = 0; ; ++ i) {
            // 源 token 的全局编号（<0 表示列表结束）
            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            // 该 token 是否为所在 chunk 的最后一个（决定 RDMA 是否携带完成标记）
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);
            // 反推源 rank（全局）、源 scale-out rank 与源 rank 内下标
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            // 各 lane 读取该 top-k 位置对应的源 scale-up rank 编号（无效为 -1）
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            // 各 lane 读取对应的源槽位下标（用于 scale-up rank 布局）
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                // 遇到结束标记退出循环
                break;

            // Scaleup rank mask
            // scale-up rank 掩码：收集本 token 涉及的所有 scale-up rank
            // rank 数不能超过 64
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up peers");
            // 依 rank 数选择掩码宽度
            using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
            // warp 归约出“涉及哪些 scale-up rank”的位掩码
            const auto scaleup_mask = ptx::reduce_or(
                stored_src_scaleup_rank_idx >= 0 ?
                (mask_t(1) << stored_src_scaleup_rank_idx) : mask_t(0));
            // 各 lane 判断自己负责的 scale-up rank 是否被本 token 需要
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_is_scaleup_rank_needed[j] = (scaleup_mask >> (j * 32 + lane_idx)) & 1;

            // Wait all tails to arrive
            // 等待所有需要的 scale-up 对端 tail 到达（即它们已把本 token 的数据发出）
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                // 检查是否所有需要的 rank 都满足“已收计数 < 缓存 tail”（即数据已就绪）
                bool arrived = true;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                if (ptx::all(arrived))
                    // warp 全员满足则结束等待
                    return true;

                // Reload cached
                // 重新加载各对端 tail 的最新值（acquire 语义保证可见性）
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    // lane 对应的全局 scale-up rank 号
                    const auto k = j * 32 + lane_idx;
                    stored_cached_scaleup_tail[j] = j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks ?
                        ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                // 最后一次检查仍失败则打印超时诊断信息
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                               channel_idx, j * 32 + lane_idx,
                               stored_num_tokens_recv[j],
                               stored_cached_scaleup_tail[j],
                               stored_is_scaleup_rank_needed[j]);
                    }
                }
                // 返回 false 继续轮询
                return false;
            });

            // Increase received count
            // 递增各 lane 负责的 rank 的已收计数
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);

            if constexpr (not kAllowMultipleReduction) {
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                // 禁用多重归约的情形：需要把 scale-up 对端的全部数据逐份转发给 scale-out 对端
                // TODO: 让 scale-up warp 直接把数据放进 `send_buffer`？
                // 本 token 在 scale-up 缓冲中的全局槽位
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                // 各 top-k 位置的有效掩码：展开布局只需槽位有效；
                // 非展开布局需按 scale-up rank 去重（同一 rank 只转发一次）
                auto topk_valid_mask = kUseExpandedLayout ?
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                if (ptx::elect_one_sync()) {
                    // 唯一 lane 串行处理各有效 top-k 位置
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        if ((topk_valid_mask & (1u << k)) == 0u)
                            // 该 top-k 位置无效则跳过
                            continue;

                        // Issue TMA load, and wait
                        // 发起 TMA load：scale-up 缓冲的对应槽位 -> 共享内存，并用 mbarrier 等待完成
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        // 设定 mbarrier 传输字节计数
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        // 等待 load 完成
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        // Issue TMA store, and wait
                        // 计算接收/发送缓冲：目标是本 scale-out rank 时直接写本地接收缓冲
                        const auto recv_buffer_ptr = scaleout_recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx).get_base_ptr();
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr : scaleout_send_buffer.get_rank_buffer(k).get_token_buffer(i).get_base_ptr();
                        // TMA store：共享内存 -> 发送缓冲（或本地接收缓冲）
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        // 提交并等待完成（RDMA 源数据必须落地）
                        ptx::tma_store_commit();
                        ptx::tma_store_wait();

                        // Issue IBGDA
                        // 发起 IBGDA RDMA：先清掉已处理的 top-k 位
                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx) {
                            // 跨节点才需要 RDMA put；最后一个有效 top-k 且 token 为 chunk 末尾时不聚合
                            gin.put<ncclTeamTagRail>(
                                recv_buffer_ptr,
                                send_buffer_ptr,
                                token_layout.get_num_bytes<false>(),
                                src_scaleout_rank_idx,
                                topk_valid_mask == 0 and is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                            );
                        }
                    }
                }
                // warp 同步
                __syncwarp();
            } else {
                // NOTES: we must do deduplicate and only add once from one rank
                // NOTES: 必须做去重，保证同一 rank 的数据只累加一次
                // 归约有效掩码：同一 scale-up rank 只保留首次出现的 lane，且该 rank 需有效
                auto reduce_valid_mask = ptx::gather(
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // Calculate the source buffer index
                // 计算各 lane 对应的源缓冲区下标
                int stored_src_buffer_idx = 0;
                if constexpr (kUseScaleupRankLayout) {
                    // scale-up rank 布局：源下标 = rank 号 * 每 rank token 上限 + 源槽位
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    // top-k 布局：源下标 = top-k 位置 * 每 rank token 上限 + scale-out 偏移后的槽位
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }

                // Preprocess top-k indices
                // 预处理 top-k 下标：从掩码提取有效槽位并交换出源缓冲下标
                int topk_slot_idx[kNumTokensInScaleupLayout];
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,
                    [=](const int& idx) {
                        // warp 交换换取对应槽位的源缓冲下标
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                // Do reduce
                // 执行归约：从 scale-up 缓冲的各槽位读出并累加到共享内存
                // 注册重配时可用更大的展开因子（寄存器更充裕）
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>();
                combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumScaleoutRanks)>(
                    lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    /* Get source base */ /* 源基址：scale-up 缓冲中该槽位的 token 数据 */
                    /* Get source base */ [=](const int& slot_idx) {
                        // get_token_buffer 第二参数 true 表示下标已是“打包”的全局槽位
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ /* 复用共享内存前先冲刷上一条 TMA 并发射其 RDMA */
                    /* Wait buffer release */ [=]() {
                        // 冲刷上一 token 的 TMA store 并发出对应 RDMA put
                        flush_last_tma_and_issue_rdma();
                    }
                );

                // Merge topk weights
                // NOTES: the slot indices must follow the master lane
                // 合并 top-k 权重
                // NOTES: 槽位下标必须跟随主 lane（同 rank 的代表 lane）
                stored_src_buffer_idx = ptx::exchange(
                    stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                if (stored_src_scaleup_rank_idx >= 0) {
                    // 从源 token 缓冲的权重区读出本 lane 对应的 top-k 权重
                    tma_buffer.get_topk_weights_ptr()[lane_idx] =
                        scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                    .get_topk_weights_ptr()[lane_idx];
                }
                // fence：权重写入对 TMA 可见
                ptx::tma_store_fence();
                // warp 同步：让主 lane 能看到其他 lane 的写入
                __syncwarp(); // Necessary to let the leader lane see the writes

                // Assign send and receive buffers
                // NOTES: as we only have 1 destination, we will use "send" as "recv" for local transfer
                // 分配发送与接收缓冲
                // NOTES: 只有一个目的地，本地传输时把“发送缓冲”当作“接收缓冲”用
                int scaleout_recv_buffer_rank_idx;
                if constexpr (kUseScaleoutRankLayout) {
                    // scale-out rank 布局：直接用本 rank 号定位接收缓冲
                    scaleout_recv_buffer_rank_idx = scaleout_rank_idx;
                } else {
                    // top-k 布局：用有效 lane 中的主 lane（即 top-k 位置）定位
                    const int src_topk_idx = ptx::get_master_lane_idx(ptx::gather(stored_src_scaleup_rank_idx >= 0));
                    scaleout_recv_buffer_rank_idx = src_topk_idx;
                }
                // 接收缓冲：本 rank 对应槽位中该源 token 的位置
                const auto recv_token_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_recv_buffer_rank_idx).get_token_buffer(src_token_idx);
                // 发送缓冲：目标是本 rank 时复用接收缓冲，否则用本 channel 的发送分块
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                    recv_token_buffer :
                    scaleout_send_buffer.get_token_buffer(i);

                // Write into scale-out send buffer or local rank recv buffer bypass
                // 写入 scale-out 发送缓冲，或本地 rank 接收缓冲的旁路
                if (ptx::elect_one_sync()) {
                    // TMA store：共享内存归约结果 -> 发送/接收缓冲
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    // 提交 TMA store 批次
                    ptx::tma_store_commit();
                }
                // warp 同步
                __syncwarp();

                // Record RDMA info to issue later
                // 记录 RDMA 信息供之后（与下一 token 的归约重叠地）发射
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_is_token_last_in_chunk = is_token_last_in_chunk;
                last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();
                last_send_token_buffer_ptr = send_token_buffer.get_base_ptr();
            }
        }

        // Issue the last RDMA
        // 发射最后一条滞留的 RDMA（仅多重归约路径有暂存）
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_issue_rdma();

        // Clean scaleup tails
        // 清理本 channel 的 scale-up tail（供下一轮使用）
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            // lane 对应的全局 scale-up rank 号
            const auto k = j * 32 + lane_idx;
            if (j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks)
                // 直接写 0 清零（下轮 barrier 会重新同步）
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        // warp 同步
        __syncwarp();

        // Update, wait and clean
        // 更新、等待并清理 scale-out 侧的完成信号
        // scale-out rank 数不能超过 32（lane 数上限）
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");
        // 期望信号：低 32 位为 1（各对端各加 1），高位为 0
        const auto expected_signal = math::pack2<int, int64_t>(1, 0);
        // 先冲刷本 warp 的所有 GIN 请求，保证之前的 RDMA 全部发出
        gin.flush<ncclCoopWarp>();
        if (lane_idx < kNumScaleoutRanks) {
            // Update remote tails
            // 用远端原子加（带 release）通知各 scale-out 对端：本 channel 已完成一次处理
            gin.red_add_rel<ncclTeamTagRail>(
                workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx),
                expected_signal, lane_idx);
        }
        // warp 同步
        __syncwarp();

        // Wait tail arrival
        // 等待各 scale-out 对端的到达信号
        if (lane_idx < kNumScaleoutRanks) {
            // 本 lane 负责等待的对端信号地址
            const auto wait_ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx);
            // 带超时的轮询等待
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                // acquire 读取远端信号值
                const auto signal = ptx::ld_acquire_sys<int64_t>(wait_ptr);
                if (signal == expected_signal) {
                    // Clean for next usages
                    // 为下次使用清零信号
                    *wait_ptr = 0;
                    return true;
                }

                if (is_last_check) {
                    // 超时时打印诊断信息
                    printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                           "channel: %d, lane: %d, signal: %lld, expected: %lld\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           channel_idx, lane_idx,
                           signal, expected_signal);
                }
                // 继续轮询
                return false;
            });
        }
        // warp 同步
        __syncwarp();
    }

    // No barrier at epilogue
    // 收尾处不再需要屏障（由上层 kernel 配合 PDL/依赖保证）
}

}  // namespace deep_ep::ep
// 结束 deep_ep::ep 命名空间
