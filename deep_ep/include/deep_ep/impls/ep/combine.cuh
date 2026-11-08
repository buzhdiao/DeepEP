#pragma once
// 头文件保护：防止重复包含

#include <nccl_device.h>
// 引入 NCCL 设备端 API：对称内存窗口（ncclWindow_t）、设备通信句柄、GIN 等

#include <deep_ep/comm/barrier.cuh>
// 引入 GPU 间全局 barrier（含 GIN/RDMA 信号收发）
#include <deep_ep/common/math.cuh>
// 引入数学工具：ceil_div、advance_ptr 等
#include <deep_ep/common/ptx.cuh>
// 引入 PTX 封装：TMA load/store、mbarrier、warp 原语等
#include <deep_ep/layout/ep/token.cuh>
// 引入 token/buffer 布局抽象
#include <deep_ep/layout/ep/workspace.cuh>
// 引入 EP workspace 布局（barrier 信号区）

#include <deep_ep/impls/ep/combine_utils.cuh>
// 引入 combine 归约核心工具（combine_reduce、compute_topk_slots 等）


namespace deep_ep::ep {
// 进入 deep_ep::ep 命名空间

// EP combine 的发送端 kernel：把本地激活值按 dispatch 时的来源关系
// 写回（经 NVLink 直达或 RDMA 中转）各目标 rank 的接收缓冲区，供对端做归约
// kIsScaleupNVLink：scale-up 域是否走 NVLink（决定 barrier 与 NCCL team 的选择）
template <bool kIsScaleupNVLink,
          bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          int kNumRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          // QP 数量与超时周期数（RDMA 面向通道的配置）
          int kNumQPs, int64_t kNumTimeoutCycles,
          // 线程块大小 = warp 数 * 32
          int kNumThreads = kNumWarps * 32,
          // hidden 维度字节数（BF16）
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          // 是否按 rank 维度组织布局
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          // 布局中每槽位代表的 token 数
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          // NCCL team 类型：scale-up 走 LSA（节点内），否则走 World（跨节点）
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
// launch_bounds：固定块大小、每 SM 至少 1 个块
__global__ void __launch_bounds__(kNumThreads, 1)
combine_impl(nv_bfloat16* x,
             float* topk_weights,
             int* src_metadata, int* psum_num_recv_tokens_per_scaleup_rank,
             const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* buffer, void* workspace,
             const int rank_idx,
             int num_reduced_tokens) {
    // Utils
    // 本线程块所在的 SM 编号
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 块内线程编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // warp 编号按 rank 偏移轮转，错开不同 rank 上 warp 与通道的对应关系
    const auto warp_idx = (ptx::get_warp_idx() + rank_idx) % kNumWarps;
    // warp 内 lane 编号
    const auto lane_idx = ptx::get_lane_idx();
    // 全局 warp 编号：warp 优先铺满不同 SM（使尾波均匀）
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;
    // 是否为“展开发送”模式：展开布局且不允许多重归约时，需要把每个 top-k 副本都发出去
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;

    // We should assign the real number of received tokens if without CPU sync
    // 不经 CPU 同步时，真实收到的 token 数需从 GPU 端的累计前缀和读取
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        // 用最后一个 scale-up rank 的前缀和作为实际 token 总数
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumRanks - 1);

    // Buffer layouts
    // 动态共享内存：按 TMA 对齐要求声明，作 TMA 双向搬运的中转
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // token 布局：hidden 字节、无 scale、top-k 个权重槽位、不用 FP8
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    // 本 warp 独占的共享内存双缓冲（kNumWarps 份、每份 1 token），带 mbarrier
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        // 取本 warp 的 rank 分块中的第 0 个 token 缓冲
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    // 接收缓冲区：挂在 buffer 指针上，按 kNumTokensInLayout 个槽位组织
    const auto recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, buffer);
    // 发送缓冲区：紧跟在接收缓冲区之后，每槽位 token 数在展开发送时乘上 kNumTopk
    const auto send_buffer = layout::BufferLayout<false>(
        token_layout, kNumRanks,
        kNumMaxTokensPerRank * (kDoExpandedSend ? kNumTopk : 1),
        // 发送缓冲区从接收缓冲区的末尾开始
        recv_buffer.get_buffer_end_ptr());

    // Init TMA
    // mbarrier 的当前相位（双缓冲翻转用）
    ptx::arrival_phase phase = 0;
    // 取本 warp TMA 缓冲对应的 mbarrier 地址（共享内存）
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        // 由 warp 中选出的一个 lane 初始化 mbarrier（到达计数为 1），并做 fence
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // warp 同步，保证 mbarrier 初始化对全体 lane 可见
    __syncwarp();

    // Expanding send mode must not be backward
    // 展开发送模式不能用于反向传播（无 top-k 权重）
    if constexpr (kDoExpandedSend)
        // 编译期条件下的运行时断言
        EP_DEVICE_ASSERT(topk_weights == nullptr);

    // Gin handle
    // We treat each warp as a "channel"
    // Gin handle：把每个 warp 视作一条“通道”（channel），据此分配 QP 与共享模式
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumWarps>(sm_idx, warp_idx);
    // 用设备通信句柄、对称窗口和 QP 配置构造 GIN（GPU Initiated Networking）句柄
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Full barrier to ensure the remote buffer is available
    // 全局 barrier：确保所有 rank 的接收缓冲区都已就绪（对称内存窗口可访问）
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    // 启动屏障（kCombineTag0 阶段），跨 kNumRanks 同步
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag0, false, false, true>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);

    // Do TMA writes into the remote buffers
    // 将总 token 数均分给所有 warp：向上取整得到每个 warp 负责的 token 数
    int num_tokens_per_warp = math::ceil_div(num_reduced_tokens, kNumSMs * kNumWarps);
    // 本 warp 负责区间的起始下标
    const int token_start_idx = num_tokens_per_warp * global_warp_idx;
    // 本 warp 负责区间的结束下标（不越过总数）
    const int token_end_idx = min(token_start_idx + num_tokens_per_warp, num_reduced_tokens);
    // 逐 token 处理本 warp 的区间
    for (int i = token_start_idx; i < token_end_idx; ++ i) {
        // The master slot index during dispatch
        // dispatch 阶段记录的元数据步长：2 个整数字段 + kNumTopk 个槽位下标
        constexpr int kMetadataStride = 2 + kNumTopk;
        // 源 token 在源 rank 内的下标（对每 rank 容量取模）
        const int src_token_idx = __ldg(src_metadata + i * kMetadataStride) % kNumMaxTokensPerRank;
        // 源 rank 与源 top-k 位置被编码在同一个整数中
        const int src_rank_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1);
        // 解码出源 rank 编号
        const int src_rank_idx = src_rank_topk_idx / kNumTopk;
        // 解码出源 top-k 槽位编号
        const int src_topk_idx = src_rank_topk_idx % kNumTopk;

        // Directly to the remote or via RDMA
        // 判断源 rank 是否可经 NVLink 直达（否则需走 RDMA 中转）
        const bool nvlink_bypass = gin.is_nvlink_accessible<team_t>(src_rank_idx);
        // 计算本 token 的目标缓冲区（master token buffer），按路径分支
        layout::TokenLayout master_token_buffer = [=]() {
            // NVLink bypass
            // NVLink 直达路径
            if (nvlink_bypass) {
                // rank 布局按本 rank 号选槽位，否则按源 top-k 位置选；定位到源 token 位置
                auto token_buffer = recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx).get_token_buffer(src_token_idx);
                // 将本地指针换算成源 rank 对称内存窗口中的远端指针
                token_buffer.set_base_ptr(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx));
                // 返回远端 token 缓冲
                return token_buffer;
            }

            // Use RDMA
            // RDMA 路径：先写入本地发送缓冲区中对应源 rank 的槽位
            return send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx);
        }();

        // Hidden requirements
        // hidden 维度约束：必须能被一个 warp 单轮 int4 搬运的 BF16 数量整除
        EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
        // 依据架构与对齐选择 combine 的向量类型
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
        // hidden 维度的向量总数
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);

        // Read source indices for expand mode
        // 展开布局下读取各 lane 的源 top-k 槽位下标
        int stored_topk_slot_idx = -1;
        if constexpr (kUseExpandedLayout) {
            if (lane_idx < kNumTopk)
                // lane 对应的 top-k 位置读出其在本地 x 中的槽位下标
                stored_topk_slot_idx = __ldg(src_metadata + i * kMetadataStride + (2 + lane_idx));
            // warp 同步，保证所有 lane 都已读出
            __syncwarp();
        }

        // 3 cases:
        //  - no expand + no reduce, or expand + no reduce
        //  - expand + reduce
        //  - expand + send all
        // 三种情况：不展开或展开但不归约 / 展开且本地归约 / 展开且全量发送
        // 收集 warp 内“槽位有效”的掩码
        auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
        // 无需本地归约的条件：非展开布局，或（允许多重归约且仅 1 个有效槽位）
        auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
        if (no_local_reduce) {
            // token 在输出张量中的下标默认就是循环下标
            int token_idx_in_tensor = i;
            if constexpr (kUseExpandedLayout)
                // 展开布局下需从主 lane 的槽位下标经 warp 交换换算出真实 token 下标
                token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

            // No reduce
            // 不归约：直接从本地 x 经共享内存中转拷贝到目标缓冲
            if (ptx::elect_one_sync()) {
                // 由唯一的 lane 执行 TMA 流水，避免重复
                // 计算本地 x 中该 token 的加载地址
                const auto load_ptr =
                    math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                // 等待上一轮 TMA store 完成，腾出共享内存缓冲
                ptx::tma_store_wait();
                // 发起 1D TMA load：全局 x -> 共享内存
                ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                // 到达 mbarrier 并设定期望传输字节数（tx 计数）
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                // 自旋等待 mbarrier 相位翻转（即 load 完成且数据可见）
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                // 发起 1D TMA store：共享内存 -> 目标 token 缓冲（远端或发送缓冲区）
                ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                // 提交 TMA store 批次
                ptx::tma_store_commit();
            }
            // warp 同步
            __syncwarp();
        } else if constexpr (kAllowMultipleReduction) {
            // Do local reduction
            // 本地归约路径
            // Sort valid top-k indices to front
            // 把有效的 top-k 槽位下标整理到数组前部
            int topk_slot_idx[kNumTopk];
            compute_topk_slots(
                topk_slot_idx, reduce_valid_mask,
                [=](const int& idx) {
                    // 通过 warp 交换换取各有效槽位在本地 x 中的下标
                    return ptx::exchange(stored_topk_slot_idx, idx);
                }
            );

            // Reduce into shared memory
            // 编译期求展开因子
            constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
            // 调用通用归约：从本地 x 的多个槽位读出并累加到共享内存
            combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                /* Get source base */ /* 源基址：本地 x 中该槽位 token 的地址 */
                /* Get source base */ [=](const int& slot_idx) {
                    // 以向量类型指针前进 slot_idx 个 token 的距离
                    return math::advance_ptr<combine_vec_t>(
                        x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                },
                /* Wait buffer release */ /* 复用共享内存前等待上次 TMA store 完成 */
                /* Wait buffer release */ [=]() {
                    // 等待本 warp 的 TMA store 排空
                    ptx::tma_store_wait();
                    // warp 同步
                    __syncwarp();
                }
            );
            // fence 使共享内存写入对 TMA 异步代理可见
            ptx::tma_store_fence();
            // warp 同步
            __syncwarp();

            // Issue TMA stores
            // 发射 TMA store 到目标缓冲区
            if (ptx::elect_one_sync()) {
                // 共享内存归约结果 -> master token 缓冲（远端或发送缓冲）
                ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                // 提交批次
                ptx::tma_store_commit();
            }
            // warp 同步
            __syncwarp();
        } else {
            // No local reduction, send all data (expanded send)
            // 无本地归约的全量展开发送：逐个 top-k 槽位发出
            #pragma unroll
            for (int k = 0; k < kNumTopk; ++ k) {
                // 通过 warp 交换得到第 k 个槽位的下标
                const auto slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                if (slot_idx >= 0) {
                    // 该槽位 token 在本地 x 中的地址（int4 视角）
                    const auto src_token_ptr = math::advance_ptr<int4>(x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                    // 目标 rank k 的接收缓冲区中该源 token 的位置
                    const auto token_buffer = recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx);
                    if (ptx::elect_one_sync()) {
                        // 唯一 lane 执行 TMA 拷贝流水
                        // Load
                        // 先从本地 x 加载到共享内存
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), src_token_ptr, mbarrier_ptr, kNumHiddenBytes);
                        // 设定 mbarrier 的传输字节计数
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        // 等待 load 完成
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        if (nvlink_bypass) {
                            // Write into the same position
                            // NVLink 直达：直接写入远端对称窗口中的同一位置
                            ptx::tma_store_1d(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx),
                                              tma_buffer.get_base_ptr(), kNumHiddenBytes);
                            // 提交 TMA store
                            ptx::tma_store_commit();
                        } else {
                            // Write to the RDMA send buffer
                            // RDMA 路径：先写入本地发送缓冲区的对应槽位
                            const auto send_token_buffer =
                                send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx * kNumTopk + k);
                            // 共享内存 -> 发送缓冲区
                            ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                            // 提交并等待落盘（对本地缓冲）
                            ptx::tma_store_commit();
                            ptx::tma_store_wait();

                            // Issue RDMA
                            // 通过 GIN 发起 RDMA put：发送缓冲区 -> 远端 token 缓冲
                            gin.put<team_t>(token_buffer.get_base_ptr(), send_token_buffer.get_base_ptr(),
                                            kNumHiddenBytes, src_rank_idx);
                        }
                    }
                    // warp 同步后再处理下一个槽位
                    __syncwarp();
                }
            }
        }

        // Write topk weights
        // 写出 top-k 权重（展开发送模式不写；无权重指针也不写）
        if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
            // 无效槽位的默认权重为 0
            float value = 0;
            if constexpr (kUseExpandedLayout) {
                if (stored_topk_slot_idx >= 0)
                    // 展开布局：lane 对应槽位有效时读出其权重
                    value = __ldg(topk_weights + stored_topk_slot_idx);
            } else {
                // 非展开布局：按 token 与 lane 定位权重
                value = __ldg(topk_weights + (i * kNumTopk + lane_idx));
            }
            // 将权重写入 master token 缓冲的 top-k 权重区
            master_token_buffer.get_topk_weights_ptr()[lane_idx] = value;
        }
        // warp 同步
        __syncwarp();

        // Wait send buffer's TMA store and issue RDMA send
        // NOTES: `kDoExpandedSend` mode has already issued
        // 等待发送缓冲区的 TMA store 完成后发起 RDMA 发送
        // NOTES: `kDoExpandedSend` 模式已在上面发过
        if (not kDoExpandedSend and not nvlink_bypass and ptx::elect_one_sync()) {
            // 等待本 warp 的 TMA store 排空
            ptx::tma_store_wait();
            // 计算远端接收缓冲区中目标位置（rank 布局按本 rank，否则按源 top-k 位置）
            const auto dst_ptr = recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx)
                .get_token_buffer(src_token_idx).get_base_ptr();
            // GIN RDMA put：本地发送缓冲 -> 远端接收缓冲
            gin.put<team_t>(dst_ptr, master_token_buffer.get_base_ptr(),
                            // 传输字节数取该 token 缓冲的实际大小（不含 scale）
                            master_token_buffer.get_num_bytes<false>(), src_rank_idx);
        }
    }

    // Final barrier to ensure data arrival
    // 结束屏障：确保所有 rank 的数据都已到达（kCombineTag1 阶段）
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag1, true, true, false>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::ep
// 结束 deep_ep::ep 命名空间
