#pragma once
// 头文件保护：避免重复包含

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
// 引入 GPU 栅栏协议、指针运算、PTX 封装、通用工具与 bucket 工作区布局

namespace deep_ep::bucket {
// DeepEP bucket 集合通信实现命名空间


// ============ 内核 1：纯 RDMA reduce-scatter（FP32 精度） ============
// 模板参数：kNumRanks 跨机 RDMA rank 数；kNumSMs 每 rank 的 SM 数；
// kNumIssueWarps 发送 warp 数；kNumReduceWarps 归约 warp 数；kNumQPs QP 总数；
// kNumTimeoutCycles 超时周期；kWithScale 是否乘缩放因子；kContextIdx 上下文编号；
// kNumIssueChunksPerBatch / kNumReduceChunksPerBatch 收发批次大小；其余为派生常量
template <int kNumRanks, int kNumSMs,
          int kNumIssueWarps, int kNumReduceWarps,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumIssueChunksPerBatch = 8,
          int kNumReduceChunksPerBatch = 8,
          int kNumWarps = kNumIssueWarps + kNumReduceWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumGlobalIssueWarps = kNumSMs * kNumIssueWarps,
          int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps,
          int kNumGlobalThreads = kNumSMs * kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限制每 SM 仅 1 个 CTA 驻留
rdma_reduce_scatter(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                    void* workspace_ptr, void* buffer,
                    const float scale,
                    const int num_buckets,
                    const __grid_constant__ comm::BucketList buckets) {
    // 跨机 reduce-scatter：发送 warp 按环序把本地 chunk RDMA put 到各对端的暂存槽位；
    // 归约 warp 分批（流水线）等待对端数据到达后，用 warp 级寄存器加法逐个累加进本地输出分片，
    // 并通过 put_value 把槽位信用（head）回传给发送方
    // Check slots
    // 检查槽位容量
    constexpr int kNumBytesPerChunk = 32 * 1024;
    // 编译期常量：每个 chunk 32KB
    constexpr int kNumElemsPerChunk = kNumBytesPerChunk / sizeof(float);
    // 每个 chunk 的 float 元素数
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumBytesPerChunk / kNumRanks;
    // 每个 rank 可用的暂存槽位数（环形流水线深度）

    EP_STATIC_ASSERT(kNumRanks * kNumSlotsPerRank <= layout::ReduceScatterSignals::kNumTotalSlots,
                     "Reduce-scatter slots exceed the signal workspace");
    // 编译期断言：信号槽总数不得超过信号工作区容量
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRDMARanks, "Too many RDMA ranks");
    // 编译期断言：RDMA rank 数不得超过上限
    EP_STATIC_ASSERT(kNumIssueChunksPerBatch > 0 and kNumIssueChunksPerBatch <= 32,
                     "Issue chunk batch size must be between 1 and 32");
    // 编译期断言：发送批次大小必须在 1~32 之间（受 warp lane 数限制）
    EP_STATIC_ASSERT(kNumReduceChunksPerBatch > 0 and kNumReduceChunksPerBatch <= 32,
                     "Reduce chunk batch size must be between 1 and 32");
    // 编译期断言：归约批次大小必须在 1~32 之间

    // Registers
    // 寄存器预算：不同角色的 warp 通过 warpgroup 重分配获得不同配额
    constexpr int kNumIssueRegisters = 112;
    // 发送 warp 每线程的寄存器配额
    constexpr int kNumReduceRegisters = 144;
    // 归约 warp 每线程的寄存器配额（需要更多寄存器做向量累加）
    constexpr int kNumInitialRegisters = (65536 / kNumThreads / 8) * 8;
    // 内核启动时的初始每线程寄存器数（按硬件上限与线程数折算）
    EP_STATIC_ASSERT(kNumIssueWarps % 4 == 0 and kNumReduceWarps % 4 == 0,
                     "Register roles must align to warp groups");
    // 编译期断言：两类 warp 数量必须按 4（warp 组）对齐
    EP_STATIC_ASSERT(kNumIssueWarps * kNumIssueRegisters + kNumReduceWarps * kNumReduceRegisters <=
                     kNumWarps * kNumInitialRegisters,
                     "Too many registers");
    // 编译期断言：重分配后的寄存器总量不得超过初始配额总量

    // Indices
    // 计算各类索引
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 CTA（SM）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内线程编号
    const auto lane_idx = ptx::get_lane_idx();
    // warp 内 lane 编号
    const auto warp_idx = ptx::get_warp_idx();
    // 块内 warp 编号（前 kNumIssueWarps 为发送 warp，其余为归约 warp）
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    // 本进程在铁路式 RDMA 团队中的 rank 编号

    // Init Gin
    // 初始化 GIN（RDMA 发起）通道
    // TODO: find a better and more explanable way
    // 待办：寻找更优雅、更易解释的 QP 分配方式
    constexpr int kNumSendQPsPerWarp = 4;
    // 每个发送 warp 使用 4 个 QP（lane 分片到不同 QP 以分散负载）
    const int qp_idx = [&] {
        // 立即调用 lambda 计算本 warp 的 QP 编号
        if (warp_idx < kNumIssueWarps) {
            // 发送 warp 分支
            // Send issue lanes stripe across the QPs assigned to their issue warp.
            // 发送 lane 在其所属发送 warp 分配到的 QP 间条带化（stripe）分布
            return ((sm_idx * kNumIssueWarps + warp_idx) * kNumSendQPsPerWarp +
                    lane_idx % kNumSendQPsPerWarp) % kNumQPs;
            // 按 (全局 warp 序号 * 每 warp QP 数 + lane 交替) 对 QP 总数取模
        }

        // Place credit QPs after send QPs, wrapping when the pool is shared.
        // 信用（归约）QP 排在发送 QP 之后；QP 池共享时取模回绕
        return (kNumGlobalIssueWarps * kNumSendQPsPerWarp + sm_idx * kNumReduceWarps + warp_idx - kNumIssueWarps) % kNumQPs;
        // 归减 warp 的 QP 编号在全体发送 QP 之后依次排列
    }();
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx);
    // 构造绑定到该 QP 的 GIN 封装

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    // 取得 bucket 工作区
    auto& bucket_signals = workspace.signals[kContextIdx];
    // 本上下文的信号区
    auto& barrier_signals = bucket_signals.barrier;
    // 栅栏信号数组
    auto& signals = bucket_signals.reduce_scatter;
    // reduce-scatter 专用的 head/tail 信用环
    const auto chunks = layout::ChunkView2D<kNumRanks, kNumSlotsPerRank, kNumBytesPerChunk>{workspace.chunk_storage};
    // chunk 存储二维视图：按 (rank, slot) 寻址

    // Every launch starts a fresh receive-credit ring.
    // 每次内核启动都要把接收信用环（head/tail）清零
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;
    // 全局线程编号
    #pragma unroll
    // 提示编译器展开清零循环
    for (int i = global_thread_idx; i < kNumRanks * kNumSlotsPerRank; i += kNumGlobalThreads) {
        // 全网格线程跨步覆盖所有 (rank, slot) 组合
        const auto j = i / kNumSlotsPerRank, k = i % kNumSlotsPerRank;
        // j 为 rank 编号，k 为槽位编号
        *signals.get_head_ptr<kNumSlotsPerRank>(j, k) = 0;
        // 清零该槽位的 head 计数器（消费进度）
        *signals.get_tail_ptr<kNumSlotsPerRank>(j, k) = 0;
        // 清零该槽位的 tail 计数器（到达进度）
    }
    comm::gpu_barrier<true, kNumRanks, 1,
                   kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                   kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
    // 跨 RDMA 域的入口 GPU 栅栏（flush 所有已分配 QP）：让清零后的信用可见并确保输入数据就绪

    // Different warp roles
    // 以下按 warp 编号划分两种角色
    if (warp_idx < kNumIssueWarps) {
        // ============ 角色 1：发送 warp ============
        ptx::warpgroup_reg_realloc<kNumIssueRegisters, kNumThreads>();
        // warpgroup 级寄存器重分配：发送 warp 使用较小配额

        // Issuing by a *result* view, using multiple lanes to issue at the same time
        // 以"结果视角"发起发送：多个 lane 同时向不同对端发出各自的数据块
        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalIssueWarps>(num_buckets, buckets, sm_idx * kNumIssueWarps + warp_idx);
        // chunk 迭代器：以全局发送 warp 数为步长遍历本 warp 负责的 chunk
        while (iterator.next()) {
            // 迭代下一个 chunk
            const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
            // 本 chunk 使用的暂存槽位（环形轮转）
            const auto head_target = iterator.chunk_idx / kNumSlotsPerRank;
            // head 目标值：该槽位此前被对端消费完的轮次
            const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);
            // 本 rank 该槽位的 tail 计数器（数据送达对端后由 put 信号自增）

            #pragma unroll 1
            // 禁止展开（批次与 rank 数相关）
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumIssueChunksPerBatch) {
                // 按批次遍历环序中的其他对端（peer_offset 从 1 开始，跳过自己）
                const int num_batch_chunks = min(kNumIssueChunksPerBatch, kNumRanks - peer_offset);
                // 本批实际的对端数（末批可能不满）
                const auto dst_rank_idx = (rank_idx + peer_offset + lane_idx) % kNumRanks;
                // 每个 lane 对应一个目的 rank（环序）
                const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(dst_rank_idx, slot_idx);
                // 目的 rank 消费进度 head（由对端归约 warp 回传信用）
                comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                    // 带超时的自旋等待；仅本批参与的对端 lane 需要等待
                    // NOTES: no need to do acquire
                    // 说明：此处无需 acquire 语义，只需观察信用变化
                    return ptx::ld_volatile<int64_t>(head_ptr) >= head_target;
                    // volatile 读 head，槽位已释放则返回 true
                });
                __syncwarp();
                // 同步 warp：等待结果对全 warp 可见

                // Issue RDMA
                // 发起 RDMA 发送
                if (lane_idx < num_batch_chunks) {
                    // 仅有效 lane 发送
                    const auto src_ptr = math::advance_ptr(buffer, iterator.bucket.offset + dst_rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);
                    // 源地址：目的 rank 的分片在本桶中对应 chunk 的位置
                    gin.put<ncclTeamTagRail>(
                        chunks.get_chunk_ptr(rank_idx, slot_idx),
                        // 目的：对端 chunk 存储区中本 rank 的该槽位
                        src_ptr, iterator.num_chunk_bytes, dst_rank_idx, 0,
                        ncclGin_StrongVASignalAdd {
                            .signalWindow = window,
                            // 信号落在对称窗口上
                            .signalOffset = gin.get_sym_offset(tail_ptr),
                            // 信号偏移指向本 rank 该槽位的 tail 计数器
                            .value = 1
                            // 数据送达后 tail 自增 1，通知对端可读
                        });
                }
                __syncwarp();
                // 同步 warp，本批发送完成
            }
        }
    } else {
        // ============ 角色 2：归约 warp ============
        ptx::warpgroup_reg_realloc<kNumReduceRegisters, kNumThreads>();
        // warpgroup 级寄存器重分配：归约 warp 使用较大配额

        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalReduceWarps>(num_buckets, buckets, sm_idx * kNumReduceWarps + warp_idx - kNumIssueWarps);
        // chunk 迭代器：以全局归约 warp 数为步长遍历本 warp 负责的 chunk
        while (iterator.next()) {
            // 迭代下一个待归约 chunk
            const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
            // 本 chunk 的暂存槽位
            const auto tail_target = iterator.chunk_idx / kNumSlotsPerRank + 1;
            // tail 目标值：该槽位本轮已被写入的总轮次
            const bool needs_credit = static_cast<int64_t>(iterator.chunk_idx) + kNumSlotsPerRank < iterator.num_total_chunks;
            // 该槽位之后是否还会被复用（决定是否需要回传信用）

            // Reduce
            // 寄存器归约
            EP_STATIC_ASSERT(kNumElemsPerChunk % 256 == 0, "Unaligned chunk");
            // 编译期断言：chunk 元素数必须按 256 对齐（warp 级迭代的最小粒度）
            constexpr int kNumVecsPerIter = 8;
            // 每次迭代每个线程处理的 float4 向量数
            EP_STATIC_ASSERT(kNumElemsPerChunk % (32 * kNumVecsPerIter * 4) == 0, "Unaligned reduction iteration");
            // 编译期断言：归约迭代必须与 chunk 大小对齐
            const int num_vecs = iterator.num_chunk_bytes / static_cast<int>(sizeof(float4));
            // 实际向量数（桶尾 chunk 可能不满）
            const bool full_chunk = iterator.num_chunk_bytes == kNumBytesPerChunk;
            // 是否为完整 chunk（决定是否需要越界谓词保护）
            const auto dst_ptr = math::advance_ptr<float4>(buffer, iterator.bucket.offset + rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);
            // 输出地址：本 rank 分片在缓冲中的对应 chunk 位置
            #pragma unroll 1
            // 禁止展开
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumReduceChunksPerBatch) {
                // 按批次遍历环序中的其他源 rank
                const int num_batch_chunks = min(kNumReduceChunksPerBatch, kNumRanks - peer_offset);
                // 本批实际的源数
                const bool is_last_batch = peer_offset + num_batch_chunks == kNumRanks;
                // 是否为最后一批（决定何时乘 scale）

                // Wait arrival in reverse cyclic order
                // 按逆环序等待数据到达（与发送方向相反，利于流水线）
                const auto wait_rank_idx = (rank_idx - peer_offset - (lane_idx % kNumRanks) + 2 * kNumRanks) % kNumRanks;
                // 本 lane 等待的源 rank（逆环序）
                const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(wait_rank_idx, slot_idx);
                // 源 rank 该槽位的 tail 计数器
                comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                    // 带超时的自旋等待；仅本批参与的对端 lane 需要等待
                    return ptx::ld_acquire_sys(tail_ptr) >= tail_target;
                    // acquire（系统级）读 tail，保证看到对端写入的数据
                });
                __syncwarp();
                // 同步 warp

                #pragma unroll 1
                // 禁止展开
                for (int i = 0; i < kNumElemsPerChunk / 4 / (32 * kNumVecsPerIter); ++ i) {
                    // 逐轮遍历 chunk 内的向量（每轮每 lane 处理 kNumVecsPerIter 个 float4）
                    const auto vec_idx = i * 32 * kNumVecsPerIter + lane_idx;
                    // 本 lane 的向量基索引
                    float4 values[kNumVecsPerIter];
                    // 累加寄存器组
                    if (full_chunk) {
                        // 完整 chunk：无需越界检查
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * 32);
                            // 读出本地部分和（各 lane 相距 32 向量交错布局）
                    } else {
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * 32, num_vecs, vec_idx + k * 32);
                            // 带谓词读出（越界部分被屏蔽）
                    }

                    #pragma unroll 1
                    // 禁止展开
                    for (int j = 0; j < num_batch_chunks; ++ j) {
                        // 逐个累加本批各源 rank 的数据
                        const auto src_rank_idx = (rank_idx - peer_offset - j + 2 * kNumRanks) % kNumRanks;
                        // 逆环序中的源 rank
                        const auto src_ptr = static_cast<const float4*>(
                            chunks.get_chunk_ptr(src_rank_idx, slot_idx)) + vec_idx;
                        // 源地址：该源 rank 槽位中的 chunk（向量视图）
                        float4 packed_values[kNumVecsPerIter];
                        // 本源 rank 的暂存寄存器组
                        if (full_chunk) {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * 32);
                                // 读出源数据（无越界检查）
                        } else {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * 32, num_vecs, vec_idx + k * 32);
                                // 带谓词读出源数据
                        }

                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::fadd4(values[k], packed_values[k]);
                            // 逐元素 float4 加法累加进寄存器
                    }

                    // Scale
                    // 缩放
                    if constexpr (kWithScale) {
                        // 编译期开关：仅在需要缩放时包含此代码
                        if (is_last_batch) {
                            // 只在最后一批累加完后乘一次 scale
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                values[k] = ptx::fmul4(values[k], scale);
                                // 逐元素乘以 scale
                        }
                    }

                    // Store
                    // 写回
                    if (full_chunk) {
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * 32, values[k]);
                            // 普通写回归约结果
                    } else {
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * 32, values[k], num_vecs, vec_idx + k * 32);
                            // 带谓词写回（越界部分被屏蔽）
                    }
                }
                __syncwarp();
                // 同步 warp，本批归约完成

                // Release slots of this batch; the final use of a slot needs no credit
                // 释放本批使用的槽位（回传信用）；槽位最后一次使用无需回传信用
                if (needs_credit) {
                    // 该槽位还会被后续 chunk 复用时才需要回传
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);
                    // 本 rank 该槽位的 head 计数器
                    const auto release_rank_idx = (rank_idx - peer_offset - (lane_idx % kNumRanks) + 2 * kNumRanks) % kNumRanks;
                    // 本 lane 对应的源 rank（信用要发给数据的发送方）
                    if (lane_idx < num_batch_chunks)
                        gin.put_value<ncclTeamTagRail>(head_ptr, static_cast<int64_t>(tail_target), release_rank_idx);
                        // 用单值 RDMA put 把 tail_target 写入发送方的 head（释放其槽位）
                }
                __syncwarp();
                // 同步 warp
            }
        }
    }

    // Ensure no later writes
    // 出口栅栏：确保不再有更晚的写入（所有 put 已全局可见）
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
    // 跨 RDMA 域的出口 GPU 栅栏（带 QP flush 语义）
}

// ============ 内核 2：纯 RDMA reduce-scatter（FP32 输入经确定性随机舍入压缩为 BF16 传输） ============
// 模板参数：kNumCastWarps 类型转换 warp 数；kNumIssueWarps 发送 warp 数；kNumReduceWarps 归约 warp 数；
// 其余含义同上；kNumCastWarpsPerGroup / kNumReduceWarpsPerGroup 为组内 warp 数
template <int kNumRanks, int kNumSMs,
          int kNumCastWarps, int kNumIssueWarps, int kNumReduceWarps,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumIssueChunksPerBatch = 8,
          int kNumReduceChunksPerBatch = 8,
          int kNumCastWarpsPerGroup = 4,
          int kNumReduceWarpsPerGroup = 4,
          int kNumWarps = kNumCastWarps + kNumIssueWarps + kNumReduceWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumCastGroups = kNumCastWarps / kNumCastWarpsPerGroup,
          int kNumReduceGroups = kNumReduceWarps / kNumReduceWarpsPerGroup,
          int kNumCastThreadsPerGroup = kNumCastWarpsPerGroup * 32,
          int kNumReduceThreadsPerGroup = kNumReduceWarpsPerGroup * 32,
          int kNumGlobalThreads = kNumSMs * kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限制每 SM 仅 1 个 CTA 驻留
rdma_reduce_scatter_bf16(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                    void* workspace_ptr, void* buffer,
                    const float scale,
                    const int num_buckets,
                    const __grid_constant__ comm::BucketList buckets) {
    // 带宽优化的 RDMA reduce-scatter：传输前用确定性随机舍入（stochastic rounding）把 FP32
    // chunk 打包成 BF16（数据量减半），对端以 FP32 精度累加 BF16 输入；
    // 角色：cast warp 负责压缩、issue warp 负责发送/信用、reduce warp 负责累加
    // Check slots
    // 检查槽位容量
    constexpr int kNumCommBytesPerChunk = (kNumRanks <= 8 ? 64 : 32) * 1024;
    // 编译期常量：每个 chunk 的"通信（BF16）"字节数
    constexpr int kNumBytesPerChunk = 2 * kNumCommBytesPerChunk;
    // 对应的 FP32 逻辑 chunk 字节数（是通信字节的 2 倍）
    constexpr int kNumElemsPerChunk = kNumBytesPerChunk / sizeof(float);
    // 每个 FP32 chunk 的元素数
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumCommBytesPerChunk / kNumRanks;
    // 每个 rank 可用的暂存槽位数（按通信字节数计）

    EP_STATIC_ASSERT(kNumRanks * kNumSlotsPerRank <= layout::ReduceScatterSignals::kNumTotalSlots,
                     "Reduce-scatter slots exceed the signal workspace");
    // 编译期断言：信号槽总数不得超过信号工作区容量
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRDMARanks, "Too many RDMA ranks");
    // 编译期断言：RDMA rank 数不得超过上限
    EP_STATIC_ASSERT(kNumIssueChunksPerBatch % kNumCastWarpsPerGroup == 0,
                     "Issue chunk batch size must be evenly divided across cast warps in a group");
    // 编译期断言：发送批次必须能被组内 cast warp 数整除
    EP_STATIC_ASSERT(kNumIssueChunksPerBatch > 0 and kNumIssueChunksPerBatch <= 32,
                     "Issue chunk batch size must be between 1 and 32");
    // 编译期断言：发送批次大小必须在 1~32 之间
    EP_STATIC_ASSERT(kNumReduceChunksPerBatch > 0 and kNumReduceChunksPerBatch <= 32,
                     "Reduce chunk batch size must be between 1 and 32");
    // 编译期断言：归约批次大小必须在 1~32 之间
    EP_STATIC_ASSERT(kNumCastWarpsPerGroup > 0 and kNumCastWarps % kNumCastWarpsPerGroup == 0,
                     "Cast warps must be evenly divided across groups");
    // 编译期断言：cast warp 必须能被均匀分组
    EP_STATIC_ASSERT(kNumReduceWarpsPerGroup > 0 and kNumReduceWarps % kNumReduceWarpsPerGroup == 0,
                     "Reduce warps must be evenly divided across groups");
    // 编译期断言：归约 warp 必须能被均匀分组
    EP_STATIC_ASSERT(kNumIssueWarps == kNumCastGroups + kNumReduceGroups,
                     "Each cast and reduce group requires one issue warp");
    // 编译期断言：每个 cast 组与每个归约组各需一个发送 warp（前者发数据、后者管信用）
    constexpr int kNumGlobalCastGroups = (kNumSMs * kNumCastWarps) / kNumCastWarpsPerGroup;
    // 全局 cast 组总数
    constexpr int kNumGlobalReduceGroups = (kNumSMs * kNumReduceWarps) / kNumReduceWarpsPerGroup;
    // 全局归约组总数

    // Advancing to the next chunk must not depend on the credit delayed until its barrier.
    // 推进到下一个 chunk 时不能依赖被延迟到命名屏障之后才归还的信用。
    // This also guarantees that each group's final chunk needs no returned credits.
    // 这同时保证每个组最后一个 chunk 不需要归还信用。
    EP_STATIC_ASSERT(kNumGlobalReduceGroups < kNumSlotsPerRank, "Pipelined reduce credits require more slots than global reduce groups");
    // 编译期断言：流水线化的信用归还要求槽位数多于全局归约组数

    // Registers
    // 寄存器预算
    constexpr int kNumCastRegisters = 80;
    // cast warp 每线程的寄存器配额
    constexpr int kNumIssueRegisters = 80;
    // 发送 warp 每线程的寄存器配额
    constexpr int kNumReduceRegisters = 120;
    // 归约 warp 每线程的寄存器配额
    constexpr int kNumInitialRegisters = (65536 / kNumThreads / 8) * 8;
    // 内核启动时的初始每线程寄存器数
    EP_STATIC_ASSERT(kNumCastWarps % 4 == 0 and kNumIssueWarps % 4 == 0 and kNumReduceWarps % 4 == 0,
                     "Register roles must align to warp groups");
    // 编译期断言：三类 warp 数量必须按 4（warp 组）对齐
    EP_STATIC_ASSERT(kNumCastWarps * kNumCastRegisters + kNumIssueWarps * kNumIssueRegisters +
                     kNumReduceWarps * kNumReduceRegisters <= kNumWarps * kNumInitialRegisters,
                     "Too many registers");
    // 编译期断言：重分配后的寄存器总量不得超过初始配额总量

    // Barriers
    // 命名屏障：cast 组与归约组分别与各自的发送 warp 同步
    constexpr int kCastBarrierBaseIdx = 1;
    // cast 组命名屏障的起始编号
    constexpr int kReduceBarrierBaseIdx = kCastBarrierBaseIdx + kNumCastGroups;
    // 归约组命名屏障的起始编号（排在 cast 之后）
    EP_STATIC_ASSERT(kCastBarrierBaseIdx + kNumCastGroups + kNumReduceGroups - 1 < 16, "Too many cast/reduce groups for named barriers");
    // 编译期断言：命名屏障编号必须小于 16（硬件限制）

    // Indices
    // 计算各类索引
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 CTA（SM）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内线程编号
    const auto lane_idx = ptx::get_lane_idx();
    // warp 内 lane 编号
    const auto warp_idx = ptx::get_warp_idx();
    // 块内 warp 编号（决定角色）
    const auto global_warp_idx = sm_idx * kNumWarps + warp_idx;
    // 全局 warp 编号
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    // 本进程在铁路式 RDMA 团队中的 rank 编号

    // Init Gin
    // 初始化 GIN（RDMA 发起）通道
    // TODO: find a better and more explanable way
    // 待办：寻找更优雅、更易解释的 QP 分配方式
    constexpr int kNumSendQPsPerWarp = 4;
    // 每个发送 warp 使用 4 个 QP
    const int qp_idx = [&] {
        // 立即调用 lambda 计算本 warp 的 QP 编号
        if (warp_idx >= kNumCastWarps and warp_idx < kNumCastWarps + kNumCastGroups) {
            // 数据发送 warp 分支（紧随 cast warp 之后）
            // Send issue lanes stripe across the QPs assigned to their cast group.
            // 发送 lane 在其所属 cast 组分配到的 QP 间条带化分布
            const int cast_group_idx = warp_idx - kNumCastWarps;
            // 本发送 warp 服务的 cast 组编号
            return ((sm_idx * kNumCastGroups + cast_group_idx) * kNumSendQPsPerWarp +
                    lane_idx % kNumSendQPsPerWarp) % kNumQPs;
            // 按 (全局 cast 组序号 * 每 warp QP 数 + lane 交替) 对 QP 总数取模
        }

        if (warp_idx >= kNumCastWarps + kNumCastGroups and warp_idx < kNumCastWarps + kNumIssueWarps) {
            // 信用发送 warp 分支（服务归约组）
            // Place credit QPs after send QPs, wrapping when the pool is shared.
            // 信用 QP 排在数据发送 QP 之后；QP 池共享时取模回绕
            const int reduce_group_idx = warp_idx - kNumCastWarps - kNumCastGroups;
            // 本信用 warp 服务的归约组编号
            return (kNumGlobalCastGroups * kNumSendQPsPerWarp + sm_idx * kNumReduceGroups + reduce_group_idx) % kNumQPs;
            // 在全体数据发送 QP 之后依次排列
        }

        return 0;
        // cast warp 不做 RDMA 发送，使用 QP 0
    }();
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx);
    // 构造绑定到该 QP 的 GIN 封装

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    // 取得 bucket 工作区
    auto& bucket_signals = workspace.signals[kContextIdx];
    // 本上下文的信号区
    auto& barrier_signals = bucket_signals.barrier;
    // 栅栏信号数组
    auto& signals = bucket_signals.reduce_scatter;
    // reduce-scatter 专用的 head/tail 信用环
    const auto chunks = layout::ChunkView2D<kNumRanks, kNumSlotsPerRank, kNumCommBytesPerChunk>{workspace.chunk_storage};
    // chunk 存储二维视图（按 BF16 通信字节数寻址）

    // Every launch starts a fresh receive-credit ring.
    // 每次内核启动都要把接收信用环（head/tail）清零
    #pragma unroll
    // 提示编译器展开清零循环
    for (int i = sm_idx * kNumThreads + thread_idx; i < kNumRanks * kNumSlotsPerRank; i += kNumGlobalThreads) {
        // 全网格线程跨步覆盖所有 (rank, slot) 组合
        const auto j = i / kNumSlotsPerRank, k = i % kNumSlotsPerRank;
        // j 为 rank 编号，k 为槽位编号
        *signals.get_head_ptr<kNumSlotsPerRank>(j, k) = 0;
        // 清零该槽位的 head 计数器
        *signals.get_tail_ptr<kNumSlotsPerRank>(j, k) = 0;
        // 清零该槽位的 tail 计数器
    }
    comm::gpu_barrier<true, kNumRanks, 1,
                   kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                   kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
    // 跨 RDMA 域的入口 GPU 栅栏（flush 所有已分配 QP）

    // Different warp roles
    // 以下按 warp 编号划分三种角色
    if (warp_idx < kNumCastWarps) {
        // ============ 角色 1：cast warp（FP32 → BF16 确定性随机舍入压缩） ============
        ptx::warpgroup_reg_realloc<kNumCastRegisters, kNumThreads>();
        // warpgroup 级寄存器重分配

        // A group owns a chunk offset; its cast warps keep disjoint destination batches.
        // 每个 cast 组负责一个 chunk 偏移；组内各 cast warp 处理互不重叠的目的批次
        const int cast_group_idx = warp_idx / kNumCastWarpsPerGroup;
        // 本 warp 所属 cast 组编号
        const int cast_warp_idx_in_group = warp_idx % kNumCastWarpsPerGroup;
        // 组内 cast warp 编号
        constexpr int kNumCastChunksPerBatchPerWarp = kNumIssueChunksPerBatch / kNumCastWarpsPerGroup;
        // 组内每个 cast warp 负责的每批 chunk 数
        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalCastGroups>(num_buckets, buckets, sm_idx * kNumCastGroups + cast_group_idx);
        // chunk 迭代器：以全局 cast 组数为步长遍历本组负责的 chunk
        while (iterator.next()) {
            // 迭代下一个 chunk
            const auto chunk_offset_bytes = iterator.chunk_offset_in_bucket;
            // 本 chunk 在桶内的字节偏移
            const int num_chunk_bytes = iterator.num_chunk_bytes;
            // 本 chunk 的 FP32 字节数
            const bool full_chunk = num_chunk_bytes == kNumBytesPerChunk;
            // 是否为完整 chunk
            const auto num_bucket_bytes = iterator.bucket.num_bytes;
            // 该桶每 rank 分片的字节数

            #pragma unroll 1
            // 禁止展开
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumIssueChunksPerBatch) {
                // 按批次遍历环序中的其他目的 rank
                const int i = peer_offset + cast_warp_idx_in_group * kNumCastChunksPerBatchPerWarp;
                // 本 cast warp 负责的目的 rank 起始偏移
                const int num_batch_chunks = max(0, min(kNumCastChunksPerBatchPerWarp, min(kNumRanks, peer_offset + kNumIssueChunksPerBatch) - i));
                // 本 cast warp 实际负责的目的数（可能为 0 或不足）

                // Convert this warp's destination chunks before handing the group batch to its issue warp.
                // 在把整批交给组内发送 warp 之前，先完成本 warp 各目的 chunk 的类型转换
                #pragma unroll 1
                // 禁止展开
                for (int batch_idx = 0; batch_idx < num_batch_chunks; ++ batch_idx) {
                    // 逐个处理本 warp 负责的目的 chunk
                    const auto dst_rank_idx = (rank_idx + i + batch_idx) % kNumRanks;
                    // 环序中的目的 rank

                    // Pack this destination's FP32 chunk into BF16 in its first half.
                    // 把该目的 FP32 chunk 的 BF16 打包结果写回其前半部分（原地压缩，节省缓冲）
                    constexpr int kNumVecsPerIter = 8;
                    // 每次迭代每 lane 处理的 longlong4（32 字节，8 个 float）向量数
                    EP_STATIC_ASSERT(kNumVecsPerIter > 0 and kNumElemsPerChunk % (256 * kNumVecsPerIter) == 0,
                                     "Unaligned send iteration");
                    // 编译期断言：发送迭代必须与 chunk 大小对齐
                    const int num_vecs = num_chunk_bytes / static_cast<int>(sizeof(longlong4_t));
                    // 实际的 longlong4 向量数（BF16 视图）
                    const auto chunk_ptr = math::advance_ptr<longlong4_t>(buffer, iterator.bucket.offset + dst_rank_idx * num_bucket_bytes + chunk_offset_bytes);
                    // 该目的 chunk 的基址（longlong4 视图）
                    const uint32_t global_vec_idx_base = static_cast<uint32_t>((dst_rank_idx * num_bucket_bytes + chunk_offset_bytes) / static_cast<int64_t>(sizeof(longlong4_t)));
                    // 该 chunk 在全局缓冲中的向量基索引（用于推导舍入随机位）
                    #pragma unroll 1
                    // 禁止展开
                    for (int j = 0; j < kNumElemsPerChunk / 256; j += kNumVecsPerIter) {
                        // 逐轮遍历 chunk 内的向量
                        const auto vec_idx = j * 32 + lane_idx;
                        // 本 lane 的向量索引（各 lane 相距 32 交错）
                        const uint32_t global_vec_idx = global_vec_idx_base + static_cast<uint32_t>(vec_idx);
                        // 全局向量索引
                        const int num_remaining_vecs = num_vecs - vec_idx;
                        // 剩余向量数（用于越界保护）
                        longlong4_t values[kNumVecsPerIter];
                        // 读出的原始 FP32 数据（longlong4 = 4 个 float4）
                        if (full_chunk) {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++k)
                                values[k] = ptx::ld_with_gt_pred(chunk_ptr + vec_idx + k * 32);
                                // 读出原始数据（无越界检查）
                        } else {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                values[k] = ptx::ld_with_gt_pred(chunk_ptr + vec_idx + k * 32, num_remaining_vecs, k * 32);
                                // 带谓词读出（越界部分被屏蔽）
                        }

                        float4 packed_values[kNumVecsPerIter];
                        // BF16 打包结果（每个 longlong4 压缩为一个 float4）
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k) {
                            #pragma unroll
                            // 展开内层循环
                            for (int half_idx = 0; half_idx < 2; ++ half_idx) {
                                // 每个 longlong4 分成两个 float4 半部处理
                                const auto& value = reinterpret_cast<const float4*>(&values[k])[half_idx];
                                // 取第 half_idx 个 float4
                                // Derive rounding bits from the FP32 bit patterns and bucket-relative float4 vector index.
                                // 由 FP32 位模式与桶内 float4 向量索引推导舍入随机位。
                                const auto& bits = reinterpret_cast<const uint4&>(value);
                                // 把 float4 按 4 个 uint32 位模式解释
                                uint32_t h = bits.x * 0x5671D42Bu + bits.y * 0x9995E499u +
                                             bits.z * 0xACE1B8A5u + bits.w * 0xE153538Du +
                                             global_vec_idx * 2 + static_cast<uint32_t>(k * 2 * 32) + static_cast<uint32_t>(half_idx);
                                // 对数据位与位置索引做乘法混合（SplitMix 风格哈希）
                                h ^= h >> 23;
                                // 哈希混淆第一步
                                h *= 0x7FEB352Du;
                                // 哈希混淆第二步
                                h ^= h >> 16;
                                // 哈希混淆第三步
                                uint32_t h1 = h * 0x846CA68Bu;
                                // 派生前一对 float 的随机位
                                h1 ^= h1 >> 11;
                                // 再混淆
                                uint32_t h2 = h * 0xD35A2D97u;
                                // 派生另一对 float 的随机位
                                h2 ^= h2 >> 11;
                                // 再混淆

                                // Use separate rounding bits for the two FP32 pairs.
                                // 为两个 FP32 数对使用不同的舍入随机位。
                                const auto xy = ptx::cvt_rs_bf16x2(make_float2(value.x, value.y), h1);
                                // 带随机舍入地把 (x,y) 转成 2 个 BF16
                                const auto zw = ptx::cvt_rs_bf16x2(make_float2(value.z, value.w), h2);
                                // 带随机舍入地把 (z,w) 转成 2 个 BF16
                                auto& packed = reinterpret_cast<uint2*>(&packed_values[k])[half_idx];
                                // 打包结果中的第 half_idx 个 uint2（4 个 BF16）
                                packed.x = reinterpret_cast<const uint32_t&>(xy);
                                // 存入 xy 的 BF16 对
                                packed.y = reinterpret_cast<const uint32_t&>(zw);
                                // 存入 zw 的 BF16 对
                            }
                        }

                        // Every lane must finish reading before any lane overwrites FP32 data.
                        // 在任何 lane 覆写 FP32 数据之前，所有 lane 必须已完成读取。
                        __syncwarp();
                        // warp 同步：避免读-写竞争（原位压缩）

                        if (full_chunk) {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                ptx::st_with_gt_pred(reinterpret_cast<float4*>(chunk_ptr) + vec_idx + k * 32, packed_values[k]);
                                // 把 BF16 打包结果写回 chunk 前半部分（无越界检查）
                        } else {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                ptx::st_with_gt_pred(reinterpret_cast<float4*>(chunk_ptr) + vec_idx + k * 32, packed_values[k], num_remaining_vecs, k * 32);
                                // 带谓词写回（越界部分被屏蔽）
                        }
                    }
                    __syncwarp();
                    // warp 同步：本目的 chunk 压缩完成
                }

                ptx::named_barrier_unaligned<kNumCastThreadsPerGroup + 32>(kCastBarrierBaseIdx + cast_group_idx);
                // 命名屏障：通知组内发送 warp 整批压缩完成（cast 组全部线程 + 32 个发送线程）
            }
        }
    } else if (warp_idx < kNumCastWarps + kNumIssueWarps) {
        // ============ 角色 2：发送 warp（数据发送与信用归还各司其职） ============
        ptx::warpgroup_reg_realloc<kNumIssueRegisters, kNumThreads>();
        // warpgroup 级寄存器重分配

        const int issue_warp_idx = warp_idx - kNumCastWarps;
        // 块内发送 warp 编号
        if (issue_warp_idx < kNumCastGroups) {
            // ---- 数据发送 warp：负责把 cast 组压缩好的 BF16 chunk 发出去 ----
            // One send issue warp services all destination batches from its cast group.
            // 一个数据发送 warp 服务其 cast 组的所有目的批次
            const int cast_group_idx = issue_warp_idx;
            // 本发送 warp 配对的 cast 组编号

            auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalCastGroups>(num_buckets, buckets, sm_idx * kNumCastGroups + cast_group_idx);
            // chunk 迭代器：与配对 cast 组相同的 chunk 序列
            while (iterator.next()) {
                // 迭代下一个 chunk
                const auto chunk_offset_bytes = iterator.chunk_offset_in_bucket;
                // 本 chunk 在桶内的字节偏移
                const int num_comm_chunk_bytes = iterator.num_chunk_bytes / 2;
                // BF16 通信字节数（FP32 的一半）
                const auto num_bucket_bytes = iterator.bucket.num_bytes;
                // 该桶每 rank 分片的字节数
                const auto global_chunk_idx = iterator.chunk_idx;
                // 全局 chunk 序号
                const auto slot_idx = global_chunk_idx % kNumSlotsPerRank;
                // 本 chunk 使用的暂存槽位
                const auto head_target = global_chunk_idx / kNumSlotsPerRank;
                // head 目标值：该槽位此前被对端消费完的轮次
                const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);
                // 本 rank 该槽位的 tail 计数器

                #pragma unroll 1
                // 禁止展开
                for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumIssueChunksPerBatch) {
                    // 按批次遍历环序中的其他目的 rank
                    ptx::named_barrier_unaligned<kNumCastThreadsPerGroup + 32>(kCastBarrierBaseIdx + cast_group_idx);
                    // 等待 cast 组完成本批压缩（与 cast 组在命名屏障会合）

                    // All peers in this cast group batch use distinct lanes of the issue warp.
                    // 本批中 cast 组的所有目的 rank 由发送 warp 的不同 lane 分头发送
                    const int num_batch_chunks = min(kNumIssueChunksPerBatch, kNumRanks - peer_offset);
                    // 本批实际的目的数
                    const auto dst_rank_idx = (rank_idx + peer_offset + lane_idx) % kNumRanks;
                    // 每个 lane 对应一个目的 rank
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(dst_rank_idx, slot_idx);
                    // 目的 rank 消费进度 head
                    comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                        // 带超时的自旋等待；仅有效 lane 需要等待
                        // NOTES: no need to do acquire
                        // 说明：此处无需 acquire 语义，只需观察信用变化
                        return ptx::ld_volatile<int64_t>(head_ptr) >= head_target;
                        // volatile 读 head，槽位已释放则返回 true
                    });
                    __syncwarp();
                    // 同步 warp

                    if (lane_idx < num_batch_chunks) {
                        // 仅有效 lane 发送
                        const auto src_ptr = math::advance_ptr(buffer, iterator.bucket.offset + dst_rank_idx * num_bucket_bytes + chunk_offset_bytes);
                        // 源地址：目的 rank 分片中该 chunk 的 BF16 前半部分
                        gin.put<ncclTeamTagRail>(
                            chunks.get_chunk_ptr(rank_idx, slot_idx),
                            // 目的：对端 chunk 存储区中本 rank 的该槽位
                            src_ptr, num_comm_chunk_bytes, dst_rank_idx, 0,
                            ncclGin_StrongVASignalAdd {
                                .signalWindow = window,
                                // 信号落在对称窗口上
                                .signalOffset = gin.get_sym_offset(tail_ptr),
                                // 信号偏移指向本 rank 该槽位的 tail 计数器
                                .value = 1
                                // 数据送达后 tail 自增 1
                            });
                    }
                    __syncwarp();
                    // 同步 warp，本批发送完成
                }
            }
        } else {
            // ---- 信用发送 warp：等待数据到达并把上一批的信用（head）回传给发送方 ----
            // One receive issue warp waits and returns credits for its reduce group.
            // 一个信用 warp 为其归约组等待数据到达并归还信用
            const int reduce_group_idx = issue_warp_idx - kNumCastGroups;
            // 本信用 warp 服务的归约组编号
            auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalReduceGroups>(num_buckets, buckets, sm_idx * kNumReduceGroups + reduce_group_idx);
            // chunk 迭代器：与配对归约组相同的 chunk 序列
            bool has_previous_credit = false;
            // 上一批是否存在待归还的信用
            int previous_src_rank_idx = 0;
            // 上一批信用对应的源 rank
            int64_t previous_tail_target = 0;
            // 上一批信用对应的 tail 目标值
            int64_t* previous_head_ptr = nullptr;
            // 上一批信用要写入的 head 地址
            while (iterator.next()) {
                // 迭代下一个 chunk
                const auto global_chunk_idx = iterator.chunk_idx;
                // 全局 chunk 序号
                const auto slot_idx = global_chunk_idx % kNumSlotsPerRank;
                // 本 chunk 的暂存槽位
                const auto tail_target = global_chunk_idx / kNumSlotsPerRank + 1;
                // tail 目标值：该槽位本轮已被写入的总轮次
                const bool needs_credit = static_cast<int64_t>(global_chunk_idx) + kNumSlotsPerRank < iterator.num_total_chunks;
                // 该槽位之后是否还会被复用
                const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);
                // 本 rank 该槽位的 head 计数器（信用回传目的地）

                #pragma unroll 1
                // 禁止展开
                for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumReduceChunksPerBatch) {
                    // 按批次遍历环序中的其他源 rank
                    const int num_batch_chunks = min(kNumReduceChunksPerBatch, kNumRanks - peer_offset);
                    // 本批实际的源数
                    const auto src_rank_idx = (rank_idx - peer_offset - (lane_idx % kNumRanks) + 2 * kNumRanks) % kNumRanks;
                    // 本 lane 等待的源 rank（逆环序，与归约组一致）
                    const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(src_rank_idx, slot_idx);
                    // 源 rank 该槽位的 tail 计数器
                    comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                        // 带超时的自旋等待数据到达
                        return ptx::ld_acquire_sys(tail_ptr) >= tail_target;
                        // acquire（系统级）读 tail
                    });
                    __syncwarp();
                    // 同步 warp

                    // Publish this batch's arrival and confirm the previous batch is consumed.
                    // 发布本批到达消息，并确认上一批已被归约组消费。
                    ptx::named_barrier_unaligned<kNumReduceThreadsPerGroup + 32>(kReduceBarrierBaseIdx + reduce_group_idx);
                    // 与归约组在命名屏障会合（归约组可安全使用本批数据）

                    if (has_previous_credit)
                        gin.put_value<ncclTeamTagRail>(previous_head_ptr, previous_tail_target, previous_src_rank_idx);
                        // 用单值 RDMA put 把上一批的信用（tail_target）归还给其发送方
                    __syncwarp();
                    // 同步 warp

                    // Keep the previous batch's slot and generation even across chunk/bucket boundaries.
                    // 即使跨越 chunk/桶边界，也保留上一批的槽位与代次（信用延迟一批归还）。
                    has_previous_credit = needs_credit and lane_idx < num_batch_chunks;
                    // 记录本批是否有待归还的信用（本 lane 有效且槽位会被复用）
                    previous_head_ptr = head_ptr;
                    // 记录本批的 head 地址
                    previous_tail_target = static_cast<int64_t>(tail_target);
                    // 记录本批的 tail 目标值
                    previous_src_rank_idx = src_rank_idx;
                    // 记录本批的源 rank
                }
            }
            // The last owned chunk cannot reuse its slot, so no final credit or group barrier is needed.
            // 最后一个 chunk 不会复用其槽位，因此无需归还最终信用或再做组屏障。
        }
    } else {
        // ============ 角色 3：归约 warp（FP32 精度累加 BF16 输入） ============
        ptx::warpgroup_reg_realloc<kNumReduceRegisters, kNumThreads>();
        // warpgroup 级寄存器重分配

        const int reduce_warp_idx = warp_idx - kNumCastWarps - kNumIssueWarps;
        // 块内归约 warp 编号
        const int reduce_warp_idx_in_group = reduce_warp_idx % kNumReduceWarpsPerGroup;
        // 组内归约 warp 编号
        const int reduce_warpgroup_idx = reduce_warp_idx / kNumReduceWarpsPerGroup;
        // 本 warp 所属归约组编号
        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalReduceGroups>(num_buckets, buckets, sm_idx * kNumReduceGroups + reduce_warpgroup_idx);
        // chunk 迭代器：以全局归约组数为步长遍历本组负责的 chunk
        while (iterator.next()) {
            // 迭代下一个待归约 chunk
            const auto chunk_offset_bytes = iterator.chunk_offset_in_bucket;
            // 本 chunk 在桶内的字节偏移
            const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
            // 本 chunk 的暂存槽位
            const bool full_chunk = iterator.num_chunk_bytes == kNumBytesPerChunk;
            // 是否为完整 chunk

            // Reduce
            // 寄存器归约
            constexpr int kNumVecsPerIter = 8;
            // 每次迭代每线程处理的 longlong4 向量数
            constexpr int kNumThreadsPerReduceGroup = kNumReduceWarpsPerGroup * 32;
            // 归约组内线程总数
            EP_STATIC_ASSERT(kNumVecsPerIter > 0 and kNumElemsPerChunk % (256 * kNumVecsPerIter * kNumReduceWarpsPerGroup) == 0,
                             "Unaligned reduction iteration");
            // 编译期断言：归约迭代必须与 chunk 大小对齐
            const int num_vecs = iterator.num_chunk_bytes / static_cast<int>(sizeof(longlong4_t));
            // 实际的 longlong4 向量数
            const auto num_bucket_bytes = iterator.bucket.num_bytes;
            // 该桶每 rank 分片的字节数
            const auto dst_ptr = math::advance_ptr<longlong4_t>(buffer, iterator.bucket.offset + rank_idx * num_bucket_bytes + chunk_offset_bytes);
            // 输出地址：本 rank 分片中该 chunk 的位置（FP32，longlong4 视图）

            // Reduce remote peers in reverse cyclic order, storing partial FP32 sums between batches.
            // 按逆环序归约远端对端，批次之间把部分 FP32 和保存在输出缓冲中。
            #pragma unroll 1
            // 禁止展开
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumReduceChunksPerBatch) {
                // 按批次遍历环序中的其他源 rank
                const int num_batch_chunks = min(kNumReduceChunksPerBatch, kNumRanks - peer_offset);
                // 本批实际的源数
                const bool is_last_batch = peer_offset + num_batch_chunks == kNumRanks;
                // 是否为最后一批

                ptx::named_barrier_unaligned<kNumReduceThreadsPerGroup + 32>(kReduceBarrierBaseIdx + reduce_warpgroup_idx);
                // 与信用 warp 在命名屏障会合：确认本批数据已到达（由信用 warp 的等待保证）

                #pragma unroll 1
                // 禁止展开
                for (int i = 0; i < kNumElemsPerChunk / 8 / kNumThreadsPerReduceGroup; i += kNumVecsPerIter) {
                    // 逐轮遍历 chunk 内的向量（组内线程并行）
                    const auto vec_idx = i * kNumThreadsPerReduceGroup + reduce_warp_idx_in_group * 32 + lane_idx;
                    // 本线程的向量索引（组内所有 warp/lane 连续覆盖）
                    const int num_remaining_vecs = num_vecs - vec_idx;
                    // 剩余向量数（用于越界保护）
                    // The first batch loads the local contribution; later batches load the partial sum.
                    // 第一批加载本地贡献；后续批次加载已保存的部分和。
                    longlong4_t values[kNumVecsPerIter];
                    // 累加寄存器组（FP32）
                    if (full_chunk) {
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup);
                            // 读出本地数据（无越界检查）
                    } else {
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup, num_remaining_vecs, k * kNumThreadsPerReduceGroup);
                            // 带谓词读出（越界部分被屏蔽）
                    }

                    #pragma unroll 1
                    // 禁止展开
                    for (int j = 0; j < num_batch_chunks; ++ j) {
                        // 逐个累加本批各源 rank 的 BF16 数据
                        const auto src_rank_idx = (rank_idx - peer_offset - j + kNumRanks) % kNumRanks;
                        // 逆环序中的源 rank
                        const auto src_ptr = static_cast<const float4*>(
                            chunks.get_chunk_ptr(src_rank_idx, slot_idx)) + vec_idx;
                        // 源地址：该源 rank 槽位中的 BF16 chunk（float4 视图）
                        float4 packed_values[kNumVecsPerIter];
                        // 读出的 BF16 打包数据
                        if (full_chunk) {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * kNumThreadsPerReduceGroup);
                                // 读出 BF16 数据（无越界检查）
                        } else {
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * kNumThreadsPerReduceGroup, num_remaining_vecs, k * kNumThreadsPerReduceGroup);
                                // 带谓词读出 BF16 数据
                        }

                        const auto fp32x2_view = reinterpret_cast<float2*>(values);
                        // 把累加寄存器按 float2 对（FP32 数对）视图解释
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k) {
                            #pragma unroll
                            // 展开内层循环
                            for (int half_idx = 0; half_idx < 2; ++ half_idx) {
                                // 每个 BF16 打包向量分两半处理
                                const auto& packed = reinterpret_cast<const uint2*>(&packed_values[k])[half_idx];
                                // 取第 half_idx 个 BF16 对（uint2 = 4 个 BF16）
                                ptx::accumulate(fp32x2_view[k * 4 + half_idx * 2], reinterpret_cast<const nv_bfloat162&>(packed.x));
                                // 把 BF16 对 (x,y) 以 FP32 精度累加进部分和
                                ptx::accumulate(fp32x2_view[k * 4 + half_idx * 2 + 1], reinterpret_cast<const nv_bfloat162&>(packed.y));
                                // 把 BF16 对 (z,w) 以 FP32 精度累加进部分和
                            }
                        }
                    }

                    if constexpr (kWithScale) {
                        // 编译期开关：仅在需要缩放时包含此代码
                        // Scale only the final sum, never an intermediate batch.
                        // 只对最终的和乘以 scale，绝不对中间批次缩放。
                        if (is_last_batch) {
                            // 只在最后一批累加完后缩放
                            #pragma unroll
                            // 展开内层循环
                            for (int k = 0; k < kNumVecsPerIter; ++ k) {
                                #pragma unroll
                                // 展开内层循环
                                for (int half_idx = 0; half_idx < 2; ++ half_idx) {
                                    auto& value = reinterpret_cast<float4*>(&values[k])[half_idx];
                                    // 取第 half_idx 个 float4
                                    value = ptx::fmul4(value, scale);
                                    // 逐元素乘以 scale
                                }
                            }
                        }
                    }

                    if (full_chunk) {
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup, values[k]);
                            // 写回（部分）和（无越界检查）
                    } else {
                        #pragma unroll
                        // 展开内层循环
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup, values[k], num_remaining_vecs, k * kNumThreadsPerReduceGroup);
                            // 带谓词写回（越界部分被屏蔽）
                    }
                }
            }
        }
    }

    // Ensure no later writes
    // 出口栅栏：确保不再有更晚的写入（所有 put 已全局可见）
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
    // 跨 RDMA 域的出口 GPU 栅栏（带 QP flush 语义）
}

}  // namespace deep_ep::bucket
// 退出命名空间
