#pragma once
// 头文件保护：避免重复包含

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
// 引入 GPU 栅栏协议、指针运算、PTX（含 TMA/mbarrier 封装）、通用工具与 bucket 工作区布局

namespace deep_ep::bucket {
// DeepEP bucket 集合通信实现命名空间

// 模板参数：kNumRanks 跨机 RDMA rank 数；kNumSMs 每 rank 的 SM 数；
// kNumIssueWarps 每个块的发送（issue）warp 数；kNumReduceWarps 每个块的归约（reduce）warp 数；
// kNumSmemBytesPerReduceWarp 每个归约 warp 的共享内存字节数；kNumQPs QP 总数；
// kNumTimeoutCycles 超时周期；kWithScale 是否乘缩放因子；kContextIdx 上下文编号；其余为派生常量
template <int kNumRanks, int kNumSMs,
          int kNumIssueWarps, int kNumReduceWarps,
          int kNumSmemBytesPerReduceWarp,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumWarps = kNumIssueWarps + kNumReduceWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumGlobalIssueWarps = kNumSMs * kNumIssueWarps,
          int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps,
          int kNumGlobalThreads = kNumSMs * kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限制每 SM 仅 1 个 CTA 驻留
rdma_all_reduce(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                void* workspace_ptr, void* buffer,
                const float scale,
                const int num_buckets,
                const __grid_constant__ comm::BucketList buckets) {
    // 跨机（纯 RDMA 路径）all-reduce：
    // 1) 发送 warp 通过 GIN RDMA put 把本地分片按环序发到各对端的暂存 chunk 槽位，并用 tail 计数通知；
    // 2) 归约 warp 用 TMA 把各对端 chunk 读入共享内存逐个累加（f32 reduce-add），得到本 rank 的归约结果；
    // 3) 再把归约后的 chunk RDMA put 广播给所有对端，实现 all-reduce
    // Check slots
    // 检查 chunk 槽位容量
    // Chunk indices fit in int for every supported workspace size.
    // 说明：在所有支持的工作区大小下，chunk 索引都能放进 int
    EP_STATIC_ASSERT(kNumRanks > 1, "RDMA all-reduce requires multiple RDMA ranks");
    // 编译期断言：RDMA all-reduce 至少需要 2 个 rank
    constexpr int kNumChunkBytes = (kNumRanks <= 64 ? 64 : 32) * 1024;
    // 编译期常量：每个 chunk 的大小（rank 数不超过 64 时为 64KB，否则 32KB）
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumChunkBytes / kNumRanks;
    // 每个 rank 在 chunk 存储区中可用的槽位数量（流水线深度）
    EP_STATIC_ASSERT(kNumSlotsPerRank > 0, "Insufficient chunk storage for RDMA all-reduce");
    // 编译期断言：chunk 存储区必须至少容纳每 rank 1 个槽位
    EP_STATIC_ASSERT(kNumSmemBytesPerReduceWarp > 0 and
                     kNumSmemBytesPerReduceWarp % kNumTMAAlignmentBytes == 0,
                     "Invalid shared-memory size per reduce warp");
    // 编译期断言：每个归约 warp 的共享内存大小必须为正且按 TMA 对齐
    EP_STATIC_ASSERT(2 * kNumRanks * kNumSlotsPerRank <= layout::AllReduceSignals::kNumTotalSlots,
                     "All-reduce slots exceed the signal workspace");
    // 编译期断言：双向（head+tail）信号槽位总数不得超过信号工作区容量

    // Issue and reduce warps must form complete warp groups
    // 发送与归约 warp 必须构成完整的 warp 组（4 的倍数）
    EP_STATIC_ASSERT(kNumIssueWarps % 4 == 0 and kNumReduceWarps % 4 == 0,
                     "Issue and reduce warps must align to warp groups");
    // 编译期断言：两类 warp 数量都必须是 4 的倍数

    // Indices
    // 计算各类索引
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 CTA（SM）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内线程编号
    const auto warp_idx = ptx::get_warp_idx();
    // 块内 warp 编号（前 kNumIssueWarps 为发送 warp，其余为归约 warp）
    const auto lane_idx = ptx::get_lane_idx();
    // warp 内 lane 编号（0~31）
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    // 本进程在铁路式（rail）RDMA 团队中的 rank 编号
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;
    // 全局线程编号，用于跨网格分摊工作

    // Init Gin
    // 初始化 GIN（RDMA 发起）通道
    constexpr int kNumChannelsPerSM = kNumIssueWarps + kNumReduceWarps;
    // 每个 SM 的通信通道数等于 warp 总数（每 warp 一个通道）
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, warp_idx);
    // 根据全局 warp 编号选择 QP 及其共享模式（独占或跨 CTA 共享）
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx, sharing_mode);
    // 构造绑定到该 QP 的 GIN 封装

    // Construct workspace layouts
    // 构造工作区布局引用
    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    // 取得 bucket 工作区
    auto& bucket_signals = workspace.signals[kContextIdx];
    // 本上下文的信号区
    auto& barrier_signals = bucket_signals.barrier;
    // 栅栏信号数组
    auto& signals = bucket_signals.all_reduce;
    // all-reduce 专用的 head/tail 计数器
    const auto rs_head = signals.counters;
    // head 指针：每对 (rank, slot) 一个，表示该槽位已被消费（可复用）
    const auto rs_tail = rs_head + kNumRanks * kNumSlotsPerRank;
    // tail 指针：表示该槽位已收到数据（可读取），紧接在 head 区之后

    // Shared memory and mbarriers for TMA reduction
    // 为 TMA 归约准备共享内存与 mbarrier
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // 动态共享内存：按 TMA 对齐，每个归约 warp 独占一段作为 TMA 暂存
    __shared__ __align__(8) uint64_t mbarriers[kNumReduceWarps];
    // 静态共享内存：每个归约 warp 一个 mbarrier（64 位硬件同步对象）

    // Every launch starts with fresh signal counters
    // 每次内核启动都要把信号计数器清零
    EP_STATIC_ASSERT(sizeof(layout::AllReduceSignals) % sizeof(int4) == 0 and
                     alignof(layout::AllReduceSignals) % alignof(int4) == 0,
                     "Signals must be a multiple of int4");
    // 编译期断言：信号区大小与对齐必须是 int4（16 字节）的倍数，便于向量化清零
    constexpr auto kNumSignalVecs = static_cast<int>(sizeof(layout::AllReduceSignals) / sizeof(int4));
    // 信号区占用的 int4 向量数
    const auto signal_vecs = reinterpret_cast<int4*>(&signals);
    // 把信号区按 int4 视图解释
    #pragma unroll
    // 提示编译器展开清零循环
    for (int i = global_thread_idx; i < kNumSignalVecs; i += kNumGlobalThreads)
        signal_vecs[i] = make_int4(0, 0, 0, 0);
    // 全网格线程跨步把所有信号计数器清零
    if (thread_idx < kNumReduceWarps)
        ptx::mbarrier_init_with_fence(reinterpret_cast<ptx::mbarrier*>(mbarriers + thread_idx), 1);
        // 前 kNumReduceWarps 个线程各自初始化一个 mbarrier，期望到达次数为 1（TMA 完成通知）

    // Barrier to make the cleared signals visible and ensure data arrival
    // 栅栏：让清零后的信号全局可见，并确保所有 rank 的输入数据均已就绪
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
    // 跨 RDMA 域的 GPU 栅栏（flush 所有已分配的 QP），等待全体到达

    const auto chunks = layout::ChunkView2D<kNumRanks, kNumSlotsPerRank, kNumChunkBytes>{workspace.chunk_storage};
    // 构造 chunk 存储的二维视图：按 (rank, slot) 寻址，每槽一个 chunk

    if (warp_idx < kNumIssueWarps) {
        // ============ 发送 warp：负责把本地 chunk 通过 RDMA put 发往各对端 ============
        // Each issue warp sends its chunks through a single elected lane and GIN channel
        // 每个发送 warp 由选出的单个 lane 通过一个 GIN 通道发送其负责的 chunk
        const int global_issue_warp_idx = sm_idx * kNumIssueWarps + warp_idx;
        // 全局发送 warp 编号，用于跨 warp 分摊 chunk
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalIssueWarps>(
            num_buckets, buckets, global_issue_warp_idx);
        // chunk 迭代器：以全局发送 warp 数为步长，遍历所有桶中属于本 warp 的 chunk
        if (ptx::elect_one_sync()) {
            // warp 内选出单个线程执行发送，避免重复
            while (iterator.next()) {
                // 迭代下一个 chunk
                const int dst_rank_idx = iterator.chunk_idx % kNumRanks;
                // 目的 rank 按环序编号：chunk_idx 对 kNumRanks 取模
                if (dst_rank_idx == rank_idx)
                    continue;
                // 本 rank 的数据无需发送给自己
                const int chunk_idx_in_shard = iterator.chunk_idx / kNumRanks;
                // 该数据块在本 rank 分片内的全局 chunk 序号
                const int slot_idx = chunk_idx_in_shard % kNumSlotsPerRank;
                // 按 (chunk 序号 mod 槽位数) 轮转使用暂存槽位（环形流水线）
                const int64_t rs_head_target = chunk_idx_in_shard / kNumSlotsPerRank;
                // 该槽位需要被对端消费完的轮次编号（head 目标值）
                const auto rs_head_ptr = rs_head + dst_rank_idx * kNumSlotsPerRank + slot_idx;
                // 指向对端视角下该 (rank, slot) 的 head 计数器（对称内存中即本端镜像地址）
                const auto rs_tail_ptr = rs_tail + rank_idx * kNumSlotsPerRank + slot_idx;
                // 指向本 rank 该槽位的 tail 计数器（对端收到数据后自增）

                // Wait release
                // 等待槽位释放：对端 head 达到目标轮次后才可复用该槽位
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    // 带超时的自旋等待；is_last_check 表示最后一次检查（用于打印超时信息）
                    // NOTES: no need to acquire
                    // 说明：此处无需 acquire 语义，只需观察计数变化
                    const auto head_val = ptx::ld_volatile<int64_t>(rs_head_ptr);
                    // 以 volatile 读获取 head 当前值，绕过缓存
                    if (head_val >= rs_head_target)
                        return true;
                    // 槽位已释放，可以写入
                    if (is_last_check) {
                        printf("DeepEP RDMA all-reduce send timeout: rank %d, peer %d, "
                               "chunk %d, head target %lld, head %lld\n",
                               rank_idx, dst_rank_idx, iterator.chunk_idx,
                               static_cast<long long>(rs_head_target), static_cast<long long>(head_val));
                        // 最后一次检查仍失败则打印发送超时诊断信息
                    }
                    return false;
                    // 继续等待
                });

                const auto src_ptr = math::advance_ptr(
                    buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
                // 计算该 chunk 在本地输入缓冲中的源地址
                gin.put<ncclTeamTagRail>(
                    chunks.get_chunk_ptr(rank_idx, slot_idx),
                    // 目的地址：对端 chunk 存储区中本 rank 的该槽位（对称内存，偏移即本端视图）
                    src_ptr, iterator.num_chunk_bytes, dst_rank_idx, 0,
                    ncclGin_StrongVASignalAdd {
                        .signalWindow = window,
                        // 信号落在对称窗口上
                        .signalOffset = gin.get_sym_offset(rs_tail_ptr),
                        // 信号偏移指向本 rank 该槽位的 tail 计数器
                        .value = 1,
                        // 数据送达后把 tail 自增 1，通知对端可读
                    });
            }
        }
    } else {
        // ============ 归约 warp：消费各对端发来的 chunk 并做本地归约 ============
        // Reduce warps consume their owned chunks in the deterministic cyclic rank order
        // 归约 warp 按确定性的环序（rank 循环顺序）消费属于自己的 chunk
        const int reduce_warp_idx = warp_idx - kNumIssueWarps;
        // 块内归约 warp 编号（0 起）
        const int global_reduce_warp_idx = sm_idx * kNumReduceWarps + reduce_warp_idx;
        // 全局归约 warp 编号
        const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(mbarriers + reduce_warp_idx);
        // 本归约 warp 的 mbarrier
        const auto smem_ptr = smem + reduce_warp_idx * kNumSmemBytesPerReduceWarp;
        // 本归约 warp 独占的共享内存段
        ptx::arrival_phase phase = 0;
        // mbarrier 相位计数，随每轮等待翻转
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumRanks * kNumGlobalReduceWarps>(
            num_buckets, buckets, rank_idx + kNumRanks * global_reduce_warp_idx);
        // chunk 迭代器：本 rank 需要归约的 chunk（起点 rank_idx，步长为全网格归约 warp 数乘 rank 数）

        while (iterator.next()) {
            // 迭代下一个待归约 chunk
            const int chunk_idx_in_shard = iterator.chunk_idx / kNumRanks;
            // 该数据块在分片内的全局 chunk 序号
            const int slot_idx = chunk_idx_in_shard % kNumSlotsPerRank;
            // 对应的暂存槽位编号（与发送端一致）
            const int64_t rs_tail_target = chunk_idx_in_shard / kNumSlotsPerRank + 1;
            // tail 目标值：该槽位本轮已被写入的总轮次（第 1 次从 1 开始计数）
            const auto rs_head_ptr = rs_head + rank_idx * kNumSlotsPerRank + slot_idx;
            // 本 rank 该槽位的 head 计数器（消费完成后自增，通知发送端可复用）
            const auto out_ptr = math::advance_ptr<float>(
                buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
            // 归约结果写回本地输入缓冲中的对应位置（原地累加）

            // Without scaling, all peers use TMA reduction
            // 不带缩放时，所有对端的 chunk 全部用 TMA reduce-add 硬件累加
            // With scaling, leave the last peer for a warp-wide register reduction
            // 带缩放时，把最后一个对端留给 warp 级寄存器归约（以便乘 scale）
            constexpr int kNumTMAPeers = kWithScale ? kNumRanks - 2 : kNumRanks - 1;
            // 走 TMA 路径的对端数量（kWithScale 时少一个）
            #pragma unroll 1
            // 禁止展开，避免代码膨胀（循环次数与 rank 数相关）
            for (int i = 1; i <= kNumTMAPeers; ++ i) {
                // 按环序依次处理第 i 个对端
                const int src_rank_idx = (rank_idx + i) % kNumRanks;
                // 环序中的源 rank
                const auto rs_tail_ptr = rs_tail + src_rank_idx * kNumSlotsPerRank + slot_idx;
                // 源 rank 该槽位的 tail 计数器（其所有者会往这里发数据）

                if (ptx::elect_one_sync()) {
                    // 选出单个线程执行等待与 TMA 操作
                    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                        // 带超时的自旋等待对端数据到达
                        const auto tail_val = ptx::ld_acquire_sys(rs_tail_ptr);
                        // 以 acquire（系统级）语义读 tail，保证看到对端写入的数据
                        if (tail_val >= rs_tail_target)
                            return true;
                        // 数据已到达
                        if (is_last_check) {
                            printf("DeepEP RDMA all-reduce receive timeout: rank %d, peer %d, "
                                   "chunk %d, tail target %lld, tail %lld\n",
                                   rank_idx, src_rank_idx, iterator.chunk_idx,
                                   static_cast<long long>(rs_tail_target), static_cast<long long>(tail_val));
                            // 最后一次检查仍失败则打印接收超时诊断信息
                        }
                        return false;
                        // 继续等待
                    });

                    const auto recv_ptr = chunks.get_chunk_ptr(src_rank_idx, slot_idx);
                    // 取得该槽位中源 rank chunk 的地址
                    #pragma unroll 1
                    // 禁止展开：chunk 可能远大于共享内存段，需分多次搬运
                    for (int offset = 0; offset < iterator.num_chunk_bytes; offset += kNumSmemBytesPerReduceWarp) {
                        // 按共享内存段大小把 chunk 分块搬运并累加
                        const int num_bytes = min(kNumSmemBytesPerReduceWarp, iterator.num_chunk_bytes - offset);
                        // 本轮搬运的字节数（最后一块可能不足）
                        ptx::tma_store_wait();
                        // 等待上一轮 TMA store（reduce-add 写出）完成，避免覆盖共享内存
                        ptx::tma_load_1d(
                            smem_ptr, math::advance_ptr<const void>(recv_ptr, offset),
                            mbarrier_ptr, num_bytes);
                        // 发起 1D TMA load：把对端 chunk 的一块读入共享内存，完成后由 mbarrier 通知
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, num_bytes);
                        // 设置 mbarrier 的事务字节数并到达一次
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                        // 自旋等待 mbarrier 相位翻转，即本块 TMA load 完成
                        ptx::tma_store_reduce_add_f32(
                            math::advance_ptr<float>(out_ptr, offset), smem_ptr, num_bytes);
                        // 发起 TMA store reduce-add：把共享内存中的块以 f32 加法累加进输出（全局内存）
                        ptx::tma_store_commit();
                        // 提交 TMA store 异步操作
                    }
                    ptx::tma_store_wait();
                    // 等待所有 TMA store 彻底完成
                    gin.signal<ncclTeamTagRail>(
                        src_rank_idx,
                        ncclGin_StrongVASignalAdd {
                            .signalWindow = window,
                            // 信号落在对称窗口上
                            .signalOffset = gin.get_sym_offset(rs_head_ptr),
                            // 通知源 rank：本端 head 自增 1，槽位已消费可复用
                            .value = 1,
                            // 自增量 1
                        });
                }
                __syncwarp();
                // 同步 warp：让 elected 线程的进度对全 warp 可见
            }

            if constexpr (kWithScale) {
                // 带缩放时：最后一个对端用 warp 级寄存器归约处理
                // The final rank is reduced and scaled in registers
                // 最后一个 rank 的数据在寄存器中完成归约并乘以 scale
                constexpr int kNumVecBytes = sizeof(float4);
                // 向量宽度：16 字节（4 个 float）
                const int src_rank_idx = (rank_idx + kNumRanks - 1) % kNumRanks;
                // 环序中最后一个对端（rank 前一个）
                const auto rs_tail_ptr = rs_tail + src_rank_idx * kNumSlotsPerRank + slot_idx;
                // 其 tail 计数器
                if (ptx::elect_one_sync()) {
                    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                        const auto tail_val = ptx::ld_acquire_sys(rs_tail_ptr);
                        // acquire 读等待其数据到达
                        if (tail_val >= rs_tail_target)
                            return true;
                        // 数据已到达
                        if (is_last_check) {
                            printf("DeepEP RDMA all-reduce receive timeout: rank %d, peer %d, "
                                   "chunk %d, tail target %lld, tail %lld\n",
                                   rank_idx, src_rank_idx, iterator.chunk_idx,
                                   static_cast<long long>(rs_tail_target), static_cast<long long>(tail_val));
                            // 超时诊断信息
                        }
                        return false;
                        // 继续等待
                    });
                }
                __syncwarp();
                // 让等待结果对全 warp 可见

                const auto recv_ptr = static_cast<const float4*>(chunks.get_chunk_ptr(src_rank_idx, slot_idx));
                // 该对端 chunk 的向量视图
                const auto out_vec_ptr = reinterpret_cast<float4*>(out_ptr);
                // 输出（已含前 kNumTMAPeers 个对端的部分和）的向量视图
                const int num_vecs = iterator.num_chunk_bytes / kNumVecBytes;
                // 实际需要处理的向量数
                constexpr int kNumMaxVecs = kNumChunkBytes / kNumVecBytes;
                // 一个 chunk 最多包含的向量数
                #pragma unroll
                // 展开循环：每轮 warp 处理 32 个向量
                for (int i = 0; i < math::constexpr_ceil_div(kNumMaxVecs, 32); ++ i) {
                    const int vec_idx = i * 32 + lane_idx;
                    // lane 负责的向量索引
                    const auto x = ptx::ld_with_gt_pred(out_vec_ptr + vec_idx, num_vecs, vec_idx);
                    // 读出输出中的部分和（越界谓词保护）
                    const auto y = ptx::ld_with_gt_pred(recv_ptr + vec_idx, num_vecs, vec_idx);
                    // 读出最后一个对端的 chunk（越界谓词保护）
                    const auto value = ptx::fmul4(ptx::fadd4(x, y), scale);
                    // 逐元素相加完成归约，再乘以 scale
                    ptx::st_with_gt_pred(out_vec_ptr + vec_idx, value, num_vecs, vec_idx);
                    // 写回归约结果（越界谓词保护）
                }
                __syncwarp();
                // 同步 warp 确保写出完成
                if (ptx::elect_one_sync()) {
                    gin.signal<ncclTeamTagRail>(
                        src_rank_idx,
                        ncclGin_StrongVASignalAdd {
                            .signalWindow = window,
                            // 信号落在对称窗口上
                            .signalOffset = gin.get_sym_offset(rs_head_ptr),
                            // 通知该对端槽位已消费
                            .value = 1,
                            // 自增量 1
                        });
                }
            }
            __syncwarp();
            // 同步 warp，确保本 chunk 处理完毕

            // Broadcast the reduced chunk to every peer
            // 把归约完成的 chunk 广播给所有其他对端（完成 all-reduce 的分发阶段）
            for (int i = lane_idx; i < kNumRanks - 1; i += 32) {
                // 32 个 lane 并行发往不同目的 rank
                const int dst_rank_idx = (rank_idx + i + 1) % kNumRanks;
                // 环序中除自身外的所有目的 rank
                gin.put<ncclTeamTagRail>(out_ptr, out_ptr, iterator.num_chunk_bytes, dst_rank_idx);
                // RDMA put：把结果 chunk 写到对端输出缓冲的相同偏移处（源/目的偏移一致）
            }
            __syncwarp();
            // 同步 warp，确保广播全部发出
        }
    }

    // Flush QPs and ensure the visiblity of preceding puts on responders
    // 冲刷所有 QP，确保此前的 put 在响应端（对端）上全局可见
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
    // 出口 GPU 栅栏（带 QP flush 语义），保证广播写全部落地后才允许内核结束
}

}  // namespace deep_ep::bucket
// 退出命名空间
