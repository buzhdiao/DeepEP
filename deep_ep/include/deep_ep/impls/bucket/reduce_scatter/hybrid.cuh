#pragma once
// 头文件保护：避免重复包含

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
// 引入 GPU 栅栏协议、指针运算、PTX（含 multimem/TMA/mbarrier 封装）、通用工具与 bucket 工作区布局

namespace deep_ep::bucket {
// DeepEP bucket 集合通信实现命名空间


// 模板参数：kNumRDMARanks 跨机 RDMA rank（节点）数；kNumNVLRanks 节点内 NVLink rank（卡）数；
// kNumSMs 每 rank 的 SM 数；kNumMultimemWarps 多播归约 warp 总数；kNumMultimemWarpsPerGroup 每组多播 warp 数；
// kNumIssueWarps 发送 warp 数；kNumReduceWarps 归约 warp 数；kNumSmemBytesPerReduceWarp 每归约 warp 的共享内存；
// kNumQPs QP 总数；kNumTimeoutCycles 超时周期；kWithScale 是否乘缩放因子；kContextIdx 上下文编号；其余为派生常量
template <int kNumRDMARanks, int kNumNVLRanks,
          int kNumSMs,
          int kNumMultimemWarps, int kNumMultimemWarpsPerGroup, int kNumIssueWarps, int kNumReduceWarps,
          int kNumSmemBytesPerReduceWarp,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumThreads = (kNumMultimemWarps + kNumIssueWarps + kNumReduceWarps) * 32,
          int kNumMultimemGroups = kNumMultimemWarps / kNumMultimemWarpsPerGroup,
          int kNumMultimemThreadsPerGroup = kNumMultimemWarpsPerGroup * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限制每 SM 仅 1 个 CTA 驻留
hybrid_reduce_scatter(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                      void* workspace_ptr, void* buffer, void* buffer_multimem,
                      const float scale,
                      const int num_buckets,
                      const __grid_constant__ comm::BucketList buckets) {
    // 混合（NVLink + RDMA）reduce-scatter：两级归约。
    // 第一级（节点内）：multimem warp 用 NVLS 多播读对同一节点内各卡的输入做硬件求和，得到"节点内部分和"；
    // 第二级（跨节点）：issue warp 把节点内部分和 RDMA put 发给其他节点，reduce warp 用 TMA 把收到的
    // 各节点部分和累加进本 rank 的输出分片，形成最终的全局归约结果
    // Check slots
    // 检查 chunk 槽位容量
    // Chunk indices fit in int for every supported workspace size.
    // 说明：在所有支持的工作区大小下，chunk 索引都能放进 int
    constexpr int kNumChunkBytes = 32 * 1024;
    // 编译期常量：每个 chunk 32KB
    constexpr int kNumIssueChunksPerBatch = 2;
    // 每批发送的 chunk 数（对端分组粒度，用于让发送与归约重叠）
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumChunkBytes / kNumRDMARanks;
    // 每个 RDMA rank 可用的暂存槽位数（环形流水线深度）
    EP_STATIC_ASSERT(kNumSlotsPerRank > 0, "Insufficient chunk storage for hybrid reduce-scatter");
    // 编译期断言：chunk 存储区必须至少容纳每 rank 1 个槽位
    EP_STATIC_ASSERT(kNumMultimemWarps % kNumMultimemWarpsPerGroup == 0,
                     "Multimem warps must be evenly divided across groups");
    // 编译期断言：多播 warp 必须能被均匀分组
    EP_STATIC_ASSERT(kNumIssueWarps == kNumMultimemGroups,
                     "Each multimem group requires one issue warp");
    // 编译期断言：每个多播组与一个发送 warp 一一配对
    EP_STATIC_ASSERT(kNumMultimemWarps % 4 == 0 and (kNumIssueWarps + kNumReduceWarps) % 4 == 0,
                     "Register roles must align to warp groups");
    // 编译期断言：寄存器重分配的两组 warp 都必须是 4 的倍数（warp 组对齐）
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Insufficient issue lanes for RDMA peers");
    // 编译期断言：RDMA 对端数不能超过一个 warp 的 32 个 lane
    EP_STATIC_ASSERT(kNumSmemBytesPerReduceWarp > 0 and
                     kNumSmemBytesPerReduceWarp % kNumTMAAlignmentBytes == 0,
                     "Invalid shared-memory size per reduce warp");
    // 编译期断言：每个归约 warp 的共享内存大小必须为正且按 TMA 对齐
    EP_STATIC_ASSERT(kNumRDMARanks * kNumSlotsPerRank <= layout::ReduceScatterSignals::kNumTotalSlots,
                     "Reduce-scatter slots exceed the signal workspace");
    // 编译期断言：head/tail 信号槽位总数不得超过信号工作区容量

    // Registers
    // 寄存器预算：多播归约线程需要更多寄存器，通过 warpgroup 动态重分配
    constexpr int kNumMultimemRegisters = 192;
    // 多播归约 warp 每线程的寄存器配额
    constexpr int kNumNonMultimemRegisters = 96;
    // 发送/归约 warp 每线程的寄存器配额
    EP_STATIC_ASSERT(32 * (kNumMultimemWarps * kNumMultimemRegisters +
                          (kNumIssueWarps + kNumReduceWarps) * kNumNonMultimemRegisters) <= 65536 - 2048,
                     "Too many registers");
    // 编译期断言：全部 warp 的寄存器总量不得超过每 SM 上限（预留 2048 个给系统）

    // Barriers
    // 命名屏障：用于多播组与配对发送 warp 之间的同步
    constexpr int kMultimemBarrierBaseIdx = 1;
    // 命名屏障的起始编号（0 号被其他用途占用）
    EP_STATIC_ASSERT(kMultimemBarrierBaseIdx + kNumMultimemGroups - 1 < 16, "Too many multimem groups for named barriers");
    // 编译期断言：命名屏障编号必须小于 16（硬件限制）

    // Indices
    // 计算各类索引
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 CTA（SM）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内线程编号
    const auto warp_idx = ptx::get_warp_idx();
    // 块内 warp 编号（决定角色：多播归约 / 发送 / 归约）
    const auto lane_idx = ptx::get_lane_idx();
    // warp 内 lane 编号
    const auto rdma_rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    // 本进程在铁路式 RDMA 团队中的节点编号
    const auto nvl_rank_idx = ncclTeamLsa(nccl_dev_comm).rank;
    // 本进程在机内 NVLink 团队中的卡编号
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;
    // 全局线程编号，用于跨网格分摊工作

    // Init Gin
    // 初始化 GIN（RDMA 发起）通道
    constexpr int kNumChannelsPerSM = kNumMultimemGroups + kNumReduceWarps;
    // 每个 SM 的通信通道数：多播组数 + 归约 warp 数
    constexpr int kNumUsedQPs = std::min(kNumQPs, kNumSMs * kNumChannelsPerSM);
    // 实际使用的 QP 数：不超过全部通道数
    const int channel_idx = warp_idx < kNumMultimemWarps ?
        warp_idx / kNumMultimemWarpsPerGroup : warp_idx - kNumMultimemWarps;
    // 通道编号：多播 warp 按其所属组编号，发送/归减 warp 依次连续编号
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, channel_idx);
    // 根据全局通道编号选择 QP 及其共享模式
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx, sharing_mode);
    // 构造绑定到该 QP 的 GIN 封装

    // Shared memory and mbarriers
    // 准备共享内存与 mbarrier（供归约 warp 的 TMA 使用）
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // 动态共享内存：按 TMA 对齐，每个归约 warp 独占一段作为 TMA 暂存
    __shared__ __align__(8) uint64_t mbarriers[kNumReduceWarps];
    // 静态共享内存：每个归约 warp 一个 mbarrier
    if (thread_idx < kNumReduceWarps)
        ptx::mbarrier_init_with_fence(reinterpret_cast<ptx::mbarrier*>(mbarriers + thread_idx), 1);
        // 前 kNumReduceWarps 个线程各自初始化一个 mbarrier（期望到达次数 1）
    __syncthreads();
    // 块内同步：确保 mbarrier 初始化对全块可见

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    // 取得 bucket 工作区
    auto& bucket_signals = workspace.signals[kContextIdx];
    // 本上下文的信号区
    auto& barrier_signals = bucket_signals.barrier;
    // 栅栏信号数组
    auto& signals = bucket_signals.reduce_scatter;
    // reduce-scatter 专用的 head/tail 信用环（收发信用计数器）

    // Every launch starts a fresh receive-credit ring.
    // 每次内核启动都要把接收信用环（head/tail）清零
    for (int i = global_thread_idx; i < kNumRDMARanks * kNumSlotsPerRank; i += kNumSMs * kNumThreads) {
        // 全网格线程跨步覆盖所有 (对端 rank, 槽位) 组合
        const int peer_rank_idx = i / kNumSlotsPerRank;
        // 该组合对应的对端 rank
        const int slot_idx = i % kNumSlotsPerRank;
        // 该组合对应的槽位
        *signals.get_head_ptr<kNumSlotsPerRank>(peer_rank_idx, slot_idx) = 0;
        // 清零该槽位的 head 计数器（消费进度）
        *signals.get_tail_ptr<kNumSlotsPerRank>(peer_rank_idx, slot_idx) = 0;
        // 清零该槽位的 tail 计数器（到达进度）
    }
    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag,
                      false, true, true>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx,
        sm_idx, thread_idx);
    // 跨 RDMA 域 + NVLink 域的入口 GPU 栅栏：让清零后的计数器全局可见，并确保输入数据就绪

    // Different warp roles
    // 以下按 warp 编号划分三种角色
    if (warp_idx < kNumMultimemWarps) {
        // ============ 角色 1：多播归约 warp（节点内 NVLink 归约） ============
        ptx::warpgroup_reg_realloc<kNumMultimemRegisters, kNumThreads>();
        // warpgroup 级寄存器重分配：为多播归约线程腾出更大的寄存器配额

        // Multimem reduce
        // 多播硬件归约
        constexpr int kNumGlobalMultimemGroups = kNumSMs * kNumMultimemGroups;
        // 全局多播组总数，用于跨组分摊 chunk
        const int multimem_group_idx = warp_idx / kNumMultimemWarpsPerGroup;
        // 本 warp 在组内所属的多播组编号
        const int multimem_thread_idx = warp_idx % kNumMultimemWarpsPerGroup * 32 + lane_idx;
        // 组内线程编号（组内所有 warp 的 lane 连续编号），用于分摊元素

        // NOTES: must visit the same chunk sequence as the paired issue warp
        // 说明：必须与配对的发送 warp 以完全相同的顺序访问 chunk，保证同步正确
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalMultimemGroups>(num_buckets, buckets, sm_idx * kNumMultimemGroups + multimem_group_idx);
        // chunk 迭代器：以全局多播组数为步长遍历本组负责的 chunk
        while (iterator.next()) {
            // 迭代下一个 chunk
            // NVLink domain reduction
            // 节点内（NVLink 域）归约
            #pragma unroll 1
            // 禁止展开（循环次数与 RDMA rank 数相关）
            for (int peer_begin = 0; peer_begin < kNumRDMARanks; peer_begin += kNumIssueChunksPerBatch) {
                // 按批次遍历所有 RDMA 对端（每批 kNumIssueChunksPerBatch 个）
                const int num_batch_peers = min(kNumIssueChunksPerBatch, kNumRDMARanks - peer_begin);
                // 本批实际的对端数（末批可能不满）
                #pragma unroll 1
                // 禁止展开
                for (int i = 0; i < num_batch_peers; ++ i) {
                    // 逐个处理本批对端
                    const auto dst_rdma_rank_idx = (rdma_rank_idx + peer_begin + i) % kNumRDMARanks;
                    // 环序中的目标 RDMA rank（该对端节点拥有此 shard 的那部分）
                    const auto dst_rank_idx = dst_rdma_rank_idx * kNumNVLRanks + nvl_rank_idx;
                    // 折算成全局 rank 编号（节点编号乘每节点卡数 + 本卡编号）
                    const auto src_offset = iterator.bucket.offset + dst_rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket;
                    // 该 (目标 rank, 本卡) 对应数据在缓冲中的偏移
                    const auto src_multimem_ptr = math::advance_ptr<const float4>(buffer_multimem, src_offset);
                    // 多播窗口视图：multimem 读会聚合节点内所有卡的输入
                    const auto src_ptr = math::advance_ptr<float4>(buffer, src_offset);
                    // 普通缓冲视图：归约结果写回这里，供发送 warp 发出
                    utils::iterate_ld_st<kNumMultimemWarpsPerGroup, 32, 16>(
                        multimem_thread_idx,
                        iterator.num_chunk_bytes / static_cast<int>(sizeof(float4)),
                        // 以 float4 为单位的元素数
                        src_multimem_ptr, src_ptr,
                        [&](const float4* ptr, const int& lhs = 1, const int& rhs = 0) {
                            // 加载回调：从多播地址做硬件求和归约
                            const auto v = ptx::multimem_ld_reduce_add_f32_with_gt_pred(ptr, lhs, rhs);
                            // multimem.ld_reduce.add 聚合节点内所有 rank 同位置数据
                            return kWithScale ? ptx::fmul4(v, scale) : v;
                            // 需要时对部分和乘以 scale
                        },
                        [](float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) { ptx::st_with_gt_pred(ptr, value, lhs, rhs); });
                        // 存储回调：普通 store 把节点内部分和写入本地缓冲
                }

                // Notify issue warps
                // 通知配对的发送 warp：本批节点的节点内部分和已就绪
                ptx::named_barrier_unaligned<kNumMultimemThreadsPerGroup + 32>(kMultimemBarrierBaseIdx + multimem_group_idx);
                // 命名屏障：多播组全部线程 + 配对发送 warp（32 线程）一起到达
            }
        }
    } else {
        ptx::warpgroup_reg_realloc<kNumNonMultimemRegisters, kNumThreads>();
        // 非 多播 warp 的寄存器重分配（较小的配额）

        if (warp_idx < kNumMultimemWarps + kNumIssueWarps) {
            // ============ 角色 2：发送 warp（把节点内部分和 RDMA put 到其他节点） ============
            // RDMA issue
            // RDMA 发送
            constexpr int kNumGlobalMultimemGroups = kNumSMs * kNumMultimemGroups;
            // 全局多播组总数（与配对的多播组使用相同的迭代参数）
            const auto multimem_group_idx = warp_idx - kNumMultimemWarps;
            // 本发送 warp 配对的多播组编号
            const auto chunks = layout::ChunkView2D<kNumRDMARanks, kNumSlotsPerRank, kNumChunkBytes>{workspace.chunk_storage};
            // chunk 存储二维视图：按 (rank, slot) 寻址

            // NOTES: must visit the same chunk sequence as the paired multimem group
            // 说明：必须与配对的多播组以完全相同的顺序访问 chunk
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalMultimemGroups>(num_buckets, buckets, sm_idx * kNumMultimemGroups + multimem_group_idx);
            // chunk 迭代器：与配对多播组相同
            while (iterator.next()) {
                // 迭代下一个 chunk
                const int slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
                // 本 chunk 使用的暂存槽位（环形轮转）
                const int head_target = iterator.chunk_idx / kNumSlotsPerRank;
                // head 目标值：该槽位此前被对端消费完的轮次
                #pragma unroll 1
                // 禁止展开
                for (int peer_begin = 0; peer_begin < kNumRDMARanks; peer_begin += kNumIssueChunksPerBatch) {
                    // 按批次遍历所有 RDMA 对端，与多播组同步
                    const int num_batch_peers = min(kNumIssueChunksPerBatch, kNumRDMARanks - peer_begin);
                    // 本批实际对端数
                    const int dst_rdma_rank_idx = (rdma_rank_idx + peer_begin + lane_idx) % kNumRDMARanks;
                    // 每个 lane 对应一个目标 RDMA rank（环序）
                    const int dst_rank_idx = dst_rdma_rank_idx * kNumNVLRanks + nvl_rank_idx;
                    // 折算成全局 rank 编号
                    const auto src_ptr = math::advance_ptr<float4>(
                        buffer, iterator.bucket.offset + dst_rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);
                    // 源地址：多播 warp 刚写好的节点内部分和
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(dst_rdma_rank_idx, slot_idx);
                    // 对端消费进度 head（由对端 reduce warp 自增）
                    const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(rdma_rank_idx, slot_idx);
                    // 本端到达进度 tail（数据落到对端后由 put 信号自增）

                    // Wait until the previous generation has been consumed
                    // 等待该槽位的上一代数据已被对端消费完毕，才能复用
                    // NOTES: this only depends on the remote consumers, so it overlaps with the local reduction
                    // 说明：该等待只依赖远端消费者，因此可与本地归约重叠进行
                    comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_peers, [=](const bool& is_last_check) {
                        // 带超时的自旋等待；第二个参数表示本 lane 是否参与等待
                        const auto head = ptx::ld_volatile<int64_t>(head_ptr);
                        // volatile 读 head，绕过缓存观察远端更新
                        if (head >= head_target)
                            return true;
                        // 槽位已释放
                        if (is_last_check) {
                            printf("DeepEP hybrid reduce-scatter send timeout: RDMA rank %d, NVL rank %d, "
                                   "peer %d, chunk %d, head target %d, head %lld\n",
                                   rdma_rank_idx, nvl_rank_idx, dst_rdma_rank_idx,
                                   iterator.chunk_idx, head_target, static_cast<long long>(head));
                            // 超时打印发送诊断信息
                        }
                        return false;
                        // 继续等待
                    });

                    // Wait reduction finish
                    // 等待配对多播组完成本批对端的节点内归约
                    ptx::named_barrier_unaligned<kNumMultimemThreadsPerGroup + 32>(kMultimemBarrierBaseIdx + multimem_group_idx);
                    // 与多播组在同一条命名屏障上会合

                    // Issue RDMA
                    // 发起 RDMA 发送
                    if (lane_idx < num_batch_peers and dst_rdma_rank_idx == rdma_rank_idx) {
                        // 目标是本节点自己
                        // The local chunk is already in place
                        // 本地分片数据已在原地，无需网络传输
                        ptx::red_add_rel_gpu(tail_ptr, 1);
                        // 直接以 release 语义把本端 tail 自增 1（发布"数据已可读"）
                    } else if (lane_idx < num_batch_peers) {
                        // 目标是真正的远端节点
                        // Push the reduced chunk and publish its arrival
                        // 推送归约后的 chunk 并发布其到达信号
                        gin.put<ncclTeamTagRail>(
                            chunks.get_chunk_ptr(rdma_rank_idx, slot_idx),
                            // 目的：对端 chunk 存储区中本 rank 的该槽位
                            src_ptr, iterator.num_chunk_bytes, dst_rdma_rank_idx,
                            ncclGinOptFlagsDefault,
                            // 默认选项标志
                            ncclGin_StrongVASignalAdd {
                                .signalWindow = window,
                                // 信号落在对称窗口上
                                .signalOffset = gin.get_sym_offset(tail_ptr),
                                // 信号偏移指向本端 tail 计数器（对端视角同址）
                                .value = 1,
                                // 数据送达后 tail 自增 1
                            });
                    }
                    __syncwarp();
                    // 同步 warp，保证本批发送完成后再进入下一批
                }
            }
        } else {
            // ============ 角色 3：归约 warp（把远端发来的部分和累加进本地输出） ============
            // Reduce warps: accumulate remote chunks into the local output
            // 归约 warp：把远端节点发来的 chunk 累加进本地输出分片
            // Only 1 lane is involved
            // 仅需 warp 内 1 个 lane 参与控制流
            if (ptx::elect_one_sync()) {
                // 选出单个线程执行
                constexpr int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps;
                // 全局归约 warp 总数

                const int rank_idx = rdma_rank_idx * kNumNVLRanks + nvl_rank_idx;
                // 本进程的全局 rank 编号
                const int reduce_warp_idx = warp_idx - kNumMultimemWarps - kNumIssueWarps;
                // 块内归约 warp 编号（0 起）
                const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(mbarriers + reduce_warp_idx);
                // 本归约 warp 的 mbarrier
                const auto smem_ptr = smem + reduce_warp_idx * kNumSmemBytesPerReduceWarp;
                // 本归约 warp 独占的共享内存段
                const auto chunks = layout::ChunkView2D<kNumRDMARanks, kNumSlotsPerRank, kNumChunkBytes>{workspace.chunk_storage};
                // chunk 存储二维视图
                ptx::arrival_phase phase = 0;
                // mbarrier 相位计数，随每轮等待翻转

                auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalReduceWarps>(num_buckets, buckets, sm_idx * kNumReduceWarps + reduce_warp_idx);
                // chunk 迭代器：以全局归约 warp 数为步长遍历本 warp 负责的 chunk
                while (iterator.next()) {
                    // 迭代下一个待归约 chunk
                    const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
                    // 本 chunk 的暂存槽位
                    const auto tail_target = iterator.chunk_idx / kNumSlotsPerRank + 1;
                    // tail 目标值：该槽位本轮已被写入的总轮次
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(rdma_rank_idx, slot_idx);
                    // 本端该槽位的 head 计数器（消费完成后自增，释放槽位）
                    const auto dst_ptr = math::advance_ptr<float>(
                        buffer, iterator.bucket.offset + rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);
                    // 输出地址：本 rank 在该桶中的分片位置

                    #pragma unroll 1
                    // 禁止展开
                    for (int i = 0; i < kNumRDMARanks; ++ i) {
                        // 按逆环序逐个处理所有 RDMA 源 rank（i=0 为本地）
                        const int src_rdma_rank_idx = (rdma_rank_idx - i + kNumRDMARanks) % kNumRDMARanks;
                        // 环序中的源 RDMA rank
                        const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(src_rdma_rank_idx, slot_idx);
                        // 源 rank 该槽位的 tail 计数器（其发送方会自增）

                        // Wait arrival
                        // 等待该源 rank 的 chunk 到达
                        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                            // 带超时的自旋等待
                            const auto tail = ptx::ld_acquire_sys(tail_ptr);
                            // acquire（系统级）读 tail，保证看到对端写入的数据
                            if (tail >= tail_target)
                                return true;
                            // 数据已到达
                            if (is_last_check) {
                                printf("DeepEP hybrid reduce-scatter receive timeout: RDMA rank %d, NVL rank %d, "
                                       "peer %d, chunk %d, tail target %d, tail %lld\n",
                                       rdma_rank_idx, nvl_rank_idx, src_rdma_rank_idx,
                                       iterator.chunk_idx, tail_target, static_cast<long long>(tail));
                                // 超时打印接收诊断信息
                            }
                            return false;
                            // 继续等待
                        });

                        // No local reduction
                        // 源是本地（i == 0）时无需再做归约
                        if (i == 0) {
                            // Release the local generation
                            // 释放本地该代槽位（数据已在输出缓冲中就位）
                            ptx::red_add_rel_gpu(head_ptr, 1);
                            // 以 release 语义把 head 自增 1，允许发送方复用该槽位
                            continue;
                            // 跳过本地数据，继续处理下一个源
                        }

                        // Reduce the remote chunk through shared memory into the output
                        // 通过共享内存把远端 chunk 累加进输出
                        // NOTES: TMA atomic add can ensure deterministic if no other SMs involved
                        // 说明：只要没有其他 SM 同时累加同一地址，TMA 原子加即可保证确定性
                        const auto recv_buffer_ptr = chunks.get_chunk_ptr(src_rdma_rank_idx, slot_idx);
                        // 取得该槽位中源 rank chunk 的地址
                        #pragma unroll 1
                        // 禁止展开：chunk 大于共享内存段时需分多轮
                        for (int offset = 0; offset < iterator.num_chunk_bytes; offset += kNumSmemBytesPerReduceWarp) {
                            // 按共享内存段大小分块搬运并累加
                            const int num_bytes = min(kNumSmemBytesPerReduceWarp, iterator.num_chunk_bytes - offset);
                            // 本轮搬运的字节数
                            ptx::tma_store_wait();
                            // 等待上一轮 TMA store 完成，避免覆盖共享内存
                            ptx::tma_load_1d(
                                smem_ptr,
                                math::advance_ptr<const void>(recv_buffer_ptr, offset),
                                mbarrier_ptr, num_bytes);
                            // 发起 1D TMA load：把远端 chunk 的一块读入共享内存
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, num_bytes);
                            // 设置 mbarrier 事务字节数并到达一次
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // 等待 mbarrier 相位翻转（本块 load 完成）
                            ptx::tma_store_reduce_add_f32(math::advance_ptr<float>(dst_ptr, offset), smem_ptr, num_bytes);
                            // 发起 TMA store reduce-add：把该块以 f32 加法累加进输出
                            ptx::tma_store_commit();
                            // 提交 TMA store 异步操作
                        }

                        // Release the remote ring slot only if a later chunk will reuse it.
                        // 仅当后续 chunk 还会复用该环形槽位时，才释放远端槽位
                        if (iterator.chunk_idx < iterator.num_total_chunks - kNumSlotsPerRank) {
                            gin.signal<ncclTeamTagRail>(
                                src_rdma_rank_idx,
                                ncclGin_StrongVASignalAdd {
                                    .signalWindow = window,
                                    // 信号落在对称窗口上
                                    .signalOffset = gin.get_sym_offset(head_ptr),
                                    // 通知源 rank：本端 head 自增 1，槽位可复用
                                    .value = 1,
                                    // 自增量 1
                                });
                        }
                    }
                }
            }
        }
    }

    // Barrier to ensure data arrival
    // 出口栅栏：确保所有数据传输都已完成并在对端可见
    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumUsedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag,
                      true, true, false>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx,
        sm_idx, thread_idx);
    // 跨 RDMA + NVLink 域的 GPU 栅栏（带 QP flush 语义），保证所有 put 全部落地
}

}  // namespace deep_ep::bucket
// 退出命名空间
