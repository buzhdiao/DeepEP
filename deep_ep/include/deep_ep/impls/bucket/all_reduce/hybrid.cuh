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

// 设备端内联辅助函数：通过 TMA 双缓冲流水线把一块数据经 multimem 多播写回节点内所有卡（all-gather 阶段）
template <int kNumTileBytes>
__device__ __forceinline__ void
hybrid_ar_broadcast(const int8_t* src, float4* dst, int num_bytes, int8_t* smem,
                    ptx::mbarrier* mbarrier, ptx::arrival_phase& phase) {
    // 参数：src 源数据（chunk 或缓冲）；dst 多播窗口目的地址；num_bytes 总字节数；
    // smem 双缓冲共享内存；mbarrier 同步屏障；phase mbarrier 相位（随迭代翻转）
    if (ptx::elect_one_sync()) {
        // 选出单个线程发起第一块 TMA load（流水线预热）
        const int count = min(kNumTileBytes, num_bytes);
        // 第一块的字节数（不足一个 tile 时取剩余量）
        ptx::tma_load_1d(smem, src, mbarrier, count);
        // 发起 1D TMA load：把第一块读入共享内存第 0 个缓冲
        ptx::mbarrier_arrive_and_set_tx(mbarrier, count);
        // 设置 mbarrier 事务字节数并到达一次
    }
    // Steady state of the pipeline
    // 流水线稳态：一边等当前块一边预取下一块（双缓冲交替）
    #pragma unroll 1
    // 禁止展开（迭代次数与数据量相关）
    for (int offset = 0; offset < num_bytes; offset += kNumTileBytes) {
        // 逐 tile 处理
        const int slot = (offset / kNumTileBytes) % 2;
        // 当前 tile 使用的缓冲槽（0/1 交替，构成双缓冲）
        const int count = min(kNumTileBytes, num_bytes - offset);
        // 当前 tile 的字节数
        if (ptx::elect_one_sync()) {
            // 单线程执行同步与预取
            ptx::mbarrier_wait_and_flip_phase(mbarrier, phase);
            // 等待当前 tile 的 TMA load 完成（相位翻转）
            ptx::tma_store_wait_read();
            // 等待上一次对该缓冲的异步读（multimem cp）彻底结束，避免数据竞争
            if (offset + kNumTileBytes < num_bytes) {
                // 还有后续数据则预取下一块到另一个缓冲
                const int next_count = min(kNumTileBytes, num_bytes - offset - kNumTileBytes);
                // 下一块的字节数
                ptx::tma_load_1d(smem + (1 - slot) * kNumTileBytes, src + offset + kNumTileBytes, mbarrier, next_count);
                // 预取下一 tile 到另一个缓冲槽
                ptx::mbarrier_arrive_and_set_tx(mbarrier, next_count);
                // 设置事务字节数并到达
            }
        }
        __syncwarp();
        // 同步 warp：让 elected 线程的进度对全 warp 可见
        if (ptx::elect_one_sync()) {
            // 单线程发起多播写
            ptx::multimem_cp_async_bulk(dst + offset / sizeof(float4), smem + slot * kNumTileBytes, count);
            // multimem 批量异步拷贝：把共享内存中的 tile 多播写入节点内所有卡的同一偏移
            ptx::tma_store_commit();
            // 提交异步存储操作
        }
        __syncwarp();
        // 同步 warp
    }
    if (ptx::elect_one_sync())
        ptx::tma_store_wait_read();
        // 等待最后一次多播读全部完成
    __syncwarp();
    // 同步 warp 后返回
}

// 模板参数：kNumRDMARanks 跨机 RDMA rank（节点）数；kNumNVLRanks 节点内 NVLink rank（卡）数；
// kNumSMs 每 rank 的 SM 数；kNumMultimemWarpsPerGroup 每组多播 warp 数；kNumIssueWarps 发送 warp 数；
// kNumReduceWarps 归约 warp 数；kNumBroadcastWarps 广播 warp 数；两种共享内存配额；kNumQPs QP 总数；
// kNumTimeoutCycles 超时周期；kWithScale 是否乘缩放因子；kContextIdx 上下文编号；kNumChunkBytes chunk 大小
template <int kNumRDMARanks, int kNumNVLRanks,
          int kNumSMs,
          int kNumMultimemWarpsPerGroup, int kNumIssueWarps, int kNumReduceWarps, int kNumBroadcastWarps,
          int kNumSmemBytesPerReduceWarp, int kNumSmemBytesPerBroadcastWarp,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumChunkBytes,
          int kNumMultimemWarps = kNumMultimemWarpsPerGroup * kNumIssueWarps,
          int kNumThreads = (kNumMultimemWarps + kNumIssueWarps + kNumReduceWarps + kNumBroadcastWarps) * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限制每 SM 仅 1 个 CTA 驻留
hybrid_all_reduce(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                  void* workspace_ptr, void* buffer, void* buffer_multimem,
                  const float scale,
                  const int num_buckets,
                  const __grid_constant__ comm::BucketList buckets) {
    // 混合（NVLink + RDMA）all-reduce，分四个阶段由不同角色的 warp 协作完成：
    // 1) 多播 warp：NVLS 硬件归约得到节点内部分和；
    // 2) 发送 warp：把部分和 RDMA put 到各对端节点（reduce-scatter 阶段）；
    // 3) 归约 warp：累加所有节点的部分和得到全局归约结果，并回传给各对端；
    // 4) 广播 warp：把完全归约的结果经 multimem 多播回节点内所有卡（all-gather 阶段）
    // Check slots
    // 检查槽位与拓扑约束
    constexpr int kNumGlobalIssueWarps = kNumSMs * kNumIssueWarps;
    // 全局发送 warp 总数
    constexpr int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps;
    // 全局归约 warp 总数
    constexpr int kNumSlots = layout::ChunkStorage::kNumBytes / kNumChunkBytes / kNumRDMARanks;
    // 每个 RDMA rank 可用的暂存槽位数（环形流水线深度）
    constexpr int kNumSignalSlots = 2 * kNumRDMARanks * kNumSlots + kNumGlobalReduceWarps * kNumRDMARanks;
    // 信号槽总数：RS 双向（bcst_head + rs_tail）加 AG 单向（ag_tail）
    EP_STATIC_ASSERT(kNumRDMARanks > 1 and kNumRDMARanks <= 128 and kNumNVLRanks > 1,
                     "Unsupported hybrid topology");
    // 编译期断言：混合拓扑要求至少 2 个节点（不超过 128）且节点内至少 2 卡
    EP_STATIC_ASSERT(kNumSMs >= 2 and kNumSlots > 0, "Invalid ring or SM count");
    // 编译期断言：SM 数至少为 2，槽位数必须为正
    EP_STATIC_ASSERT(kNumSmemBytesPerReduceWarp > 0 and kNumSmemBytesPerReduceWarp % kNumTMAAlignmentBytes == 0,
                     "Invalid shared memory size per reduce warp");
    // 编译期断言：归约 warp 的共享内存必须为正且按 TMA 对齐
    EP_STATIC_ASSERT(kNumSignalSlots <= layout::AllReduceSignals::kNumTotalSlots,
                     "Hybrid all-reduce signals exceed the workspace");
    // 编译期断言：信号槽总数不得超过信号工作区容量

    // Warp roles
    // warp 角色相关常量
    constexpr int kMultimemUnroll = 16;
    // 多播加载/存储迭代的一次展开批大小
    constexpr int kNumBroadcastTileBytes = kNumSmemBytesPerBroadcastWarp / 2;
    // 广播 warp 的 tile 大小：共享内存对半分为双缓冲
    EP_STATIC_ASSERT(kNumSmemBytesPerBroadcastWarp > 0 and
                     kNumSmemBytesPerBroadcastWarp % (2 * kNumTMAAlignmentBytes) == 0,
                     "Invalid shared memory size per broadcast warp");
    // 编译期断言：广播 warp 的共享内存必须为正且按双缓冲 TMA 对齐
    constexpr int kNumEmissionLanes = 8;
    // 广播/发送回传时使用的 lane 数（每个 lane 维护一条到对端的流）
    constexpr int kNumPartitions = kNumBroadcastWarps / kNumReduceWarps;
    // 广播 warp 的分区数：每个归约 warp 对应的广播 warp 组数
    EP_STATIC_ASSERT(kNumMultimemWarps % 4 == 0 and (kNumIssueWarps + kNumReduceWarps + kNumBroadcastWarps) % 4 == 0,
                     "Invalid warp groups");
    // 编译期断言：寄存器重分配的两组 warp 必须按 4（warp 组）对齐
    EP_STATIC_ASSERT(kNumBroadcastWarps >= kNumReduceWarps and kNumBroadcastWarps % kNumReduceWarps == 0 and
                     kNumPartitions <= kNumRDMARanks and math::constexpr_ceil_div(kNumRDMARanks, kNumPartitions) <= 32,
                     "Each broadcast lane must own at most one AG stream");
    // 编译期断言：每个广播 lane 至多负责一条 all-gather 流（对端区间可被 lane 一一覆盖）

    // Registers
    // 寄存器预算
    constexpr int kNumMultimemRegisters = 176;
    // 多播归约 warp 每线程的寄存器配额
    constexpr int kNumOtherRegisters = 144;
    // 其他 warp 每线程的寄存器配额
    EP_STATIC_ASSERT(32 * (kNumMultimemWarps * kNumMultimemRegisters +
                          (kNumIssueWarps + kNumReduceWarps + kNumBroadcastWarps) * kNumOtherRegisters) <= 65536 - 2048,
                     "Register redistribution exceeds the CTA pool");
    // 编译期断言：寄存器重分配总量不得超过每 SM 上限（预留 2048 给系统）

    // Barriers
    // 命名屏障：用于多播组与配对发送 warp 之间的同步
    constexpr int kMultimemBarrierBaseIdx = 1;
    // 命名屏障起始编号
    EP_STATIC_ASSERT(kNumIssueWarps > 0 and kMultimemBarrierBaseIdx + kNumIssueWarps - 1 < 16,
                     "Invalid named barrier count");
    // 编译期断言：命名屏障编号必须小于 16（硬件限制）

    // Indices
    // 计算各类索引
    const int sm_idx = blockIdx.x;
    // 当前 CTA（SM）编号
    const int thread_idx = threadIdx.x;
    // 块内线程编号
    const int warp_idx = ptx::get_warp_idx();
    // 块内 warp 编号（决定角色）
    const int lane_idx = ptx::get_lane_idx();
    // warp 内 lane 编号
    const int rdma_rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    // 本进程在铁路式 RDMA 团队中的节点编号
    const int nvl_rank_idx = ncclTeamLsa(nccl_dev_comm).rank;
    // 本进程在机内 NVLink 团队中的卡编号

    // Init Gin
    // 初始化 GIN（RDMA 发起）通道
    constexpr int kNumChannels = kNumIssueWarps + kNumReduceWarps;
    // 通道数：发送 warp 数 + 归约 warp 数
    const int channel_idx = warp_idx < kNumMultimemWarps ?
        warp_idx / kNumMultimemWarpsPerGroup : (warp_idx - kNumMultimemWarps) % kNumChannels;
    // 通道编号：多播 warp 按所属组编号，其余 warp 依次连续编号
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannels>(sm_idx, channel_idx);
    // 根据全局通道编号选择 QP 及其共享模式
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx, sharing_mode);
    // 构造绑定到该 QP 的 GIN 封装

    // Shared memory and mbarriers
    // 准备共享内存与 mbarrier
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // 动态共享内存：按 TMA 对齐，归约/广播 warp 各自独占一段
    __shared__ __align__(8) uint64_t mbarriers[kNumReduceWarps + kNumBroadcastWarps];
    // 每个归约 warp 与广播 warp 各一个 mbarrier

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    // 取得 bucket 工作区
    auto& signals = workspace.signals[kContextIdx].all_reduce;
    // 本上下文 all-reduce 专用信号区
    auto& barrier_signals = workspace.signals[kContextIdx].barrier;
    // 栅栏信号数组
    const auto bcst_head = signals.counters;
    // 广播 head：对端广播进度（发送 warp 复用槽位前等待它）
    const auto rs_tail = bcst_head + kNumRDMARanks * kNumSlots;
    // reduce-scatter tail：各节点部分和的到达计数
    const auto ag_tail = rs_tail + kNumRDMARanks * kNumSlots;
    // all-gather tail：完全归约结果的回传到达计数
    const auto chunks = layout::ChunkView2D<kNumRDMARanks, kNumSlots, kNumChunkBytes>{workspace.chunk_storage};
    // chunk 存储二维视图：按 (rank, slot) 寻址

    // Every launch starts with fresh signal counters
    // 每次内核启动都要把信号计数器清零
    EP_STATIC_ASSERT(sizeof(layout::AllReduceSignals) % sizeof(int4) == 0 and
                     alignof(layout::AllReduceSignals) % alignof(int4) == 0,
                     "Signals must be a multiple of int4");
    // 编译期断言：信号区大小与对齐必须是 int4（16 字节）的倍数
    constexpr int kNumSignalVecs = math::constexpr_ceil_div(kNumSignalSlots * sizeof(int64_t), sizeof(int4));
    // 需要清零的 int4 向量数
    for (int i = sm_idx * kNumThreads + thread_idx; i < kNumSignalVecs; i += kNumSMs * kNumThreads)
        reinterpret_cast<int4*>(&signals)[i] = make_int4(0, 0, 0, 0);
    // 全网格线程跨步把所有信号计数器清零
    if (thread_idx < kNumReduceWarps + kNumBroadcastWarps)
        ptx::mbarrier_init_with_fence(reinterpret_cast<ptx::mbarrier*>(mbarriers + thread_idx), 1);
        // 前 (归约+广播) 个线程各自初始化一个 mbarrier

    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx,
        sm_idx, thread_idx);
    // 跨 RDMA + NVLink 域的入口 GPU 栅栏：让清零后的信号可见并确保输入数据就绪

    // Different warp roles
    // 以下按 warp 编号划分三种角色
    if (warp_idx < kNumMultimemWarps) {
        // ============ 角色 1：多播归约 warp（节点内 NVLink 归约） ============
        // Intra-node reduction
        // 节点内归约
        ptx::warpgroup_reg_realloc<kNumMultimemRegisters, kNumThreads>();
        // warpgroup 级寄存器重分配：为多播归约线程腾出更大的寄存器配额
        const int group_idx = warp_idx / kNumMultimemWarpsPerGroup;
        // 本 warp 所属的多播组编号
        const int group_thread_idx = (warp_idx % kNumMultimemWarpsPerGroup) * 32 + lane_idx;
        // 组内线程编号（组内所有 warp 的 lane 连续编号）
        const int global_group_idx = sm_idx * kNumIssueWarps + group_idx;
        // 全局组编号（与发送 warp 一一配对）
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumNVLRanks * kNumGlobalIssueWarps>(
            num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * global_group_idx);
        // chunk 迭代器：本卡需要归约的 chunk（起点为本卡编号，步长为卡数乘全局组数）
        while (iterator.next()) {
            // 迭代下一个 chunk
            const auto src = math::advance_ptr<const float4>(buffer_multimem, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
            // 多播窗口源地址：multimem 读会聚合节点内所有卡的输入
            const auto dst = math::advance_ptr<float4>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
            // 普通缓冲目的地址：节点内部分和写回这里
            const int num_vecs = iterator.num_chunk_bytes / static_cast<int>(sizeof(float4));
            // 以 float4 为单位的元素数
            const auto load = [](const float4* ptr, const int& lhs = 1, const int& rhs = 0) {
                return ptx::multimem_ld_reduce_add_f32_with_gt_pred(ptr, lhs, rhs);
                // 加载回调：multimem.ld_reduce.add 做节点内硬件求和
            };
            const auto store = [](float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) {
                ptx::st_with_gt_pred(ptr, value, lhs, rhs);
                // 存储回调：普通 store 写入本地缓冲
            };
            const auto process = [&](const float4& value) { return kWithScale ? ptx::fmul4(value, scale) : value; };
            // 后处理回调：按需乘以 scale
            utils::iterate_ld_st<kNumMultimemWarpsPerGroup, 32, kMultimemUnroll>(
                group_thread_idx, num_vecs, src, dst, load, store, process);
            // 组内线程并行地"多播读-（缩放）-写回"
            // Notify the issue warp
            // 通知配对的发送 warp：本 chunk 的节点内归约已完成
            ptx::named_barrier_unaligned<(kNumMultimemWarpsPerGroup + 1) * 32>(kMultimemBarrierBaseIdx + group_idx);
            // 命名屏障：多播组全部线程 + 配对发送 warp 一起到达
        }
    } else {
        ptx::warpgroup_reg_realloc<kNumOtherRegisters, kNumThreads>();
        // 非多播 warp 的寄存器重分配（较小的配额）
        if (warp_idx < kNumMultimemWarps + kNumIssueWarps) {
            // ============ 角色 2：发送 warp（reduce-scatter 阶段的 RDMA put） ============
            // RS Issue warp
            // reduce-scatter 发送 warp
            const int group_idx = warp_idx - kNumMultimemWarps;
            // 本发送 warp 配对的多播组编号
            const int global_group_idx = sm_idx * kNumIssueWarps + group_idx;
            // 全局组编号
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumNVLRanks * kNumGlobalIssueWarps>(
                num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * global_group_idx);
            // chunk 迭代器：与配对多播组相同的 chunk 序列
            while (iterator.next()) {
                // 迭代下一个 chunk
                const int rail_seq = iterator.chunk_idx / kNumNVLRanks;
                // 折算到 rail（节点）维度的序列号（同节点各卡共享一份）
                const int owner = rail_seq % kNumRDMARanks;
                // 该分片（chunk 所在 shard）的属主节点（环序）
                const int owner_seq = rail_seq / kNumRDMARanks;
                // 属主维度上的全局 chunk 序号
                const int slot = owner_seq % kNumSlots;
                // 对端暂存槽位（环形轮转）
                const int round = owner_seq / kNumSlots;
                // 槽位复用的轮次编号
                if (ptx::elect_one_sync()) {
                    // 单线程执行信用等待
                    // Wait until the previous generation has been consumed
                    // 等待该槽位的上一代广播结果已被对端消费（bcst head 信用）
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& last) {
                        // 带超时的自旋等待
                        const auto head = ptx::ld_acquire_sys(bcst_head + owner * kNumSlots + slot);
                        // acquire（系统级）读对端视角的广播 head
                        if (last and head < round)
                            printf("Hybrid all_reduce credit timeout r=%d n=%d owner=%d chunk=%d slot=%d round=%d head=%lld\n",
                                   rdma_rank_idx, nvl_rank_idx, owner, iterator.chunk_idx, slot, round,
                                   static_cast<long long>(head));
                            // 超时打印信用等待诊断信息
                        return head >= round;
                        // head 达到本轮次则槽位可复用
                    });
                }
                // Wait for the group's intra-node reduction
                // 等待配对多播组完成本 chunk 的节点内归约
                ptx::named_barrier_unaligned<(kNumMultimemWarpsPerGroup + 1) * 32>(kMultimemBarrierBaseIdx + group_idx);
                // 与多播组在同一条命名屏障上会合
                if (ptx::elect_one_sync()) {
                    // 单线程执行发送
                    auto* tail = rs_tail + rdma_rank_idx * kNumSlots + slot;
                    // 本节点该槽位的 rs_tail（到达计数，对称内存中各节点镜像同址）
                    if (owner == rdma_rank_idx) {
                        // 属主是自己：数据已在本地
                        ptx::red_add_rel_gpu(tail, 1);
                        // 直接以 release 语义把 tail 自增 1（发布"数据可读"）
                    } else {
                        gin.put<ncclTeamTagRail>(
                            chunks.get_chunk_ptr(rdma_rank_idx, slot), math::advance_ptr<void>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket),
                            // 目的：属主 chunk 存储区中本节点的槽位；源：本地缓冲中的节点内部分和
                            iterator.num_chunk_bytes, owner, 0,
                            ncclGin_StrongVASignalAdd{window, gin.get_sym_offset(tail), 1});
                            // RDMA put 附带强信号：送达后把属主视角的 rs_tail 自增 1
                    }
                }
                __syncwarp();
                // 同步 warp
            }
        } else if (warp_idx < kNumMultimemWarps + kNumChannels) {
            // ============ 角色 3：归约 warp（累加远端部分和并回传全归约结果） ============
            // Reduce warps: Accumulate remote chunks and send fully reduced chunks back
            // 归约 warp：累加远端 chunk 并把完全归约的结果发回各对端
            const int reduce_warp_idx = warp_idx - kNumMultimemWarps - kNumIssueWarps;
            // 块内归约 warp 编号（0 起）
            const int global_reduce_warp_idx = sm_idx * kNumReduceWarps + reduce_warp_idx;
            // 全局归约 warp 编号
            const auto mbarrier = reinterpret_cast<ptx::mbarrier*>(mbarriers + reduce_warp_idx);
            // 本归约 warp 的 mbarrier
            const auto shared = smem + reduce_warp_idx * kNumSmemBytesPerReduceWarp;
            // 本归约 warp 独占的共享内存段
            ptx::arrival_phase phase = 0;
            // mbarrier 相位计数
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumNVLRanks * kNumRDMARanks * kNumGlobalReduceWarps>(
                num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * (rdma_rank_idx + kNumRDMARanks * global_reduce_warp_idx));
            // chunk 迭代器：本节点需要最终归约的 chunk（步长跨卡与节点维度）
            while (iterator.next()) {
                // 迭代下一个待归约 chunk
                const int owner_seq = iterator.chunk_idx / (kNumNVLRanks * kNumRDMARanks);
                // 属主维度上的全局 chunk 序号
                const int slot = owner_seq % kNumSlots;
                // 对应的暂存槽位
                const int target = owner_seq / kNumSlots + 1;
                // tail 目标值：该槽位本轮已被写入的总轮次
                const auto input = math::advance_ptr<void>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
                // 本地部分和（第一累加项）
                const auto out = chunks.get_chunk_ptr(rdma_rank_idx, slot);
                // 归约结果暂存到本节点 chunk 存储区的该槽位

                // Wait for all inputs before overwriting this slot's reduced result
                // 在覆写该槽位的归约结果前，必须等齐所有节点的输入
                // NOTES: New arrivals also acknowledge consumption of the previous result
                // 说明：新到达的数据同时隐含确认了上一轮结果已被对端取走
                for (int source = lane_idx; source < kNumRDMARanks; source += 32) {
                    // 各 lane 分摊源节点，并行等待
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& last) {
                        // 带超时的自旋等待
                        const auto tail = ptx::ld_acquire_sys(rs_tail + source * kNumSlots + slot);
                        // acquire 读各源节点的 rs_tail
                        if (last and tail < target)
                            printf("Hybrid all_reduce RS timeout r=%d n=%d source=%d chunk=%d slot=%d target=%d tail=%lld\n",
                                   rdma_rank_idx, nvl_rank_idx, source, iterator.chunk_idx, slot, target,
                                   static_cast<long long>(tail));
                            // 超时打印 RS 等待诊断信息
                        return tail >= target;
                        // 该源的数据已到达
                    });
                }
                __syncwarp();
                // 同步 warp：确保所有输入就绪
                if (ptx::elect_one_sync()) {
                    // 单线程执行 TMA 归约
                    #pragma unroll 1
                    // 禁止展开
                    for (int step = 0; step < kNumRDMARanks; ++ step) {
                        // 按环序逐个累加各节点
                        const int source = (rdma_rank_idx + step) % kNumRDMARanks;
                        // 环序中的源节点
                        const auto src = step == 0 ? input : chunks.get_chunk_ptr(source, slot);
                        // 第 0 步取本地部分和，其余取对端槽位中的部分和
                        #pragma unroll 1
                        // 禁止展开：chunk 分多轮搬运
                        for (int offset = 0; offset < iterator.num_chunk_bytes; offset += kNumSmemBytesPerReduceWarp) {
                            // 按共享内存段大小分块处理
                            const int count = min(kNumSmemBytesPerReduceWarp, iterator.num_chunk_bytes - offset);
                            // 本轮搬运的字节数
                            ptx::tma_store_wait();
                            // 等待上一轮 TMA store 完成
                            ptx::tma_load_1d(shared, math::advance_ptr<const void>(src, offset), mbarrier, count);
                            // 发起 1D TMA load：把该块读入共享内存
                            ptx::mbarrier_arrive_and_set_tx(mbarrier, count);
                            // 设置事务字节数并到达
                            ptx::mbarrier_wait_and_flip_phase(mbarrier, phase);
                            // 等待本块 load 完成
                            if (step == 0)
                                ptx::tma_store_1d(math::advance_ptr<void>(out, offset), shared, count);
                                // 第一个源：普通 TMA store 把数据写入槽位（建立初值）
                            else
                                ptx::tma_store_reduce_add_f32(math::advance_ptr<float>(out, offset), shared, count);
                                // 后续源：TMA store reduce-add 硬件累加进槽位
                            ptx::tma_store_commit();
                            // 提交异步存储
                        }
                        ptx::tma_store_wait();
                        // 等待该源的所有块写完
                    }
                    ptx::red_add_rel_gpu(ag_tail + global_reduce_warp_idx * kNumRDMARanks + rdma_rank_idx, 1);
                    // 以 release 语义自增本（归约 warp, 节点）的 ag_tail：完全归约结果已就绪
                }
                __syncwarp();
                // 同步 warp

                #pragma unroll 1
                // 禁止展开
                for (int d = lane_idx; lane_idx < kNumEmissionLanes and d < kNumRDMARanks - 1; d += kNumEmissionLanes) {
                    // 前 kNumEmissionLanes 个 lane 并行把结果发往各对端（每个 lane 一条流）
                    const int peer = (rdma_rank_idx + 1 + d) % kNumRDMARanks;
                    // 环序中除自身外的目的节点
                    gin.put<ncclTeamTagRail>(
                        math::advance_ptr<void>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket),
                        out, iterator.num_chunk_bytes, peer, 0,
                        // 目的：对端输出缓冲（属主分片位置）；源：槽位中的完全归约结果
                        ncclGin_StrongVASignalAdd{window, gin.get_sym_offset(ag_tail + global_reduce_warp_idx * kNumRDMARanks + rdma_rank_idx), 1});
                        // RDMA put 附带强信号：送达后自增本端的 ag_tail
                }
                __syncwarp();
                // 同步 warp
            }
        } else {
            // ============ 角色 4：广播 warp（把完全归约结果多播回节点内所有卡） ============
            // Intra-node broadcast
            // 节点内广播
            const int broadcast_warp_idx = warp_idx - kNumMultimemWarps - kNumChannels;
            // 块内广播 warp 编号（0 起）
            const int global_reduce_warp_idx = sm_idx * kNumReduceWarps + broadcast_warp_idx % kNumReduceWarps;
            // 本广播 warp 归属的归约 warp（ag_tail 计数器按归约 warp 索引）
            // Each broadcast warp watches a contiguous range of peers
            // 每个广播 warp 监视一段连续的属主节点区间
            const int partition_idx = broadcast_warp_idx / kNumReduceWarps;
            // 本广播 warp 的分区编号
            const int peer_begin = partition_idx * kNumRDMARanks / kNumPartitions;
            // 本分区负责的起始属主节点
            const int peer_end = (partition_idx + 1) * kNumRDMARanks / kNumPartitions;
            // 本分区负责的结束属主节点（不含）
            const int peer = peer_begin + lane_idx;
            // 本 lane 负责的属主节点（每 lane 一条 all-gather 流）
            constexpr int kStride = kNumNVLRanks * kNumRDMARanks * kNumGlobalReduceWarps;
            // 广播流的全局步长（跨卡与节点维度）
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kStride>(
                num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * (peer + kNumRDMARanks * global_reduce_warp_idx));
            // chunk 迭代器：本 lane 负责的 (属主, 卡) 组合的全部 chunk
            iterator.next();
            // 先推进到第一个 chunk（用于计算本 lane 的总交付数）
            const int length = peer < peer_end and iterator.chunk_idx < iterator.num_total_chunks
                ? (iterator.num_total_chunks - 1 - iterator.chunk_idx) / kStride + 1 : 0;
            // 本 lane 需要广播的 chunk 数（无效 lane 为 0）
            const auto mbarrier = reinterpret_cast<ptx::mbarrier*>(mbarriers + kNumReduceWarps + broadcast_warp_idx);
            // 本广播 warp 的 mbarrier（排在归约 warp 的之后）
            const auto shared = smem + kNumReduceWarps * kNumSmemBytesPerReduceWarp + broadcast_warp_idx * 2 * kNumBroadcastTileBytes;
            // 本广播 warp 独占的双缓冲共享内存（两块 tile）
            ptx::arrival_phase phase = 0;
            // mbarrier 相位计数
            int delivered = 0;
            // 本 lane 已交付（广播完成）的 chunk 数

            while (ptx::any(delivered < length)) {
                // warp 内任一 lane 还有待广播的 chunk 就继续循环
                unsigned ready = 0;
                // 就绪位图：记录哪些 lane 的下一个 chunk 已到达
                comm::timeout_while<kNumTimeoutCycles>([&](const bool& last) {
                    // 带超时的自旋等待：直到至少一条流有新数据到达
                    const auto arrived = delivered < length ? ptx::ld_acquire_sys(ag_tail + global_reduce_warp_idx * kNumRDMARanks + peer) : delivered;
                    // acquire 读本 lane 对应属主的 ag_tail（无任务的 lane 视为已交付）
                    ready = ptx::gather(arrived > delivered);
                    // 归并各 lane 的"有新数据"标志成 warp 位图
                    if (last and delivered < length and arrived <= delivered)
                        printf("Hybrid all_reduce AG timeout r=%d n=%d owner=%d j=%d delivered=%d length=%d tail=%lld\n",
                               rdma_rank_idx, nvl_rank_idx, peer, global_reduce_warp_idx, delivered, length,
                               static_cast<long long>(arrived));
                        // 超时打印 AG 等待诊断信息
                    return ready != 0;
                    // 至少一条流就绪即可继续
                });
                __syncwarp();
                // 同步 warp
                while (ready) {
                    // 逐个处理就绪的流
                    const int selected_lane_idx = ptx::ffs(ready);
                    // 取最低就绪位对应的 lane
                    const int owner = peer_begin + selected_lane_idx;
                    // 该 lane 对应的属主节点
                    int chunk_idx = 0;
                    // 选中 lane 提供的 chunk 全局序号
                    int num_bytes = 0;
                    // 该 chunk 的字节数
                    int64_t offset = 0;
                    // 该 chunk 在缓冲中的偏移
                    if (lane_idx == selected_lane_idx) {
                        // 仅被选中的 lane 提供元信息并推进自己的迭代器
                        chunk_idx = iterator.chunk_idx;
                        // 当前 chunk 序号
                        num_bytes = iterator.num_chunk_bytes;
                        // 当前 chunk 字节数
                        offset = iterator.bucket.offset + iterator.chunk_offset_in_bucket;
                        // 当前 chunk 偏移
                        iterator.next();
                        // 推进到下一个 chunk
                        ++ delivered;
                        // 交付计数加一
                    }
                    chunk_idx = ptx::exchange(chunk_idx, selected_lane_idx);
                    // 把选中 lane 的 chunk 序号广播给全 warp
                    num_bytes = ptx::exchange(num_bytes, selected_lane_idx);
                    // 广播字节数
                    offset = ptx::exchange(offset, selected_lane_idx);
                    // 广播偏移
                    const int slot = chunk_idx / (kNumNVLRanks * kNumRDMARanks) % kNumSlots;
                    // 由 chunk 序号反推暂存槽位
                    const auto src = owner == rdma_rank_idx
                        ? static_cast<const int8_t*>(chunks.get_chunk_ptr(rdma_rank_idx, slot))
                        // 属主是自己：直接从本地槽位读取
                        : math::advance_ptr<const int8_t>(buffer, offset);
                        // 否则从输出缓冲读取（由归约 warp 回传的结果）
                    hybrid_ar_broadcast<kNumBroadcastTileBytes>(
                        src, math::advance_ptr<float4>(buffer_multimem, offset), num_bytes, shared, mbarrier, phase);
                    // 调用双缓冲 TMA + multimem 多播广播：把结果写回节点内所有卡
                    if (ptx::elect_one_sync())
                        ptx::red_add_rel_gpu(bcst_head + owner * kNumSlots + slot, 1);
                        // 以 release 语义自增该属主槽位的 bcst_head：结果已多播，槽位可复用
                    __syncwarp();
                    // 同步 warp
                    ready &= ready - 1;
                    // 清除最低就绪位，处理下一条流
                }
            }
        }
    }

    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks, kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx);
    // 出口 GPU 栅栏（带 QP flush 语义）：确保所有 RDMA put 在对端全局可见后内核才结束
}

}  // namespace deep_ep::bucket
// 退出命名空间
