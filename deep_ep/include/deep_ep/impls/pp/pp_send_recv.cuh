#pragma once
// 头文件保护：保证本文件只被包含一次

#include <algorithm>
// 引入 std::min 等算法
#include <cooperative_groups.h>
// 引入协作组（用于网格级同步 this_grid().sync()）
#include <utility>
// 引入 std::pair / std::make_pair

#include <deep_ep/comm/barrier.cuh>
// 引入通信层超时等待原语（comm::timeout_while）
#include <deep_ep/comm/handle.cuh>
// 引入 NCCL Gin 通信句柄封装（NCCLGin）
#include <deep_ep/common/compiled.cuh>
// 引入编译期常量与静态断言（EP_STATIC_ASSERT 等）
#include <deep_ep/common/exception.cuh>
// 引入设备端断言（EP_DEVICE_ASSERT）
#include <deep_ep/common/math.cuh>
// 引入常用数学工具（ceil_div、advance_ptr 等）
#include <deep_ep/common/ptx.cuh>
// 引入 PTX 内联工具（mbarrier、TMA、elect_one_sync 等）
#include <deep_ep/layout/pp/workspace.cuh>
// 引入流水线并行的信号布局（PPSignals）

namespace deep_ep::pp {
// pp 命名空间：流水线并行（pipeline parallelism）相邻 rank 收发实现

// Returns `(local_idx, peer_idx)`. `local_idx` is this rank's buffer half for the peer (0 for the
// next rank, 1 for the previous rank) and `peer_idx` is the symmetric index the peer uses to
// address the same half.
// 返回 `(local_idx, peer_idx)`。`local_idx` 是本 rank 面向该对端使用的缓冲半区编号
// （发往下一个 rank 用 0，发往上一个 rank 用 1）；`peer_idx` 是对端在对称寻址本 rank
// 同一半区时使用的索引（方向恰好相反）。
template <int kNumRanks>
__device__ __forceinline__ std::pair<int, int> get_pp_buffer_offset(
    const int& src_rank_idx, const int& dst_rank_idx) {
    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;
    // 计算源 rank 在环形拓扑中的下一个 rank
    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
    // 若目标就是下一 rank：本端用半区 0、对端视其为半区 1；否则（上一 rank）反之
}

template <int kNumSMs,
          int kNumSmemBytes,
          int kNumStages = 2,
          int kNumTMABytesPerStage = math::constexpr_align<int, false>(
              (kNumSmemBytes - kNumStages * static_cast<int>(sizeof(ptx::mbarrier))) / kNumStages,
              kNumTMAAlignmentBytes),
          int kNumTMABlocksPerStage = kNumTMABytesPerStage / kNumTMAAlignmentBytes>
__device__ __forceinline__ void pp_tma_copy(
    // 模板参数：kNumSMs 总 SM 数；kNumSmemBytes 动态共享内存总字节数；
    // kNumStages 流水线级数；kNumTMABytesPerStage 每级 TMA 缓冲字节数（对齐到 TMA 对齐要求）；
    // kNumTMABlocksPerStage 每级能容纳的 TMA 块数
    void* src_ptr, void* dst_ptr,
    const int64_t& num_bytes, const int& sm_idx) {
    // 用 TMA 在共享内存中转，把 num_bytes 字节从 src_ptr 多级流水线地拷到 dst_ptr
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // 声明按 TMA 对齐的动态共享内存
    const auto tma_buffers = smem;
    // 前段作为各级 TMA 数据缓冲
    const auto mbarriers = reinterpret_cast<ptx::mbarrier*>(smem + kNumStages * kNumTMABytesPerStage);
    // 末尾紧接着放各级的 mbarrier
    EP_STATIC_ASSERT(kNumTMABytesPerStage > 0, "Invalid shared memory bytes");
    // 编译期断言：每级缓冲必须为正
    EP_STATIC_ASSERT(kNumStages >= 2, "Need at least 2 stages for pipelining");
    // 编译期断言：流水线至少需要两级

    // Init mbarriers
    // 初始化各级 mbarrier
    ptx::arrival_phase phases[kNumStages];
    // 记录各级当前等待的相位（用于交替翻转）
    #pragma unroll
    for (int s = 0; s < kNumStages; ++ s)
        phases[s] = 0, ptx::mbarrier_init_with_fence(mbarriers + s, 1);
    // 相位清零；初始化每级 mbarrier 期望到达数为 1（单个发起线程），并带栅栏发布

    // Work partitioning across SMs
    // 按 SM 平均切分整块拷贝任务
    EP_DEVICE_ASSERT(num_bytes % kNumTMAAlignmentBytes == 0);
    // 设备端断言：总字节数必须按 TMA 对齐
    const auto num_tma_blocks = num_bytes / kNumTMAAlignmentBytes;
    // 总 TMA 块数
    const auto num_tma_blocks_per_sm = math::ceil_div<int64_t>(num_tma_blocks, kNumSMs);
    // 每个 SM 平均分到的块数（向上取整）
    const auto start_block_idx = sm_idx * num_tma_blocks_per_sm;
    // 本 SM 负责的起始块编号
    const auto end_block_idx = std::min(start_block_idx + num_tma_blocks_per_sm, num_tma_blocks);
    // 本 SM 负责的结束块编号（不超过总块数）
    const auto num_iterations = math::ceil_div<int64_t>(end_block_idx - start_block_idx, kNumTMABlocksPerStage);
    // 本 SM 的流水线迭代次数（每次迭代处理一级缓冲能容纳的块数）

    auto get_iter_info = [&](const int64_t& iter_idx) {
        // 内联 lambda：由迭代编号得到字节偏移与本迭代的事务字节数
        const auto i = start_block_idx + iter_idx * kNumTMABlocksPerStage;
        // 本迭代的第一个 TMA 块编号
        const auto offset = i * kNumTMAAlignmentBytes;
        // 换算成字节偏移
        const auto num_transaction_bytes =
            std::min<int>(kNumTMABlocksPerStage, end_block_idx - i) * kNumTMAAlignmentBytes;
        // 事务字节数 = 本迭代实际块数（末尾可能不满）× 对齐字节数
        return std::make_pair(offset, num_transaction_bytes);
        // 返回 (偏移, 字节数) 二元组
    };

    // Fill pipeline: issue loads for the first kNumStages iterations
    // 填充流水线：先为前 kNumStages 次迭代发起 TMA 加载
    for (int64_t iter_idx = 0; iter_idx < kNumStages and iter_idx < num_iterations; ++ iter_idx) {
        const auto [load_offset, num_load_bytes] = get_iter_info(iter_idx);
        // 取本迭代的加载偏移与字节数
        ptx::tma_load_1d(
            tma_buffers + iter_idx * kNumTMABytesPerStage,
            math::advance_ptr(src_ptr, load_offset),
            mbarriers + iter_idx, num_load_bytes);
        // 发起 1D TMA 加载：源全局内存 -> 本级共享内存缓冲，完成时通知本级 mbarrier
        ptx::mbarrier_arrive_and_set_tx(mbarriers + iter_idx, num_load_bytes);
        // 预置该 mbarrier 的期望事务字节数（TMA 完成字节数与之匹配即放行）
    }

    for (int64_t iter_idx = 0; iter_idx < num_iterations; ++ iter_idx) {
        // 主循环：逐迭代完成"等加载 -> 存储 -> 预取下一批"
        const auto stage_idx = static_cast<int>(iter_idx % kNumStages);
        // 本迭代使用的流水线级号（循环复用）
        const auto [store_offset, num_store_bytes] = get_iter_info(iter_idx);
        // 取本迭代的存储偏移与字节数

        // Wait this stage's load and issue store
        // 等待本级加载完成后发起存储
        ptx::mbarrier_wait_and_flip_phase(mbarriers + stage_idx, phases[stage_idx]);
        // 阻塞等待本级 mbarrier 完成当前相位，并翻转下一轮等待的相位
        ptx::tma_store_1d(
            math::advance_ptr(dst_ptr, store_offset),
            tma_buffers + stage_idx * kNumTMABytesPerStage,
            num_store_bytes);
        // 发起 1D TMA 存储：本级共享内存缓冲 -> 目标全局内存
        ptx::tma_store_commit();
        // 提交存储操作（进入 TMA 存储队列）

        // Prefetch: wait until this stage's store is completed, then issue next load
        // 预取：等本级的存储完成（缓冲可复用）后，再发起 kNumStages 之后的下一批加载
        const auto next_iter_idx = iter_idx + kNumStages;
        // 与本迭代共用同一级的后续迭代编号
        if (next_iter_idx < num_iterations) {
            ptx::tma_store_wait();
            // 等待所有已提交的 TMA 存储完成，确保本级缓冲不再被读取
            const auto [load_offset, num_load_bytes] = get_iter_info(next_iter_idx);
            // 取下一批的加载偏移与字节数
            ptx::tma_load_1d(
                tma_buffers + stage_idx * kNumTMABytesPerStage,
                math::advance_ptr(src_ptr, load_offset),
                mbarriers + stage_idx, num_load_bytes);
            // 复用本级缓冲发起下一次 TMA 加载
            ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_load_bytes);
            // 重置本级 mbarrier 的事务字节数（相位已翻转，等待新一批完成）
        }
    }

    // Drain all outstanding stores
    // 排空所有未完成的 TMA 存储，保证拷贝全部落盘
    ptx::tma_store_wait();
}

// Send `x` to an adjacent rank. `x` is first staged into a local send slot (TMA), then pushed into
// the peer's recv slot through Gin. One slot is released back to us once the peer has consumed it.
// 把张量 `x` 发送给相邻 rank。`x` 先由 TMA 中转到本地发送槽位，再经 Gin 推送到对端的接收槽位；
// 对端消费完后会把一个槽位释放回本端（环形限流，最多 num_max_inflight_tensors 个在途张量）。
template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
          int kNumQPs,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_send_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             // 模板参数：SM 数、rank 总数、共享内存字节数、QP 数、超时周期数
             // 内核参数：NCCL 通信对象与窗口、源数据指针及字节数、环形缓冲与工作区、
             //          本 rank 与目标 rank 编号、单张量最大字节数、最大在途张量数
             void* x, const int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int dst_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 SM（块）编号
    auto& pp = *static_cast<layout::PPSignals*>(workspace);
    // 引用流水线并行信号区（发送/接收计数、到达/释放信号）
    const auto [local_idx_in_dst, dst_idx_in_local] = get_pp_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);
    // 本端在对端缓冲中的半区编号 local_idx_in_dst；对端在本端缓冲中的半区编号 dst_idx_in_local

    EP_STATIC_ASSERT(kNumQPs > 0 and kNumQPs <= 2, "Invalid number of PP QPs");
    // 编译期断言：QP 数必须为 1 或 2

    // Buffer offsets
    // 计算环形缓冲槽位偏移
    const auto send_count = pp.send_count[dst_idx_in_local];
    // 本端已向该方向发送的张量总数（历史累计，每次递增）
    const auto slot_idx = send_count % num_max_inflight_tensors;
    // 当前使用的槽位号：按在途张量数取模循环复用
    auto send_buffer_ptr = math::advance_ptr(
        buffer, ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    // 本地发送槽位地址：缓冲分四个半区段（0/1 为接收，2/3 为发送），按方向与槽位定位
    auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((local_idx_in_dst + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    // 对端接收槽位（对称寻址）地址：使用对端面向本端的接收半区（0/1）

    // Wait buffer slot release and do TMA
    // 等待槽位被对端释放，然后用 TMA 拷入发送槽位
    if (ptx::elect_one_sync()) {
        // 每 SM 选出一个代表线程执行等待与 TMA
        const auto release_target = send_count - num_max_inflight_tensors + 1;
        // 需要等待的目标释放数：腾出当前要用的那个槽位
        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            if (ptx::ld_acquire_sys(&pp.release[dst_idx_in_local]) >= release_target)
                return true;
            // 以系统范围 acquire 语义读释放计数，达到目标即槽位可用
            if (is_last_check)
                printf("DeepEP PP send timeout, rank %d, peer %d, recv buffer is full\n", rank_idx, dst_rank_idx);
            // 超时打印：接收缓冲已满（对端消费过慢）
            return false;
            // 继续等待
        });
        pp_tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);
        // 各 SM 分片协作：把 x 通过 TMA 多级流水拷入本地发送槽位
    }
    cooperative_groups::this_grid().sync();
    // 网格级同步：确保整块 x 都已写入发送槽位后才能发起 RDMA

    // Issue RDMA put
    // 发起 RDMA put：把发送槽位内容推到对端接收槽位
    if (ptx::elect_one_sync()) {
        // 每 SM 选出一个代表线程
        // Split buffer among QPs
        // 把数据按 QP 数均分，多个 QP 并行传输
        const auto num_bytes_per_qp = num_x_bytes / kNumQPs / kNumRDMAAlignmentBytes * kNumRDMAAlignmentBytes;
        // 每个 QP 负责的字节数（向下取整到 RDMA 对齐单位）
        for (int context_idx = sm_idx; context_idx < kNumQPs; context_idx += kNumSMs) {
            // SM 轮流认领 QP：SM 数多于 QP 数时前面的 SM 各管一个，其余空闲
            const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, context_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
            // 构造该 QP 的 Gin 句柄；资源按 CTA 共享
            const auto offset = context_idx * num_bytes_per_qp;
            // 本 QP 负责的数据在张量内的起始偏移
            const auto num_bytes = context_idx == kNumQPs - 1 ? num_x_bytes - offset : num_bytes_per_qp;
            // 最后一个 QP 吃掉剩余的全部字节（含对齐取整的零头）
            const ncclGin_StrongVASignalAdd remote_action {
                .signalWindow = nccl_window,
                .signalOffset = gin.get_sym_offset(&pp.arrival[local_idx_in_dst][context_idx]),
                .value = 1,
            };
            // 定义远端强序 VA 信号动作：数据到达后在对端窗口的 arrival[方向][QP] 计数上原子加 1
            // Always signal, as the receiver waits on every QP even if it has no data
            // 必须始终发信号：即使某 QP 没有数据，接收端也在每个 QP 上等待
            if (num_bytes > 0) {
                gin.put<ncclTeamTagWorld>(
                    math::advance_ptr(recv_buffer_ptr, offset), math::advance_ptr(send_buffer_ptr, offset),
                    num_bytes, dst_rank_idx, 0, remote_action);
                // RDMA put：把发送槽位本 QP 分片写入对端接收槽位对应位置，完成后触发远端信号
            } else {
                gin.signal<ncclTeamTagWorld>(dst_rank_idx, remote_action);
                // 该 QP 无数据（数据太小或分片为空）时只发信号
            }
        }
        if (sm_idx == 0)
            pp.send_count[dst_idx_in_local] += 1;
        // 由 0 号 SM 统一把发送计数加一（记录槽位已被占用）
    }
}

// Receive `x` from an adjacent rank. The data is staged in our recv slot by the peer, then copied
// out with TMA. Once copied, we release the slot back to the peer.
// 从相邻 rank 接收张量。对端已把数据放入本端接收槽位；本端用 TMA 拷出，
// 拷完后把槽位释放回对端（供其复用发送槽位）。
template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
          int kNumQPs,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_recv_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             // 模板参数：SM 数、rank 总数、共享内存字节数、QP 数、超时周期数
             // 内核参数：NCCL 通信对象与窗口、输出张量及字节数、环形缓冲与工作区、
             //          本 rank 与来源 rank 编号、单张量最大字节数、最大在途张量数
             void* x, const int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int src_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 SM（块）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内线程编号（本内核每块仅 1 warp）
    auto& pp = *static_cast<layout::PPSignals*>(workspace);
    // 引用流水线并行信号区
    const auto [src_idx_in_local, local_idx_in_src] = get_pp_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);
    // 来源 rank 在本端缓冲中的半区编号 src_idx_in_local；本端在来源缓冲中的半区编号 local_idx_in_src
    EP_STATIC_ASSERT(kNumQPs > 0 and kNumQPs <= 2, "Invalid number of PP QPs");
    // 编译期断言：QP 数必须为 1 或 2

    // Gin handle
    // 构造 Gin 句柄（QP 0，资源按 CTA 共享；仅用于最后的释放信号）
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Buffer offsets
    // 计算接收槽位偏移
    const auto recv_count = pp.recv_count[src_idx_in_local];
    // 本端已从该方向接收的张量总数
    const auto slot_idx = recv_count % num_max_inflight_tensors;
    // 本次要读的槽位号
    auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    // 接收槽位地址：按来源方向半区与槽位号定位

    // Wait arrival and copy from the buffer into a new tensor
    // 等待每个 QP 的到达信号，然后把数据从缓冲拷出成新张量
    if (thread_idx < kNumQPs) {
        // 前 kNumQPs 个线程各负责等待一个 QP 的到达信号
        const auto& context_idx = thread_idx;
        // 线程号即 QP 编号
        const auto arrival_target = recv_count + 1;
        // 需要等待的到达目标值：历史计数加一（本次传输）
        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            const auto arrival = ptx::ld_acquire_sys(&pp.arrival[src_idx_in_local][context_idx]);
            // 以系统范围 acquire 语义读该 QP 的到达计数
            if (arrival >= arrival_target)
                return true;
            // 达到目标即该 QP 分片的数据已全部写入
            if (is_last_check)
                printf("DeepEP PP recv timeout, rank %d, peer %d, context %d, arrival %lld, target %lld\n",
                       rank_idx, src_rank_idx, context_idx,
                       static_cast<long long>(arrival), static_cast<long long>(arrival_target));
            // 超时打印：rank、对端、QP、当前到达数与目标值
            return false;
            // 继续等待
        });
    }
    __syncwarp();
    // warp 内同步：等所有 QP 的到达信号都满足后再拷贝
    if (ptx::elect_one_sync()) {
        pp_tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);
        // 各 SM 分片协作：把接收槽位的数据 TMA 拷出到输出张量 x
    }
    cooperative_groups::this_grid().sync();
    // 网格级同步：确保整块数据都拷完后才能释放槽位

    // Release the slot back to the peer
    // 把槽位释放回对端
    if (sm_idx == 0 and ptx::elect_one_sync()) {
        // 由 0 号 SM 选出的一个代表线程执行释放
        gin.signal<ncclTeamTagWorld>(
            src_rank_idx,
            ncclGin_StrongVASignalAdd {
                .signalWindow = nccl_window,
                .signalOffset = gin.get_sym_offset(&pp.release[local_idx_in_src]),
                .value = 1,
            });
        // 向来源 rank 发信号：对本端窗口中 release[本端在对端眼里的方向] 计数原子加一
        // （该地址对称映射到对端视角，即对端的 release[它面向本端的方向]）
        pp.recv_count[src_idx_in_local] += 1;
        // 本端接收计数加一（推进下次读取的槽位）
    }
}

}  // namespace deep_ep::pp
// 结束 pp 命名空间
