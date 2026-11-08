#pragma once
// 头文件保护：避免重复包含

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/comm/handle.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
// 引入 GPU 栅栏协议、NCCL 通信句柄、指针运算、PTX 辅助函数与 bucket 工作区布局

namespace deep_ep::bucket {
// DeepEP bucket 集合通信实现命名空间

// 模板参数：kNumRDMARanks 跨机（RDMA）rank 数；kNumNVLRanks 机内（NVLink）rank 数；
// kNumThreads 每 CTA 线程数；kNumTimeoutCycles 超时周期；kContextIdx 上下文编号；
// kNumQPsPerChunk 每个 chunk 使用的 RDMA 队列对（QP）数量
template <int kNumRDMARanks, int kNumNVLRanks, int kNumThreads,
          int64_t kNumTimeoutCycles, int kContextIdx,
          int kNumQPsPerChunk = layout::AllGatherSignals::kNumQPsPerChunk>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限制每 SM 仅 1 个 CTA 驻留
hybrid_all_gather(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                  void* workspace_ptr, void* buffer,
                  const int num_buckets, const int num_chunks, const int num_chunk_bytes,
                  const int rdma_rank_idx, const int nvl_rank_idx, const int rank_idx,
                  const __grid_constant__ comm::BucketList buckets) {
    // 混合（NVLink + RDMA）all-gather 内核：先用 RDMA put 把本地分片发往远端节点，
    // 再由远端节点内部通过 NVLink 完成机内分发；此处实现的是 RDMA put + 远端信号（tails）一侧
    // One SM per RDMA peer, one warp per QP
    // 网格划分：每个 SM（CTA）负责一个 RDMA 对端节点，每个 warp 负责一个 QP
    EP_STATIC_ASSERT(kNumRDMARanks > 1 and kNumNVLRanks > 1, "Hybrid all-gather requires both RDMA and NVLink ranks");
    // 编译期断言：混合路径要求 RDMA 与 NVLink 两类 rank 都存在
    EP_STATIC_ASSERT(kNumNVLRanks <= kNumThreads and kNumQPsPerChunk * 32 <= kNumThreads, "Insufficient threads");
    // 编译期断言：线程数必须足够覆盖 NVLink rank 数与所有 QP 的 warp 分配

    // Indices: SM index is the destination RDMA rank, warp index is the QP index
    // 索引约定：SM 编号即目标 RDMA rank 编号，warp 编号即 QP 编号
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 CTA（= 目标 RDMA rank）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内线程编号
    const auto warp_idx = ptx::get_warp_idx();
    // 块内 warp 编号（= QP 编号）

    // NOTES: all QPs of this SM are only used by this CTA, the QP index only matters for the issuing warps
    // 说明：本 SM 的所有 QP 只被本 CTA 独占使用，QP 编号仅对发起请求的 warp 有意义
    const comm::NCCLGin gin(nccl_dev_comm, window, warp_idx % kNumQPsPerChunk, NCCL_GIN_RESOURCE_SHARING_CTA);
    // 构造 GIN 封装：按 warp 选取 QP，资源共享粒度为 CTA 级
    auto& bucket_signals = static_cast<layout::BucketWorkspace*>(workspace_ptr)->signals[kContextIdx];
    // 取出本上下文的信号区（包含 all-gather 的 tails 与 barrier 信号）
    auto& signals = bucket_signals.all_gather;
    // all-gather 专用的 tails 信号数组（每个 QP 一组）

    // Clear tails
    // 清零所有 chunk 的 tails 信号，为新一轮通信做准备
    for (int i = sm_idx * kNumThreads + thread_idx; i < num_chunks; i += kNumRDMARanks * kNumThreads) {
        // 用全体 CTA 的线程跨步覆盖所有 chunk 索引
        #pragma unroll
        // 提示编译器展开内层循环
        for (int j = 0; j < kNumQPsPerChunk; ++ j)
            signals.tails[j][i] = 0;
        // 每个 QP 的 tail 数组在该 chunk 处清零
    }

    // Entry barrier: make the cleared tails visible before any peer signals
    // 入口栅栏：在对端发出任何信号之前，先让清零后的 tails 对全体可见
    // NOTES: no flush is needed, the previous puts are already confirmed by the tail signals
    // 说明：无需 flush，此前的写入已经由 tail 信号机制确认
    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumRDMARanks, kNumThreads, kNumQPsPerChunk,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, bucket_signals.barrier,
        rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx);
    // 跨 RDMA 域与 NVLink 域的 GPU 栅栏，等待所有节点/卡全部到达

    // Each warp is responsible for a QP
    // 每个 warp 负责一个 QP 发起数据传输
    if (sm_idx == rdma_rank_idx or warp_idx >= kNumQPsPerChunk)
        return;
    // 本节点自身的 SM（不需要给自己发数据）以及超出 QP 数量的 warp 直接退出

    // Issue in order: each QP sends its own part of every chunk and signals once per chunk
    // 按序发起：每个 QP 发送每个 chunk 中属于自己的部分，并对每个 chunk 发一次信号
    if (ptx::elect_one_sync()) {
        // warp 内选出单个活跃线程，避免重复发起相同的 RDMA 操作
        int chunk_idx = 0;
        // 全局 chunk 计数器（跨桶连续编号）
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            // 逐桶遍历
            const auto& [bucket_offset, num_bucket_bytes] = buckets[bucket_idx];
            // 解构桶描述：桶在 buffer 中的偏移与该桶每 rank 分片的字节数
            const auto shard_ptr = math::advance_ptr(buffer, bucket_offset + rank_idx * num_bucket_bytes);
            // 本 rank 在该桶内分片的起始地址
            for (int64_t chunk_offset = 0; chunk_offset < num_bucket_bytes; chunk_offset += num_chunk_bytes, ++ chunk_idx) {
                // 将桶按 num_chunk_bytes 切成多个 chunk，逐 chunk 发送
                // Split the chunk evenly across QPs, only the last chunk of a bucket may be uneven
                // 将 chunk 均匀切分给各 QP，仅桶内最后一个 chunk 可能不均匀
                const auto num_bytes_in_chunk = static_cast<int>(min(num_bucket_bytes - chunk_offset, static_cast<int64_t>(num_chunk_bytes)));
                // 当前 chunk 实际字节数（桶尾可能不足一个完整 chunk）
                const auto num_bytes_per_qp = math::align(math::ceil_div(num_bytes_in_chunk, kNumQPsPerChunk), kNumRDMAAlignmentBytes);
                // 每个 QP 分到的字节数（向上取整并对齐到 RDMA 传输对齐要求）
                const auto qp_offset = min(warp_idx * num_bytes_per_qp, num_bytes_in_chunk);
                // 本 QP 在 chunk 内的起始偏移（越界时钳制到 chunk 末尾）
                const auto num_bytes = min(num_bytes_per_qp, num_bytes_in_chunk - qp_offset);
                // 本 QP 实际发送的字节数（不超过 chunk 剩余量）

                // NOTES: always signal even if this QP has nothing to send, so that the CE waits stay uniform
                // 说明：即使该 QP 无数据可发也要发信号，使拷贝引擎（CE）的等待行为在各 QP 间保持一致
                const ncclGin_WeakVASignalInc remote_action {
                    .signalWindow = window,
                    // 远端信号落在对端的对称内存窗口上
                    .signalOffset = gin.get_sym_offset(signals.tails[warp_idx] + chunk_idx),
                    // 信号偏移指向本 QP 对应 tail 数组中该 chunk 的槽位（自增语义）
                };
                if (num_bytes > 0) {
                    // 本 QP 在该 chunk 中确有数据
                    const auto ptr = math::advance_ptr(shard_ptr, chunk_offset + qp_offset);
                    // 计算本 QP 负责的数据段起始地址
                    gin.put<ncclTeamTagRail>(ptr, ptr, num_bytes, sm_idx, 0, remote_action);
                    // RDMA put：把数据写到目标节点（sm_idx 对应的远端 rank），写完后附带自增 tail 信号
                } else {
                    gin.signal<ncclTeamTagRail>(sm_idx, remote_action);
                    // 无数据时仅发送独立的信号操作，保证 tail 计数仍然自增
                }
            }
        }
    }
}

}  // namespace deep_ep::bucket
// 退出命名空间
