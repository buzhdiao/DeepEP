#pragma once
// 头文件保护：确保本头文件在单个编译单元中只被包含一次

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/comm/handle.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
// 引入 GPU 端 barrier 协议、NCCL 通信句柄封装、指针运算辅助、PTX 内建函数以及 bucket 工作区布局

namespace deep_ep::bucket {
// 进入 DeepEP bucket（桶式分桶集合通信）实现的命名空间

// 模板参数：kNumRanks 参与通信的 rank 总数；kNumSMs 每个网卡代理占用的 SM 数；
// kNumTimeoutCycles barrier 超时周期数；kContextIdx 使用的上下文编号；其余为派生的线程/warp 配置
template <int kNumRanks, int kNumSMs, int64_t kNumTimeoutCycles, int kContextIdx,
          int kNumThreads = 512,
          int kNumWarps = kNumThreads / 32,
          int kNumGlobalWarps = kNumSMs * kNumWarps>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限定每个 SM 最多驻留 1 个 CTA，保证寄存器/共享资源用量受控
rdma_all_gather(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                void* workspace_ptr, void* buffer,
                const int num_buckets,
                const __grid_constant__ comm::BucketList buckets) {
    // 参数：NCCL 设备端通信对象与对称内存窗口；workspace_ptr 指向 GPU 工作区；
    // buffer 为本地输入缓冲（含各 rank 分片布局）；num_buckets 为桶数量；buckets 为常量内存中的桶描述列表
    // Indices
    // 以下计算各类索引
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 CTA 所在的 SM（block）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 当前块内线程编号
    const auto lane_idx = ptx::get_lane_idx();
    // 当前线程在 warp 内的 lane 编号（0~31）
    const auto warp_idx = ptx::get_warp_idx();
    // 当前块内 warp 编号
    const auto global_warp_idx = sm_idx * kNumWarps + warp_idx;
    // 全局 warp 编号：跨所有 SM 统一编号，用于把目标 rank 分摊到各 warp
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    // 本进程在"铁路式（rail）"RDMA 团队中的 rank 编号

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    // 取得 bucket 工作区引用（存放 barrier 信号等跨 rank 同步状态）
    auto& barrier_signals = workspace.signals[kContextIdx].barrier;
    // 取出本上下文对应的 barrier 信号数组

    // Entry barrier: all ranks have their local shard in place.
    // 入口栅栏：确保所有 rank 的本地分片数据均已就绪，才能开始远端读取/写入
    const comm::NCCLGin gin(nccl_dev_comm, window);
    // 构造 GIN（GPU Initiated Networking）发起 RDMA 操作的封装，绑定通信对象与窗口
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
    // 执行跨 rank（NVLink 域内 1 级 + RDMA 域 kNumRanks 级）的 GPU 栅栏，等待全体到达

    for (int dst_rank_idx = global_warp_idx; dst_rank_idx < kNumRanks; dst_rank_idx += kNumGlobalWarps) {
        // 每个 global warp 负责一个目标 rank，跨步循环覆盖所有远端 rank
        if (dst_rank_idx == rank_idx)
            continue;
        // 本 rank 的数据已在本地，无需自我拷贝

        for (int bucket_idx = lane_idx; bucket_idx < num_buckets; bucket_idx += 32) {
            // warp 内 32 个 lane 分摊桶，跨步遍历本 rank 的所有桶
            const auto& [offset, num_bytes] = buckets[bucket_idx];
            // 解构桶描述：offset 为该桶在 buffer 内的字节偏移，num_bytes 为每个 rank 分片的字节数
            const auto ptr = math::advance_ptr(buffer, offset + rank_idx * num_bytes);
            // 计算本 rank 在该桶中分片的起始地址
            gin.put<ncclTeamTagRail>(ptr, ptr, num_bytes, dst_rank_idx);
            // 通过 RDMA put 把本 rank 的分片直接写到目标 rank 缓冲中的对应槽位（源/目的偏移相同，all-gather 摆放）
        }
        __syncwarp();
        // 同步 warp 内各 lane，确保本目标 rank 的所有 put 发出后再切换到下一个目标
    }
}

}  // namespace deep_ep::bucket
// 退出命名空间
