#pragma once
// 头文件保护：避免重复包含

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
// 引入 GPU 栅栏协议、指针运算、PTX（含 multimem 指令封装）与通用加载/存储迭代工具

namespace deep_ep::bucket {
// DeepEP bucket 集合通信实现命名空间

// 模板参数：kNumRanks 机内 NVLink rank 数；kNumSMs 使用的 SM 数；kNumThreads 每 CTA 线程数；
// kNumTimeoutCycles 超时周期；kWithScale 是否对归约结果乘以缩放因子；kContextIdx 上下文编号
template <int kNumRanks, int kNumSMs, int kNumThreads,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx>
__global__ void __launch_bounds__(kNumThreads, 1)
// 限制每 SM 仅 1 个 CTA 驻留
nvlink_reduce_scatter(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                      void* workspace_ptr,
                      void* buffer, void* buffer_multimem,
                      const float scale,
                      const int num_buckets,
                      const __grid_constant__ comm::BucketList buckets) {
    // 基于 NVLink 多播（NVLS/multimem）的机内 reduce-scatter：
    // 每个 rank 用 multimem.ld_reduce 对多播窗口上的全体输入做硬件求和，并把本 rank 拥有的分片写到普通缓冲
    // 参数：nccl_dev_comm 设备端通信对象；window 对称内存窗口；buffer 普通输出缓冲；
    // buffer_multimem 注册为多播地址的输入缓冲；scale 缩放因子；num_buckets 桶数量；buckets 桶描述列表
    // Indices
    // 计算各类索引
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 CTA（SM）编号
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 块内线程编号
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;
    // 全局线程编号，用于把数据元素分摊到整个网格
    const auto rank_idx = ncclTeamLsa(nccl_dev_comm).rank;
    // 本进程在 LSA（机内 NVLink 团队）中的 rank 编号

    // Barrier to ensure data is arrived
    // 入口栅栏：确保所有 rank 的输入数据都已就绪并可见
    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    // 取得 bucket 工作区引用
    auto& barrier_signals = workspace.signals[kContextIdx].barrier;
    // 本上下文的 barrier 信号数组
    const comm::NCCLGin gin(nccl_dev_comm, window, 0);
    // 构造 GIN 封装（使用 QP 0，仅用于栅栏信号）
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, 0, rank_idx, sm_idx, thread_idx);
    // 纯 NVLink 域（RDMA 层数为 1）的 GPU 栅栏，等待全部机内 rank 到达

    // Reduce scatter
    // 归约散发：每个 rank 归约出属于自己的那一份分片
    for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
        // 逐桶处理
        const auto& bucket = buckets[bucket_idx];
        // 取出桶描述（偏移与总字节数）
        const auto src_offset = bucket.offset + rank_idx * bucket.num_bytes;
        // 本 rank 分片（目的槽位）在缓冲中的偏移
        const auto src = math::advance_ptr<const float4>(buffer_multimem, src_offset);
        // 多播窗口上的源地址：multimem 读会聚合所有 rank 同位置的数据
        const auto dst = math::advance_ptr<float4>(buffer, src_offset);
        // 普通输出缓冲上本 rank 分片的目的地址

        // TODO: merge iterations between buckets
        // 待办：合并相邻桶之间的迭代以提高效率
        utils::iterate_ld_st<kNumSMs, kNumThreads, 8>(
            global_thread_idx,
            bucket.get_num_elems<float4>(),
            // 迭代元素总数（以 float4 为单位）
            src, dst,
            [](const float4* src_ptr, const int& lhs = 1, const int& rhs = 0) { return ptx::multimem_ld_reduce_add_f32_with_gt_pred(src_ptr, lhs, rhs); },
            // 加载阶段：multimem.ld_reduce.add 对所有 rank 的同位置 float4 做硬件求和归约
            [](float4* dst_ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) { ptx::st_with_gt_pred(dst_ptr, value, lhs, rhs); },
            // 存储阶段：普通 store 把归约结果写入本 rank 自己的分片（reduce-scatter 的语义）
            [&](const float4& v) { return kWithScale ? ptx::fmul4(v, scale) : v; });
            // 可选的后处理：按需对归约结果逐元素乘以 scale
    }

    // Ensure no overlapping writes
    // 出口栅栏：确保各 rank 的写入互不重叠、全部完成后才能进入下一阶段
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, 0, rank_idx, sm_idx, thread_idx);
    // 再次执行 GPU 栅栏（带 flush 语义），保证所有写入全局可见
}

}  // namespace deep_ep::bucket
// 退出命名空间
