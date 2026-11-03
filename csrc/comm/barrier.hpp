#pragma once

// 引入通信流的获取与流间等待工具
#include "stream.hpp"
// 引入 NCCL 通信上下文与内核启动的公共 API
#include "../kernels/comm/api.hpp"
// 引入屏障内核的启动封装
#include "../kernels/comm/barrier.hpp"

// 通信相关工具所在的命名空间
namespace deep_ep::comm {

// 对给定通信组执行一次组内屏障同步（所有 rank 到齐后才能继续）
// 参数含义依次为：通信组上下文、屏障信号区指针、执行内核的流、GPU 超时周期数、
// 是否带 CPU 侧同步、是否使用顺序（sequential）屏障、是否刷新写操作、是否等待通信流
static void barrier(const Context& group,
                    void* signals,
                    const at::cuda::CUDAStream& stream,
                    const int64_t& num_gpu_timeout_cycles,
                    const bool& with_cpu_sync,
                    const bool& sequential,
                    const bool& flush_stores = true,
                    const bool& wait_comm_stream = false) {
    // Sync
    // 前置同步：确保屏障启动前的状态一致
    if (with_cpu_sync) {
        // 需要时直接同步整个 GPU 设备，保证所有工作已完成
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    } else if (wait_comm_stream) {
        // 否则让调用方传入的流等待通信流上的已有工作
        stream_wait(stream, get_comm_stream());
    }

    // Launch
    // 启动屏障内核：通过对称内存信号区与其他 rank 交换完成标记
    launch_barrier(
        // NCCL 设备端通信句柄
        group.dev_comm, group.window,
        // 屏障信号所在的共享/对称内存区
        signals,
        // 本进程在 scale-out（跨机）与 scale-up（机内）两个维度上的编号
        group.scaleout_rank_idx, group.scaleup_rank_idx,
        // 两个维度上的 rank 总数
        group.num_scaleout_ranks, group.num_scaleup_ranks,
        // GPU 端等待的超时周期数
        num_gpu_timeout_cycles,
        // scale-up 方向是否为 NVLink 直连
        group.is_scaleup_nvlink,
        // 是否在屏障前刷新（flush）各处的写操作
        flush_stores, sequential,
        // 屏障内核所运行的 CUDA 流
        stream);

    // Sync
    // 后置同步：确保屏障完成后调用方的后续工作基于同步后的状态
    if (with_cpu_sync) {
        // 需要时再次同步整个 GPU 设备，等待屏障内核结束
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    } else if (wait_comm_stream) {
        // To work with APIs with previous events
        // 让通信流反过来等待屏障流，便于与带 previous_event 的 API 配合使用
        // 为了与带有 previous_event 的 API 配合使用
        stream_wait(get_comm_stream(), stream);
    }
}

}  // namespace deep_ep::comm
