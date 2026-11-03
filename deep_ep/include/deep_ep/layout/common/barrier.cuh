#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cstdint>
// 引入定宽整型
#include <cuda_runtime.h>
// 引入 CUDA 运行时基础定义

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量（kNumMaxSignalBytes）与静态断言
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) BarrierSignals {
    // 各子系统共用的屏障信号：按 RDMA 对齐；实现一个两相位（2-phase）的 NVL 屏障
    unsigned long long nvl_barrier_counter;
    // NVL 屏障到达计数器：跨节点/节点内 rank 原子累加，用于检测全体到达
    int nvl_barrier_signals[2];
    // 两相位各自的信号标志：交替使用避免上一相位残留值影响下一相位

    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() {
        // 取屏障到达计数器的地址
        return &nvl_barrier_counter;
    }

    __forceinline__ __device__ __host__ int* get_nvl_barrier_signal_ptr(const int& phase) {
        // 取指定相位（0 或 1）的信号标志地址
        return nvl_barrier_signals + phase;
    }
};
EP_STATIC_ASSERT(sizeof(BarrierSignals) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：信号区总大小不得超过预留的信号字节上限

}  // namespace deep_ep::layout
// 结束 layout 命名空间
