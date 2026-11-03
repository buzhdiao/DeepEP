#pragma once  // 头文件保护，防止被重复包含

#include <deep_ep/comm/barrier.cuh>  // 全局屏障实现（gpu_barrier 等）
#include <deep_ep/common/compiled.cuh>  // 编译期常量（kFlushAllAllocatedQPs 等）
#include <deep_ep/layout/common/barrier.cuh>  // 屏障信号布局定义
#include <deep_ep/common/ptx.cuh>  // PTX 指令封装


// deep_ep::comm 命名空间：屏障 kernel 的实现
namespace deep_ep::comm {

template <bool kIsScaleupNVLink,  // 节点内是否走 NVLink 直连屏障
          int kNumSMs, int kNumThreads,  // 网格 SM 数与每块线程数
          int kNumScaleoutRanks, int kNumScaleupRanks,  // 跨节点与节点内 rank 总数
          int64_t kNumTimeoutCycles,  // 屏障超时周期数
          bool kFlushStores, bool kSequential>  // 是否刷新存储；是否串行执行跨节点/节点内屏障
__global__ void __launch_bounds__(kNumThreads, 1)  // 限制每块线程数并保证每 SM 至多 1 块
barrier_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* signals,  // NCCL 设备通信对象、对称窗口、屏障信号缓冲
             const int scaleout_rank_idx, const int scaleup_rank_idx) {  // 跨节点与节点内 rank 编号
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);  // 取当前 SM（block）与线程编号

    auto& barrier_signals = *static_cast<layout::BarrierSignals*>(signals);  // 将裸指针还原为屏障信号结构引用
    const auto gin = NCCLGin(nccl_dev_comm, nccl_window, 0);  // 以 QP 0 构造 Gin 通信封装
    if constexpr (kSequential) {  // 串行模式：先跨节点后节点内
        // Scaleout barrier  // 跨节点屏障
        if constexpr (kNumScaleoutRanks > 1)  // 跨节点多于 1 个 rank 才需要
            gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                        kNumSMs, kNumThreads, kFlushAllAllocatedQPs, kNumTimeoutCycles, kKernelBarrierTag,
                        kFlushStores, true, false>(  // 起始同步、结尾不同步
                gin, barrier_signals, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, true, false);  // 仅执行跨节点部分

        // Scaleup barrier, and it needs to flush the RDMA requests issued by scaleout barrier  // 节点内屏障，同时需要刷新跨节点屏障发出的 RDMA 请求
        gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                    kNumSMs, kNumThreads, kFlushAllAllocatedQPs, kNumTimeoutCycles, kKernelBarrierTag, kFlushStores, true, false>(
            gin, barrier_signals, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, false, true);  // 仅执行节点内部分
    } else {  // 混合模式：一次调用同时完成跨节点与节点内屏障
        gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                    kNumSMs, kNumThreads, kFlushAllAllocatedQPs, kNumTimeoutCycles, kKernelBarrierTag,
                    kFlushStores, true, false>(  // 默认 do_scaleout/do_scaleup 均为 true
            gin, barrier_signals, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }
}

}  // namespace deep_ep::comm
