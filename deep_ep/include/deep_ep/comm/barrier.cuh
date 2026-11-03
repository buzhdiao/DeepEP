#pragma once  // 头文件保护，防止被重复包含

#include <cooperative_groups.h>  // 协作组：用于 grid 级同步
#include <nccl.h>  // NCCL 主头文件
#include <nccl_device.h>  // NCCL 设备端 API

#include <deep_ep/comm/handle.cuh>  // NCCLGin 通信封装
#include <deep_ep/common/ptx.cuh>  // PTX 指令封装
#include <deep_ep/layout/common/barrier.cuh>  // 屏障信号布局定义（BarrierSignals）

// deep_ep::comm 命名空间：设备端全局屏障实现
namespace deep_ep::comm {

static constexpr int64_t kNumOneSecCycles = 2000000000;  // An approximation of the GPU clock at 2000 MHz  // 1 秒对应的时钟周期数（按 2GHz 估算）

// Some reserved tags  // 预留的 barrier 标签（用于超时诊断信息区分调用来源）
static constexpr int kDeviceBarrierTag = 0;  // 设备端发起的屏障
static constexpr int kKernelBarrierTag = 1;  // 独立 barrier kernel 发起的屏障
static constexpr int kDispatchTag0 = 2;  // dispatch 阶段通道 0
static constexpr int kDispatchTag1 = 3;  // dispatch 阶段通道 1
static constexpr int kCombineTag0 = 4;  // combine 阶段通道 0
static constexpr int kCombineTag1 = 5;  // combine 阶段通道 1
static constexpr int kHybridDispatchTag0 = 6;  // 混合（scaleup+scaleout）dispatch 通道 0
static constexpr int kHybridDispatchTag1 = 7;  // 混合 dispatch 通道 1
static constexpr int kHybridCombineTag0 = 8;  // 混合 combine 通道 0
static constexpr int kHybridCombineTag1 = 9;  // 混合 combine 通道 1

// Some reserved count  // 预留的 QP 计数特殊值
static constexpr int kFlushAllAllocatedQPs = -1;  // 表示刷新/使用所有已分配的 QP

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const bool& condition, const func_t& func,
                                              int64_t start_clock = 0) {  // 带超时检测的 while 循环：条件满足时反复调用 func
    // User may share a start clock for multiple waits  // 调用者可传入共享的起始时钟，用于多次等待统一计时
    if (start_clock == 0)  // 未提供起始时钟
        start_clock = clock64();  // 取当前 GPU 时钟周期计数

    while (condition) {  // 主循环：条件成立则继续
        const bool timeout = clock64() - start_clock >= kNumTimeoutCycles;  // 判断是否已超过超时周期数
        if (func(timeout))  // 调用回调；回调返回 true 表示成功，可提前退出
            break;  // 正常退出循环

        if (timeout) {  // 超时且回调未宣布成功
            // Wait another 1 second to let all threads print information and trap  // 再等 1 秒，让所有线程完成信息打印
            start_clock = clock64();  // 重置起始时钟
            while (clock64() - start_clock < kNumOneSecCycles) {}  // 忙等约 1 秒
            ptx::trap();  // 触发致命陷阱，终止整个 kernel
        }
    }
}

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const func_t& func, const int64_t& start_clock = 0) {  // 恒真条件的重载（自旋直到回调成功或超时）
    timeout_while<kNumTimeoutCycles, func_t>(true, func, start_clock);  // 复用上面的通用实现，条件恒为 true
}

template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(  // 计算某 SM 某通道应使用的 QP 编号与资源共享模式
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp = false) {  // SM 编号、SM 内通道编号、是否为通知 warp
    constexpr auto kSharingCTA = NCCL_GIN_RESOURCE_SHARING_CTA;  // 按 CTA（SM）共享 Gin 资源
    constexpr auto kSharingGrid = kNumSMs == 1 ? NCCL_GIN_RESOURCE_SHARING_CTA : NCCL_GIN_RESOURCE_SHARING_GPU;  // 单 SM 时按 CTA 共享，否则按整个 GPU 共享

    // Only one QP  // 只有一个 QP 的退化情况
    if constexpr (kNumQPs == 1)  // 编译期判断
        return {0, kSharingGrid};  // 所有通道共用 QP 0 与网格级共享

    // The notify warp always use 1 SM and 1 QP  // 通知 warp 固定使用 QP 0 且独占
    if (is_notify_warp)  // 运行时判断
        return {0, kSharingCTA};  // 使用 QP 0、CTA 级共享

    // Data channels  // 数据通道的 QP 分配
    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);  // 有通知 warp 时数据通道从 QP 1 开始，否则从 0 开始
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;  // 可供数据通道使用的 QP 数
    if constexpr (kNumSMs <= kNumAvailableQPs) {  // QP 数量充足：每个 SM 独占若干 QP
        // A single SM uses an entire QP
        // e.g., 3 SMs and 10 QPs
        // SM 0: 0 3 6 9
        // SM 1: 1 4 7
        // SM 2: 2 5 8  // 保留原注释：轮转分配示例
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));  // 本 SM 分得的 QP 数（前 kNumAvailableQPs%kNumSMs 个 SM 多分一个）
        return {kQPStartIdx + sm_idx + (channel_in_sm_idx % num_qps_in_sm) * kNumSMs, kSharingCTA};  // 按列轮转取 QP 编号，CTA 级共享
    } else {  // QP 不足：所有 SM 轮流共享全部 QP
        // All SMs share all QPs  // 全体 SM 共享所有 QP
        const auto global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;  // 全局通道编号
        return {kQPStartIdx + (global_channel_idx % kNumAvailableQPs), kSharingGrid};  // 对可用 QP 取模分配，网格级共享
    }
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag>
__forceinline__ __device__ void nvlink_barrier_wo_local_sync(  // NVLink（节点内）屏障：仅用 1 个 SM，不做本地 grid 同步
    const NCCLGin& gin,  // Gin 通信封装
    layout::BarrierSignals& signals,  // 屏障信号结构（计数器与信号）
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {  // rank、SM、线程编号
    // This barrier only uses 1 SM  // 本屏障只由 SM 0 执行
    if (kNumSMs > 1 and sm_idx > 0)  // 多 SM 且不是 SM 0
        return;  // 直接返回

    // Read the current barrier phase first  // 先读取当前屏障的相位状态
    const auto counter_ptr = signals.get_nvl_barrier_counter_ptr();  // NVLink 屏障状态计数器指针
    const int status = static_cast<int>((*counter_ptr) & 3);  // 取低 2 位作为状态：位 0 为相位，位 1 为符号
    const int phase = status & 1, sign = status >> 1;  // phase：当前相位（0/1）；sign：本轮到达方向的符号
    const auto signal_ptr = signals.get_nvl_barrier_signal_ptr(phase);  // 当前相位对应的信号数组指针

    for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {  // 每线程负责一部分远端 rank
        const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(signal_ptr, i);  // 取远端 rank 对应的信号对称指针
        ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);  // 向远端信号做 release 原子加（符号决定加/减，用于相位区分）
    }
    __syncthreads();  // 等待 block 内所有线程完成远端加法

    // NOTES: we need `2^64 / 1e6 / 3600 / 24 / 365 = 571000` years to make the counter overflow (1 barrier per us)
    // 注：按每微秒一次屏障计算，2^64 计数器溢出需约 57 万年，无需担心回绕
    // Add the phase counter  // 递增相位计数器
    if (thread_idx == 0)  // 仅 0 号线程
        atomicAdd(counter_ptr, 1);  // 原子递增状态计数器（进入下一相位）
    __syncwarp();  // warp 内同步，保证计数器更新可见

    // Check timeout  // 等待所有远端信号到达目标值（带超时）
    const auto target = sign ? 0 : kNumRanks;  // 目标值：符号为 1 时期望归零，否则期望累加到 kNumRanks
    timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=](const bool& is_last_check) {  // 仅 0 号线程执行等待循环
        const auto signal = ptx::ld_acquire_sys<int>(signal_ptr);  // 以系统作用域 acquire 读取当前信号值
        if (signal == target)  // 已达目标
            return true;  // 屏障完成

        if (is_last_check) {  // 最后一次检查（即将超时）
            printf("DeepEP NVLink barrier timeout, tag: %d, nvl: %d, thread: %d, "
                   "status: %d, signal: %d, phase: %d, target: %d, counter: %llu\n",  // 打印诊断信息
                   kTag, rank_idx, thread_idx, status, signal, phase, target,
                   *counter_ptr);
        }
        return false;  // 继续等待
    });
    __syncwarp();  // warp 内同步
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles,
          typename team_t, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true,
          int kNumWarps = kNumThreads / 32>
__forceinline__ __device__ void gin_barrier_wo_local_sync(  // 基于 Gin 信号的屏障：不做本地同步（grid 同步由调用者负责）
    const NCCLGin& gin,  // Gin 通信封装
    const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 横向（跨节点）与纵向（节点内）rank 编号
    const int& sm_idx, const int& thread_idx) {  // SM 与线程编号
    const auto& nccl_dev_comm = gin.nccl_dev_comm;  // NCCL 设备端通信对象
    const auto global_warp_idx = sm_idx * kNumWarps + (thread_idx / 32);  // 全局 warp 编号
    const int& rank_idx = (std::is_same_v<team_t, ncclTeamTagWorld>) ? scaleup_rank_idx : scaleout_rank_idx;  // world 团队用节点内编号，rail 团队用跨节点编号
    const int num_qps = kNumQPs == kFlushAllAllocatedQPs ? nccl_dev_comm.ginContextCount : kNumQPs;  // 特殊值 -1 表示使用全部已分配 QP

    // Flush all QPs by all SMs (only needed for release semantics)  // 由所有 SM 分摊刷新全部 QP（保证此前写入发出，仅在需要 release 语义时）
    if constexpr (kFlushStores) {  // 编译期开关
        for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {  // 全局 warp 轮流分摊各 QP
            ncclGin(nccl_dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA).flush(ncclCoopWarp());  // 以 warp 为单位 flush 每个 QP
        }

        // World-team barriers may mix direct NVLink writes with GIN signals, so a system-scope
        // threadfence is required to make the NVLink writes visible before publishing the barrier.
        // 注：world 团队屏障可能混用 NVLink 直写与 Gin 信号，需要系统级栅栏确保 NVLink 写先于屏障发布可见
        if constexpr (std::is_same_v<team_t, ncclTeamTagWorld>)  // 仅 world 团队需要
            ptx::fence_acq_rel_sys();  // 系统作用域 acquire-release 栅栏

        // NOTES: we can not use `kNumSMs` to judge, as maybe only part of the SMs will call this function.
        // 注：不能用模板参数 kNumSMs 判断是否需要 grid 同步，因为可能只有部分 SM 调用本函数
        (gridDim.x > 1) ? cooperative_groups::this_grid().sync() : __syncthreads();  // 多 block 时 grid 同步，单 block 时 block 内同步
    }

    if (sm_idx == 0) {  // 仅 SM 0 负责屏障信令
        // Use QP 0 to do barrier: signal slot `rank_idx` on every rank, then wait for every rank's slot  // 用 QP 0：向每个 rank 的本 rank 槽位发信号，再等待所有 rank 发来的信号
        const NCCLGin barrier_gin(nccl_dev_comm, gin.nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);  // 以 QP 0、CTA 共享模式构造屏障专用 Gin
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads)  // 每线程负责若干远端 rank
            barrier_gin.signal<team_t>(i, ncclGin_SignalInc{static_cast<ncclGinSignal_t>(rank_idx)});  // 向 rank i 的第 rank_idx 号信号槽自增
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads)  // 再逐 rank 等待
            barrier_gin.wait_signal<team_t, kNumTimeoutCycles>(i, rank_idx);  // 等待 rank i 发到本 rank 第 rank_idx 槽的信号（带超时）
    }
}

template <bool kIsScaleupNVLink, int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag, bool kFlushStores = true>
__forceinline__ __device__ void scaleup_barrier_wo_local_sync(  // 横向扩展（节点内）屏障：按链路类型分派
    const NCCLGin& gin,  // Gin 通信封装
    layout::BarrierSignals& signals,  // 屏障信号结构
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {  // rank、SM、线程编号
    if constexpr (kIsScaleupNVLink) {  // 节点内走 NVLink 直连
        nvlink_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumTimeoutCycles, kTag>(  // 使用 NVLink 屏障
            gin, signals, rank_idx, sm_idx, thread_idx);
    } else {  // 否则走 Gin（RDMA）屏障，world 团队视图
        gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagWorld, kTag, kFlushStores>(  // 使用 Gin 屏障
            gin, 1, rank_idx, sm_idx, thread_idx);  // 传入占位的 scaleout 编号 1
    }
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true>
__forceinline__ __device__ void scaleout_barrier_wo_local_sync(  // 纵向扩展（跨节点）屏障：固定使用 rail 团队的 Gin 屏障
    const NCCLGin& gin,  // Gin 通信封装
    const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 跨节点与节点内 rank 编号
    const int& sm_idx, const int& thread_idx) {  // SM 与线程编号
    gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagRail, kTag, kFlushStores>(  // 以 rail 团队视图调用 Gin 屏障
        gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
}

template <bool kIsScaleupNVLink,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true>
__forceinline__ __device__ void gpu_barrier(const NCCLGin& gin,  // 全局屏障：可同时/分别完成节点内与跨节点屏障
                                            layout::BarrierSignals& signals,  // 屏障信号结构
                                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 跨节点与节点内 rank 编号
                                            const int& sm_idx, const int& thread_idx,  // SM 与线程编号
                                            bool do_scaleout = true, bool do_scaleup = true) {  // 是否执行跨节点/节点内屏障
    // A general TMA store wait to prevent proxy memory issues  // 通用 TMA 存储等待，防止异步代理的内存可见性问题
    if constexpr (kFlushStores) {  // 编译期开关
        ptx::tma_store_commit();  // 提交当前批量存储组
        ptx::tma_store_wait();  // 等待全部 TMA 存储完成
        __syncwarp();  // warp 内同步
    }

    // All the SMs should wait  // 所有 SM 在屏障开始前汇合
    if constexpr (kSyncAtStart) {  // 编译期开关
        cooperative_groups::this_grid().sync();  // grid 级同步
    } else {  // 不同步则必须无存储需要刷新
        EP_STATIC_ASSERT(not kFlushStores, "No data to be flushed");  // 静态检查：起始不同步时不能要求刷新
    }

    do_scaleout &= kNumScaleoutRanks > 1;  // 跨节点只有 1 个 rank 时无需屏障
    do_scaleup &= kNumScaleupRanks > 1;  // 节点内只有 1 个 rank 时无需屏障
    if (do_scaleup and do_scaleout) {  // 两者都需要
        // Do scaleup and scaleout barrier in parallel  // 节点内与跨节点屏障并行执行
        EP_DEVICE_ASSERT(kNumSMs >= 2 and "At least 2 SMs for a hybrid barrier");  // 混合屏障至少需要 2 个 SM
        if (sm_idx == 0) {  // SM 0 负责节点内屏障
            // First SM do the scaleup barrier  // 第一个 SM 执行节点内屏障
            scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(  // 节点内屏障
                gin, signals, scaleup_rank_idx, sm_idx, thread_idx);

            // We need an extra grid sync, as the scaleout barrier will do a sync after flush, before the barrier.
            // 我们需要额外一次 grid 同步：跨节点屏障在 flush 之后、屏障本身之前还会做一次同步
            // NOTES: this is kind of hacky  // 保留原注释：此做法有些取巧
            if constexpr (kFlushStores)  // 仅在需要刷新时
                cooperative_groups::this_grid().sync();  // 额外 grid 同步对齐两侧进度
        } else {  // 其余 SM 负责跨节点屏障
            // The remaining SMs do the scaleout barrier  // 其余 SM 执行跨节点屏障
            scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs - 1, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(  // 注意 SM 数减 1、SM 编号减 1
                gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx - 1, thread_idx);
        }
    } else if (do_scaleup) {  // 仅节点内屏障
        // Scaleup only  // 仅执行节点内屏障
        scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(  // 节点内屏障
            gin, signals, scaleup_rank_idx, sm_idx, thread_idx);
    } else if (do_scaleout) {  // 仅跨节点屏障
        // Scaleout only  // 仅执行跨节点屏障
        scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(  // 跨节点屏障
            gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }

    // All the SMs should wait  // 屏障完成后所有 SM 再次汇合
    if constexpr (kSyncAtEnd)  // 编译期开关
        cooperative_groups::this_grid().sync();  // grid 级同步
}

}  // namespace deep_ep::comm
