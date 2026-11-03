#pragma once
// 头文件保护：保证本文件只被包含一次

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量（kNumMaxRanks、kNumMaxSignalBytes）与静态断言
#include <deep_ep/common/exception.cuh>
// 引入统一断言宏（EP_UNIFIED_ASSERT）
#include <deep_ep/layout/common/barrier.cuh>
// 引入公共屏障信号（BarrierSignals）

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) EPSignals {
    // EP（专家并行）全局信号区：按 RDMA 对齐；存放同步计数器与统计值
    static constexpr int kNumMaxExperts = 2048;
    // 支持的最大专家总数
    static constexpr int kNumMaxExpertsPerRank = 512;
    // 每 rank 支持的最大专家数
    static constexpr int kNumMaxChannels = 1280;
    // 支持的最大通信 channel 数

    BarrierSignals barrier_signals;
    // NVL 屏障信号（互斥/到达计数等）
    int64_t notify_reduction[kNumMaxRanks + kNumMaxExperts];
    // 归约通知计数：前 kNumMaxRanks 项按 rank 记录，其后按专家记录
    int64_t scaleup_rank_expert_count[2][kNumMaxRanks + kNumMaxExperts];
    // 扩容（scale-up）方向计数：双缓冲（0/1 两套，发送/接收交替使用），
    // 前 kNumMaxRanks 项为各 rank 的 token 计数，其后为各专家的 token 计数
    int scaleup_atomic_sender_count[kNumMaxRanks];
    // 扩容方向各 rank 的原子发送方计数
    int scaleout_rank_expert_count[2][kNumMaxRanks + kNumMaxExperts];
    // 扩缩（scale-out）方向计数：双缓冲，前 kNumMaxRanks 项按 rank，其后按 (scale-out rank, 专家) 展开
    int64_t scaleout_channel_signaled_tail[kNumMaxChannels][kNumMaxRanks];
    // 扩缩方向：每个 channel 对每个 scale-out rank 已发信号（确认）的尾指针
    int channel_scaleup_tail[kNumMaxChannels][kNumMaxRanks];
    // 扩容方向：每个 channel 对每个 rank 已消费的尾指针
    int lb_chunk_idx_counter;
    // 负载均衡（EPLB）权重复制的全局块认领计数器
};
EP_STATIC_ASSERT(sizeof(EPSignals) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：信号区总大小不得超过预留的信号字节上限

struct EPWorkspaceLayout {
    // EP 工作区视图：包装 EPSignals 并提供按维度取址的便捷接口
    EPSignals& signals;
    // 底层信号区引用

    int num_ranks;
    // rank 总数 = scale-out rank 数 × scale-up rank 数
    int num_scaleout_ranks, num_scaleup_ranks;
    // 跨节点（scale-out）维度 rank 数与节点内（scale-up）维度 rank 数
    int num_experts, num_experts_per_rank;
    // 专家总数与每 rank 专家数

    __forceinline__ __device__ __host__
    EPWorkspaceLayout(void* base,
                      const int& num_scaleout_ranks,
                      const int& num_scaleup_ranks,
                      const int& num_experts):
        signals(*static_cast<EPSignals*>(base)),
        // 把基地址解释为信号区并绑定引用
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),
        // 计算总 rank 数
        num_scaleout_ranks(num_scaleout_ranks),
        // 保存 scale-out rank 数
        num_scaleup_ranks(num_scaleup_ranks),
        // 保存 scale-up rank 数
        num_experts(num_experts),
        // 保存专家总数
        num_experts_per_rank(num_experts / num_ranks) {
        // 计算每 rank 专家数
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        // 断言：专家数必须能被 rank 数整除
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        // 断言：rank 数不超上限
        EP_UNIFIED_ASSERT(num_experts <= EPSignals::kNumMaxExperts);
        // 断言：专家总数不超上限
        EP_UNIFIED_ASSERT(num_experts_per_rank <= EPSignals::kNumMaxExpertsPerRank);
        // 断言：每 rank 专家数不超上限
    }

    __forceinline__ __device__ __host__ static constexpr int64_t get_num_bytes() {
        // 工作区总字节数：即信号区结构体大小
        return sizeof(EPSignals);
    }

    __forceinline__ __device__ __host__ BarrierSignals& get_barrier_signals() const {
        // 取公共屏障信号
        return signals.barrier_signals;
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        // 取归约通知计数数组首地址
        return signals.notify_reduction;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        // 取扩容方向的 rank/expert 计数数组（双缓冲之一）：
        // 发送方用另一套（not 发送缓冲），避免与接收方的清零/读写互相踩踏
        return signals.scaleup_rank_expert_count[not kIsSendBuffer];
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {
        // 取扩容方向"按 rank"计数段首地址（数组前 kNumMaxRanks 项）
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {
        // 取扩容方向"按专家"计数段首地址：紧跟 rank 段之后
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + num_scaleup_ranks;
    }

    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {
        // 取扩容方向各 rank 的原子发送方计数数组首地址
        return signals.scaleup_atomic_sender_count;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {
        // 取扩缩方向的 rank/expert 计数数组（双缓冲之一，规则同扩容方向）
        return signals.scaleout_rank_expert_count[not kIsSendBuffer];
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {
        // 取扩缩方向指定 (scale-out rank, scale-up rank) 的 rank 计数地址：
        // 布局为 [rank 0: 每个 scale-up rank 一项][rank 1: ...]...
        return get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() +
               scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        // 取扩缩方向指定 (scale-out rank, 专家) 的专家计数地址：
        // 布局：先跳过 rank 段（共 num_ranks = scale-out 数 × scale-up 数 项），
        //       再按 [scale-out rank × 每 rank 专家数 + 专家编号] 偏移
        return get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() + num_ranks +
               scaleout_rank_idx * num_scaleup_ranks * num_experts_per_rank + expert_idx;
    }

    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr(
        const int& channel_idx, const int& scaleout_rank_idx) const {
        // 取扩缩方向 (channel, scale-out rank) 的已确认尾指针地址
        return &signals.scaleout_channel_signaled_tail[channel_idx][scaleout_rank_idx];
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        // 取扩容方向 (channel, scale-up rank) 的消费尾指针地址
        return &signals.channel_scaleup_tail[channel_idx][scaleup_rank_idx];
    }

    __forceinline__ __device__ __host__ int* get_lb_chunk_idx_counter_ptr() const {
        // 取 EPLB 权重复制的全局块认领计数器地址
        return &signals.lb_chunk_idx_counter;
    }
};

}  // namespace deep_ep::layout
// 结束 layout 命名空间
