#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cstdint>
// 引入定宽整型（int64_t 等）

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量（kNumMaxSignalBytes）与静态断言
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) ReduceScatterSignals {
    // ReduceScatter 子系统的信号区：按 RDMA 对齐；每槽位一对头/尾指针
    static constexpr int kNumTotalSlots = 8192;
    // 槽位总数（各 rank 之间按 kNumSlotsPerRank 划分）
    int64_t heads[kNumTotalSlots];
    // 头指针数组：消费者已就绪/可读的进度
    int64_t tails[kNumTotalSlots];
    // 尾指针数组：生产者已写入/已发送的进度

    template <int kNumSlotsPerRank>
    __forceinline__ __device__ int64_t* get_head_ptr(const int& rank_idx, const int& slot_idx) {
        // 取指定 (rank, 槽位) 的头指针地址：偏移 = rank 编号 × 每 rank 槽位数 + 槽位编号
        return heads + rank_idx * kNumSlotsPerRank + slot_idx;
    }

    template <int kNumSlotsPerRank>
    __forceinline__ __device__ int64_t* get_tail_ptr(const int& rank_idx, const int& slot_idx) {
        // 取指定 (rank, 槽位) 的尾指针地址：偏移 = rank 编号 × 每 rank 槽位数 + 槽位编号
        return tails + rank_idx * kNumSlotsPerRank + slot_idx;
    }
};
EP_STATIC_ASSERT(sizeof(ReduceScatterSignals) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：信号区总大小不得超过预留的信号字节上限

}  // namespace deep_ep::layout
// 结束 layout 命名空间
