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

struct alignas(kNumRDMAAlignmentBytes) AllReduceSignals {
    // AllReduce 子系统的信号区：按 RDMA 对齐
    static constexpr int kNumTotalSlots = 16384;
    // 槽位总数（每个在途的 AllReduce 分片占一个槽位）
    int64_t counters[kNumTotalSlots];
    // 完成计数数组：各 rank 对应槽位原子累加，达到参与数即表示该分片归约完成
};
EP_STATIC_ASSERT(sizeof(AllReduceSignals) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：信号区总大小不得超过预留的信号字节上限

}  // namespace deep_ep::layout
// 结束 layout 命名空间
