#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cstdint>
// 引入定宽整型（uint64_t 等）

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量（kNumMaxSignalBytes）与静态断言
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) AllGatherSignals {
    // AllGather 子系统的信号区：按 RDMA 对齐
    static constexpr int kNumQPsPerChunk = 8;
    // 每个 chunk 使用的 QP（队列对）数：多 QP 并行传输同一 chunk
    static constexpr int kNumMaxChunks = 8192;
    // 支持的最大 chunk 数
    uint64_t tails[kNumQPsPerChunk][kNumMaxChunks];
    // 完成信号尾指针：[QP 编号][chunk 编号]，各 QP 独立记录该 chunk 已推送的字节进度

};
EP_STATIC_ASSERT(sizeof(AllGatherSignals) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：信号区总大小不得超过预留的信号字节上限

}  // namespace deep_ep::layout
// 结束 layout 命名空间
