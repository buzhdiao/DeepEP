#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cstdint>
// 引入定宽整型（uint64_t 等）

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量（kNumMaxNVLRanks、kNumMaxSignalBytes）与静态断言
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) CopyEngineSignals {
    // 拷贝引擎（copy engine）的信号区：按 RDMA 对齐
    uint64_t base[kNumMaxNVLRanks];
    // 各 NVL rank 的写指针基址/进度：记录对端已拷贝到的位置，用于断点续拷与顺序控制
};
EP_STATIC_ASSERT(sizeof(CopyEngineSignals) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：信号区总大小不得超过预留的信号字节上限

}  // namespace deep_ep::layout
// 结束 layout 命名空间
