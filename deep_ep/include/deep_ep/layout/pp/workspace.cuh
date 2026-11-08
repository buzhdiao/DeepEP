#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cstdint>
// 引入定宽整型（int64_t 等）

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量（kNumMaxSignalBytes）与静态断言
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏
#include <deep_ep/layout/common/barrier.cuh>
// 引入公共屏障信号（BarrierSignals）

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

// Pipeline-parallel send/recv state, indexed by the ring direction
// 流水线并行的收发状态：按环形方向（0/1，即上一 rank / 下一 rank）索引
struct alignas(kNumRDMAAlignmentBytes) PPSignals {
    BarrierSignals barrier;
    // 公共屏障信号
    int64_t send_count[2] = {};
    // 每个方向的累计发送计数（决定当前使用的环形槽位）
    int64_t recv_count[2] = {};
    // 每个方向的累计接收计数（决定当前读取的环形槽位）
    int64_t arrival[2][2] = {};  // Ring direction, QP index
                                 // 到达计数：[环形方向][QP 编号]，发送端 put 完成后远端原子加一
    int64_t release[2] = {};
    // 释放计数：[环形方向]，接收端拷贝完数据后释放槽位，通知对端可复用
};
EP_STATIC_ASSERT(sizeof(PPSignals) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：信号区总大小不得超过预留的信号字节上限

}  // namespace deep_ep::layout
// 结束 layout 命名空间
