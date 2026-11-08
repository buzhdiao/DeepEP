#pragma once
// 头文件保护：保证本文件只被包含一次

#include <deep_ep/layout/bucket/all_gather.cuh>
// 引入 AllGather 信号定义
#include <deep_ep/layout/bucket/all_reduce.cuh>
// 引入 AllReduce 信号定义
#include <deep_ep/layout/common/barrier.cuh>
// 引入公共屏障信号定义
#include <deep_ep/layout/bucket/chunk.cuh>
// 引入分块存储（ChunkStorage）定义
#include <deep_ep/layout/bucket/copy_engine.cuh>
// 引入拷贝引擎信号定义
#include <deep_ep/layout/bucket/reduce_scatter.cuh>
// 引入 ReduceScatter 信号定义
#include <deep_ep/common/compiled.cuh>
// 引入编译期常量与静态断言
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) BucketSignals {
    // Bucket 子系统的信号区：按 RDMA 对齐，聚合各算子与屏障的信号
    BarrierSignals barrier;
    // 公共屏障信号
    CopyEngineSignals copy_engine;
    // 拷贝引擎信号（各 rank 的写指针基址）
    AllGatherSignals all_gather;
    // AllGather 完成信号（各 QP×chunk 的尾指针）
    ReduceScatterSignals reduce_scatter;
    // ReduceScatter 头/尾指针信号
    AllReduceSignals all_reduce;
    // AllReduce 完成计数信号
};
EP_STATIC_ASSERT(sizeof(BucketSignals) * kNumMaxContexts <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：单上下文信号大小 × 最大上下文数不得超过预留的信号字节上限

struct alignas(kNumRDMAAlignmentBytes) BucketWorkspace {
    // Bucket 工作区：按 RDMA 对齐，由每个上下文一套的信号数组 + 分块数据存储组成
    BucketSignals signals[kNumMaxContexts];
    // 每个通信上下文独立的一套信号
    ChunkStorage chunk_storage;
    // 分块数据存储：各算子实际交换的数据放在这 256 MiB 缓冲中
};

}  // namespace deep_ep::layout
// 结束 layout 命名空间
