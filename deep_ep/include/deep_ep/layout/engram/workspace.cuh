#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cstdint>
// 引入定宽整型（uint32_t 等）
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏（EP_STATIC_ASSERT）
#include <deep_ep/layout/common/barrier.cuh>
// 引入公共屏障信号（BarrierSignals）

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) EngramWorkspace {
    // Engram 工作区：按 RDMA 对齐；存放屏障信号与各层的发起屏障计数器
    static constexpr int kNumMaxLayers = 128;
    // 支持的最大层数
    static constexpr int kNumMaxPeerContexts = 2048;
    // 每 (对端, 上下文) 组合的计数器上限

    BarrierSignals barrier;
    // 公共屏障信号
    uint32_t issue_counters[kNumMaxLayers][kNumMaxPeerContexts];
    // 发起屏障计数器：issue warp 在每层结束时用原子加计数，
    // 最后一个到达的 warp 负责 flush QP 并把计数清零，其余 warp 等待清零后再进下一层

    __forceinline__ __device__ __host__
    uint32_t* get_issue_counter_ptr(const int layer_idx, const int peer_context_idx) {
        // 取指定 (层, 对端上下文) 的计数器地址：偏移 = 层编号 × 每层组合数 + 组合编号
        return &issue_counters[layer_idx][peer_context_idx];
    }
};
EP_STATIC_ASSERT(sizeof(EngramWorkspace) <= kNumMaxSignalBytes, "Too many signals");
// 编译期断言：工作区总大小不得超过预留的信号字节上限

// The Engram buffer layout.
// The GPU segment gives each layer a fixed-size region in one receive buffer:
//   [layer 0: num_max_tokens * num_entries_per_token entries] ... [layer (L - 1): ...]
// Each rank's CPU/GPU RDMA storage shard packs the per-layer storages back-to-back:
//   [layer 0: entries of layer 0] ... [layer (L - 1): entries of layer (L - 1)]
// Engram 缓冲布局说明。
// GPU 接收缓冲把每个层划成一个固定大小的区域，各层依次排列：
//   [层 0: num_max_tokens × num_entries_per_token 个条目] ... [层 L-1: ...]
// 每个 rank 的 CPU/GPU RDMA 存储分片则把各层存储首尾相接打包：
//   [层 0 的条目] ... [层 L-1 的条目]
template <int kNumRanks, int kNumSFShards, int kHiddenBytes, int kEntriesPerToken, int... kEntriesPack>
struct EngramLayout {
    // 模板参数：rank 总数、SF 分片数、单条目 hidden 字节数、每 token 条目数、各层条目数（参数包）
    static constexpr int kNumLayers = sizeof...(kEntriesPack);
    // 层数 = 各层条目数参数包的长度
    static constexpr int kNumSFOwnersPerShard = kNumRanks / kNumSFShards;
    // 每个 SF 分片（owner rank）聚合的 rank 数
    static constexpr int kNumHiddenBytes = kHiddenBytes;
    // 单个条目的 hidden 字节数（条目定长）
    static constexpr int kNumEntriesPerToken = kEntriesPerToken;
    // 每 token 的条目数（接收缓冲按此展开）

    __forceinline__ __device__ __host__
    static constexpr int get_num_entries(const int& layer_idx) {
        // 取指定层的条目数（存储分片中该层的长度）
        constexpr int kNumEntriesPerLayer[kNumLayers] = {kEntriesPack...};
        // 把参数包展开成每层条目数的编译期数组
        return kNumEntriesPerLayer[layer_idx];
        // 按层号取值
    }

    __forceinline__ __device__ __host__
    static constexpr int64_t get_num_prefix_entries(const int& layer_idx) {
        // 计算存储分片中指定层之前所有层的条目总数（前缀和）
        int64_t num_prefix_entries = 0;
        // 累加器
        for (int i = 0; i < layer_idx; ++ i)
            num_prefix_entries += get_num_entries(i);
        // 逐层累加层 i 的条目数
        return num_prefix_entries;
        // 返回前缀条目数
    }

    __forceinline__ __device__ __host__
    static int64_t get_storage_byte_offset(const int& layer_idx, const int& entry_idx) {
        // 计算存储分片内的字节偏移：布局为各层条目首尾相接
        // 偏移 = (前面各层条目总数 + 本层内条目编号) × 单条目字节数
        return (get_num_prefix_entries(layer_idx) + entry_idx) * kNumHiddenBytes;
    }

    __forceinline__ __device__ __host__
    static int64_t get_recv_byte_offset(const int& layer_idx, const int& entry_idx, const int& num_max_tokens) {
        // 计算接收缓冲内的字节偏移：每层是一个定长区域，层内按条目编号排列
        // 偏移 = (层编号 × 每 token 条目数 × 最大 token 数 + 条目编号) × 单条目字节数
        return (layer_idx * kNumEntriesPerToken * num_max_tokens + entry_idx) * kNumHiddenBytes;
    }

    __forceinline__ __device__ __host__
    static constexpr int get_num_sf_rows_per_shard(const int& layer_idx) {
        // 每个 SF 分片为指定层保存的行数 = 每分片聚合的 rank 数 × 该层条目数
        return kNumSFOwnersPerShard * get_num_entries(layer_idx);
    }

    __forceinline__ __device__ __host__
    static int64_t get_sf_row_idx(const int& layer_idx, const int& global_idx) {
        // 计算 SF 行号：条目落在其归属分片的第几行
        // 行号 = 前面各层在该分片内的行总数 + 全局编号对该层行数的余数
        return kNumSFOwnersPerShard * get_num_prefix_entries(layer_idx) +
               global_idx % get_num_sf_rows_per_shard(layer_idx);
    }

    __forceinline__ __device__ __host__
    static int get_sf_shard_idx(const int& layer_idx, const int& global_idx) {
        // 计算 SF 分片编号：全局编号整除该层每分片行数，得到归属哪个 owner 分片
        return global_idx / get_num_sf_rows_per_shard(layer_idx);
    }
};

}  // namespace deep_ep::layout
// 结束 layout 命名空间
