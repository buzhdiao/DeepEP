#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cstdint>
// 引入定宽整型（int8_t 等）

#include <deep_ep/comm/bucket.cuh>
// 引入通信层的 Bucket / BucketList 定义
#include <deep_ep/common/compiled.cuh>
// 引入编译期常量与静态断言
#include <deep_ep/common/exception.cuh>
// 引入静态断言宏
#include <deep_ep/common/math.cuh>
// 引入数学工具（ceil_div 等）

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

struct alignas(kNumRDMAAlignmentBytes) ChunkStorage {
    // 分块存储的底层缓冲：按 RDMA 对齐，容量固定
    static constexpr int kNumBytes = 256 * 1024 * 1024;
    // 存储容量：256 MiB
    EP_STATIC_ASSERT(kNumBytes % kNumRDMAAlignmentBytes == 0, "Invalid RDMA alignment");
    // 编译期断言：容量必须按 RDMA 对齐字节数整除

    int8_t data[kNumBytes];
    // 原始字节缓冲：所有 chunk 视图都落在其中
};

template <int kNumChunks, int kNumBytesPerChunk>
struct ChunkView1D {
    // 一维分块视图：把存储视为连续 kNumChunks 个等大 chunk
    EP_STATIC_ASSERT(kNumChunks > 0, "Invalid number of chunks");
    // 编译期断言：chunk 数必须为正
    EP_STATIC_ASSERT(kNumBytesPerChunk > 0 and kNumBytesPerChunk % kNumRDMAAlignmentBytes == 0,
                     "Invalid chunk size");
    // 编译期断言：chunk 大小必须为正且按 RDMA 对齐
    EP_STATIC_ASSERT(kNumChunks * kNumBytesPerChunk <= ChunkStorage::kNumBytes,
                     "Chunk view exceeds the storage capacity");
    // 编译期断言：视图总大小不得超过存储容量

    ChunkStorage& storage;
    // 底层存储引用

    __forceinline__ __device__ void* get_chunk_ptr(const int& chunk_idx) const {
        // 取指定 chunk 的指针：偏移 = chunk 编号 × 每 chunk 字节数
        const int chunk_offset = chunk_idx * kNumBytesPerChunk;
        // 计算该 chunk 在存储内的字节偏移
        return storage.data + chunk_offset;
        // 返回数据区中该偏移处的指针
    }
};

template <int kNumRanks, int kNumChunksPerRank, int kNumBytesPerChunk>
struct ChunkView2D {
    // 二维分块视图：存储按 rank 分区，每 rank 再分 kNumChunksPerRank 个 chunk
    EP_STATIC_ASSERT(kNumRanks > 0, "Invalid number of ranks");
    // 编译期断言：rank 数必须为正
    EP_STATIC_ASSERT(kNumChunksPerRank > 0, "Invalid number of chunks per rank");
    // 编译期断言：每 rank chunk 数必须为正
    EP_STATIC_ASSERT(kNumBytesPerChunk > 0 and kNumBytesPerChunk % kNumRDMAAlignmentBytes == 0,
                     "Invalid chunk size");
    // 编译期断言：chunk 大小必须为正且按 RDMA 对齐
    EP_STATIC_ASSERT(kNumRanks * kNumChunksPerRank * kNumBytesPerChunk <= ChunkStorage::kNumBytes,
                     "Chunk view exceeds the storage capacity");
    // 编译期断言：视图总大小不得超过存储容量

    ChunkStorage& storage;
    // 底层存储引用

    __forceinline__ __device__ void* get_chunk_ptr(
        const int& rank_idx, const int& chunk_idx) const {
        // 取 (rank, chunk) 的指针：先展平成一维 chunk 编号，再乘 chunk 大小得到字节偏移
        const int global_chunk_idx = rank_idx * kNumChunksPerRank + chunk_idx;
        // 展平：全局 chunk 编号 = rank 编号 × 每 rank chunk 数 + rank 内 chunk 编号
        const int chunk_offset = global_chunk_idx * kNumBytesPerChunk;
        // 计算字节偏移
        return storage.data + chunk_offset;
        // 返回数据区中该偏移处的指针
    }
};

// Iterates over all buckets as one flat sequence of `kNumBytesPerChunk`-sized chunks, e.g.
//     [bucket 0: chunk 0, 1, ..., n0 - 1][bucket 1: chunk n0, n0 + 1, ..., n0 + n1 - 1]...
// Each iterator visits chunks `start_chunk_idx, start_chunk_idx + kStride, ...`, so `kStride` iterators
// with distinct starts in `[0, kStride)` cover every chunk exactly once
// 把所有 bucket 展平成一串大小为 kNumBytesPerChunk 的 chunk 并遍历，例如：
//     [bucket 0: chunk 0, 1, ..., n0-1][bucket 1: chunk n0, n0+1, ..., n0+n1-1]...
// 每个迭代器访问 start_chunk_idx, start_chunk_idx+kStride, ... 这些 chunk，
// 因此 kStride 个起始编号互不相同的迭代器（起始都在 [0, kStride)）恰好不重不漏覆盖所有 chunk
template <int kNumBytesPerChunk, int kStride>
struct ChunkIterator {
    const int num_buckets;
    // bucket 总数
    const comm::BucketList& buckets;
    // bucket 列表（每个 bucket 记录一段字节范围）
    int num_total_chunks = 0;           // Total across all buckets, including each bucket's partial tail
                                        // 所有 bucket 的 chunk 总数（含每个 bucket 末尾的不满 chunk）

    // States, valid after a successful `next()`
    // 状态字段：仅在 next() 返回 true 后有效
    comm::Bucket bucket;
    // 当前 chunk 所属的 bucket
    int chunk_idx;                      // Global chunk index across all buckets
                                        // 跨所有 bucket 的全局 chunk 编号
    int num_chunk_bytes;                // Only the last chunk of a bucket may be smaller than `kNumBytesPerChunk`
                                        // 本 chunk 字节数（只有 bucket 的最后一个 chunk 可能不满）
    int64_t chunk_offset_in_bucket;     // Byte offset of this chunk within `bucket`
                                        // 本 chunk 在所属 bucket 内的字节偏移

    // NOTES: `int` is enough for chunk indices, as a chunk is normally at least several KB, which covers several TB in total
    // 备注：chunk 编号用 int 即可——每个 chunk 通常至少几 KB，int 已能覆盖数 TB 的总量
    __forceinline__ __device__ ChunkIterator(const int& num_buckets, const comm::BucketList& buckets, const int& start_chunk_idx):
        num_buckets(num_buckets), buckets(buckets), chunk_idx(start_chunk_idx - kStride) {
        // 构造：记录 bucket 列表；chunk_idx 先回退一个步长，使首次 next() 恰好指向 start_chunk_idx
        for (int i = 0; i < num_buckets; ++ i)
            num_total_chunks += static_cast<int>(math::ceil_div<int64_t>(buckets[i].num_bytes, kNumBytesPerChunk));
        // 预先累加每个 bucket 的 chunk 数（字节数向上取整到 chunk 大小）
    }

    __forceinline__ __device__ bool next() {
        // 推进到本迭代器的下一个 chunk；遍历完返回 false
        // Move chunk
        // 编号前进一个步长
        chunk_idx += kStride;

        // Locate bucket
        // 定位编号所在的 bucket（顺着 bucket 顺序推进起点）
        while (bucket_idx < num_buckets) {
            bucket = buckets[bucket_idx];
            // 取当前 bucket
            const int bucket_end_chunk_idx = bucket_start_chunk_idx + static_cast<int>(math::ceil_div<int64_t>(bucket.num_bytes, kNumBytesPerChunk));
            // 该 bucket 的结束 chunk 编号（不含）：起始编号 + 该 bucket 的 chunk 数
            if (chunk_idx < bucket_end_chunk_idx) {
                // 编号落在当前 bucket 内：解析出 chunk 信息
                chunk_offset_in_bucket = static_cast<int64_t>(chunk_idx - bucket_start_chunk_idx) * kNumBytesPerChunk;
                // chunk 在 bucket 内的字节偏移 = (编号 - bucket 起始编号) × chunk 大小
                // NOTES: only the last chunk may be partial, so the remainder fits in `int` without a 64-bit `min`
                // 备注：只有最后一个 chunk 可能不满，余数必在 int 范围内，无需 64 位 min
                const bool is_last_chunk = chunk_idx + 1 == bucket_end_chunk_idx;
                // 判断是否为该 bucket 的最后一个 chunk
                num_chunk_bytes = is_last_chunk ? static_cast<int>(bucket.num_bytes - chunk_offset_in_bucket) : kNumBytesPerChunk;
                // 最后一个 chunk 取剩余字节；其余都是完整的 chunk 大小
                return true;
                // 成功取到一个 chunk
            }

            // Move into the next bucket
            // 越过当前 bucket：推进 bucket 起始编号与 bucket 下标
            bucket_start_chunk_idx = bucket_end_chunk_idx;
            bucket_idx += 1;
        }
        return false;
        // 所有 bucket 都已遍历完
    }

private:
    int bucket_idx = 0, bucket_start_chunk_idx = 0;
    // 内部状态：当前 bucket 下标、当前 bucket 第一个 chunk 的全局编号
};

}  // namespace deep_ep::layout
// 结束 layout 命名空间
