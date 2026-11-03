#pragma once  // 头文件保护，防止被重复包含

#include <cstddef>  // offsetof 等定义
#include <cstdint>  // 定宽整数类型

#include <deep_ep/common/exception.cuh>  // EP_STATIC_ASSERT 静态断言宏

// deep_ep::comm 命名空间：通信桶（bucket）描述符定义
namespace deep_ep::comm {

static constexpr int kNumMaxBuckets = 64;  // 一个桶列表最多支持的桶数

// Reduce-scatter/all-gather use the shard view, all-reduce uses the full view  // reduce-scatter/all-gather 使用分片视图，all-reduce 使用完整视图
struct Bucket {  // 描述通信缓冲中一个桶的元信息
    // Byte offset from the beginning of the bucket buffer to the full tensor (i.e. shard 0)  // 从桶缓冲区起始到完整张量（即分片 0）的字节偏移
    int64_t offset;  // 桶在缓冲区内的字节偏移
    // Number of bytes, either one shard or the full tensor (see `BucketList`)  // 字节数：既可是一个分片，也可是完整张量（见 BucketList）
    int64_t num_bytes;  // 桶的字节长度

    template <typename dtype_t>
    __forceinline__ __device__ __host__ int64_t get_num_elems() const {  // 计算本桶按指定类型可容纳的元素个数
        return num_bytes / static_cast<int64_t>(sizeof(dtype_t));  // 字节数除以元素大小
    }
};

EP_STATIC_ASSERT(sizeof(Bucket) == 16, "Invalid bucket descriptor size");  // 桶描述符必须恰好 16 字节（便于 GPU 间传输/打包）
EP_STATIC_ASSERT(offsetof(Bucket, offset) == 0, "Invalid bucket offset field");  // offset 字段必须位于偏移 0
EP_STATIC_ASSERT(offsetof(Bucket, num_bytes) == 8, "Invalid bucket num_bytes field");  // num_bytes 字段必须位于偏移 8

struct BucketList {  // 桶的定长集合，可在主机与设备间直接拷贝
    Bucket buckets[kNumMaxBuckets] = {};  // 最多 64 个桶，默认零初始化

    __forceinline__ __device__ __host__ Bucket& operator[](const int& bucket_idx) {  // 非常量访问：返回指定桶的引用
        return buckets[bucket_idx];  // 直接下标访问
    }

    __forceinline__ __device__ __host__ const Bucket& operator[](const int& bucket_idx) const {  // 常量访问：返回指定桶的只读引用
        return buckets[bucket_idx];  // 直接下标访问
    }
};

}  // namespace deep_ep::comm
