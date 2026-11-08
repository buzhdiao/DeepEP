#pragma once  // 头文件保护，防止被重复包含

// Make CLion CUDA indexing work  // 让 CLion 的 CUDA 索引能正常工作（IDE 场景下伪造架构宏）
#ifdef __CLION_IDE__  // 仅在 CLion IDE 解析时生效
#define __CUDA_ARCH__ 900  // 伪造 GPU 架构为 SM90，便于 IDE 补全与跳转
#define __CUDACC_RDC__  // 伪造可重定位设备代码宏
#define __CUDACC__  // 伪造 NVCC 编译器宏
#endif

// Remove Torch restrictions  // 解除 PyTorch 对半精度类型运算符/转换的限制宏
#ifdef __CUDA_NO_HALF_CONVERSIONS__  // 若定义了“禁用 half 转换”
#undef __CUDA_NO_HALF_CONVERSIONS__  // 取消定义，恢复 half 类型转换
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__  // 若定义了“禁用 half 运算符”
#undef __CUDA_NO_HALF_OPERATORS__  // 取消定义
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__  // 若定义了“禁用 half2 运算符”
#undef __CUDA_NO_HALF2_OPERATORS__  // 取消定义
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__  // 若定义了“禁用 bf16 转换”
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__  // 取消定义
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__  // 若定义了“禁用 bf16x2 运算符”
#undef __CUDA_NO_BFLOAT162_OPERATORS__  // 取消定义
#endif

#include <cstdint>  // 定宽整数类型
#include <cuda_bf16.h>  // bfloat16 类型支持
#include <cuda_runtime.h>  // CUDA 运行时 API

#include <cuda_fp8.h>  // FP8 浮点类型支持

// Compatibility: 256 bits LD/ST instructions  // 兼容性：256 位加载/存储指令所需的 longlong4_t
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000  // CUDA 13.0 及以上
using longlong4_t = longlong4_32a;  // 使用官方的 32 字节对齐 longlong4 变体（可安全用于 256 位 LD/ST）
#define make_longlong4_t make_longlong4_32a  // 对应的构造函数宏
#else  // 旧 CUDA 版本
struct alignas(32) longlong4_t { long long x, y, z, w; };  // 自行定义 32 字节对齐的 4×64 位结构体
__device__ __forceinline__ longlong4_t make_longlong4_t(  // 自行实现对应的构造函数
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};  // 逐字段初始化
}
#endif

#ifndef EP_NUM_TOPK_IDX_BITS  // topk 索引位宽宏（未定义时才定义）
#define EP_NUM_TOPK_IDX_BITS 64  // 默认使用 64 位整型表示 topk 索引
#endif

namespace deep_ep {

template <int kNumBits> struct int_with_bits;  // 按位宽选择整型类型的元编程模板（主模板仅声明）
template <> struct int_with_bits<8>  { using type = int8_t;  };  // 8 位 -> int8_t
template <> struct int_with_bits<16> { using type = int16_t; };  // 16 位 -> int16_t
template <> struct int_with_bits<32> { using type = int32_t; };  // 32 位 -> int32_t
template <> struct int_with_bits<64> { using type = int64_t; };  // 64 位 -> int64_t

using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;  // topk 索引类型：位宽由 EP_NUM_TOPK_IDX_BITS 决定

union sf_pack_t {  // 标量因子（scale factor）打包联合体：同一块内存两种视图
    float fp32;  // FP32 视图
    int ue8m0x4;  // 4×UE8M0（FP8 指数格式）打包视图
};

constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);  // 16 字节（一个向量寄存器）内可容纳的 SF 包数

// Alignment requirements  // 各类对齐要求（单位：字节）
constexpr int kNumTMAAlignmentBytes = 32;  // TMA 拷贝要求 32 字节对齐
constexpr int kNumRDMAAlignmentBytes = 64;  // RDMA 传输要求 64 字节对齐
constexpr int kNumAllocationAlignmentBytes = 2 * 1024 * 1024;  // 显存分配按 2MB 大页对齐
static_assert(kNumRDMAAlignmentBytes % kNumTMAAlignmentBytes == 0);  // RDMA 对齐必须是 TMA 对齐的整数倍
static_assert(kNumAllocationAlignmentBytes % kNumRDMAAlignmentBytes == 0);  // 分配对齐必须是 RDMA 对齐的整数倍

// Communication limits  // 通信规模上限常量
constexpr int kNumMaxRanks = 1024;  // 支持的最大 rank 数
constexpr int kNumMaxRDMARanks = kNumMaxRanks;  // 最大 RDMA rank 数
constexpr int kNumMaxNVLRanks = kNumMaxRanks;  // 最大 NVLink rank 数
constexpr int kNumMaxScaleoutRanks = kNumMaxRanks;  // 最大纵向扩展（跨节点）rank 数
constexpr int kNumMaxScaleupRanks = kNumMaxRanks;  // 最大横向扩展（节点内）rank 数
constexpr int kNumMaxContexts = 8;  // 最大 CUDA context（通信上下文）数
constexpr int kNumMaxQPs = 1024;  // 最大 RDMA QP（队列对）数
constexpr int kDefaultQPDepth = 1024;  // QP 默认深度
constexpr int kNumMaxSignalBytes = 16 * 1024 * 1024;  // 信号区最大字节数（16MB）

} // namespace deep_ep
