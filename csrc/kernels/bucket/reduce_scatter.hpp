#pragma once // 头文件保护：只被包含一次

#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板
#include <string> // std::string：拼接模板实参

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API

#include <deep_ep/comm/bucket.cuh> // bucket 通信公共定义（kNumMaxBuckets、BucketList）
#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）
#include <deep_ep/common/math.cuh> // 数学工具（ceil_div、align 等）

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架

namespace deep_ep::bucket { // bucket 命名空间：bucket 级集合通信 kernel 封装

// 启动 bucket 级 reduce-scatter：按 bucket 分组做跨 rank 归约并散到各 rank，支持混合/纯 RDMA（fp32、bf16）/纯 NVLink 路径
static void launch_reduce_scatter(
    const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& window, // NCCL 设备通信器；对称内存窗口
    void* workspace, void* buffer, void* buffer_multimem, // workspace（信号量）、数据缓冲区与 multimem 缓冲区
    const int& num_buckets, const comm::BucketList& buckets, // bucket 数量与列表（每 bucket 的偏移与字节数）
    const int& num_rdma_ranks, const int& num_nvl_ranks, // 跨机 RDMA rank 数与机内 NVLink rank 数
    const int& num_sms, const int& num_qps, // 占用的 SM 数；QP 数量
    const int64_t& num_timeout_cycles, const float& scale, // GPU 超时周期数；归约后的缩放系数
    const std::string& comm_precision, const int& context_idx, // 通信精度（fp32 或 bf16）；workspace 上下文索引
    const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    // Checks // 基本合法性检查
    EP_HOST_ASSERT(comm_precision == "fp32" or comm_precision == "bf16"); // 通信精度只支持 fp32 或 bf16
    EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets); // bucket 数不得超过 kNumMaxBuckets 上限
    EP_HOST_ASSERT(num_rdma_ranks > 0 and num_nvl_ranks > 0); // 两层域 rank 数必须为正
    EP_HOST_ASSERT(num_rdma_ranks > 1 or num_nvl_ranks > 1); // 至少一层域的规模大于 1（否则无需通信）
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms()); // SM 数不得超过物理上限
    EP_HOST_ASSERT((comm_precision != "bf16" or (num_rdma_ranks > 1 and num_nvl_ranks == 1)) and // bf16 通信仅支持纯 RDMA reduce-scatter
                   "BF16 communication is only supported for RDMA-only reduce-scatter");

    if (num_rdma_ranks > 1 and num_nvl_ranks > 1) { // 分支一：混合模式（跨机 RDMA + 机内 NVLink）
        // Hybrid RDMA and NVLink // 混合 RDMA 与 NVLink 路径
        EP_HOST_ASSERT(num_sms > 1 and "Hybrid reduce-scatter requires at least 2 SMs"); // 混合 reduce-scatter 至少需要 2 个 SM

        constexpr int kNumMultimemWarps = 8; // 8 个 multimem warp：经 NVLink multimem 组播搬运数据
        constexpr int kNumMultimemWarpsPerGroup = 4; // multimem warp 按 4 个一组划分
        constexpr int kNumIssueWarps = 2; // 2 个 issue warp：发起跨机 RDMA 读写
        constexpr int kNumReduceWarps = 2; // 2 个 reduce warp：接收与归约（TMA 搬运 + 累加）
        constexpr int kNumSmemBytesPerReduceWarp = 64 * 1024; // 每个归约 warp 的 TMA 缓冲 64 KiB
        constexpr int kNumThreads = (kNumMultimemWarps + kNumIssueWarps + kNumReduceWarps) * 32; // 总线程数 =（multimem + issue + reduce）warp 数 × 32
        constexpr int kNumSmemBytes = kNumReduceWarps * kNumSmemBytesPerReduceWarp; // 动态共享内存总量 = 归约 warp 数 × 每 warp 缓冲
        EP_HOST_ASSERT(kNumSmemBytes + kNumReduceWarps * sizeof(uint64_t) <= jit->device.get_num_smem_bytes() and // TMA 缓冲不得超过共享内存上限
                       "Hybrid reduce-scatter TMA buffers exceed the shared-memory limit");

        // Compile // 编译：实例化混合 reduce-scatter 模板
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/hybrid.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::hybrid_reduce_scatter<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                         num_rdma_ranks, num_nvl_ranks,
                            num_sms, // 模板实参：SM 数
                            kNumMultimemWarps, // multimem warp 数
                            kNumMultimemWarpsPerGroup, // 每组 multimem warp 数
                            kNumIssueWarps, // issue warp 数
                            kNumReduceWarps, // reduce warp 数
                            kNumSmemBytesPerReduceWarp, // 每归约 warp 的 TMA 缓冲大小
                            num_qps, // QP 数
                            num_timeout_cycles, scale != 1.0f, // 超时；是否缩放
                            context_idx)); // workspace 上下文索引

        jit->launch( // 协作式启动：grid=num_sms 个 block，cluster 视奇偶取 1 或 2
            kernel, { // 启动配置
                .stream = stream, // 目标 CUDA 流
                .num_smem_bytes = kNumSmemBytes, // 动态共享内存大小
                .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
                .block_dim = dim3(kNumThreads, 1, 1), // block 维度：multimem/issue/reduce 各类 warp
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
                .cooperative = true, // 协作启动：支持 grid 内全局同步
            },
            nccl_dev_comm, window, // 设备通信器与对称内存窗口
            workspace, buffer, buffer_multimem, // workspace、数据缓冲区与 multimem 缓冲区
            scale, num_buckets, // 缩放系数与 bucket 数
            buckets // bucket 列表
        );
    } else if (num_rdma_ranks > 1 and comm_precision == "bf16") { // 分支二：纯 RDMA + bf16 精度（需先做 fp32→bf16 转换）
// bf16 路径的 warp 配置：cast warp 做 fp32→bf16 转换，issue/reduce warp 按域规模调整分组与批量
        const int num_cast_warps = num_rdma_ranks < 4 ? 4 : 8; // bf16 转换 warp 数：RDMA 域小取 4，否则取 8
        const int num_issue_warps = num_rdma_ranks < 4 ? 8 : 4; // issue warp 数：RDMA 域小取 8，否则取 4
        const int num_reduce_warps = num_rdma_ranks < 4 ? 4 : 8; // 归约 warp 数：RDMA 域小取 4，否则取 8
        const int cast_group = num_rdma_ranks < 4 ? 1 : 4; // 转换 warp 的分组大小
        const int reduce_group = num_rdma_ranks < 4 ? 1 : 4; // 归约 warp 的分组大小
        const int issue_batch = num_rdma_ranks < 4 ? 1 : (num_rdma_ranks < 16 ? 4 : 8); // issue 侧的批量提交数：按域规模选 1/4/8
        const int reduce_batch = num_rdma_ranks < 4 ? 1 : 8; // 归约侧的批量提交数：域小取 1，否则取 8
        const int num_threads = (num_cast_warps + num_issue_warps + num_reduce_warps) * 32; // 总线程数 =（cast + issue + reduce）warp 数 × 32

        // Compile // 编译：实例化 bf16 版 RDMA reduce-scatter 模板（需按具体架构编译）
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_reduce_scatter_bf16<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                         num_rdma_ranks, num_sms, num_cast_warps, num_issue_warps, num_reduce_warps, num_qps,
                            num_timeout_cycles, scale != 1.0f, context_idx, // 超时；是否缩放；上下文索引
                            issue_batch, reduce_batch, cast_group, reduce_group), // 批量与分组模板实参
            {.arch = jit->device.get_arch(/* use_arch_family = */ false)}); // 编译选项：使用设备的具体架构（而非架构家族）

        jit->launch( // 协作式启动：grid=num_sms 个 block
            kernel, { // 启动配置
                .stream = stream, // 目标 CUDA 流
                .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
                .block_dim = dim3(num_threads, 1, 1), // block 维度：cast/issue/reduce 各类 warp
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
                .cooperative = true, // 协作启动：支持 grid 内全局同步
            },
            nccl_dev_comm, window, // 设备通信器与对称内存窗口
            workspace, buffer, // workspace（信号量）与数据缓冲区
            scale, num_buckets, // 缩放系数与 bucket 数
            buckets // bucket 列表
        );
    } else if (num_rdma_ranks > 1) { // 分支三：纯 RDMA（fp32）
        // RDMA only // 仅 RDMA 路径
        constexpr int kNumIssueWarps = 8; // 8 个 issue warp：发起跨机 RDMA 读写
        constexpr int kNumReduceWarps = 8; // 8 个 reduce warp：接收与归约
        constexpr int kNumThreads = (kNumIssueWarps + kNumReduceWarps) * 32; // 总线程数 = 16 warp × 32
        const int num_batch_chunks = num_rdma_ranks >= 8 ? 8 : num_rdma_ranks; // 批量 chunk 数：RDMA 域 ≥8 时取 8，否则等于域大小

        // Compile // 编译：实例化 RDMA reduce-scatter 模板
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_reduce_scatter<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                         num_rdma_ranks, num_sms,
                            kNumIssueWarps, kNumReduceWarps, num_qps, // issue/reduce warp 数与 QP 数
                            num_timeout_cycles, scale != 1.0f, context_idx, // 超时；是否缩放；上下文索引
                            num_batch_chunks, num_batch_chunks)); // issue/reduce 两侧的批量 chunk 数

        jit->launch( // 协作式启动：grid=num_sms 个 block，每 block 16 个 warp
            kernel, { // 启动配置
                .stream = stream, // 目标 CUDA 流
                .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
                .block_dim = dim3(kNumThreads, 1, 1), // block 维度：issue + reduce warp
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
                .cooperative = true, // 协作启动：支持 grid 内全局同步
            },
            nccl_dev_comm, window, // 设备通信器与对称内存窗口
            workspace, buffer, // workspace（信号量）与数据缓冲区
            scale, num_buckets, // 缩放系数与 bucket 数
            buckets // bucket 列表
        );
    } else { // 分支四：纯机内 NVLink
        // NVLink only // 仅 NVLink 路径
        constexpr int kNumThreads = 1024; // 单 block 1024 线程

        // Compile // 编译：实例化 NVLink reduce-scatter 模板
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/nvlink.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::nvlink_reduce_scatter<{}, {}, {}, {}, {}, {}>);
}}
)",                         num_nvl_ranks, num_sms, kNumThreads,
                            num_timeout_cycles, scale != 1.0f, context_idx)); // 超时；是否缩放；上下文索引

        jit->launch( // 协作式启动：grid=num_sms 个 block，每 block 1024 线程
            kernel, { // 启动配置
                .stream = stream, // 目标 CUDA 流
                .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
                .block_dim = dim3(kNumThreads, 1, 1), // block 维度：1024 线程
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
                .cooperative = true, // 协作启动：支持 grid 内全局同步
            },
            nccl_dev_comm, window, // 设备通信器与对称内存窗口
            workspace, buffer, buffer_multimem, // workspace、数据缓冲区与 multimem 缓冲区
            scale, num_buckets, // 缩放系数与 bucket 数
            buckets // bucket 列表
        );
    }
}

}  // namespace deep_ep::bucket
