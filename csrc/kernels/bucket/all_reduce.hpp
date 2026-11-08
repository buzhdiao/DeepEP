#pragma once // 头文件保护：只被包含一次

#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装

#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）
#include <deep_ep/common/math.cuh> // 数学工具（ceil_div、align 等）
#include <deep_ep/layout/bucket/workspace.cuh> // bucket 通信 workspace 布局（信号量槽数等）

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架
#include "../comm/api.hpp" // comm::Context：通信上下文

namespace deep_ep::bucket { // bucket 命名空间：bucket 级集合通信 kernel 封装

// 启动 bucket 级 all-reduce：按 bucket 分组做多对多归约，自动选择混合（RDMA+NVLink）/纯 NVLink/纯 RDMA 三条路径
static void launch_all_reduce(
    const comm::Context& context, // 通信上下文
    const int& num_buckets, const comm::BucketList& buckets, // bucket 数量与列表（每 bucket 的偏移与字节数）
    const int& context_idx, const int& num_sms, // 本上下文在 workspace 信号量数组中的索引
    const int64_t& num_timeout_cycles, const float& scale, // 占用的 SM 数
    const at::cuda::CUDAStream& stream) { // GPU 超时周期数；归约后的缩放系数
    // Checks // 启动 kernel 的 CUDA 流
    EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets); // 基本合法性检查
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms()); // bucket 数不得超过 kNumMaxBuckets 上限
 // SM 数不得超过物理上限
    if (context.num_rdma_ranks > 1 and context.num_nvl_ranks > 1) { // 混合模式：跨机 RDMA 与机内 NVLink 同时启用
        EP_HOST_ASSERT(context.buffer_multimem != nullptr); // 混合 all-reduce 依赖 multimem 缓冲区
        EP_HOST_ASSERT(num_sms >= 2 and "Hybrid all-reduce requires at least 2 SMs"); // 混合路径至少需要 2 个 SM
        EP_HOST_ASSERT(context.num_rdma_ranks <= 128 and context.num_rdma_ranks * context.num_nvl_ranks <= 1024); // 规模限制：RDMA 域 ≤128，总 rank ≤1024
 // 跨机 RDMA rank 数与机内 NVLink rank 数
        const int rdma = context.num_rdma_ranks, nvl = context.num_nvl_ranks; // multimem warp 数：RDMA 域 ≤32 时取 8，否则取 4
        const int mm_warps = rdma <= 32 ? 8 : 4; // 1 个 warp 负责发起 RDMA 通信
        constexpr int issue_warps = 1; // 1 个 warp 负责本机 TMA 搬运与归约
        constexpr int reduce_warps = 1; // 广播 warp 下限：每个 warp 经 multimem 覆盖 32 个 RDMA rank
        const int min_broadcast_warps = reduce_warps * math::ceil_div(rdma, 32); // 广播 warp 数向上对齐到 4 的倍数
        const int broadcast_warps = math::align(issue_warps + reduce_warps + min_broadcast_warps, 4) - issue_warps - reduce_warps; // chunk 大小：RDMA 域 ≤4 用 256 KiB，否则 64 KiB
        const int num_chunk_bytes = (rdma <= 4 ? 256 : 64) * 1024; // 广播 warp 多时压缩每 warp 的共享内存配额
        // Limit total SMEM usage when more broadcast warps are needed // 归约 warp 的 TMA 缓冲大小（64/32 KiB）
        const int num_reduce_smem_bytes = (broadcast_warps <= 2 ? 64 : 32) * 1024; // 广播 warp 的 TMA 缓冲大小（32/8 KiB）
        const int num_broadcast_smem_bytes = (broadcast_warps <= 2 ? 32 : 8) * 1024; // 总线程数 = 各类 warp 数之和 × 32
        const int num_threads = (mm_warps * issue_warps + issue_warps + reduce_warps + broadcast_warps) * 32; // 动态共享内存总量
        const int num_smem_bytes = reduce_warps * num_reduce_smem_bytes + broadcast_warps * num_broadcast_smem_bytes; // TMA 缓冲 + 信号量不得超过共享内存上限
        EP_HOST_ASSERT(num_smem_bytes + (reduce_warps + broadcast_warps) * sizeof(uint64_t) <= jit->device.get_num_smem_bytes()); // workspace 中每 RDMA rank 可容纳的 chunk 槽位数
        const int num_slots = layout::ChunkStorage::kNumBytes / num_chunk_bytes / rdma; // 信号量槽位不得超过 workspace 总容量
        EP_HOST_ASSERT(2 * rdma * num_slots + num_sms * reduce_warps * rdma <= layout::AllReduceSignals::kNumTotalSlots and
                       "Hybrid all-reduce signals exceed the workspace");

        // Compile // 编译：实例化混合 all-reduce 模板
        const auto kernel = jit->compile("bucket_all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/hybrid.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::hybrid_all_reduce<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", rdma, nvl, num_sms, mm_warps, issue_warps, reduce_warps, broadcast_warps, num_reduce_smem_bytes, num_broadcast_smem_bytes,
            context.num_allocated_qps, num_timeout_cycles, scale != 1.0f, context_idx, num_chunk_bytes)); // 模板实参：QP 数、超时、是否缩放、上下文索引、chunk 大小
        // Launch // 协作式启动：grid=num_sms 个 block，cluster 视奇偶取 1 或 2
        const auto options = deep_jit::cuda::LaunchOptions { // 启动配置
            .stream = stream, // 目标 CUDA 流
            .num_smem_bytes = num_smem_bytes, // 动态共享内存大小
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(num_threads, 1, 1), // block 维度：mm/issue/reduce/broadcast 各类 warp
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
            .cooperative = true, // 协作启动：支持 grid 内全局同步
        };
        jit->launch( // 提交 kernel 启动
            kernel, options, // kernel 与启动配置
            context.dev_comm, context.window, // 设备通信器与对称内存窗口
            context.workspace, context.buffer, context.buffer_multimem, // workspace（信号量）、数据缓冲区与 multimem 缓冲区
            scale, num_buckets, buckets // 缩放系数与 bucket 列表
        );
    } else if (context.num_rdma_ranks == 1) { // 分支：机内 NVLink 域
        // NVLink only // 仅 NVLink 路径：所有 rank 都在同一 NVLink 域内
        EP_HOST_ASSERT(context.num_nvl_ranks > 1 and context.buffer_multimem != nullptr); // 需要 multimem 缓冲区支持组播归约

        // Compile // 编译：实例化 NVLink all-reduce 模板
        constexpr int kNumThreads = 1024; // 单 block 1024 线程
        const auto kernel = jit->compile("bucket_all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/nvlink.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::nvlink_all_reduce<{}, {}, {}, {}, {}, {}>);
}}
)", context.num_nvl_ranks, num_sms, kNumThreads,
            num_timeout_cycles, scale != 1.0f, context_idx)); // 模板实参：超时、是否缩放、上下文索引

        // Launch // 协作式启动：grid=num_sms 个 block，每 block 1024 线程
        const auto options = deep_jit::cuda::LaunchOptions { // 启动配置
            .stream = stream, // 目标 CUDA 流
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(kNumThreads, 1, 1), // block 维度：1024 线程
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：偶数 SM 配 2 卡 cluster
            .cooperative = true, // 协作启动：支持 grid 内全局同步
        };
        jit->launch( // 提交 kernel 启动
            kernel, options, // kernel 与启动配置
            context.dev_comm, context.window, // 设备通信器与对称内存窗口
            context.workspace, context.buffer_multimem, // workspace 与 multimem 缓冲区（multimem.ld_reduce 组播归约）
            scale, num_buckets, // 缩放系数与 bucket 数
            buckets // bucket 列表
        );
    } else { // 分支：纯跨机路径
        // RDMA only // 仅 RDMA 路径（无机内 NVLink 对端）
        constexpr int kNumIssueWarps = 8; // 8 个 warp 负责发起 RDMA 读写
        constexpr int kNumReduceWarps = 8; // 8 个 warp 负责接收与归约
        constexpr int kNumSmemBytesPerReduceWarp = 16 * 1024; // 每个归约 warp 的 TMA 缓冲 16 KiB
        constexpr int kNumThreads = (kNumIssueWarps + kNumReduceWarps) * 32; // 总线程数 = 16 warp × 32
        constexpr int kNumSmemBytes = kNumReduceWarps * kNumSmemBytesPerReduceWarp; // 动态共享内存总量
        EP_HOST_ASSERT(kNumSmemBytes + kNumReduceWarps * sizeof(uint64_t) <= jit->device.get_num_smem_bytes() and // TMA 缓冲不得超过共享内存上限
                       "RDMA all-reduce TMA buffers exceed the shared-memory limit");

        // Compile // 编译：实例化 RDMA all-reduce 模板
        const auto kernel = jit->compile("bucket_all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_all_reduce<{}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, num_sms,
            kNumIssueWarps, kNumReduceWarps, kNumSmemBytesPerReduceWarp, // 模板实参：issue/reduce warp 数与 TMA 缓冲大小
            context.num_allocated_qps, // 已分配 QP 数
            num_timeout_cycles, scale != 1.0f, context_idx)); // 超时、是否缩放、上下文索引

        // Launch // 协作式启动：grid=num_sms 个 block，每 block 16 个 warp
        const auto options = deep_jit::cuda::LaunchOptions { // 启动配置
            .stream = stream, // 目标 CUDA 流
            .num_smem_bytes = kNumSmemBytes, // 动态共享内存大小
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(kNumThreads, 1, 1), // block 维度：issue + reduce warp
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：偶数 SM 配 2 卡 cluster
            .cooperative = true, // 协作启动：支持 grid 内全局同步
        };
        jit->launch( // 提交 kernel 启动
            kernel, options, // kernel 与启动配置
            context.dev_comm, context.window, // 设备通信器与对称内存窗口
            context.workspace, context.buffer, // workspace（信号量）与数据缓冲区
            scale, num_buckets, // 缩放系数与 bucket 数
            buckets // bucket 列表
        );
    }
}
 // 结束 bucket 命名空间
}  // namespace deep_ep::bucket
