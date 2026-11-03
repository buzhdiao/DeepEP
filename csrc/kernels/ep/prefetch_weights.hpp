#pragma once // 头文件保护：只被包含一次

#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板
#include <string> // std::string

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API
#include <nccl_device.h> // NCCL 设备端 API（ncclWindow_t 等）

#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）
#include <deep_ep/common/ptx.cuh> // PTX 工具（mbarrier 等）
#include <deep_ep/layout/ep/eplb.cuh> // EPLB（专家负载均衡）的布局定义

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架
#include "../comm/api.hpp" // comm::Context：通信上下文（窗口、workspace、QP 等）

// ep 命名空间：专家并行（Expert Parallelism）kernel 封装
namespace deep_ep::ep {

// 启动 EPLB 权重预取 kernel：把下一层要用的专家权重提前搬到 GPU（隐藏权重加载延迟）
static void launch_lb_prefetch_weights(const comm::Context& context, // 通信上下文
                                       const int& num_weights, // 每专家权重的元素个数
                                       const layout::EPWeightList& weights, // 权重列表（各专家的地址与大小）
                                       const int* redundancy_mapping, // 冗余专家 -> 真实专家 的映射表
                                       const int& num_redundant_experts, // 冗余专家数
                                       const int& num_local_experts, // 本地专家数
                                       const int& num_sms, // 占用的 SM 数
                                       const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    // Each warp owns one fixed-size chunk buffer and handles both its load and store // 每个 warp 私有一个定长 chunk 缓冲，自己负责加载与存储
    constexpr int num_chunk_bytes = 12 * 1024; // 每个 chunk 12 KiB
    constexpr int num_bytes_per_warp = num_chunk_bytes + sizeof(ptx::mbarrier); // 每 warp 的共享内存开销 = chunk 缓冲 + 1 个 mbarrier
    const int num_smem_bytes = jit->device.get_num_smem_bytes(); // 设备可用共享内存上限
    constexpr int num_smem_layout_bytes = sizeof(layout::EPLBSharedMemoryLayout); // 布局元数据占用的共享内存
    EP_HOST_ASSERT(num_smem_layout_bytes + num_bytes_per_warp <= num_smem_bytes); // 校验共享内存够用
    const int num_warps = (num_smem_bytes - num_smem_layout_bytes) / num_bytes_per_warp; // 按剩余共享内存计算每 block 的 warp 数
    EP_HOST_ASSERT(num_warps <= 32); // 单 block 最多 32 个 warp
    const int num_threads = num_warps * 32; // 每 block 线程数 = warp 数 × 32

    // Compile // JIT 编译：实例化 lb_prefetch_weights 模板
    const auto kernel = jit->compile("ep_lb_prefetch_weights", std::format(R"(
#include <deep_ep/impls/ep/prefetch_weights.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::lb_prefetch_weights<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_sms, num_warps, context.num_nvl_ranks, num_local_experts, num_redundant_experts,
    num_weights, num_chunk_bytes, context.num_gpu_timeout_cycles));

    // Launch // 协作式启动：grid=num_sms 个 block，每 block num_threads 线程；奇偶 SM 配成 2 卡 cluster
    jit->launch(
        kernel, {
            .stream = stream.stream(), // 目标 CUDA 流
            .num_smem_bytes = num_smem_bytes, // 动态共享内存大小
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(num_threads, 1, 1), // block 维度：num_warps 个 warp
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
            .cooperative = true, // 协作启动：支持 grid 内全局同步
        },
        redundancy_mapping, context.dev_comm, context.window, context.workspace, // 冗余映射、设备通信器、对称窗口、workspace
        weights // 权重列表（设备端读取）
    );
}

}  // namespace deep_ep::ep // 结束 ep 命名空间
