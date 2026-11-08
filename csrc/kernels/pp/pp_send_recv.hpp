#pragma once // 头文件保护：只被包含一次

#include <algorithm> // std::min
#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板
#include <string> // std::string

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API

#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）
#include <deep_ep/common/math.cuh> // 数学工具（对齐、整除等）

#include "../comm/api.hpp" // comm::Context：通信上下文
#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架

// pp 命名空间：流水线并行（Pipeline Parallelism）的点对点收发 kernel 封装
namespace deep_ep::pp {

// 计算可用的 QP 数量：最多使用 2 个 QP（受已分配数限制）
static int get_num_qps(const comm::Context& context) {
    const auto num_qps = std::min(2, context.num_allocated_qps); // 取 2 与已分配 QP 数的较小者
    return num_qps; // 返回实际使用的 QP 数
}

// 启动流水线并行发送 kernel：把本地张量 x 经 RDMA 发给目标 rank
static void launch_pp_send(
    const comm::Context& context, // 通信上下文
    void* x, const int64_t& num_x_bytes, // 待发送数据指针及其字节数
    const int& dst_rank_idx, // 目标 rank 编号
    const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors, // 单张量最大字节数；最多同时在途的张量数
    const int& num_sms, // 占用的 SM 数
    const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    constexpr int kNumThreads = 32; // 每 block 仅 1 个 warp（32 线程）：发送主要靠 QP 发起，无需更多线程
    const auto num_qps = get_num_qps(context); // 本 kernel 使用的 QP 数
    const auto num_smem_bytes = jit->device.get_num_smem_bytes(); // 设备可用共享内存上限
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms()); // 校验 SM 数合法

    // Compile // JIT 编译：实例化 pp_send_impl 模板
    const auto kernel = jit->compile("pp_send", std::format(R"(
#include <deep_ep/impls/pp/pp_send_recv.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::pp::pp_send_impl<{}, {}, {}, {}, {}>);
}}
)", num_sms, context.num_ranks, num_smem_bytes, num_qps, context.num_gpu_timeout_cycles));

    // Launch // 协作式启动：grid=num_sms 个 block，每 block 32 线程
    jit->launch(
        kernel, {
            .stream = stream, // 目标 CUDA 流
            .num_smem_bytes = num_smem_bytes, // 动态共享内存大小
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(kNumThreads, 1, 1), // block 维度：单 warp
            .cooperative = true, // 协作启动：支持 grid 内全局同步
        },
        context.dev_comm, context.window, // NCCL 设备通信器与对称内存窗口（RDMA 路径）
        x, num_x_bytes, // 待发送数据及大小
        context.buffer, context.workspace, // 对称数据缓冲区与 workspace（信号量）
        context.rank_idx, dst_rank_idx, // 本 rank 与目标 rank
        num_max_tensor_bytes, num_max_inflight_tensors // 张量大小上限与在途上限
    );
}

// 启动流水线并行接收 kernel：从源 rank 经 RDMA 收取张量到本地 x
static void launch_pp_recv(
    const comm::Context& context, // 通信上下文
    void* x, const int64_t& num_x_bytes, // 接收缓冲区指针及其字节数
    const int& src_rank_idx, // 源 rank 编号
    const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors, // 单张量最大字节数；最多同时在途的张量数
    const int& num_sms, // 占用的 SM 数
    const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    constexpr int kNumThreads = 32; // 每 block 仅 1 个 warp：接收主要轮询 QP 完成状态
    const auto num_qps = get_num_qps(context); // 本 kernel 使用的 QP 数
    const auto num_smem_bytes = jit->device.get_num_smem_bytes(); // 设备可用共享内存上限
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms()); // 校验 SM 数合法

    // Compile // JIT 编译：实例化 pp_recv_impl 模板
    const auto kernel = jit->compile("pp_recv", std::format(R"(
#include <deep_ep/impls/pp/pp_send_recv.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::pp::pp_recv_impl<{}, {}, {}, {}, {}>);
}}
)", num_sms, context.num_ranks, num_smem_bytes, num_qps, context.num_gpu_timeout_cycles));

    // Launch // 协作式启动：grid=num_sms 个 block，每 block 32 线程
    jit->launch(
        kernel, {
            .stream = stream, // 目标 CUDA 流
            .num_smem_bytes = num_smem_bytes, // 动态共享内存大小
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(kNumThreads, 1, 1), // block 维度：单 warp
            .cooperative = true, // 协作启动：支持 grid 内全局同步
        },
        context.dev_comm, context.window, // NCCL 设备通信器与对称内存窗口（RDMA 路径）
        x, num_x_bytes, // 接收缓冲区及大小
        context.buffer, context.workspace, // 对称数据缓冲区与 workspace（信号量）
        context.rank_idx, src_rank_idx, // 本 rank 与源 rank
        num_max_tensor_bytes, num_max_inflight_tensors // 张量大小上限与在途上限
    );
}

}  // namespace deep_ep::pp // 结束 pp 命名空间
