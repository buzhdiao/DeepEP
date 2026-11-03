#pragma once // 头文件保护：只被包含一次

#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API
#include <nccl_device.h> // NCCL 设备端 API（ncclWindow_t 等）

#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（NCCL_CHECK 等）

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架

// comm 命名空间：基于 NCCL 设备 API 的集合通信 kernel 封装
namespace deep_ep::comm {

// 启动跨层级 barrier kernel（scale-out 跨机 + scale-up 机内两层同步）
static void launch_barrier(const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window, // NCCL 设备通信器；对称内存窗口
                           void* signals, // barrier 信号量缓冲区（GPU 侧）
                           const int& scaleout_rank_idx, const int& scaleup_rank_idx, // 本 rank 在两层域中的编号
                           const int& num_scaleout_ranks, const int& num_scaleup_ranks, // 两层域内的 rank 总数
                           const int64_t& num_timeout_cycles, // GPU 侧自旋等待的超时周期数
                           const bool& is_scaleup_nvlink, // scale-up 域是否走 NVLink（否则走 GIN/RDMA）
                           const bool& flush_stores, const bool& sequential, // 是否 flush store；是否串行执行两层 barrier
                           const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    // Number of threads equals to the number of ranks // 线程数固定为 512（每线程对应一个对端 rank）
    constexpr auto kNumThreads = 512;

    // NOTES: only the parallel hybrid kernel needs 2 SMs; the sequential mode does scaleout and
    // scaleup one after another, so a single SM is sufficient.
    // 并行混合模式需要 2 个 SM（跨机与机内 barrier 同时进行）；串行模式依次执行两层，1 个 SM 即可
    const auto num_sms = (not sequential and num_scaleout_ranks > 1) ? 2 : 1;

    // Compile // JIT 编译：实例化 barrier_impl 模板，模板参数由上述运行期值填充
    const auto kernel = jit->compile("barrier", std::format(R"(
#include <deep_ep/impls/comm/barrier.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::comm::barrier_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", is_scaleup_nvlink, num_sms, kNumThreads,
        num_scaleout_ranks, num_scaleup_ranks,
        num_timeout_cycles, flush_stores, sequential));

    // Launch // 以协作式（cooperative）方式启动：grid=num_sms 个 block，每 block 512 线程
    jit->launch(
        kernel, {
            .stream = stream.stream(), // 目标 CUDA 流
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：1~2 个 block（SM）
            .block_dim = dim3(kNumThreads, 1, 1), // block 维度：512 线程
            .cooperative = true, // 协作启动：允许 grid 内全局同步
        },
        nccl_dev_comm, nccl_window, // 传入设备通信器与对称内存窗口
        signals, scaleout_rank_idx, scaleup_rank_idx // 信号缓冲区与本 rank 在两层域中的编号
    );
}

}  // namespace deep_ep::comm // 结束 comm 命名空间
