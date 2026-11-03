#pragma once // 头文件保护：只被包含一次

#include <cstddef> // std::size_t
#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板与函数名
#include <string> // std::string
#include <vector> // std::vector

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API
#include <nccl_device.h> // NCCL 设备端 API（ncclWindow_t、ncclGinRequest_t 等）

#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架

// engram 命名空间：KV/激活缓存（engram）远端拉取 kernel 封装
namespace deep_ep::engram {

// 启动 engram 拉取 kernel：按层按条目从远端 rank 经 GIN（RDMA）拉取数据到本地
static void launch_engram_fetch(const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window, // NCCL 设备通信器；主对称内存窗口
                                ncclWindow_t* storage_windows, void* fetched, void* workspace, // RDMA 存储分片窗口数组；拉取结果缓冲；workspace
                                int* indices, // 条目索引（要拉取哪些条目）
                                ncclGinRequest_t* last_gin_requests, // 最近一次 GIN 请求句柄（用于后续 wait）
                                void* sf_shards, void* fetched_sf, // scale-factor 分片源；拉取到的 scale-factor
                                const int& sf_token_stride, const int& sf_hidden_stride, // scale-factor 的 token/hidden 维步长
                                const int& sf_layer_stride, // scale-factor 的 layer 维步长
                                const int& num_sf_shards, // scale-factor 分片数
                                const std::vector<int>& num_entries_per_layer, // 每层的条目数（编译为模板参数列表）
                                const int& hidden, const int& elem_size, const int& num_sf_packs, // 隐藏维度；元素字节数；scale-factor 包数
                                const int& num_entries_per_token, // 每 token 的条目数
                                const int& num_tokens, const int& num_max_tokens, // 本批 token 数；最大 token 数
                                const int& num_rdma_peers, const int& num_ranks_per_rdma_peer, // RDMA 对端数；每对端的 rank 数
                                const int& num_qps, const int& flush_depth, // QP 数；flush 深度（RDMA 写回流水线深度）
                                const int64_t& num_timeout_cycles, // GPU 侧自旋等待超时（周期数）
                                const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    constexpr int kNumSFWarpsPerSM = 4; // 每个用于 scale-factor 拷贝的 warp 数上限
    const auto num_sms = jit->device.get_num_sms(); // 占满整个 GPU 的 SM 数

    const auto num_sf_warps_per_sm = num_sf_packs > 0 ? kNumSFWarpsPerSM : 0; // 有 scale-factor 包才分配 SF warp
    const auto num_issue_warps_per_sm = 32 - num_sf_warps_per_sm; // 其余 warp 全部用于发起 RDMA 拉取（issue warp）

    const auto num_threads = (num_sf_warps_per_sm + num_issue_warps_per_sm) * 32; // 每 block 线程数 = warp 数 × 32
    EP_HOST_ASSERT(flush_depth > 0); // flush 深度必须为正
    EP_HOST_ASSERT(sf_hidden_stride != 1); // scale-factor 的 hidden 步长不能为 1（否则布局不支持）
    EP_HOST_ASSERT(num_sms * num_issue_warps_per_sm >= num_rdma_peers * num_qps); // issue warp 必须覆盖所有 对端×QP 组合

    std::string num_entries_per_layer_string; // 把每层条目数拼成逗号分隔串，作为模板参数
    num_entries_per_layer_string.reserve(num_entries_per_layer.size() * 12); // 预留空间避免反复分配
    for (const auto num_entries : num_entries_per_layer) { // 遍历每层的条目数
        if (not num_entries_per_layer_string.empty()) // 已有内容则先加分隔符
            num_entries_per_layer_string += ", "; // 逗号分隔
        num_entries_per_layer_string += std::to_string(num_entries); // 追加本层条目数
    }
    const auto func_name = std::format( // 拼出完整的实例化函数名（含 EngramLayout 模板参数）
        "engram_fetch_impl<{}, {}, {}, {}, deep_ep::layout::EngramLayout<{}, {}, {}, {}, {}>, {}, {}, {}, {}, {}, {}>",
        num_sms, num_qps, num_sf_warps_per_sm, num_issue_warps_per_sm, // SM 数、QP 数、SF warp 数、issue warp 数
        num_rdma_peers * num_ranks_per_rdma_peer, num_sf_shards, // 远端总 rank 数；scale-factor 分片数
        hidden * elem_size, num_entries_per_token, // 每 token 隐藏字节数；每 token 条目数
        num_entries_per_layer_string, // 每层条目数列表
        num_sf_packs, num_max_tokens, // scale-factor 包数；最大 token 数
        num_rdma_peers, num_ranks_per_rdma_peer, // RDMA 对端数；每对端 rank 数
        flush_depth, num_timeout_cycles); // flush 深度；超时周期数

    // Compile // JIT 编译：实例化 engram_fetch_impl
    const auto kernel = jit->compile("engram_fetch", std::format(R"(
#include <deep_ep/impls/engram/engram_fetch.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::engram::{});
}}
)", func_name));

    // Launch // 启动：grid=num_sms 个 block，每 block 含 SF warp 与 issue warp
    jit->launch(
        kernel, {
            .stream = stream.stream(), // 目标 CUDA 流
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(num_threads, 1, 1), // block 维度：SF + issue 两类 warp
        },
        nccl_dev_comm, nccl_window, // 设备通信器与主窗口
        storage_windows, fetched, workspace, // 存储分片窗口、结果缓冲、workspace
        indices, // 条目索引
        last_gin_requests, // GIN 请求句柄（供 fetch_wait 使用）
        static_cast<sf_pack_t*>(sf_shards), static_cast<sf_pack_t*>(fetched_sf), // scale-factor 源与目的
        sf_token_stride, sf_hidden_stride, sf_layer_stride, // scale-factor 三维步长
        num_tokens // 本批 token 数
    );
}

// 启动 engram 拉取的等待 kernel：轮询各 GIN 上下文的请求完成状态，确保全部拉取落地
static void launch_engram_fetch_wait(ncclGinRequest_t* last_gin_requests, // 上一阶段记录的 GIN 请求句柄
                                     const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window, // 设备通信器；主窗口
                                     const int& num_rdma_peers, // RDMA 对端数
                                     const int& num_active_gin_contexts, // 活跃的 GIN 上下文（QP 组）数
                                     const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    constexpr int kNumEngramFetchWaitThreads = 1024; // 单 block 1024 线程：每线程负责轮询一组请求

    // Compile // JIT 编译：实例化 engram_fetch_wait_impl
    const auto func_name = std::format(
        "engram_fetch_wait_impl<{}, {}>", num_rdma_peers, kNumEngramFetchWaitThreads); // 模板参数：对端数、线程数
    const auto kernel = jit->compile("engram_fetch_wait", std::format(R"(
#include <deep_ep/impls/engram/engram_fetch_wait.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::engram::{});
}}
)", func_name));

    // Launch // 启动：单 block、1024 线程，专门用于等待 RDMA 完成信号
    jit->launch(
        kernel, {
            .stream = stream.stream(), // 目标 CUDA 流
            .grid_dim = dim3(1, 1, 1), // grid 维度：1 个 block
            .block_dim = dim3(kNumEngramFetchWaitThreads, 1, 1), // block 维度：1024 线程
        },
        nccl_dev_comm, nccl_window, // 设备通信器与主窗口
        last_gin_requests, // 要等待的 GIN 请求句柄
        num_active_gin_contexts // 活跃 GIN 上下文数
    );
}

}  // namespace deep_ep::engram // 结束 engram 命名空间
