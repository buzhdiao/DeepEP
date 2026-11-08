#pragma once // 头文件保护：只被包含一次

#include <algorithm> // std::min
#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板
#include <string> // std::string：拼接模板名

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API
#include <nccl_device.h> // NCCL 设备端 API（ncclWindow_t 等）

#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）
#include <deep_ep/layout/ep/token.cuh> // EP token 布局（TokenLayout：隐藏数据 + scale-factor + topk 元数据）

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架

namespace deep_ep::ep { // ep 命名空间：专家并行 kernel 封装

constexpr int kNumNotifyWarps = 4; // 通知（notify）warp 数：4 个 warp 专职统计/通知元数据

static int get_num_notify_smem_bytes(const int& num_ranks, const int& num_experts) { // 计算 notify warp 需要的共享内存字节数
    return math::align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int); // 每 rank、每专家各一个计数器，按 4 warp × 32 线程对齐
}

static layout::TokenLayout get_dispatch_token_layout( // 构造 dispatch 的 token 布局：隐藏字节 + scale-factor 字节 + topk
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) { // 隐藏数据大小 = hidden × elem_size；scale-factor = num_sf_packs × 包大小；启用 padding
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
}

// 启动 dispatch kernel：把本机 token 按专家路由分发到远端 rank（纯 scale-up 走 NVLink，混合模式加跨机 RDMA 一级）
static void launch_dispatch(void* x, void* sf, // 通信上下文
                            topk_idx_t* topk_idx, float* topk_weights, // 输入 token 数据指针；scale-factor 指针
                            int* cumulative_local_expert_recv_stats, // 每 token 的专家索引与权重
                            int* psum_num_recv_tokens_per_scaleup_rank, // 本地各专家累计接收 token 数统计（nullptr 表示不统计）
                            int* psum_num_recv_tokens_per_expert, // 各 scale-up rank 接收 token 数的前缀和
                            int* num_unaligned_recv_tokens_per_expert, // 各专家接收 token 数的前缀和
                            int* dst_buffer_slot_idx, // 未对齐填充下各专家的接收 token 数
                            int* token_metadata_at_forward, // 每 token 在远端缓冲区中的目标槽位索引
                            const int& num_tokens, const int& num_max_tokens_per_rank, // 转发阶段使用的 token 元数据（混合模式用）
                            const int& hidden, const int& elem_size, // 本批 token 数；每 rank 最大 token 数
                            const int& num_sf_packs, const int& sf_token_stride, const int& sf_hidden_stride, // 隐藏维度；单元素字节数
                            const int& num_experts, const int& num_topk, const int& expert_alignment, // scale-factor 包数及其 token/hidden 维步长
                            const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window, // 专家数；topk 数；专家对齐粒度
                            void* buffer, // NCCL 设备通信器与对称内存窗口
                            void* workspace, void* mapped_host_workspace, // 对称数据缓冲区（跨 rank 交换 token 用）
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx, // workspace（信号量）与映射到主机内存的 workspace
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks, // 本 rank 在两层逻辑域中的编号
                            const bool& is_scaleup_nvlink, // 两层逻辑域的 rank 总数
                            const int& num_sms, const int& num_channels_per_sm, // scale-up 域是否走 NVLink
                            const int& num_smem_bytes, // 占用 SM 数；每 SM 的通道数
                            const int& num_qps, const int64_t& num_timeout_cycles, // 动态共享内存字节数
                            const bool& cached_mode, // QP 数；GPU 超时周期数
                            const bool& do_cpu_sync, // 缓存模式：跳过元数据统计，复用上次槽位索引
                            const at::cuda::CUDAStream& stream) { // 是否与 CPU 做同步（GPU-CPU 协同计数）
    // Cached mode does not support expert token counting // 启动 kernel 的 CUDA 流
    if (cached_mode) // 缓存模式不支持专家 token 计数
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats == nullptr); // 缓存模式下必须关闭接收统计

    // Utils
    const auto num_ranks = num_scaleout_ranks * num_scaleup_ranks; // 总 rank 数 = 跨机 × 机内

    // Notify warps // notify warp 配置
    // TODO: why don't we use 4 notify warps? // （原注释：为何不用 4 个 notify warp）
    const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps; // 非缓存模式启用 4 个 notify warp，缓存模式为 0
    const bool reuse_slot_indices = cached_mode; // 缓存模式复用上次的槽位索引
    const int num_notify_smem_bytes = cached_mode ? 0 : get_num_notify_smem_bytes(num_ranks, num_experts); // notify 共享内存：每 rank/专家一个 int 计数器
    EP_HOST_ASSERT(num_notify_warps % 4 == 0); // notify warp 数必须是 4 的倍数

    // Other warps // 其余 warp 的配置
    int num_dispatch_warps = 0; // dispatch warp 数：负责 token 打包与 RDMA 发送
    int num_scaleout_warps = 0, num_forward_warps = 0; // 混合模式：scale-out（跨机）与 forward（转发）warp 数
    int num_threads = 0; // 总线程数

    // Maximize shared memory utilization // 按共享内存容量最大化利用
    if (num_scaleout_ranks == 1) { // 纯 scale-up 模式（无跨机）
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk); // token 布局决定每个 token 占用的字节数
        num_dispatch_warps = std::min<int>( // dispatch warp 数 = 共享内存能容纳的 token 数与 32-notify 的较小者
            (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(), 32 - num_notify_warps); // get_num_bytes<true> 含填充后的每 token 字节数
        num_threads = (num_notify_warps + num_dispatch_warps) * 32; // 总线程数 =（notify + dispatch）warp 数 × 32
    } else {
        // Hybrid kernels // 混合 kernel：跨机 + 机内两级分发
        num_scaleout_warps = num_channels_per_sm; // scale-out warp 数 = 每 SM 通道数（每通道负责一组跨机 rank）
        num_forward_warps = num_channels_per_sm; // forward warp 数 = 每 SM 通道数（负责机内转发）
        num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32; // 总线程数 =（notify + scaleout + forward）warp 数 × 32
    }

    // Compile // 编译：按模式实例化 dispatch 或 hybrid_dispatch 模板
    std::string header_name, func_name; // 头文件名与实例化函数名
    if (num_scaleout_ranks == 1) { // 纯 scale-up 分支
        header_name = "dispatch"; // 编译 dispatch 头
        func_name = std::format("dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
            is_scaleup_nvlink, // 是否 NVLink scale-up
            do_cpu_sync, // 是否 CPU 同步
            reuse_slot_indices, // 是否复用槽位索引（缓存模式）
            num_sms, // SM 数
            num_notify_warps, num_dispatch_warps, // notify 与 dispatch warp 数
            num_scaleup_ranks, // scale-up 域 rank 数
            hidden * elem_size, num_sf_packs, // 每 token 隐藏字节数；scale-factor 包数
            num_max_tokens_per_rank, // 每 rank 最大 token 数
            num_experts, num_topk, expert_alignment, // 专家数、topk 数、专家对齐粒度
            num_qps, num_timeout_cycles); // QP 数与超时周期数
    } else {
        header_name = "hybrid_dispatch"; // 编译 hybrid_dispatch 头
        func_name = std::format("hybrid_dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
            do_cpu_sync, // 是否 CPU 同步
            reuse_slot_indices, // 是否复用槽位索引
            num_sms, // SM 数
            num_notify_warps, num_scaleout_warps, num_forward_warps, // notify、scale-out、forward warp 数
            num_scaleout_ranks, num_scaleup_ranks, // 两层域 rank 数
            hidden * elem_size, num_sf_packs, // 每 token 隐藏字节数；scale-factor 包数
            num_max_tokens_per_rank, // 每 rank 最大 token 数
            num_experts, num_topk, expert_alignment, // 专家数、topk 数、专家对齐粒度
            num_qps, num_timeout_cycles); // QP 数与超时周期数
    }
    const auto kernel = jit->compile("dispatch", std::format(R"(
#include <deep_ep/impls/ep/{}.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::{});
}}
)", header_name, func_name));

    // Launch // 协作式启动：grid=num_sms 个 block，cluster 视奇偶取 1 或 2
    const auto options = deep_jit::cuda::LaunchOptions { // 启动配置
        .stream = stream.stream(), // 目标 CUDA 流
        .num_smem_bytes = num_smem_bytes, // 动态共享内存大小
        .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
        .block_dim = dim3(num_threads, 1, 1), // block 维度：按模式配置的各 warp
        .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
        .cooperative = true, // 协作启动：支持 grid 内全局同步
    };
    if (num_scaleout_ranks == 1) { // 纯 scale-up 模式的启动参数
        jit->launch( // 提交 kernel 启动
            kernel, options, // kernel 与启动配置
            x, static_cast<sf_pack_t*>(sf), topk_idx, topk_weights, // 输入 token、scale-factor、topk 索引与权重
            cumulative_local_expert_recv_stats, // 本地专家接收统计
            psum_num_recv_tokens_per_scaleup_rank, // scale-up rank 接收前缀和
            psum_num_recv_tokens_per_expert, // 专家接收前缀和
            num_unaligned_recv_tokens_per_expert, // 未对齐接收统计
            dst_buffer_slot_idx, // 目标槽位索引
            num_tokens, // token 数
            sf_token_stride, sf_hidden_stride, // scale-factor 步长
            nccl_dev_comm, nccl_window, // 设备通信器与窗口
            buffer, // 对称缓冲区
            workspace, mapped_host_workspace, // workspace 与主机映射 workspace
            scaleup_rank_idx // scale-up rank 编号
        );
    } else { // 混合模式的启动参数（多传 token 元数据与两层 rank 编号）
        jit->launch( // 提交 kernel 启动
            kernel, options, // kernel 与启动配置
            x, static_cast<sf_pack_t*>(sf), topk_idx, topk_weights, // 输入 token、scale-factor、topk 索引与权重
            cumulative_local_expert_recv_stats, // 本地专家接收统计
            psum_num_recv_tokens_per_scaleup_rank, // scale-up rank 接收前缀和
            psum_num_recv_tokens_per_expert, // 专家接收前缀和
            num_unaligned_recv_tokens_per_expert, // 未对齐接收统计
            dst_buffer_slot_idx, // 目标槽位索引
            token_metadata_at_forward, // 转发阶段 token 元数据
            num_tokens, // token 数
            sf_token_stride, sf_hidden_stride, // scale-factor 步长
            nccl_dev_comm, nccl_window, // 设备通信器与窗口
            buffer, // 对称缓冲区
            workspace, mapped_host_workspace, // workspace 与主机映射 workspace
            scaleout_rank_idx, scaleup_rank_idx // 两层域 rank 编号
        );
    }
}

// 启动 dispatch 的拷贝收尾 kernel：从缓冲区把收到的 token 整理成接收布局（与分发 kernel 用 PDL 重叠）
static void launch_dispatch_copy_epilogue(void* buffer, void* workspace, // workspace（信号量）与数据缓冲区
                                          int* psum_num_recv_tokens_per_scaleup_rank, // scale-up rank 接收前缀和
                                          int* psum_num_recv_tokens_per_expert, // 专家接收前缀和
                                          void* recv_x, void* recv_sf, // 接收 token 数据与 scale-factor
                                          topk_idx_t* recv_topk_idx, float* recv_topk_weights, // 接收 token 的 topk 索引与权重
                                          int* recv_src_metadata, // 接收 token 的源元数据
                                          int* channel_linked_list, // 通道链表（同通道 token 的串接顺序）
                                          int* num_unaligned_recv_tokens_per_expert, // 未对齐接收统计
                                          const int& num_recv_tokens, const int& num_max_tokens_per_rank, // 接收 token 数；每 rank 最大 token 数
                                          const int& num_hidden_bytes, // 隐藏数据字节数
                                          const int& num_sf_packs, const int& recv_sf_token_stride, const int& recv_sf_hidden_stride, // scale-factor 包数及其接收侧步长
                                          const int& num_experts, const int& num_topk, const int& expert_alignment, // 专家数、topk 数、专家对齐粒度
                                          const int& scaleout_rank_idx, const int& scaleup_rank_idx, // 两层域 rank 编号
                                          const int& num_scaleout_ranks, const int& num_scaleup_ranks, // 两层域 rank 总数
                                          const int& num_sms, const int& num_smem_bytes, // SM 数；动态共享内存字节数
                                          const int& num_channels, // 通道数
                                          const bool& do_expand, const bool& cached_mode, // 是否展开布局；是否缓存模式
                                          const bool& do_zero_padding, // 是否做零填充
                                          const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    // Maximize shared memory utilization
    const auto token_layout = layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true); // 按共享内存容量最大化利用
    const auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32); // 每 token 字节数决定 warp 数上限
    const auto num_threads = num_warps * 32; // warp 数 = 共享内存可容纳 token 数与 32 的较小者
 // 总线程数 = warp 数 × 32
    // Compile
// 编译：实例化 dispatch_copy_epilogue 模板（PDL 依赖上一个 kernel）
    const auto kernel = jit->compile("dispatch_copy_epilogue", std::format(R"(
#include <deep_ep/impls/ep/dispatch_copy_epilogue.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", do_expand, cached_mode, do_zero_padding,
        num_sms, num_channels, num_warps, // SM 数、通道数、warp 数
        num_scaleout_ranks, num_scaleup_ranks, // 两层域 rank 总数
        num_hidden_bytes, num_sf_packs, // 隐藏字节数；scale-factor 包数
        num_max_tokens_per_rank, // 每 rank 最大 token 数
        num_experts, num_topk, expert_alignment)); // 专家数、topk 数、专家对齐粒度

    // Launch // 启动：启用 PDL（Programmatic Dependent Launch）与 dispatch kernel 重叠
    jit->launch( // 提交 kernel 启动
        kernel, { // 启动配置
            .stream = stream.stream(), // 目标 CUDA 流
            .num_smem_bytes = num_smem_bytes, // 动态共享内存大小
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(num_threads, 1, 1), // block 维度：num_warps 个 warp
            .enable_pdl = true, // 启用 PDL：与前一 kernel 程序化依赖启动
        },
        buffer, workspace, // 缓冲区与 workspace
        psum_num_recv_tokens_per_scaleup_rank, // scale-up rank 接收前缀和
        psum_num_recv_tokens_per_expert, // 专家接收前缀和
        recv_x, recv_sf, recv_topk_idx, recv_topk_weights, // 接收数据、scale-factor、topk 索引与权重
        recv_src_metadata, // 源元数据
        channel_linked_list, // 通道链表
        num_unaligned_recv_tokens_per_expert, // 未对齐接收统计
        num_recv_tokens, // 接收 token 数
        recv_sf_token_stride, recv_sf_hidden_stride, // scale-factor 接收步长
        scaleout_rank_idx, scaleup_rank_idx // 两层域 rank 编号
    );
}

}  // namespace deep_ep::ep
