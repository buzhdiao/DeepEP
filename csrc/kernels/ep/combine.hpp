#pragma once // 头文件保护：只被包含一次

#include <algorithm> // std::min
#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板
#include <string> // std::string：拼接模板名

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API

#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）
#include <deep_ep/layout/ep/token.cuh> // EP token 布局（TokenLayout）

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架

namespace deep_ep::ep { // ep 命名空间：专家并行 kernel 封装

static layout::TokenLayout get_combine_token_layout( // 构造 combine 的 token 布局：只有隐藏数据（bf16）+ topk，无 scale-factor
    const int& hidden, const int& elem_size, const int& num_topk) { // 隐藏数据 = hidden × elem_size；scale-factor 为 0；关闭 padding
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false); // topk 数用于布局中的元数据区
}

// 启动 combine kernel：把分发到各专家的 token 结果加权归约回原 rank（纯 scale-up 走 NVLink，混合模式加跨机一级）
static void* launch_combine(void* x, // 输入 token 数据（bf16）
                            void* topk_weights, // 每 token 的 topk 权重
                            int* src_metadata, // 源元数据（token 来自哪个 rank/槽位）
                            int* psum_num_recv_tokens_per_scaleup_rank, // scale-up rank 接收 token 数前缀和
                            int* token_metadata_at_forward, // 转发阶段的 token 元数据（混合模式用）
                            int* channel_linked_list, // 通道链表（同通道 token 顺序）
                            const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window, // NCCL 设备通信器与对称内存窗口
                            void* buffer, void* workspace, // 对称数据缓冲区与 workspace（信号量）
                            const int& num_reduced_tokens, const int& num_max_tokens_per_rank, // 参与归约的 token 数；每 rank 最大 token 数
                            const int& hidden, // 隐藏维度
                            const int& num_experts, const int& num_topk, // 专家数与 topk 数
                            const int& num_qps, const int64_t& num_timeout_cycles, // QP 数；GPU 超时周期数
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks, // 两层逻辑域 rank 总数
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx, // 本 rank 在两层逻辑域中的编号
                            const bool& is_scaleup_nvlink, // scale-up 域是否走 NVLink
                            const int& num_sms, const int& num_smem_bytes, // SM 数；动态共享内存字节数
                            const int& num_channels, // 通道数
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction, // 是否展开布局；是否允许多次归约
                            const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    // Maximize shared memory utilization // 按共享内存容量最大化利用
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk); // combine 布局：bf16 隐藏数据 + topk 元数据
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32); // warp 数 = 共享内存可容纳 token 数与 32 的较小者

    // Decide warps // 确定 warp 分工
    int num_scaleup_warps = 0, num_forward_warps = 0; // scale-up（机内）与 forward（转发）warp 数
    if (num_scaleout_ranks > 1) { // 混合模式（有跨机）
        EP_HOST_ASSERT(num_channels % num_sms == 0 and // 通道数必须被 SM 数整除（与 dispatch 配置匹配）
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        EP_HOST_ASSERT(num_channels / num_sms <= 16); // 每 SM 通道数不得超过 16

        num_scaleup_warps = num_forward_warps = num_channels / num_sms; // scale-up 与 forward warp 数 = 每 SM 通道数
        num_warps = num_scaleup_warps + num_forward_warps; // block 内 warp 总数为两类 warp 之和
        EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and // 校验共享内存足够容纳这些 warp 的 token 缓冲
                       "Invalid combine SM count, please try to match your dispatch config");
    }

    const auto num_threads = num_warps * 32; // 总线程数 = warp 数 × 32

    // Compile // 编译：按模式实例化 combine 或 hybrid_combine 模板
    std::string header_name, func_name; // 头文件名与实例化函数名
    if (num_scaleout_ranks == 1) { // 纯 scale-up 分支
        header_name = "combine"; // 编译 combine 头
        func_name = std::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                is_scaleup_nvlink, // 是否 NVLink scale-up
                                use_expanded_layout, allow_multiple_reduction, // 是否展开布局；是否允许多次归约
                                num_sms, // SM 数
                                num_threads / 32, // block 内 warp 总数（纯模式不分两类）
                                num_scaleup_ranks * num_scaleout_ranks, // 总 rank 数
                                hidden, // 隐藏维度
                                num_max_tokens_per_rank, // 每 rank 最大 token 数
                                num_experts, // 专家数
                                num_topk, // topk 数
                                num_qps, num_timeout_cycles); // QP 数与超时周期数
    } else {
        header_name = "hybrid_combine"; // 编译 hybrid_combine 头
        func_name = std::format("hybrid_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                use_expanded_layout, allow_multiple_reduction, // 是否展开布局；是否允许多次归约
                                num_sms, // SM 数
                                num_scaleup_warps, num_forward_warps, // scale-up 与 forward warp 数
                                num_scaleout_ranks, num_scaleup_ranks, // 两层域 rank 数
                                hidden, // 隐藏维度
                                num_max_tokens_per_rank, // 每 rank 最大 token 数
                                num_experts, // 专家数
                                num_topk, // topk 数
                                num_qps, // QP 数
                                num_timeout_cycles); // 超时周期数
    }
    const auto kernel = jit->compile("combine", std::format(R"(
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
        .block_dim = dim3(num_threads, 1, 1), // block 维度：num_threads 线程
        .cluster_dim = dim3(2 - (num_sms % 2), 1, 1), // cluster 维度：num_sms 为偶数时取 2，否则取 1
        .cooperative = true, // 协作启动：支持 grid 内全局同步
    };
    if (num_scaleout_ranks == 1) {
        jit->launch( // 纯 scale-up 模式的启动参数
            kernel, options, // 提交 kernel 启动
            static_cast<nv_bfloat16*>(x), static_cast<float*>(topk_weights), // kernel 与启动配置
            src_metadata, psum_num_recv_tokens_per_scaleup_rank, // 输入 bf16 token 与 topk 权重
            nccl_dev_comm, nccl_window, // 源元数据与 scale-up 接收前缀和
            buffer, workspace, // 设备通信器与窗口
            scaleup_rank_idx, // 缓冲区与 workspace
            num_reduced_tokens // scale-up rank 编号
        ); // 参与归约的 token 数
    } else {
        jit->launch( // 混合模式的启动参数（多传转发元数据与两层编号）
            kernel, options, // 提交 kernel 启动
            static_cast<nv_bfloat16*>(x), static_cast<float*>(topk_weights), // kernel 与启动配置
            src_metadata, // 输入 bf16 token 与 topk 权重
            psum_num_recv_tokens_per_scaleup_rank, // 源元数据
            token_metadata_at_forward, // scale-up 接收前缀和
            channel_linked_list, // 转发阶段 token 元数据
            nccl_dev_comm, nccl_window, // 通道链表
            buffer, workspace, // 设备通信器与窗口
            scaleout_rank_idx, scaleup_rank_idx, // 缓冲区与 workspace
            num_reduced_tokens // 两层域 rank 编号
        ); // 参与归约的 token 数
    }

    // Return the buffer to be reduced
    if (num_scaleout_ranks == 1)
        return buffer;

    // For hybrid mode, we have to skip the scale-up buffer // 混合模式：跳过 scale-up 段，返回下一级归约缓冲区的末尾
    const bool is_scaleup_buffer_rank_layout = // 判断 scale-up 段是否按 rank 布局：允许多次归约且 scale-up rank 数 ≤ topk 时成立
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    const auto scaleup_buffer = layout::BufferLayout<false>( // 构造 scale-up 段的缓冲区布局
        token_layout, 
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk, // 按 rank 布局或按 topk 布局选择分区数
        num_scaleout_ranks * num_max_tokens_per_rank, // 总 token 容量 = 跨机 rank 数 × 每 rank 最大 token 数
        buffer);
    return scaleup_buffer.get_buffer_end_ptr(); // 返回 scale-up 段之后的缓冲区末尾指针
}

// 启动 combine 的归约收尾 kernel：把多级归约缓冲区合并成最终输出（与 combine kernel 用 PDL 重叠）
static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights, // 归约后的 topk 权重
                                           topk_idx_t* combined_topk_idx, // 归约后的 topk 索引
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank, // 归并 token 数；每 rank 最大 token 数
                                           const int& hidden, // 隐藏维度
                                           const int& num_experts, const int& num_topk, // 专家数与 topk 数
                                           void* reduce_buffer, // reduce 缓冲区（多级归约的中间结果）
                                           void* bias_0, void* bias_1, // 两级偏置（可加到输出上）
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks, // 两层逻辑域 rank 总数
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx, // 本 rank 在两层逻辑域中的编号
                                           const int& num_sms, const int& num_smem_bytes, // SM 数；动态共享内存字节数
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction, // 是否展开布局；是否允许多次归约
                                           const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024 // 按共享内存容量最大化利用
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false); // warp 太多可能劣化性能，限制到 1024 线程内
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32); // 此布局只关心隐藏数据大小
    const auto num_threads = num_warps * 32; // warp 数 = 共享内存可容纳 token 数与 32 的较小者
 // 总线程数 = warp 数 × 32
    // Compile
// 编译：实例化 combine_reduce_epilogue 模板（PDL 与前序 kernel 重叠）
    const auto kernel = jit->compile("combine_reduce_epilogue", std::format(R"(
#include <deep_ep/impls/ep/combine_reduce_epilogue.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", use_expanded_layout, allow_multiple_reduction,
        num_sms, num_threads / 32, // SM 数与 warp 数
        num_scaleout_ranks, num_scaleup_ranks, // 两层域 rank 总数
        hidden, // 隐藏维度
        num_max_tokens_per_rank, // 每 rank 最大 token 数
        num_experts, num_topk)); // 专家数与 topk 数

    // Launch // 启动：启用 PDL（Programmatic Dependent Launch）
    jit->launch( // 提交 kernel 启动
        kernel, { // 启动配置
            .stream = stream.stream(), // 目标 CUDA 流
            .num_smem_bytes = num_smem_bytes, // 动态共享内存大小
            .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
            .block_dim = dim3(num_threads, 1, 1), // block 维度：num_threads 线程
            .enable_pdl = true, // 启用 PDL：与前一 kernel 程序化依赖启动
        },
        static_cast<nv_bfloat16*>(combined_x), // 输出 bf16 token
        combined_topk_weights, // 归约后的 topk 权重
        combined_topk_idx, // 归约后的 topk 索引
        reduce_buffer, // reduce 中间缓冲区
        bias_0, bias_1, // 两级偏置
        num_combined_tokens, // 归并 token 数
        scaleout_rank_idx, scaleup_rank_idx // 两层域 rank 编号
    );
}

}  // namespace deep_ep::ep
