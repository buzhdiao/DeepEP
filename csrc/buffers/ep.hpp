#pragma once

// 引入 c10 的整数累乘工具 multiply_integers
#include <c10/util/accumulate.h>
// 引入 CUDA 运行时 API（锁页内存分配等）
#include <cuda_runtime.h>
// 引入 std::shared_ptr / make_shared
#include <memory>
// 引入 std::vector
#include <vector>
// 引入 pybind11 的 std::function 支持（延迟 epilogue 钩子用）
#include <pybind11/functional.h>

// 引入 DeepEP 公共编译常量与内核声明
#include <deep_ep/common/compiled.cuh>
// 引入 EP token 布局定义
#include <deep_ep/layout/ep/token.cuh>
// 引入 EP workspace（信号区）布局定义
#include <deep_ep/layout/ep/workspace.cuh>
// 引入 EPLB（专家负载均衡）布局定义
#include <deep_ep/layout/ep/eplb.cuh>

// 引入缓冲区基类 BufferBase
#include "base.hpp"
// 引入 NCCL 通信上下文与公共通信 API
#include "../comm/api.hpp"
// 引入 EP 内核（dispatch/combine 等）的启动封装
#include "../kernels/ep/api.hpp"
// 引入 JIT 运行时（查询设备 SM/shared memory 等）
#include "../runtime/jit.hpp"
// 引入事件句柄
#include "../utils/event.hpp"
// 引入张量工具（形状解包等）
#include "../utils/tensor.hpp"

// EP（专家并行）缓冲区所在的命名空间
namespace deep_ep::ep {

// EP 缓冲区：实现 MoE 的 dispatch（分发）与 combine（合并）核心通信逻辑
class EPBuffer: public BufferBase {
    // Buffer bytes exclude workspace
    // Memory layout: [Workspace, buffer]
    // Buffer 的字节数不包含 workspace
    // 内存布局: [Workspace, buffer]
    // 通信 buffer 的字节数（不含 workspace）
    int64_t num_buffer_bytes;

    // Host workspace
    // 宿主机端 workspace（锁页内存）及其映射到 GPU 的地址
    void *host_workspace, *mapped_host_workspace;

    // Whether to use hybrid mode (scale-out with scale-up)
    // 是否启用混合模式（跨机 scale-out 与机内 scale-up 相结合）
    bool allow_hybrid_mode;

    // Whether to allow multiple reductions
    // 是否允许多次归约（影响 combine 缓冲区的布局与行为）
    bool allow_multiple_reduction;

    // Whether to prefer overlapping communication with compute (use more SMs and channels if false)
    // 是否优先与计算重叠通信（若为 false 则占用更多 SM 与通道）
    bool prefer_overlap_with_compute;

    // Some EP hybrid mode settings
    // EP 混合模式的一些常量设置
    // 每个 SM 支持的最大通道数
    static constexpr int kNumMaxChannelsPerSM = 8;
    // 可使用的最大 SM 数
    static constexpr int kNumMaxSMs = 160;
    // 全局最大通道数 = 每 SM 通道数 x SM 数
    static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;

public:
    // NCCL 通信上下文
    std::shared_ptr<comm::Context> context;

    // For load balance
    // 用于负载均衡（EPLB）的存储张量（buffer 中划出的 LB 区域）
    torch::Tensor lb_storage;

    // 构造函数：创建通信上下文、锁页 host workspace 与 LB 存储区
    EPBuffer(const int& rank_idx, const int& num_ranks,
             const int64_t& nccl_comm,
             const int64_t& num_buffer_bytes,
             const int64_t& num_lb_buffer_bytes,
             const bool& allow_hybrid_mode,
             const bool& allow_multiple_reduction,
             const bool& prefer_overlap_with_compute,
             const std::optional<int>& sl_idx, const int& num_allocated_qps,
             const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
             const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),
        num_buffer_bytes(num_buffer_bytes),
        allow_hybrid_mode(allow_hybrid_mode),
        allow_multiple_reduction(allow_multiple_reduction),
        prefer_overlap_with_compute(prefer_overlap_with_compute) {
        // 断言 buffer 大小为正且按分配对齐
        EP_HOST_ASSERT(num_buffer_bytes > 0 and num_buffer_bytes % kNumAllocationAlignmentBytes == 0);
        // 断言 LB buffer 大小非负且按分配对齐
        EP_HOST_ASSERT(num_lb_buffer_bytes >= 0 and num_lb_buffer_bytes % kNumAllocationAlignmentBytes == 0);

        // Workspace is aligned to 2 MB so that it sits cleanly at the front of the GPU segment
        // Workspace 按 2MB 对齐，使其干净地位于 GPU 段的起始位置
        // 计算 workspace 大小并按分配对齐
        const auto num_workspace_bytes = math::align<int64_t>(
            layout::EPWorkspaceLayout::get_num_bytes(), kNumAllocationAlignmentBytes);

        // 创建 NCCL 通信上下文：workspace 与 buffer 大小、混合模式、QP 与超时配置
        context = std::make_shared<comm::Context>(
            nccl_comm, symmetric::shared_comm_t{}, num_ranks, rank_idx,
            num_workspace_bytes, num_buffer_bytes + num_lb_buffer_bytes, 0, true,
            allow_hybrid_mode, sl_idx, num_allocated_qps,
            0, num_cpu_timeout_secs, num_gpu_timeout_secs);
        // 设置基类中的主上下文
        main_context = context;
        // 获取 workspace（信号区）的引用
        auto& workspace = *static_cast<layout::EPSignals*>(context->workspace);
        // 注册屏障信号区，供同步使用
        context->set_barrier_signals(&workspace.barrier_signals);

        // Expose only the LB region; the preceding bytes remain reserved for EP communication.
        // 只把 LB 区域暴露为张量；前面的字节保留给 EP 通信使用。
        // 以通信 buffer 为后端创建字节张量（空删除器，生命周期由 context 管理）
        lb_storage = torch::from_blob(
            context->buffer, {num_buffer_bytes + num_lb_buffer_bytes}, [context = context](void*) {},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        ).narrow(0, num_buffer_bytes, num_lb_buffer_bytes);

        // Allocate host workspaces
        // 分配锁页 host workspace
        // 分配可映射的锁页内存
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::EPWorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));
        // 获取该锁页内存映射到 GPU 的设备指针
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
        // 将 host workspace 清零
        std::memset(host_workspace, 0, layout::EPWorkspaceLayout::get_num_bytes());

        // We should call a barrier at the end
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
        // 最后应该调用一次 barrier
        // 这个 barrier 应当由 Python 的 `dist.barrier` 来调用
        // 注意：这里不要调用我们自己的 barrier，因为 workspace 还没有就绪
    }

    // 析构函数：按销毁策略处理资源释放
    ~EPBuffer() noexcept(false) override {
        // 基类辅助：必要时自动调用 destroy()，否则打印泄漏警告
        destroy_on_destruction("EP");
    }

    // 销毁缓冲区：屏障同步后释放所有资源
    void destroy() override {
        // 断言尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // Finish all works on all GPUs
        // 等待所有 GPU 上的工作完成
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);

        // Deallocate host workspaces
        // 释放锁页 host workspace
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));

        // Destroy NCCL context
        // 结束并释放 NCCL 通信上下文
        context->finalize();

        // Destroy load-balance storage
        // 释放负载均衡存储张量
        lb_storage = torch::Tensor();

        // Cannot use anymore
        // 之后不能再使用
        destroyed = true;
    }

    // 流控制前奏（简单版）：获取计算流与通信流，并让通信流等待前置事件或计算流
    static torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event) {
        // 获取当前计算流
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        // 获取全局唯一的通信流
        const auto comm_stream = comm::get_comm_stream();

        // Wait previous tasks to finish
        // 等待之前的任务完成
        if (previous_event.has_value()) {
            // 有前置事件：让通信流等待该事件
            comm::stream_wait(comm_stream, previous_event.value());
        } else {
            // 无前置事件：让通信流等待计算流
            comm::stream_wait(comm_stream, compute_stream);
        }
        // 返回计算流（供收尾时切回）
        return compute_stream;
    }

    // 流控制前奏（完整版）：额外决定是否把当前流切换为通信流用于分配张量
    static torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                           const bool& allocate_on_comm_stream) {
        // Assertion for safety
        // `previous_event` implicitly means the overlapping computation kernels are launched first,
        // in order not to use the memory on the compute stream, we must allocate on the communication stream.
        // If you launch the communication kernels firstly, then `previous_event` must be unnecessary.
        // 安全断言
        // 传入 `previous_event` 隐含表示重叠的计算内核先被启动，
        // 为了不占用计算流上的内存，必须在通信流上分配张量。
        // 如果先启动通信内核，则 `previous_event` 必然是不需要的。
        if (previous_event.has_value())
            EP_HOST_ASSERT(allocate_on_comm_stream);

        // Allocate all tensors on communication stream if set
        // NOTES: do not allocate tensors upfront!
        // 若设置了在通信流上分配，则之后所有张量分配都发生在通信流上
        // 注意：不要提前分配张量！
        const auto compute_stream = stream_control_prologue(previous_event);
        if (allocate_on_comm_stream)
            // 把当前线程的 CUDA 流切换为通信流
            at::cuda::setCurrentCUDAStream(comm::get_comm_stream());
        // 返回计算流（供收尾时切回）
        return compute_stream;
    }

    // 流控制收尾：登记张量、返回事件，并在需要时切回计算流
    static std::optional<EventHandle> stream_control_epilogue(const tensor_list_t& tensors,
                                                              const at::cuda::CUDAStream& compute_stream,
                                                              const bool& allocate_on_comm_stream,
                                                              const bool& async_with_compute_stream) {
        // Ensure memory access safety between two streams
        // 保证两条流之间的内存访问安全
        const auto comm_stream = comm::get_comm_stream();
        // 待返回的事件句柄
        std::optional<EventHandle> event;
        if (async_with_compute_stream) {
            // 异步模式：在通信流上记录事件
            event = EventHandle(comm_stream);

            // NOTES: this environment only applies to V2 APIs
            // 注意：此环境变量仅作用于 V2 API
            if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
                // 延迟登记：把张量列表挂到事件上由 Python 侧处理
                event->tensors_to_record = tensors;
            } else {
                // 默认路径：对每个存在的张量在计算流与通信流上都做 record_stream
                for (auto& t: tensors) if (t.has_value()) {
                    t->record_stream(compute_stream);
                    t->record_stream(comm_stream);
                }
            }
        } else {
            // 同步模式：让计算流等待通信流完成
            comm::stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        // 切回计算流
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // The CUDA event marking the finishing
        // 返回标志通信完成与否的 CUDA 事件
        return event;
    }

    // 计算 dispatch 所需的通信 buffer 字节数（直接模式与混合模式分别布局）
    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& hidden, const int& num_sf_packs, const int& num_topk,
                                            const int& elem_size,
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                            const bool& is_scaleup_nvlink) {
        // 总 rank 数 = scale-up 与 scale-out 两个维度的乘积
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        // 获取 dispatch 的 token 布局（含 SF 与 top-k 信息）
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct dispatch
            // 直接 dispatch（单 scale-out 域）
            // 发送侧布局：NVLink 直连时无需远端槽位（0），否则按 1 个对端
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
            // 接收侧布局：来自所有 rank 的 token
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_ranks, num_max_tokens_per_rank);
            // 返回发送与接收布局字节数之和
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid dispatch
            // 混合 dispatch（跨机 + 机内两级）
            // scale-up 接收缓冲：来自各 scale-up rank 的 token
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
            // scale-out 发送缓冲：只发给单个跨机对端
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, 1, num_max_tokens_per_rank);
            // scale-out 接收缓冲：额外预留按通道划分的容量
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleout_ranks,
                /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);
            // 返回三部分布局字节数之和
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    // 计算 combine 所需的通信 buffer 字节数（直接模式与混合模式分别布局）
    static int64_t get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const bool& is_scaleup_nvlink,
                                           const bool& allow_multiple_reduction) {
        // 总 rank 数 = scale-up 与 scale-out 两个维度的乘积
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        // 获取 combine 的 token 布局（BF16，含 top-k）
        const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct combine
            // 直接 combine（单 scale-out 域）
            // 多次归约时布局内的 token 数取 rank 数与 top-k 的较小值，否则为 top-k
            const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;
            // 发送侧布局：NVLink 直连无需槽位，否则按全部 rank
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : num_ranks,
                // For single reduction cases, the maximum number of received tokens is
                // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
                // 单次归约情况下，接收 token 的最大数量为
                // `num_ranks * num_topk * num_max_tokens_per_rank`（我们假设 `do_expand=True` 的坏情况）
                num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
            // 接收侧布局
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
            // 返回发送与接收布局字节数之和
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid combine
            // 混合 combine（跨机 + 机内两级）
            // scale-up 布局内的 token 数
            const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
            // scale-out 布局内的 token 数
            const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
            // scale-up 接收缓冲
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
            // scale-out 接收缓冲
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
            // scale-out 发送缓冲：额外预留按通道划分的容量
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, allow_multiple_reduction ? 1 : num_topk,
                /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
                num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));
            // 返回三部分布局字节数之和
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    // 静态计算 EP buffer 的实际分配大小：取 dispatch 与 combine 需求的较大值并按 2MB 对齐
    static int64_t calculate_buffer_size(const int64_t& nccl_comm,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch,
                                         const bool& allow_hybrid_mode,
                                         const bool& allow_multiple_reduction) {
        // 断言每 rank 最大 token 数与隐藏维均为正
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

        // The worst case SF bytes must be less than the main part
        // 最坏情况的 SF 字节数必须小于主体部分
        EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

        // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
        // 注意：代码中存在大量 `kNumTopk <= 32` 的限制，因此统一用 32 来计算 token 大小
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Topology
        // 拓扑信息
        // 获取物理域大小：RDMA rank 数与 NVLink rank 数
        const auto [num_rdma_ranks, num_nvl_ranks] = comm::get_physical_domain_size(nccl_comm);
        // 获取逻辑域大小：scale-out 与 scale-up rank 数
        const auto [num_scaleout_ranks, num_scaleup_ranks] = comm::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
        // 判断 scale-up 方向是否为 NVLink 直连
        const auto is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

        // Dispatch size
        // Dispatch 所需大小
        // 元素大小：FP8 dispatch 用 1 字节，否则 BF16 用 2 字节
        const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        // SF 包数：FP8 时按每 32 隐藏维一包估算（近似值）
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0; // An approximation for number of SF packs // SF 包数的近似值
        // 计算 dispatch 布局字节数
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink);

        // Combine layout
        // Combine 布局
        // 计算 combine 布局字节数
        const auto num_combine_bytes = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink, allow_multiple_reduction);

        // Return the maximum of those layouts, aligned to 2 MB
        // 返回两者中的较大值，并按 2MB 对齐
        return math::align<int64_t>(std::max(num_dispatch_bytes, num_combine_bytes), kNumAllocationAlignmentBytes);
    }

    // MoE dispatch：把 token 按专家索引分发到对应 rank；返回接收结果与元信息
    pybind11::tuple
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const std::optional<torch::Tensor>& cached_channel_linked_list,
             const int& num_max_tokens_per_rank,
             const int& num_experts, const int& expert_alignment,
             const int& num_sms, const int& num_qps,
             const std::optional<EventHandle>& previous_event,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf,
             const bool& defer_epilogue) const {
        // Check SM count
        // 检查 SM 数
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        // 混合 dispatch 至少需要 2 个 SM
        EP_HOST_ASSERT((num_sms > 1 or context->num_scaleout_ranks == 1 or context->num_scaleup_ranks == 1) and
                       "Hybrid dispatch requires at least 2 SMs");

        // Zero padding only makes sense with expand mode
        // 零填充只在 expand 模式下有意义
        EP_HOST_ASSERT(not do_zero_padding or do_expand);

        // Cached mode must have responding handles
        // 缓存模式必须提供配套的句柄
        // 是否为缓存模式：由是否提供缓存的接收 token 数判断
        const bool cached_mode = cached_num_recv_tokens.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(cached_num_recv_tokens.has_value());
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
            EP_HOST_ASSERT(cached_num_expanded_tokens.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_scaleup_rank.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());
            EP_HOST_ASSERT(cached_recv_src_metadata.has_value());
            // 断言源元数据张量在 GPU 上且连续
            EP_HOST_ASSERT(cached_recv_src_metadata->is_cuda() and cached_recv_src_metadata->is_contiguous());

            // Hybrid kernels require more
            // 混合内核还需要额外的元数据
            if (context->num_scaleout_ranks > 1) {
                EP_HOST_ASSERT(cached_token_metadata_at_forward.has_value());
                EP_HOST_ASSERT(cached_channel_linked_list.has_value());
            }
        }

        // Check data tensor
        // 检查数据张量
        // 解包输入形状：token 数与隐藏维
        const auto [num_tokens, hidden] = get_shape<2>(x);
        // 每个 token 的隐藏数据字节数
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());
        // 计算本地专家数：总专家数按 rank 数均分
        const auto num_local_experts = num_experts / context->num_ranks;
        // 断言输入在 GPU 上且连续
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        // 断言隐藏维字节数满足 16 字节（int4）对齐
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        // 断言 token 数不超过上限
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        // Check SF stuffs
        // 检查 SF（scale factor）相关内容
        // SF 包数（默认 0 表示无 SF）
        int num_sf_packs = 0;
        // SF 数据指针（默认空）
        void* sf_ptr = nullptr;
        // SF 的 token 与隐藏两个方向的步长
        int sf_token_stride = 0, sf_hidden_stride = 0;
        if (sf.has_value()) {
            // SF must be FP32 or packed UE8M0x4
            // SF 必须是 FP32 或打包的 UE8M0x4
            // 解包 SF 形状：token 数与 SF 包数
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
            // 断言 SF 的 token 数与输入一致
            EP_HOST_ASSERT(num_tokens == num_tokens_);
            // 断言 SF 在 GPU 上
            EP_HOST_ASSERT(sf->is_cuda());
            // 断言 SF 元素大小等于 SF 包大小
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            // 记录 SF 包数
            num_sf_packs = num_sf_packs_;
            // 记录 SF 数据指针
            sf_ptr = sf->data_ptr();
            // 记录 token 方向步长
            sf_token_stride = sf->stride(0);
            // 记录隐藏方向步长
            sf_hidden_stride = sf->stride(1);
        }

        // Check top-k stuffs
        // 检查 top-k 相关内容
        // 解包 topk_idx 形状：token 数与 top-k 值
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);
        // 断言 token 数与输入一致
        EP_HOST_ASSERT(num_tokens == num_tokens_);
        // 断言 topk_idx 的标量类型与 topk_idx_t 匹配
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        // 断言 topk_idx 在 GPU 上且连续
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

        // Weights are optional for training backward
        // 权重是可选的（训练反向传播时可不提供）
        // top-k 权重指针（默认空）
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            // 解包权重形状并断言 token 数一致
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__);
            // 断言权重在 GPU 上且连续
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            // 记录权重数据指针
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        // 专家接收计数器
        // 累计本地专家接收统计的指针（默认空）
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            // 解包形状（本地专家数）并断言一致
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());
            // 断言张量在 GPU 上且连续
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());
            // 断言专家数与配置一致
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            // 记录计数器指针
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // Stream control
        // All new tensor allocations should happen after this
        // 流控制
        // 此之后才允许分配新的张量
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream);
        // 获取通信流
        const auto comm_stream = comm::get_comm_stream();

        // The number of received tokens per expert
        // This is useful for expanding mode
        // 每个专家接收的 token 数
        // 该信息在 expand 模式下非常有用
        // 断言专家数可被 rank 数整除
        EP_HOST_ASSERT(num_experts % context->num_ranks == 0);
        // 取缓存的每专家前缀和，否则先置为空张量
        auto psum_num_recv_tokens_per_expert = cached_psum_num_recv_tokens_per_expert.value_or(torch::Tensor());
        if (cached_mode) {
            // 缓存模式：校验形状与属性
            const auto& [num_local_experts_] = get_shape<1>(psum_num_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_cuda() and psum_num_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
            // 注意：expand 模式下输入是排他前缀和，非 expand 模式下是包含前缀和
            // 分配长度为本地专家数 + 1 的 int 张量
            psum_num_recv_tokens_per_expert = torch::empty(
                {num_local_experts + 1}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // The unaligned (actual) number of received tokens per expert
        // Written by the dispatch kernel's notify warps, used by the epilogue for zero padding
        // 每个专家未对齐（实际）的接收 token 数
        // 由 dispatch 内核的通知 warp 写入，供收尾阶段的零填充使用
        auto num_unaligned_recv_tokens_per_expert = cached_num_unaligned_recv_tokens_per_expert.value_or(torch::Tensor());
        // 其数据指针（稍后填充）
        int* num_unaligned_recv_tokens_per_expert_ptr = nullptr;
        if (cached_mode) {
            // 缓存模式：校验形状与属性
            const auto& [num_local_experts_] = get_shape<1>(num_unaligned_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.is_cuda() and num_unaligned_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            // 分配长度为本地专家数的 int 张量
            num_unaligned_recv_tokens_per_expert = torch::empty(
                {num_local_experts}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }
        // 取其数据指针
        num_unaligned_recv_tokens_per_expert_ptr = num_unaligned_recv_tokens_per_expert.data_ptr<int>();

        // The prefix sum tensor of number of received tokens from each rank
        // Will also be used in combine as the dispatch handle
        // 每个 scale-up rank 接收 token 数的前缀和张量
        // 在 combine 中还会被用作 dispatch 句柄
        auto psum_num_recv_tokens_per_scaleup_rank = cached_psum_num_recv_tokens_per_scaleup_rank.value_or(torch::Tensor());
        if (cached_mode) {
            // 缓存模式：校验形状与属性
            const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
            EP_HOST_ASSERT(num_scaleup_ranks == context->num_scaleup_ranks);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        } else {
            // 分配长度为 scale-up rank 数的 int 张量
            psum_num_recv_tokens_per_scaleup_rank = torch::empty(
                {context->num_scaleup_ranks}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // Decide number of channels by shared memory consumption
        // Only for hybrid version
        // 根据 shared memory 消耗决定通道数
        // 仅混合模式使用
        // 每 SM 通道数与总通道数（非混合模式均为 1）
        int num_channels_per_sm = 1, num_channels = 1;
        // 获取设备可用的 shared memory 字节数
        const int num_smem_bytes = jit->device.get_num_smem_bytes();
        if (context->num_scaleout_ranks > 1) {
            // 获取 dispatch 与 combine 的 token 布局以估算每通道占用
            const auto dispatch_token_layout = get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk);
            const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
            // 断言 SM 数不超过上限
            EP_HOST_ASSERT(num_sms <= kNumMaxSMs);
            // 按 dispatch 布局与通知区占用的 shared memory 计算每 SM 可容纳的通道数
            num_channels_per_sm = std::min<int>(
                (num_smem_bytes - get_num_notify_smem_bytes(context->num_ranks, num_experts)) / dispatch_token_layout.get_num_bytes<true>(),
                32 - kNumNotifyWarps);
            // 再按 combine 布局的 shared memory 消耗收窄
            num_channels_per_sm = std::min<int>(
                num_smem_bytes / combine_token_layout.get_num_bytes<true>(),
                num_channels_per_sm);
            // 一半通道给通知 warp、一半给数据 warp，并限制在每 SM 上限内
            num_channels_per_sm = std::min<int>(
                /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
            // 若不优先与计算重叠，则把每 SM 通道数限制在 4 以内
            if (not prefer_overlap_with_compute)
                num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
            // 计算总通道数
            num_channels = num_sms * num_channels_per_sm;
            // 调试模式打印每 SM 的通道数
            if (get_env<int>("EP_BUFFER_DEBUG"))
                printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);
        }

        // Non-hybrid mode handles
        // 非混合模式的句柄
        // token 目的槽位索引（非混合模式为 [num_tokens, num_topk]）
        auto dst_buffer_slot_idx = cached_dst_buffer_slot_idx.value_or(torch::Tensor());
        if (context->num_scaleout_ranks == 1) {
            if (cached_mode) {
                // 缓存模式：校验形状与属性
                const auto [num_tokens__, num_topk_] = get_shape<2>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                // Allocate a new tensor
                // 分配新的张量
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            }
        }

        // Hybrid mode handles
        // 混合模式的句柄
        // 前向 token 元数据与每通道链表（混合模式专用）
        std::optional<torch::Tensor> token_metadata_at_forward, channel_linked_list;
        // 两者的数据指针（默认空）
        int *token_metadata_at_forward_ptr = nullptr, *channel_linked_list_ptr = nullptr;
        if (context->num_scaleout_ranks > 1) {
            // The token destination slot idx during forward
            // `[i, j, k, l]` means: from channel i from scale-out peer k, the j-th token's index in the l-th rank buffer
            // NOTES: Used primarily for cached mode
            // TODO: May make it a linked list to remove the redundant info in `token_metadata_at_forward`
            // 前向过程中的 token 目的槽位索引
            // `[i, j, k, l]` 表示：来自通道 i、跨机对端 k 的第 j 个 token 在第 l 个 rank 的 buffer 中的索引
            // 注意：主要供缓存模式使用
            // TODO: 或许可以改成链表，以去除 `token_metadata_at_forward` 中的冗余信息
            // 计算每通道的最大 token 数
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
            if (cached_mode) {
                // 缓存模式：校验四维形状与属性
                const auto [num_channels_, num_scaleout_ranks_, num_max_tokens_per_channel_, num_topk_] =
                    get_shape<4>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_channels == num_channels_ and context->num_scaleout_ranks == num_scaleout_ranks_ and
                               num_max_tokens_per_channel == num_max_tokens_per_channel_ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                // 分配新的四维张量
                dst_buffer_slot_idx = torch::empty(
                    {num_channels, context->num_scaleout_ranks, num_max_tokens_per_channel, num_topk},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }

            // The token metadata during forward
            // `[i, j]` means: in channel i, the j-th forwarded token's metadata
            // Info contains:
            //   - Scaleout rank index and source token index in the original rank (0)
            //   - Whether the token is the last one in the chunk (1)
            //   - cached top-k scaleup peer indices (top-k)
            //   - each selections' destination slot indices (top-k)
            // 前向过程中的 token 元数据
            // `[i, j]` 表示：通道 i 中第 j 个被前转 token 的元数据
            // 元数据包含:
            //   - 跨机 rank 索引与源 token 在原始 rank 中的索引（0）
            //   - 该 token 是否为分块中的最后一个（1）
            //   - 缓存的 top-k scale-up 对端索引（top-k 个）
            //   - 每个选择的目的槽位索引（top-k 个）
            // 计算每通道最大前转 token 数（多加 1 作为哨兵）
            const auto num_max_forwarded_tokens = context->num_scaleout_ranks * num_max_tokens_per_channel + 1;
            // 元数据维度数：2 个基础字段 + 每 top-k 选择 2 个字段
            const auto num_forward_metadata_dims = 2 + num_topk * 2;
            if (cached_mode) {
                // 缓存模式：使用缓存的元数据并校验
                token_metadata_at_forward = cached_token_metadata_at_forward;
                const auto [num_channels_, num_max_forwarded_tokens_, num_forward_metadata_dims_] = get_shape<3>(token_metadata_at_forward.value());
                EP_HOST_ASSERT(num_channels == num_channels_ and num_max_forwarded_tokens == num_max_forwarded_tokens_
                               and num_forward_metadata_dims == num_forward_metadata_dims_);
                EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
                EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);
            } else {
                // 分配新的三维元数据张量
                token_metadata_at_forward = torch::empty(
                    {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            // 取元数据指针
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();

            // Per-scaleup-peer-per-channel linked list
            // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
            // 每个 scale-up 对端、每通道一条的链表
            // `[i, j, k]` 表示：来自通道 i、scale-up 对端 k 的第 j 个 token 在 combine 输入中的索引
            if (cached_mode) {
                // 缓存模式：使用缓存的链表并校验
                channel_linked_list = cached_channel_linked_list;
                const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
                channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
                EP_HOST_ASSERT(num_channels == num_channels__);
                EP_HOST_ASSERT(d1_ == context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
                EP_HOST_ASSERT(d2_ == context->num_scaleup_ranks);
                EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
                EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
            } else {
                // 分配新的链表张量
                channel_linked_list = torch::empty(
                    // Index 0 of the list means the starting item
                    // 链表中索引 0 表示起始项
                    {num_channels,
                    context->num_scaleout_ranks * num_max_tokens_per_channel + 1,
                    context->num_scaleup_ranks},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            // 取链表指针
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
        }

        // Check buffer size
        // 检查 buffer 大小是否足够
        EP_HOST_ASSERT(get_dispatch_buffer_size(
                       num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, x.element_size(),
                       context->num_scaleout_ranks, context->num_scaleup_ranks,
                       context->is_scaleup_nvlink) <= num_buffer_bytes);

        // Ready and clean host workspace for this round
        // 就绪并清空本轮使用的 host workspace
        // 构造 host workspace 的布局视图
        const auto host_workspace_layout = layout::EPWorkspaceLayout(
            host_workspace,
            context->num_scaleout_ranks,
            context->num_scaleup_ranks,
            num_experts);
        // 清零各 scale-up rank 的计数区
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), context->num_scaleup_ranks, 0);
        // 清零各本地专家的计数区
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);
        // 全量内存栅栏，保证清零对后续 GPU 写入可见
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Do dispatch into the buffers (with SM limitation)
        // 执行 dispatch（限制使用的 SM 数）
        launch_dispatch(x.data_ptr(), sf_ptr,
                        topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,
                        cumulative_local_expert_recv_stats_ptr,
                        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),
                        num_unaligned_recv_tokens_per_expert_ptr,
                        dst_buffer_slot_idx.data_ptr<int>(),
                        token_metadata_at_forward_ptr,
                        num_tokens, num_max_tokens_per_rank,
                        hidden, x.element_size(),
                        num_sf_packs, sf_token_stride, sf_hidden_stride,
                        num_experts, num_topk, expert_alignment,
                        context->dev_comm, context->window,
                        context->buffer,
                        context->workspace, mapped_host_workspace,
                        context->scaleout_rank_idx, context->scaleup_rank_idx,
                        context->num_scaleout_ranks, context->num_scaleup_ranks,
                        context->is_scaleup_nvlink,
                        num_sms, num_channels_per_sm,
                        num_smem_bytes,
                        num_qps, context->num_gpu_timeout_cycles,
                        cached_mode, do_cpu_sync,
                        comm_stream);

        // For tensor recording
        // 供 record_stream 使用的张量列表
        tensor_list_t tensors_to_record = {
            x, sf, topk_idx, topk_weights,
            cumulative_local_expert_recv_stats,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            num_unaligned_recv_tokens_per_expert,
            dst_buffer_slot_idx,
            token_metadata_at_forward,
            channel_linked_list};

        // Epilogue can be deferred, so it is a lambda
        // 收尾可以被延迟执行，因此封装为 lambda
        auto epilogue = [=, this](const at::cuda::CUDAStream& stream,
                                  const std::optional<std::reference_wrapper<tensor_list_t>>& tensors_to_record_opt) mutable {
            // Received token counters
            // 接收 token 的各类计数
            int num_recv_tokens = 0, num_expanded_tokens = 0;
            // CPU 轮询的游标：scale-up rank 与本地专家
            int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;
            // 每个专家接收 token 数列表
            std::vector<int> num_recv_tokens_per_expert_list;

            // Assign these values according to modes
            // 按模式赋予这些值
            if (cached_mode) {
                // Cached mode
                // 缓存模式
                // 断言缓存模式不允许 CPU 同步
                EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
                // 直接采用缓存的接收 token 数
                num_recv_tokens = cached_num_recv_tokens.value();
                num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
                num_expanded_tokens = cached_num_expanded_tokens.value();
            } else if (do_cpu_sync) {
                // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
                // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
                // unless we release GIL here.
                // 在 dispatch 中，CPU 会忙等直到 GPU 从其他 rank 接收到张量大小元数据，这可能耗时较长。
                // 如果 DeepEP 的用户需要在其他线程执行别的 Python 代码（如 KV 传输），
                // 若不在这里释放 GIL，其代码会因 GIL 而被阻塞。
                pybind11::gil_scoped_release release;

                // Non-cached mode with sync
                // 非缓存模式且需要同步：记录起始时间
                const auto start_cpu_time = std::chrono::high_resolution_clock::now();
                // 忙等直到所有计数就绪
                while (true) {
                    // 是否全部就绪
                    bool ready = true;

                    // Read number of received tokens from each scaleup rank
                    // 逐个读取来自每个 scale-up rank 的接收 token 数
                    while (counter_scaleup_rank_idx < context->num_scaleup_ranks and ready) {
                        // 从 host workspace 解码计数（带就绪位编码）
                        const auto count = math::encode_decode_positive(
                            host_workspace_layout.get_scaleup_rank_count_ptr<false>()[counter_scaleup_rank_idx]);
                        // 若该计数已就绪则累加并前移游标
                        if ((ready = math::is_decoded_positive_ready(count))) {
                            num_recv_tokens += count;
                            ++ counter_scaleup_rank_idx;
                        }
                    }

                    // Read expert counts
                    // 逐个读取各本地专家的计数
                    while (counter_local_expert_idx < num_local_experts and ready) {
                        // 从 host workspace 解码专家计数
                        const auto count = math::encode_decode_positive(
                            host_workspace_layout.get_scaleup_expert_count_ptr<false>()[counter_local_expert_idx]);
                        // 若就绪则记录并累加展开 token 数
                        if ((ready = math::is_decoded_positive_ready(count))) {
                            num_recv_tokens_per_expert_list.push_back(count);
                            num_expanded_tokens += count;
                            ++ counter_local_expert_idx;
                        }
                    }

                    // Ready and do next steps
                    // 就绪后执行下一步
                    // 构造调试信息字符串的 lambda
                    const auto get_buffer_info = [&]() {
                        std::stringstream ss;
                        ss << "CPU side received count (scaleup: " << context->scaleup_rank_idx << "): ";
                        for (int i = 0; i < context->num_scaleup_ranks + num_local_experts; ++ i) {
                            ss << host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i];
                            ss << (i == context->num_scaleup_ranks - 1 ? " # ": " ");
                        }
                        return ss.str();
                    };
                    if (ready) {
                        // 调试模式打印缓冲信息
                        if (get_env<int>("EP_BUFFER_DEBUG"))
                            printf("%s\n", get_buffer_info().c_str());
                        // 全部就绪，退出忙等
                        break;
                    }

                    // Timeout checks
                    // 超时检查
                    const auto now = std::chrono::high_resolution_clock::now();
                    // 超过 CPU 超时秒数则抛出带行号信息的异常
                    if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > context->num_cpu_timeout_secs)
                        throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
                }
            } else {
                // Non-cached mode without CPU sync, allocate with the worst case
                // 非缓存模式且不做 CPU 同步：按最坏情况分配
                num_recv_tokens = num_max_tokens_per_rank * context->num_ranks;
                num_expanded_tokens = context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_local_experts);
                num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
                num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
            }

            // Allocate received tensors
            // `recv_src_metadata` includes source token indices and buffer slot indices
            // 分配接收侧张量
            // `recv_src_metadata` 包含源 token 索引与 buffer 槽位索引
            // expand 模式按展开数分配，否则按接收数
            const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
            // 分配接收数据张量
            auto recv_x = torch::empty({num_allocated_tokens, hidden}, x.options());
            // 接收 SF 张量（可选）
            auto recv_sf = std::optional<torch::Tensor>();
            // 接收 top-k 索引张量（可选）
            auto recv_topk_idx = std::optional<torch::Tensor>();
            // 接收 top-k 权重张量（可选）
            auto recv_topk_weights = std::optional<torch::Tensor>();
            // 接收源元数据：缓存模式复用，否则新分配
            auto recv_src_metadata = cached_mode ?
                cached_recv_src_metadata.value() :
                torch::empty({num_recv_tokens, num_topk + 2},
                             torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

            // Optional tensors
            // 可选张量
            // 接收 SF 数据指针（默认空）
            void* recv_sf_ptr = nullptr;
            // 接收 top-k 索引指针（默认空）
            topk_idx_t* recv_topk_idx_ptr = nullptr;
            // 接收权重指针（默认空）
            float* recv_topk_weights_ptr = nullptr;
            // 接收 SF 的步长（默认 0）
            int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
            if (sf.has_value()) {
                if (not use_tma_aligned_col_major_sf) {
                    // 行主序布局：token 方向步长为包数
                    recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;
                } else {
                    // TMA-aligned layout for the next GEMM input
                    // 供下一个 GEMM 输入使用的 TMA 对齐布局
                    recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks);
                }
                // 按步长分配接收 SF 张量
                recv_sf = torch::empty_strided({num_allocated_tokens, num_sf_packs},
                                               {recv_sf_token_stride, recv_sf_hidden_stride},
                                               sf->options());
                // 取其数据指针
                recv_sf_ptr = recv_sf->data_ptr();
            }
            if (not do_expand) {
                // 非 expand 模式才需要接收 top-k 索引
                recv_topk_idx = torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());
                recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
            }
            if (topk_weights.has_value()) {
                // expand 模式下权重为单列，否则与 top-k 同宽
                recv_topk_weights = do_expand ?
                    torch::empty({num_allocated_tokens}, topk_weights->options()) :
                    torch::empty({num_allocated_tokens, num_topk}, topk_weights->options());
                recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
            }

            // Process prefix sum, in expanding mode, it is also atomic counters
            // 处理前缀和；expand 模式下它同时被用作原子计数器
            if (not cached_mode) {
                if (do_expand) {
                    // Slice the exclusive part and do atomic additions into inclusive
                    // 切出排他部分，内核通过原子加法转成包含前缀和
                    psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
                } else {
                    // Slice the inclusive part (and will not be used in the epilogue)
                    // 切出包含部分（收尾阶段不会再使用）
                    psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
                }
            }
            // 断言前缀和长度等于本地专家数
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) == num_local_experts);

            // Launch copy kernels with full SMs
            // 使用全部 SM 启动拷贝收尾内核
            launch_dispatch_copy_epilogue(context->buffer, context->workspace,
                                          psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                                          psum_num_recv_tokens_per_expert.data_ptr<int>(),
                                          recv_x.data_ptr(), recv_sf_ptr,
                                          recv_topk_idx_ptr, recv_topk_weights_ptr,
                                          recv_src_metadata.data_ptr<int>(),
                                          channel_linked_list_ptr,
                                          num_unaligned_recv_tokens_per_expert_ptr,
                                          num_recv_tokens, num_max_tokens_per_rank,
                                          num_hidden_bytes,
                                          num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,
                                          num_experts, num_topk, expert_alignment,
                                          context->scaleout_rank_idx, context->scaleup_rank_idx,
                                          context->num_scaleout_ranks, context->num_scaleup_ranks,
                                          jit->device.get_num_sms(),
                                          jit->device.get_num_smem_bytes(),
                                          num_channels,
                                          do_expand, cached_mode,
                                          do_zero_padding,
                                          stream);

            // 组装 Python 返回的元组
            auto result = pybind11::make_tuple(
                recv_x, recv_sf,
                recv_topk_idx, recv_topk_weights,
                num_recv_tokens, num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                token_metadata_at_forward,
                channel_linked_list);

            // For non-deferring tensor recording
            // 非延迟模式下的张量登记
            if (tensors_to_record_opt.has_value()) {
                // 取出登记列表的引用
                auto& tensors = tensors_to_record_opt->get();
                // 依次登记接收侧生成的张量
                tensors.push_back(recv_x);
                tensors.push_back(recv_sf);
                tensors.push_back(recv_topk_idx);
                tensors.push_back(recv_topk_weights);
                tensors.push_back(recv_src_metadata);
            }
            // 返回结果元组
            return result;
        };

        // Defer epilogue: record the created tensors only, and return intermediately
        // NOTES: CPU sync will be deferred, too
        // 延迟收尾：只登记已创建的张量，然后立即返回
        // 注意：CPU 同步同样会被延迟
        if (defer_epilogue) {
            // 断言延迟收尾必须在异步模式下
            EP_HOST_ASSERT(async_with_compute_stream);
            // 执行流收尾（异步）
            const auto event = stream_control_epilogue(
                tensors_to_record, compute_stream, allocate_on_comm_stream, true);
            // 打包收尾钩子：Python 侧后续调用时在当前流上执行
            std::function<pybind11::object()> epilogue_hook = [epilogue = std::move(epilogue)]() mutable {
                return epilogue(at::cuda::getCurrentCUDAStream(), std::nullopt);
            };
            // 返回 (空结果, 事件, 收尾钩子)
            return pybind11::make_tuple(pybind11::none(), event, epilogue_hook);
        }

        // Do epilogue intermediately and record all tensors
        // 立即执行收尾并登记所有张量
        auto result = epilogue(comm_stream, std::ref(tensors_to_record));
        // 执行流收尾
        const auto event = stream_control_epilogue(
            tensors_to_record, compute_stream, allocate_on_comm_stream, async_with_compute_stream);
        // 返回 (结果, 事件, 空钩子)
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    // MoE combine：把各 rank 上经过专家计算的结果合并回来（含可选归约权重与偏置）
    pybind11::tuple
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const std::optional<torch::Tensor>& token_metadata_at_forward,
            const std::optional<torch::Tensor>& channel_linked_list,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms, const int& num_qps,
            const std::optional<EventHandle>& previous_event,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout,
            const bool& defer_epilogue) const {
        // Check SM count
        // 检查 SM 数
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        // 混合 combine 至少需要 2 个 SM
        EP_HOST_ASSERT((num_sms > 1 or context->num_scaleout_ranks == 1 or context->num_scaleup_ranks == 1) and
                       "Hybrid combine requires at least 2 SMs");
        // 延迟收尾必须在异步模式下
        EP_HOST_ASSERT(not defer_epilogue or async_with_compute_stream);

        // Check data
        // 检查数据
        // 解包输入形状：token 数与隐藏维
        const auto [num_tokens, hidden] = get_shape<2>(x);
        // 断言输入在 GPU 上且连续
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        // 断言输入为 BF16 类型
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        // 断言隐藏维字节数满足 int4 对齐
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);

        // Check tensors at dispatch
        // 检查 dispatch 阶段的张量
        // 解包合并后的 top-k 索引形状：token 数与 top-k 值
        const auto [num_combined_tokens, num_topk] = get_shape<2>(combined_topk_idx);
        // 解包 scale-up rank 前缀和的长度
        const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
        // 断言 top-k 索引在 GPU 上且连续
        EP_HOST_ASSERT(combined_topk_idx.is_cuda() and combined_topk_idx.is_contiguous());
        // 断言其标量类型与 topk_idx_t 匹配
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        // 断言 scale-up rank 数与上下文一致
        EP_HOST_ASSERT(num_scaleup_ranks == context->num_scaleup_ranks);
        // 断言前缀和在 GPU 上且连续
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
        // 断言其为 int 类型
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        // 断言合并后的 token 数不超过上限
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);

        // Check metadata
        // For reduction mode, `num_tokens_` means the number of unexpanded tokens
        // 检查元数据
        // 归约模式下 `num_tokens_` 表示未展开的 token 数
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
        // 断言元数据行数与布局模式匹配
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));
        // 断言元数据列数为 top-k + 2
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
        // 断言元数据在 GPU 上且连续
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
        // 断言其为 int 类型
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

        // Check optional tensors
        // 检查可选张量
        if (topk_weights.has_value()) {
            if (use_expanded_layout) {
                // 展开布局：权重是一维的（每合并 token 一个）
                const auto [num_tokens__] = get_shape<1>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__);
            } else {
                // 标准布局：权重与 top-k 同形
                const auto [num_tokens__, num_topk__] = get_shape<2>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);
            }
            // 断言权重在 GPU 上且连续
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            // 断言权重为 float 类型
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
        }

        // 把两个偏置放入列表便于统一校验
        const auto bias_opts = std::vector({bias_0, bias_1});
        // 逐个校验偏置
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                // 取出该偏置
                auto bias = bias_opts[i].value();
                // 断言为二维、在 GPU 上且连续
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_cuda() and bias.is_contiguous());
                // 断言数据类型与输入一致
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                // 断言形状为 [合并 token 数, 隐藏维]
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            }
        }

        // Stream control
        // All new tensor allocations should happen after this
        // 流控制
        // 此之后才允许分配新的张量
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream);
        // 获取通信流
        const auto comm_stream = comm::get_comm_stream();

        // Check buffer size
        // 检查 buffer 大小是否足够
        EP_HOST_ASSERT(get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk,
                                               context->num_scaleout_ranks, context->num_scaleup_ranks,
                                               context->is_scaleup_nvlink, allow_multiple_reduction) <= num_buffer_bytes);

        // Optional configs and metadata for hybrid combine
        // 混合 combine 的可选配置与元数据
        // 总通道数（非混合模式为 1）
        int num_channels = 1;
        // 前向 token 元数据指针（默认空）
        int* token_metadata_at_forward_ptr = nullptr;
        // 通道链表指针（默认空）
        int* channel_linked_list_ptr = nullptr;
        if (context->num_scaleout_ranks > 1) {
            // The token metadata during forward
            // 前向过程中的 token 元数据
            const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward.value());
            // 计算每通道最大 token 数
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);
            // 记录总通道数
            num_channels = num_channels_;
            // 取元数据指针
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();
            // 校验第一维大小
            EP_HOST_ASSERT(d1 == context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            // 校验第二维大小（2 + 2 * top-k）
            EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
            // 断言元数据在 GPU 上且连续
            EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
            // 断言其为 int 类型
            EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);

            // Per-scaleup-peer-per-channel linked list
            // 每个 scale-up 对端、每通道一条的链表
            const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
            // 取链表指针
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
            // 校验通道数
            EP_HOST_ASSERT(num_channels == num_channels__);
            // 校验链表长度
            EP_HOST_ASSERT(d1_ == context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            // 校验对端数
            EP_HOST_ASSERT(d2_ == context->num_scaleup_ranks);
            // 断言链表在 GPU 上且连续
            EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
            // 断言其为 int 类型
            EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
        }

        // Push data into remote buffers
        // NOTES: we don't use `num_hidden_bytes` due to enable later quantization possibility
        // 把数据推送到远端的 buffer
        // 注意：这里不使用 `num_hidden_bytes`，以便将来支持延迟量化
        const auto reduce_buffer = launch_combine(
            x.data_ptr(),
            topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            token_metadata_at_forward_ptr,
            channel_linked_list_ptr,
            context->dev_comm, context->window,
            context->buffer, context->workspace,
            num_reduced_tokens, num_max_tokens_per_rank,
            hidden, num_experts, num_topk,
            num_qps, context->num_gpu_timeout_cycles,
            context->num_scaleout_ranks, context->num_scaleup_ranks,
            context->scaleout_rank_idx, context->scaleup_rank_idx,
            context->is_scaleup_nvlink,
            num_sms, jit->device.get_num_smem_bytes(),
            num_channels,
            use_expanded_layout, allow_multiple_reduction,
            comm_stream);

        // For tensor recording
        // 供 record_stream 使用的张量列表
        tensor_list_t tensors_to_record = {
            x, topk_weights, bias_0, bias_1,
            src_metadata, combined_topk_idx,
            psum_num_recv_tokens_per_scaleup_rank,
            token_metadata_at_forward, channel_linked_list};

        // Epilogue can be deferred, so it is a lambda
        // 收尾可以被延迟执行，因此封装为 lambda
        auto epilogue = [=, this](const at::cuda::CUDAStream& stream,
                                  const std::optional<std::reference_wrapper<tensor_list_t>>& tensors_to_record_opt) {
            // Allocate output tensors
            // 分配输出张量
            auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
            // 合并后的 top-k 权重（可选）
            auto combined_topk_weights = std::optional<torch::Tensor>();
            // 其数据指针（默认空）
            float* combined_topk_weights_ptr = nullptr;
            if (topk_weights.has_value()) {
                // 提供权重时才分配并取指针
                combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
                combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
            }

            // Resolve pointers here to retain bias tensors in a deferred epilogue
            // 在这里解析偏置指针，以便延迟收尾时偏置张量仍被 lambda 持有
            void* bias_ptrs[2] = {
                bias_opts[0].has_value() ? bias_opts[0]->data_ptr() : nullptr,
                bias_opts[1].has_value() ? bias_opts[1]->data_ptr() : nullptr
            };

            // Combine pushed data
            // 对推送来的数据执行合并
            launch_combine_reduce_epilogue(combined_x.data_ptr(),
                                           combined_topk_weights_ptr,
                                           combined_topk_idx.data_ptr<topk_idx_t>(),
                                           num_combined_tokens, num_max_tokens_per_rank,
                                           hidden,
                                           num_experts, num_topk,
                                           reduce_buffer,
                                           bias_ptrs[0], bias_ptrs[1],
                                           context->num_scaleout_ranks, context->num_scaleup_ranks,
                                           context->scaleout_rank_idx, context->scaleup_rank_idx,
                                           jit->device.get_num_sms(),
                                           jit->device.get_num_smem_bytes(),
                                           use_expanded_layout, allow_multiple_reduction,
                                           stream);

            if (tensors_to_record_opt.has_value()) {
                // 取出登记列表的引用
                auto& tensors = tensors_to_record_opt->get();
                // 登记合并输出张量
                tensors.push_back(combined_x);
                tensors.push_back(combined_topk_weights);
            }
            // 返回 (合并结果, 合并权重)
            return pybind11::make_tuple(combined_x, combined_topk_weights);
        };

        // Defer epilogue
        // 延迟收尾
        if (defer_epilogue) {
            // 执行流收尾（异步）
            const auto event = stream_control_epilogue(
                tensors_to_record, compute_stream, allocate_on_comm_stream, true);
            // 打包收尾钩子供 Python 侧后续调用
            std::function<pybind11::object()> epilogue_hook = [epilogue = std::move(epilogue)]() {
                return epilogue(at::cuda::getCurrentCUDAStream(), std::nullopt);
            };
            // 返回 (空结果, 事件, 收尾钩子)
            return pybind11::make_tuple(pybind11::none(), event, epilogue_hook);
        }

        // Do epilogue
        // 立即执行收尾
        auto result = epilogue(comm_stream, std::ref(tensors_to_record));
        // 执行流收尾
        const auto event = stream_control_epilogue(
            tensors_to_record, compute_stream, allocate_on_comm_stream, async_with_compute_stream);
        // 返回 (结果, 事件, 空钩子)
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    // 负载均衡（EPLB）：预取冗余专家与本地专家的权重到 GPU
    std::optional<EventHandle>
    lb_prefetch_weights(const std::vector<torch::Tensor>& redundant_expert_weights,
                        const std::vector<torch::Tensor>& expert_weights,
                        const torch::Tensor& redundancy_mapping,
                        const int& num_sms,
                        const std::optional<EventHandle>& previous_event) const {
        // Checks
        // 各项检查
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        // 断言冗余专家权重与专家权重数量一致
        EP_HOST_ASSERT(redundant_expert_weights.size() == expert_weights.size());
        // 断言列表非空
        EP_HOST_ASSERT(not expert_weights.empty());
        // 断言不超过布局支持的最大权重条目数
        EP_HOST_ASSERT(redundant_expert_weights.size() <= layout::kNumMaxWeightEntries);
        // 断言冗余映射在 GPU 上且连续
        EP_HOST_ASSERT(redundancy_mapping.is_cuda() and redundancy_mapping.is_contiguous());
        // 断言冗余映射为二维
        EP_HOST_ASSERT(redundancy_mapping.dim() == 2);
        // 断言第一维等于 NVLink rank 数
        EP_HOST_ASSERT(redundancy_mapping.size(0) == context->num_nvl_ranks);
        // 断言其为 int 类型
        EP_HOST_ASSERT(redundancy_mapping.scalar_type() == torch::kInt);

        // 冗余专家数与权重列表长度、本地专家数
        const int num_redundant_experts = redundancy_mapping.size(1);
        const int num_weights = static_cast<int>(expert_weights.size());
        const int num_local_experts = expert_weights.front().size(0);
        // 断言本地专家数为正
        EP_HOST_ASSERT(num_local_experts > 0);

        // 构造权重描述列表
        layout::EPWeightList weights;
        // 逐项校验并登记
        for (int i = 0; i < num_weights; ++ i) {
            // 取第 i 项的冗余权重与本地权重
            const auto& redundant_expert_weight = redundant_expert_weights[i];
            const auto& expert_weight = expert_weights[i];
            // 断言两者均在 GPU 上且连续
            EP_HOST_ASSERT(redundant_expert_weight.is_cuda() and redundant_expert_weight.is_contiguous());
            EP_HOST_ASSERT(expert_weight.is_cuda() and expert_weight.is_contiguous());
            // 断言维数至少为 1
            EP_HOST_ASSERT(redundant_expert_weight.dim() >= 1);
            EP_HOST_ASSERT(expert_weight.dim() >= 1);
            // 断言首维分别匹配冗余专家数与本地专家数
            EP_HOST_ASSERT(redundant_expert_weight.size(0) == num_redundant_experts);
            EP_HOST_ASSERT(expert_weight.size(0) == num_local_experts);
            // 计算单个专家权重的字节数（除首维外各维的乘积 x 元素大小）
            const int64_t num_bytes_per_expert = c10::multiply_integers(redundant_expert_weight.sizes().slice(1)) *
                                                  redundant_expert_weight.element_size();
            // 断言冗余权重与本地权重的每专家字节数一致
            EP_HOST_ASSERT(num_bytes_per_expert == c10::multiply_integers(expert_weight.sizes().slice(1)) *
                                                   expert_weight.element_size());
            // 断言每专家字节数按 16 对齐
            EP_HOST_ASSERT(num_bytes_per_expert % 16 == 0);

            // 登记该权重条目
            weights[i] = {
                .redundant_expert_weights = redundant_expert_weight.data_ptr(),
                .expert_weights = expert_weight.data_ptr(),
                .num_bytes_per_expert = num_bytes_per_expert
            };
        }

        // Stream control
        // 流控制
        const auto compute_stream = stream_control_prologue(previous_event);

        // Launch
        // 启动预取内核
        launch_lb_prefetch_weights(
            *context, num_weights, weights, redundancy_mapping.data_ptr<int>(),
            num_redundant_experts, num_local_experts,
            num_sms, comm::get_comm_stream());

        // Stream epilogue
        // 流收尾：登记所有参与张量
        tensor_list_t tensors_to_record = {redundancy_mapping};
        tensors_to_record.insert(tensors_to_record.end(), redundant_expert_weights.begin(), redundant_expert_weights.end());
        tensors_to_record.insert(tensors_to_record.end(), expert_weights.begin(), expert_weights.end());
        // 返回事件
        return stream_control_epilogue(tensors_to_record, compute_stream, false, true);
    }

    // 负载均衡（EPLB）：把对端的冗余专家梯度归约到本地专家梯度上
    std::optional<EventHandle>
    lb_reduce_grads(const torch::Tensor& redundant_expert_grads,
                    const torch::Tensor& expert_grads,
                    const torch::Tensor& redundancy_mapping,
                    const int& num_sms,
                    const std::optional<EventHandle>& previous_event) const {
        // Checks
        // 各项检查
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        // 断言各张量均在 GPU 上且连续
        EP_HOST_ASSERT(redundant_expert_grads.is_cuda() and redundant_expert_grads.is_contiguous());
        EP_HOST_ASSERT(expert_grads.is_cuda() and expert_grads.is_contiguous());
        EP_HOST_ASSERT(redundancy_mapping.is_cuda() and redundancy_mapping.is_contiguous());
        // 断言维度均为二维
        EP_HOST_ASSERT(redundant_expert_grads.dim() == 2);
        EP_HOST_ASSERT(expert_grads.dim() == 2);
        EP_HOST_ASSERT(redundancy_mapping.dim() == 2);
        // 断言冗余映射第一维等于 NVLink rank 数
        EP_HOST_ASSERT(redundancy_mapping.size(0) == context->num_nvl_ranks);
        // 断言其为 int 类型
        EP_HOST_ASSERT(redundancy_mapping.scalar_type() == torch::kInt);

        // 冗余专家数与（本地专家数, 隐藏维）
        const int num_redundant_experts = redundancy_mapping.size(1);
        const auto [num_local_experts, hidden] = get_shape<2>(expert_grads);

        // 断言形状与类型匹配
        EP_HOST_ASSERT(redundant_expert_grads.size(0) == num_redundant_experts);
        EP_HOST_ASSERT(redundant_expert_grads.size(1) == hidden);
        EP_HOST_ASSERT(redundant_expert_grads.scalar_type() == torch::kFloat);
        EP_HOST_ASSERT(expert_grads.scalar_type() == torch::kFloat);

        // Stream control
        // 流控制
        const auto compute_stream = stream_control_prologue(previous_event);

        // Launch: accumulate peers' redundant gradients into the local expert gradients
        // 启动内核：把对端的冗余梯度累加到本地专家梯度上
        launch_lb_reduce_grads(
            *context, redundant_expert_grads.data_ptr<float>(), expert_grads.data_ptr<float>(),
            redundancy_mapping.data_ptr<int>(), num_redundant_experts, num_local_experts, hidden,
            num_sms, comm::get_comm_stream());

        // Stream epilogue
        // 流收尾：登记张量并返回事件
        return stream_control_epilogue(
            {redundant_expert_grads, expert_grads, redundancy_mapping},
            compute_stream, false, true);
    }
};

// 向 Python 模块注册 EP 缓冲区的绑定接口
static void register_apis(pybind11::module_& m) {
    // 将 EPBuffer 暴露给 Python，并绑定构造函数与各方法
    pybind11::class_<EPBuffer, BufferBase>(m, "EPBuffer")
        .def(pybind11::init<int, int, int64_t, int64_t, int64_t, bool, bool, bool, std::optional<int>, int, int, int, bool>())
        // 只读属性：通信上下文
        .def_readonly("context", &EPBuffer::context)
        // 只读属性：负载均衡存储张量
        .def_readonly("lb_storage", &EPBuffer::lb_storage)
        // 方法：dispatch 分发
        .def("dispatch", &EPBuffer::dispatch)
        // 方法：combine 合并
        .def("combine", &EPBuffer::combine)
        // 方法：预取负载均衡权重
        .def("lb_prefetch_weights", &EPBuffer::lb_prefetch_weights)
        // 方法：归约负载均衡梯度
        .def("lb_reduce_grads", &EPBuffer::lb_reduce_grads);
    // 注册静态函数：计算 EP buffer 大小
    m.def("calculate_ep_buffer_size", &EPBuffer::calculate_buffer_size);
    // 注册静态函数：获取 EP buffer 的对齐字节数
    m.def("get_ep_buffer_alignment", [=]() {
        return kNumAllocationAlignmentBytes;
    });

}

}  // namespace deep_ep::ep
