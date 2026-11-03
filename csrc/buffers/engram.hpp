#pragma once

// 引入标准算法
#include <algorithm>
// 引入数学函数
#include <cmath>
// 引入 CUDA 运行时 API（cudaMemcpyAsync 等）
#include <cuda_runtime.h>
// 引入 std::function
#include <functional>
// 引入 std::shared_ptr
#include <memory>
// 引入 std::reduce 等数值工具
#include <numeric>
// 引入 std::optional
#include <optional>
// 引入 std::tuple
#include <tuple>
// 引入 std::vector
#include <vector>
// 引入 pybind11 的 std::function 支持
#include <pybind11/functional.h>
// 引入 PyTorch 的 Python 绑定支持
#include <torch/python.h>

// 引入 DeepEP 公共编译常量
#include <deep_ep/common/compiled.cuh>
// 引入对齐等数学工具
#include <deep_ep/common/math.cuh>
// 引入 Engram workspace 布局定义
#include <deep_ep/layout/engram/workspace.cuh>

// 引入缓冲区基类 BufferBase
#include "base.hpp"
// 引入 NCCL 通信上下文与公共通信 API
#include "../comm/api.hpp"
// 引入 Engram 抓取内核的启动封装
#include "../kernels/engram/engram.hpp"
// 引入张量工具（形状解包等）
#include "../utils/tensor.hpp"

// Engram（按条目索引的专家参数存储）通信所在的命名空间
namespace deep_ep::engram {

// Engram 缓冲区：写入/抓取按层组织的专家参数条目，支持 FP8 scale 因子
class EngramBuffer: public BufferBase {
    // QP 深度预留量：保证 QP 中留有余量用于 flush 操作
    static constexpr int kFlushDepthReserve = 256;

    // 是否允许混合模式（scale-out 与 scale-up 结合）
    bool allow_hybrid_mode;
    // 每个 QP 的队列深度
    int qp_depth;

    // 每层的条目数列表
    std::vector<int> num_entries_per_layer;
    // scale 因子的数据类型（FP8 抓取时使用）
    torch::ScalarType sf_dtype;
    // 隐藏维大小与元素字节数（默认 0，由 set_config 设置）
    int hidden = 0, elem_size = 0;
    // 单次抓取的最大 token 数
    int num_max_tokens = 0;
    // 每个 token 的 scale 因子包数
    int num_sf_packs = 0;
    // scale 因子分片的总字节数
    int64_t num_sf_shard_bytes = 0;

public:
    // NCCL 通信上下文
    std::shared_ptr<comm::Context> context;

    // 构造函数：创建通信上下文并注册屏障信号区
    EngramBuffer(const int& rank_idx, const int& num_ranks,
                 const int64_t& nccl_comm, const symmetric::shared_comm_t& shared_comm,
                 const int64_t& num_gpu_buffer_bytes, const int64_t& num_rdma_storage_bytes,
                 const bool& use_cpu_rdma_storage,
                 const bool& allow_hybrid_mode,
                 const std::optional<int>& sl_idx, const int& num_allocated_qps, const int& qp_depth,
                 const int& num_gpu_timeout_secs,
                 const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),
        allow_hybrid_mode(allow_hybrid_mode),
        qp_depth(qp_depth) {
        // 计算 workspace（EngramWorkspace）大小并按分配对齐
        const auto num_workspace_bytes = math::align<int64_t>(
            sizeof(layout::EngramWorkspace),
            kNumAllocationAlignmentBytes);
        // 创建 NCCL 通信上下文，包含 GPU buffer 与 RDMA 存储的配置
        context = std::make_shared<comm::Context>(
            nccl_comm, shared_comm, num_ranks, rank_idx,
            num_workspace_bytes, num_gpu_buffer_bytes, num_rdma_storage_bytes, use_cpu_rdma_storage,
            allow_hybrid_mode, sl_idx, num_allocated_qps, qp_depth,
            0, num_gpu_timeout_secs);
        // 设置基类中的主上下文
        main_context = context;
        // 获取 workspace 的引用并注册屏障信号区
        auto& workspace = *static_cast<layout::EngramWorkspace*>(context->workspace);
        context->set_barrier_signals(&workspace.barrier);
    }

    // 析构函数：按销毁策略处理资源释放
    ~EngramBuffer() noexcept(false) override {
        // 基类辅助：必要时自动调用 destroy()，否则打印泄漏警告
        destroy_on_destruction("Engram");
    }

    // 销毁缓冲区：先屏障同步，再释放通信上下文
    void destroy() override {
        // 断言尚未销毁
        EP_HOST_ASSERT(not destroyed);
        // 执行组内屏障，确保所有 rank 都完成工作后再销毁
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);
        // 结束并释放 NCCL 通信上下文
        context->finalize();
        // 清空上下文共享指针
        context.reset();
        // 标记已销毁
        destroyed = true;
    }

    // 设置/更新配置：各层条目数、隐藏维、元素大小、最大 token 数等
    // 同时会与所有 rank 屏障同步，保证配置切换的全局一致性
    void set_config(const std::vector<int>& new_num_entries_per_layer,
                    const int& new_hidden, const int& new_elem_size,
                    const int& new_num_max_tokens, const int& num_entries_per_token,
                    const int& new_num_sf_packs) {
        // 先屏障同步，确保所有 rank 同时切换配置
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);

        // 断言最大 token 数与每 token 条目数为正
        EP_HOST_ASSERT(new_num_max_tokens > 0 and num_entries_per_token > 0);
        // 断言层数不超过 workspace 支持的最大层数
        EP_HOST_ASSERT(new_num_entries_per_layer.size() <= layout::EngramWorkspace::kNumMaxLayers);
        // 断言每条目字节数满足 RDMA 对齐
        EP_HOST_ASSERT(new_hidden * new_elem_size % kNumRDMAAlignmentBytes == 0);
        // 累计总条目数
        int64_t num_total_entries = 0;
        // 遍历每层并校验条目数为正
        for (const auto& num_entries: new_num_entries_per_layer) {
            EP_HOST_ASSERT(num_entries > 0);
            num_total_entries += num_entries;
        }
        // 断言总条目占用的 RDMA 存储不超过分配的大小
        EP_HOST_ASSERT(num_total_entries * new_hidden * new_elem_size <= context->num_rdma_storage_bytes);

        // 计算接收侧所需字节：层数 x 最大 token 数 x 每 token 条目数 x 每条目字节数
        const auto num_recv_bytes = static_cast<int64_t>(new_num_entries_per_layer.size()) *
                                    new_num_max_tokens * num_entries_per_token * new_hidden * new_elem_size;
        // 计算 scale 因子分片字节数（无 SF 时为 0；有则按分配对齐）
        num_sf_shard_bytes = new_num_sf_packs == 0 ? 0 : math::align<int64_t>(
            num_total_entries * context->num_rdma_ranks * new_num_sf_packs * sizeof(sf_pack_t),
            kNumAllocationAlignmentBytes);
        // 断言接收数据与 SF 分片合计不超过 GPU buffer 大小
        EP_HOST_ASSERT(num_recv_bytes + num_sf_shard_bytes <= context->num_gpu_buffer_bytes);

        // 保存各层条目数
        num_entries_per_layer = new_num_entries_per_layer;
        // 保存隐藏维与元素字节数
        hidden = new_hidden, elem_size = new_elem_size;
        // 保存最大 token 数
        num_max_tokens = new_num_max_tokens;
        // 保存 SF 包数
        num_sf_packs = new_num_sf_packs;
    }

    // 把本地的专家参数写入 RDMA 存储区（可选写入 FP8 scale 因子）
    // 写入前后各执行一次屏障，保证写入的全局可见性
    void write(const std::vector<torch::Tensor>& storages,
               const std::optional<std::vector<torch::Tensor>>& sfs) {
        // 写入前屏障同步
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);

        // 获取当前计算流，后续拷贝均在该流上异步执行
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        // 计算层数
        const auto num_layers = num_entries_per_layer.size();
        // 断言传入的存储张量数量与层数一致
        EP_HOST_ASSERT(num_layers > 0 and storages.size() == num_layers);

        // RDMA 存储区的写入游标，从本地存储起始地址开始
        auto* storage_ptr = context->local_rdma_storage_ptr;
        // 逐层拷贝专家参数
        for (int i = 0; i < num_layers; ++ i) {
            // 取第 i 层的存储张量
            const auto& storage = storages[i];
            // 解包形状：条目数与隐藏维
            const auto [num_entries, storage_hidden] = get_shape<2>(storage);
            // 断言形状与配置一致
            EP_HOST_ASSERT(num_entries == num_entries_per_layer[i] and storage_hidden == hidden);
            // 断言数据类型为 BF16 或 FP8(e4m3)
            EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16 or
                           storage.scalar_type() == torch::kFloat8_e4m3fn);
            // 断言张量在 GPU 上且连续
            EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());
            // 在计算流上异步拷贝该层参数到 RDMA 存储
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                storage_ptr,
                storage.data_ptr(), storage.nbytes(),
                cudaMemcpyDefault, compute_stream));
            // 前移写入游标
            storage_ptr = math::advance_ptr(storage_ptr, storage.nbytes());
        }

        // 若提供 scale 因子，则同样分片写入 GPU buffer 末尾区域
        if (sfs.has_value()) {
            // 断言 SF 张量数量与层数一致
            EP_HOST_ASSERT(sfs->size() == num_layers);
            // 记录 SF 的数据类型（以第一层为准）
            sf_dtype = sfs->front().scalar_type();
            // SF 分片区的写入游标：GPU buffer 末尾的 num_sf_shard_bytes 区域
            auto* shard_ptr = math::advance_ptr(context->buffer,
                                                context->num_gpu_buffer_bytes - num_sf_shard_bytes);
            // 逐层拷贝 SF 分片
            for (int i = 0; i < num_layers; ++ i) {
                // 取第 i 层的 SF 张量
                const auto& sf = sfs->at(i);
                // 解包形状：行数与 SF 包数
                const auto [num_rows, num_packs] = get_shape<2>(sf);
                // 断言在 GPU、连续且元素大小等于 SF 包大小
                EP_HOST_ASSERT(sf.is_cuda() and sf.is_contiguous() and sf.element_size() == sizeof(sf_pack_t));
                // 断言 SF 包数与配置一致
                EP_HOST_ASSERT(num_packs == num_sf_packs);

                // 计算本 rank 需要拷贝的分片字节数（NVLink 域内均分）
                const auto num_shard_bytes = sf.nbytes() / context->num_nvl_ranks;
                // 在计算流上异步拷贝本 rank 的分片
                CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                    shard_ptr,
                    math::advance_ptr(sf.data_ptr(), context->nvl_rank_idx * num_shard_bytes),
                    num_shard_bytes, cudaMemcpyDefault, compute_stream));
                // 前移 SF 写入游标
                shard_ptr = math::advance_ptr(shard_ptr, num_shard_bytes);
            }
        }

        // 写入后屏障同步，保证所有 rank 的写入全局可见
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);
    }

    // 按 indices 抓取各层专家参数；返回每层一个完成钩子，钩子执行后才能读取结果
    // indices 形状为 [num_layers, num_tokens, num_entries_per_token]，元素为条目全局索引
    std::vector<std::function<pybind11::object()>>
    fetch(const torch::Tensor& indices, const int& num_qps,
          const bool& use_tma_aligned_col_major_sf) const {
        // 是否使用 FP8 抓取：由 SF 包数是否为正决定
        const auto use_fp8 = num_sf_packs > 0;
        // 抓取结果的数据类型：FP8 或 BF16
        const auto fetched_dtype = use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
        // 解包 indices 形状：层数、token 数、每 token 条目数
        const auto [num_layers, num_tokens, num_entries_per_token] = get_shape<3>(indices);
        // 断言索引类型为 int
        EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
        // 断言索引张量在 GPU 上且连续
        EP_HOST_ASSERT(indices.is_cuda() and indices.is_contiguous());
        // 断言层数与配置一致
        EP_HOST_ASSERT(num_layers == num_entries_per_layer.size());
        // 断言 token 数不超过配置上限
        EP_HOST_ASSERT(num_tokens <= num_max_tokens);

        // 每层抓取区的字节数：最大 token 数 x 每 token 条目数 x 每条目字节数
        const auto num_layer_bytes = static_cast<int64_t>(num_max_tokens) * num_entries_per_token *
                                     hidden * elem_size;
        // 各层抓取结果张量列表（以 GPU buffer 为后端的视图）
        std::vector<torch::Tensor> fetched;
        // 逐层创建抓取结果视图
        for (int i = 0; i < num_layers; ++ i)
            fetched.push_back(torch::from_blob(
                math::advance_ptr(context->buffer, i * num_layer_bytes),
                {num_tokens, hidden * num_entries_per_token},
                torch::TensorOptions().dtype(fetched_dtype).device(torch::kCUDA)
            ));

        // FP8 时各层的 SF 结果张量列表
        std::vector<torch::Tensor> fetched_sf;
        // SF 结果的底层存储（所有层共用）
        torch::Tensor sf_backing;
        // SF 分片区（源）指针
        void* sf_shards_ptr = nullptr;
        // SF 结果（目的）指针
        void* fetched_sf_ptr = nullptr;
        // SF 的 token/隐藏/层三个方向的步长
        int sf_token_stride = 0, sf_hidden_stride = 0, sf_layer_stride = 0;
        if (use_fp8) {
            if (use_tma_aligned_col_major_sf) {
                // 列主序 TMA 对齐布局：token 方向步长为 1，隐藏方向按包对齐
                sf_token_stride = 1, sf_hidden_stride = math::align(num_tokens, kNumAlignedSFPacks);
            } else {
                // 行主序布局：隐藏方向步长为 1，token 方向为每 token 的包数
                sf_token_stride = num_entries_per_token * num_sf_packs, sf_hidden_stride = 1;
            }
            // 层间步长按布局分别计算
            sf_layer_stride = use_tma_aligned_col_major_sf
                ? num_entries_per_token * num_sf_packs * sf_hidden_stride
                : num_entries_per_token * num_sf_packs * num_tokens;
            // 分配所有层共用的 SF 底层存储
            sf_backing = torch::empty({num_layers, sf_layer_stride},
                torch::TensorOptions().dtype(sf_dtype).device(torch::kCUDA));
            // 逐层创建 SF 结果的 strided 视图
            for (int i = 0; i < num_layers; ++ i)
                fetched_sf.push_back(sf_backing[i].as_strided({num_tokens, num_entries_per_token * num_sf_packs},
                                                               {sf_token_stride, sf_hidden_stride}));
            // 取 SF 结果存储的起始指针
            fetched_sf_ptr = sf_backing.data_ptr();
            // 取 SF 分片源区指针：GPU buffer 末尾的 SF 区域
            sf_shards_ptr = math::advance_ptr(context->buffer,
                                              context->num_gpu_buffer_bytes - num_sf_shard_bytes);
        }

        // 断言总 rank 数是 GIN 最小步长的整数倍
        EP_HOST_ASSERT(context->num_ranks % context->gin_min_stride == 0);
        // 计算 RDMA 对端数量
        const auto num_rdma_peers = context->num_ranks / context->gin_min_stride;
        // 断言 QP 总数不超过 workspace 支持的最大对端上下文数
        EP_HOST_ASSERT(num_rdma_peers * num_qps <= layout::EngramWorkspace::kNumMaxPeerContexts);
        // 实际激活的 GIN 上下文数：不超过 QP 数，也不超过 token 数/32
        const auto num_active_gin_contexts =
            std::min(num_qps, math::ceil_div(num_tokens, 32));

        // 分配各层各 QP 的 GIN 请求对象，用于后续等待完成
        const auto last_gin_requests = torch::empty(
            {num_layers, num_rdma_peers * num_qps, sizeof(ncclGinRequest_t)},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );

        // 启动 Engram 抓取内核：按索引从各 rank 的 RDMA 存储读取条目
        launch_engram_fetch(
            // NCCL 设备端通信句柄与窗口
            context->dev_comm, context->window,
            // 各 rank RDMA 存储的 GPU 端窗口
            context->rdma_storage_windows_gpu,
            // GPU buffer 与 workspace 地址
            context->buffer,
            context->workspace,
            // 索引张量指针
            indices.data_ptr<int>(),
            // GIN 请求对象指针
            static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
            // SF 分片源与结果目的指针
            sf_shards_ptr, fetched_sf_ptr,
            // SF 三个方向的步长
            sf_token_stride, sf_hidden_stride, sf_layer_stride,
            // NVLink 域 rank 数
            context->num_nvl_ranks,
            // 各层条目数
            num_entries_per_layer,
            // 隐藏维、元素字节数、SF 包数
            hidden, elem_size, num_sf_packs,
            // 每 token 条目数
            num_entries_per_token,
            // token 数与最大 token 数
            num_tokens, num_max_tokens,
            // RDMA 对端数
            num_rdma_peers,
            // GIN 最小步长
            context->gin_min_stride,
            // QP 数与预留后的队列深度
            num_qps, qp_depth - kFlushDepthReserve,
            // GPU 超时周期数
            context->num_gpu_timeout_cycles,
            // 在当前计算流上启动
            at::cuda::getCurrentCUDAStream()
        );

        // 为每层生成一个完成钩子：执行时等待该层抓取完成并返回结果
        std::vector<std::function<pybind11::object()>> hooks;
        // 逐层构造钩子
        for (int i = 0; i < num_layers; ++ i) {
            // 钩子捕获本层下标与上下文，执行时调用等待内核
            hooks.emplace_back([=, this]() -> pybind11::object {
                // 启动等待内核：等待本层所有 GIN 请求完成
                launch_engram_fetch_wait(
                    static_cast<ncclGinRequest_t*>(last_gin_requests[i].data_ptr()),
                    context->dev_comm,
                    context->window,
                    num_rdma_peers,
                    num_active_gin_contexts,
                    at::cuda::getCurrentCUDAStream()
                );
                // FP8 时返回 (参数, SF) 元组，否则仅返回参数张量
                return use_fp8 ? pybind11::make_tuple(fetched[i], fetched_sf[i]) : pybind11::cast(fetched[i]);
            });
        }
        // 返回各层的完成钩子
        return hooks;
    }

    // 静态计算所需存储大小：返回 (GPU buffer 字节数, RDMA 存储字节数)
    static std::tuple<int64_t, int64_t> calculate_storage_size(
            const int64_t& nccl_comm,
            const std::vector<int>& num_entries_per_layer, const int& hidden, const int& elem_size,
            const int& num_max_tokens, const int& num_entries_per_token, const int& num_sf_packs) {
        // 查询物理域大小：RDMA rank 数与 NVLink rank 数
        const auto [num_rdma_ranks, num_nvl_ranks] = comm::get_physical_domain_size(nccl_comm);
        // 汇总所有层的总条目数
        const auto num_total_entries = std::reduce(
            num_entries_per_layer.begin(), num_entries_per_layer.end(), int64_t(0));

        // 每条目字节数
        const auto num_bytes_per_entry = hidden * elem_size;
        // 断言满足 RDMA 对齐
        EP_HOST_ASSERT(num_bytes_per_entry % kNumRDMAAlignmentBytes == 0);
        // 接收区字节数：层数 x 最大 token 数 x 每 token 条目数，按分配对齐
        const auto num_recv_bytes = math::align<int64_t>(
            num_entries_per_layer.size() * num_max_tokens * num_entries_per_token * num_bytes_per_entry,
            kNumAllocationAlignmentBytes);
        // RDMA 存储字节数：总条目数 x 每条目字节数，按分配对齐
        const auto num_rdma_storage_bytes = math::align<int64_t>(
            num_total_entries * num_bytes_per_entry, kNumAllocationAlignmentBytes);

        // SF 区字节数（无 SF 时为 0；有则覆盖所有 rank 的条目并按分配对齐）
        const auto num_sf_bytes = num_sf_packs == 0 ? 0 : math::align<int64_t>(
            num_total_entries * num_rdma_ranks * num_sf_packs * sizeof(sf_pack_t),
            kNumAllocationAlignmentBytes);
        // 返回 GPU buffer 总量与 RDMA 存储量
        return {num_recv_bytes + num_sf_bytes, num_rdma_storage_bytes};
    }

    // 根据理论性能模型给出推荐的 QP 配置：返回 (QP 数, QP 深度, 是否需要 flush)
    static std::tuple<int, int, bool> get_theoretical_config(const int64_t& nccl_comm,
                                                       const int& num_layers,
                                                       const int& num_max_tokens,
                                                       const int& num_entries_per_token,
                                                       const float& nic_read_mpps,
                                                       const int& num_max_read_requests_per_qp) {
        // 假设的读取往返时延：10 微秒
        constexpr float kReadRTTSeconds = 10e-6f;

        // 断言所有输入参数均为正
        EP_HOST_ASSERT(num_layers > 0 and num_max_tokens > 0 and num_entries_per_token > 0 and
                       nic_read_mpps > 0 and num_max_read_requests_per_qp > 0);

        // 查询 GIN 最小步长（对端数量）
        const auto [_, num_rdma_peers] = comm::get_gin_min_stride(nccl_comm);

        // 带宽时延积模型：计算满足吞吐所需的 QP 数（至少为 1）
        const auto num_qps = std::max(static_cast<int>(std::ceil(
            nic_read_mpps * 1e6f * kReadRTTSeconds / num_max_read_requests_per_qp / num_rdma_peers)), 1);

        // 计算总请求数
        const auto num_requests = num_layers * num_max_tokens * num_entries_per_token;
        // 每个 QP 槽位需要承载的请求数
        const auto num_required_slots = math::ceil_div(num_requests, num_qps * num_rdma_peers);
        // 超过硬件上限 32768 时：以最小可行 QP 数返回并要求 flush
        if (num_required_slots > 32768) {
            // 按上限反推最少 QP 数并留 5% 余量
            const auto num_min_qps = static_cast<int>(math::ceil_div(num_requests, 32768 * num_rdma_peers) * 1.05);
            // 返回最少 QP 数、最大槽位深度、需要 flush
            return {num_min_qps, 32768, true};
        }

        // 否则从默认深度开始，按 2 的幂扩展到覆盖所需槽位（不超过 32768）
        auto qp_depth = kDefaultQPDepth;
        while (qp_depth < num_required_slots and qp_depth < 32768) qp_depth *= 2;
        // 返回计算的 QP 数与深度，无需 flush
        return {num_qps, qp_depth, false};
    }

    // 创建共享 RDMA 存储的句柄（跨进程共享物理内存用）
    static symmetric::shared_handle_t create_shared_handle(const int64_t& num_rdma_storage_bytes,
                                                           const bool& use_cpu_rdma_storage) {
        // 直接委托给对称内存管理器的工厂方法
        return symmetric::HybridElasticSymmetricMemory::create_shared_handle(num_rdma_storage_bytes, use_cpu_rdma_storage);
    }
};

// 向 Python 模块注册 Engram 缓冲区的绑定接口
static void register_apis(pybind11::module_& m) {
    // 将 EngramBuffer 暴露给 Python，并绑定构造函数
    pybind11::class_<EngramBuffer, BufferBase>(m, "EngramBuffer")
        .def(pybind11::init<int, int, int64_t, symmetric::shared_comm_t, int64_t, int64_t, bool, bool, std::optional<int>, int, int, int, bool>())
        // 只读属性：通信上下文
        .def_readonly("context", &EngramBuffer::context)
        // 方法：设置配置
        .def("set_config", &EngramBuffer::set_config)
        // 方法：写入参数
        .def("write", &EngramBuffer::write)
        // 方法：按索引抓取参数
        .def("fetch", &EngramBuffer::fetch)
        // 静态方法：创建共享句柄
        .def_static("create_shared_handle", &EngramBuffer::create_shared_handle)
        // 静态方法：计算存储大小
        .def_static("calculate_storage_size", &EngramBuffer::calculate_storage_size)
        // 静态方法：获取理论最优配置
        .def_static("get_theoretical_config", &EngramBuffer::get_theoretical_config);
}

}  // namespace deep_ep::engram
