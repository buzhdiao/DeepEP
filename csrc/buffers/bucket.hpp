#pragma once

// 引入 std::array
#include <array>
// 引入数学函数
#include <cmath>
// 引入固定宽度整型
#include <cstdint>
// 引入 std::function
#include <functional>
// 引入 std::shared_ptr
#include <memory>
// 引入 std::optional
#include <optional>
// 引入 std::string
#include <string>
// 引入 std::tuple
#include <tuple>
// 引入 std::vector
#include <vector>

// 引入 pybind11 的 std::function 支持
#include <pybind11/functional.h>
// 引入 pybind11 的 STL 容器转换支持
#include <pybind11/stl.h>
// 引入 PyTorch 的 Python 绑定支持
#include <torch/python.h>

// 引入 DeepEP 公共编译常量
#include <deep_ep/common/compiled.cuh>
// 引入异常与宿主断言宏
#include <deep_ep/common/exception.cuh>
// 引入对齐等数学工具
#include <deep_ep/common/math.cuh>
// 引入通用屏障布局定义
#include <deep_ep/layout/common/barrier.cuh>
// 引入 bucket workspace 布局定义
#include <deep_ep/layout/bucket/workspace.cuh>
// 引入 bucket reduce-scatter 布局定义
#include <deep_ep/layout/bucket/reduce_scatter.cuh>

// 引入缓冲区基类 BufferBase
#include "base.hpp"
// 引入 NCCL 通信上下文与公共通信 API
#include "../comm/api.hpp"
// 引入屏障函数
#include "../comm/barrier.hpp"
// 引入通信流工具
#include "../comm/stream.hpp"
// 引入 bucket all-gather 内核的启动封装
#include "../kernels/bucket/all_gather.hpp"
// 引入 bucket all-reduce 内核的启动封装
#include "../kernels/bucket/all_reduce.hpp"
// 引入 bucket reduce-scatter 内核的启动封装
#include "../kernels/bucket/reduce_scatter.hpp"
// 引入 JIT 运行时
#include "../runtime/jit.hpp"
// 引入事件句柄
#include "../utils/event.hpp"

// bucket 式集合通信（reduce-scatter / all-reduce / all-gather）所在的命名空间
namespace deep_ep::bucket {

// Bucket 缓冲区：把多组张量组织成桶（bucket）批量执行集合通信
class BucketBuffer: public BufferBase {
public:
    // 多个通信组的上下文列表（支持一个缓冲区管理多个 NCCL 通信域）
    std::vector<std::shared_ptr<comm::Context>> contexts;
    // 以主上下文 buffer 为后端的存储张量（所有桶的输入输出都必须落在此区域）
    torch::Tensor storage;

private:
    // 存储区的总字节数
    int64_t num_storage_bytes;
    // NVLink 与 RDMA 的带宽（字节/秒），用于 all-gather 的自适应参数
    float nvl_bandwidth, rdma_bandwidth;

    // Allocator for operation indices in sequence
    // 按顺序为各上下文分配操作序号的分配器
    mutable std::array<uint64_t, kNumMaxContexts> seq_indices = {};

    // 流控制收尾：在通信流上记录事件并登记张量
    static EventHandle stream_control_epilogue(const std::vector<torch::Tensor>& tensors = {}) {
        // 获取全局唯一的通信流
        const auto comm_stream = comm::get_comm_stream();
        // 在通信流上创建事件句柄
        auto event = EventHandle(comm_stream);
        if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
            // 环境变量开启时：延迟登记，把张量列表挂到事件上由 Python 侧处理
            event.tensors_to_record.assign(tensors.begin(), tensors.end());
        } else {
            // 默认路径：对每个张量在通信流上做 record_stream 保护
            for (const auto& tensor: tensors)
                tensor.record_stream(comm_stream);
        }
        // 返回事件句柄
        return event;
    }

public:
    // 构造函数：为每个通信组创建上下文并校验组号一致性
    BucketBuffer(const std::vector<int>& rank_indices,
                 const std::vector<int>& num_ranks,
                 const std::vector<int64_t>& nccl_comms,
                 const int64_t& num_storage_bytes,
                 const std::optional<int>& sl_idx,
                 const int& num_gpu_timeout_secs,
                 const float& nvl_gbs, const float& rdma_gbs,
                 const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),
        num_storage_bytes(num_storage_bytes),
        // 把用户提供的 Gbps 带宽转换为字节/秒
        nvl_bandwidth(nvl_gbs * 1e9f), rdma_bandwidth(rdma_gbs * 1e9f) {
        // 断言带宽参数为正
        EP_HOST_ASSERT(nvl_gbs > 0 and rdma_gbs > 0);
        // 计算通信组数量
        const auto num_groups = static_cast<int>(nccl_comms.size());
        // 断言组数合法且各组参数长度匹配
        EP_HOST_ASSERT(num_groups > 0 and rank_indices.size() == num_groups and num_ranks.size() == num_groups);
        // 断言组数不超过 workspace 支持的最大上下文数
        EP_HOST_ASSERT(num_groups <= kNumMaxContexts and
                       "Too many contexts for the bucket workspace");
        // 断言存储大小为正且按分配对齐
        EP_HOST_ASSERT(num_storage_bytes > 0 and num_storage_bytes % kNumAllocationAlignmentBytes == 0);

        // 获取当前计算流（后续 NCCL all-gather 校验用它）
        const auto stream = at::cuda::getCurrentCUDAStream();
        // 计算 workspace 大小并按分配对齐
        const auto num_workspace_bytes = math::align<int64_t>(
            sizeof(layout::BucketWorkspace),
            kNumAllocationAlignmentBytes);
        // 预留上下文向量容量
        contexts.reserve(num_groups);
        // 逐组创建通信上下文
        for (int i = 0; i < num_groups; ++ i) {
            // 查询该通信域的物理域大小
            const auto [num_rdma_ranks, num_nvl_ranks] = comm::get_physical_domain_size(nccl_comms[i]);
            // 单 RDMA 域时分配 1 个 QP，否则按最大 QP 数均分
            const auto num_allocated_qps = num_rdma_ranks == 1 ? 1 : kNumMaxQPs / num_rdma_ranks;
            // 创建该组的上下文；除第一组外共享第一组的存储句柄
            contexts.emplace_back(std::make_shared<comm::Context>(
                nccl_comms[i], symmetric::shared_comm_t{}, num_ranks[i], rank_indices[i],
                num_workspace_bytes, num_storage_bytes, 0, true,
                true, sl_idx, num_allocated_qps,
                0, 0, num_gpu_timeout_secs, num_nvl_ranks > 1,
                contexts.empty() ? nullptr : contexts.front()));
            // 获取该组 workspace 的引用
            auto& workspace = *static_cast<layout::BucketWorkspace*>(contexts.back()->workspace);
            // 注册该组的屏障信号区
            contexts.back()->set_barrier_signals(&workspace.signals[i].barrier);

            // All ranks of a group must have the same context id
            // 同一组的所有 rank 必须得到相同的上下文 id
            // 分配接收 all-gather 结果的张量
            const auto gathered = torch::empty({num_ranks[i]}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
            // 构造只含本组 id 的发送张量
            const auto send = torch::full({1}, i, gathered.options());
            // 通过 NCCL all-gather 交换各 rank 的组 id
            NCCL_CHECK(ncclAllGather(send.data_ptr<int>(), gathered.data_ptr<int>(), 1, ncclInt, reinterpret_cast<ncclComm_t>(nccl_comms[i]), stream));
            // 断言所有 rank 的组 id 一致
            EP_HOST_ASSERT(torch::eq(gathered, i).all().item<bool>());
        }
        // 以第一组作为主上下文
        main_context = contexts.front();
        // 以主上下文的 buffer 为后端创建字节类型存储视图
        storage = torch::from_blob(
            main_context->buffer,
            {num_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    }

    // 析构函数：按销毁策略处理资源释放
    ~BucketBuffer() noexcept(false) override {
        // 基类辅助：必要时自动调用 destroy()，否则打印泄漏警告
        destroy_on_destruction("bucket");
    }

    // 对一组桶执行 reduce-scatter：每个 rank 获得自己分片的归约结果
    // 返回 (各分片张量, 事件, 使用的 SM 数)
    std::tuple<std::vector<torch::Tensor>, EventHandle, int> reduce_scatter(
        const std::vector<torch::Tensor>& srcs,
        const int& context_idx,
        const int& num_sms,
        const std::string& comm_precision,
        const float& scale) const {
        // 断言缓冲区尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // Check config
        // 检查配置参数
        // 桶的数量即输入张量数量
        const auto num_buckets = static_cast<int>(srcs.size());
        // 断言桶数合法
        EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
        // 断言上下文索引有效
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        // 断言通信精度只支持 fp32 或 bf16
        EP_HOST_ASSERT((comm_precision == "fp32" or comm_precision == "bf16") and
                       "Reduce-scatter comm_precision must be fp32 or bf16");
        // 断言 SM 数为正（未实现 0-SM 版本）
        EP_HOST_ASSERT(num_sms > 0 and "0-SM reduce-scatter is not implemented");
        // 断言 SM 数不超过设备上限
        EP_HOST_ASSERT(num_sms <= jit->device.get_num_sms());

        // Check context
        // 检查上下文
        // 取出本次操作使用的通信上下文
        const auto& context = contexts[context_idx];

        // Create bucket list
        // 构建桶列表
        // 存储区的起始地址
        const auto storage_begin = reinterpret_cast<uintptr_t>(storage.data_ptr());
        // 存储区的结束地址
        const auto storage_end = storage_begin + num_storage_bytes;
        // 桶描述列表（偏移 + 字节数）
        comm::BucketList buckets;
        // 输出分片张量列表
        std::vector<torch::Tensor> dsts;
        // 预留输出列表容量
        dsts.reserve(srcs.size());
        // 逐桶校验并登记
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            // TODO: support other dtypes
            // TODO: 支持其他数据类型
            // 取第 bucket_idx 个源张量
            const auto& src = srcs[bucket_idx];
            // 断言张量在 GPU 上且连续
            EP_HOST_ASSERT(src.is_cuda() and src.is_contiguous());
            // 断言为 float 类型且非空
            EP_HOST_ASSERT(src.scalar_type() == torch::kFloat and src.numel() > 0);
            // 断言字节数按 rank 数与 longlong4 对齐（内核要求）
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * sizeof(longlong4_t)) == 0);
            // 断言字节数满足 TMA 对齐
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * kNumTMAAlignmentBytes) == 0);

            // TODO: check overlapping
            // TODO: 检查内存重叠
            // 源张量的起始地址
            const auto src_begin = reinterpret_cast<uintptr_t>(src.data_ptr());
            // 源张量的结束地址
            const auto src_end = src_begin + src.nbytes();
            // 断言源张量必须落在本缓冲区的存储区内
            EP_HOST_ASSERT(src_begin >= storage_begin and src_end <= storage_end and
                           "Input tensors must belong to this BucketBuffer");
            // 断言起始地址按 longlong4 对齐
            EP_HOST_ASSERT(src_begin % alignof(longlong4_t) == 0);
            // 断言起始地址满足 TMA 对齐
            EP_HOST_ASSERT(src_begin % kNumTMAAlignmentBytes == 0);

            // TODO: support >1D slicing
            // TODO: 支持多维切片
            // 计算每个 rank 的分片元素数
            const auto shard_numel = src.numel() / context->num_ranks;
            // 切出本 rank 的分片视图
            auto dst = src.view({-1}).slice(
                0, context->rank_idx * shard_numel, (context->rank_idx + 1) * shard_numel);

            // Create bucket (shard view)
            // 创建桶（分片视图）
            buckets[bucket_idx] = {
                // 桶在存储区内的字节偏移
                .offset = static_cast<int64_t>(src_begin - storage_begin),
                // 本 rank 分片的字节数
                .num_bytes = static_cast<int64_t>(dst.nbytes()),
            };
            // 记录输出分片
            dsts.push_back(dst);
        }

        // Launch
        // 启动内核
        // 获取当前计算流
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        // 获取通信流
        const auto comm_stream = comm::get_comm_stream();
        // 让通信流等待计算流上的已有工作
        comm::stream_wait(comm_stream, compute_stream);
        // 启动 reduce-scatter 内核
        launch_reduce_scatter(
            // NCCL 设备端通信句柄与窗口
            context->dev_comm, context->window,
            // workspace、buffer 与 multimem 地址
            context->workspace, context->buffer, context->buffer_multimem,
            // 桶数与桶列表
            num_buckets, buckets,
            // RDMA 与 NVLink 域 rank 数
            context->num_rdma_ranks, context->num_nvl_ranks,
            // SM 数、QP 数与超时周期数
            num_sms, context->num_allocated_qps, context->num_gpu_timeout_cycles,
            // 缩放因子、通信精度、上下文索引与通信流
            scale, comm_precision, context_idx, comm_stream);

        // 在通信流上记录事件并登记张量
        auto event = stream_control_epilogue();
        // 返回分片张量、事件与 SM 数
        return {dsts, event, num_sms};
    }

    // 对一组桶执行原位 all-reduce：归约结果直接覆盖源张量
    // 返回 (各输出张量视图, 事件, 使用的 SM 数)
    std::tuple<std::vector<torch::Tensor>, EventHandle, int> all_reduce(
        const std::vector<torch::Tensor>& srcs,
        const int& context_idx,
        const int& num_sms,
        const float& scale) const {
        // 断言缓冲区尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // Check config
        // 检查配置参数
        // 桶的数量即输入张量数量
        const auto num_buckets = static_cast<int>(srcs.size());
        // 断言桶数合法
        EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
        // 断言上下文索引有效
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        // 断言 SM 数为正（未实现 0-SM 版本）
        EP_HOST_ASSERT(num_sms > 0 and "0-SM all-reduce is not implemented");
        // 断言 SM 数不超过设备上限
        EP_HOST_ASSERT(num_sms <= jit->device.get_num_sms());

        // Check context
        // 检查上下文
        // 取出本次操作使用的通信上下文
        const auto& context = contexts[context_idx];
        // 断言单 NVLink 域时无需 multimem，多域时必须支持 multimem
        EP_HOST_ASSERT(context->num_nvl_ranks == 1 or context->buffer_multimem != nullptr);

        // Create bucket list
        // 构建桶列表
        // 存储区的起始地址
        const auto storage_begin = reinterpret_cast<uintptr_t>(storage.data_ptr());
        // 存储区的结束地址
        const auto storage_end = storage_begin + num_storage_bytes;
        // 桶描述列表
        comm::BucketList buckets;
        // 输出张量视图列表
        std::vector<torch::Tensor> dsts;
        // 预留输出列表容量
        dsts.reserve(num_buckets);
        // 逐桶校验并登记
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            // 取第 bucket_idx 个源张量
            const auto& src = srcs[bucket_idx];
            // 断言张量在 GPU 上且连续
            EP_HOST_ASSERT(src.is_cuda() and src.is_contiguous());
            // 断言为 float 类型且非空
            EP_HOST_ASSERT(src.scalar_type() == torch::kFloat and src.numel() > 0);
            // Each rank reduces one shard (numel / num_ranks) of float4 vectors
            // 每个 rank 归约 float4 向量的一小片（numel / num_ranks）
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * sizeof(float4)) == 0);
            // 断言字节数满足 TMA 对齐
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * kNumTMAAlignmentBytes) == 0);

            // 源张量的起始地址
            const auto src_begin = reinterpret_cast<uintptr_t>(src.data_ptr());
            // 源张量的结束地址
            const auto src_end = src_begin + src.nbytes();
            // 断言源张量必须落在本缓冲区的存储区内
            EP_HOST_ASSERT(src_begin >= storage_begin and src_end <= storage_end and
                           "Input tensors must belong to this BucketBuffer");
            // 断言起始地址按 float4 对齐
            EP_HOST_ASSERT(src_begin % alignof(float4) == 0);
            // 断言起始地址满足 TMA 对齐
            EP_HOST_ASSERT(src_begin % kNumTMAAlignmentBytes == 0);

            // In-place all-reduce: the full reduced tensor overwrites the source
            // Create bucket (full view)
            // 原位 all-reduce：完整归约结果覆盖源张量
            // 创建桶（全量视图）
            buckets[bucket_idx] = {
                // 桶在存储区内的字节偏移
                .offset = static_cast<int64_t>(src_begin - storage_begin),
                // 整个源张量的字节数
                .num_bytes = static_cast<int64_t>(src.nbytes()),
            };
            // 输出即源张量的展平视图（原位）
            dsts.push_back(src.view({-1}));
        }

        // Launch
        // 启动内核
        // 获取当前计算流
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        // 获取通信流
        const auto comm_stream = comm::get_comm_stream();
        // 让通信流等待计算流上的已有工作
        comm::stream_wait(comm_stream, compute_stream);
        // 启动 all-reduce 内核
        launch_all_reduce(
            // 通信上下文、桶数与桶列表、上下文索引
            *context, num_buckets, buckets, context_idx,
            // SM 数、超时周期数与缩放因子
            num_sms, context->num_gpu_timeout_cycles, scale, comm_stream);

        // 在通信流上记录事件并登记张量
        auto event = stream_control_epilogue();
        // 返回输出张量、事件与 SM 数
        return {dsts, event, num_sms};
    }

    // In-place without `dsts`, otherwise `srcs` can be out of the storage (pure NVLink only)
    // 原位模式时不提供 dsts（否则 srcs 可以位于存储区之外，仅限纯 NVLink 场景）
    // 对一组桶执行 all-gather：把各 rank 的分片拼成完整张量
    pybind11::tuple all_gather(
        const std::vector<torch::Tensor>& srcs,
        const std::vector<torch::Tensor>& dsts,
        const int& context_idx,
        const int& num_sms) const {
        // 断言缓冲区尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // Check config
        // 检查配置参数
        // 桶的数量即输入张量数量
        const auto num_buckets = static_cast<int>(srcs.size());
        // 断言桶数合法
        EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
        // 断言输出列表为空（原位）或与桶数一致
        EP_HOST_ASSERT(dsts.empty() or static_cast<int>(dsts.size()) == num_buckets);
        // 断言上下文索引有效
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        // 断言 SM 数必须为 0（all-gather 使用自适应/多线程路径）
        EP_HOST_ASSERT(num_sms == 0);

        // Check context
        // 检查上下文
        // 取出本次操作使用的通信上下文
        const auto& context = contexts[context_idx];

        // Create bucket list
        // 构建桶列表
        // 桶描述列表（输出侧）
        comm::BucketList buckets;
        // 输入侧桶列表：仅在提供外部 dsts 时启用
        auto src_buckets = dsts.empty() ? std::nullopt : std::make_optional<comm::BucketList>();
        // 汇聚结果的张量视图列表
        std::vector<torch::Tensor> gathered;
        // 预留容量
        gathered.reserve(num_buckets);
        // 逐桶校验并登记
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            // 取第 bucket_idx 个源张量
            const auto& src = srcs[bucket_idx];
            // 断言张量在 GPU 上且连续
            EP_HOST_ASSERT(src.is_cuda() and src.is_contiguous());
            // 断言张量非空
            EP_HOST_ASSERT(src.numel() > 0);
            // 若提供输出，则断言输出同样在 GPU 上且连续
            EP_HOST_ASSERT(dsts.empty() or (dsts[bucket_idx].is_cuda() and dsts[bucket_idx].is_contiguous()));

            // 分片的字节数即源张量字节数
            const auto num_shard_bytes = static_cast<int64_t>(src.nbytes());
            // 源张量相对存储区基址的偏移（可为负，表示在存储区之外）
            const auto src_offset = math::ptr_diff(src.data_ptr(), storage.data_ptr());
            // 桶（输出区）的偏移：原位时由源偏移减去本 rank 之前的分片推出，否则取 dst 的偏移
            const auto bucket_offset = dsts.empty() ? src_offset - context->rank_idx * num_shard_bytes :
                                                     math::ptr_diff(dsts[bucket_idx].data_ptr(), storage.data_ptr());
            // 若提供输出，则断言输出大小等于 rank 数 x 分片大小
            EP_HOST_ASSERT(dsts.empty() or static_cast<int64_t>(dsts[bucket_idx].nbytes()) == context->num_ranks * num_shard_bytes);
            // 断言源与输出不重叠，除非源恰好是输出的本地分片
            EP_HOST_ASSERT((src_offset == bucket_offset + context->rank_idx * num_shard_bytes or
                            src_offset + num_shard_bytes <= bucket_offset or
                            src_offset >= bucket_offset + context->num_ranks * num_shard_bytes) and
                           "Input tensors overlapping with the output must be the local shard");
            // 断言输出桶完整落在存储区内
            EP_HOST_ASSERT(bucket_offset >= 0 and
                           bucket_offset + context->num_ranks * num_shard_bytes <= num_storage_bytes and
                           "Output tensors must belong to this BucketBuffer");

            // Create bucket (shard view)
            // 创建桶（分片视图）
            buckets[bucket_idx] = {
                // 桶在存储区内的字节偏移
                .offset = bucket_offset,
                // 单个分片的字节数
                .num_bytes = num_shard_bytes,
            };
            if (src_buckets.has_value()) {
                // External sources must be decoded against the selected context's buffer
                // 外部源必须相对所选上下文的 buffer 解码偏移
                // 判断源张量是否完整落在存储区内
                const auto src_in_buffer = src_offset >= 0 and src_offset + num_shard_bytes <= num_storage_bytes;
                // 登记输入桶：在存储区内用存储偏移，否则用相对上下文 buffer 的偏移
                (*src_buckets)[bucket_idx] = {
                    .offset = src_in_buffer ? src_offset : math::ptr_diff(src.data_ptr(), context->buffer),
                    .num_bytes = num_shard_bytes,
                };
            }

            // 创建汇聚结果视图：指向输出桶起点的完整长度张量
            gathered.push_back(torch::from_blob(
                math::advance_ptr(storage.data_ptr(), bucket_offset),
                {src.numel() * context->num_ranks},
                torch::TensorOptions().dtype(src.dtype()).device(src.device())));
        }

        // Launch
        // 启动内核
        // 获取当前计算流
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        // 获取通信流
        const auto comm_stream = comm::get_comm_stream();
        // 让通信流等待计算流上的已有工作
        comm::stream_wait(comm_stream, compute_stream);
        // 启动 all-gather 内核（带带宽参数以自适应调度）
        launch_all_gather(*context, num_buckets, buckets, src_buckets, context_idx,
                          seq_indices[context_idx], nvl_bandwidth, rdma_bandwidth, comm_stream);

        // Record and epilogue
        // 登记源张量并生成事件
        auto event = stream_control_epilogue(srcs);
        // 收尾钩子：RDMA 域大于 1 时需要先做屏障确认远端写入完成，再返回结果
        std::function epilogue = [context, gathered = std::move(gathered)]() {
            if (context->num_rdma_ranks > 1) {
                // NOTES: the hybrid path confirms every put by the tail signals before the barrier, no flush needed
                // 注意：混合路径在屏障前通过尾部信号确认每一次 put，无需额外 flush
                comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                              context->num_gpu_timeout_cycles, false, true, context->num_nvl_ranks == 1);
            }
            // 返回汇聚结果张量列表
            return pybind11::cast(gathered);
        };
        // 返回事件与收尾钩子
        return pybind11::make_tuple(event, epilogue);
    }

    // 销毁缓冲区：逐组屏障同步后释放全部上下文
    void destroy() override {
        // 断言缓冲区尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // 逐组执行屏障，确保所有 rank 都完成工作
        for (int context_idx = 0; context_idx < contexts.size(); ++ context_idx) {
            // 取出该组上下文
            const auto& context = contexts[context_idx];
            // 执行组内屏障
            comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                          context->num_gpu_timeout_cycles, true, true);
        }
        // 逐组结束并释放通信上下文
        for (const auto& item: contexts)
            item->finalize();

        // 释放存储张量
        storage = torch::Tensor();
        // 清空上下文列表
        contexts.clear();
        // 标记已销毁
        destroyed = true;
    }
};

// 向 Python 模块注册 Bucket 缓冲区的绑定接口
static void register_apis(pybind11::module_& m) {
    // 将 BucketBuffer 暴露给 Python，并绑定带命名参数的构造函数
    pybind11::class_<BucketBuffer, BufferBase>(m, "BucketBuffer")
        .def(pybind11::init<std::vector<int>, std::vector<int>, std::vector<int64_t>, int64_t, std::optional<int>, int, float, float, bool>(),
             py::arg("rank_indices"),  // 参数：各组本 rank 编号
             py::arg("num_ranks"),  // 参数：各组总 rank 数
             py::arg("nccl_comms"),  // 参数：各组 NCCL 通信域句柄
             py::arg("num_buffer_bytes"),  // 参数：存储区字节数
             py::arg("sl_idx"),  // 参数：可选的 SL 索引
             py::arg("num_gpu_timeout_secs"),  // 参数：GPU 超时秒数
             py::arg("nvl_gbs"), py::arg("rdma_gbs"),  // 参数：NVLink 与 RDMA 带宽（Gbps）
             py::arg("explicitly_destroy") = false)  // 参数：是否要求显式销毁
        // 只读属性：各通信组上下文
        .def_readonly("contexts", &BucketBuffer::contexts)
        // 只读属性：存储张量
        .def_readonly("storage", &BucketBuffer::storage)
        // 方法：reduce-scatter（可指定通信精度与缩放因子）
        .def("reduce_scatter", &BucketBuffer::reduce_scatter,
             py::arg("srcs"),  // 参数：源张量列表
             py::arg("context_idx"),  // 参数：上下文索引
             py::arg("num_sms"),  // 参数：SM 数
             py::arg("comm_precision") = "fp32",  // 参数：通信精度，默认 fp32
             py::arg("scale") = 1.0f)  // 参数：缩放因子，默认 1
        // 方法：原位 all-reduce
        .def("all_reduce", &BucketBuffer::all_reduce,
             py::arg("srcs"),  // 参数：源张量列表
             py::arg("context_idx"),  // 参数：上下文索引
             py::arg("num_sms"),  // 参数：SM 数
             py::arg("scale") = 1.0f)  // 参数：缩放因子，默认 1
        // 方法：all-gather（支持原位或外部输出）
        .def("all_gather", &BucketBuffer::all_gather,
             py::arg("srcs"),  // 参数：源张量列表
             py::arg("dsts"),  // 参数：输出张量列表（可为空表示原位）
             py::arg("context_idx") = 0,  // 参数：上下文索引，默认 0
             py::arg("num_sms") = 0);  // 参数：SM 数，必须为 0
}

}  // namespace deep_ep::bucket
