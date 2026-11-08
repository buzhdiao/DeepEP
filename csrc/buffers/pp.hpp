#pragma once

// 引入标准算法（本文件主要为对齐计算）
#include <algorithm>
// 引入固定宽度整型
#include <cstdint>
// 引入 std::shared_ptr
#include <memory>
// 引入 std::optional
#include <optional>
// 引入 std::vector
#include <vector>

// 引入 pybind11 的 STL 容器转换支持
#include <pybind11/stl.h>
// 引入 PyTorch 的 Python 绑定支持
#include <torch/python.h>

// 引入 DeepEP 公共编译常量
#include <deep_ep/common/compiled.cuh>
// 引入异常与宿主断言宏
#include <deep_ep/common/exception.cuh>
// 引入对齐计算等数学工具
#include <deep_ep/common/math.cuh>
// 引入 PP 信号区布局定义
#include <deep_ep/layout/pp/workspace.cuh>

// 引入缓冲区基类 BufferBase
#include "base.hpp"
// 引入 NCCL 通信上下文与公共通信 API
#include "../comm/api.hpp"
// 引入屏障函数
#include "../comm/barrier.hpp"
// 引入 PP send/recv 内核的启动封装
#include "../kernels/pp/pp_send_recv.hpp"
// 引入 JIT 运行时（查询设备属性）
#include "../runtime/jit.hpp"

// 流水线并行（pipeline parallelism）通信所在的命名空间
namespace deep_ep::pp {

// PP 缓冲区：在环形相邻 rank 之间做点对点 send/recv 的缓冲区封装
class PPBuffer: public BufferBase {
public:
    // NCCL 通信上下文（包含 workspace、buffer、窗口等）
    std::shared_ptr<comm::Context> context;
    // 以通信 buffer 为后端的存储张量（字节视图）
    torch::Tensor storage;

private:
    // PP settings
    // PP 相关配置
    // 单个张量的最大字节数（已对齐）
    int64_t num_max_tensor_bytes;
    // 允许同时在途（in-flight）的最大张量个数
    int num_max_inflight_tensors;

public:
    // 构造函数：创建通信上下文并分配存储
    PPBuffer(const int& rank_idx, const int& num_ranks, const int64_t& nccl_comm,
             const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors,
             const std::optional<int>& sl_idx,
             const int& num_gpu_timeout_secs,
             const bool& explicitly_destroy):
        BufferBase(explicitly_destroy) {
        // 断言张量大小上限与在途张量数均为正
        EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);

        // Each send/recv buffer contains the previous and next rank in the ring
        // 每个 send/recv 缓冲区同时覆盖环形上相邻的前一个与后一个 rank
        // 对齐并保存单张量最大字节数（按 RDMA 对齐）
        this->num_max_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, kNumRDMAAlignmentBytes);
        // 保存最大在途张量数
        this->num_max_inflight_tensors = num_max_inflight_tensors;
        // 计算总存储大小：环形两侧共 4 份缓冲区 x 在途张量数，再按分配对齐
        const auto num_storage_bytes = math::align<int64_t>(
            4 * this->num_max_tensor_bytes * this->num_max_inflight_tensors,
            kNumAllocationAlignmentBytes);

        // 计算 workspace（PP 信号区）大小并按分配对齐
        const auto num_workspace_bytes = math::align<int64_t>(
            sizeof(layout::PPSignals),
            kNumAllocationAlignmentBytes);
        // 查询物理域大小以决定 QP 分配数量
        const auto [num_rdma_ranks, _] = comm::get_physical_domain_size(nccl_comm);
        // 单 RDMA 域时分配 1 个 QP，否则按最大 QP 数均分
        const auto num_allocated_qps = num_rdma_ranks == 1 ? 1 : kNumMaxQPs / num_rdma_ranks;
        // 创建 NCCL 通信上下文：workspace 与存储大小、超时等参数一并传入
        context = std::make_shared<comm::Context>(
            nccl_comm, symmetric::shared_comm_t{}, num_ranks, rank_idx,
            num_workspace_bytes, num_storage_bytes, 0, true,
            false, sl_idx, num_allocated_qps,
            0, 0, num_gpu_timeout_secs, false);
        // 设置基类中的主上下文
        main_context = context;
        // 以通信 buffer 为后端创建字节类型的存储张量视图
        storage = torch::from_blob(
            context->buffer,
            {num_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

        // 获取 workspace 中的 PP 信号区引用
        auto& pp = *static_cast<layout::PPSignals*>(context->workspace);
        // 注册屏障信号区，供 destroy 时同步使用
        context->set_barrier_signals(&pp.barrier);

        // We should call a barrier at the end
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
        // 最后应该调用一次 barrier
        // 这个 barrier 应当由 Python 的 `dist.barrier` 来调用
        // 注意：这里不要调用我们自己的 barrier，因为 workspace 还没有就绪
    }

    // 析构函数：按销毁策略处理资源释放
    ~PPBuffer() noexcept(false) override {
        // 基类辅助：必要时自动调用 destroy()，否则打印泄漏警告
        destroy_on_destruction("PP");
    }

    // 向环形上相邻的下游 rank 异步发送张量 x
    void send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
        // 断言缓冲区尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // Check input
        // 检查输入张量
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() > 0);
        // 断言地址与大小满足 TMA 对齐，且不超过单张量上限
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(x.data_ptr()) % kNumTMAAlignmentBytes == 0 and
                       x.nbytes() % kNumTMAAlignmentBytes == 0 and x.nbytes() <= num_max_tensor_bytes);
        // 断言目标 rank 必须是环形上的相邻 rank（前一个或后一个）
        EP_HOST_ASSERT((dst_rank_idx == (context->rank_idx + 1) % context->num_ranks or
                        dst_rank_idx == (context->rank_idx + context->num_ranks - 1) % context->num_ranks) and
                       "PP send requires an adjacent rank in the ring");

        // 启动 PP 发送内核；未指定 SM 数时使用设备的全部 SM
        launch_pp_send(*context, x.data_ptr(), x.nbytes(), dst_rank_idx,
                       num_max_tensor_bytes, num_max_inflight_tensors,
                       num_sms == 0 ? jit->device.get_num_sms() : num_sms,
                       at::cuda::getCurrentCUDAStream());
    }

    // 从环形上相邻的上游 rank 异步接收数据到张量 x
    void recv(const torch::Tensor& x, const int& src_rank_idx, const int& num_sms) const {
        // 断言缓冲区尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // Check input
        // 检查输入张量
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() > 0);
        // 断言地址与大小满足 TMA 对齐，且不超过单张量上限
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(x.data_ptr()) % kNumTMAAlignmentBytes == 0 and
                       x.nbytes() % kNumTMAAlignmentBytes == 0 and x.nbytes() <= num_max_tensor_bytes);
        // 断言源 rank 必须是环形上的相邻 rank（前一个或后一个）
        EP_HOST_ASSERT((src_rank_idx == (context->rank_idx + 1) % context->num_ranks or
                        src_rank_idx == (context->rank_idx + context->num_ranks - 1) % context->num_ranks) and
                       "PP recv requires an adjacent rank in the ring");

        // 启动 PP 接收内核；未指定 SM 数时使用设备的全部 SM
        launch_pp_recv(*context, x.data_ptr(), x.nbytes(), src_rank_idx,
                       num_max_tensor_bytes, num_max_inflight_tensors,
                       num_sms == 0 ? jit->device.get_num_sms() : num_sms,
                       at::cuda::getCurrentCUDAStream());
    }

    // 销毁缓冲区：先同步再释放所有资源
    void destroy() override {
        // 断言缓冲区尚未销毁
        EP_HOST_ASSERT(not destroyed);

        // Flush outstanding RDMA operations before deregistering, as a peer may still signal us
        // 在注销内存注册之前先冲刷所有未完成的 RDMA 操作，因为对端可能仍在向我们发信号
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);
        // 结束并释放 NCCL 通信上下文
        context->finalize();
        // 释放存储张量
        storage = torch::Tensor();
        // 清空上下文指针
        context = nullptr;
        // 标记已销毁，之后不能再使用
        destroyed = true;
    }
};

// 向 Python 模块注册 PP 缓冲区的绑定接口
static void register_apis(pybind11::module_& m) {
    // 将 PPBuffer 暴露给 Python，并绑定带命名参数的构造函数
    pybind11::class_<PPBuffer, BufferBase>(m, "PPBuffer")
        .def(pybind11::init<int, int, int64_t, int64_t, int, std::optional<int>, int, bool>(),
             py::arg("rank_idx"),  // 参数：本 rank 编号
             py::arg("num_ranks"),  // 参数：总 rank 数
             py::arg("nccl_comm"),  // 参数：NCCL 通信域句柄
             py::arg("num_max_tensor_bytes"),  // 参数：单张量最大字节数
             py::arg("num_max_inflight_tensors"),  // 参数：最大在途张量数
             py::arg("sl_idx"),  // 参数：可选的 SL 索引
             py::arg("num_gpu_timeout_secs"),  // 参数：GPU 超时秒数
             py::arg("explicitly_destroy") = false)  // 参数：是否要求显式销毁
        // 只读属性：存储张量
        .def_readonly("storage", &PPBuffer::storage)
        // 方法：向相邻 rank 发送张量
        .def("send", &PPBuffer::send,
             py::arg("x"),  // 参数：待发送张量
             py::arg("dst_rank_idx"),  // 参数：目标 rank
             py::arg("num_sms") = 0)  // 参数：使用的 SM 数，0 表示全部
        // 方法：从相邻 rank 接收张量
        .def("recv", &PPBuffer::recv,
             py::arg("x"),  // 参数：接收目标张量
             py::arg("src_rank_idx"),  // 参数：源 rank
             py::arg("num_sms") = 0);  // 参数：使用的 SM 数，0 表示全部
}

}  // namespace deep_ep::pp
