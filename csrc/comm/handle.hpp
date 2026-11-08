#pragma once

// 引入屏障函数
#include "barrier.hpp"
// 引入 NCCL Context / 通信创建相关的底层 API
#include "../kernels/comm/api.hpp"

// 通信模块所在的命名空间
namespace deep_ep::comm {

// 向 Python 模块注册通信相关的绑定接口（Context 类与若干自由函数）
static void register_apis(pybind11::module_& m) {
    // 将 NCCL 通信上下文 Context 以共享指针形式暴露给 Python
    pybind11::class_<Context, std::shared_ptr<Context>>(m, "Context")
        // 绑定方法：查询物理域大小（RDMA rank 数与 NVLink rank 数）
        .def("get_physical_domain_size", &Context::get_physical_domain_size)
        // 绑定方法：查询逻辑域大小（scale-out / scale-up 维度划分）
        .def("get_logical_domain_size", &Context::get_logical_domain_size)
        // 只读属性：workspace 的字节数
        .def_readonly("num_workspace_bytes", &Context::num_workspace_bytes)
        // 只读属性：GPU 通信 buffer 的字节数
        .def_readonly("num_gpu_buffer_bytes", &Context::num_gpu_buffer_bytes)
        // 只读属性：RDMA 存储区的字节数
        .def_readonly("num_rdma_storage_bytes", &Context::num_rdma_storage_bytes);

    // 注册 Python 端可用的 barrier 函数：对给定上下文执行组内屏障
    m.def("barrier", [](const std::shared_ptr<Context>& context,
                        const bool& with_cpu_sync, const bool& sequential, const bool& wait_comm_stream) {
        // 断言上下文指针非空
        EP_HOST_ASSERT(context != nullptr);
        // 断言屏障信号区已初始化
        EP_HOST_ASSERT(context->barrier_signals != nullptr);
        // 调用底层屏障：强制 flush_stores，其余参数透传
        barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                context->num_gpu_timeout_cycles, with_cpu_sync, sequential, true, wait_comm_stream);
    });

    // 绑定获取本地 NCCL unique id 的函数
    m.def("get_local_nccl_unique_id", &get_local_unique_id);
    // 绑定创建 NCCL 通信域的函数
    m.def("create_nccl_comm", &create_nccl_comm);
    // 绑定销毁 NCCL 通信域的函数
    m.def("destroy_nccl_comm", &destroy_nccl_comm);
    // 绑定查询物理域大小的函数
    m.def("get_physical_domain_size", &get_physical_domain_size);
    // 绑定查询逻辑域大小的函数
    m.def("get_logical_domain_size", &get_logical_domain_size);
    // 绑定获取通信流的函数（返回 torch::Stream 便于 Python 侧使用）
    m.def("get_comm_stream", []() -> torch::Stream { return get_comm_stream(); });
}

}  // namespace deep_ep::comm
