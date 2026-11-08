#pragma once

// 引入 PyTorch 的 CUDA 上下文（当前流等）
#include <ATen/cuda/CUDAContext.h>
// 引入 std::shared_ptr / make_shared
#include <memory>
// 引入 pybind11 核心
#include <pybind11/pybind11.h>

// 引入 DeepEP 的宿主端断言宏 EP_HOST_ASSERT
#include <deep_ep/common/exception.cuh>

// 引入可空张量列表类型 tensor_list_t
#include "tensor.hpp"

// DeepEP 主命名空间
namespace deep_ep {

// 事件句柄：包装 torch::Event，用于跨流/跨 API 传递完成信号
struct EventHandle {
    // 内部持有的 CUDA 事件（共享指针便于拷贝传递）
    std::shared_ptr<torch::Event> event;
    // 需要在此事件生命周期内登记 record_stream 的张量列表（异步释放保护）
    tensor_list_t tensors_to_record;

    // 默认构造：在当前 CUDA 流上记录事件
    EventHandle() {
        // 创建一个 CUDA 类型的事件对象
        event = std::make_shared<torch::Event>(torch::kCUDA);
        // 在当前流上记录：事件完成即代表当前流此前的工作完成
        event->record(at::cuda::getCurrentCUDAStream());
    }

    // 显式构造：在指定流上记录事件
    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        // 创建一个 CUDA 类型的事件对象
        event = std::make_shared<torch::Event>(torch::kCUDA);
        // 在指定流上记录事件
        event->record(stream);
    }

    // 拷贝构造：使用默认行为（共享底层事件）
    EventHandle(const EventHandle& other) = default;

    // 让当前 CUDA 流等待本事件完成（Python 侧主要调用此方法）
    void current_stream_wait() const { at::cuda::getCurrentCUDAStream().unwrap().wait(*event); }
};

// 向 Python 模块注册 EventHandle 的绑定接口
static void register_event_apis(pybind11::module_& m) {
    // 将 EventHandle 结构体暴露给 Python
    pybind11::class_<EventHandle>(m, "EventHandle")
        // 绑定默认构造函数
        .def(pybind11::init<>())
        // 绑定 current_stream_wait 方法
        .def("current_stream_wait", &EventHandle::current_stream_wait);
}

}  // namespace deep_ep
