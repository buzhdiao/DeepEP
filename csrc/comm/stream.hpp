#pragma once

// 引入 PyTorch 的 Python 绑定支持
#include <torch/python.h>

// 引入事件句柄 EventHandle 的定义
#include "../utils/event.hpp"

// 通信相关的通用工具命名空间
namespace deep_ep::comm {

// 获取进程内全局唯一的通信专用 CUDA 流（首次调用时从流池创建，之后复用）
static at::cuda::CUDAStream get_comm_stream() {
    // 静态局部变量保证只创建一次；true 表示创建高优先级的异步流
    static const auto comm_stream = at::cuda::getStreamFromPool(true);
    // 返回这个全局唯一的通信流
    return comm_stream;
}

// 在指定流上创建并记录一个 CUDA 事件，用于跨流同步
static torch::Event create_event(const at::cuda::CUDAStream& stream) {
    // 创建一个 CUDA 类型的事件对象
    auto event = torch::Event(torch::kCUDA);
    // 在给定流上记录事件：事件完成即代表该流上此前提交的工作全部完成
    event.record(stream);
    // 返回已记录的事件
    return event;
}

// 让 `stream` 等待另一条流 `event_stream` 上已有的工作完成（流间同步）
static void stream_wait(const at::cuda::CUDAStream& stream, const at::cuda::CUDAStream& event_stream) {
    // 断言两条流不是同一条流，避免自己等待自己导致死锁
    EP_HOST_ASSERT(stream.id() != event_stream.id());
    // 在 `event_stream` 上记录事件，并让 `stream` 等待该事件
    stream.unwrap().wait(create_event(event_stream));
}

// 让 `stream` 等待一个已记录好的事件句柄所代表的事件
static void stream_wait(const at::cuda::CUDAStream& stream, const EventHandle& event) {
    // 直接让 `stream` 等待句柄内部持有的 CUDA 事件
    stream.unwrap().wait(*event.event);
}

}  // namespace deep_ep::comm
