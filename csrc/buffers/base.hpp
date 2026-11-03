#pragma once

// 引入 printf/fflush 所需头文件
#include <cstdio>
// 引入 std::shared_ptr
#include <memory>
// 引入 pybind11 核心
#include <pybind11/pybind11.h>

// 引入 NCCL 通信上下文 Context 的定义
#include "../kernels/comm/api.hpp"

// DeepEP 主命名空间
namespace deep_ep {

// 所有缓冲区（EP/Engram/Bucket/PP）的公共基类，负责析构时的统一销毁语义
class BufferBase {
protected:
    // 是否要求用户显式调用 destroy()（为 false 时允许析构函数中自动销毁）
    bool explicitly_destroy;
    // 销毁标记：防止重复销毁
    bool destroyed = false;

    // 构造函数：记录销毁策略
    explicit BufferBase(const bool& explicitly_destroy): explicitly_destroy(explicitly_destroy) {}

    // Called from the derived destructor, while its members and destroy() implementation are still available.
    // 从派生类析构函数中调用，此时派生类的成员和 destroy() 实现仍然可用。
    // 按销毁策略在对象析构时完成资源释放（仅在未显式销毁且尚未销毁时调用 destroy()）
    void destroy_on_destruction(const char* name) {
        // 未要求显式销毁且尚未销毁时，执行派生类的 destroy()
        if (not explicitly_destroy and not destroyed) {
            destroy();
            // 销毁后清空主上下文的共享指针，释放通信资源
            main_context = nullptr;
        }

        // 若仍未销毁（例如要求显式销毁但用户没有调用），打印泄漏警告
        if (not destroyed) {
            std::printf("`destroy()` is not called before DeepEP %s buffer destruction, which can leak resources.\n", name);
            // 立即刷新 stdout，保证警告及时输出
            std::fflush(stdout);
        }
    }

public:
    // 主通信上下文：子类构造时赋值，供打印内存占用等公共逻辑使用
    std::shared_ptr<comm::Context> main_context;

    // 虚析构函数：允许 noexcept(false)（destroy 可能抛出异常）
    virtual ~BufferBase() noexcept(false) = default;
    // 纯虚函数：由各子类实现的具体销毁逻辑
    virtual void destroy() = 0;

    // 打印当前缓冲区的内存占用情况（workspace、GPU buffer、RDMA 存储等）
    void print_memory_usage() const {
        // 断言缓冲区尚未销毁且主上下文有效
        EP_HOST_ASSERT(not destroyed and main_context != nullptr);
        // 计算 workspace 占用（GiB）
        const auto num_workspace_gibs = main_context->num_workspace_bytes / float(1 << 30);
        // 计算 GPU 通信 buffer 占用（GiB）
        const auto num_buffer_gibs = main_context->num_gpu_buffer_bytes / float(1 << 30);
        // 计算 GPU 侧 RDMA 存储占用（使用 CPU 存储时为 0）
        const auto num_gpu_storage_gibs = main_context->use_cpu_rdma_storage ? 0.0 : main_context->num_rdma_storage_bytes / float(1 << 30);
        // 计算 CPU 侧 RDMA 存储占用（未使用 CPU 存储时为 0）
        const auto num_cpu_storage_gibs = main_context->use_cpu_rdma_storage ? main_context->num_rdma_storage_bytes / float(1 << 30) : 0.0;
        // 格式化打印各部分内存及总量信息
        std::printf("[DeepEP memory usage] rank_idx=%d, num_ranks=%d, GPU total %.3f GiB "
                    "(workspace %.3f GiB, buffer %.3f GiB, RDMA storage %.3f GiB), CPU total %.3f GiB\n",
                    main_context->rank_idx, main_context->num_ranks,
                    num_workspace_gibs + num_buffer_gibs + num_gpu_storage_gibs,
                    num_workspace_gibs, num_buffer_gibs, num_gpu_storage_gibs, num_cpu_storage_gibs);
        // 立即刷新 stdout
        std::fflush(stdout);
    }
};

}  // namespace deep_ep

// 基类的 Python 绑定注册所在的子命名空间
namespace deep_ep::base {

// 向 Python 模块注册 BufferBase 的绑定接口
static void register_apis(pybind11::module_& m) {
    // 将 BufferBase 暴露给 Python
    pybind11::class_<BufferBase>(m, "BufferBase")
        // 只读属性：主通信上下文
        .def_readonly("main_context", &BufferBase::main_context)
        // 方法：打印内存占用
        .def("print_memory_usage", &BufferBase::print_memory_usage)
        // 方法：销毁缓冲区
        .def("destroy", &BufferBase::destroy);
}

}  // namespace deep_ep::base
