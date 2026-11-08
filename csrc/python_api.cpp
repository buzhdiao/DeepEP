// 引入 pybind11 核心头文件
#include <pybind11/pybind11.h>
// 引入 PyTorch 的 Python 绑定支持
#include <torch/python.h>

// 引入 DeepEP 公共编译常量（如 topk_idx_t、对齐字节数）
#include <deep_ep/common/compiled.cuh>

// 引入各缓冲区基类与实现的注册接口
#include "buffers/base.hpp"
#include "buffers/bucket.hpp"
#include "buffers/engram.hpp"
#include "buffers/ep.hpp"
#include "buffers/pp.hpp"
// 引入通用通信 API 的注册接口
#include "comm/api.hpp"
// 引入 JIT 运行时的注册接口
#include "runtime/jit.hpp"
// 引入事件句柄的注册接口
#include "utils/event.hpp"

// 若构建系统未定义扩展名，则使用默认的 _C
#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// 定义 Python 扩展模块：把 C++ 各子模块的 API 统一挂载到 DeepEP 的 Python 包中
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 设置模块文档字符串
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    // The integer type of top-k indices
    // top-k 索引所使用的整数类型（映射为 torch 标量类型暴露给 Python）
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);
    // 注册获取内存分配对齐字节数的函数
    m.def("get_num_allocation_alignment", []() { return deep_ep::kNumAllocationAlignmentBytes; });
    // 注册获取 TMA 对齐字节数的函数
    m.def("get_num_tma_alignment", []() { return deep_ep::kNumTMAAlignmentBytes; });
    // 注册获取 RDMA 对齐字节数的函数
    m.def("get_num_rdma_alignment", []() { return deep_ep::kNumRDMAAlignmentBytes; });

    // JIT API
    // JIT API：注册 JIT 运行时相关的 Python 接口
    deep_ep::register_apis(m);

    // Event API
    // 事件 API：注册 EventHandle 相关接口
    deep_ep::register_event_apis(m);

    // Common communication APIs
    // 通用通信 API：注册 Context、barrier 等接口
    deep_ep::comm::register_apis(m);

    // Common buffer interface
    // 通用缓冲区基类接口：注册 BufferBase
    deep_ep::base::register_apis(m);

    // Register EP buffer APIs
    // 注册 EP（专家并行）缓冲区 API
    deep_ep::ep::register_apis(m);

    // Register Engram buffer APIs
    // 注册 Engram 缓冲区 API
    deep_ep::engram::register_apis(m);

    // Register bucket communication APIs
    // 注册 bucket（桶式 reduce-scatter / all-reduce / all-gather）通信 API
    deep_ep::bucket::register_apis(m);

    // Register pipeline-parallel send/recv APIs
    // 注册流水线并行（PP）的 send/recv API
    deep_ep::pp::register_apis(m);
}
