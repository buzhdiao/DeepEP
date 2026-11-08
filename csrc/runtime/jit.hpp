#pragma once

// 引入文件系统路径工具
#include <filesystem>
// 引入 std::shared_ptr
#include <memory>
// 引入 std::string
#include <string>

// 引入 DeepEP 公共编译常量（如 EP_NUM_TOPK_IDX_BITS）
#include <deep_ep/common/compiled.cuh>
// 引入 deep_jit 的 CUDA 运行时后端
#include <deep_jit/backend/cuda/backend.hpp>
// 引入 deep_jit 的 Python 绑定注册接口
#include <deep_jit/python_api.hpp>

// 引入环境变量读取工具 get_env
#include "../utils/system.hpp"

// DeepEP 主命名空间
namespace deep_ep {

// 全局 JIT 编译运行时：懒初始化（首次使用时才真正创建 CUDA 编译运行时）
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

// 初始化 JIT 运行时：library_root_path 为 DeepEP 库根目录，nccl_root_path 为 NCCL 安装根目录
inline void init_jit(const std::string& library_root_path, const std::string& nccl_root_path) {
    // 将库根目录转换为绝对路径
    const auto library_root = std::filesystem::absolute(library_root_path);
    // 计算 DeepEP 头文件所在目录（根目录下的 include）
    const auto library_include_dir = library_root / "include";
    // 计算 NCCL 头文件目录的绝对路径
    const auto nccl_include_dir = std::filesystem::absolute(
        std::filesystem::path(nccl_root_path) / "include");
    // 组装 JIT 编译配置：库根目录、名称 "EP"、默认参数、包含目录与源码前缀
    const auto config = deep_jit::Config(
        library_root,
        "EP",
        {},
        {library_include_dir},
        {"deep_ep/"});

    // 以懒初始化方式创建运行时：lambda 只在首次访问 jit 时执行
    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config, nccl_include_dir] {
        // 按配置创建 CUDA JIT 运行时实例
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);

        // Keep the NCCL include path in compiler options so it participates in the cache key.
        // DeepEP's own include path stays path-independent because its contents are tracked by the parser.
        // 将 NCCL 头文件路径保留在编译选项中，使其参与编译缓存键的计算。
        // DeepEP 自身的头文件路径与路径无关，因为其内容已被解析器跟踪。
        // 获取默认编译选项中 nvcc 旗标的可变引用
        auto& nvcc_flags = *runtime->default_compiler_options.nvcc_flags;
        // 追加 nvcc 诊断抑制选项，屏蔽已知无关警告
        nvcc_flags.emplace_back("--diag-suppress=39,161,174,177,186,940,3012");
        // 追加宿主编译器选项：关闭过时声明与 ABI 相关警告
        nvcc_flags.emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        // 追加 NCCL 头文件包含路径
        nvcc_flags.emplace_back("-I" + nccl_include_dir.string());
        // 定义 top-k 索引位宽宏，保证编译期与运行时一致
        nvcc_flags.emplace_back("-DEP_NUM_TOPK_IDX_BITS=" + std::to_string(EP_NUM_TOPK_IDX_BITS));
        // 启用 NCCL GIN 的 GDAKI 后端
        nvcc_flags.emplace_back("-DNCCL_GIN_GDAKI_ENABLE=1");
        // 关闭 NCCL GIN 的 PROXY 后端
        nvcc_flags.emplace_back("-DNCCL_GIN_PROXY_ENABLE=0");
        // 关闭 NCCL GIN 的 GPI 后端
        nvcc_flags.emplace_back("-DNCCL_GIN_GPI_ENABLE=0");
        // 关闭 NCCL GIN 的 EFA GDA 后端
        nvcc_flags.emplace_back("-DNCCL_GIN_EFA_GDA_ENABLE=0");
        // 环境变量 EP_GIN_GDAKI_DEBUG 非零时，启用 GDAKI 的调试宏
        if (get_env<int>("EP_GIN_GDAKI_DEBUG", 0))
            nvcc_flags.emplace_back("-DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1");
        // 返回配置完成的运行时实例
        return runtime;
    });
}

// 向 Python 模块注册 JIT 相关 API（deep_jit 自身接口与 init_jit 函数）
inline void register_apis(pybind11::module_& module) {
    // 注册 deep_jit 提供的 Python API（围绕全局 jit 运行时）
    deep_jit::register_python_api(module, jit);
    // 绑定 init_jit 函数，供 Python 侧完成 JIT 初始化
    module.def("init_jit", &init_jit);
}

}  // namespace deep_ep
