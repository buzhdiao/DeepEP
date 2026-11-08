#pragma once

// 引入 CUDA 驱动 API 的声明（cuModuleLoad 等）
#include <cuda.h>
// 引入动态链接库加载接口 dlopen/dlsym
#include <dlfcn.h>

// 引入 DeepEP 的宿主端断言宏 EP_HOST_ASSERT
#include <deep_ep/common/exception.cuh>

// DeepEP 主命名空间
namespace deep_ep {

// Lazy loading all driver symbols
// 懒加载所有 CUDA 驱动符号
// 获取 CUDA 驱动动态库的句柄；首次调用时通过 dlopen 打开 libcuda.so.1
static void* get_driver_handle() {
    // 函数级静态变量：缓存已打开的库句柄
    static void* handle = nullptr;
    // 尚未加载时才执行 dlopen
    if (handle == nullptr) {
        // 以懒解析、本地符号可见性打开 CUDA 驱动库
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        // 断言加载成功，否则说明环境中缺少 CUDA 驱动
        EP_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    // 返回库句柄
    return handle;
}

// Macro to expand CUDA API aliases before stringification
// 宏定义：在字符串化之前先展开 CUDA API 的别名
// 把标识符转换为其字面字符串（借助两层宏以正确展开别名）
#define EP_CUDA_STRINGIFY(name) #name

// Macro to define wrapper functions named `lazy_cu{API name}`
// 宏定义：生成名为 `lazy_cu{API 名}` 的包装函数
// 说明：宏体为多行，续行符后不能添加注释；作用是按需通过 dlsym 查找驱动函数指针，
//       首次调用时解析符号并缓存，之后直接调用，从而避免启动时强链接 libcuda
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), EP_CUDA_STRINGIFY(name))); \
        EP_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}

// 为下列各个 CUDA 驱动 API 分别生成懒加载包装函数 lazy_cuXxx
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemSetAccess);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRetainAllocationHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAddressRange_v2);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressReserve);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressFree);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemMap);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemUnmap);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemCreate);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRelease);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuCtxGetDevice);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemImportFromShareableHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemExportToShareableHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAllocationGranularity);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGet);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuStreamBatchMemOp);

}  // namespace deep_ep
