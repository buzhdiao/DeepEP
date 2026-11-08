#pragma once  // 头文件保护，防止被重复包含

#include <exception>  // std::exception 基类
#include <string>  // std::string
#include <sstream>  // std::stringstream，用于拼接错误信息

#ifndef EP_STATIC_ASSERT  // 若尚未定义静态断言宏
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)  // 定义为标准 static_assert：条件不满足时输出 reason 编译报错
#endif

// 异常类：携带异常名称、源文件、行号与错误描述，供各 CHECK/ASSERT 宏抛出
class EPException : public std::exception {
private:
    std::string message = {};  // 缓存拼接好的完整错误消息

public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {  // 构造：记录异常名、文件、行号与错误内容
        std::stringstream ss;  // 用于拼接消息的字符串流
        ss << name << " exception (" << file << ":" << line << "): " << error;  // 拼成 "xxx exception (file:line): error" 格式
        message = ss.str();  // 缓存最终消息
    }

    const char* what() const noexcept override { return message.c_str(); }  // 实现 what()：返回缓存的错误消息
};

#define EPExceptionWithLineInfo(name, message) EPException(name, __FILE__, __LINE__, message)  // 辅助宏：自动填入当前文件名与行号构造 EPException

#ifndef EP_HOST_ASSERT  // 主机端断言宏（未定义时才定义）
// EP_HOST_ASSERT：条件不满足时抛出 C++ 异常，消息为条件的字符串化形式；do-while 包裹保证在 if/else 中安全使用
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

#ifndef EP_HOST_UNREACHABLE  // 主机端“不可达”宏（未定义时才定义）
#define EP_HOST_UNREACHABLE(reason) (throw EPException("Assertion", __FILE__, __LINE__, reason))  // 直接抛异常，reason 为说明文字
#endif

#ifndef EP_DEVICE_ASSERT  // 设备端断言宏（未定义时才定义）
// EP_DEVICE_ASSERT：条件不满足时打印出错位置与条件，然后触发 trap 终止整个 kernel
#define EP_DEVICE_ASSERT(cond)                                                             \
    do {                                                                                   \
        if (not(cond)) {                                                                   \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            asm("trap;");                                                                  \
        }                                                                                  \
    } while (0)
#endif

#ifndef EP_UNIFIED_ASSERT  // 主机/设备统一断言宏（未定义时才定义）
#ifdef __CUDA_ARCH__  // 设备代码编译时
#define EP_UNIFIED_ASSERT(cond) EP_DEVICE_ASSERT(cond)  // 走设备端断言（printf + trap）
#else  // 主机代码编译时
#define EP_UNIFIED_ASSERT(cond) EP_HOST_ASSERT(cond)  // 走主机端断言（抛异常）
#endif
#endif

#ifndef CUDA_RUNTIME_CHECK  // CUDA Runtime API 调用检查宏（未定义时才定义）
// CUDA_RUNTIME_CHECK：执行 cmd 并检查返回的 cudaError_t，失败时抛出携带错误码/错误名/错误串的异常
#define CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != cudaSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
        throw EPException("CUDA runtime", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef CUDA_DRIVER_CHECK  // CUDA Driver API 调用检查宏（未定义时才定义）
// CUDA_DRIVER_CHECK：执行 cmd 并检查 CUresult，失败时用惰性加载的错误查询函数取错误信息并抛异常
#define CUDA_DRIVER_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != CUDA_SUCCESS) { \
        std::stringstream ss; \
        const char *name, *info; \
        lazy_cuGetErrorName(e, &name), lazy_cuGetErrorString(e, &info); \
        ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; \
        throw EPException("CUDA driver", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef NCCL_CHECK  // NCCL 调用检查宏（未定义时才定义）
// NCCL_CHECK：执行 cmd 并检查 ncclResult_t，失败时抛出携带 ncclGetLastError 信息的异常
#define NCCL_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != ncclSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << ncclGetLastError(nullptr) << ")"; \
        throw EPException("NCCL", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif
