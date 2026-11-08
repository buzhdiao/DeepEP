#pragma once

// 引入标准输入输出（用于 sscanf）
#include <cstdio>
// 引入标准库工具（用于 getenv）
#include <cstdlib>
// 引入 std::string
#include <string>
// 引入编译期类型判断工具 std::is_same_v
#include <type_traits>

// 引入 DeepEP 的宿主端断言宏 EP_HOST_ASSERT
#include <deep_ep/common/exception.cuh>

// DeepEP 主命名空间
namespace deep_ep {

// ReSharper disable once CppNotAllPathsReturnValue
// 读取环境变量并按模板类型转换的通用函数；name 为变量名，default_value 为缺省值
template <typename dtype_t>
static dtype_t get_env(const std::string& name, const dtype_t& default_value = dtype_t()) {
    // 从进程环境中读取该环境变量的 C 字符串
    const auto c_str = std::getenv(name.c_str());
    // 未设置时返回默认值
    if (c_str == nullptr)
        return default_value;

    // Read the env and convert to the desired type
    // 读取环境变量并转换为期望的类型
    if constexpr (std::is_same_v<dtype_t, std::string>) {
        // 目标类型为 std::string：直接以字符串返回
        return std::string(c_str);
    } else if constexpr (std::is_same_v<dtype_t, int>) {
        // 目标类型为 int：用 sscanf 解析十进制整数
        int value;
        std::sscanf(c_str, "%d", &value);
        return value;
    } else {
        // 其余类型不支持：直接触发宿主端断言失败
        EP_HOST_ASSERT(false and "Unexpected type");
    }
}

} // deep_ep
