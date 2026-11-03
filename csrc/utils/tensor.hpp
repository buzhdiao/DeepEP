#pragma once

// 引入 std::optional
#include <optional>
// 引入 std::vector
#include <vector>

// 引入 PyTorch 的 Python 绑定支持
#include <torch/python.h>

// 引入 DeepEP 的宿主端断言宏 EP_HOST_ASSERT
#include <deep_ep/common/exception.cuh>

// DeepEP 主命名空间
namespace deep_ep {

// 类型别名：可空张量的列表，广泛用于流上的 record_stream 登记
using tensor_list_t = std::vector<std::optional<torch::Tensor>>;

// 把张量的形状解包成 `kNumDims` 维的 std::tuple<int...>，维度不符则断言失败
template <int kNumDims>
static auto get_shape(const torch::Tensor& t) {
    // 宿主端断言：张量维数必须与模板参数一致
    EP_HOST_ASSERT(t.dim() == kNumDims);
    // 构造一个捕获 t 的泛型 lambda，配合编译期索引序列展开各维大小
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        // 将每一维大小（long）转换为 int，打包成元组返回
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<kNumDims>());
}

// 从可空张量中取出指定类型的数据指针；张量不存在时返回 nullptr
template <typename dtype_t = void>
static dtype_t* get_data_ptr(const std::optional<torch::Tensor>& t) {
    // 有值则取其数据指针，否则返回空指针
    return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;
}

}  // namespace deep_ep
