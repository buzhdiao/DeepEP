#pragma once  // 头文件保护，防止被重复包含

// deep_ep::math 命名空间：主机/设备通用的数学与指针、打包小工具
namespace deep_ep::math {

template <typename T>
__forceinline__ __device__ __host__ T ceil_div(T a, T b) {  // 向上取整除法：ceil(a/b)
    return (a + b - 1) / b;  // 经典写法：先加 b-1 再整除
}

template <typename T>
__forceinline__ __device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {  // constexpr 版向上取整除法（可用于编译期常量）
    return (a + b - 1) / b;  // 同上
}

template <typename T, bool kDoCeilAlignment = true>
__forceinline__ __device__ __host__ T align(T a, T b) {  // 将 a 按 b 对齐（默认向上取整对齐）
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;  // 向上：先上取整再乘回；否则直接向下截断
}

template <typename T, bool kDoCeilAlignment = true>
__forceinline__ __device__ __host__ constexpr T constexpr_align(T a, T b) {  // constexpr 版对齐函数
    return (kDoCeilAlignment ? constexpr_ceil_div(a, b) : (a / b)) * b;  // 同上
}

template <typename dtype_t>
__forceinline__ __device__ __host__ bool is_decoded_positive_ready(const dtype_t& value) {  // 判断解码后的值是否表示“就绪”（正数）
    return value >= 0;  // 编码约定：负值表示未就绪，>=0 表示就绪
}

template <typename dtype_t>
__forceinline__ __device__ __host__ dtype_t encode_decode_positive(const dtype_t& value) {  // 正数与负数编码的互换（自反编解码）
    return -value - static_cast<dtype_t>(1);  // 取负再减 1：编码 x -> -x-1，解码同样满足该映射
}

template <typename dtype_t = void>
__forceinline__ __device__ __host__ dtype_t* advance_ptr(void* ptr, const int64_t num_bytes) {  // 按字节数前移指针并转为目标类型
    return reinterpret_cast<dtype_t*>(static_cast<int8_t*>(ptr) + num_bytes);  // 以 int8_t 视图做字节级偏移
}

__forceinline__ __device__ __host__ ptrdiff_t ptr_diff(const void* ptr, const void* base) {  // 计算两指针的字节差
    return static_cast<const int8_t*>(ptr) - static_cast<const int8_t*>(base);  // 按 int8_t 视图做字节级相减
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {  // 将两个半宽值打包为一个全宽值
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");  // 要求两倍大小关系成立
    dtype_b_t packed;  // 打包结果
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);  // 以半宽类型视图访问打包缓冲
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;  // 依次写入低半与高半
    return packed;  // 返回打包值
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ std::tuple<dtype_a_t, dtype_a_t> unpack2(const dtype_b_t& packed) {  // 将全宽值解包为两个半宽值（返回 tuple）
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");  // 要求两倍大小关系成立
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);  // 以半宽类型视图读取
    dtype_a_t x = unpacked_ptr[0], y = unpacked_ptr[1];  // 取出低半与高半
    return {x, y};  // 以元组返回
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {  // 解包为两个半宽值（引用输出版本）
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");  // 要求两倍大小关系成立
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);  // 以半宽类型视图读取
    x = unpacked_ptr[0], y = unpacked_ptr[1];  // 分别写入输出引用
}

}  // namespace deep_ep::math
