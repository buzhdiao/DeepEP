#pragma once  // 头文件保护，防止被重复包含

#include <cstdint>  // 定宽整数类型

// deep_ep::utils 命名空间：通用设备端工具
namespace deep_ep::utils {

struct Identity {  // 恒等函数对象：默认的处理函数，原样返回输入
    template <typename value_t>
    __device__ __forceinline__ value_t operator()(value_t value) const {  // 调用运算符
        return value;  // 直接返回输入值
    }
};

// 分块迭代“加载-处理-存储”的通用模板：
// 线程被分为 kNumThreadGroups 组、每组 kNumThreadsPerGroup 线程，每线程处理 kNumElemsPerThread 个元素，
// 同组线程处理的元素在内存中间隔 kThreadStride（跨步访问，利于合并/交错）
template <int kNumThreadGroups, int kNumThreadsPerGroup, int kNumElemsPerThread,
          typename index_t, typename ld_ptr_t, typename st_ptr_t,
          typename ld_func_t, typename st_func_t,
          typename process_func_t = Identity,  // 元素处理函数，默认恒等
          int kThreadStride = kNumThreadGroups * kNumThreadsPerGroup>  // 线程总步长
__device__ __forceinline__ void iterate_ld_st(const int& global_thread_idx,  // 全局线程编号
                                              const index_t& num_elems,  // 元素总数
                                              const ld_ptr_t& ld_ptr, const st_ptr_t& st_ptr,  // 读指针与写指针
                                              const ld_func_t& ld_func,  // 读取函数（支持越界感知的重载）
                                              const st_func_t& st_func,  // 存储函数
                                              const process_func_t& process_func = process_func_t{}) {  // 处理函数实例
    constexpr int kStride = kNumElemsPerThread * kThreadStride;  // 主循环一轮覆盖的元素总数（本线程每轮处理的跨度）
    constexpr int kMaxElemOffset = (kNumElemsPerThread - 1) * kThreadStride;  // 本线程一轮中最大的元素偏移
    using ld_value_t = decltype(ld_func(ld_ptr));  // 由读取函数推导读取值类型
    using st_value_t = decltype(process_func(ld_func(ld_ptr)));  // 由处理函数推导存储值类型

    // Main loop  // 主循环：整块完整处理（不越界）
    index_t elem_begin_idx = global_thread_idx;  // 本线程负责的起始元素下标
    for (; elem_begin_idx + kMaxElemOffset < num_elems; elem_begin_idx += kStride) {  // 只要一轮不越界就继续
        const auto ld_base = ld_ptr + elem_begin_idx;  // 本轮读取的基地址
        ld_value_t ld_values[kNumElemsPerThread];  // 本轮读取的元素寄存器缓冲
        #pragma unroll  // 展开读取循环
        for (int i = 0; i < kNumElemsPerThread; ++ i) {  // 逐个读取本线程的元素
            const auto elem_offset = i * kThreadStride;  // 第 i 个元素相对基址的偏移
            ld_values[i] = ld_func(ld_base + elem_offset);  // 从读取指针载入
        }
        st_value_t st_values[kNumElemsPerThread];  // 本轮待存储的元素缓冲
        #pragma unroll  // 展开处理循环
        for (int i = 0; i < kNumElemsPerThread; ++ i) {  // 逐个处理
            st_values[i] = process_func(ld_values[i]);  // 调用处理函数变换
        }
        const auto st_base = st_ptr + elem_begin_idx;  // 本轮存储的基地址
        #pragma unroll  // 展开存储循环
        for (int i = 0; i < kNumElemsPerThread; ++ i) {  // 逐个写回
            const auto elem_offset = i * kThreadStride;  // 第 i 个元素相对基址的偏移
            st_func(st_base + elem_offset, st_values[i]);  // 写入存储指针
        }
    }

    // Tail  // 尾部：剩余不足一整轮的元素（可能越界，交给带边界信息的重载）
    if (elem_begin_idx < num_elems) {  // 还有剩余元素才处理
        const auto ld_base = ld_ptr + elem_begin_idx;  // 尾部读取基地址
        const auto num_remaining_elems = static_cast<int>(num_elems - elem_begin_idx);  // 剩余元素个数
        ld_value_t ld_values[kNumElemsPerThread];  // 尾部读取缓冲
        #pragma unroll  // 展开读取循环
        for (int i = 0; i < kNumElemsPerThread; ++ i) {  // 逐个读取
            const auto elem_offset = i * kThreadStride;  // 第 i 个元素的偏移
            ld_values[i] = ld_func(ld_base + elem_offset, num_remaining_elems, elem_offset);  // 带剩余长度与偏移的越界感知读取
        }
        st_value_t st_values[kNumElemsPerThread];  // 尾部处理结果缓冲
        #pragma unroll  // 展开处理循环
        for (int i = 0; i < kNumElemsPerThread; ++ i) {  // 逐个处理
            st_values[i] = process_func(ld_values[i]);  // 调用处理函数变换
        }
        const auto st_base = st_ptr + elem_begin_idx;  // 尾部存储基地址
        #pragma unroll  // 展开存储循环
        for (int i = 0; i < kNumElemsPerThread; ++ i) {  // 逐个写回
            const auto elem_offset = i * kThreadStride;  // 第 i 个元素的偏移
            st_func(st_base + elem_offset, st_values[i], num_remaining_elems, elem_offset);  // 带边界信息的越界感知存储
        }
    }
}

}  // namespace deep_ep::utils
