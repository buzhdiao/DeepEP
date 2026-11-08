#pragma once
// 头文件保护：保证本文件在单个编译单元中只被包含一次

#include <stdexcept>
// 引入标准异常库，用于在编译期参数非法时抛出 logic_error


namespace deep_ep::ep {
// 进入 DeepEP 的 expert parallel（EP）命名空间

// 根据“是否允许多重归约 / rank 数 / top-k 数”决定归约布局是否按 rank 维度组织
template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>
constexpr bool use_rank_layout() {
    // 若不允许多重归约，则不能使用 rank 布局（同一 rank 的多个 expert 无法区分来源）
    if constexpr (not kAllowMultipleReduction)
        return false;
    // 只有当 rank 数不超过 top-k 时，按 rank 布局才比按 top-k 布局更省
    return kNumRanks <= kNumTopk;
}

// 返回上述布局中每个元素（槽位）对应的“token 数”：rank 布局下为 rank 数，否则为 top-k 数
template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>
constexpr int get_num_tokens_in_layout() {
    // rank 布局：每个槽位代表一个 rank；否则每个槽位代表一个 top-k 位置
    return use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>() ? kNumRanks : kNumTopk;
}

// 编译期计算最大展开因子：在 kMaxUnrollFactor 到 1 之间找到最大的 i，
// 使得 kLength 恰好能被 (一个 warp 的线程数 32 * i) 整除，保证每个 warp 每轮处理整数个向量
template <int kLength, int kMaxUnrollFactor, int kWarpSize = 32>
constexpr int get_max_unroll_factor() {
    // 从最大因子向下尝试
    for (int i = kMaxUnrollFactor; i >= 1; -- i)
        // 长度能被 warp 宽度乘以因子整除即合法（warp 内每 lane 处理 i 个向量，无尾差）
        if (kLength % (kWarpSize * i) == 0)
            return i;
    // 找不到任何合法因子说明长度与 warp 宽度不匹配，直接报错
    throw std::logic_error("Invalid length, cannot find unrolling factor");
}

// Determine the vector type for combine loads/stores based on arch and hidden size alignment
// 根据 GPU 架构和 hidden size 的对齐情况，决定 combine 加载/存储所用的向量类型
template <int kHiddenBytes>
struct CombineVecTraits {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // On SM100+, use `longlong4_t` (32 bytes) if hidden is aligned, otherwise fall back to `int4` (16 bytes)
    // NOTES: we observe some performance degrade with `longlong4_t`, temporarily disable it
    // SM100+（Blackwell）本可用 32 字节的 longlong4_t，但实测有性能回退，暂时关闭
    static constexpr bool kUseLonglong4 = false;
    // 按开关选择 longlong4_t（32B）或 int4（16B）作为向量类型
    using vec_t = std::conditional_t<kUseLonglong4, longlong4_t, int4>;
#else
    // 旧架构上一律使用 16 字节的 int4 向量
    using vec_t = int4;
#endif
};

// 从 32 位掩码中按“最低置位位”依次取出至多 kNumValidTopk 个槽位下标，
// 并通过 fetch_func 无条件地做 warp 内数据交换（避免发散分支 BRA.DIV 造成性能损失）
template <int kNumValidTopk, typename fetch_func_t>
__device__ __forceinline__
void compute_topk_slots(int (&topk_slot_idx)[kNumValidTopk], uint32_t mask,
                        const fetch_func_t& fetch_func) {
    // 对每个 top-k 槽位循环
    #pragma unroll
    for (int k = 0; k < kNumValidTopk; ++ k) {
        // __ffs 找到掩码中最低的置位位（1-based），减 1 得到 0-based 下标
        const int lowest_idx = __ffs(mask) - 1;
        // Here we perform the exchange unconditionally to avoid `BRA.DIV`
        // 无条件执行 warp 交换，即使下标非法也做，避免分支发散
        const auto fetched = fetch_func(lowest_idx);
        // 清除刚处理过的最低置位位（mask & mask-1 是经典技巧）
        mask &= mask - 1;
        // 下标合法则记录取回的槽位，否则记 -1 表示该槽位无效
        topk_slot_idx[k] = lowest_idx >= 0 ? fetched : -1;
    }
}

// combine 的核心归约函数：每个 warp 负责一段 hidden 维度，
// 从各个源槽位（rank 或 top-k）的接收缓冲区读取数据并累加，结果写入目标缓冲区
// kHiddenVec：hidden 维度被划分的向量数；kUnrollFactor：每 lane 每轮处理的向量数
template <int kHiddenVec, int kUnrollFactor, int kNumExpectedTopk, int kNumValidTopk,
          typename vec_t, typename get_src_buffer_ptr_func_t, typename wait_buffer_func_t>
__device__ __forceinline__
void combine_reduce(const int& lane_idx, int (&topk_slot_idx)[kNumValidTopk],
                    vec_t* dst_buffer_ptr,
                    const get_src_buffer_ptr_func_t& get_src_buffer_ptr_func,
                    const wait_buffer_func_t& wait_buffer_func,
                    vec_t* bias_0 = nullptr, vec_t* bias_1 = nullptr) {
    // 每个向量包含的 BF16 元素个数（int4=16B 即 8 个 BF16）
    constexpr int kNumElemsPerVec = sizeof(vec_t) / sizeof(nv_bfloat16);
    // 要求向量元素数为偶数，才能按 bf16x2（4 字节）成对处理
    EP_STATIC_ASSERT(kNumElemsPerVec % 2 == 0, "Invalid number of elements");
    // 要求 hidden 向量数能被（展开因子*warp 宽度）整除，保证每轮无残余
    EP_STATIC_ASSERT(kHiddenVec % (kUnrollFactor * 32) == 0, "Invalid unrolling");

    // We use BF16 add as much as possible, as casting is slow
    // 快速路径判定：无 bias，且有效 top-k 不超过 2 个时，可全程用 BF16 加法，跳过 FP32 累加
    const bool enable_hadd_bypass =
        (bias_0 == nullptr and bias_1 == nullptr) and
        (kNumValidTopk <= 2 or topk_slot_idx[2] < 0);
    // 至少要有 1 个有效 top-k
    EP_STATIC_ASSERT(kNumValidTopk > 0, "Invalid top-k");

    if (enable_hadd_bypass) {
        // 快速路径：仅 BF16 相加，不做 BF16->FP32->BF16 的转换
        // #pragma unroll 1 禁止外层展开（轮数较多，展开无益且膨胀代码）
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            // Read values 0
            // 读取第 0 个槽位的数据
            const auto slot_0 = topk_slot_idx[0];
            // 由槽位号计算第 0 个源缓冲区的基址
            const auto src_base_ptr_0 = get_src_buffer_ptr_func(slot_0);
            // 本 lane 的展开寄存器数组，零初始化
            vec_t values_0[kUnrollFactor] = {};
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                // 带谓词（slot 合法才真正读）的全局内存加载，warp 内 32 个 lane 覆盖 j*32 偏移的一段
                values_0[j] = ptx::ldg_with_ge_pred(
                    src_base_ptr_0 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_0, 0);
            }

            // Read values 1
            // 读取第 1 个槽位的数据
            vec_t values_1[kUnrollFactor] = {};
            // 只有 top-k 大于 1 时才有第 1 个槽位，否则置 -1（谓词失效）
            const auto slot_1 = kNumValidTopk == 1 ? -1 : topk_slot_idx[1];
            // 第 1 个源缓冲区基址
            const auto src_base_ptr_1 = get_src_buffer_ptr_func(slot_1);
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                // 同样带谓词加载第 1 个槽位的数据
                values_1[j] = ptx::ldg_with_ge_pred(
                    src_base_ptr_1 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_1, 0);
            }

            // Wait buffer releases for the first write
            // 第一轮写入前先等待目标缓冲区被释放（barrier/上游消费完成），只等一次
            if (i == 0)
                wait_buffer_func();

            // Reduce into shared memory
            // 将寄存器数组按 bf16x2 视角解释，便于成对相加
            const auto bf162_view_0 = reinterpret_cast<nv_bfloat162*>(values_0);
            const auto bf162_view_1 = reinterpret_cast<nv_bfloat162*>(values_1);
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                // 逐个 bf16x2 做 BF16 加法（避免 FP32 转换开销）
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    bf162_view_0[j * (kNumElemsPerVec / 2) + l] += bf162_view_1[j * (kNumElemsPerVec / 2) + l];
                // 将累加结果写回目标缓冲区（本 warp 对应的共享内存/TMA 暂存区）
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = values_0[j];
            }
        }
    } else {
        // 通用路径：可能带 bias、top-k 数量任意，需用 FP32 累加保证精度
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            // Add bias
            // FP32 累加器：每 lane 持有 kUnrollFactor * kNumElemsPerVec / 2 个 float2
            float2 reduced[kUnrollFactor * kNumElemsPerVec / 2] = {};
            // 局部 lambda：读取一段数据并累加进 reduced（用于 bias 或某个槽位）
            const auto add_bias = [&](const vec_t* base_ptr) {
                // Read
                // 读取本 lane 负责的各向量（不带谓词，bias 指针必然有效）
                vec_t values[kUnrollFactor];
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j)
                    values[j] = ptx::ldg(base_ptr + i * (kUnrollFactor * 32) + j * 32 + lane_idx);

                // Reduce
                // 按 bf16x2 视角逐对累加到 FP32 累加器中
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    ptx::accumulate(reduced[j], bf162_view[j]);
            };
            // 存在 bias_0 时先累加 bias_0（三元运算符+void() 规避语句表达式）
            bias_0 != nullptr ? add_bias(bias_0) : void();
            // 存在 bias_1 时再累加 bias_1
            bias_1 != nullptr ? add_bias(bias_1) : void();

            #pragma unroll
            for (int k = 0; k < kNumValidTopk; ++ k) {
                // We have a limitation on `k` to reduce the branch instruction count
                // 从 kNumExpectedTopk 开始，一旦槽位无效即提前退出，减少分支指令数
                if (k >= kNumExpectedTopk and topk_slot_idx[k] < 0)
                    break;

                // Read values
                // 计算第 k 个槽位的源缓冲区基址
                const auto src_base_ptr = get_src_buffer_ptr_func(topk_slot_idx[k]);
                vec_t values[kUnrollFactor] = {};
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j) {
                    // 带谓词加载：槽位非法时不真正访存，读到 0
                    values[j] = ptx::ldg_with_ge_pred(
                        src_base_ptr + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), topk_slot_idx[k], 0);
                }

                // Reduce
                // 将该槽位的数据累加进 FP32 累加器
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    ptx::accumulate(reduced[j], bf162_view[j]);
            }

            // Wait buffer releases for the first write
            // 第一轮写目标缓冲区前等待其释放（仅首次需要）
            if (i == 0)
                wait_buffer_func();

            // Cast into shared memory
            // 将 FP32 累加结果舍入转换回 BF16 并写入目标缓冲区
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                // 每次转换一个向量
                vec_t casted_value;
                // 以 bf16x2 视角访问该向量
                auto bf162_view = reinterpret_cast<nv_bfloat162*>(&casted_value);
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    // 一对 float 舍入打包成一个 bf16x2（__float22bfloat162_rn 为就近舍入）
                    bf162_view[l] = __float22bfloat162_rn(reduced[j * (kNumElemsPerVec / 2) + l]);
                // 写入目标缓冲区中本 lane 负责的位置
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = casted_value;
            }
        }
    }
}

}  // namespace deep_ep::ep
// 结束 deep_ep::ep 命名空间
