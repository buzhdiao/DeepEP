#pragma once
// 头文件保护：保证本文件只被包含一次

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量（TMA/RDMA 对齐字节数）与静态断言
#include <deep_ep/common/exception.cuh>
// 引入统一断言宏（EP_UNIFIED_ASSERT，主机/设备均可用）
#include <deep_ep/common/math.cuh>
// 引入对齐与指针偏移工具（math::align、advance_ptr 等）
#include <deep_ep/common/ptx.cuh>
// 引入 PTX 工具（mbarrier 类型等）

namespace deep_ep::layout {
// layout 命名空间：各通信子系统的数据布局定义

struct TokenLayout {
    // 描述"单个 token 槽位"在缓冲中的逐字节布局：
    //   [hidden | SF | metadata (可选) | mbarrier (可选)]
    // 三段数据各自按 TMA 对齐字节数对齐，总和再按 RDMA 对齐字节数对齐
    int num_hidden_bytes, num_sf_bytes;
    // hidden 激活值的字节数与 SF（scale factor）的字节数
    // NOTES: the top-k index is always 32-bit
    // 备注：top-k 索引固定使用 32 位整数
    bool with_metadata;
    // 是否携带元数据区（top-k 索引、权重、来源 token 信息）
    int num_topk, num_metadata_bytes;
    // top-k 个数与元数据区总字节数
    void* base;
    // 该 token 槽位的基地址

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        num_hidden_bytes(num_hidden_bytes),
        // 初始化 hidden 字节数
        num_sf_bytes(num_sf_bytes),
        // 初始化 SF 字节数
        // Metadata includes: top-k indices, weight and source rank/token index
        // 元数据包含：top-k 索引、top-k 权重、来源 rank/token 索引
        with_metadata(with_metadata),
        // 初始化是否带元数据
        num_topk(num_topk),
        // 初始化 top-k 个数
        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +
                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),
        // 元数据字节布局：
        //   [top-k 索引: num_topk×4B][top-k 权重: num_topk×4B]
        //   +（若带来源信息）[链表头: 4B][链表 next: num_topk×4B]
        base(base) {
        // 初始化基地址
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        // 编译期断言：int 与 float 同为 4 字节（元数据混排的前提）
        EP_UNIFIED_ASSERT(num_hidden_bytes % kNumTMAAlignmentBytes == 0);
        // 断言：hidden 字节数必须按 TMA 对齐（保证 SF 段起点天然对齐）
    }

    template <bool kWithMBarrier, typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        // 计算单 token 槽位的总字节数；kWithMBarrier 决定是否额外追加 mbarrier 存储区
        // dtype_t 允许用 int 或 int64_t 承载结果
        const auto num_data_bytes = math::align<int>(
            math::align(num_hidden_bytes, kNumTMAAlignmentBytes) +
            math::align(num_sf_bytes, kNumTMAAlignmentBytes) +
            math::align(num_metadata_bytes, kNumTMAAlignmentBytes), kNumRDMAAlignmentBytes);
        // 数据区布局（三段各自 TMA 对齐）：
        //   offset 0:                   hidden（向上对齐到 TMA 对齐）
        //   + SF 段:                    SF 字节数向上对齐到 TMA 对齐
        //   + metadata 段:              元数据字节数向上对齐到 TMA 对齐
        //   三段之和再向上对齐到 RDMA 对齐字节数（整槽位可被 RDMA 原子传输）
        const auto num_bytes =
            num_data_bytes + math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, kNumTMAAlignmentBytes);
        // 总字节数 = 数据区 + （可选的）mbarrier 存储区（8 字节向上对齐到 TMA 对齐），位于槽位末尾
        return static_cast<dtype_t>(num_bytes);
        // 按目标类型返回
    }

    __forceinline__ __device__ __host__ void* get_base_ptr() const {
        // 返回槽位基地址
        return base;
    }

    __forceinline__ __device__ __host__ void set_base_ptr(void* ptr) {
        // 设置槽位基地址
        base = ptr;
    }

    __forceinline__ __device__ __host__ void* get_hidden_ptr() const {
        // hidden 段指针：位于槽位最前端，偏移 0
        return get_base_ptr();
    }

    __forceinline__ __device__ __host__ sf_pack_t* get_sf_ptr() const {
        // SF 段指针：偏移 = align(hidden 字节数, TMA 对齐)，紧跟 hidden 段之后
        return math::advance_ptr<sf_pack_t>(base, math::align(num_hidden_bytes, kNumTMAAlignmentBytes));
    }

    __forceinline__ __device__ __host__ int* get_metadata_ptr() const {
        // 元数据段指针：偏移 = align(hidden, TMA 对齐) + align(SF, TMA 对齐)
        return math::advance_ptr<int>(get_sf_ptr(), math::align(num_sf_bytes, kNumTMAAlignmentBytes));
    }

    __forceinline__ __device__ __host__ int* get_topk_idx_ptr() const {
        // top-k 索引指针：元数据段开头，num_topk 个 int
        return get_metadata_ptr();
    }

    __forceinline__ __device__ __host__ float* get_topk_weights_ptr() const {
        // top-k 权重指针：偏移 = top-k 索引之后，num_topk 个 float
        return math::advance_ptr<float>(get_metadata_ptr(), num_topk * sizeof(int));
    }

    __forceinline__ __device__ __host__ int* get_src_token_global_idx_ptr() const {
        // 来源 token 全局索引指针：偏移 = 索引区 + 权重区之后，1 个 int
        return math::advance_ptr<int>(get_topk_weights_ptr(), num_topk * sizeof(float));
    }

    __forceinline__ __device__ __host__ int* get_linked_list_idx_ptr() const {
        // 链表 next 指针：紧跟来源 token 索引之后的 num_topk 个 int
        // （构成"来源 token -> 本 token 各 top-k 槽位"的反向链表）
        return get_src_token_global_idx_ptr() + 1;
    }

    __forceinline__ __device__ ptx::mbarrier* get_mbarrier_ptr() const {
        // mbarrier 指针：位于槽位末尾，偏移 = 不含 mbarrier 的总字节数
        return math::advance_ptr<ptx::mbarrier>(base, get_num_bytes<false>());
    }
};

template <bool kWithMBarrier>
struct BufferLayout {
    // 描述由多个 token 槽位组成的收发缓冲：按 rank 分区，每 rank 内按 token 排列
    // kWithMBarrier 透传给 TokenLayout，决定每个 token 是否附带 mbarrier 存储区
    TokenLayout token_layout;
    // 单 token 槽位布局（不含具体地址）
    int num_ranks;
    // rank 总数（缓冲被均分为 num_ranks 个 rank 区）
    int num_max_tokens_per_rank;
    // 每 rank 区最多容纳的 token 数

    void* base;
    // 缓冲基地址

    __forceinline__ __device__ __host__
    BufferLayout(const TokenLayout& token_layout,
                 const int& num_ranks,
                 const int& max_num_tokens_per_rank,
                 void* base = nullptr) :
        token_layout(token_layout),
        // 保存 token 槽位布局
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        // 保存 rank 数与每 rank 最大 token 数
        base(base) {}
        // 保存基地址

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_token() const {
        // 单 token 槽位总字节数（按是否含 mbarrier 计算）
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_rank() const {
        // 单 rank 区总字节数 = 每 rank 最大 token 数 × 单 token 字节数
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {
        // 整个缓冲总字节数 = rank 区数 × 单 rank 区字节数
        return get_num_bytes_per_rank() * num_ranks;
    }

    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {
        // 返回缓冲末尾指针（基地址 + 总字节数）
        return math::advance_ptr(base, get_num_bytes());
    }

    __forceinline__ __device__ __host__
    BufferLayout get_rank_buffer(const int& rank_idx) const {
        // 取指定 rank 的子缓冲视图：偏移 = rank 区编号 × 单 rank 区字节数
        // 子视图视作只有 1 个 rank 区、每 rank token 数不变
        return BufferLayout(token_layout,
                            1, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    template <int kNumTokensPerChannel>
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx) const {
        // 取指定 channel 的子缓冲视图：token 按 channel 数交叉划分，
        // 偏移 = 单 token 字节数 × 每 channel token 数 × channel 编号
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % kNumTokensPerChannel == 0);
        // 断言：每 rank token 数必须能被每 channel token 数整除
        return BufferLayout(token_layout,
                            // Do not use `num_max_tokens_per_rank / kNumTokensPerChannel` as the false stride
                            // 注意：不要把 `num_max_tokens_per_rank / kNumTokensPerChannel` 当作错误步长使用
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * kNumTokensPerChannel * channel_idx);
    }

    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {
        // 取指定 token 的槽位视图：偏移 = token 编号 × 单 token 字节数
        // global=true 表示 token_idx 是跨 rank 区的全局编号（要求 num_ranks==1 或显式声明全局）
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        // 断言：仅单 rank 区时才能用局部编号，否则必须传 global=true
        return TokenLayout(token_layout.num_hidden_bytes, token_layout.num_sf_bytes, token_layout.num_topk, token_layout.with_metadata,
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx);
    }
};

}  // namespace deep_ep::layout
// 结束 layout 命名空间
