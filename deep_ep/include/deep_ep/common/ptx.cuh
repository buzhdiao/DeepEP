#pragma once  // 头文件保护，防止被重复包含

#include <cuda/barrier>  // CUDA 标准库 barrier，用于封装 mbarrier 类型
#include <cuda_bf16.h>  // CUDA bfloat16 半精度浮点类型支持

#include <deep_ep/common/compiled.cuh>  // 编译期常量与架构兼容性定义（EP_STATIC_ASSERT 等）
#include <deep_ep/common/exception.cuh>  // 异常与断言宏

// deep_ep::ptx 命名空间：底层 PTX 内联汇编的轻量封装，供上层通信内核使用
namespace deep_ep::ptx {

/// Declarations  // 类型声明
using mbarrier = cuda::barrier<cuda::thread_scope_block>;  // mbarrier：块内可见的异步内存屏障对象（对应 PTX mbarrier）
using arrival_phase = uint32_t;  // 屏障相位编号（0/1 交替翻转），用于 mbarrier 的 parity 等待

#ifdef __CUDACC__  // 以下设备端代码仅在 NVCC 编译时可用

/// Exceptions  // 异常
__forceinline__ __device__ void trap() {  // 设备端主动触发致命陷阱，终止整个 kernel
    asm volatile("trap;");  // PTX trap 指令：使当前 kernel 网格立即进入错误终止状态
}

/// Thread layout  // 线程布局
__forceinline__ __device__ int get_warp_idx() {  // 获取当前线程所属 warp 在 block 内的编号
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);  // 全 warp 同步 shuffle：广播 0 号 lane 的 threadIdx.x/32（warp 编号）
}

__forceinline__ __device__ int get_lane_idx() {  // 获取当前线程在 warp 内的 lane 编号（0-31）
    int lane_idx;  // 存放 lane 编号
    asm volatile("mov.s32 %0, %laneid;" : "=r"(lane_idx));  // PTX mov：读取特殊寄存器 %laneid（当前 lane 编号）
    return lane_idx;  // 返回 lane 编号
}

/// Election  // 选举
__forceinline__ __device__ int elect_one_sync() {  // 从活跃 warp 中选出唯一代表线程，返回本线程是否被选中
    int pred = 0;  // 选中标志，未选中为 0
    // elect.sync：在活跃掩码内同步选举出恰好一个线程并置其谓词；@p mov：仅被选中者写 1
    asm volatile(
        "{\n"  // PTX 汇编块开始
        ".reg .b32 %%rx;\n"  // 声明 32 位临时寄存器 %rx
        ".reg .pred %%px;\n"  // 声明谓词寄存器 %px
        "      elect.sync %%rx|%%px, %1;\n"  // 在 %1（活跃掩码）内选举：成员编号写入 %rx，选中标志写入 %px
        "@%%px mov.s32 %0, 1;\n"  // 若本线程被选中（%px 为真），将 %0 置 1
        "}\n"  // PTX 汇编块结束
        : "+r"(pred)  // %0：读写操作数 pred
        : "r"(0xffffffff));  // %1：全 warp 活跃掩码
    return pred;  // 返回 1 表示本线程是选举出的代表线程
}

/// TMA and `cp.async`  // TMA 批量异步拷贝与 cp.async
__forceinline__ __device__ void mbarrier_init_with_fence(mbarrier* ptr, const int& arrive_count = 1) {  // 初始化共享内存中的 mbarrier 对象
    // mbarrier.init：以到达线程数 %0 初始化 %1 指向的共享内存 mbarrier
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::
                 "r"(arrive_count), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // %0：到达线程数；%1：mbarrier 的共享内存地址（通用地址转换为 shared 窗口偏移）
    // fence.mbarrier_init.release.cluster：release 栅栏，确保 mbarrier 初始化结果对集群内其他线程可见
    asm volatile("fence.mbarrier_init.release.cluster;" ::);
}

__forceinline__ __device__ void mbarrier_invalidate(mbarrier* ptr) {  // 使 mbarrier 对象失效（复用其共享内存前调用）
    // mbarrier.inval：将 %0 指向的 mbarrier 标记为无效
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // %0：mbarrier 的共享内存地址
}

__forceinline__ __device__ void mbarrier_arrive(mbarrier* ptr) {  // 当前线程到达 mbarrier（不带字节数预期）
    // mbarrier.arrive：到达 %0 指向的 mbarrier 并使到达计数加 1，`_` 表示丢弃返回的计数快照
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // %0：mbarrier 的共享内存地址
}

__forceinline__ __device__ void mbarrier_arrive_and_set_tx(mbarrier* ptr, const int& num_bytes) {  // 到达 mbarrier 并附加期望字节数（expect-tx）
    // mbarrier.arrive.expect_tx：到达的同时把 %0（字节数）累加到 tx-count，TMA 拷贝完成时会按字节扣减
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // %0：期望到达字节数；%1：mbarrier 地址
}

__forceinline__ __device__ void mbarrier_wait_and_flip_phase(mbarrier* ptr, arrival_phase& phase) {  // 自旋等待 mbarrier 翻转到指定相位，然后翻转本地相位
    // mbarrier.try_wait.parity：按相位 %1 轮询检查 mbarrier %0 是否完成；成功跳 DONE，否则跳回 LAB_WAIT 重试
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        ".reg .pred       P1; \n\t"  // 声明谓词寄存器 P1
        "LAB_WAIT: \n\t"  // 等待循环标号
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"  // 尝试按相位 %1 等待 mbarrier %0，结果写入 P1；%2 为挂起时间提示（纳秒级）
        "@P1 bra DONE; \n\t"  // 若 P1 为真（等待成功），跳转到 DONE
        "bra     LAB_WAIT; \n\t"  // 否则跳回 LAB_WAIT 继续自旋
        "DONE: \n\t"  // 等待结束标号
        "}" ::  // PTX 汇编块结束
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),  // %0：mbarrier 的共享内存地址
        "r"(phase), "r"(0x989680));  // %1：目标相位；%2：0x989680 = 10000000（try_wait 的挂起提示）
    phase ^= 1;  // 翻转调用方的相位状态（0↔1 交替）
}

template <int kNumBytes>
__forceinline__ __device__ void st_bulk(void* smem_ptr) {  // 将 kNumBytes 字节的共享内存批量清零
    EP_STATIC_ASSERT(kNumBytes % 8 == 0, "`st.bulk` requires size to be a multiple of 8");  // st.bulk 要求大小为 8 的倍数
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 使用硬件 st.bulk 指令
    if (elect_one_sync()) {  // warp 内选举一个代表线程执行，其余线程跳过
        // st.bulk.weak：从 %0 开始的 %1 字节共享内存整体写 0（weak 表示弱内存序、不保证排序）
        asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;\n" ::
                     "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),  // %0：共享内存起始地址
                     "r"(kNumBytes)  // %1：清零的字节长度
                     : "memory");  // 告知编译器存在内存副作用，禁止重排
    }
#else  // 旧架构回退方案：普通 64 位写循环清零
    #pragma unroll  // 提示编译器展开循环
    for (int i = get_lane_idx(); i < kNumBytes / 8; i += 32)  // 每个 lane 间隔 32 处理一个 8 字节单元
        static_cast<uint64_t*>(smem_ptr)[i] = 0;  // 将该 8 字节单元清零
#endif
    __syncwarp();  // warp 内同步，保证清零结果对所有 lane 可见
}

__forceinline__ __device__ void tma_store_fence() {  // 发出异步代理栅栏，保证此前的共享内存写对 TMA 存储可见
    asm volatile("fence.proxy.async.shared::cta;");  // PTX fence.proxy.async：通用代理与异步（TMA）代理之间的栅栏
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {  // 等待直到最多还剩 kNumRemainingWaits 个未完成的 TMA 存储组
    // cp.async.bulk.wait_group：等待批量异步拷贝组完成到只剩 %0 个（等待写全部落盘）
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");  // %0：允许剩余未完成的组数（编译期常量）
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait_read() {  // 同上，但只需等到源数据可安全复用（读完成即可）
    // cp.async.bulk.wait_group.read：等待组内拷贝“读源完成”，即共享内存源缓冲可以覆盖
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(kNumRemainingWaits) : "memory");  // %0：允许剩余未完成的组数
}

enum L2CacheHint: int64_t {  // L2 缓存驻留提示（以编码形式随 TMA 指令传入）
    kEvictFirst = 0x12f0000000000000ll,  // 尽快逐出策略：加载的数据很快被消费，优先腾出缓存
    kEvictNormal = 0x1000000000000000ll  // 正常逐出策略：按默认缓存替换策略处理
};

__forceinline__ __device__ void tma_load_1d(
    const void* dst_ptr, const void* src_ptr, mbarrier* ptr, const int& num_bytes,
    const L2CacheHint& hint = L2CacheHint::kEvictFirst) {  // TMA 一维加载：全局内存 -> 共享内存，完成字节数由 mbarrier 记账
    // NOTES: normally, the loaded part will be evicted soon. 注：加载的数据通常很快被消费，故默认使用“尽快逐出”提示
    // cp.async.bulk...mbarrier::complete_tx::bytes：批量拷贝，完成后自动将实际字节数计入 mbarrier 的 tx-count
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),  // %0：目标共享内存地址
        "l"(src_ptr),  // %1：源全局内存地址
        "r"(num_bytes),  // %2：拷贝字节数
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),  // %3：用于完成记账的 mbarrier 地址
        "l"(hint)  // %4：L2 缓存驻留提示
        : "memory");  // 内存副作用，禁止编译器重排
}

__forceinline__ __device__ void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const int& num_bytes,
    const L2CacheHint& hint = L2CacheHint::kEvictNormal) {  // TMA 一维存储：共享内存 -> 全局内存，按 bulk_group 记账
    // NOTES: normally, the stored part will be used soon. 注：存储的数据很快会被使用，故默认使用“正常驻留”提示
    // cp.async.bulk.global.shared::cta.bulk_group：批量存储，加入当前线程的 bulk 组，稍后用 commit/wait_group 管理
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),  // %0：目标全局内存地址
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),  // %1：源共享内存地址
                 "r"(num_bytes),  // %2：拷贝字节数
                 "l"(hint)  // %3：L2 缓存驻留提示
                 : "memory");  // 内存副作用
}

__forceinline__ __device__ void tma_store_reduce_add_f32(
    float* dst_ptr, const void* src_ptr, const int& num_bytes) {  // TMA 归约存储：把共享内存的 f32 数据以加法方式归约到全局内存
    // cp.reduce.async.bulk...add.f32：异步批量归约，对目标地址执行 add 操作（而非覆盖写）
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n" ::
        "l"(dst_ptr),  // %0：目标全局内存地址（归约目标）
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),  // %1：源共享内存地址
        "r"(num_bytes)  // %2：归约字节数
        : "memory");  // 内存副作用
}

__forceinline__ __device__ void tma_store_commit() {  // 提交当前线程的所有待处理批量异步拷贝为一个 bulk 组
    asm volatile("cp.async.bulk.commit_group;");  // PTX commit_group：将之前发出的 cp.async.bulk 打包成组，供 wait_group 等待
}

__forceinline__ __device__ float4 multimem_ld_reduce_add_f32_with_gt_pred(const float4* ptr, const int& lhs = 1, const int& rhs = 0) {  // 多播读归约：对 NVLink 域内所有 rank 的同一地址做加法归约并加载
    float4 value = make_float4(0, 0, 0, 0);  // 存放归约结果，初始化为 0
    // setp.gt.s32：比较 %5 > %6 生成谓词 p；@p multimem.ld_reduce...add：仅当谓词为真时，
    // 对多播地址 %4 处的 4 个 f32 执行跨 rank 加法归约（relaxed.gpu 弱内存序），结果写入 %0-%3
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.gt.s32 p, %5, %6;\n\t"  // p = (lhs > rhs)，作为执行开关
        "  @p multimem.ld_reduce.relaxed.gpu.global.add.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"  // 条件执行：多播加法归约加载 4 个 f32
        "}"  // PTX 汇编块结束
        : "+f"(value.x), "+f"(value.y), "+f"(value.z), "+f"(value.w)  // %0-%3：读写操作数，接收归约结果
        : "l"(ptr), "r"(lhs), "r"(rhs)  // %4：多播地址；%5/%6：谓词比较的两个输入
        : "memory");  // 内存副作用
    return value;  // 返回跨 rank 归约后的值
}

// Multicast store: writes `value` to the same address on every rank in the NVLink domain.
// 多播存储：将 value 写入 NVLink 域内每个 rank 的同一地址（用于广播）
__forceinline__ __device__ void multimem_st_f32_with_gt_pred(float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) {
    // setp.gt.s32：生成谓词 p = (lhs > rhs)；@p multimem.st：条件执行多播写
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.gt.s32 p, %5, %6;\n\t"  // p = (lhs > rhs)，作为执行开关
        "  @p multimem.st.relaxed.gpu.global.v4.f32 [%0], {%1, %2, %3, %4};\n\t"  // 条件执行：将 4 个 f32 多播写入所有 rank 的同一地址
        "}"  // PTX 汇编块结束
        :: "l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w), "r"(lhs), "r"(rhs)  // %0：多播地址；%1-%4：写入值；%5/%6：谓词输入
        : "memory");  // 内存副作用
}

__forceinline__ __device__ void multimem_cp_async_bulk(
    const void* dst_ptr, const void* src_ptr, const int& num_bytes) {  // 多播批量拷贝：从多播源读取并写入共享内存（聚合跨 rank 读）
    // multimem.cp.async.bulk：从多播地址批量拷贝到共享内存，硬件自动聚合多个 rank 的读请求
    asm volatile("multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                 "l"(dst_ptr),  // %0：目标共享内存地址
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),  // %1：源共享内存地址（多播映射）
                 "r"(num_bytes)  // %2：拷贝字节数
                 : "memory");  // 内存副作用
}

template <class dtype_t>
__forceinline__ __device__ void cp_async_ca(const dtype_t* gmem_src, const dtype_t* smem_dst) {  // 异步拷贝 4/8/16 字节：全局内存 -> 共享内存（ca = 全级缓存）
    EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8 or sizeof(dtype_t) == 16, "Invalid dtype bytes");  // 仅支持 4/8/16 字节类型
    // cp.async.ca：非阻塞异步拷贝；L2::128B 表示以 128 字节粒度使用 L2 缓存；ca 表示 L1/L2 都缓存
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], %2;\n" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),  // %0：目标共享内存地址
                 "l"(gmem_src),  // %1：源全局内存地址
                 "n"(sizeof(dtype_t)));  // %2：拷贝字节数（编译期常量）
}

__forceinline__ __device__ void cp_async_mbarrier_arrive(mbarrier* ptr) {  // 将 cp.async 的完成事件绑定到 mbarrier
    // cp.async.mbarrier.arrive：此前提交的 cp.async 全部完成后，自动到达 %0 指向的 mbarrier
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // %0：mbarrier 的共享内存地址
}

/// Barriers  // 屏障
template <int kNumThreads>
__forceinline__ __device__ void named_barrier(const int& idx) {  // 具名 barrier：按编号 idx 同步 kNumThreads 个线程
    // Equivalent to `barrier.sync.aligned`, which requires all threads run the same location of code.
    // 注：等价于 barrier.sync.aligned，要求所有参与线程执行同一指令位置
    asm volatile("bar.sync %0, %1;" ::"r"(idx), "r"(kNumThreads));  // PTX bar.sync：%0 为 barrier 编号，%1 为参与线程数
}

template <int kNumThreads>
__forceinline__ __device__ void named_barrier_unaligned(const int& idx) {  // 非对齐具名 barrier：允许不同线程从不同代码位置到达
    // barrier.sync（非 aligned 版本）：同一编号、线程数合计足够的线程在此汇合
    asm volatile("barrier.sync %0, %1;" ::"r"(idx), "r"(kNumThreads) : "memory");  // %0：barrier 编号；%1：参与线程数；带内存副作用
}

/// LD/ST instructions  // 带谓词的加载/存储指令
__forceinline__ __device__ int4 ldg_with_ge_pred(const int4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {  // 带谓词（>=）的只读缓存加载 int4
    int4 ret = make_int4(0, 0, 0, 0);  // 默认返回 0，谓词不满足时保持 0
    // setp.ge.s32：生成谓词 p = (lhs >= rhs)；@p ld...nc：条件执行非一致（read-only，等同 __ldg）加载，
    // L1::no_allocate 不占用 L1，L2::cache_hint 附带缓存驻留提示 %7
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.ge.s32 p, %5, %6;\n\t"  // p = (lhs >= rhs)
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32 {%0, %1, %2, %3}, [%4], %7;\n\t"  // 条件执行：nc 只读缓存加载 4 个 s32
        "}"  // PTX 汇编块结束
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)  // %0-%3：读写操作数，接收加载结果
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)  // %4：地址；%5/%6：谓词输入；%7：缓存提示
        : "memory");  // 内存副作用
    return ret;  // 返回加载（或默认 0）结果
}

__forceinline__ __device__ int4 ldg_with_gt_pred(const int4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {  // 带谓词（>）的只读缓存加载 int4
    int4 ret = make_int4(0, 0, 0, 0);  // 默认返回 0
    // setp.gt.s32：p = (lhs > rhs)；其余同上：nc 只读缓存条件加载
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.gt.s32 p, %5, %6;\n\t"  // p = (lhs > rhs)
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32 {%0, %1, %2, %3}, [%4], %7;\n\t"  // 条件执行：nc 只读缓存加载 4 个 s32
        "}"  // PTX 汇编块结束
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)  // %0-%3：接收加载结果
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)  // %4：地址；%5/%6：谓词输入；%7：缓存提示
        : "memory");  // 内存副作用
    return ret;  // 返回加载结果
}

__forceinline__ __device__ int4 ld_with_ge_pred(const int4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {  // 带谓词（>=）的普通（可写）加载 int4
    int4 ret = make_int4(0, 0, 0, 0);  // 默认返回 0
    // 与 ldg 版本的区别：不带 .nc，走普通（coherent）全局加载路径
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.ge.s32 p, %5, %6;\n\t"  // p = (lhs >= rhs)
        "  @p ld.L1::no_allocate.L2::cache_hint.global.v4.s32 {%0, %1, %2, %3}, [%4], %7;\n\t"  // 条件执行：普通加载 4 个 s32
        "}"  // PTX 汇编块结束
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)  // %0-%3：接收加载结果
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)  // %4：地址；%5/%6：谓词输入；%7：缓存提示
        : "memory");  // 内存副作用
    return ret;  // 返回加载结果
}

__forceinline__ __device__ float4 ld_with_gt_pred(const float4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {  // 带谓词（>）的普通加载 float4
    float4 ret = make_float4(0, 0, 0, 0);  // 默认返回 0
    // 普通全局加载 4 个 f32，条件与缓存策略同上
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.gt.s32 p, %5, %6;\n\t"  // p = (lhs > rhs)
        "  @p ld.L1::no_allocate.L2::cache_hint.global.v4.f32 {%0, %1, %2, %3}, [%4], %7;\n\t"  // 条件执行：普通加载 4 个 f32
        "}"  // PTX 汇编块结束
        : "+f"(ret.x), "+f"(ret.y), "+f"(ret.z), "+f"(ret.w)  // %0-%3：接收加载结果
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)  // %4：地址；%5/%6：谓词输入；%7：缓存提示
        : "memory");  // 内存副作用
    return ret;  // 返回加载结果
}

#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // 256 位加载/存储需要 SM100+
__forceinline__ __device__ longlong4_t ld_with_gt_pred(const longlong4_t* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {  // 带谓词（>）的普通加载 longlong4_t（256 位）
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);  // 默认返回 0
    // v4.s64：一次加载 4 个 64 位整数（共 256 位）
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.gt.s32 p, %5, %6;\n\t"  // p = (lhs > rhs)
        "  @p ld.L1::no_allocate.L2::cache_hint.global.v4.s64 {%0, %1, %2, %3}, [%4], %7;\n\t"  // 条件执行：普通加载 4 个 s64
        "}"  // PTX 汇编块结束
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)  // %0-%3：接收加载结果
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)  // %4：地址；%5/%6：谓词输入；%7：缓存提示
        : "memory");  // 内存副作用
    return ret;  // 返回加载结果
}

__forceinline__ __device__ longlong4_t ldg_with_ge_pred(const longlong4_t* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {  // 带谓词（>=）的只读缓存加载 longlong4_t
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);  // 默认返回 0
    // .nc 只读（非一致）加载，其余同上
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.ge.s32 p, %5, %6;\n\t"  // p = (lhs >= rhs)
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s64 {%0, %1, %2, %3}, [%4], %7;\n\t"  // 条件执行：nc 只读缓存加载 4 个 s64
        "}"  // PTX 汇编块结束
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)  // %0-%3：接收加载结果
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)  // %4：地址；%5/%6：谓词输入；%7：缓存提示
        : "memory");  // 内存副作用
    return ret;  // 返回加载结果
}

__forceinline__ __device__ longlong4_t ldg(const longlong4_t* ptr) {  // 无条件只读缓存加载 longlong4_t（256 位）
    longlong4_t ret;  // 存放加载结果
    // .nc 只读加载：数据为只读时可用，绕过 L1 一致性协议，缓存效率更高
    asm volatile(
        "ld.L1::no_allocate.global.nc.v4.s64 {%0, %1, %2, %3}, [%4];\n\t"  // nc 只读加载 4 个 s64
        : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)  // %0-%3：输出加载结果
        : "l"(ptr)  // %4：源地址
        : "memory");  // 内存副作用
    return ret;  // 返回加载结果
}
#endif

__forceinline__ __device__ int4 ldg(const int4* ptr) {  // 只读缓存加载 int4，直接复用 CUDA 内建函数
    return __ldg(ptr);  // __ldg：经只读数据缓存（texture 路径）的非一致加载
}

template <typename dtype_t>
__forceinline__ __device__ void st_with_ge_pred(dtype_t* ptr, dtype_t value, const int& lhs = 1, const int& rhs = 0) {  // 带谓词（>=）的 4 字节全局存储
    EP_STATIC_ASSERT(sizeof(dtype_t) == 4, "Invalid data type");  // 仅支持 4 字节类型
    auto view = *reinterpret_cast<int*>(&value);  // 将值按 32 位整数视图传给 PTX
    // setp.ge.s32：p = (lhs >= rhs)；@p st.global.s32：条件执行存储
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.ge.s32 p, %2, %3;\n\t"  // p = (lhs >= rhs)
        "  @p st.global.s32 [%0], %1;\n\t"  // 条件执行：存储 1 个 s32 到全局内存
        "}"
        :: "l"(ptr), "r"(view), "r"(lhs), "r"(rhs)  // %0：地址；%1：存储值；%2/%3：谓词输入
        : "memory");  // 内存副作用
}


__forceinline__ __device__ void st_with_gt_pred(float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) {  // 带谓词（>）的全局存储 float4（128 位）
    // setp.gt.s32：p = (lhs > rhs)；@p st.global.v4.f32：条件执行 4 个 f32 的向量存储
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.gt.s32 p, %5, %6;\n\t"  // p = (lhs > rhs)
        "  @p st.global.v4.f32 [%0], {%1, %2, %3, %4};\n\t"  // 条件执行：存储 4 个 f32
        "}"
        :: "l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w), "r"(lhs), "r"(rhs)  // %0：地址；%1-%4：存储值；%5/%6：谓词输入
        : "memory");  // 内存副作用
}

#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // 256 位存储需要 SM100+

__forceinline__ __device__ void st_with_gt_pred(longlong4_t* ptr, const longlong4_t& value, const int& lhs = 1, const int& rhs = 0) {  // 带谓词（>）的全局存储 longlong4_t（256 位）
    // v4.s64：一次存储 4 个 64 位整数
    asm volatile(
        "{\n\t"  // PTX 汇编块开始
        "  .reg .pred p;\n\t"  // 声明谓词寄存器 p
        "  setp.gt.s32 p, %5, %6;\n\t"  // p = (lhs > rhs)
        "  @p st.global.v4.s64 [%0], {%1, %2, %3, %4};\n\t"  // 条件执行：存储 4 个 s64
        "}"
        :: "l"(ptr), "l"(value.x), "l"(value.y), "l"(value.z), "l"(value.w), "r"(lhs), "r"(rhs)  // %0：地址；%1-%4：存储值；%5/%6：谓词输入
        : "memory");  // 内存副作用
}
#endif

template <typename dtype_t>
__forceinline__ __device__ dtype_t ld_volatile(const void* ptr) {  // volatile 语义加载：绕过常规缓存优化，常用于轮询标志位
    if constexpr (sizeof(dtype_t) == 4) {  // 4 字节类型分支
        uint32_t value;  // 中转整型值
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));  // PTX ld.volatile：volatile 加载 32 位全局内存（每次真实访存）
        return reinterpret_cast<const dtype_t&>(value);  // 按目标类型重新解释返回
    } else if constexpr (sizeof(dtype_t) == 8) {  // 8 字节类型分支
        uint64_t value;  // 中转整型值
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr));  // volatile 加载 64 位全局内存
        return reinterpret_cast<const dtype_t&>(value);  // 按目标类型重新解释返回
    } else {  // 不支持的长度
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");  // 编译期报错
    }
}

__forceinline__ __device__ void red_add(const int64_t* ptr, const int64_t& value) {  // GPU 作用域原子加（弱内存序）
    // TODO(NVCC): why don't NVCC support `s64`?  // 保留原注释：NVCC 为何不支持 s64
    asm volatile("red.gpu.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));  // PTX red：GPU 作用域原子加，%0 目标地址，%1 加数
}

__forceinline__ __device__ void red_add_rel_sys(const int* ptr, const int& value) {  // 系统作用域、release 内存序的原子加（int）
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));  // PTX red.release.sys：release 语义原子加，作用域覆盖 CPU 与所有 GPU
}

__forceinline__ __device__ void red_add_rel_sys(const int64_t* ptr, const int64_t& value) {  // 系统作用域、release 内存序的原子加（int64）
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));  // PTX red.release.sys：64 位 release 原子加
}

__forceinline__ __device__ void red_add_rel_gpu(const int* ptr, const int& value) {  // GPU 作用域、release 内存序的原子加（int）
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));  // PTX red.release.gpu：release 语义原子加，作用域为单个 GPU（NVLink 域）
}

__forceinline__ __device__ void red_add_rel_gpu(const int64_t* ptr, const int64_t& value) {  // GPU 作用域、release 内存序的原子加（int64）
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));  // PTX red.release.gpu：64 位 release 原子加
}

template <typename dtype_t>
__forceinline__ __device__ dtype_t ld_acquire_sys(const dtype_t* ptr) {  // 系统作用域、acquire 内存序的加载
    if constexpr (sizeof(dtype_t) == 4) {  // 4 字节类型分支
        uint32_t value;  // 中转整型值
        asm volatile("ld.acquire.sys.L1::no_allocate.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));  // PTX ld.acquire.sys：acquire 语义加载，保证后续读取不早于本读
        return reinterpret_cast<const dtype_t&>(value);  // 按目标类型重新解释返回
    } else if constexpr (sizeof(dtype_t) == 8) {  // 8 字节类型分支
        uint64_t value;  // 中转整型值
        asm volatile("ld.acquire.sys.L1::no_allocate.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr));  // acquire 语义加载 64 位
        return reinterpret_cast<const dtype_t&>(value);  // 按目标类型重新解释返回
    } else {  // 不支持的长度
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");  // 编译期报错
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_relaxed_gpu(void* ptr, dtype_t value) {  // GPU 作用域、relaxed（弱）内存序的存储
    if constexpr (sizeof(dtype_t) == 4) {  // 4 字节类型分支
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);  // 按位重解释为整数
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));  // PTX st.relaxed.gpu：弱内存序存储，作用域为单 GPU
    } else if constexpr (sizeof(dtype_t) == 8) {  // 8 字节类型分支
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);  // 按位重解释为整数
        asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));  // 弱内存序存储 64 位
    } else {  // 不支持的长度
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");  // 编译期报错
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_relaxed_sys(void* ptr, dtype_t value) {  // 系统作用域、relaxed（弱）内存序的存储
    if constexpr (sizeof(dtype_t) == 4) {  // 4 字节类型分支
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);  // 按位重解释为整数
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));  // PTX st.relaxed.sys：弱内存序存储，作用域覆盖 CPU 与所有 GPU
    } else if constexpr (sizeof(dtype_t) == 8) {  // 8 字节类型分支
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);  // 按位重解释为整数
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));  // 弱内存序存储 64 位
    } else {  // 不支持的长度
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");  // 编译期报错
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_release_sys(void* ptr, dtype_t value) {  // 系统作用域、release 内存序的存储
    if constexpr (sizeof(dtype_t) == 4) {  // 4 字节类型分支
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);  // 按位重解释为整数
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));  // PTX st.release.sys：release 语义存储，此前的写对其后的读者可见
    } else if constexpr (sizeof(dtype_t) == 8) {  // 8 字节类型分支
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);  // 按位重解释为整数
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));  // release 语义存储 64 位
    } else {  // 不支持的长度
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");  // 编译期报错
    }
}

// Adjust registers  // 调整寄存器数量
// Initial register adjustment assuming the budget for __launch_bounds__(kNumThreads, 1).
// 注：以 __launch_bounds__(kNumThreads, 1) 的寄存器预算为基准进行的初始寄存器数量假设
template <int kNumRegs, int kNumThreads>
__device__ __forceinline__ void warpgroup_reg_realloc() {  // warpgroup 级寄存器重分配（setmaxnreg）
    EP_STATIC_ASSERT(kNumRegs % 8 == 0 and kNumRegs >= 24 and kNumRegs <= 256, "Invalid register target");  // 目标寄存器数必须合法（8 对齐、24-256）
    constexpr int kNumInitialRegisters = (65536 / kNumThreads / 8) * 8;  // 按每 SM 64K 寄存器文件、8 对齐计算的初始每线程寄存器数

    if constexpr (kNumInitialRegisters > 255) {  // 初始寄存器数已超过 PTX 每线程 255 的上限
        return;  // 无需也无法调整
    }

    if constexpr (kNumRegs >= kNumInitialRegisters)  // 需要更多寄存器
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));  // PTX setmaxnreg.inc：整 warpgroup 对齐地增加每线程寄存器上限
    else  // 需要更少寄存器
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));  // PTX setmaxnreg.dec：释放寄存器给同 warpgroup 中需要更多寄存器的 warp
}

/// General fences  // 通用内存栅栏
__device__ __forceinline__ void fence_acquire_gpu() {  // GPU 作用域 acquire 栅栏
    asm volatile("fence.acquire.gpu;" ::: "memory");  // PTX fence.acquire.gpu：保证栅栏后的读取不会早于栅栏前的读取被观察到
}

__device__ __forceinline__ void fence_acq_rel_sys() {  // 系统作用域 acquire-release 栅栏
    asm volatile("fence.acq_rel.sys;" ::: "memory");  // PTX fence.acq_rel.sys：兼具 acquire 与 release 语义，作用域覆盖 CPU 与所有 GPU
}

/// Intrinsics  // warp 级内建函数封装
template <typename dtype_t>
__device__ __forceinline__ dtype_t exchange(dtype_t ptr, const int& src_lane_idx) {  // warp 内交换任意宽度的值：从指定 lane 取值
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");  // 类型大小必须是 int 的整数倍
    const auto send_int_values = reinterpret_cast<int*>(&ptr);  // 将发送值按 int 数组展开
    dtype_t recv_dtype;  // 接收结果
    auto recv_int_values = reinterpret_cast<int*>(&recv_dtype);  // 接收缓冲也按 int 数组展开
    #pragma unroll  // 展开逐 int 的 shuffle 循环
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)  // 逐个 32 位字交换
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);  // 全 warp 同步 shuffle：读取 src_lane_idx 线程的第 i 个字
    return recv_dtype;  // 返回重组后的完整值
}

__device__ __forceinline__ unsigned gather(const bool& value) {  // 收集 warp 内各线程的布尔值为 32 位掩码
    return __ballot_sync(0xffffffff, value);  // ballot：真线程的对应位置 1
}

__device__ __forceinline__ bool all(const bool& value) {  // 判断 warp 内所有线程的 value 是否都为真
    return __all_sync(0xffffffff, value);  // 全体为真才返回真
}

__device__ __forceinline__ bool any(const bool& value) {  // 判断 warp 内是否存在线程的 value 为真
    return __any_sync(0xffffffff, value);  // 任一为真即返回真
}

__device__ __forceinline__ unsigned reduce_or(const unsigned& value) {  // warp 内 32 位按位或归约
    return __reduce_or_sync(0xffffffff, value);  // 硬件 reduce.or 指令
}

__device__ __forceinline__ unsigned long long reduce_or(const unsigned long long& value) {  // warp 内 64 位按位或归约
    const auto low = __reduce_or_sync(0xffffffff, static_cast<unsigned>(value));  // 归约低 32 位
    const auto high = __reduce_or_sync(0xffffffff, static_cast<unsigned>(value >> 32));  // 归约高 32 位
    return (static_cast<unsigned long long>(high) << 32) | low;  // 高低拼接回 64 位结果
}

__device__ __forceinline__ int reduce_add(const int& value) {  // warp 内整数加法归约
    return __reduce_add_sync(0xffffffff, value);  // 硬件 reduce.add 指令
}

__device__ __forceinline__ unsigned match(const int& value) {  // 返回 warp 内与 value 相等的线程掩码
    return __match_any_sync(0xffffffff, value);  // match.any：值相同的线程对应位为 1
}

__device__ __forceinline__ int fns(const unsigned& value, const int& offset) {  // 查找从低位起第 offset 个被置位的位编号
    return __fns(value, 0, offset);  // find-nth-set-bit：返回 value 中从 LSB 起第 offset 个 1 的位置
}

template <typename dtype_t>
__device__ __forceinline__ auto ffs(const dtype_t& value) {  // 求最低置位位的编号（0 起）
    if constexpr (sizeof(dtype_t) == 4) {  // 32 位版本
        return __ffs(static_cast<int>(value)) - 1;  // __ffs 返回 1 起编号，此处转 0 起
    } else {  // 64 位版本
        EP_STATIC_ASSERT(sizeof(dtype_t) == 8, "Invalid data type");  // 仅支持 8 字节
        return __ffsll(static_cast<long long>(value)) - 1;  // 64 位 ffs，同样转 0 起
    }
}

__device__ __forceinline__ int get_master_lane_idx(const unsigned& mask) {  // 求 mask 中最高置位位对应的 lane 编号（主 lane）
    // Equivalent to `31 - __clz(mask)`  // 等价于 31 - __clz(mask)。此处用 PTX 位查找指令实现
    int highest_idx;  // 存放最高置位位编号
    asm volatile("bfind.u32 %0, %1;" : "=r"(highest_idx) : "r"(mask));  // PTX bfind：查找 %1 的最高有效置位位编号写入 %0
    return highest_idx;  // 返回主 lane 编号
}

__device__ __forceinline__ bool deduplicate(const int& value, const int& lane_idx) {  // 判断本线程是否为组内相同 value 的去重代表
    return get_master_lane_idx(match(value)) == lane_idx;  // 值相同的线程中只有主 lane 返回真
}

__device__ __forceinline__ int warp_inclusive_sum(int value, const int& lane_idx) {  // warp 内包含式前缀和（含自身）
    #pragma unroll  // 展开 5 轮迭代
    for (int offset = 1; offset < 32; offset <<= 1) {  // Hillis-Steele 扫描：步长 1,2,4,8,16
        const auto synced = __shfl_up_sync(0xffffffff, value, offset);  // 取上方 offset 处线程的当前值
        if (lane_idx >= offset)  // 越过 warp 边界的 lane 不参与累加
            value += synced;  // 累加上游值
    }
    return value;  // 返回包含自身的前缀和
}

__device__ __forceinline__ int warp_exclusive_sum(const int& value, const int& lane_idx) {  // warp 内排他式前缀和（不含自身）
    return warp_inclusive_sum(value, lane_idx) - value;  // 包含式前缀和减去自身即得排他前缀和
}

__device__ __forceinline__ float4 fadd4(const float4& a, const float4& b) {  // float4 逐分量加法
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 使用打包的双 float 加法指令
    const auto xy = __fadd2_rn(make_float2(a.x, a.y), make_float2(b.x, b.y));  // 一次指令完成 x、y 两个分量的加法（就近舍入）
    const auto zw = __fadd2_rn(make_float2(a.z, a.w), make_float2(b.z, b.w));  // 一次指令完成 z、w 两个分量的加法
    return make_float4(xy.x, xy.y, zw.x, zw.y);  // 组装回 float4
#else  // 旧架构：逐分量普通加法
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);  // 4 次标量加法
#endif
}

__device__ __forceinline__ float4 fmul4(const float4& a, const float4& b) {  // float4 逐分量乘法
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 使用打包的双 float 乘法指令
    const auto xy = __fmul2_rn(make_float2(a.x, a.y), make_float2(b.x, b.y));  // 一次指令完成 x、y 两个分量的乘法
    const auto zw = __fmul2_rn(make_float2(a.z, a.w), make_float2(b.z, b.w));  // 一次指令完成 z、w 两个分量的乘法
    return make_float4(xy.x, xy.y, zw.x, zw.y);  // 组装回 float4
#else  // 旧架构：逐分量普通乘法
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);  // 4 次标量乘法
#endif
}

__device__ __forceinline__ float4 fmul4(const float4& a, const float& b) {  // float4 与标量的逐分量乘法
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 使用打包的双 float 乘法指令
    const auto b2 = make_float2(b, b);  // 将标量复制成 float2 以便打包运算
    const auto xy = __fmul2_rn(make_float2(a.x, a.y), b2);  // 一次指令完成 x、y 两个分量的乘法
    const auto zw = __fmul2_rn(make_float2(a.z, a.w), b2);  // 一次指令完成 z、w 两个分量的乘法
    return make_float4(xy.x, xy.y, zw.x, zw.y);  // 组装回 float4
#else  // 旧架构：逐分量普通乘法
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);  // 4 次标量乘法
#endif
}

__device__ __forceinline__ void accumulate(float2& a, nv_bfloat162 b) {  // 将 bfloat16x2 累加到 float2
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 走硬件融合指令
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100.
    // 注：在 SM100 上用 add.rn.f32.bf16 指令完成“bf16 转 f32 + 加法”的融合操作
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));  // 将 b.x（bf16）转 f32 并就近舍入累加到 a.x
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));  // 将 b.y（bf16）转 f32 并就近舍入累加到 a.y
#else  // 旧架构：显式转换后逐分量相加
    const auto [x, y] = __bfloat1622float2(b);  // bf16x2 解包并转换为两个 float
    a.x += x, a.y += y;  // 逐分量累加
#endif
}

__device__ __forceinline__ nv_bfloat162 cvt_rs_bf16x2(const float2& value, const uint32_t& bits) {  // float2 转 bf16x2（带随机舍入）
    uint32_t result;  // 打包结果
    asm("cvt.rs.bf16x2.f32 %0, %2, %1, %3;" : "=r"(result) : "f"(value.x), "f"(value.y), "r"(bits));  // PTX cvt.rs：随机舍入的 f32->bf16x2 打包转换，bits 为随机数
    return reinterpret_cast<const nv_bfloat162&>(result);  // 按目标类型重新解释返回
}

#endif

} // namespace deep_ep::ptx
