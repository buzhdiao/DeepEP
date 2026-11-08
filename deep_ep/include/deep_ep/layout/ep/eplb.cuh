#pragma once
// 头文件保护：保证本文件只被包含一次

#include <deep_ep/common/compiled.cuh>
// 引入编译期常量与静态断言
#include <deep_ep/common/exception.cuh>
// 引入断言宏（EP_STATIC_ASSERT / EP_DEVICE_ASSERT）
#include <deep_ep/common/math.cuh>
// 引入数学工具（ceil_div 等）
#ifdef __CUDACC__
#include <cub/block/block_scan.cuh>
// 引入 CUB 块级前缀和（BlockScan，用于任务压缩）
#include <deep_ep/comm/handle.cuh>
// 引入 NCCL Gin 通信句柄（get_sym_ptr 等对称寻址接口）
#endif

namespace deep_ep::layout {
// layout 命名空间：数据布局定义

// One exchange pair between an original expert and a redundant expert slot
// 一条 EPLB（专家并行负载均衡）任务：把一个"原专家"的权重交换到某个"冗余专家槽位"
struct EPLBTask {
    int local_expert_idx;
    // 本 rank 上要复制的原专家在其本地位中的编号
    int dst_nvl_rank_idx;
    // 冗余专家槽位所在的 NVL rank 编号（拷贝目标节点）
    int redundant_expert_idx;
    // 该 NVL rank 上冗余专家槽位的编号（目标槽位）
};

struct alignas(kNumTMAAlignmentBytes) EPLBSharedMemoryLayout {
    // EPLB 任务列表的共享内存布局：按 TMA 对齐；开头为任务数，其后是任务数组
    static constexpr int kNumBytes = 32 * 1024;
    // 该共享内存区的固定大小：32 KiB
    static constexpr int kNumMaxTasks = (kNumBytes - sizeof(int)) / sizeof(EPLBTask);
    // 最大任务数 =（总字节数 - 计数字段 4B）/ 单任务 12B

    int num_tasks;
    // 本 rank 构建出的任务数
    EPLBTask tasks[kNumMaxTasks];
    // 任务数组

#ifdef __CUDACC__
    template <int kNumRanks, int kNumLocalExperts, int kNumRedundantExperts, int kNumThreads, int kNumWorkspaceBytes>
    __forceinline__ __device__ void build_tasks(const int* redundancy_mapping, const int& rank_idx, const int& thread_idx,
                                               void* smem_workspace_ptr) {
        // 从全局冗余映射表构建本 rank 的任务列表
        // 模板参数：rank 总数、每 rank 本地专家数、每 rank 冗余专家槽位数、块内线程数、工作区字节数
        // redundancy_mapping[i]（i = rank×冗余数 + 槽位）：该槽位应复制的原专家全局编号，-1 表示空槽
        EP_STATIC_ASSERT(kNumLocalExperts > 0, "Invalid number of local experts");
        // 编译期断言：本地专家数必须为正

        // Reuse TMA storage for the block scan
        // 复用 TMA 存储区作为 CUB BlockScan 的临时工作区
        using BlockScan = cub::BlockScan<int, kNumThreads>;
        // 块级整型前缀和类型
        EP_STATIC_ASSERT(sizeof(typename BlockScan::TempStorage) <= kNumWorkspaceBytes, "Insufficient shared-memory workspace");
        // 编译期断言：CUB 工作区大小必须不超过预留字节数
        auto& smem_workspace = *static_cast<typename BlockScan::TempStorage*>(smem_workspace_ptr);
        // 把外部传入的共享内存指针转成 BlockScan 工作区

        // Compact local expert exchanges in mapping order
        // 按映射表顺序压缩出"落在本 rank"的交换任务
        int num_found_tasks = 0;
        // 迄今为止累计发现并写入的任务数
        for (int base = 0; base < kNumRanks * kNumRedundantExperts; base += kNumThreads) {
            // 分块遍历整个映射表（每轮每线程处理一个表项）
            const int i = base + thread_idx;
            // 本线程本轮负责的表项线性编号
            const int expert_idx = i < kNumRanks * kNumRedundantExperts ? redundancy_mapping[i] : -1;
            // 读出该表项的原专家全局编号；越界填 -1
            const int is_local = expert_idx >= 0 and expert_idx / kNumLocalExperts == rank_idx;
            // 判断该原专家是否落在本 rank（全局编号整除本地专家数得到所属 rank）
            int task_idx, num_iter_found_tasks;
            // task_idx：本线程该任务在轮内的前缀编号；num_iter_found_tasks：本轮命中总数
            BlockScan(smem_workspace).ExclusiveSum(is_local, task_idx, num_iter_found_tasks);
            // 块级排他前缀和：得到本线程任务在轮内的插入位置，并统计本轮命中数
            if (is_local) {
                // 命中的线程把任务写入共享内存数组
                task_idx += num_found_tasks;
                // 加上此前轮次累计数，得到全局任务编号
                EP_DEVICE_ASSERT(task_idx < kNumMaxTasks);
                // 设备端断言：不得超过任务数组容量
                tasks[task_idx] = {
                    .local_expert_idx = expert_idx % kNumLocalExperts,
                    // 原专家在本 rank 内的本地编号（全局编号对本地专家数取模）
                    .dst_nvl_rank_idx = i / kNumRedundantExperts,
                    // 目标 NVL rank：表项编号整除每 rank 冗余槽位数
                    .redundant_expert_idx = i % kNumRedundantExperts
                    // 目标冗余槽位：表项编号对每 rank 冗余槽位数取模
                };
            }
            num_found_tasks += num_iter_found_tasks;
            // 累加本轮发现的任务数
            __syncthreads();
            // 块内同步：保证 CUB 工作区复用安全、任务写入顺序一致
        }

        // Publish task count
        // 发布任务总数
        if (ptx::get_warp_idx() == 0 and ptx::elect_one_sync())
            num_tasks = num_found_tasks;
        // 由 0 号 warp 选出的唯一线程写入任务数
        __syncthreads();
        // 块内同步：确保所有线程都能看到任务数
    }
#endif
};
EP_STATIC_ASSERT(sizeof(EPLBSharedMemoryLayout) <= EPLBSharedMemoryLayout::kNumBytes, "LB metadata exceeds shared-memory capacity");
// 编译期断言：结构体实际大小不得超过预定的 32 KiB

static constexpr int kNumMaxWeightEntries = 16;
// 权重列表最多容纳的条目数

struct EPWeight {
    // 一条权重描述：某冗余专家的权重缓冲 + 其源专家的权重缓冲
    void* redundant_expert_weights;
    // 冗余专家权重缓冲基址（本地对称内存，目标端）
    void* expert_weights;
    // 原专家权重缓冲基址（本地，源端；各专家权重首尾相接）
    int64_t num_bytes_per_expert;
    // 单个专家权重的字节数
};

struct EPWeightList {
    // 权重条目的定长列表，默认零初始化，支持下标访问
    EPWeight weights[kNumMaxWeightEntries] = {};

    __forceinline__ __device__ __host__ EPWeight& operator[](const int& weight_idx) {
        // 可变下标访问：返回指定条目的引用
        return weights[weight_idx];
    }

    __forceinline__ __device__ __host__ const EPWeight& operator[](const int& weight_idx) const {
        // 只读下标访问
        return weights[weight_idx];
    }
};

#ifdef __CUDACC__
// Claim chunks in weight, chunk, task order
// 按"权重条目 -> 块 -> 任务"的顺序认领拷贝分块（多个块并行竞争全局原子计数器）
template <int kNumWeights, int kNumBytesPerChunk>
struct EPWeightIterator {
    const comm::NCCLGin& gin;
    // Gin 句柄：用于把目标地址翻译成对端对称指针
    const EPLBSharedMemoryLayout& smem_layout;
    // 共享内存中的任务列表
    const EPWeight* weights;
    // 权重条目数组
    const int num_tasks;
    // 任务总数（每块在所有任务间并行推进）
    int* chunk_idx_counter;
    // 全局原子块计数器（位于工作区，多 SM 竞争认领）

    // States, valid after a successful `next()`
    // 状态字段：仅在 next() 返回 true 后有效
    int num_chunk_bytes;
    // 当前块的实际字节数（最末块可能不满）
    void *src_ptr, *dst_ptr;
    // 当前块的源指针与目标（对称）指针

    __forceinline__ __device__ EPWeightIterator(const comm::NCCLGin& gin, const EPLBSharedMemoryLayout& smem_layout,
                                                const EPWeight* weights,
                                                int* chunk_idx_counter):
        gin(gin), smem_layout(smem_layout), weights(weights), num_tasks(smem_layout.num_tasks),
        chunk_idx_counter(chunk_idx_counter) {
        // 初始化各引用与从共享内存读出的任务数
    }

    __forceinline__ __device__ bool next() {
        // 认领下一个待拷贝块；全部拷完时返回 false
        // Claim a chunk
        // 原子认领一个块编号（减去前面被其他调用跳过的块数，得到逻辑块号）
        int chunk_idx = atomicAdd(chunk_idx_counter, 1) - num_skipped_chunks;

        // Advance weight monotonically
        // 权重条目只前进不回退（块号在"条目内块数×任务数"空间中单调递增）
        while (weight_idx < kNumWeights) {
            const int num_chunks = static_cast<int>(math::ceil_div<int64_t>(weights[weight_idx].num_bytes_per_expert, kNumBytesPerChunk)) * num_tasks;
            // 当前权重条目占用的总块数 = ceil(单专家字节数/块大小) × 任务数
            if (chunk_idx < num_chunks)
                break;
            // 块号落在当前条目内：停止推进

            chunk_idx -= num_chunks;
            // 否则扣减该条目的块数，继续检查下一条目
            num_skipped_chunks += num_chunks;
            // 记录已跨越的物理块数（供下一次认领时修正）
            weight_idx ++;
            // 推进到下一个权重条目
        }
        if (weight_idx == kNumWeights)
            return false;
        // 所有条目都已耗尽：没有更多块可拷

        // Locate task and resolve pointers
        // 定位任务并解析出源/目标指针
        const auto& [redundant_expert_weights, expert_weights, num_bytes_per_expert] = weights[weight_idx];
        // 解构当前权重条目：目标基址、源基址、单专家字节数
        const auto& [local_expert_idx, dst_nvl_rank_idx, redundant_expert_idx] = smem_layout.tasks[chunk_idx % num_tasks];
        // 解构任务：块号对任务数取模决定"拷哪个专家到哪个槽位"
        const int64_t chunk_offset = static_cast<int64_t>(chunk_idx / num_tasks) * kNumBytesPerChunk;
        // 块号整除任务数得到"专家权重内的块序号"，乘块大小得到字节偏移
        num_chunk_bytes = static_cast<int>(min(static_cast<int64_t>(kNumBytesPerChunk), num_bytes_per_expert - chunk_offset));
        // 本块字节数：块大小与剩余字节数取较小者（最末块可能不满）
        src_ptr = math::advance_ptr(expert_weights, local_expert_idx * num_bytes_per_expert + chunk_offset);
        // 源指针：专家权重基址 + 本地专家偏移 × 单专家字节数 + 块内偏移
        dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(math::advance_ptr(redundant_expert_weights,
            redundant_expert_idx * num_bytes_per_expert + chunk_offset), dst_nvl_rank_idx);
        // 目标指针：把本地对称地址（冗余权重基址 + 槽位偏移 + 块内偏移）翻译成目标 NVL rank 的 LSA 指针
        return true;
        // 成功取到一个块
    }

private:
    int weight_idx = 0, num_skipped_chunks = 0;
    // 内部状态：当前权重条目下标、此前跳过的物理块总数
};
#endif

}  // namespace deep_ep::layout
// 结束 layout 命名空间
