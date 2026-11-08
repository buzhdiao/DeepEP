// 头文件保护:保证本头文件只被包含一次
#pragma once

// NCCL 主库头文件(设备通信相关类型)
#include <nccl.h>
// NCCL 设备端 API 头文件(设备通信句柄 ncclDevComm_t、对称内存窗口 ncclWindow_t 等)
#include <nccl_device.h>

// GPU 全局屏障(跨 rank、跨 SM 的内核级同步)实现
#include <deep_ep/comm/barrier.cuh>
// 编译期通用常量(如 kNumTMAAlignmentBytes 等)
#include <deep_ep/common/compiled.cuh>
// 静态断言与设备端断言工具
#include <deep_ep/common/exception.cuh>
// 编译期数学工具(advance_ptr 等)
#include <deep_ep/common/math.cuh>
// PTX 内联汇编封装(TMA、mbarrier、elect_one_sync 等)
#include <deep_ep/common/ptx.cuh>
// EP 工作区布局(屏障信号、LB chunk 计数器等)
#include <deep_ep/layout/ep/workspace.cuh>
// EPLB(专家负载均衡)的共享内存布局与权重迭代器
#include <deep_ep/layout/ep/eplb.cuh>


// DeepEP 的 EP(expert parallel)实现命名空间
namespace deep_ep::ep {

// EPLB 梯度归约内核:把部署在冗余(镜像)位置上的专家梯度,
// 以原子加的方式经 TMA reduce 归约回主专家所在的 rank,与 lb_prefetch_weights 对偶。
// 模板参数:
template <int kNumSMs,                    // GPU 的 SM 数量
          int kNumWarps,                  // 每个 SM 的 warp 数(每个 warp 独立处理一个 chunk 流水线)
          int kNumRanks,                  // 全局 rank 总数
          int kNumLocalExperts,           // 每个 rank 上的本地专家数量
          int kNumRedundantExperts,       // 冗余(重复部署)专家数量
          int kNumChunkBytes,             // 每个 TMA chunk 的字节数
          int64_t kNumTimeoutCycles>      // 屏障轮询的超时周期数
// 启动边界:每 SM 固定 kNumWarps*32 线程,且每 SM 至多驻留 1 个 block
__global__ void __launch_bounds__(kNumWarps * 32, 1)
lb_reduce_grads(const int* redundancy_mapping,
                const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                void* workspace,
                const __grid_constant__ layout::EPWeight weight) {
    // 下标工具:取 SM 编号与块内线程编号
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    // 取当前 warp 在 block 内的编号
    const auto warp_idx = ptx::get_warp_idx();

    // 静态检查
    // 总线程数由 warp 数推导
    constexpr int kNumThreads = kNumWarps * 32;
    // warp 数必须落在 (0, 32] 区间
    EP_STATIC_ASSERT(kNumWarps > 0 and kNumWarps <= 32, "Invalid number of warps");

    // 共享内存布局
    // 声明动态共享内存首指针,按 TMA 对齐要求对齐
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // 共享内存首段为 EPLB 任务布局(任务队列、锁等),以引用方式重解释
    auto& smem_layout = *reinterpret_cast<layout::EPLBSharedMemoryLayout*>(smem);
    // 紧随其后的是各 warp 的 TMA 缓冲区数组
    auto* tma_buffers = math::advance_ptr(smem, sizeof(smem_layout));
    // 本 warp 专属的 TMA 缓冲区(按 warp 编号偏移)
    auto* tma_buffer = math::advance_ptr(tma_buffers, warp_idx * kNumChunkBytes);
    // TMA 缓冲区之后是每 warp 一个的 mbarrier(共享内存中,用于 TMA 完成通知)
    auto* mbarrier = math::advance_ptr<ptx::mbarrier>(tma_buffers, kNumWarps * kNumChunkBytes) + warp_idx;

    // 通信句柄
    // 构造 NCCL GIN 上下文(使用 0 号 QP,不共享),提供对称内存窗口上的远程访问原语
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, 0);
    // 解析工作区布局(本内核只关心屏障信号与 LB chunk 下标计数器)
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace, 1, kNumRanks, kNumRanks * kNumLocalExperts);

    // 构建归约任务
    // 通过 LSA(节点内)团队查询本进程的 rank 编号
    const int rank_idx = ncclTeamLsa(nccl_dev_comm).rank;
    // 全体线程合作,把需要归约的冗余专家梯度任务填入共享内存任务队列,
    // tma_buffers 作为构建过程的临时工作区
    smem_layout.build_tasks<kNumRanks, kNumLocalExperts, kNumRedundantExperts, kNumThreads, kNumWarps * kNumChunkBytes>(
        redundancy_mapping, rank_idx, thread_idx, tma_buffers);

    // 初始化屏障与 chunk 计数器
    // 注意:必须在共享工作区复用之后再初始化 mbarrier,因为它们有专用缓存
    // 每个 warp 由 elected 线程把 mbarrier 初始化为期望到达次数 1(单线程提交 TMA)
    if (ptx::elect_one_sync()) {
        // 初始化本 warp 的 mbarrier 并附带内存栅栏
        ptx::mbarrier_init_with_fence(mbarrier, 1);

        // 清理任务获取计数器
        // 仅由 SM 0 的 0 号 warp 把全局 chunk 下标计数器清零,供迭代器从头分配任务
        if (sm_idx == 0 and warp_idx == 0)
            *workspace_layout.get_lb_chunk_idx_counter_ptr() = 0;
    }
    // warp 内同步,保证 mbarrier 初始化对全 warp 可见
    __syncwarp();

    // 入口屏障:等待所有 rank 的所有 SM 就绪(任务与计数器初始化完成)
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1, kNumTimeoutCycles, comm::kKernelBarrierTag, false, false, true>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);

    // 归约主循环
    // 每个 warp 只由 elected 线程驱动(单线程发射 TMA,其余 lane 空闲)
    if (ptx::elect_one_sync()) {
        // 构造权重迭代器(单权重张量,与预取内核不同):原子地领取下一个 chunk 任务
        auto iterator = layout::EPWeightIterator<1, kNumChunkBytes>(
            gin, smem_layout, &weight, workspace_layout.get_lb_chunk_idx_counter_ptr());
        // mbarrier 相位计数器,随等待次数翻转
        ptx::arrival_phase phase = 0;
        // 逐 chunk 迭代,直到任务队列耗尽
        while (iterator.next()) {
            // 加载冗余梯度
            // 设置 mbarrier 期望字节数并到达
            ptx::mbarrier_arrive_and_set_tx(mbarrier, iterator.num_chunk_bytes);
            // 发起 1D TMA 加载:冗余位置上的梯度(远端对称内存)→ 本 warp 的共享内存缓冲,
            // 使用 L2 缓存提示 kEvictNormal(该数据只会被读一次)
            ptx::tma_load_1d(tma_buffer, iterator.dst_ptr, mbarrier, iterator.num_chunk_bytes,
                             /* TODO: why this is faster? */ ptx::L2CacheHint::kEvictNormal);
            // 以相位翻转方式等待 TMA 加载完成
            ptx::mbarrier_wait_and_flip_phase(mbarrier, phase);

            // 累加并释放共享缓冲区
            // 发起 TMA reduce store:f32 逐元素加,把共享缓冲中的梯度累加到主专家所在位置
            ptx::tma_store_reduce_add_f32(static_cast<float*>(iterator.src_ptr), tma_buffer, iterator.num_chunk_bytes);
            // 提交本次 TMA store 组
            ptx::tma_store_commit();
            // 等待读侧完成(共享缓冲可安全复用),但不等待写全完成
            ptx::tma_store_wait_read<0>();
        }
    }
    // warp 内同步,等待驱动线程完成全部归约
    __syncwarp();

    // 完成屏障:确保所有 rank 的归约都已完成后才允许内核退出
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1, kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::ep
