#pragma once
// 头文件保护：保证本文件在同一个编译单元中只被包含一次

#include <deep_ep/comm/handle.cuh>
// 引入 NCCL Gin 通信句柄封装（NCCLGin）
#include <deep_ep/common/compiled.cuh>
// 引入编译期常量与静态断言宏（EP_STATIC_ASSERT 等）


namespace deep_ep::engram {
// engram 命名空间：RDMA 直读（engram fetch）相关实现

template <int kNumRDMAPeers, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
// 模板参数：kNumRDMAPeers 为 RDMA 对端数；kNumThreads 为每块线程数
// __launch_bounds__ 限制每块最多 kNumThreads 线程且每 SM 至少驻留 1 个块
engram_fetch_wait_impl(const ncclDevComm_t nccl_dev_comm,
                       const ncclWindow_t nccl_window,
                       ncclGinRequest_t* last_gin_requests,
                       const int num_active_gin_contexts) {
    // 等待 engram_fetch_impl 发出的所有 RDMA GET 请求完成：
    // 逐个 Gin 上下文读取"最后一次请求"记录并在其上 wait，从而确认数据已全部到达
    // 参数：nccl_dev_comm 设备端 NCCL 通信对象；nccl_window 对称内存窗口
    //      last_gin_requests 每个对端上下文的最后请求记录；num_active_gin_contexts 活跃 Gin 上下文数
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // 取当前块内线程编号
    EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
    // 编译期断言：ncclGinRequest_t 必须恰好是 16 字节（int4），便于按向量加载
    const auto num_active_peer_contexts = num_active_gin_contexts * kNumRDMAPeers;
    // 活跃上下文总数 = 活跃 Gin 上下文数 × RDMA 对端数（每个对端每个上下文各有一条待等待的请求）

    for (int peer_context_idx = thread_idx; peer_context_idx < num_active_peer_contexts; peer_context_idx += kNumThreads) {
        // 网格步长循环：每个线程按 kNumThreads 为步长认领若干 (对端, 上下文) 组合
        const auto context_idx = peer_context_idx / kNumRDMAPeers;
        // 由一维编号反解出 Gin 上下文编号（同上下文的线程相邻，利于缓存）
        const auto gin = comm::NCCLGin(
            nccl_dev_comm, nccl_window, context_idx, NCCL_GIN_RESOURCE_SHARING_THREAD);
        // 构造该上下文的 Gin 句柄；资源按线程共享（NCCL_GIN_RESOURCE_SHARING_THREAD）

        auto last_gin_req_int4 = __ldg(reinterpret_cast<int4*>(last_gin_requests + peer_context_idx));
        // 只读缓存加载 16 字节的最后请求记录（以 int4 向量化读取）
        auto last_gin_req = *reinterpret_cast<ncclGinRequest_t*>(&last_gin_req_int4);
        // 将 int4 位模式重新解释回 ncclGinRequest_t 类型
        gin.wait(last_gin_req, cuda::memory_order_relaxed);
        // 在最后一条请求上阻塞等待其完成；完成即代表该上下文本层的所有 GET 都已写入
        // 使用 relaxed 内存序：此处只关心传输完成本身，无需额外同步语义
    }
}

}  // namespace deep_ep::engram
// 结束 engram 命名空间
