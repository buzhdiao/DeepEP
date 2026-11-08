#pragma once
// 头文件保护：保证本文件只被包含一次

#include <cuda/atomic>
// 引入 libcu++ 原子操作（cuda::atomic_ref 等）
#include <deep_ep/comm/barrier.cuh>
// 引入通信层超时等待原语（comm::timeout_while）
#include <deep_ep/common/compiled.cuh>
// 引入编译期常量与静态断言（EP_STATIC_ASSERT 等）
#include <deep_ep/common/math.cuh>
// 引入常用数学工具（ceil_div 等）
#include <deep_ep/common/ptx.cuh>
// 引入 PTX 内联工具（warp 索引、mbarrier、内存栅栏等）
#include <deep_ep/layout/engram/workspace.cuh>
// 引入 Engram 工作区与布局定义（EngramWorkspace、EngramLayout）


namespace deep_ep::engram {
// engram 命名空间：RDMA 直读（engram fetch）相关实现

template <int kNumRDMAPeers, int kNumGinContexts, int kNumIssueWarps>
struct EngramIssueWarpLayout {
    // 描述一个"发起 warp"（issue warp）在整个线程网格中的职责划分：
    // 每个 issue warp 负责某个 (RDMA 对端, Gin 上下文) 组合下的一部分 token 分片
    int rdma_peer_idx;
    // 本 warp 负责的 RDMA 对端编号
    int gin_context_idx;
    // 本 warp 负责的 Gin 上下文（QP 组）编号
    int peer_context_idx;
    // 本 warp 在 (对端, 上下文) 组合空间中的扁平编号，用于定位工作区计数器
    int num_peer_context_warps;
    // 分到同一 (对端, 上下文) 组合的 issue warp 总数
    int peer_warp_idx;
    // 本 warp 在负责同一 RDMA 对端的所有 warp 中的编号
    int num_peer_warps;
    // 分到同一 RDMA 对端的 issue warp 总数

    __device__ __forceinline__ explicit EngramIssueWarpLayout(const int issue_warp_idx) {
        // 由 issue warp 的全局扁平编号构造各维度职责划分
        constexpr int kNumPeerContexts = kNumRDMAPeers * kNumGinContexts;
        // (对端, 上下文) 组合总数

        peer_context_idx = issue_warp_idx % kNumPeerContexts;
        // 对组合总数取模：得到本 warp 的扁平组合编号
        rdma_peer_idx = peer_context_idx % kNumRDMAPeers;
        // 再对对端数取模：得到 RDMA 对端编号
        gin_context_idx = peer_context_idx / kNumRDMAPeers;
        // 整除对端数：得到 Gin 上下文编号
        num_peer_context_warps = math::ceil_div(kNumIssueWarps - peer_context_idx, kNumPeerContexts);
        // 计算分到本组合的 warp 数：从本 warp 的扁平编号开始，每隔 kNumPeerContexts 取一个，向上取整

        peer_warp_idx = issue_warp_idx / kNumRDMAPeers;
        // 同一对端内 warp 的编号（相邻编号的 warp 服务不同对端的同位置）
        num_peer_warps = math::ceil_div(kNumIssueWarps - rdma_peer_idx, kNumRDMAPeers);
        // 分到本对端的 warp 总数
    }

    __device__ __forceinline__ int get_num_active_peer_context_warps(const int num_token_tiles) const {
        // 计算本 (对端, 上下文) 组合下真正有 token 要处理的活跃 warp 数：
        // 取"上下文内 token 分片数"与"本组合 warp 数"中的较小者
        const auto num_token_warps = gin_context_idx < num_token_tiles
            ? math::ceil_div(num_token_tiles - gin_context_idx, kNumGinContexts) : 0;
        // 本上下文覆盖的 token 分片数：从本上下文的起始分片起每隔 kNumGinContexts 取一个；不足则为 0
        return num_token_warps < num_peer_context_warps ? num_token_warps : num_peer_context_warps;
        // 活跃 warp 数 = min(token 分片覆盖的 warp 数, 本组合的 warp 总数)
    }
};

template <int kNumSMs, int kNumQPs,
          int kNumSFWarpsPerSM, int kNumIssueWarpsPerSM,
          typename engram_layout_t,
          int kNumSFPacks,
          int kNumMaxTokens,
          int kNumRDMAPeers,
          int kNumRanksPerRDMAPeer,
          int kFlushDepth,
          int64_t kNumTimeoutCycles,
          int kNumSFWarps = kNumSMs * kNumSFWarpsPerSM,
          int kNumIssueWarps = kNumSMs * kNumIssueWarpsPerSM,
          int kNumThreads = (kNumSFWarpsPerSM + kNumIssueWarpsPerSM) * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_impl(
    // Engram 取回内核：按每层的路由索引（indices），从各 RDMA 对端的存储分片中
    // 把本 rank 需要的 hidden 条目异步 GET 到本地接收缓冲，并顺带取回 scale factor（SF）分片
    const ncclDevComm_t nccl_dev_comm,
    // 设备端 NCCL 通信对象
    const ncclWindow_t nccl_window,
    // 本 rank 在对称内存中的窗口句柄
    const ncclWindow_t* storage_windows,
    // 各对端 rank 的 CPU/GPU RDMA 存储分片窗口数组
    void* fetched,
    // 本地接收缓冲区（按 EngramLayout 的 recv 布局存放各层 hidden 条目）
    void* workspace_ptr,
    // 工作区指针（EngramWorkspace：层间发起屏障计数器等）
    int* indices,
    // 路由结果：indices[layer][token][entry] = 条目在全局中的编号（-1 表示无效）
    ncclGinRequest_t* last_gin_requests,
    // 每个 (层, 对端上下文) 的最后一次 Gin 请求记录，供等待内核使用
    const sf_pack_t* sf_shards,
    // 各对端持有的 SF（scale factor）分片（对称内存）
    sf_pack_t* fetched_sf,
    // 本地取回后的 SF 输出缓冲
    const int sf_token_stride,
    // SF 输出中同一 token 相邻条目之间的步长（单位：sf_pack_t）
    const int sf_hidden_stride,
    // SF 输出中同一 hidden 通道相邻打包之间的步长
    const int sf_layer_stride,
    // SF 输出中层与层之间的步长
    const int num_tokens
    // 本批 token 数
) {
    constexpr int kWarpSize = 32;
    // CUDA warp 大小
    constexpr int kNumLayers = engram_layout_t::kNumLayers;
    // 层数（由布局模板参数包推导）
    constexpr int kNumEntriesPerToken = engram_layout_t::kNumEntriesPerToken;
    // 每 token 的条目数（top-k 数量）

    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 SM（块）编号
    const auto warp_idx = ptx::get_warp_idx();
    // 块内 warp 编号
    const auto lane_idx = ptx::get_lane_idx();
    // warp 内 lane 编号

    // Two warp-sized halves form one circular pending-request buffer per issue warp.
    // 每个 issue warp 有一个环形"待发起请求"缓冲：由两个 warp 大小（2×32）的半区组成
    __shared__ int64_t pending_requests[kNumIssueWarpsPerSM][2 * kWarpSize];
    // 共享内存中的待请求缓冲：64 位项 = (全局索引 << 32) | 条目索引

    if (warp_idx < kNumIssueWarpsPerSM) {
        // 前半部分 warp 作为 issue warp：负责按对端/上下文划分发起 RDMA GET
        constexpr int kNumPeerContexts = kNumRDMAPeers * kNumQPs;
        // (对端, QP) 组合总数
        const auto issue_warp_idx = sm_idx * kNumIssueWarpsPerSM + warp_idx;
        // 本 issue warp 在全部 issue warp 中的全局编号
        const auto num_token_tiles = math::ceil_div(num_tokens, kWarpSize);
        // token 分片（tile）数：每片覆盖 32 个 token
        const auto issue_layout = EngramIssueWarpLayout<
            kNumRDMAPeers, kNumQPs, kNumIssueWarps>(issue_warp_idx);
        // 由全局编号解析出本 warp 的 (对端, 上下文) 职责
        if (issue_layout.peer_warp_idx >= num_token_tiles)
            return;
        // 若本 warp 在对端内的编号超出 token 分片数，则无事可做，直接退出

        const auto num_active_peer_context_warps =
            issue_layout.get_num_active_peer_context_warps(num_token_tiles);
        // 本 (对端, 上下文) 组合下活跃的 issue warp 数（用于层间屏障）
        const auto rdma_peer_rank = issue_layout.rdma_peer_idx * kNumRanksPerRDMAPeer;
        // 本对端在 NCCL world 中的起始 rank 编号（一个 RDMA 对端可能聚合多个 rank）
        const auto doorbell_threshold = kFlushDepth >= num_active_peer_context_warps
            ? kFlushDepth / num_active_peer_context_warps : 1;
        // 门铃阈值：把 flush 深度摊到各活跃 warp 上，攒够这么多个请求才触发一次门铃
        const auto gin = comm::NCCLGin(
            nccl_dev_comm, nccl_window, issue_layout.gin_context_idx, NCCL_GIN_RESOURCE_SHARING_GPU);
        // 构造本 warp 的 Gin 句柄；资源按 GPU 共享（NCCL_GIN_RESOURCE_SHARING_GPU）
        auto& workspace = *static_cast<layout::EngramWorkspace*>(workspace_ptr);
        // 引用工作区（层间发起屏障计数器）
        auto* warp_pending_requests = pending_requests[warp_idx];
        // 本 warp 专属的待请求缓冲片段

        const auto num_entries = num_tokens * kNumEntriesPerToken;
        // 本批有效条目总数
        #pragma unroll
        for (int layer_idx = 0; layer_idx < kNumLayers; ++ layer_idx) {
            // 逐层处理：每层的路由索引不同，需分别发起请求
            int num_issued_requests = 0;
            // 本层累计已发起的请求数（用于门铃计数）
            int num_pending_requests = 0;
            // 环形缓冲中尚未发起的请求数
            int pending_head = 0;
            // 环形缓冲当前读取（发起）方向的半区起点（0 或 kWarpSize）
            const auto num_entries_per_rank = engram_layout_t::get_num_entries(layer_idx);
            // 本层每 rank 的存储条目数（存储分片中该层的长度）
            const auto peer_global_idx_begin = issue_layout.rdma_peer_idx * kNumRanksPerRDMAPeer * num_entries_per_rank;
            // 本对端负责的全局条目编号区间起点
            const auto peer_global_idx_end = (issue_layout.rdma_peer_idx + 1) * kNumRanksPerRDMAPeer * num_entries_per_rank;
            // 本对端负责的全局条目编号区间终点（不含）

            const auto issue_batch = [&](const int batch_size, const bool is_last_batch = false) {
                // 内联 lambda：把环形缓冲中的 batch_size 个请求一次性批量发出
                const auto rings_doorbell = is_last_batch or num_issued_requests % doorbell_threshold + batch_size >= doorbell_threshold;
                // 判断本批是否触发门铃：最后一批必触发；否则看累计计数是否跨过门铃阈值
                num_issued_requests += batch_size;
                // 累计已发起请求数
                const auto options = rings_doorbell ? ncclGinOptFlagsDefault : ncclGinOptFlagsAggregateRequests;
                // 触发门铃的批次用默认标志；否则用"聚合请求"标志以推迟通知、减少门铃次数

                if (lane_idx < batch_size) {
                    // 前 batch_size 个 lane 各负责发出一个请求（一个 lane 一条 GET）
                    const auto request = warp_pending_requests[pending_head + lane_idx];
                    // 从环形缓冲读取本 lane 的 64 位请求项
                    const auto entry_idx = static_cast<int>(request);
                    // 低 32 位：条目在接收缓冲中的（本地）条目编号
                    const auto global_idx = static_cast<int>(request >> 32);
                    // 高 32 位：条目的全局编号（决定源 rank 与源位置）
                    const auto peer_local_idx = global_idx - peer_global_idx_begin;
                    // 全局编号相对本对端起始的局部偏移
                    const auto intra_peer_rank_idx = peer_local_idx / num_entries_per_rank;
                    // 该条目位于对端聚合的哪个 rank 上（对端内 rank 序号）
                    const auto local_entry_idx = peer_local_idx % num_entries_per_rank;
                    // 该条目在该 rank 存储分片内的条目序号
                    const auto src_byte_offset = engram_layout_t::get_storage_byte_offset(layer_idx, local_entry_idx);
                    // 计算源端存储分片内的字节偏移（各层条目首尾相接）
                    const auto dst_byte_offset = engram_layout_t::get_recv_byte_offset(
                        layer_idx, entry_idx, kNumMaxTokens);
                    // 计算本地接收缓冲内的字节偏移（按层×每 token 条目×最大 token 数布局）
                    gin.get<ncclTeamTagWorld, ncclCoopThread, ncclGin_SegmentMixed>(
                        storage_windows[intra_peer_rank_idx], src_byte_offset,
                        nccl_window, gin.get_sym_offset(math::advance_ptr(fetched, dst_byte_offset)),
                        engram_layout_t::kNumHiddenBytes,
                        rdma_peer_rank,
                        options);
                    // 发起 RDMA GET：从对端 rank 的存储窗口读 kNumHiddenBytes 字节到本地对称内存偏移处
                    // ncclGin_SegmentMixed：源地址按窗口混合寻址；rdma_peer_rank 指定目标 rank
                }
            };

            for (int token_tile_idx = issue_layout.peer_warp_idx;
                 token_tile_idx < num_token_tiles;
                 token_tile_idx += issue_layout.num_peer_warps) {
                // 本 warp 以"对端内 warp 数"为步长轮流认领 token 分片
                const auto first_entry_idx = token_tile_idx * kWarpSize * kNumEntriesPerToken;
                // 本分片第一个条目的全局条目序号（每 token 有 kNumEntriesPerToken 个条目）

                for (int round_idx = 0; round_idx < kNumEntriesPerToken; ++round_idx) {
                    // 每轮处理一种 entry 槽位：32 个 lane 恰好覆盖 32 个 token 的同一槽位
                    const auto entry_idx = first_entry_idx + round_idx * kWarpSize + lane_idx;
                    // 本 lane 负责的条目在路由数组中的线性编号
                    const auto global_idx = entry_idx < num_entries
                        ? __ldg(indices + layer_idx * num_entries + entry_idx)
                        : -1;
                    // 只读缓存读取全局编号；越界部分填 -1 表示无效
                    const auto belongs_to_peer =
                        global_idx >= peer_global_idx_begin && global_idx < peer_global_idx_end;
                    // 判断该条目是否属于本对端负责的编号区间
                    const auto peer_mask = __ballot_sync(0xffffffffu, belongs_to_peer);
                    // warp 内投票：得到哪些 lane 命中本对端的位掩码

                    if (belongs_to_peer) {
                        // 命中的 lane 把请求写入环形缓冲
                        const auto lower_lanes_mask = (1u << lane_idx) - 1;
                        // 低于本 lane 的所有 lane 的掩码
                        const auto pending_idx =
                            (pending_head + num_pending_requests +
                             __popc(peer_mask & lower_lanes_mask)) &
                            (2 * kWarpSize - 1);
                        // 本 lane 在环形缓冲中的写入位置：当前队尾 + 低 lane 命中个数（前缀和），对 64 取模
                        warp_pending_requests[pending_idx] =
                            (static_cast<int64_t>(global_idx) << 32) | entry_idx;
                        // 打包请求项：高 32 位全局编号、低 32 位本地条目编号
                    }
                    __syncwarp();
                    // warp 内同步：保证写入完成后再更新队尾计数
                    num_pending_requests += __popc(peer_mask);
                    // 队尾前进：本轮命中本对端的 lane 数

                    // Keep the final batch to ring the doorbell before flush_async
                    // 保留最后一批请求不发起，以便在 flush_async 前触发门铃
                    if (num_pending_requests > kWarpSize) {
                        // 超过半个环形缓冲（32 条）就先批量发起一批
                        issue_batch(kWarpSize);
                        // 一次性发出 32 条请求
                        pending_head ^= kWarpSize;
                        // 读半区在两个半区间翻转（0<->32）
                        num_pending_requests -= kWarpSize;
                        // 队列长度减去已发出的一批
                    }
                }
            }

            if (num_pending_requests > 0) {
                // 本层结束时若仍有剩余请求，作为最后一批发出
                issue_batch(num_pending_requests, true);
                // 最后一批：无论门铃阈值如何都触发门铃
                __syncwarp();
                // warp 内同步，确保所有 lane 发起完成后再进入屏障阶段
            }

            if (ptx::elect_one_sync()) {
                // 每 warp 选出一个代表线程执行屏障/flush 逻辑
                auto* issue_counter = workspace.get_issue_counter_ptr(
                    layer_idx, issue_layout.peer_context_idx);
                // 取本 (层, 对端上下文) 的到达计数器地址
                const auto request_ptr =
                    last_gin_requests + layer_idx * kNumPeerContexts + issue_layout.peer_context_idx;
                // 本上下文最后一次请求记录的存放位置
                if (num_active_peer_context_warps == 1) {
                    gin.flush_async<ncclTeamTagWorld, ncclCoopThread>(rdma_peer_rank, request_ptr);
                    // 只有一个活跃 warp 时无需协调，直接 flush 该对端 QP 并记录最后请求
                } else {
                    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> num_arrivals(*issue_counter);
                    // 用设备范围的原子引用包装到达计数器

                    // Publish this issue warp's last_issued_get update. The last warp
                    // alone acquires all earlier arrivals before flushing the QP.
                    // 发布本 warp 最后一条 GET 的更新；由最后一个到达的 warp 先获取
                    // （acquire）之前所有 warp 的发布，再统一 flush QP
                    const auto arrival_idx = num_arrivals.fetch_add(1, cuda::memory_order_release);
                    // 以 release 语义原子递增到达计数，返回到达序号
                    if (arrival_idx + 1 == num_active_peer_context_warps) {
                        // 本 warp 是最后一个到达者
                        ptx::fence_acquire_gpu();
                        // GPU 范围 acquire 栅栏：看到其他 warp 的最后请求记录更新
                        gin.flush_async<ncclTeamTagWorld, ncclCoopThread>(
                            rdma_peer_rank, request_ptr);
                        // 统一 flush 该对端 QP，记录最后请求
                        ptx::st_relaxed_gpu(issue_counter, 0u);
                        // 以 relaxed 语义把计数器清零，供下一层复用
                    } else if (layer_idx + 1 < kNumLayers) {
                        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                            // 非最后到达且还有后续层：等待计数器被最后一个到达者清零
                            if (ptx::ld_volatile<uint32_t>(issue_counter) == 0)
                                return true;
                            // 计数器已清零说明 flush 已完成，可以进入下一层
                            if (is_last_check) {
                                printf("DeepEP Engram issue barrier timeout, peer: %d, "
                                       "gin context: %d, layer: %d\n",
                                       issue_layout.rdma_peer_idx, issue_layout.gin_context_idx, layer_idx);
                                // 最后一次检查仍超时：打印对端、上下文与层号便于排查
                            }
                            return false;
                            // 继续等待
                        });
                    }
                }
            }
            __syncwarp();
            // warp 内同步：保证本层收尾逻辑对所有 lane 可见后再进入下一层
        }
    } else if constexpr (kNumSFPacks > 0) {
        // 后半部分 warp 作为 SF（scale factor）warp：直接通过对称内存读取 SF 分片
        constexpr int kNumPacksPerVec = sizeof(uint64_t) / sizeof(sf_pack_t);
        // 一个 64 位向量能装下多少个 sf_pack
        constexpr int kNumVecs = kNumSFPacks / kNumPacksPerVec;
        // 每个条目的 SF 需要多少个 64 位向量
        EP_STATIC_ASSERT(kNumSFPacks % kNumPacksPerVec == 0, "Invalid number of SF packs");
        // 编译期断言：SF 打包数必须能被向量大小整除

        const auto sf_warp_idx = sm_idx * kNumSFWarpsPerSM + warp_idx - kNumIssueWarpsPerSM;
        // 本 SF warp 在全部 SF warp 中的全局编号
        const auto num_entries = num_tokens * kNumEntriesPerToken;
        // 本批有效条目总数
        const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window);
        // 构造默认上下文的 Gin 句柄（仅用于把对称地址转成对端可寻址指针）

        const auto copy_sf_entry = [&](const int layer_idx, const int entry_idx) {
            // 内联 lambda：把一个条目（token×槽位）的 SF 数据从对端分片拷到本地
            const auto global_idx = __ldg(indices + layer_idx * num_entries + entry_idx);
            // 读取该条目的全局编号
            if (global_idx < 0)
                return;
            // 无效条目直接跳过

            const auto sf_row_idx = engram_layout_t::get_sf_row_idx(layer_idx, global_idx);
            // 计算 SF 在分片内的行号（按所有者 rank 分段）
            const auto sf_shard_idx = engram_layout_t::get_sf_shard_idx(layer_idx, global_idx);
            // 计算该条目的 SF 归属哪个分片（owner rank）
            const auto* sf_shard = gin.get_sym_ptr<ncclTeamTagLsa>(const_cast<sf_pack_t*>(sf_shards), sf_shard_idx);
            // 把 SF 分片基址翻译成第 sf_shard_idx 个 LSA 域内 rank 的对称指针
            const auto* src = reinterpret_cast<const uint64_t*>(sf_shard + sf_row_idx * kNumSFPacks);
            // 指向该条目 SF 数据起始（按 64 位向量读取）

            uint64_t values[kNumVecs];
            // 暂存本条目的 SF 向量
            #pragma unroll
            for (int i = 0; i < kNumVecs; ++ i)
                values[i] = __ldg(src + i);
            // 只读缓存逐向量加载

            auto* dst = fetched_sf + layer_idx * sf_layer_stride +
                (entry_idx / kNumEntriesPerToken) * sf_token_stride +
                (entry_idx % kNumEntriesPerToken) * kNumSFPacks * sf_hidden_stride;
            // 计算输出位置：层步长 + token 步长 + 槽位×打包数×hidden 步长
            #pragma unroll
            for (int i = 0; i < kNumVecs; ++ i) {
                #pragma unroll
                for (int j = 0; j < kNumPacksPerVec; ++ j)
                    dst[(i * kNumPacksPerVec + j) * sf_hidden_stride] = reinterpret_cast<const sf_pack_t*>(&values[i])[j];
                // 把每个 64 位向量拆成若干 sf_pack，按 hidden 步长散写到输出
            }
        };

        #pragma unroll
        for (int layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
            // 逐层拷贝 SF
            const auto num_token_tiles = math::ceil_div(num_tokens, kWarpSize);
            // token 分片数
            for (int i = sf_warp_idx; i < kNumEntriesPerToken * num_token_tiles; i += kNumSFWarps) {
                // 把 (槽位×分片) 的二维任务空间按 SF warp 总数轮流切分
                const auto entry_slot_idx = i / num_token_tiles;
                // 反解出槽位编号（0..kNumEntriesPerToken-1）
                const auto token_tile_idx = i % num_token_tiles;
                // 反解出 token 分片编号
                const auto token_idx = token_tile_idx * kWarpSize + lane_idx;
                // 本 lane 负责的 token 编号
                if (token_idx < num_tokens)
                    copy_sf_entry(layer_idx, token_idx * kNumEntriesPerToken + entry_slot_idx);
                // 越界 token 跳过；否则拷贝该 (token, 槽位) 的 SF
            }
        }
    }
}

}  // namespace deep_ep::engram
// 结束 engram 命名空间
