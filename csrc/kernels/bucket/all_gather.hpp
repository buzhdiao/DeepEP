#pragma once // 头文件保护：只被包含一次

#include <algorithm> // std::min
#include <cmath> // std::pow 等数学函数
#include <cstdint> // 定宽整数类型
#include <format> // std::format：拼接 JIT 编译模板
#include <limits> // std::numeric_limits
#include <optional> // std::optional
#include <vector> // std::vector

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <nccl.h> // NCCL 主机端 API

#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（EP_HOST_ASSERT 等）
#include <deep_ep/common/math.cuh> // 数学工具（ceil_div、align、advance_ptr 等）
#include <deep_ep/layout/bucket/workspace.cuh> // bucket 通信 workspace 布局（AllGatherSignals 等）

#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架
#include "../comm/api.hpp" // comm::Context：通信上下文
#include "../driver/api.hpp" // driver：拷贝引擎（CE）push/sync 接口

namespace deep_ep::bucket { // bucket 命名空间：bucket 级集合通信 kernel 封装

// 启动 bucket 级 all-gather：把各 rank 的分片聚齐，自动选择纯 RDMA/纯 NVLink（CE 直推）/混合（RDMA 收 + NVLink 转发）三条路径
static void launch_all_gather(
    const comm::Context& context, // 通信上下文
    const int& num_buckets, const comm::BucketList& buckets, // bucket 数量与列表（每 bucket 的偏移与字节数）
    const std::optional<comm::BucketList>& src_buckets, // 可选的源 bucket 列表（输入不在标准缓冲区布局时使用）
    const int& context_idx, // 本上下文在 workspace 信号量数组中的索引
    uint64_t& seq_idx, // 拷贝引擎同步用的序列号（跨调用递增）
    const float& nvl_bandwidth, const float& rdma_bandwidth, // 机内 NVLink 带宽与跨机 RDMA 带宽（用于估算最优 chunk 数）
    const at::cuda::CUDAStream& stream) { // 启动 kernel 的 CUDA 流
    EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets); // bucket 数不得超过 kNumMaxBuckets 上限
    EP_HOST_ASSERT((context.num_rdma_ranks == 1 or not src_buckets.has_value()) and // 缓冲区外输入只支持纯 NVLink 路径
                   "Out-of-buffer inputs are only supported by the pure NVLink path");

    if (context.num_rdma_ranks > 1 and context.num_nvl_ranks == 1) { // 分支一：纯 RDMA（机内只有 1 个 NVLink rank）
        // Pure RDMA // 纯 RDMA 路径
        constexpr int kNumThreads = 512; // 每 block 512 线程
        const auto num_sms = jit->device.get_num_sms(); // 占满全部 SM

        // Compile // 编译：实例化 RDMA all-gather 模板（模板实参：RDMA 域大小、SM 数、超时、上下文索引）
        const auto kernel = jit->compile("bucket_rdma_all_gather", std::format(R"(
#include <deep_ep/impls/bucket/all_gather/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_all_gather<{}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, num_sms, context.num_gpu_timeout_cycles, context_idx));
 // 协作式启动：grid=全部 SM，每 block 512 线程
        // Launch
        jit->launch( // 提交 kernel 启动
            kernel, { // 启动配置
                .stream = stream, // 目标 CUDA 流
                .grid_dim = dim3(num_sms, 1, 1), // grid 维度：num_sms 个 block
                .block_dim = dim3(kNumThreads, 1, 1), // block 维度：512 线程
                .cooperative = true, // 协作启动：支持 grid 内全局同步
            },
            context.dev_comm, context.window, // 设备通信器与对称内存窗口
            context.workspace, context.buffer, // workspace（信号量）与数据缓冲区
            num_buckets, buckets // bucket 数与 bucket 列表
        );
    } else if (context.num_rdma_ranks == 1) { // 分支二：纯 NVLink（无跨机 RDMA）
        // Pure NVLink // 纯 NVLink 路径：不用 kernel，直接用拷贝引擎（CE）环形推送分片
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context.workspace); // 取 workspace 引用（含各上下文的 CE 信号量）
        auto& signals = workspace.signals[context_idx].copy_engine; // 取本上下文的 copy-engine 信号量组
        EP_HOST_ASSERT(context.num_nvl_ranks <= kNumMaxNVLRanks and // NVLink rank 数不得超过 workspace 信号量容量
                       "Number of NVLink ranks exceeds the maximum workspace size");

        // Build copy descriptors // 构建拷贝描述符：为每个对端与每个 bucket 生成一条 CE 拷贝
        std::vector<size_t> sizes; // 各条拷贝的字节数
        std::vector<void*> dst_ptrs, src_ptrs; // 各条拷贝的目的与源地址
        std::vector<uint64_t*> write_signals, wait_signals; // 写信号与等待信号的地址
        write_signals.reserve(context.num_nvl_ranks - 1); // 预留写信号容量（除本地外的每个对端一个）
        wait_signals.reserve(context.num_nvl_ranks - 1); // 预留等待信号容量
        for (int rank_offset = 1; rank_offset < context.num_nvl_ranks; ++ rank_offset) { // 按环形顺序遍历所有 NVLink 对端
            // Wait/write signals are same for prologue and epilogue // 前同步（等上一轮）与后同步（写本轮）使用相同的一组信号
            const auto peer_rank_idx = (context.nvl_rank_idx + rank_offset) % context.num_nvl_ranks; // 对端 rank 编号：从本 rank 出发环形偏移
            write_signals.push_back( // 写信号：对端缓冲区中属于本 rank 的槽位（经对称地址）
                static_cast<uint64_t*>(context.get_sym_ptr(signals.base, peer_rank_idx)) + context.nvl_rank_idx); // 指向对端信号数组中本 rank 的槽位
            wait_signals.push_back(signals.base + peer_rank_idx); // 等待信号：本地信号数组中对端的槽位

            // Create copies // 为每个 bucket 生成拷贝：本 rank 的分片推到对端
            for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) { // 遍历每个 bucket
                const auto& [offset, num_bytes] = buckets[bucket_idx]; // 取该 bucket 的偏移与字节数
                const auto shard_offset = offset + context.nvl_rank_idx * num_bytes; // 本 rank 分片在该 bucket 中的偏移
                const auto src_offset = src_buckets.has_value() ? (*src_buckets)[bucket_idx].offset : shard_offset; // 源偏移：缓冲区外输入用源 bucket 布局，否则与本分片相同
                dst_ptrs.push_back(context.get_sym_ptr(math::advance_ptr(context.buffer, shard_offset), peer_rank_idx)); // 目的地址：对端对称缓冲区中的对应分片位置
                src_ptrs.push_back(math::advance_ptr(context.buffer, src_offset)); // 源地址：本地缓冲区中的分片
                sizes.push_back(num_bytes); // 拷贝字节数为该 bucket 的大小
            }
        }

        // Out-of-buffer inputs also need a local copy
        for (int bucket_idx = 0; src_buckets.has_value() and bucket_idx < num_buckets; ++ bucket_idx) { // 缓冲区外输入还需要一次本地拷贝（搬到标准布局位置）
            const auto& [offset, num_bytes] = buckets[bucket_idx]; // 遍历每个 bucket
            const auto shard_offset = offset + context.nvl_rank_idx * num_bytes; // 取该 bucket 的偏移与字节数
            if ((*src_buckets)[bucket_idx].offset == shard_offset) // 目标分片偏移
                continue; // 源已在标准布局则跳过

            dst_ptrs.push_back(math::advance_ptr(context.buffer, shard_offset)); // 目的地址：标准布局中的分片位置
            src_ptrs.push_back(math::advance_ptr(context.buffer, (*src_buckets)[bucket_idx].offset)); // 源地址：缓冲区外的原始输入位置
            sizes.push_back(num_bytes); // 拷贝字节数
        }

        // Wait arrival & push & write // 三步式同步：等对端就绪 -> 推送数据 -> 写信号通知对端
        driver::copy_engine_sync(stream, driver::increase_seq_idx(stream, seq_idx), write_signals, wait_signals); // 第一步：递增序列号，等待各对端信号（同时写本地信号）
        driver::copy_engine_push(stream, dst_ptrs, src_ptrs, sizes); // 第二步：批量提交所有 CE 拷贝
        driver::copy_engine_sync(stream, driver::increase_seq_idx(stream, seq_idx), write_signals, wait_signals); // 第三步：再次递增序列号并写信号，通知对端数据已送达
    } else if (context.num_rdma_ranks > 1 and context.num_nvl_ranks > 1) { // 分支三：混合模式（跨机 RDMA + 机内 NVLink 都大于 1）
        // Hybrid: one SM issues RDMA to one peer with `kNumQPsPerChunk` QPs, the CE forwards arrived chunks over NVLink // 混合：一个 SM 经多个 QP 向一个 RDMA 对端收数据，拷贝引擎再把到齐的 chunk 经 NVLink 转发
        constexpr int kNumThreads = 256; // 每 block 256 线程
        constexpr int kNumQPsPerChunk = layout::AllGatherSignals::kNumQPsPerChunk; // 每个 chunk 使用的 QP 数（与 workspace 信号量布局一致）
        EP_HOST_ASSERT(context.num_rdma_ranks <= jit->device.get_num_sms() and "Insufficient SMs for RDMA peers"); // RDMA 对端数不得超过 SM 数（每个对端占一个 SM）
        EP_HOST_ASSERT(kNumQPsPerChunk <= context.num_allocated_qps and "Insufficient QPs"); // QP 数量必须足够

        int64_t num_shard_bytes = 0; // 累计所有 bucket 的本 rank 分片字节数
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) // 遍历每个 bucket
            num_shard_bytes += buckets[bucket_idx].num_bytes; // 累加该 bucket 的分片字节数

        // Choose the number of chunks by estimating the single-stream CE timeline: the CE pushes the local shard, then
        // forwards each chunk wave (one chunk from every RDMA peer) once it lands, paying a launch gap per wave.
        // The chunk is floored so that each QP puts at least 512 KiB, where the NIC efficiency is stable
        constexpr int kNumMinChunkBytes = 4 * 1024 * 1024; // 每 chunk 最小 4 MiB：保证每个 QP 至少承载 512 KiB，NIC 效率才稳定
        const auto push_t = num_shard_bytes * (context.num_nvl_ranks - 1) / nvl_bandwidth; // 本地分片经 NVLink 推送的时间（不含转发）
        const auto rdma_t = num_shard_bytes * (context.num_rdma_ranks - 1) / rdma_bandwidth; // 所有 RDMA 接收的总时间（不含分 chunk 并行收益）
        const auto forward_t = push_t * (context.num_rdma_ranks - 1); // CE 转发总时间：每波转发都要等本地推送完成
        int num_best_chunks = 1; // 最优 chunk 数（枚举上限 64）
        float best_t = std::numeric_limits<float>::infinity(); // 当前最优完成时间
        for (int i = 1; i <= std::min<int64_t>(64, num_shard_bytes / kNumMinChunkBytes); ++ i) { // 枚举 chunk 数 i，估算每种切分下的总完成时间
            // CE-bound: forward all waves after the local push or the first wave; RDMA-bound: forward the last wave
            // NOTES: `10e-6` is CE overhead
            const auto t = std::max(std::max(push_t, rdma_t / i) + forward_t + i * (10e-6f), rdma_t + forward_t / i); // CE 受限：本地推送或第一波之后串行转发所有波（每次波间有 CE 启动开销 10e-6）；RDMA 受限：转发最后一波
            if (t < best_t) // 若该切分更优则记录
                best_t = t, num_best_chunks = i; // 更新最优时间与最优 chunk 数
        }
        // Calculate in 64 bits, then bound each chunk before narrowing its byte count. // 先用 64 位计算再收敛，避免中间溢出；字节数按 QP 数与 RDMA 对齐要求向上对齐
        const auto num_chunk_bytes = static_cast<int>(std::clamp<int64_t>( // 把每 chunk 字节数夹在 [4 KiB, ChunkStorage 容量] 之间
            math::align<int64_t>( // 按 kNumQPsPerChunk × RDMA 对齐字节数向上对齐
                math::ceil_div<int64_t>(num_shard_bytes, num_best_chunks), // 先算出平均每 chunk 的字节数（向上取整）
                kNumQPsPerChunk * kNumRDMAAlignmentBytes), // 下限 4 KiB、上限为 ChunkStorage 单槽容量
            4 * 1024, layout::ChunkStorage::kNumBytes
        ));

        // One completion signal per chunk on every QP // 每个 chunk 在每条 QP 上需要一个完成信号
        int num_chunks = 0; // 统计 chunk 总数
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) // 遍历每个 bucket
            num_chunks += static_cast<int>(math::ceil_div<int64_t>(buckets[bucket_idx].num_bytes, num_chunk_bytes)); // 累加该 bucket 按 chunk 大小切出的块数（向上取整）
        EP_HOST_ASSERT(num_chunks <= layout::AllGatherSignals::kNumMaxChunks and "Too many chunks"); // chunk 数不得超过 workspace 信号量容量

        // Compile // 编译：实例化混合 all-gather 模板（模板实参：两层域大小、线程数、超时、上下文索引）
        const auto kernel = jit->compile("bucket_hybrid_all_gather", std::format(R"(
#include <deep_ep/impls/bucket/all_gather/hybrid.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::hybrid_all_gather<{}, {}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, context.num_nvl_ranks, kNumThreads, context.num_gpu_timeout_cycles, context_idx));

        // Launch // 协作式启动：grid=RDMA 对端数（每对端一个 SM），每 block 256 线程
        jit->launch( // 提交 kernel 启动
            kernel, { // 启动配置
                .stream = stream, // 目标 CUDA 流
                .grid_dim = dim3(context.num_rdma_ranks, 1, 1), // grid 维度：RDMA 对端数个 block
                .block_dim = dim3(kNumThreads, 1, 1), // block 维度：256 线程
                .cooperative = true, // 协作启动：支持 grid 内全局同步
            },
            context.dev_comm, context.window, // 设备通信器与对称内存窗口
            context.workspace, context.buffer, // workspace（信号量）与数据缓冲区
            num_buckets, num_chunks, num_chunk_bytes, context.rdma_rank_idx, context.nvl_rank_idx, context.rank_idx, // bucket 数、chunk 数与大小、本 rank 在两层域中的编号
            buckets // bucket 列表
        );

        // NVLink peers' buffers // 收集所有 NVLink 对端的缓冲区地址，供 CE 转发使用
        std::vector<void*> lsa_buffers; // NVLink 对端缓冲区（LSA 域内互访）列表
        lsa_buffers.reserve(context.num_nvl_ranks - 1); // 预留容量（除本地外每个对端一个）
        for (int rank_offset = 1; rank_offset < context.num_nvl_ranks; ++ rank_offset) { // 按环形顺序遍历所有 NVLink 对端
            const int peer_rank_idx = (context.nvl_rank_idx + rank_offset) % context.num_nvl_ranks; // 对端 rank 编号
            lsa_buffers.push_back(context.get_sym_ptr(context.buffer, peer_rank_idx)); // 经对称地址取对端缓冲区指针
        }

        std::vector<size_t> sizes; // CE 转发用的批量描述符
        std::vector<void*> dst_ptrs, src_ptrs; // push_to_nvl_peers：把本地缓冲区某段推到所有 NVLink 对端的同偏移处
        const auto push_to_nvl_peers = [&](const int64_t& src_offset, const size_t& num_bytes) { // 源地址：本地缓冲区加上源偏移
            const auto src_ptr = math::advance_ptr(context.buffer, src_offset); // 对每个 NVLink 对端生成一条拷贝
            for (const auto& lsa_buffer: lsa_buffers) { // 源固定为本地地址
                src_ptrs.push_back(src_ptr); // 目的为对端缓冲区的同偏移地址
                dst_ptrs.push_back(math::advance_ptr(lsa_buffer, src_offset)); // 拷贝字节数
                sizes.push_back(num_bytes);
            } // flush_pushes：一次性提交积压的拷贝并清空描述符
        }; // 批量提交 CE 拷贝
        const auto flush_pushes = [&]() { // 清空源地址列表
            driver::copy_engine_push(stream, dst_ptrs, src_ptrs, sizes); // 清空目的地址列表
            src_ptrs.clear(); // 清空大小列表
            dst_ptrs.clear();
            sizes.clear();
        };

        // Local shards over NVLink // 先把本地各 bucket 的分片经 NVLink 推给所有对端
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) { // 遍历每个 bucket
            const auto& [offset, num_bytes] = buckets[bucket_idx]; // 取该 bucket 的偏移与字节数
            push_to_nvl_peers(offset + context.rank_idx * num_bytes, num_bytes); // 推送本 rank 的分片（源偏移 = bucket 偏移 + 本 rank × bucket 大小）
        } // 提交本批拷贝
        flush_pushes();

        // Forward RDMA results over NVLink, chunk by chunk // 再逐 chunk 转发 RDMA 收到的远端分片
        auto& signals = static_cast<layout::BucketWorkspace*>(context.workspace)->signals[context_idx].all_gather; // 取本上下文的 all-gather 信号量（每 QP 每 chunk 一个尾指针）
        std::vector<uint64_t*> wait_signals(kNumQPsPerChunk); // 等待信号数组（每条 QP 一个）
        int chunk_idx = 0; // chunk 计数器
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) { // 遍历每个 bucket
            const auto& [offset, num_bucket_bytes] = buckets[bucket_idx]; // 取该 bucket 的偏移与字节数
            for (int64_t chunk_offset = 0; chunk_offset < num_bucket_bytes; chunk_offset += num_chunk_bytes, ++ chunk_idx) { // 按 chunk 大小遍历该 bucket 的每个 chunk
                // Wait for all RDMA peers to finish this chunk on every QP // 等待所有 RDMA 对端在每条 QP 上完成该 chunk
                for (int qp_idx = 0; qp_idx < kNumQPsPerChunk; ++ qp_idx) // 为每条 QP 设置等待信号
                    wait_signals[qp_idx] = signals.tails[qp_idx] + chunk_idx; // 指向该 QP 尾指针数组的第 chunk_idx 个槽位
                driver::copy_engine_wait(stream, wait_signals, context.num_rdma_ranks - 1); // CE 等待：直到每个对端都完成该 chunk（目标值 = 对端数 - 1）

                const auto num_bytes = std::min<int64_t>(num_bucket_bytes - chunk_offset, num_chunk_bytes); // 该 chunk 的实际字节数（末块可能不足）
                for (int rank_offset = 1; rank_offset < context.num_rdma_ranks; ++ rank_offset) { // 环形遍历除本地外的所有 RDMA 对端
                    const int peer_rank_idx = (context.rdma_rank_idx + rank_offset) % context.num_rdma_ranks; // 对端 RDMA rank 编号
                    const auto src_rank_idx = peer_rank_idx * context.num_nvl_ranks + context.nvl_rank_idx; // 对端分片对应的全局源 rank 编号
                    push_to_nvl_peers(offset + src_rank_idx * num_bucket_bytes + chunk_offset, num_bytes); // 把该 chunk 经 NVLink 推给本机所有 NVLink 对端
                } // 提交本批拷贝
                flush_pushes();
            }
        }
        EP_HOST_ASSERT(chunk_idx == num_chunks); // 实际处理的 chunk 数必须与预算一致

        // Exit barrier is in the epilogue
    } // 结束 barrier 已放在 kernel 的收尾阶段
}

}  // namespace deep_ep::bucket
