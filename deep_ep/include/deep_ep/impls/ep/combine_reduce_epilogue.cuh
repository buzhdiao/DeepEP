#pragma once
// 头文件保护：防止重复包含

#include <deep_ep/common/compiled.cuh>
// 引入编译期公共定义（静态断言、对齐常量等）
#include <deep_ep/common/ptx.cuh>
// 引入内联 PTX 封装：warp 原语、TMA store、mbarrier 等底层操作
#include <deep_ep/layout/ep/token.cuh>
// 引入 EP 的 token/buffer 布局抽象，用于在接收缓冲区上定位 rank/token/scale

#include <deep_ep/impls/ep/combine_utils.cuh>
// 引入 combine 归约的核心工具函数（combine_reduce、CombineVecTraits 等）


namespace deep_ep::ep {
// 进入 deep_ep::ep 命名空间

// combine 的收尾（epilogue）kernel：把 dispatch/combine 各 rank 缓冲区里的
// 已聚合激活值按 top-k 归约，输出最终的 combined 激活、top-k 权重与 top-k 索引
// 模板参数：布局开关、SM 数/warp 数、scale-out/scale-up rank 数、hidden 大小等
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          // TODO: merge these two variables into one (ensure the whole file does not contain "scaleup")
          // TODO: 这两个变量后续应合并（确保整个文件不再出现 "scaleup"）
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          // 每个线程块的线程数 = warp 数 * 32
          int kNumThreads = kNumWarps * 32,
          // hidden 维度的字节数（BF16，每元素 2 字节）
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          // 非 hybrid 模式下实际参与通信的 rank 数（单 scale-out 时取 scale-up rank 数）
          int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleoutRanks,
          // 是否按 rank 维度组织归约布局（见 combine_utils.cuh 的判定函数）
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          // 布局中每个槽位代表的 token 数（rank 数或 top-k 数）
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>()>
// launch_bounds：限定线程块大小为 kNumThreads、每 SM 至少驻留 1 个块，辅助寄存器分配
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* recv_buffer,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    // 每个 scale-out rank 拥有的 expert 数
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    // 每个（scale-up * scale-out）rank 拥有的 expert 数
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    // 静态断言：expert 数必须能被总 rank 数整除
    EP_STATIC_ASSERT(kNumExperts % (kNumScaleupRanks * kNumScaleoutRanks) == 0, "Invalid number of experts or ranks");

    // Utils
    // 当前线程块对应的 SM 编号
    const auto sm_idx = static_cast<int>(blockIdx.x);
    // 当前 warp 在块内的编号与 lane 编号
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    // 全局 warp 编号：warp 维度优先映射到不同 SM，使最后一批任务均匀落在各 SM 上
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;   // NOTES: Here we prioritize distributing tasks to different SMs to ensure that the last wave is evenly concentrated on each SM.

    // Load buffers from scale-out or scale-up ranks
    // 动态共享内存：按 TMA 对齐要求（128B）声明，作为 TMA store 的中转缓冲
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    // 通信侧 token 布局：hidden 字节数、无 scale、top-k 个槽位、不使用 FP8
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    // 通信接收缓冲区布局：按 kNumTokensInLayout 个槽位、每槽位最多 kNumMaxTokensPerRank 个 token 组织
    const auto comm_buffer = layout::BufferLayout<false>(
        comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);

    // Store buffers
    // 输出侧 token 布局：无 scale、无 top-k 槽位
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    // 输出缓冲区：combined_x 按 num_combined_tokens 个 token 组织
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);
    // 本 warp 独占的一块共享内存作为 TMA store 暂存区（kNumWarps 份、每份 1 token）
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)
        // 取本 warp 对应的 rank 分块
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    // Bias layout
    // bias_0 缓冲区布局（可能为空指针，见下方判空）
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);
    // bias_1 缓冲区布局
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);

    // Will block until the main combine kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    // PDL（Programmatic Dependent Launch）：等待主 combine kernel 完成且其数据全部可见
    // NOTES: 使用了 PDL，请不要用 `__ldg`（会绕过依赖可见性语义）
    cudaGridDependencySynchronize();

    // Read from buffers and do reduction
    // 以 warp 为单位跨步遍历所有待归约 token（grid-stride：每次跨 kNumWarps * kNumSMs 个）
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {
        // Preprocess all indices
        // 预处理该 token 的全部 top-k 索引（每个 lane 负责一个 top-k 位置）
        int stored_dst_rank_idx = -1, stored_dst_expert_idx = -1;
        // top-k 数不能超过 32（一个 warp 的 lane 数上限）
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        if (lane_idx < kNumTopk) {
            // lane 对应的 top-k 位置读出目标 expert 全局编号
            stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            // 由 expert 编号反推目标 rank：单 scale-out 时按每 rank expert 数划分，否则按 scale-out 划分
            stored_dst_rank_idx = stored_dst_expert_idx >= 0 ?
                stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout) : -1;
        }
        // 同步 warp，保证各 lane 写出的 rank/expert 索引对其他 lane 可见
        __syncwarp();

        // Sort valid top-k indices to front
        // 计算去重键并判断是否需要去重（把无效/重复的 top-k 槽位“挤掉”）
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {
                // Activations are never reduced before
                // 展开布局且不允许多重归约：激活从未被归约过，无需去重
                return {false, 0};
            } else if constexpr (kNumScaleoutRanks != 1 and not kUseExpandedLayout and not kAllowMultipleReduction) {
                // Hybrid mode without expanded layout and multiple reduction. Should deduplicate on a per-rank basis
                // hybrid 模式（无展开布局、无多重归约）：按每 rank 的 expert 编号去重
                return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / kNumExpertsPerRank : -1};
            } else {
                // Should deduplicate on a per-rank (for non-hybrid mode) or a per-scale-rank (for hybrid mode) basis
                // 其余情况：非 hybrid 按 rank 去重，hybrid 按 scale-out rank 去重
                return {true, stored_dst_rank_idx};
            }
        }();
        // 计算归约有效掩码：去重模式下仅保留每个去重键首次出现的 lane，且目标 rank 必须有效
        auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            // 不去重时只要目标 rank 有效即参与归约
            ptx::gather(stored_dst_rank_idx >= 0);
        // 每个 lane 持有一份槽位下标数组（长度为布局 token 数）
        int topk_slot_idx[kNumTokensInLayout];
        // 从掩码中提取各有效槽位；rank 布局下通过 warp 交换把 rank 号换回本 lane
        compute_topk_slots(
            topk_slot_idx, reduce_valid_mask,
            [=](const int& idx) {
                // rank 布局：槽位对应 rank，需 exchange 换取 lane 上存的 rank 号；否则直接用槽位号
                return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
            }
        );

        // Iterate over per-hidden-chunk stage
        // 按 hidden 分块的循环阶段：先根据 hidden 字节数选择向量类型与展开参数
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
        // hidden 维度的向量总数
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);
        // 编译期求解最大展开因子（保证 kHiddenVec 能被 32*factor 整除）
        constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
        // 调用通用归约：从各槽位缓冲区读出、累加后写入本 warp 的 TMA 暂存区
        combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(
            // 当前 lane 编号
            lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
            /* 取源缓冲区基址：槽位号 -> 该 rank 该 token 的数据指针 */
            /* Get source base */ [=](const int& slot_idx) {
                // 槽位 -> rank 缓冲区 -> token 缓冲区 -> 基址（reinterpret 为向量类型指针）
                return static_cast<combine_vec_t*>(
                    comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());
            },
            /* 等待上一次 TMA store 完成后暂存区才可复用 */
            /* Wait buffer release */ [=]() {
                // 等待本 warp 所有未完成的 TMA store 结束
                ptx::tma_store_wait();
                // warp 内同步，确保全员都等到释放
                __syncwarp();
            },
            /* bias_0 为空则传 nullptr，否则指向该 token 的 bias_0 数据 */
            /* Bias 0 */ bias_0 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()),
            /* bias_1 同上 */
            /* Bias 1 */ bias_1 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr())
        );
        // TMA store 前的 fence：确保共享内存写入对异步代理可见
        ptx::tma_store_fence();
        // warp 内同步
        __syncwarp();

        // Issue TMA copy
        // 由 warp 内选出的唯一 lane 发起 TMA store
        if (ptx::elect_one_sync()) {
            // 从共享内存暂存区以 TMA 批量拷贝 kNumHiddenBytes 到全局输出缓冲区
            ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),
                              tma_buffer.get_base_ptr(), kNumHiddenBytes);
            // 提交 TMA store 批次
            ptx::tma_store_commit();
        }
        // 等待 warp 内发射完拷贝再进入下个 token 的处理
        __syncwarp();

        // Write top-k weights
        // 写出该 token 的 top-k 权重（缓冲区可能为空）
        if (combined_topk_weights != nullptr) {
            // 找出与本 lane 目标 rank 相同的“主 lane”，用于统一选取 rank 缓冲区
            const auto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));
            if (lane_idx < kNumTopk) {
                // 默认值为 0（无效槽位权重输出 0）
                float value = 0;
                if (stored_dst_rank_idx >= 0) {
                    // rank 布局按 rank 号取缓冲区，否则用主 lane 的槽位；top-k 权重区按 lane 偏移
                    const auto dst_ptr = comm_buffer
                        .get_rank_buffer(kUseRankLayout ? stored_dst_rank_idx : master_lane_idx)
                        .get_token_buffer(token_idx).get_topk_weights_ptr() + lane_idx;
                    // 从通信缓冲区读出该 top-k 位置的权重
                    value = *dst_ptr;
                }
                // 写入输出的 top-k 权重数组
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;
            }
            // warp 同步后再处理下一个 token
            __syncwarp();
        }
    }
}

}  // namespace deep_ep::ep
// 结束 deep_ep::ep 命名空间
