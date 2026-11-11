# 第 04 章 · EPBuffer 与 dispatch 全链路

## 4.1 dispatch 的契约：输入、输出与不变式

`EPBuffer.dispatch`（`deep_ep/buffers/ep.py`，约 200 行带注释的实现）是 DeepEP 使用频率最高的接口。先把它的契约写清楚：

**输入**

| 参数 | 形状/类型 | 语义 |
| --- | --- | --- |
| `x` | `[num_tokens, hidden]`，`torch.bfloat16`；FP8 模式为 `(fp8 张量, scale_factors)` 二元组 | 待分发 token 的隐藏状态；FP8 用 `torch.float8_e4m3fn` 加缩放因子 |
| `topk_idx` | `[num_tokens, num_topk]`，`deep_ep.topk_idx_t`（构建期 32/64 位，通常 `torch.int64`） | 每 token 的专家索引；`-1` 表示该槽位未选中 |
| `topk_weights` | `[num_tokens, num_topk]`，`torch.float` | 每 token 对各专家的门控权重（可省略） |
| `num_experts` / `num_max_tokens_per_rank` / `expert_alignment` | 标量 | 专家总数 / 容量上限 / 每本地专家接收数的对齐量 |
| `cumulative_local_expert_recv_stats` | `[num_local_experts]`，int | 累计式专家接收计数（在线 EP 负载监控用） |
| `handle` | 可选 `EPHandle` | 缓存的上一次 dispatch 布局（复用时 `topk_idx` 必须为 None） |

**输出**（`defer_epilogue=False` 时为五元组）

`(recv_x, recv_topk_idx, recv_topk_weights, handle, event)`：接收缓冲中的 token、随行专家索引、展开权重、路由句柄、事件。`defer_epilogue=True` 时直接返回 `EventOverlap`，等待点 `event.current_stream_wait()` 执行 epilogue 后给出四元组（`combined` 侧为二元组）。

两条关键不变式直接写进了 README 与 docstring：

1. **topk_idx 按引用存入句柄**：`"Stored by reference in the handle; must not be modified until all uses of the handle complete."`——句柄不复制路由表，因此用户必须保证句柄存活期内路由表不变。
2. **新路由需要新 dispatch**：`"新的路由决策需要一次全新的 dispatch；仅靠解码并不会使旧的路由句柄变为可复用。"`——缓存复用的是"布局"，不是路由决策本身。

## 4.2 路由的数学：从专家编号到目标 rank

设专家总数 $E$、rank 数 $P$（且 $E$ 整除 $P$），专家 $e$ 的属主为

$$
\mathrm{owner}(e) = \left\lfloor \frac{e}{E/P} \right\rfloor .
$$

于是 `topk_idx[t]` 这一行 $k$ 个编号把 token $t$ 的数据映射到最多 $k$ 个目标 rank（可能含本 rank）。dispatch 的本质是：

$$
\text{slot}(t, e) \; \text{写入} \; \text{rank}_{\mathrm{owner}(e)} \text{ 的接收缓冲区第 } \text{slot}(t,e) \text{ 槽}.
$$

接收端必须先知道"我会收到多少个 token（每 rank 每专家各多少）"，才能给每个到达的 (token, expert) 对分配槽位。这个"先计数、再搬运"的两阶段结构是 all-to-all 与生俱来的，也是 dispatch 内核复杂度的根源。

DeepEP 的参考实现（`deep_ep/utils/refs.py` 的 `dispatch`）用纯 PyTorch 写出了这个语义：接收 token **按 rank 排序、rank 内按来源 token 全局编号 `src_token_global_idx = src_rank_idx * num_max_tokens_per_rank + src_token_local_idx` 排序**，越界的 `recv_topk_idx` 置 `-1`。这个"确定性参考语义"正是内核实现所对齐的行为，也是测试（第 12 章 `calc_diff`）的判定基准。

## 4.3 TokenLayout：一个 token 槽位的逐字节解剖

`deep_ep/include/deep_ep/layout/ep/token.cuh` 定义了全链路最重要的数据结构：

```cpp
struct TokenLayout {
    // 单个 token 槽位布局：[hidden | SF | metadata(可选) | mbarrier(可选)]
    // 三段数据各自按 TMA 对齐字节数对齐，总和再按 RDMA 对齐字节数对齐
    int num_hidden_bytes, num_sf_bytes;
    bool with_metadata;                 // 是否携带元数据区
    int num_topk, num_metadata_bytes;
    // 备注：top-k 索引固定使用 32 位整数
};
```

dispatch 与 combine 用不同的构造（`csrc/kernels/ep/dispatch.hpp` / `combine.hpp`）：

```cpp
// dispatch：hidden 字节 + scale-factor 字节 + topk 元数据，启用 padding
get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk)
    = TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, /*with_metadata*/ true);

// combine：只有 hidden（BF16）+ topk 元数据，无 scale factor，关闭 padding
get_combine_token_layout(hidden, elem_size, num_topk)
    = TokenLayout(hidden * elem_size, 0, num_topk, /*with_metadata*/ false);
```

为什么这样设计？三条硬约束共同塑造了它：

1. **TMA 对齐**：TMA（Tensor Memory Accelerator）的批量异步拷贝要求描述符对齐（对齐常量为 32 字节，见 `deep_ep.get_num_tma_alignment()`）。把 hidden、SF、metadata 三段各自独立对齐，每一段都能用 TMA 单独搬运，而不必把不相关的数据捆绑在一次拷贝里。
2. **RDMA 对齐**：槽位总大小再按 RDMA 对齐（`kNumRDMAAlignmentBytes`）向上取整，保证"跨 rank 的一次远程写"从对齐地址出发。`EPSignals` 等信号区结构同样 `alignas(kNumRDMAAlignmentBytes)`。
3. **SF 与数据同槽**：FP8 量化的 scale factor（`num_sf_packs` 个打包单元）与数据放在同一个槽位里随行，接收端拿到的数据是"自描述"的——不需要第二次通信去取 scale。`use_tma_aligned_col_major_sf=True` 时 SF 进一步按 TMA 要求排成列主序，供下游 DeepGEMM 直接消费。

```text
TokenLayout 槽位（FP8 dispatch，with_metadata=true）：

字节偏移  0                H（hidden*1B，向上对齐到 32B）      +SF（向上对齐）      +META（向上对齐）
        ┌──────────────────────┬──────────────┬──────────────────┬───────────────┐
        │  hidden 激活 (fp8)    │  padding     │  scale factors   │  metadata     │ …mbarrier
        │                      │              │  (num_sf_packs)  │  (topk 相关)  │   (可选)
        └──────────────────────┴──────────────┴──────────────────┴───────────────┘
          ^ 32B 对齐                            ^ 32B 对齐         ^ 32B 对齐      整体再按 RDMA 对齐
```

## 4.4 两种输出布局：排序式与展开式

dispatch 的接收端有两种合法布局，对应训练/prefill 与分组 GEMM 两种消费方式：

**（1）排序式（非 expanded，`do_expand=False`）**：接收缓冲按"先按来源 rank、rank 内按 `src_token_global_idx`"排序，`recv_x` 形状为 `[收到的 token 总数, hidden]`，随行 `recv_topk_idx`（越界为 -1）。这是参考实现的语义，适合推理引擎按 token 组织后处理。

**（2）展开式（expanded，`do_expand=True`）**：**每 token 每专家槽一个槽位**——同一个 token 若被 3 个本地专家选中，就在接收缓冲里出现 3 行。每本地专家的接收数向上对齐到 `expert_alignment`（README 明确建议分组 GEMM 用户设为 `deep_gemm.get_mk_alignment_for_contiguous_layout()`），`do_zero_padding=True` 时把专家之间的对齐空隙清零。展开式的收益是：分组 GEMM 可以按"每专家连续区间"直接吃满矩阵，无需再排序；代价是缓冲占用变大（恰为 `Σ_e num_recv_tokens_per_expert` 加对齐空隙）。展开式下 `recv_topk_weights` 变成一维张量，**每专家行对应一个值**（README），专家计算后直接乘上即可。

`num_recv_tokens_per_expert` 前缀和（`handle.psum_num_recv_tokens_per_expert`）告诉专家 GEMM 每个专家的有效区间——在无 CPU 同步模式下，输出按配置容量分配，"有效范围"只能由这个前缀和给出（README：*"专家 GEMM 必须使用 `handle.psum_num_recv_tokens_per_expert` 来确定有效的专家范围"*）。

## 4.5 元数据家族与 EPHandle

dispatch 产生的元数据不只是"计数"，而是一族互相配合的张量。`EPBuffer._unpack_handle` 把句柄拆成十个字段，配合 `EPHandle` 的属性，可以列出完整的元数据家族：

| 元数据 | 含义 | 生产阶段 | 主要消费者 |
| --- | --- | --- | --- |
| `num_recv_tokens` | 接收 token 总数（未展开） | notify 计数 + CPU/GPU 同步 | 输出尺寸 |
| `num_expanded_tokens` | 展开后的总行数（含对齐空隙） | 槽位分配 | 展开 GEMM |
| `num_recv_tokens_per_expert_list` | 每本地专家接收数（CPU 列表） | CPU 同步 | 负载监控/静态调度 |
| `psum_num_recv_tokens_per_scaleup_rank` | 各 scale-up rank 接收数**前缀和** | notify 计数 | 内核槽位分配；句柄复用 |
| `psum_num_recv_tokens_per_expert` | 各专家接收数前缀和 | 槽位分配 | 专家 GEMM 有效范围 |
| `num_unaligned_recv_tokens_per_expert` | 未对齐时各专家接收数 | 计数阶段 | 对齐后的槽位修正 |
| `dst_buffer_slot_idx` | 每个 (token, expert) 对在远端缓冲的槽位号 | 槽位分配 | combine 逆向定位 |
| `token_metadata_at_forward` | 前向转发的来源元数据（每 token，宽度 $2 + 2k$，见 `hybrid_dispatch.cuh` 的 `kNumForwardMetadataDims`） | 发送端打包 | 混合模式 combine 还原 |
| `recv_src_metadata` | 接收端收到的来源元数据 | 接收 | combine 回程路由 |
| `channel_linked_list` | 同通道 token 的链表（`kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1`） | 发送端串联 | combine 按通道遍历 |
| `topk_idx`（引用） | 路由表本体 | 用户输入 | 一切布局的源头 |

`EPHandle` 还携带 `do_expand`、`num_experts`、`expert_alignment`、`num_max_tokens_per_rank`、`num_sms` 等配置快照，保证 combine 能在**零额外协商**的情况下还原路由。句柄可复用：`"Can be reused as a cached handle in subsequent EPBuffer.dispatch calls to skip layout recomputation"`——解码场景中路由不变时（GDN/投机解码的草稿匹配等场景），缓存 dispatch 直接重放已保存的展开布局，**跳过接收计数所需的 CPU 同步**（README："缓存的 dispatch 会重放已保存的展开布局，而无需再做一次接收计数的 CPU 同步"）。`EPHandle.deterministic_sort`（`deep_ep/buffers/ep.py` 第 119 行起）提供确定性排序路径，配合 `EPBuffer` 构造参数 `deterministic=True` 与 `check_torch_deterministic()` 的环境校验，满足训练可复现需求。

## 4.6 内核执行模型：notify warp、通道与两级路由

设备端实现分两条路径：纯 scale-up 的 `impls/ep/dispatch.cuh`（模板参数 `kIsScaleupNVLink` 表示所有对端 NVLink 可直达）与混合模式的 `impls/ep/hybrid_dispatch.cuh`。它们的骨架一致：

**阶段 A：计数与通知。** 每个目标槽位的归属需要"到达者报数"。`csrc/kernels/ep/dispatch.hpp` 定义：

```cpp
constexpr int kNumNotifyWarps = 4;                       // 专职统计/通知的 warp 数
static int get_num_notify_smem_bytes(const int& num_ranks, const int& num_experts) {
    return math::align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int);
}
```

4 个 notify warp 在共享内存里维护"每 rank + 每专家"的计数器阵列（大小按 $4 \times 32 = 128$ 个 lane 对齐），用原子加（PTX `red_add`，见第 10 章）累加到达数，最终写回 workspace 的 `notify_reduction[kNumMaxRanks + kNumMaxExperts]`（`layout/ep/workspace.cuh`：前 `kNumMaxRanks` 项按 rank 记录、其后按专家记录）。`EPSignals` 同时给出规模上限：`kNumMaxExperts = 2048`、`kNumMaxExpertsPerRank = 512`、`kNumMaxChannels = 1280`——这些常量既是缓冲区规划的下界，也解释了 QP 预分配为什么在 hybrid 模式下需要 65/129 个（每通道独立 QP + notify 专用）。

**阶段 B：槽位分配。** 计数完成后，每 rank/每专家的接收数经前缀和（`psum_*` 名称即 partial-sum 之意）映射成确定性的槽位区间。发送端为每个 (token, expert) 对计算 `dst_buffer_slot_idx` 并随数据一起（或先行）送达，使接收端可以直接写入正确位置而无需二次排序。对齐处理在此层完成：`num_unaligned_recv_tokens_per_expert` 记录未对齐计数，`expert_alignment` 生效后得到 `num_expanded_tokens`。

**阶段 C：数据搬运。** 纯 scale-up 路径经 NVLink 用 `get_sym_ptr` 得到的对端地址直接写入对端槽位（配合 TMA 批量拷贝与 mbarrier 完成通知）；hybrid 路径分两级——先经 RDMA（NCCL Gin）把数据送到目标 scale-out rank 的中转区，再由该节点内的 NVLink 二级分发到目标 rank。`hybrid_dispatch.cuh` 的模板参数展示了层级参数化：`kNumScaleupRanksPerLane = ceil(kNumScaleupRanks / 32)`（每个 lane 分摊的节点内 rank 数）、`kNumMaxTokensPerChannel = ceil(kNumMaxTokensPerRank / kNumChannels)`（通道化切分批量）。

**阶段 D：收尾（epilogue）。** `impls/ep/dispatch_copy_epilogue.cuh` 负责把通信缓冲区的数据整理成用户输出（拷出 `recv_x`、写 `recv_topk_idx/weights`），支持 `st_bulk` 等 PTX 批量存储。`defer_epilogue=True` 时这步被推迟到 `current_stream_wait()` 执行——这是 README"延迟 epilogue"一节的实现载体。

```text
dispatch 内核时间线（hybrid 模式，示意）：

 SM0..SMn ─┬─ 通道 c=0..C-1：发送端打包 token → RDMA 一级（Gin）→ 目标节点中转区
           ├─ notify warp：原子累加 (rank, expert) 计数 → workspace
           ▼
 目标节点：NVLink 二级分发到目标 rank 的槽位（按 dst_buffer_slot_idx）
           ▼
 接收端：mbarrier 通知完成 → CPU 同步（可选）读计数 → epilogue 拷出 recv_x
```

## 4.7 同步模式：CPU 数还是 GPU 数

`do_cpu_sync` 是 dispatch 语义的分水岭：

- **`do_cpu_sync=True`（训练/prefill 默认路径）**：内核完成后 CPU 同步读取接收计数，输出张量按**精确**尺寸创建，`num_recv_tokens_per_expert_list` 直接可用。代价是一次主机-设备同步，中断异步流水。
- **`do_cpu_sync=False`（解码路径）**：CPU 不同步，输出按容量分配，配合 `handle.psum_num_recv_tokens_per_expert`（GPU 侧前缀和）界定有效范围。配合 `async_with_compute_stream=True` 与 `defer_epilogue=True`，调用即刻返回事件，CPU 零等待——这正是 README 对解码场景的建议组合。

`allocate_on_comm_stream` 是配套的所有权开关：当传入 `previous_event`（前序独立计算的事件）时必须置 True，使新分配的张量所有权登记在通信流上，避免 PyTorch 的缓存分配器在计算流视角下过早复用尚未被通信内核读完的显存（第 10 章详述）。

## 4.8 一次 dispatch 的完整数据流（总结图）

```text
用户张量                设备内核                                接收端                    用户输出
topk_idx ──┐
topk_w   ──┤  发送端 SM：
x / sf   ──┤    ① 解析 topk_idx → owner rank（路由）
           │    ② 通知：原子加计数（notify warp → workspace.notify_reduction）
           │    ③ 槽位分配：前缀和 → dst_buffer_slot_idx
           │    ④ 搬运：NVLink 直写（sym_ptr）或 RDMA 一级 + NVLink 二级
           │                                          ┌────────────────┐
           └─────────────────────────────────────────▶│ 对端槽位：      │
                                                        │ [hidden|SF|md] │
                                                        └───────┬────────┘
                                                                ▼
                                      ⑤ 接收端：mbarrier 确认 → 计数汇总 →
                                         （可选 CPU sync）→ epilogue
                                                                │
                                                                ▼
                                              recv_x, recv_topk_idx,
                                              recv_topk_weights, EPHandle
```

## 4.10 延伸讨论：通道化设计的成本收益分析

第 4.6 节提到通道（channel）是内核内部的并行子流划分（上限 `kNumMaxChannels=1280`），本节把"为什么要通道化、多少通道才对"论证完整。通道化的收益来自三个方面。其一，**同步粒度细化**：若整个 all-to-all 只有一个全局"计数完成"信号，最早的通道也要等最晚的数据到达；通道化让每通道独立推进（第 6.9 节的"通道内流水"），完成时间由最慢通道而非最慢消息决定。其二，**QP 与链路的并发映射**：每通道独立 QP（第 6 章 QP 预分配的动机），多通道把流量分散到多个队列与多条网络路径，单链路热点被摊平。其三，**SM 的工作分配单位**：SM 数与通道数成对配置（估算结果"偶数且至少 4"的一部分原因），每个通道的 warp 组固定归属特定 SM，寄存器预算与共享内存按通道划分。

通道化的成本同样真实：每个通道有自己的信号区槽位（workspace 按通道扩容）、自己的屏障参与与同步开销，通道越多固定开销越大；混合模式的 QP 预分配（65/129）与通道数直接挂钩。因此通道数不是"越多越好"，而是"让每通道的流量足够摊满链路、又让通道数足以覆盖并行度"的折中——这正是解析式估算同时以 SM 数与拓扑为输入的原因（第 6 章）。调优视角下，通道数由内核按 SM 数与拓扑派生，用户通过 `num_sms` 间接控制它：SM 减半则通道减半，单通道流量加倍——当实测出现"通道饥饿"（链路利用率低而 SM 忙）或"通道过碎"（同步开销占比高）时，沿这个映射方向调整 SM 数即可。

## 4.11 延伸讨论：槽位分配的确定性结构

第 16 章从确定性专题的角度看槽位分配，本节从算法结构的角度补一个观察：dispatch 的槽位分配之所以能既高效又确定，是因为它把一个看似动态的问题转化成了**前缀和问题**。"接收端不知道自己会收到什么"的动态性，被 notify warp 的原子计数（加法交换律）吸收成"每 rank/每专家的计数"；而给定计数，前缀和唯一确定每个 (token, expert) 对的槽位——**到达顺序不影响布局**。这个转化正是第 16 章"布局与槽位层结构上偏向确定"的算法根源：原子加只承诺"和"的确定性（加法可交换），前缀和把"和"翻译成"位置"，两步都不引入顺序依赖。对比另一种常见方案——接收端用链表/队列动态挂接到达的数据（到达顺序即布局顺序）——那种方案的布局天然依赖到达时序，确定性差且需要二次整理。DeepEP 在 `channel_linked_list`（第 4 章元数据表）里也用了链表结构，但那是用于 combine 回程遍历的辅助结构，主布局仍由前缀和决定。这个"先计数、后定位"的范式是理解 EP 内核一切同步代码（notify、mbarrier、屏障）的钥匙：它们都在为"计数"与"定位"这两个确定性步骤服务。

## 4.13 延伸讨论：元数据族的成本核算

第 4.5 节列出了 `EPHandle` 的元数据家族，本节核算它的内存与带宽成本——因为它随 token 数线性增长，是缓冲区容量的隐性组成。按张量逐项估算（设每 rank 批次 $T$、top-$k$、rank 数 $P$、专家数 $E$）：`token_metadata_at_forward` 宽度为 $2 + 2k$ 个整型（`kNumForwardMetadataDims`，第 4.6 节 hybrid 内核常量），成本 $T \cdot (2+2k) \cdot 4$ 字节；`dst_buffer_slot_idx` 每对 (token, expert) 一个索引，成本 $T \cdot k \cdot 4$；`recv_src_metadata` 与接收 token 数成正比（最坏 $T \cdot k \cdot \frac{P-1}{P}$）；`channel_linked_list` 与通道容量相关（`kNumMaxTokensPerChannel * kNumScaleoutRanks + 1`，第 4 章）；前缀和族（`psum_*`）按 rank 与专家数计，成本 $P + E$ 个整型量级，可忽略。合计的量级结论：元数据约为"激活流量的一个小常数因子"——以 4 字节整型对 2 字节 BF16 激活换算，`token_metadata_at_forward` 的每 token 成本约等于两三个隐藏元素的流量。这笔账解释了两个设计选择：其一，为什么句柄复用（第 4.8 节）收益大——缓存复用省的不只是 CPU 同步，还有元数据的重算与搬运；其二，为什么 `topk_idx` 按引用存储（第 4 章不变式）——复制一份路由表的内存与带宽成本不必要，引用语义把成本转移为用户纪律（第 14.8 节清单一的依据）。容量规划（第 19.3 节）中元数据隐含在"metadata 段"与缓冲区余量里，本节的核算让这项隐含成本变得显式。

## 4.14 延伸讨论：dispatch 的失败面与前置校验

dispatch 的参数面很宽（第 4.1 节契约表），本节梳理它的失败面并强调"前置校验"的设计取向。失败面的第一类是**契约违约**：dtype 或形状不符（`x` 非 BF16/FP8 二元组、`topk_idx` 非 `topk_idx_t`）、`-1` 之外的非法专家编号、`handle` 复用时传入了 `topk_idx`（互斥约束，第 4.1 节）、`previous_event` 未配 `allocate_on_comm_stream=True`（所有权约束，第 4.7 节）——这类失败全部在 Python/C++ 前置校验中以断言拒绝，不会进入内核。第二类是**容量违约**：批次超过 `num_max_tokens_per_rank`、展开后的行数超出缓冲区——同样前置拒绝（容量是构造期冻结的，第 3 章）。第三类是**环境违约**：PyTorch 确定性模式冲突（`check_torch_deterministic`，第 16.1 节）、NVLink 连接异常（构造期已校验，运行期不重复）。值得注意的设计取向：dispatch 没有"部分失败"语义——要么入队成功（此后失败只可能是运行期环境故障，如对端掉线触发超时），要么在前置校验即失败（不产生任何 GPU 侧副作用）。这个取向与第 14 章的故障分类呼应：**把"可静态判定的错误"全部挡在内核之前**，内核只面对动态世界的不确定性（时序、网络）。对集成方的含义：断言失败一律是调用方 bug（读断言信息修代码，第 14 章习惯二），运行期超时才是运维事件（走第 14.2 节诊断链）——两类失败的处置流程完全不同，不要混同。



## 4.12 本章小结

- dispatch 的契约以 `topk_idx`（`-1` 表未选中、按引用进句柄）与 `TokenLayout`（`[hidden|SF|metadata|mbarrier]` 三段 TMA 对齐 + 整体 RDMA 对齐）为核心；FP8 的 scale factor 与数据同槽随行。
- 输出布局二选一：排序式（参考实现语义，`refs.py`）或展开式（每专家连续区间，配 `expert_alignment`/`do_zero_padding`，面向分组 GEMM）；有效范围由 `psum_num_recv_tokens_per_expert` 前缀和界定。
- 元数据家族（十个 `_unpack_handle` 字段）构成"计数 → 前缀和 → 槽位 → 回程路由"的闭环，`EPHandle` 携带配置快照支持缓存复用（重放展开布局，免 CPU 计数同步）与确定性排序。
- 内核四阶段：notify 计数（4 个 notify warp、`notify_reduction[]`）→ 槽位分配 → 两级搬运（scale-up 直写 / hybrid RDMA+NVLink）→ 可延迟的 copy epilogue。
- 同步模式二选一：`do_cpu_sync=True` 拿精确尺寸（训练/prefill），`False` 配合延迟 epilogue 与事件（解码），由 `allocate_on_comm_stream` 保证张量所有权安全。

combine 是 dispatch 的镜像与逆运算——包括它在反向传播中的角色互换——是下一章的主题。
