# 第 08 章 · EngramBuffer：远程内存与按索引取数

## 8.1 动机：当参数大过显存

EP 的 dispatch/combine 解决的是"激活值去找专家"；但还有一类问题反着来：**参数太大，不可能（或不值得）常驻每张卡**。典型场景：

- **Wide-EP / 超大专家**：专家 FFN 极大，decode 时每步只有少量 token 命中少量专家，为每张卡复制全量专家权重在显存与带宽上都不可行；
- **大嵌入表 / 稀疏特征**：每条样本只触碰表的一小部分；
- **KV 缓存卸载**：历史上下文的 KV 存在 CPU 或远端 GPU，按需拉回。

这些场景的共同结构是：数据被**分片**存放（每 rank 一片，甚至 CPU 一片），而访问是**按索引的稀疏读取**——每 token 只需要 $\text{entries}$ 个条目，远小于全量。对此，all-to-all 是错配的（它要求"发送方知道要把数据给谁"，而这里是"读取方知道自己要什么"）；正确的原语是**远程内存读（RDMA get）**。

`EngramBuffer`（`deep_ep/buffers/engram.py`，C++ 运行时在 `csrc/buffers/engram.hpp` 与 `csrc/kernels/engram/engram.hpp`）正是这个原语。其类注释直译即设计：*"GPU fetch buffer with separate CPU or GPU storage shards registered for RDMA"*——带独立 CPU/GPU 存储分片、经 RDMA 注册的 GPU 取数缓冲区；典型用途是跨节点按索引拉取大表/专家参数：`write()` 写入本 rank 分片，`fetch()` 经 RDMA read 按索引从远端分片取数，返回**逐层**的等待钩子。

## 8.2 数据模型：分片、层与条目

Engram 的数据模型有四个维度，先建立坐标系：

| 维度 | 名称 | 说明 |
| --- | --- | --- |
| 分片（shard） | `storages` 列表 | 每个 rank 持有一份（或多份 dtype/用途不同的）存储，经 RDMA 注册为窗口（窗口数组 `storage_windows`，数量为 Gin 步长 `gin_min_stride`） |
| 层（layer） | `num_entries_per_layer: List[int]` | 存储按模型层组织，每层条目数可以不同（`EngramWorkspace::kNumMaxLayers = 128`） |
| token | `num_tokens` | 一次 fetch 批量处理的所有 token |
| 每 token 条目 | `num_entries_per_token` | 每个 token 在每层要取的条目数 |

`fetch` 的输入是索引张量：

```python
fetch(indices)   # indices: [num_layers, num_tokens, num_entries_per_token] 的 int32
                 # 返回逐层的完成钩子列表
```

语义上，`indices[l][t][j]` 是第 $l$ 层、第 $t$ 个 token 要取的第 $j$ 个条目的全局编号。条目全局编号到"属主分片 + 分片内偏移"的映射由配置决定（`set_config(num_entries_per_layer, hidden, …)`）：若条目按属主连续划分，则属主为 $\lfloor \text{id} / \lceil L_l / P \rceil \rfloor$ 这类确定性函数（具体划分由 Engram 的布局定义 `EngramLayout` 决定，`deep_ep/include/deep_ep/layout/engram/workspace.cuh` 中定义了 `EngramWorkspace` 与布局常量）。README 的用法说明："使用 `get_storage_size_hint` 和 `get_theoretical_config` 来确定缓冲区及其 QP 的规模，然后使用 `set_config` 和 `write` 填充层表。"

`write(storages, …)` 把本 rank 分片注册并（首次）填充；随后同一存储可被反复原地更新——Engram 是"寄存一次、随取随用"的模型，注册成本（RDMA window、`NCCL_WIN_STRIDE` 步长对齐）只付一次。

## 8.3 取数内核：GIN 请求与分层等待

宿主侧入口 `csrc/kernels/engram/engram.hpp` 的 `launch_engram_fetch` 签名浓缩了全部要素：

```cpp
static void launch_engram_fetch(
    const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window, // NCCL 设备通信器；主对称内存窗口
    ncclWindow_t* storage_windows, void* fetched, void* workspace,            // RDMA 存储分片窗口数组；拉取结果缓冲；workspace
    int* indices,                                   // 条目索引
    ncclGinRequest_t* last_gin_requests,            // 最近一次 GIN 请求句柄（用于后续 wait）
    void* sf_shards, void* fetched_sf,              // scale-factor 分片源；拉取到的 scale-factor
    const int& sf_token_stride, const int& sf_hidden_stride, const int& sf_layer_stride,
    const int& num_sf_shards,                       // scale-factor 分片数
    const std::vector<int>& num_entries_per_layer,  // 每层的条目数（编译为模板参数列表）
    const int& hidden, const int& elem_size, const int& num_sf_packs, // 隐藏维度；元素字节数；SF 包数
    const int& num_entries_per_token, …);
```

几个值得解读的设计：

1. **`ncclGinRequest_t last_gin_requests`**：GIN 的异步请求句柄。取数不是阻塞的——内核发起 RDMA read 后立即返回请求句柄，"数据是否到达"由独立的等待内核（`impls/engram/engram_fetch_wait.cuh`）检查。发起与等待分离，让"发起下一层的请求"与"等待上一层的完成"可以交错，这正是逐层钩子的实现基础。
2. **`num_entries_per_layer` 编译为模板参数列表**：每层条目数作为编译期常量注入（配合 DeepJIT，见第 11 章），内核可按层做完全展开与编译期校验，避免运行时对变长层表的循环开销。
3. **scale-factor 分片独立成组**（`sf_shards`/`fetched_sf` + 三个 stride + `num_sf_shards`）：与 dispatch 的 TokenLayout 思想一致——量化数据的 scale 随数据一起取回，但存放在独立的分片里（dtype 不同、stride 不同），取数内核按各自的 stride 定位。`sf_token_stride`/`sf_hidden_stride`/`sf_layer_stride` 三个步长描述 SF 在"token/hidden/layer"三维上的排布，内核用它把"条目编号"翻译成 SF 分片内的地址。
4. **`fetched` 结果缓冲**：取回的数据落在本 rank 的连续缓冲区（这与对称内存主窗口 `nccl_window` 同域），供后续内核直接消费。

设备端实现（`impls/engram/engram_fetch.cuh`）依赖三件套：`cuda/atomic`（`cuda::atomic_ref` 做细粒度计数）、`comm::timeout_while`（通信层超时等待原语，防止对端异常时本端死循环）、PTX 工具（warp 索引、mbarrier、内存栅栏）。等待阶段单独成文件（`engram_fetch_wait.cuh`）意味着"发起"与"等待"甚至可以是两次内核启动——按层钩子的粒度各司其职。

## 8.4 层间流水与等待钩子

`fetch` 返回"每层一个完成钩子"（README：*"并为每层返回一个完成钩子。在使用某一层抓取到的数据之前，请调用对应的钩子"*）。这个 API 形状直接编码了流水线思想：

```text
时间 ──────────────────────────────────────────────────────▶
fetch() 发起第 0..L-1 层的全部 RDMA 读（异步，内核返回钩子）
  │
  ├─ hook[0].wait() ──▶ 用第 0 层数据做计算 ─────────┐
  ├─ hook[1].wait() ──▶ 用第 1 层数据做计算          │  每层的计算与其后层的
  ├─ …                                              │  RDMA 读在时间上重叠
  └─ hook[L-1].wait() ◀─────────────────────────────┘
```

"逐层等待"与"一次全部等待"的差异在数学上就是流水线定理：若每层数据到达时间为 $a_l$、计算耗时 $c_l$，则逐层钩子的完成时刻为 $\max_l (a_l + \sum_{l'\le l} c_{l'})$ 的量级（计算与传输重叠），而整体等待为 $\max_l a_l + \sum_l c_l$（串行）。层数越多、重叠收益越大。

注意钩子的语义：**钩子是"该层数据可用"的同步点**，不负责数据消费；且 README 强调 BF16、FP8 与 CPU 存储的完整示例见 `tests/engram/test_engram.py`。

## 8.5 workspace：issue_counters 的"最后到达者清零"协议

`deep_ep/include/deep_ep/layout/engram/workspace.cuh` 定义了 Engram 的同步骨架：

```cpp
struct alignas(kNumRDMAAlignmentBytes) EngramWorkspace {
    static constexpr int kNumMaxLayers = 128;         // 支持的最大层数
    static constexpr int kNumMaxPeerContexts = 2048;  // 每 (对端, 上下文) 组合的计数器上限

    BarrierSignals barrier;                            // 公共屏障信号
    uint32_t issue_counters[kNumMaxLayers][kNumMaxPeerContexts];
    // 发起屏障计数器：issue warp 在每层结束时用原子加计数，
    // 最后一个到达的 warp 负责 flush QP 并把计数清零，其余 warp 等待清零后再进下一层
};
```

这段注释描述了一个精巧的 **QP 批量提交门控**协议，值得完整推演：

1. 多个 issue warp 并行发起某一层的 RDMA 读请求（写入各自 QP 的发送队列）；
2. 每个 warp 完成本层发起后，对其 (layer, peer, context) 槽位原子加一（`issue_counters`）；
3. **最后一个到达的 warp**（计数达到参与 warp 数）负责调用 QP 的 **flush**——把队列中积累的请求真正推向网络——然后把计数清零；
4. 其余 warp 自旋等待"计数被清零"作为"本层发起全部完成、QP 已冲刷"的信号，再进入下一层。

为什么需要 flush 门控？RDMA 队列的提交是有成本的（门铃/队列推进），每个 warp 每发起一条请求就推进一次队列会产生大量小请求与队列竞争；批量提交把"一层的全部请求"合并成一次队列推进，既降低网络协议开销，又保证层内请求的顺序性（同层请求按队列序投递）。而"最后到达者清零"是把"两相位屏障"思想（第 10 章）压缩进一个计数器的实现：清零动作本身就是"上一相位完成"的信号，免去了独立的相位标志。

`kNumMaxPeerContexts = 2048` 的"对端 × 上下文"二维上限说明协议支持每个 warp 组与每个对端独立计数——规模再次以编译期常量形式封顶，与 `EPSignals` 的风格一致。

## 8.6 存储介质：GPU 分片与 CPU 分片

Engram 的分片可以是 GPU 显存，也可以是 **CPU 内存**（类注释：*"separate CPU or GPU storage shards"*）。两者的带宽模型完全不同：

- **GPU 分片**：远端 GPU 显存 → 本地 GPU，走 GPU↔NIC↔GPU 的 RDMA，全程 GPU 对称内存可达；
- **CPU 分片**：远端主机内存 → 本地 GPU。CPU 分片需锁页注册（可被 NIC 直接访问），读取时 NIC DMA 从主机内存搬到网络；延迟更高（要经过 PCIe 上行），但容量可以是 TiB 级——KV 缓存卸载、超大规模嵌入表的典型分层形态。

`get_storage_size_hint(group, …)` 与 `get_theoretical_config(group, …)` 这两个静态方法（`deep_ep/buffers/engram.py`）负责"算账"：前者估算需要的注册存储字节，后者返回理论配置（QP 数、分片切分等）——与 EP 的 `get_buffer_size_hint`/`get_theoretical_num_sms` 同一哲学：**先解析式估算，再真实构造**。

CPU 分片场景还牵出一个环境变量：`EP_NUM_MAX_LOCAL_RANKS`（默认 16）——"仅限 Engram：在混合模式下为 `NCCL_WIN_STRIDE` 的大小设置估算已注册存储。这只是一个容量估算，并非 rank 数量上限"（README 环境变量表）。Gin 的窗口注册要求按固定步长（stride）预留窗口槽数；DeepEP 以"每节点最多 16 个本地 rank"作为默认估算，构造时按此分配窗口数组（`rdma_storage_windows_gpu` 数量 = `gin_min_stride`）。

## 8.7 与 EP 路径的对比：读取向 vs 推送向

| | EPBuffer dispatch | EngramBuffer fetch |
| --- | --- | --- |
| 数据流方向 | 发送端主动推送（put 型 all-to-all） | 读取端主动拉取（get 型） |
| 目的地确定 | 由门控 topk 决定，一 token 多目的地 | 由索引决定，一条目一目的地 |
| 归约需求 | combine 需要加权归约 | 无归约，纯读取 |
| 同步原语 | notify 计数 + 槽位分配 | GIN 请求句柄 + 逐层钩子 |
| 规模控制 | SM/QP 解析式估算 | `get_theoretical_config` + `NCCL_WIN_STRIDE` 估算 |
| 典型延迟 | 层内关键路径（每层必经） | 可被逐层流水隐藏 |

这张对照表解释了为什么两者必须是两个缓冲区类而不是一个"万能 all-to-all"：推送向协议的核心资产是"槽位分配与计数器"，拉取向协议的核心资产是"请求句柄与分层流水"，合并只会让两者的同步原语互相污染。

## 8.8 本章小结

- Engram 解决"参数大过显存、访问按索引稀疏"的取数问题：分片（GPU/CPU）RDMA 注册一次，`fetch(indices)` 按 `[num_layers, num_tokens, num_entries_per_token]` 索引拉取，逐层钩子驱动流水消费。
- 内核用 GIN 异步请求（`ncclGinRequest_t`）分离发起与等待（`engram_fetch.cuh` / `engram_fetch_wait.cuh`），scale factor 独立分片按三维 stride 取回，层表编译进模板。
- `issue_counters[kNumMaxLayers][kNumMaxPeerContexts]` 实现"最后到达者 flush QP 并清零"的门控协议：批量提交 RDMA 请求、层间推进以清零为信号。
- 存储介质分层（GPU 快、CPU 大），`get_storage_size_hint`/`get_theoretical_config` 解析式规划，`EP_NUM_MAX_LOCAL_RANKS` 默认 16 估算 `NCCL_WIN_STRIDE` 注册规模；Engram 需要 NCCL 提供 `ncclGinOptFlagsWarpGet` 能力。
- 与 EP 的对照：推送向（put、归约、计数）与拉取向（get、流水、请求句柄）是两种协议形态，各自成类。

## 8.9 延伸讨论：取数的成本模型与分片策略

Engram 的性能取决于三个可调量：分片的放置（GPU 还是 CPU）、条目的划分（每 rank 分多少）、以及取数的批量（一次 fetch 覆盖多少层与 token）。本节给出一个把三者联系起来的成本模型。

**单条目取数的延迟分解**。一次 RDMA read 的完成时间近似为 $t_{\text{entry}} \approx t_{\text{issue}} + t_{\text{rtt}} + V_{\text{entry}} / B$，其中 $t_{\text{issue}}$ 是内核发起与 QP 提交的开销（被 8.5 节的 flush 门控摊薄）、$t_{\text{rtt}}$ 是网络往返、$V_{\text{entry}}/B$ 是传输时间。Engram 的关键优势在于**大批量同质请求**：一次 fetch 发起 $L \times T \times n_{\text{et}}$ 条读取（$L$ 层、$T$ token、每 token $n_{\text{et}}$ 条目），同一 QP 上的请求按队列序连发，往返延迟被流水掩盖，稳态吞吐趋近于 $B$（链路带宽）而非单请求速率。这解释了 8.3 节"发起与等待分离"的设计动机：发起阶段的目标是把队列填满，等待阶段的目标是按层收割。

**分片放置的选择**。GPU 分片的 $t_{\text{rtt}}$ 低（NIC 与 GPU 同机或近机）、$B$ 高，适合"每步都要取、条目较小"的负载——Wide-EP 的专家权重按需取是典型；CPU 分片牺牲延迟（多一跳 PCIe 与主机内存带宽）换取容量（TiB 级、成本每 GB 更低），适合"命中率分散、单条目大"的负载——KV 缓存与超大嵌入表。`get_storage_size_hint` 与 `get_theoretical_config`（`deep_ep/buffers/engram.py` 的两个静态方法）把"容量与 QP 规模"的算术前置到构造前，集成方应当先用它们做纸上规划，再决定放置层级——这也是 README 的使用顺序（"使用 `get_storage_size_hint` 和 `get_theoretical_config` 来确定缓冲区及其 QP 的规模，然后使用 `set_config` 和 `write` 填充层表"）的用意。

**条目划分与热点**。条目到分片的映射决定取数流量的空间分布：若热门条目聚集在少数 rank 的分片上，那些分片属主的 NIC 成为瓶颈，链路带宽 $B$ 退化为其上行带宽的极小份额。因此条目划分应当尽量打散（与 EP 专家放置同理），或对热点条目在多分片冗余存放——后者与第 7 章的专家复制是同一个思想在 Engram 上的投影。`num_entries_per_layer` 编译为模板参数列表（8.3 节）意味着层表在 `set_config` 后是静态的，划分策略的调整需要重新配置缓冲区，这把"划分"从运行时决策提升为部署决策——合理，因为划分错了的代价是持续的带宽损失，值得一次性算清。

**层表与流水深度**。逐层钩子（8.4 节）的收益取决于每层计算与传输的重叠程度：若第 $l$ 层的计算时间 $c_l$ 大于第 $l+1$ 层的到达等待 $w_{l+1}$，流水完全隐藏传输；反之则每层出现气泡。把层的顺序按"计算重、取数轻"的先发原则排列（第 0 层先发起全部请求），可以让气泡集中到流水头部而非每层重复。集成方对层表的自由度（哪些层纳入 Engram、每层条目数）就是流水的调节旋钮。

## 8.10 延伸讨论：窗口步长与注册的工程含义

第 8.6 节提到 `EP_NUM_MAX_LOCAL_RANKS`（默认 16）为 `NCCL_WIN_STRIDE` 提供估算，本节解释"窗口步长"为什么存在以及它对部署的影响。Gin 的窗口注册体系要求为每个可能的对端上下文预留窗口槽位，槽位数量以固定步长（stride）组织——这就是 `Context` 构造时 `rdma_storage_windows_gpu` 数组的大小来自 `gin_min_stride`（第 8.3 节）的原因。步长按"每节点本地 rank 数"估算（环境变量默认 16），意味着：若单节点 rank 数超过这个估算（例如单机八卡以上配置多进程 per GPU，或未来单机 16 卡以上），注册存储的估算会偏小，表现为构造时的容量断言或运行时的窗口越界——修正方法是调大 `EP_NUM_MAX_LOCAL_RANKS`（README 环境变量表明确"这只是一个容量估算，并非 rank 数量上限"）。反过来，估算过大的代价是窗口数组占用的设备内存与注册时间线性增长。这个"估算而非精确"的设计与解析式哲学（第 6 章）一脉相承：窗口步长属于平台协议约束（Gin 侧要求），DeepEP 以保守默认覆盖主流拓扑、以环境变量开放极端拓扑，避免了为每个部署精确探测的复杂度。

对部署的实践含义有三：其一，Engram 分片的数量规划要把窗口槽数计入（每分片一个窗口，数组大小为步长，第 8.3 节）；其二，CPU 分片与 GPU 分片混用时窗口注册在同一数组内组织，`set_config` 的层表变更不影响窗口结构（注册一次、随取随用，第 8.2 节）；其三，窗口步长的估算发生在导入与构造期，部署清单里应把 `EP_NUM_MAX_LOCAL_RANKS` 与节点拓扑的核对列为固定项（第 14 章初始化检查单）。

## 8.11 延伸讨论：Engram 与 KV 缓存传输的关系

第 8.1 节把"KV 缓存卸载"列为 Engram 的典型场景之一，本节展开这个场景的特殊性，因为它与专家参数取数在访问模式上有本质差异。专家参数取数的索引分布由门控决定，热点集中在少数条目，批量内重复命中率高；KV 缓存取数的索引由注意力模式决定，条目（历史 token 的键值）在批次内几乎不重复，且单条目尺寸小（每个 token 的 KV 对是 hidden 维的一对向量）。两者的差异决定了调优重点不同：前者吃带宽（热点条目重复拉取的总量），后者吃延迟与小消息效率（海量不重复的小读）。Engram 的机制为两者都留了空间——批量 fetch 的流水（第 8.4 节）对小消息场景尤其重要，因为 RDMA 小消息的固定开销只有靠并发流水摊薄；而条目尺寸大、重复率高的场景则更受益于 8.9 节的分片放置与热点冗余。集成方评估"KV 走 Engram 还是走专用 KV 传输层"时，可以按本节的两个维度（重复率、条目尺寸）判断：高重复、大条目适合 Engram 的批量取数；低重复、小条目、强时序的场景则可能需要为 KV 定制的聚合协议——这也是为什么 Engram 在 README 中定位为"实验性"且场景开放（"典型用途是跨节点按索引拉取大表/专家参数"，第 8 章），而把 KV 专属优化留给上层与社区分支探索。

## 8.12 延伸讨论：与 EPBuffer 的组合使用模式

Engram 与 EPBuffer 并非二选一，Wide-EP 类部署的正确形态往往是两者组合——本节描述两种典型组合模式及其容量核算。模式一，**常驻加取数分层**：高频专家常驻本地（EPBuffer 的标准 dispatch 路径），低频专家放远端分片（Engram 按需拉取）——门控输出先查常驻表，未命中走 `fetch`；容量核算按"常驻专家的 dispatch 容量 + 取数缓冲的批量"分别做（第 19.3 节五项分解），两者的比例由负载的长尾分布决定（第 1.10 节 $\lambda$ 分布）。模式二，**权重与激活的双路径**：激活走 dispatch（推送向，小而频），专家权重走 Engram（拉取向，大而稀疏）——这正是第 8.8 节对照表"推送向对拉取向"的工程落地：同一层内两种协议并存，各走各的 QP 与窗口（Context 的窗口数组与主窗口分离，第 8.3 节）。组合时的三个协同点：其一，时序协同——取数的逐层钩子（第 8.4 节）要与 dispatch 的事件流（第 10 章）在同一通信流上排队（第 10.11 节单流推论），等待点的设计要避免两类等待互相饿死；其二，容量协同——取数缓冲 `fetched` 与 dispatch 接收槽位共享显存预算（第 19.8 节分配优先级），分片放置（GPU 或 CPU）决定跨层带宽分配；其三，故障协同——两者的失败语义一致（集合超时或请求超时，第 10.13 节），框架的容错处理统一接入。组合模式的价值在于展示 DeepEP 的"原语可组合性"：四个缓冲区不是四个孤岛，而是同一资源模型上的四种通信原语——这正是第 20.6 节家族化拆分"共享地基"的设计意图在部署形态上的回响。

下一章补齐剩下的两块拼图：流水线点对点（PPBuffer）与批量集合通信（BucketBuffer）。
