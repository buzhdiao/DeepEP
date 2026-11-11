# 第 01 章 · 专家并行与 MoE all-to-all 通信问题

## 1.1 从稠密 Transformer 到稀疏 MoE

要理解 DeepEP 为什么存在，必须先理解 MoE（Mixture of Experts，混合专家）把"计算"和"参数"拆开之后，掉出来一个什么样的通信问题。

一个标准 Transformer 块中的前馈网络（FFN）是稠密算子：对每个 token $x \in \mathbb{R}^{h}$（$h$ 为隐藏维度），计算

$$
\mathrm{FFN}(x) = W_2 \, \sigma(W_1 x), \qquad W_1 \in \mathbb{R}^{f \times h},\ W_2 \in \mathbb{R}^{h \times f},
$$

其中 $f$ 是中间维度。每个 token 都要穿过同一组参数，参数量为 $2hf$，计算量为每 token $O(2hf)$ 次乘加。

MoE 用一组专家网络 $\{E_1, \dots, E_E\}$ 替换这一个 FFN，并用一个门控（router/gate）为每个 token 挑选 top-$k$ 个专家：

$$
y = \sum_{i \in \mathrm{TopK}(g(x))} g_i(x) \cdot E_i(x),
$$

其中 $g: \mathbb{R}^h \to \mathbb{R}^E$ 通常是一个线性层加 softmax，$g_i(x)$ 是第 $i$ 个专家的门控权重（因此 $\sum_{i \in \mathrm{TopK}} g_i(x)$ 往往被归一化为 1）。这一替换带来一个决定性的解耦：

- **参数量**正比于 $E$：$E \cdot 2hf$。把专家总数 $E$ 从 8 提到 256，模型"知识容量"线性放大。
- **每 token 计算量**只正比于 $k$：$k \cdot 2hf$。每个 token 只激活 top-$k$ 个专家，$k$ 通常为 2~8。

| 模型形态 | 每层参数量 | 每 token 每 FFN 计算量 | 每 token 实际激活参数 |
| --- | --- | --- | --- |
| 稠密 FFN | $2hf$ | $2hf$ | $2hf$ |
| MoE（$E$ 个专家，top-$k$） | $E \cdot 2hf$ | $k \cdot 2hf$ | $k \cdot 2hf$ |

"参数量与计算量解耦"是大模型扩容的关键杠杆，但它有一个隐含前提：**每个 token 必须能找到自己被分配的那 $k$ 个专家**。在单卡上这不是问题；一旦模型大到专家必须摊到几十甚至上百张卡上，"找专家"就变成了"跨设备通信"。这个通信就是本章的主角。

## 1.2 专家并行：把专家摊到设备上

**专家并行（expert parallelism，EP）**的定义朴素而彻底：把 $E$ 个专家均匀分配到 $P$ 个设备（rank）上，每个 rank 持有 $E/P$ 个专家（源码注释与参考实现都假设整除：`deep_ep/utils/refs.py` 的 `dispatch` 里就有 `assert num_experts % num_ranks == 0` 与 `num_experts_per_rank = num_experts // num_ranks`）。token 的表示（激活值）仍然按数据并行的方式切分——每个 rank 上有一批 token 的隐藏状态——但专家权重是 EP 维度上不重叠的。

于是一个 MoE 层的前向计算变成了这样的流水：

1. **门控**（本 rank 上）：对本地 $T_r$ 个 token 计算路由，得到 `topk_idx`（形状 $[T_r, k]$，元素为全局专家编号，`-1` 表示未选满）与 `topk_weights`（形状 $[T_r, k]$）。
2. **dispatch**：把"token 隐藏状态 + 路由元数据"发送给目标专家所在的 rank。一个 token 若被 $k$ 个专家选中、而这 $k$ 个专家分布在不同 rank 上，它的数据就要被复制发送到多个 rank。
3. **专家计算**：每个 rank 对"路由到本地专家"的 token 做分组专家 GEMM（grouped expert GEMM）。
4. **combine**：把专家输出沿反向路由送回 token 原属 rank，并按门控权重加权求和。

步骤 2 和步骤 4 是通信，而且它们不是"相邻 rank 交换"或"树形归约"这类规整的集合通信模式——任何一个 rank 都可能向任何一个 rank 发送 token，也可能从任何一个 rank 接收 token。这是一种 **all-to-all** 通信模式：$P \times P$ 的全互联数据交换。

```text
        MoE 层前向中的 all-to-all（P = 4，E/P = 2 个专家/rank）

  rank0 [E0 E1]   rank1 [E2 E3]   rank2 [E4 E5]   rank3 [E6 E7]
    │  token 集合 T0   │               │               │
    │  topk: {0,5}     │  topk: {3,4}  │  topk: {5,7}  │
    │                  │               │               │
    ├──────── dispatch（每 token 按 topk 目标 rank 复制发送）──────▶
    │        T0 中指向 E2..E7 的 (token,expert) 对流向 rank1/2/3
    │
    ▼ 专家计算（各 rank 分组 GEMM，只算落在本地专家上的行）
    │
    ◀──────── combine（结果按路由逆序送回，接收端加权求和）────────
```

与张量并行（TP）的 all-reduce 相比，all-to-all 有三个更"难缠"的性质：

1. **无规整模式**：通信量分布由路由结果决定，每个 token 各不相同，事先只能估计上界；接收方在收到元数据之前不知道自己会收到多少数据。
2. **尺寸动态**：每个 rank 每 层的接收 token 数是随批次的随机变量，输出张量的形状在 CPU 侧"数完"之前是未知的——这直接导致"要不要 CPU 同步"这一贯穿 DeepEP API 的设计难题（第 4 章展开）。
3. **一 token 多目的地**：top-$k$ 把同一 token 的数据复制到最多 $k$ 个目的地，通信量不是 $O(T)$ 而是 $O(T \cdot k)$ 量级。

## 1.3 通信量的定量分析

设一个 EP 组有 $P$ 个 rank、$E$ 个专家、每 rank 批次 $T$ 个 token、top-$k$、隐藏维度 $h$、激活值元素字节数 $b$（BF16 为 2，FP8 为 1 加上 scale factor）。假设门控均衡（每个 token 的 $k$ 个专家均匀散布在 $P$ 个 rank 上），则：

- 每 token 产生 $k$ 个 $(\text{token}, \text{expert})$ 对；
- 对落在"属主 rank"的专家不需要通信，落在其他 $P-1$ 个 rank 的比例是 $(P-1)/P$；
- 故每 token 平均跨设备传输的**数据份数**为 $k \cdot \frac{P-1}{P}$。

于是 dispatch 阶段每 rank 的发送数据量为

$$
V_{\text{dispatch}} \;\approx\; T \cdot k \cdot \frac{P-1}{P} \cdot (h \cdot b + b_{\text{meta}}),
$$

其中 $b_{\text{meta}}$ 是随行元数据（topk 索引、权重、来源信息等）的摊销字节数。combine 阶段对称地把专家输出送回去，通信量同量级（BF16 输出，$b=2$）。一层 MoE 前向的总通信量为

$$
V_{\text{fwd}} = V_{\text{dispatch}} + V_{\text{combine}} \approx T \cdot k \cdot \frac{P-1}{P} \cdot h \cdot (b_{\text{in}} + b_{\text{out}}).
$$

再看反向传播。记前向 dispatch 的输出（专家看到的 token）为 $X_{\text{exp}}$，combine 的输出为 $Y$。根据链式法则：

- $\partial L/\partial Y$ 需要被**发送回各个专家**以计算专家参数梯度——这与 dispatch 的数据流完全一致（"反向的 combine 是 dispatch"）；
- 专家算出的 $\partial L/\partial X_{\text{exp}}$ 需要被**送回原 rank 加权求和**得到 $\partial L/\partial x$——这与 combine 的数据流完全一致（"反向的 dispatch 是 combine"）。

README 的训练示例把这一点写得很直白：`dispatch_backward` 就是调用 `_buffer.combine(...)`，`combine_backward` 就是调用 `_buffer.dispatch(...)`（传入 `handle` 复用前向缓存的布局）。因此**一层 MoE 一个训练 micro-batch 共有 4 次同量级的 all-to-all**：

$$
V_{\text{layer}} \approx 2 \cdot T \cdot k \cdot \frac{P-1}{P} \cdot h \cdot (b_{\text{in}} + b_{\text{out}}) \times 2 \text{（前向+反向）}.
$$

以一个具体数字感受量级：$T = 4096$，$k = 8$，$h = 4096$，$b_{\text{in}} = 2$（BF16）且 FP8 dispatch 时 $b_{\text{in}} = 1 + \text{scale}$，$b_{\text{out}} = 2$，$P = 64$。则

$$
V_{\text{dispatch}} \approx 4096 \times 8 \times \frac{63}{64} \times 4096 \times 2 \,\text{B} \approx 4.0\,\text{GiB},
$$

一层前向 dispatch + combine 接近 $8\,\text{GiB}$，一个 60 层模型一个 micro-batch 仅 EP 通信就是数百 GiB 的量级。这个规模意味着：**通信带宽成为 MoE 训练/推理的一等公民资源**，任何让通信"停在 CPU 等待"或"串行于计算"的设计都会被放大成端到端性能损失。

## 1.4 三面墙：带宽墙、延迟墙、SM 墙

有了量级，就可以把约束说清楚。DeepEP 要同时翻越三面墙。

### 1.4.1 带宽墙

设节点内 NVLink 聚合带宽为 $B_{\text{nvl}}$、跨节点 RDMA 带宽为 $B_{\text{rdma}}$（DeepEP 通过 `deep_ep/utils/envs.py` 的 `get_nvlink_gbs` 与 `get_rdma_gbs` 自动探测，分别解析 `nvidia-smi` 与 `ibstat` 的输出）。一次 all-to-all 中，数据按目的地是否跨节点分成两部分：

$$
T_{\text{comm}} \;\ge\; \max\!\left( \frac{V_{\text{scale-out}}}{B_{\text{rdma}}},\; \frac{V_{\text{scale-up}}}{B_{\text{nvlink}}},\; \frac{V_{\text{rd}} + V_{\text{wr}}}{B_{\text{hbm,agg}}} \right),
$$

最后一项是 SM 从 HBM 读出、写回数据的时间——通信本质上是"读本地内存 + 走网络 + 写本地（或对端）内存"，SM 数量决定了 HBM 读写带宽的聚合值 $B_{\text{hbm,agg}}$（DeepEP 用 `get_sm_read_gbs` / `get_sm_write_gbs` 建模每 SM 的读写速率）。这个 $\max$ 结构正是第 6 章 `get_theoretical_num_sms` 解析式估算的原型：SM 太少则 HBM 搬运成为瓶颈，SM 太多则挤占计算——因此要"解析地"找到平衡点，而不是跑一遍自动调优。

### 1.4.2 延迟墙

训练与 prefill 的批量大（数千 token），带宽主导；但**推理解码（decode）**每步每 rank 往往只有一两个 token，此时通信时间被往返延迟（RTT）而非带宽主导：

$$
T_{\text{decode}} \approx \underbrace{N_{\text{msg}} \cdot t_{\text{launch}}}_{\text{内核/消息启动}} + \underbrace{N_{\text{rtt}} \cdot t_{\text{rtt}}}_{\text{网络往返}} + \underbrace{\frac{V}{B}}_{\text{传输}} .
$$

RDMA 写本身可以做到"fire-and-forget"，但**接收端必须知道数据到了**——V1 时代的低延迟协议依赖 RDMA 原子操作（atomic add）作为完成信号，而原子操作引入一次额外的 RTT（README 的实验性 Eager 分支 PR #437 的卖点正是"消除由 RDMA 原子操作引入的额外 RTT 延迟"）。V2.5 用 NCCL Gin 的设备端 API 统一了高吞吐与低延迟两种场景（"高吞吐与低延迟 API 统一为单一的 `EPBuffer` 接口"），让同一段 dispatch/combine 代码在小批量下自动偏向低延迟路径、在大批量下偏向满带宽路径。

### 1.4.3 SM 墙

GPU 上没有免费的通信：把数据从显存搬到网卡/对端显存，要么占用 SM（跑拷贝/同步指令），要么占用拷贝引擎（CE，但 CE 不适合做细粒度同步与归约）。若通信内核吃掉全部 SM，专家计算就得排队——**通信与计算的重叠（overlap）就无从谈起**。因此 DeepEP 的所有通信内核都把"占多少 SM"作为一等配置项：

- `EPBuffer.__init__` 有 `prefer_overlap_with_compute`（"若启用，我们倾向于使用更少的 SM"）；
- `dispatch`/`combine` 的 `num_sms` 参数默认为 0，表示按 `get_theoretical_num_sms` 的解析式估算取值；
- Bucket 的 all-gather 甚至可以完全不用 SM（由拷贝引擎驱动，`num_sms=0`，见第 9 章）。

三面墙合并成一个总目标函数：

$$
\min_{\text{SM 数},\ \text{QP 数},\ \text{路径},\ \text{布局}} \; T_{\text{MoE 层}} \quad \text{s.t.} \quad \text{SM 保留给专家计算},\ \text{数值精度},\ \text{确定性（可选）}.
$$

DeepEP 的答案不是运行时搜索，而是把每个变量都变成**可解析计算 + 可手动覆盖**的量：SM 数与 QP 数有理论估算公式（`get_theoretical_num_sms`、`get_theoretical_num_qps`），路径由拓扑自动选择（scale-up 直达 / hybrid 两级 / 纯 RDMA），布局由 MoE 配置算出（`calculate_ep_buffer_size`）。

## 1.5 为什么"通用集合通信库"不够用

既然 all-to-all 能用 NCCL/torch.distributed 表达，为什么还要 DeepEP？把痛点列全，后面的章节就是在逐条回答这张清单。

**（1）输出尺寸不可预知，逼出 CPU 同步。** `torch.distributed.all_to_all_single` 要求调用方预先告知发送/接收切分。MoE 的接收数量依赖路由结果，传统做法是"先发一次小的元数据 all-to-all（CPU 同步拿到计数），再发数据"——两次主机侧同步把 CUDA 流打断了，也与异步流水线天然冲突。DeepEP 的做法是把计数与数据放进**同一个设备端内核**：notify warp 在 GPU 上原子地累加接收计数（`csrc/kernels/ep/dispatch.hpp` 的 `get_num_notify_smem_bytes`、`layout/ep/workspace.cuh` 的 `notify_reduction[kNumMaxRanks + kNumMaxExperts]`），CPU 只有在需要精确尺寸时才同步（`do_cpu_sync=True`，训练/prefill 用），解码则直接用 GPU 侧计数配合容量分配（`do_cpu_sync=False`，专家 GEMM 用 `handle.psum_num_recv_tokens_per_expert` 确定有效范围）。

**（2）主机编排无法与计算重叠。** NCCL 集合通信由主机线程入队，通常独占其使用的 SM 与流。DeepEP 通信内核跑在独立的高优先级通信流上（`csrc/comm/stream.hpp` 的 `get_comm_stream`，`at::cuda::getStreamFromPool(true)`），调用返回 `EventOverlap`，允许用户"先发起通信 → 做独立计算 → `current_stream_wait()` 等结果"，配合 `defer_epilogue=True` 把 epilogue 也推迟到等待点执行（README 的"延迟 epilogue"一节）。

**（3）跨节点路径需要两级路由。** 跨节点用 RDMA、节点内用 NVLink，通用 all-to-all 把两层网络抽象成一个扁平通信域，浪费 NVLink 的近距带宽、又把所有跨节点流量压到单一路径。DeepEP 显式建模 scale-up/scale-out 两层域（`get_logical_domain_size` 返回 scale-out/scale-up 维度），hybrid 模式先经 RDMA 到目标节点、再经 NVLink 分发（`impls/ep/hybrid_dispatch.cuh`），对多平面/多轨（multi-rail）网络更友好。

**（4）部署复杂度。** V1 依赖 NVSHMEM，意味着额外的库、额外的通信域 bootstrap、与 PyTorch/NCCL 并存的内存注册管理。V2 改用 **NCCL Gin 后端**：设备端 API（`nccl_device.h` 的 `ncclWindow_t`、`ncclDevComm_t`、`ncclGinRequest_t` 等类型在 `csrc/kernels/comm/api.hpp` 与 `deep_ep/include/deep_ep/comm/handle.cuh` 中被广泛使用），能复用已有的 NCCL 通信域（`EP_REUSE_NCCL_COMM=1` 时直接取 PyTorch 进程组的 `_comm_ptr`，见 `deep_ep/comm/handle.py` 的 `get_nccl_comm_handle`），NVSHMEM 不再是依赖项。

**（5）数值与调试需求。** FP8 dispatch（`use_fp8_dispatch=True`，输入变为 `(fp8 张量, scale factor)` 二元组）、`deterministic` 确定性路由（`EPHandle.deterministic_sort`）、`cumulative_local_expert_recv_stats` 在线负载统计——这些都是通用库不提供的 MoE 专属需求。

这张需求清单正是 DeepEP API 形态的"因"：一个以**缓冲区对象**（`EPBuffer`）为中心的接口——缓冲区在初始化时按 MoE 配置与拓扑分配好对称内存、建立通信域与 QP；每次 dispatch/combine 只传张量与少量参数；返回句柄与事件支撑复用与重叠。

## 1.6 约束清单：DeepEP 必须同时满足的八件事

把 1.4/1.5 节的讨论收拢成一张设计约束表，后续章节会不断回看它：

| # | 约束 | 来源 | DeepEP 的回应（章节） |
| --- | --- | --- | --- |
| 1 | 高吞吐：大批量时吃满 RDMA/NVLink 带宽 | 1.4.1 | 两级路由 + 通道化内核 + 解析式 SM 数（第 4、6 章） |
| 2 | 低延迟：小批量 decode 时压 RTT | 1.4.2 | 统一 EPBuffer 接口、设备端同步、原子信号优化（第 4、10 章） |
| 3 | 通信与计算重叠 | 1.4.3 | 独立通信流 + EventOverlap + 延迟 epilogue + SM 预算（第 10 章） |
| 4 | 动态 shape 与精确接收计数 | 1.5(1) | GPU 端 notify 计数、可选 CPU 同步、句柄缓存（第 4 章） |
| 5 | 数值：FP8 dispatch、BF16 combine、可缩放归约 | 1.5(5) | TokenLayout 内嵌 scale factor 槽位（第 4 章）、combine 归约布局（第 5 章） |
| 6 | 更大的 scale-up/scale-out 域 | 1.5(3) | 混合模式与解析式规模上限（RDMA 域 ≤128、总 rank ≤1024 等）（第 6 章） |
| 7 | 部署简单：复用 NCCL 生态 | 1.5(4) | NCCL Gin 后端、运行时 DeepJIT 编译（第 11 章） |
| 8 | 专业化原语：EPLB 权重/梯度交换、Engram 取数、PP 点对点、Bucket 集合通信 | 1.5(5) | 四大缓冲区家族（第 7~9 章） |

其中第 8 条值得强调：MoE 的通信问题不止 dispatch/combine。热点专家需要跨 rank 复制（第 7 章）；有的参数干脆不值得常驻每张卡、应按需远程取数（第 8 章）；流水线并行需要极薄的点对点 send/recv（第 9 章）；CP/DP 需要批量化的 all-gather/reduce-scatter/all-reduce（第 9 章）。V2.5 把这些统一进四个缓冲区类，共享 `BufferBase` 生命周期与 `BufferAllocator` 布局规划——这是第 2、3 章的主题。

## 1.7 一个重要的现实约束：通信内核必须有 SM

README"注意事项"里有一条容易被忽略的硬约束："EP 的 dispatch 与 combine 需要 GPU SM；不支持零 SM 的 RDMA EP"。原因在 1.4.3 已经埋下伏笔：EP 的 all-to-all 不是纯网络搬运输，接收端的**计数、槽位分配、元数据整理、归约**都需要设备端逻辑，这些逻辑跑在 SM 上。CE（拷贝引擎）只能搬运不能计数归约，所以 EP 路径注定要"以 SM 换带宽"，也才有了"解析式估算最优 SM 数"这个 DeepEP 特色（对比：Bucket 的 all-gather 恰恰因为不需要归约与计数，才能完全交给 CE）。这条约束也解释了 AntGroup-Opt 系列分支为什么把"SM-free"作为卖点——Normal-SMFree（PR #347）通过把通信内核执行与网卡 token 传输解耦来释放 SM，本质是在 DeepEP 的框架内重新划分"谁干哪一步活"。

## 1.8 延伸讨论：为什么 all-to-all 是"最难优化的集合通信"

把 all-to-all 放进集合通信的代数里比较，可以看清它的难处从何而来。常见的集合通信可以按"每个参与方的输入如何变成输出"来分类：broadcast 是一对多复制，reduce 是多对一聚合，all-gather 是多对多复制，reduce-scatter 是多对多聚合加切分，all-reduce 是两者的组合。这些模式的共同特征是**数据流向由模式本身决定**：给定参与方集合与数据尺寸，每条链路上的流量、每个节点的收发量、以及一条最优的树形或环形调度，都可以在事先完全确定。all-to-all 则不同——它的流量矩阵是满的，但矩阵的每个元素（谁给谁发多少）不由模式决定，而由应用语义决定；对 MoE 而言就是由门控的输出决定。这带来三个层次的优化难题。

第一，**静态调度无法预先确定**。树形归约可以提前规划好"哪两两归约、在哪一层交换"，因为流量模式是常量；all-to-all 的流量矩阵每个微批次都在变，调度必须"数据驱动"——这正是 DeepEP 把计数放进设备端内核（notify warp）、把槽位分配做成前缀和（第 4 章）的根本原因：让调度决策在 GPU 上以数据为准即时完成，而不是回到 CPU 重新规划。

第二，**负载均衡是内在的随机变量**。规整集合通信中每个参与方的收发量相同（reduce-scatter 各拿一份，all-reduce 各发一份），网络可以利用这种规整性做条带化；all-to-all 中每个参与方的收发量随门控波动，热点链路会周期性出现。缓解手段有二：其一在门控侧控制分布（`tests/utils/test_gate.py` 的 ratio 参数正是对这一维度的模拟），其二在传输侧用通道与多 QP 把热点摊开（第 4 章通道化、第 6 章 QP 并发），其三在专家侧复制热点（第 7 章 EPLB）。三个手段分别在门控、传输、计算三个层面应对同一个随机变量——这也是为什么本书反复强调"MoE 通信问题是一个系统级问题"。

第三，**归约与路由的耦合**。reduce-scatter 的归约发生在"每份数据的目标地"（第 p 份归到 rank p），目标地由模式决定；all-to-all 的归约发生在"token 的出发地"，而每个 token 的出发地要到接收端整理路由元数据时才知道。combine 的"先按来源组织、接收端一次归约"（第 5 章 `kDoExpandedSend`）正是这个耦合的直接产物：路由与归约谁先谁后，是一个真正的设计自由度，而 DeepEP 把它做成了编译期开关让用户按精度与流量权衡选择。

## 1.9 延伸讨论：通信时间的三种极限与 DeepEP 的位置

为了给后续章节的优化留出"天花板"，本节把一次 all-to-all 的时间极限分三层讨论。下界是**网络极限**：数据一旦离开节点，时间不会低于流量除以网络可用带宽（对跨节点部分）或 NVLink 带宽（对节点内部分）；任何软件优化只能逼近这条线，不能突破它——这也是第 13 章调优清单把网络侧配置放在第一位的原因：如果网络没配好（SL 隔离缺失、拥塞未控、原子模式未开），软件再好也在一条打折的带宽线上工作。

第二条线是**同步极限**：即便带宽无限，通信仍受制于"接收方知道数据何时到"的同步开销。同步可以在三个位置做：CPU 轮询（最慢，打断流水）、设备端自旋（快，占 SM）、硬件信号（最快，mbarrier 与网络原子）。DeepEP 的位置在设备端与硬件信号一端：notify warp 的原子计数、mbarrier 的相位等待、Gin 的原子完成信号（第 4、10 章），把同步开销压到"内核内自旋 + 硬件通知"的量级，避免了 CPU 往返。Eager 分支消除原子 RTT（第 17 章）则是在这条线上再往硬件方向推一步的尝试。

第三条线是**重叠极限**：通信时间若能与计算时间完全重叠，对端到端时延的净贡献趋近于零——条件是通信内核与计算内核不争抢资源（SM、HBM 带宽、网络），且两者的依赖链允许交错。DeepEP 为此提供的三件套是：独立高优先级通信流（调度隔离）、SM 预算（资源隔离，`prefer_overlap_with_compute`）、延迟 epilogue（依赖链重组，把"整理数据"这类必须串行的部分推迟到等待点）。三层极限叠加起来，就是第 1.4 节那个总目标函数的几何解释：**DeepEP 做的事，是把实际通信时间从"三条极限线之和"压到"三条极限线之最大值"**——和与最大值之间的差，就是所有协议设计、同步设计与重叠设计共同赚出来的空间。

## 1.10 延伸讨论：门控分布如何塑造一切

第 1.4 节的模型假设均衡门控，本节说明当分布偏离均衡时会发生什么，以及 DeepEP 的哪些机制在吸收这种偏离。设第 $e$ 个专家的负载因子为 $\lambda_e$（等于均值负载的倍数），则该专家属主 rank 的专家计算时间为 $\sum_{e \in \text{rank}} \lambda_e \cdot \bar{c}$，而 dispatch 的网络流量在"指向该 rank 的 token 份额"维度上同样乘上 $\lambda$ 权重。分布的偏差通过两条路径伤害性能：其一，**计算路径**——负载最重的 rank 决定整层时间（木桶效应），这是第 7 章 EPLB 的动机；其二，**传输路径**——指向热点 rank 的链路队列变长，通道与 QP 的并发把队列摊到时间轴上，但单链路的带宽上限不变，这是第 4 章通道化与第 6 章多 QP 的动机。两个路径的相对重要性取决于 $k$ 与 $P$ 的关系：$k$ 大而 $P$ 小时，每个 rank 都分摊了几乎全部专家的流量，传输路径的方差被平均掉，木桶效应主导；$k$ 小而 $P$ 大时，单 token 只联系少数 rank，传输方差显著。理解这一点，集成方就能判断自己的配置应该优先投资哪一层——是门控的分布控制（gate 侧）、还是传输并发（通道与 QP）、还是专家复制（EPLB）。`cumulative_local_expert_recv_stats` 提供的在线统计是判断的数据来源，`generate_rank_count` 的 ratio 参数是离线模拟的旋钮。

## 1.12 延伸讨论：为什么通信库必须"理解模型"

本章的推导反复出现 $k$、$E$、$h$ 这些模型参数，本节把一个容易被忽视的立场挑明：**DeepEP 的每一个关键设计都以"理解 MoE 的工作方式"为前提**，这是它与"传输无关的通用通信层"的根本分野。理解体现在三个层面。第一个层面是**数据语义**：dispatch 传输的不是"一批字节"而是"带路由的 token"——因此才有 TokenLayout 的元数据段（第 4 章）、才有 `recv_topk_idx` 的 `-1` 语义（参考实现，第 16 章）、才有"权重随专家输出合并"的 combine 契约（第 5 章）。第二个层面是**负载语义**：门控分布是负载的生成机制——因此才有均衡门控假设下的解析式估算（第 6 章）、才有把倾斜作为一等问题的治理手段（第 1.10、7 章）、才有 `cumulative_local_expert_recv_stats` 这类把路由统计暴露给上层的接口。第三个层面是**生命周期语义**：MoE 层在前向与反向间的角色互换（第 5.2 节）决定了句柄要跨前向反向存活、决定了两方向共用一条链路——这些都不是"传输视角"能推出的设计。三个层面合起来回答了一个定位问题：为什么第 17 章把 DeepEP 归为"语义层"而不是"更快的传输层"——因为它的优化空间恰恰来自模型语义（动态路由产生的通信模式、负载倾斜的统计结构、前后向的对偶性），一个不理解模型的库只能优化这三者之外的常数因子。对读者的启示是反向的：读懂 MoE 的工作方式（本章前七节），就是读懂 DeepEP 一切 API 形态的前提——这也是本书把"问题章"放在最前的用意。



## 1.11 本章小结

- MoE 把参数量与计算量解耦（$E \cdot 2hf$ vs $k \cdot 2hf$），代价是每层引入 dispatch/combine 两次 all-to-all；训练时反向传播再引入两次，通信量 $O(T \cdot k \cdot \frac{P-1}{P} \cdot h)$，在真实配置下达每层数 GiB。
- all-to-all 的三大难点：通信模式无规整、输出尺寸动态、一 token 多目的地；由此产生"CPU 同步 vs 异步流水"的根本矛盾。
- 带宽墙（$T_{\text{comm}} \ge \max(V/B)$ 的三元瓶颈）、延迟墙（RTT 主导的小批量）、SM 墙（通信与计算抢 SM）共同构成设计约束；DeepEP 用解析式配置与双路径回应。
- 通用集合通信库在"计数、重叠、两级路由、部署、MoE 专属数值需求"五个方面不足；DeepEP 的 NCCL Gin 后端 + 统一 EPBuffer 接口正是针对这五点的重构，V1 的 NVSHMEM 已被移除。
- EP 通信内核必须有 SM 参与（计数/归约/同步），这决定了后续一切"SM 预算"设计。

下一章我们拆开 DeepEP 的四层结构，看看这些约束如何映射到 Python buffers / csrc / 设备端内核模板 / DeepJIT 的分层职责上。
