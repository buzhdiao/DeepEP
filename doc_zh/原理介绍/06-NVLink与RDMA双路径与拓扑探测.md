# 第 06 章 · NVLink/RDMA 双路径与拓扑探测

## 6.1 两层互连，两种物理

DeepEP 的传输模型建立在两类物理互连的干净二分上：

- **scale-up（节点内）**：NVLink。特征是高带宽、低延迟、**对称内存可达**——一块注册过的显存，对端 GPU 可以像访问本地一样直接读写（经 `get_sym_ptr` 换算地址）。NCCL 设备 API 中对应的"团队"是 **LSA**（`ncclTeamTagLsa`，NVLink 域）与 **Rail**（`ncclTeamTagRail`，同机），见 `deep_ep/include/deep_ep/comm/handle.cuh` 里 `NCCLGin` 的 `IS_TEAM_WORLD/LSA/RAIL` 宏族。
- **scale-out（跨节点）**：RDMA（InfiniBand/RoCE）。特征是延迟高一个数量级、但可扩展到数百节点；访问以**窗口（window）+ QP** 为单位，读（get）、写（put）、原子（atomic）与多播由 NCCL Gin 在设备端发起（GIN = GPU Initiated Networking）。

DeepEP 用两个查询把这两层暴露给上层（`deep_ep/comm/handle.py`）：

| 查询 | 返回 | 语义 |
| --- | --- | --- |
| `get_physical_domain_size()` | `(num_rdma_ranks, num_nvlink_ranks)` | **物理域**：按实际互连连通性划分，rank 被组织成 RDMA 域内每节点 `num_nvlink_ranks` 个 |
| `get_logical_domain_size()` | `(num_scaleout_ranks, num_scaleup_ranks)` | **逻辑域**：按配置划分的跨节点/节点内 rank 数 |

`EPBuffer.__init__` 里的换算公式（源码）：

```python
self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks   # 节点间编号
self.scaleup_rank_idx  = self.rank_idx %  self.num_scaleup_ranks   # 节点内编号
```

即线性 rank 被切成"先节点内、后节点间"的两维坐标 $(s_{\text{out}}, s_{\text{up}})$。这个两维坐标是后面一切路由、缓冲区布局与 QP 分配的基础。

## 6.2 路径矩阵：direct 与 hybrid

`EPBuffer` 构造参数 `allow_hybrid_mode`（默认 True）+ 拓扑共同决定路径选择。三种部署形态对应三种内核模板组合：

| 部署形态 | 拓扑条件 | dispatch/combine 模板 | 桶式集合通信路径 |
| --- | --- | --- | --- |
| 纯 scale-up | 单节点 NVLink 组 | `impls/ep/dispatch.cuh`（`kIsScaleupNVLink=true`） | 纯 NVLink（all-gather 可走 CE） |
| 纯 scale-out | 无 NVLink 的 RDMA 组 | dispatch/combine 的 RDMA 路径 | 纯 RDMA |
| 混合 | 多节点 + 节点内 NVLink | `impls/ep/hybrid_dispatch.cuh` / `hybrid_combine.cuh` | 混合（RDMA 收 + NVLink 转发） |

**hybrid（混合）模式**的核心思想是"两级路由"：

```text
direct（若有 NVLink 直连）:              hybrid（两级）:
  rank A ──NVLink──▶ rank B               rank A ──RDMA──▶ 目标节点某 rank ──NVLink──▶ rank B
          （一次直达）                        （一级跨节点，二级节点内分发）
```

选择 hybrid 的理由写在 `EPBuffer` 的 docstring 里：*"Hybrid mode uses hierarchical RDMA + NVLink communication to achieve higher bandwidth, and is more friendly to multi-plane/multi-rail networks."* 展开论证有三点：

1. **带宽聚合**。跨节点 RDMA 带宽通常远低于节点内 NVLink 聚合带宽。两级路由允许"每节点一个（或少数几个）RDMA 接入点 + 节点内 NVLink 扇出"，把昂贵的 RDMA 带宽用在刀刃上，节点内的二次分发用便宜的 NVLink 完成。总时延 $T \approx \max(V_{\text{rdma}}/B_{\text{rdma}}, V_{\text{nvl}}/B_{\text{nvl}})$ 的取值可以通过分配两级流量比来优化，而 direct 模式没有这个自由度。
2. **多轨友好**。multi-rail 网络里每台机器有多块网卡分摊流量；两级路由天然适配"按 rail 分片、节点内归拢"的流量形态。
3. **QP 规模可控**。direct 模式下与所有远端 rank 通信都需要连接；hybrid 把"对外"压缩为节点级，QP 数随节点数而非 rank 数增长（但 DeepEP 的 QP 预分配仍然按最坏情况配置，见 6.4）。

代价也真实存在：二级分发引入一次节点内中转拷贝（额外的 HBM 读写与 NVLink 流量），且中转 rank 成为瞬时热点。这就是为什么 `allow_hybrid_mode` 是可关闭的开关，而缓冲区大小计算（`calculate_ep_buffer_size`）也把它作为参数——hybrid 需要额外的中转窗口空间。

## 6.3 QP（Queue Pair）的预分配与 Gin 上下文

RDMA 通信的连接单元是 QP。QP 的数量与深度直接决定**在途消息并发度**：每 QP 是一个独立的发送队列，多 QP 可以让多通道（channel）的数据互不阻塞地并行投递。DeepEP 的 QP 预分配策略（`EPBuffer.__init__`）：

```python
if num_allocated_qps == 0:
    if self.allow_hybrid_mode:
        # 混合模式每个 channel 都要独立 QP，且额外一个用于 notify warp
        num_allocated_qps = 65 if check_fast_rdma_atomic_support() else 129
    else:
        num_allocated_qps = 17
```

三个数字背后的推理：

- **65 vs 129**：hybrid 模式需要大量并行 QP 供通道使用；当网卡支持**快速 RDMA 原子**（`check_fast_rdma_atomic_support()`，可用 `mlxconfig` 把 `PCI_ATOMIC_MODE` 设为 4 提升原子吞吐，见 6.6）时，完成信号可以用原子操作走数据面，notify 类小消息可以复用 QP 通道，QP 需求减半（64+1 vs 128+1）。
- **17**：非 hybrid 模式 8 个数据 QP + 备用。通道数受 `kNumMaxChannels=1280` 上限与 QP 数共同约束。
- C++ 侧创建时（`csrc/kernels/comm/context.cpp` 的 `Context` 构造）：`reqs.ginContextCount = num_allocated_qps`（**每个 QP 一个 GIN 上下文**）、`ginQueueDepth = qp_depth > 0 ? qp_depth : kDefaultQPDepth`。Engram 场景下窗口数组大小还要求 Gin 的 `NCCL_WIN_STRIDE` 步长（`gin_min_stride`），DeepEP 用 `EP_NUM_MAX_LOCAL_RANKS`（默认 16）来估算注册存储（第 8 章）。

运行时每次 dispatch/combine 用多少 QP 由 `EPBuffer.get_theoretical_num_qps(num_sms)` 估算（`num_qps=0` 时自动），PP 内核则取 `get_num_qps(context) = min(2, context.num_allocated_qps)`（`csrc/kernels/pp/pp_send_recv.hpp`）——点对点方向只有两个（上/下游），2 个 QP 已能把两个环形方向分开。

## 6.4 解析式 SM 估算：get_theoretical_num_sms 的推导

`EPBuffer.get_theoretical_num_sms(num_experts, num_topk, num_scaleout_topk=0, rdma_gbs=0, nvlink_gbs=0, sm_read_gbs=0, sm_write_gbs=0)` 是 DeepEP"免自动调优"承诺的落点。它的输入是四路带宽：

- `rdma_gbs`：跨节点带宽，0 时自动探测（`get_rdma_gbs()`，解析 `ibstat`）；
- `nvlink_gbs`：节点内带宽，0 时自动探测（`get_nvlink_gbs()`，解析 `nvidia-smi`）；
- `sm_read_gbs` / `sm_write_gbs`：每 SM 的 HBM 读/写带宽（`get_sm_read_gbs()` / `get_sm_write_gbs()`，0 时用内置默认）。

建模方法（结合 docstring *"Estimate the optimal number of SMs … based on bandwidth modeling. This assumes a balanced gate distribution"* 与第 1 章的通信量模型）：设均衡路由下每 rank 的 scale-out 流量 $V_{\text{rdma}}$ 与 scale-up 流量 $V_{\text{nvl}}$（由 $T$、$k$、$h$、topk 的跨节点分量 `num_scaleout_topk` 等推得），SM 数为 $S$，则通信时间受三重瓶颈约束：

$$
T(S) = \max\left( \frac{V_{\text{rdma}}}{B_{\text{rdma}}},\ \frac{V_{\text{nvl}}}{B_{\text{nvl}}},\ \frac{S_{\text{rd}}(S) \cdot V_{\text{rd}} + S_{\text{wr}}(S) \cdot V_{\text{wr}}}{S \cdot b_{\text{sm}}} \right),
$$

其中第三项是 SM 聚合的 HBM 读写时间——**SM 越多，每 SM 分摊的搬运量越少、总 HBM 聚合带宽越高**；而前两项与 $S$ 无关（网络是外生资源）。$S$ 的下界由"HBM 搬运时间 ≤ 网络时间"给出：

$$
S^{*} = \arg\min_{S} T(S), \qquad \text{即让第三项恰好贴近前两项的最大值。}
$$

超过 $S^{*}$ 的 SM 只会闲置（网络饱和）或挤占计算；少于 $S^{*}$ 则网络空转等 HBM。估算结果按 docstring 的约定"为偶数且至少为 4"（通道与 QP 的成对配比需要偶数 SM）。两个附加设计：

- `prefer_overlap_with_compute=True`（默认）时估算偏向**更少**的 SM（docstring：*"If enabled, we tend to use fewer SMs"）——把余量留给计算内核，这正是第 1 章"SM 墙"约束的逆问题：不是问"通信最快要多少 SM"，而是问"留下计算余量后通信最少要多少 SM"。
- 结果经 `weak_lru`（`deep_ep/utils/semantic.py` 的弱引用 LRU 装饰器）缓存：以 `self` 的弱引用为键做方法级缓存，避免 `functools.lru_cache` 把缓冲区对象钉在内存里。这是 Python 层生命周期纪律在缓存策略上的呼应。

`num_sms` 与 `num_qps` 都是**每次调用可覆盖**的参数（dispatch/combine 签名中的 `num_sms=0, num_qps=0`）——解析式估算给出的是"好默认值"，不是"唯一值"。

## 6.5 拓扑探测与前置校验

构造缓冲区前的探测链（`deep_ep/utils/envs.py`）：

| 函数 | 探测内容 | 用途 |
| --- | --- | --- |
| `check_nvlink_connections(group)` | 组内 NVLink 连通性 | PCIe 互联的机型在此被诊断（构造函数中调用） |
| `check_fast_rdma_atomic_support()` | 网卡快速原子能力 | 决定 QP 数 65/129；RDMA 原子信号路径可行性 |
| `get_nvlink_gbs()` / `get_rdma_gbs()` | 解析 `nvidia-smi` / `ibstat` | SM 估算与 bucket 的 chunk 数估算 |
| `get_sm_read_gbs()` / `get_sm_write_gbs()` | 每 SM HBM 读写速率 | SM 估算第三项 |
| `check_torch_deterministic()` | PyTorch 确定性模式状态 | dispatch 前校验与 DeepEP 确定性实现的兼容性 |

README 明确的探测前提：*"自动带宽检测使用 `nvidia-smi` 和 `ibstat`；`BucketBuffer` 目前要求 NVLink 与 RDMA 带宽均可被检测到，即使某个通信组只使用其中一种传输方式。"*——bucket 缓冲区强制两路带宽可用，因为其内核在启动时要按两路带宽比例估算 chunk 划分（第 9 章），缺一路就无法完成解析式规划。

## 6.6 网络侧调优：SL/VL、自适应路由与原子模式

README"网络配置"一节给出了三条建议，DeepEP 侧各有对应机制：

**（1）流量隔离。** InfiniBand 用虚拟通道（VL）隔离流量；专家并行负载与其他负载应分到不同 VL。DeepEP 通过 RDMA 服务等级（SL）参数选择流量类别，优先级链是：

$$
\text{EP\_OVERRIDE\_RDMA\_SL} \;>\; \text{sl\_idx（构造参数）} \;>\; \text{EP\_DEFAULT\_RDMA\_SL} \;>\; \text{NCCL 默认},
$$

网络侧的 SL→VL 映射决定实际通道。`EP_DEFAULT_RDMA_SL` 与 `EP_OVERRIDE_RDMA_SL` 都是运行时环境变量，前者是兜底默认、后者无条件覆盖——把"集群统一策略"（环境变量）与"单缓冲区微调"（参数）分成两级。

**（2）自适应路由。** 交换机把流量均匀摊到多条路径，引入额外延迟但提升各种负载下的整体带宽，README 建议开启。这对 dispatch 这类"突发、全互联"流量尤其有效——不开启时个别链路会成为通道热点。

**（3）PCI 原子模式。** 网卡 `PCI_ATOMIC_MODE` 设为 4（`sudo mlxconfig -y -d mlx5_0 set PCI_ATOMIC_MODE=4`）可提升 RDMA 原子操作性能。原子操作在 DeepEP 里承担两件事：设备端计数器的跨节点累加（dispatch 的到达统计）、以及部分完成信号的实现；`EP_DISABLE_GIN=0`（默认）时 Gin 初始化正常进行，仅 NVLink 场景可跳过。README 的实验性 Eager 分支（PR #437）则从协议上消除原子 RTT——"使用低延迟协议，消除由 RDMA 原子操作引入的额外 RTT 延迟"。

## 6.7 规模上限与解析式配置的意义

`csrc/kernels/bucket/all_reduce.hpp` 的启动校验给出了规模上限的实感：

```cpp
EP_HOST_ASSERT(context.num_rdma_ranks <= 128 and
               context.num_rdma_ranks * context.num_nvl_ranks <= 1024);  // 混合 all-reduce
```

RDMA 域 ≤ 128、总 rank ≤ 1024（README："支持更大的 scale-up 与 scale-out 域"）。这些上限不是随意的：通道数（`kNumMaxChannels=1280`）、信号区容量（`kNumMaxSignalBytes`、`EPSignals` 的计数器阵列）、QP 数、以及"每 rank 一份"的 workspace 内存都随规模线性或平方增长——解析式配置（SM 数、QP 数、缓冲区大小、chunk 划分）让这些量在**构造时**就被确定并校验，而不是在运行时撞上隐式上限。V2 发布说明把它列为 EPv2 的特性之一："解析式（analytical）SM 与 QP 数量计算——不再需要自动调优"。

## 6.8 本章小结

- scale-up（NVLink/LSA，对称直达）与 scale-out（RDMA/GIN，窗口+QP）两层互连，经 `get_physical_domain_size`/`get_logical_domain_size` 暴露，rank 编号按 `scaleout = rank // num_scaleup_ranks`、`scaleup = rank % num_scaleup_ranks` 切成两维坐标。
- 路径三态：纯 scale-up（`dispatch.cuh`）、纯 RDMA、hybrid 两级（`hybrid_*.cuh`）；hybrid 以"RDMA 聚合 + NVLink 扇出"换取带宽利用、多轨友好与可控 QP 规模，代价是一次节点内中转。
- QP 预分配 65（快速原子）/129/17，每个 QP 一个 Gin 上下文（`ginContextCount`），深度为 `kDefaultQPDepth`；PP 只需 `min(2, allocated)`。
- SM 数的解析式估算把通信时间建模为 $\max(\text{RDMA}, \text{NVLink}, \text{SM 聚合 HBM})$ 三项瓶颈，`prefer_overlap_with_compute` 时偏向少占 SM；结果偶数且 ≥4，经 `weak_lru` 缓存，每次调用可覆盖。
- 网络调优三件套：SL/VL 流量隔离（三级 SL 优先级）、自适应路由、`PCI_ATOMIC_MODE=4`；带宽探测依赖 `nvidia-smi`/`ibstat`，BucketBuffer 强制两路带宽可探测。

传输路径已经就绪。## 6.9 延伸讨论：两级路由的正确性论证与失败模式

两级路由（hybrid）引入了一个一级路径不存在的新正确性问题：**中转区的生命周期**。一级跨节点传输把数据写进"目标节点某 rank 的中转缓冲"，该数据在被二级 NVLink 分发消费之前必须保持有效。这意味着 dispatch/combine 的缓冲区必须为中转预留空间——这正是 `calculate_ep_buffer_size` 把 `allow_hybrid_mode` 作为输入参数的原因（第 3 章构造时序中的第 2 步）。若用户手动指定 `num_bytes` 而实际启用了混合模式，容量计算必须覆盖中转区，否则一级写入会与二级读取竞争同一块内存。这也是为什么 README 建议绝大多数用户走 `get_buffer_size_hint` 的推荐值而不是手写字节数：解析式计算已经把 hybrid 的额外空间计入，手写值很容易漏掉这一项。

第二个值得论证的问题是**两级之间的同步依赖**。二级分发必须等一级数据全部到达中转区才能开始吗？答案藏在通道结构里：hybrid 内核按通道划分工作（`kNumMaxTokensPerChannel`），每个通道的二级分发只依赖自己通道的一级到达信号（由 mbarrier 与计数器承载，第 10 章），通道之间互不等待。这种"通道内流水"把一级传输与二级分发的总时间从"串行相加"压缩为"首通道二级启动延迟 + 末通道一级完成时间 + 通道流水深度"的量级——通道数越多，两级重叠越充分，这也再次解释了通道数为什么是 QP 预分配与 SM 数估算的输入。

第三是**失败模式的定位**。两级路径的故障症状与单级不同：若一级 RDMA 未送达，症状是二级通道的到达计数永远不满，表现为 GPU 侧超时（`num_gpu_timeout_secs` 换算的周期上限触发，第 10 章）；若二级 NVLink 写失败，症状通常是 `get_sym_ptr` 换算出的对端地址不可写（对称内存未就绪或 rank 未完成构造握手）。因此排查 hybrid 问题时，第一步应当确认所有 rank 都完成了构造函数末尾的双重同步（第 3 章），第二步再检查 RDMA 层（SL/VL 配置、`PCI_ATOMIC_MODE`、自适应路由），最后才是内核侧。`EP_BUFFER_DEBUG=1` 打印的初始化、拓扑与缓冲区大小诊断信息是这一流程的数据来源。

最后把 direct 与 hybrid 的选择标准归纳成一个可操作的判据：**当跨节点流量占比高且多轨网络带宽能被单节点聚合利用时，选 hybrid；当 EP 组本来就以节点内为主、或网络是单轨单平面时，direct 更简单且没有中转开销**。`allow_hybrid_mode` 默认为 True 是因为生产集群（多节点多轨）是主流场景；但推理部署、单节点大模型、以及 RDMA 带宽相对充裕的环境，显式关闭 hybrid 常常是更优解——缓冲区更小、路径更短、故障面更小。这个判断不需要跑基准：看拓扑（`get_physical_domain_size` 与 `get_logical_domain_size` 的比值）就能预先做出。

## 6.10 延伸讨论：带宽探测的工程细节与失效模式

解析式配置的全部可信度都建立在带宽探测的准确性上，本节把探测链路的工程细节与失效模式讲透。探测的输入是两条命令的输出解析：`nvidia-smi`（NVLink 状态与带宽）与 `ibstat`（RDMA 端口速率），分别由 `get_nvlink_gbs` 与 `get_rdma_gbs` 完成（`deep_ep/utils/envs.py`，第 6.5 节表格）。它们的失效模式各有特点。`nvidia-smi` 侧的典型问题：容器环境挂载不完整导致命令缺失、驱动版本差异导致输出格式变化使解析失败、以及节点内 NVLink 拓扑不对称（不同 GPU 对之间的链路数不同）使"单一 GB/s 数字"只能代表折中值。`ibstat` 侧：端口未激活（InfiniBand 端口状态不是 Active）会读不到速率、RoCE 环境下工具链可能不兼容、多网卡多端口时存在"取哪个值"的歧义。README 对 BucketBuffer 的硬性要求（两路带宽都必须可探测，第 6.5 节）正是为了避免这些失效模式被静默吞掉——探测失败宁可拒绝构造，也不带着错误的比例做 chunk 规划（第 9.5 节）。

更隐蔽的一类问题是**探测值与可用值的偏差**：`ibstat` 报告的是链路物理速率，实际可用带宽要扣除协议开销、信用机制与其他流量的占用；NVLink 报告的是理论聚合值，实际单向可用值取决于方向分摊。这类系统性偏差的校准方法在第 13.1 节已经给出：用实测吞吐反推有效带宽，作为 `get_theoretical_num_sms` 的显式参数传入。运维上建议把"探测值对校准值的比值"作为集群的常备资产记录下来——比值显著低于一，往往意味着 SL 隔离或拥塞配置有问题（第 14 章问题十一），是网络健康度的一手指标。

## 6.11 延伸讨论：规模上限的来源分析

第 6.7 节引用了混合 all-reduce 的规模断言（RDMA 域不超过 128、总 rank 不超过 1024），本节分析这类上限的三个来源，帮助读者判断"我的规模能不能上"。第一个来源是**信号区容量**：workspace 里的计数器阵列按最大规模静态分配（`EPSignals` 的 `kNumMaxExperts=2048`、`kNumMaxChannels=1280`、Engram 的 `kNumMaxPeerContexts=2048`，第 4、8 章），`kNumMaxRanks` 决定屏障与计数器的宽度——这些都是编译期常量，超规模不是"变慢"而是"断言拒绝"（`EP_HOST_ASSERT`，第 14 章）。第二个来源是**同步成本**：全局屏障的参与方数量线性影响到达计数的传播延迟（跨 NVLink 域的原子加），规模越大屏障越贵，因此内核把屏障的粒度与通道绑定（按通道推进而非按消息屏障）。第三个来源是**QP 与连接数**：hybrid 模式的 QP 预分配随通道数增长，网卡侧的 QP 总量与内存注册上限构成硬约束。

三个来源共同决定了"上限是阶梯而非斜坡"：在断言允许的规模内，性能随规模平滑下降（同步与流量成本渐进上升）；超过断言则直接拒绝。集成方做大规模规划时应把断言值当作"已验证的安全区"，把接近断言的区域当作"需要实测验证的边缘区"——第 19 章的容量规划流程正是按这个思路把规模检查放在网络清单之前。

## 6.13 延伸讨论：拓扑不对称的识别与应对

前文的模型大多假设域内对称（每 rank 的 NVLink 邻居数相同、每节点的网卡数相同），真实集群的拓扑不对称并不罕见，本节讨论识别与应对。不对称的常见形态有三类。第一类，**节点内 NVLink 不对称**：不同 GPU 对之间的链路数不同（取决于互连拓扑代次与板卡布局），表现为"同一节点内某些 rank 对的直达带宽低"——识别手段是 `nvidia-smi` 的拓扑矩阵（探测链路的输入，第 6.10 节），应对是让通道与 QP 的分配偏向高带宽对（解析式配置以平均值为输入时的偏差，经第 13.13 节校准吸收）。第二类，**网卡与轨道不对称**：multi-rail 集群中某轨故障或降速，使部分 rank 的 RDMA 带宽低于同伴——识别是双端测量（第 13.13 节设计要点其三），应对是把该轨流量隔离（SL 调整）或在 hybrid 模式下调整接入点分布（第 6.2 节两级路由的自由度）。第三类，**域划分与物理不对称的错位**：逻辑域（按配置划分，第 6.1 节）假设的"每节点相同 rank 数"与实际部署（异构节点混布）不符——识别是对照 `get_physical_domain_size` 与 `get_logical_domain_size` 的返回值（构造期诊断，第 14.1 节问题二），应对是修正部署划分而非让通信库迁就（对称性是协议前提，第 3.5 节）。三类形态的共同处理原则：先让不对称"可见"（探测与测量），再决定是"绕开它"（路由与隔离）还是"修复它"（部署调整）——通信库的责任边界止于前者，后者属于集群工程。本节与第 14.4 节问题十一的关系是"因与症"：拓扑不对称是持续性的性能偏低根因之一，而吞吐异常只是它的症状。

下一章把视角转向一个特殊的负载——热点专家的复制与回收：`lb_prefetch_weights` 与 `lb_reduce_grads`。
