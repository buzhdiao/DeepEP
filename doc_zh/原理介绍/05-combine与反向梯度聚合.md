# 第 05 章 · combine 与反向梯度聚合

## 5.1 combine 的语义：dispatch 的逆运算加归约

dispatch 把 token 送向专家，combine 把专家输出送回 token。形式化地，dispatch 在接收端建立的映射是：展开式布局下，第 $e$ 个本地专家的接收缓冲中第 $j$ 行，对应某个来源 token $t$ 与其第 $r$ 个 top-k 槽位 $(t, r)$，且 $\text{topk\_idx}[t][r] = e$。combine 要做的计算是：

$$
\mathrm{combined}_x[t] \;=\; \sum_{r=1}^{k} g_r(t) \cdot \mathrm{recv\_x}\big[\mathrm{slot}(t, r)\big] \;+\; \mathrm{bias}[t],
$$

其中 $g_r(t) = \text{topk\_weights}[t][r]$（若调用方提供了 `topk_weights`），`bias` 是可选的逐 token 加项——README 指出它可以在 combine epilogue 期间加入**共享专家（shared-expert）输出或残差**，这正是 DeepSeek 系列 MoE 架构"共享专家 + 路由专家"结构的落点：与其单独为共享专家输出安排一次加法，不如让 combine 在归约的同一次内存写里捎带上。

combine 的输入必须为 **BF16**（README 明确约束："Combine 的输入必须为 BF16"）。理由是双重的：其一，专家计算以 BF16/FP8 混合精度完成，输出统一回 BF16 保证兼容；其二，归约的累加精度由内核内部保证（fp32 累加，见 5.4），若输入再引入 FP8 会让"低比特通信"的误差分析叠层失控。与 dispatch 相比，combine 的 `TokenLayout` 因此简化为 `get_combine_token_layout(hidden, elem_size, num_topk) = TokenLayout(hidden * elem_size, 0, num_topk, false)`——没有 scale factor 段（`csrc/kernels/ep/combine.hpp`）。

## 5.2 combine 在反向传播中的角色互换

这是训练中最容易被忽视、却是 DeepEP API 设计最精巧的一点。对 MoE 层

$$
y = \sum_{r=1}^{k} g_r(t)\, E_{e_r}(x_t),
$$

反向传播的两条链分别是：

$$
\frac{\partial L}{\partial x_t} = \sum_{r=1}^{k} g_r(t) \cdot \frac{\partial L}{\partial y_t}\Big|_{E_{e_r}} \qquad \text{（dispatch 的反传 = combine）},
$$

$$
\frac{\partial L}{\partial g_r(t)} = \left\langle \frac{\partial L}{\partial y_t},\; E_{e_r}(x_t) \right\rangle \qquad \text{（combine 的反传 = dispatch）}.
$$

换言之：**combine 的反向操作是 dispatch（把 $\partial L/\partial y$ 重新送到各专家求参数梯度），dispatch 的反向操作是 combine（把专家给出的 $\partial L/\partial x$ 贡献聚合回原 token）**。README 的训练示例把这两个函数分别命名为 `dispatch_backward` 与 `combine_backward`，实现上只是复用同两个 API 并传入前向缓存的 `handle`：

```python
def dispatch_backward(grad_recv_x, grad_recv_topk_weights, handle, bias=None):
    # The backward pass of MoE dispatch is actually a combine.
    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x, handle=handle, bias=bias,
        topk_weights=grad_recv_topk_weights, num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )
    return combined_grad_x, combined_grad_topk_weights, event

def combine_backward(grad_combined_x, handle):
    # The backward pass of MoE combine is actually a dispatch.
    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x, handle=handle, num_sms=_num_comm_sms,
        do_expand=True, do_zero_padding=True, use_tma_aligned_col_major_sf=True,
    )
    return grad_x, event
```

注意两个细节：`combine` 传入 `topk_weights=grad_recv_topk_weights`——反向的"门控权重梯度"就是乘法因子的梯度路径；`dispatch_backward` 返回的第二个张量是 `combined_grad_topk_weights`，即门控权重的梯度 $\partial L/\partial g_r$。整套反向不需要任何新的通信原语——handle 里缓存的 `token_metadata_at_forward`、`dst_buffer_slot_idx`、`channel_linked_list` 恰好构成逆向路由的全部信息。这就是第 4 章元数据家族"回程"一列的意义。

## 5.3 归约布局：单次归约 vs 多重归约

combine 最大的自由度在于"归约发生在哪一层"。同一 token 的 $k$ 份专家输出贡献，既可以各自独立送回原 rank 再一次归约（**单次归约**），也可以在途中按 rank 分组先做若干次局部归约（**多重归约**）。`EPBuffer` 构造参数 `allow_multiple_reduction` 的 docstring 精确描述了权衡：*"whether to allow multiple reductions in combine. If disabled, only one reduction will be done in the combine epilogue for best precision, but it may increase data transfer size."*

设备端的选择逻辑由编译期常量表达（`impls/ep/combine.cuh`）：

```cpp
constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;
```

即"不允许多重归约且使用展开布局"时，发送端按展开式逐一发送每个 (token, expert) 贡献，保证每个贡献独立到达接收端、由 combine epilogue **一次**完成归约——数值上最干净（所有加法都在接收端按 fp32 一次性累加，误差只取决于一次求和的舍入），代价是传输量保持 top-k 倍，没有任何途中合并的机会。允许多重归约时，途中可以合并同源贡献，传输量下降，但结果经历多次部分和舍入。

是否值得按 rank 组织归约，由 `impls/ep/combine_utils.cuh` 的编译期谓词决定：

```cpp
template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>
constexpr bool use_rank_layout() {
    if constexpr (not kAllowMultipleReduction) return false;   // 不允许多重归约 → 不能按 rank 合并
    return kNumRanks <= kNumTopk;   // 仅当 rank 数不超过 top-k 时，按 rank 布局才更省
}
```

这条谓词的推理值得展开：按 rank 布局能把"同一来源 rank 的多个专家贡献"合并成一份传输，收益正比于 $\min(k/P, 1)$——只有当 $k > P$（每 token 的专家数多于 rank 数）时合并才真正节省流量；$k \le P$ 时按 rank 合并几乎不产生合并机会，反而引入额外组织成本。把这样的拓扑相关决策编译进模板而非运行时判断，是 DeepJIT（第 11 章）存在的直接原因之一。

## 5.4 归约核心：combine_reduce 与展开式还原

`impls/ep/combine_utils.cuh` 提供 `combine_reduce`（归约核心循环）与 `CombineVecTraits`（向量化 traits），`compute_topk_slots` 负责在展开式布局下定位一个 token 的 $k$ 个源槽位——即求解

$$
\{\,\mathrm{slot}(t, r)\,\}_{r=1}^{k} \quad \text{s.t.} \quad \text{topk\_idx}[t][r] = e_r \in \text{本 rank 或经路由可达}.
$$

发送端内核（`impls/ep/combine.cuh` 与 `hybrid_combine.cuh`）按 dispatch 缓存的元数据把每个贡献写回原 rank 的接收缓冲区；接收端 epilogue（`impls/ep/combine_reduce_epilogue.cuh`）把"dispatch/combine 各 rank 缓冲区里的贡献"归约写出（源码注释：*"combine 的收尾（epilogue）kernel：把 dispatch/combine 各 rank 缓冲区里的 …"*）。向量化宽度由 `kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t)` 决定，展开因子 `kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>()` 在寄存器预算内最大化循环展开——通信归约内核的瓶颈是 HBM 访存，展开与向量化直接影响每次 load/store 的有效带宽。

精度路径可以做定量对比：设单次 fp32 累加的舍入误差界为 $\epsilon$，则单次归约的总误差为 $O(k \cdot \epsilon \cdot \max|g_r \cdot y_r|)$；多重归约在途中有 $m$ 个合并点，误差为 $O((k + m) \cdot \epsilon \cdot \max(\cdot))$——多重归约把误差上限抬高，换来传输量下降。`allow_multiple_reduction=True`（默认）是吞吐优先，训练框架若做损失复现或梯度异常排查，可构造 `allow_multiple_reduction=False` 的缓冲区。

## 5.5 混合模式下的 combine 与寄存器预算

`impls/ep/hybrid_combine.cuh` 与 `hybrid_dispatch.cuh` 是混合（RDMA + NVLink）两级路径的实现，其中有一个特别值得讲的内核优化——**寄存器预算分组**：

```cpp
constexpr int kNumRegistersForScaleupWarps = 40;
constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;  // = 216
if constexpr (kAdjustRegisters) { … }   // 启动时调整两组 warp 的寄存器上限
```

混合内核里存在两类负载极不均衡的 warp：负责 scale-up（节点内 NVLink）方向的 warp 与负责 forward（跨节点 RDMA 一级）方向的 warp。前者工作短平快（NVLink 延迟低、数据立即落位），后者需要维持大量在途状态（RDMA 请求、通道链表、元数据）。CUDA 默认给同内核所有线程分配同等寄存器上限，会让"轻 warp"白白占用预算，压缩"重 warp"的展开与缓存能力。`kAdjustRegisters` 在内核启动时把 256 个寄存器按 40/216 重新分配，配合 `__launch_bounds__(kNumThreads, 1)`（保证每 SM 恰好一个 block 的占用，不给编译器留占用率歧义）。JIT 质量门 `EP_JIT_CHECK_NO_SPILLS` / `EP_JIT_CHECK_NO_LOCAL_MEMORY`（第 11 章）则从反面兜底：任何导致寄存器溢出或 local memory 的配置直接拒绝编译——通信内核的延迟敏感度不允许隐性 HBM 往返。

## 5.6 bias 注入与 epilogue 的时机

combine 的 `bias` 参数（可为张量，或 `(张量, 张量)` 二元组——`EPBuffer._unpack_bias` 解包）在 **combine epilogue** 期间加入。结合第 4 章的延迟 epilogue 机制，这给出一个非常实用的重叠模式：

```python
event = combine_forward(expert_output, handle)      # 通信先行入队
# … 独立计算：此时可以算 bias（例如共享专家前向、残差加工）…
combined_x, _ = event.current_stream_wait()          # 等待点执行 epilogue（含 bias 加法）
```

README 特别指出："这也使得由中间计算产生的 combine `bias` 可以在该处被消费"——bias 可以是**通信发起时尚未就绪**的张量，只要在 `current_stream_wait()` 之前算完即可。同理，延迟 epilogue 的返回值"由第一次等待产生"，因此必须保留第一次 `wait()` 的结果供后续使用。

## 5.7 EPLB 梯度归约：combine 之外的第二类聚合

训练冗余专家（第 7 章）时，每个冗余副本都会在自己的 rank 上产生一份 FP32 专家梯度；参数更新前必须把所有副本的梯度**累加回属主专家**。这与 combine 的"多源聚合"形似，但有三个本质不同：

1. **聚合对象是参数梯度而非激活**：形状为每专家一行（`lb_reduce_grads` 要求二维 FP32 张量，每专家参数展平为一行），不是 token 流。
2. **方向是 NVLink 域内的定点聚合**：从各冗余槽位 rank 读 FP32 梯度、原地累加进 `expert_grads`，`redundancy_mapping` 决定"谁贡献给谁"。
3. **不清零**：README 明确"归约不会清除冗余梯度；在下一次累加之前请覆盖它们或将它们置零"，且调用前要"用本地贡献（或零）初始化 `expert_grads`"——原子累加语义而非覆盖语义。

实现上由 `csrc/kernels/ep/reduce_grads.hpp` 的 `launch_lb_reduce_grads(context, redundant_expert_grads, expert_grads, redundancy_mapping, num_redundant_experts, num_local_experts, hidden, num_sms, stream)` 启动，**每个 warp 私有一个 24 KiB 的定长 chunk 缓冲**（`constexpr int num_chunk_bytes = 24 * 1024`），warp 按 chunk 粒度循环：读远端、累加、写回。为什么取 24 KiB？这是"寄存器/共享内存占用 × 单次拷贝粒度"的折中：chunk 太大则每 warp 的缓冲占用把 SM 塞不下，太小则 NVLink 往返次数上升。第 7 章将对照权重预取的 12 KiB chunk 做同样的论证。

## 5.8 combine 与 dispatch 的对称性总结

```text
           前向                          反向
dispatch  ───────────────────────────▶  combine
   │  (token,expert) 对流向专家)           │  (∂L/∂y 流向专家求参数梯度)
   │                                      │
   ▼                                      ▼
专家计算                                  专家计算(参数梯度)
   │                                      │
   ▼                                      ▼
combine   ◀───────────────────────────  dispatch
(专家输出加权归约回 token)               (∂L/∂x 贡献聚合回 token)
```

四个方向共用同一组内核模板、同一套元数据、同一个句柄——差别只在模板开关（如 `kReuseSlotIndices` 缓存槽位索引时为真）与传入的权重张量。这种"前向反向共用一条链路"的设计把测试面从四套协议压缩到一套，也让确定性（`deterministic=True` + `EPHandle.deterministic_sort`）在两个方向上一致成立。

## 5.9 本章小结

- combine 的计算为 $\mathrm{combined}_x[t] = \sum_r g_r(t)\cdot \mathrm{recv\_x}[\mathrm{slot}(t,r)] + \mathrm{bias}$，输入限定 BF16，`TokenLayout` 无 SF 段；bias 可为共享专家输出或残差，并在延迟 epilogue 中消费"尚未就绪"的中间计算结果。
- 训练反向中角色互换：dispatch 的反传是 combine、combine 的反传是 dispatch，`EPHandle` 的 `token_metadata_at_forward`/`dst_buffer_slot_idx`/`channel_linked_list` 构成逆向路由的全部信息。
- 归约布局由 `kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout` 与 `use_rank_layout()`（仅当 $k \ge$ rank 数才有合并收益）编译期决定；单次归约精度最优、多重归约传输最省。
- 混合 combine 用寄存器预算分组（scale-up warp 40 / forward warp 216）适配两类 warp 的负载差异，`__launch_bounds__(kNumThreads, 1)` 固定占用，JIT 质量门拒绝溢出。
- EPLB 的 `lb_reduce_grads` 是第二类聚合（FP32 参数梯度、NVLink 域内、原子累加、不清零），每 warp 24 KiB chunk。

## 5.10 延伸讨论：归约的调度顺序与消费侧配合

combine 的归约不只是"加起来"这一步，它的正确执行还依赖一整套调度顺序上的配合，本节把这些容易被忽略的配合关系集中说清。

第一是**权重的应用位置**。combine 允许两种等价但性质不同的做法：把门控权重 $g_r$ 在发送端乘进专家输出再传输，或在接收端归约时再乘。DeepEP 的接口把选择交给数据流：前向 combine 时不传 `topk_weights` 则纯求和（权重已由专家计算侧处理或由 bias 通道吸收），反向 combine 时传入 `grad_recv_topk_weights` 作为乘法因子。从数值上看，接收端乘权重再累加可以让所有乘加共享同一段累加寄存器，避免发送端先乘引入的一次中间舍入；从传输上看，两种做法的流量完全相同。这也是为什么 README 强调"专家计算会在 combine 之前应用这些路由器权重"——权重乘法被放在专家 GEMM 之后、combine 之前的那段计算里，与通信解耦。

第二是**归约与 epilogue 的层次**。`combine_reduce_epilogue.cuh` 的职责是把"dispatch/combine 各 rank 缓冲区里的贡献"归约写出，这意味着归约的最终一次发生在接收端的收尾内核里，而不是通信内核里。这个分层让 combine 的通信部分只负责"把贡献送到接收缓冲区"，把"求和 + 写 bias + 写输出"留给 epilogue——于是 `defer_epilogue=True` 才有可能：通信可以先行入队，epilogue 推迟到 `current_stream_wait()` 的时刻，此时 bias 所需的中间计算（例如共享专家的前向输出）已经就绪。如果归约被塞进通信内核，这种延迟就无从谈起。

第三是**与专家 GEMM 的衔接**。展开式布局下，专家 GEMM 的输出就是"每专家行一份贡献"的形态，combine 的发送端按 `dst_buffer_slot_idx` 与 `token_metadata_at_forward` 把每行贡献映射回来源 token；`compute_topk_slots` 负责在接收端为一个 token 找齐它的 $k$ 个贡献槽位。这个映射的完整性依赖句柄存续期内 `topk_idx` 不被修改（第 4 章不变式），也依赖 `allow_multiple_reduction` 与构造时的布局选择保持一致——若前后两次调用的缓冲区配置不同，槽位解释就会错位。集成方最容易犯的错误是在反向传播里换了一个缓冲区或改了配置却复用旧句柄；正确的做法是 README 示例展示的那样：训练循环里前向保存的 handle 一直用到 `dispatch_backward` 与 `combine_backward` 完成。

第四是**多重归约下的顺序性问题**。允许多重归约时，途中合并的顺序由内核的通道划分与到达顺序决定，不同运行之间可能不同——这意味着多重归约在浮点意义上不是位级确定的。需要位级可复现的训练应当关闭多重归约（构造 `allow_multiple_reduction=False` 的缓冲区），让所有加法收敛到接收端 epilogue 的单一确定顺序。确定性专题（第 16 章）会把这个话题与确定性排序、参考实现语义一起完整展开。

综合这四点，combine 的设计可以用一句话概括：**把"通信"与"数学"分开**——通信内核只管把字节送到正确位置，数学（乘权重、求和、加 bias）全部集中在 epilogue 这个可延迟、可插拔的收尾点上。这个分离让 combine 同时服务前向（MoE 聚合）、反向（dispatch 的反传）与负载均衡（冗余梯度归约的近亲，第 7 章）三种语义而不需要三套内核。

## 5.11 延伸讨论：归约网络的组织形态

第 5.3 节讨论了"单次归约 vs 多重归约"的语义权衡，本节讨论归约在接收端内部的组织形态，这是理解 combine 内核工作分配的关键。展开式发送（`kDoExpandedSend`）下，接收端收到的贡献已经按槽位就位，epilogue 的归约是对"一个 token 的 $k$ 个贡献"做一次 $k$ 输入求和——工作分配天然按 token 划分，每个负责该 token 的 warp 读 $k$ 个槽位、累加、写出。这个形态的优点是每个输出的写只发生一次（无中间态），缺点是读放大：负责归约的逻辑若重复扫描接收缓冲区会浪费带宽——`compute_topk_slots`（第 5.4 节）的职责正是把"一个 token 的 $k$ 个源槽位"一次找齐，避免重复扫描。

允许多重归约时，归约被部分前移：同源贡献在途中合并（`use_rank_layout` 的按 rank 组织），接收端见到的输入数减少、单输入的尺寸增大。两种形态在 HBM 流量上的差异可以定量表述：设贡献总量为 $V$、途中合并率为 $\mu$（由 $k$ 与 $P$ 决定，0 到 1 之间），接收端的读流量从 $k$ 份贡献全量读入降到约 $(1-\mu) \cdot k + 1$ 份——合并率越高读放大越小。这解释了为什么 `use_rank_layout` 的谓词（第 5 章）要求 $k \ge P$ 才启用 rank 布局：只有此时合并机会（每 rank 的贡献数 $k/P \ge 1$）才真实存在，组织成本才有回报。

归约的数值顺序在两种形态下也不同：展开式发送时所有加法发生在接收端 epilogue 的固定循环内（顺序由槽位索引决定，天然确定）；多重归约时顺序分散在途中与接收端（时序相关）。第 16 章确定性配方的"关闭多重归约"一项，本质上是把归约网络从"分布式树"退化为"接收端单点星形"，用确定性交换传输量——这个交换在需要位级复现时价值无限，在只需要统计精度时则是浪费。集成方按第 16.5 节的两级配方选择即可。

## 5.12 延伸讨论：bias 通道的架构语义

combine 的 `bias` 参数看似一个小功能，实际上承载了 MoE 架构演进中的一个重要语义：**把"与路由无关的固定部分"从路由路径中剥离**。共享专家（shared expert）的输出对每个 token 都相同地加入，残差连接亦然——它们的共同特征是"不需要经过 all-to-all 传输"（共享专家在本 rank 即可计算，残差本来就在原 rank）。把它们的加入安排在 combine epilogue（第 5.1 节），有三个架构收益：其一，通信流量不因共享专家的存在而增加——若给共享专家单独安排一次"加法传输"，等于把一份隐藏状态的流量重复一遍；其二，bias 可以是通信发起时尚未就绪的中间计算结果（第 5.6 节），这让"共享专家前向"与"路由专家通信"天然并行；其三，数值上 bias 加法与路由求和共享同一次 FP32 累加写回（第 15 章数值链），避免一次额外的读写遍历。

`_unpack_bias` 支持张量或二元组（第 5.6 节），二元组对应"主输出 + 伴随成分"的解包语义。集成的推荐做法是把共享专家的输出在独立计算窗口（`current_stream_wait` 之前的间隙）算好并作为 bias 传入——这正是第 18.3 节时序模板中"独立计算：共享专家前向、bias 准备"那一格的含义。反过来，若框架把共享专家实现为"额外的常开专家"并让门控给它固定权重，路由路径上会多出一份恒定流量与槽位占用——两种实现的性能差异在最坏情形下接近共享专家与路由专家的尺寸比，值得在架构评审时对照本节的论证。

## 5.14 延伸讨论：combine 在推理与训练中的不同代价结构

combine 在训练与推理中的角色频率不同，代价结构也不同——区分这两者有助于解释 API 的一些细节取向。推理场景：combine 每层只出现一次（前向聚合），没有反向角色，其代价结构是"每层一次的聚合延迟"——decode 小批量下延迟主导（第 1.4.2 节），因此推理侧更依赖单次归约的确定性结构（无反向需求，多重归约的传输收益也有限），以及延迟 epilogue 里顺带完成的 bias 加法（共享专家与残差，第 5.6 节）。训练场景：combine 每层出现两次（前向聚合加反向 dispatch），且与 dispatch 的角色互换意味着 combine 的实现质量同时影响两个方向的性能——这解释了为什么归约布局（第 5.3 节）与寄存器预算（第 5.5 节）的优化在训练负载下的杠杆更大。两个场景对 `allow_multiple_reduction` 的偏好也不同：训练追求吞吐（默认开启合理，第 15.3 节），推理追求尾延迟与确定性（途中合并的时序波动会影响长尾，第 16.4 节）。集成方的推论：同一份 combine 代码服务两个场景是接口统一的胜利（第 20.1 节），但两者的调优重点应当分开建立——推理侧重叠配方的选择（第 13.5 节配方 B）与训练侧重叠配方（配方 A）的差异，本质上是代价结构差异的镜像。把场景敏感性写进容量手册（第 19.10 节混部策略），可以让"一套部署服务训练与推理"的容量决策有据可依。

下一章回到传输层：NVLink 与 RDMA 两条路径如何被组织成两级路由，SM 与 QP 数如何解析式地确定。
