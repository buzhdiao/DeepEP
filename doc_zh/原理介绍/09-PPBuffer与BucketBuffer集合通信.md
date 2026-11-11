# 第 09 章 · PPBuffer 与 BucketBuffer：流水线点对点与批量集合通信

本章覆盖两个缓冲区子系统：`PPBuffer`（流水线并行的点对点收发）与 `BucketBuffer`（面向 CP/DP 的桶式 all-gather/reduce-scatter/all-reduce）。它们是"通信模式谱系"的另外两极——最简单的点对点与最规整的集合通信，共享同一套对称内存与屏障地基（第 3、10 章）。

## 9.1 PPBuffer：流水线并行的点对点通信

### 9.1.1 问题与约束

流水线并行（PP）把模型按层切成若干 stage，micro-batch 沿 stage 链流动（1F1B、交错式等调度）。相邻 stage 之间的通信是**点对点**的：stage $i$ 的激活值发给 stage $i+1$，stage $i+1$ 的梯度发回 stage $i$。它的约束与 all-to-all 截然不同：

- **方向固定**：只有"上游/下游"两个对端；
- **尺寸规整**：激活值张量形状由模型决定，已知且通常巨大（整个 micro-batch 的一层）；
- **延迟敏感**：流水线的气泡（bubble）直接乘在通信延迟上，send/recv 必须"发起即走"；
- **部署形态**：README 明确 PP 缓冲区面向"仅含 RDMA 的通信组"（*"PP 测试需要仅含 RDMA 的通信组"*）——跨节点流水线正是 RDMA 的主场。

### 9.1.2 环形槽位协议：PPSignals

`deep_ep/buffers/pp.py` 的 `PPBuffer` 构造参数只有三个资源量：

```python
PPBuffer(group, num_max_tensor_bytes, num_max_inflight_tensors, sl_idx=None,
         num_gpu_timeout_secs=100, explicitly_destroy=False)
```

构造时按 `num_max_tensor_bytes × num_max_inflight_tensors` 预留 send/recv 槽位（README：*"请在构建时设置 `num_max_tensor_bytes` 和 `num_max_inflight_tensors` 以预留 send/recv 槽位"*）。槽位的轮转由设备端状态机管理，`deep_ep/include/deep_ep/layout/pp/workspace.cuh`：

```cpp
// 流水线并行的收发状态：按环形方向（0/1，即上一 rank / 下一 rank）索引
struct alignas(kNumRDMAAlignmentBytes) PPSignals {
    BarrierSignals barrier;
    int64_t send_count[2] = {};   // 每个方向的累计发送计数（决定当前使用的环形槽位）
    int64_t recv_count[2] = {};   // 每个方向的累计接收计数（决定当前读取的环形槽位）
    int64_t arrival[2][2] = {};   // Ring direction, QP index
};
```

协议本质是一个**按方向的环形缓冲区**：方向 $d \in \{0, 1\}$（下游 0 / 上游 1），第 $n$ 次 send 写入槽位 $n \bmod \text{num\_inflight}$，接收方按 `recv_count` 读对应槽位。`arrival[2][2]` 按方向与 QP 索引记录到达计数——因为两个方向的流量走不同的 QP（第 6 章：PP 取 `min(2, allocated)` 个 QP，恰好把两个环形方向分开），到达信号也按 QP 分开维护，避免两个方向的等待互相阻塞。

```text
方向 d 的槽位环（num_max_inflight_tensors = 3）：

  send_count[d]:  0  1  2  3  4  5  …
                    ↘  槽位 = n mod 3
  [slot0][slot1][slot2]
     ▲              ▲
  recv_count[d] 跟随；对方 arrival[d][qp] 确认"第 n 个已可读"
```

正确性由"计数配对"保证：发送方第 $n$ 次写入前，要求接收方已消费完第 $n - \text{num\_inflight}$ 次（槽位空闲）；接收方第 $n$ 次读取前，等待 `arrival` 达到 $n$。这就是典型的有界缓冲（bounded buffer）问题，只不过计数器与等待都发生在 GPU 显存的信号区里、由内核内的自旋 + 超时（`comm::timeout_while`，`num_gpu_timeout_secs=100` 换算为时钟周期上限）完成——**主机线程完全不参与每条消息的同步**。

### 9.1.3 API 与内核

```python
buffer.send(x, dst_rank_idx, num_sms=None)   # 把 x 经 RDMA 发给目标 rank
buffer.recv(x, src_rank_idx, num_sms=None)   # 从源 rank 接收进 x
```

README 的硬约束："输入与输出必须是指针和大小均按 32 字节对齐的连续 CUDA 张量"——与 TMA 对齐常数一致（第 4 章），RDMA 搬运的最小对齐粒度。宿主侧 `csrc/kernels/pp/pp_send_recv.hpp` 的 `launch_pp_send(context, x, num_x_bytes, dst_rank_idx, num_max_tensor_bytes, num_max_inflight_tensors, …)` 启动设备端发送（`impls/pp/pp_send_recv.cuh`，引入 `cooperative_groups` 以使用 `this_grid().sync()` 做网格级同步——发送内核需要整个网格确认"数据已写入槽位"后再推进计数）。

PP 与 EP 一样需要 SM（"EP 的 dispatch 与 combine 需要 GPU SM；不支持零 SM 的 RDMA EP"的原则同样适用于发起 RDMA 与维护到达信号），`num_sms` 参数同样默认走解析式估算。

## 9.2 BucketBuffer：把多个小集合通信装进桶里

### 9.2.1 问题：CP/DP 的集合通信为什么需要"批量"

上下文并行（CP）把长序列切到多卡（每卡需要 all-gather 其余卡的 KV/激活），数据并行（DP）在梯度同步时需要 reduce-scatter/all-reduce。这些集合通信的特点是**单次调用常常涉及多个中小张量**（每层若干个、加上 CPU 标量同步量等）。用 NCCL 逐张量调用的问题：

1. 每次集合通信都有固定的启动/协议开销，小张量被启动开销淹没；
2. 多次调用之间需要同步点，破坏与计算的交错；
3. 张量的显存位置不受控——集合通信（尤其 NVLink multimem 归约与 RDMA 窗口访问）要求操作数位于**注册过的对称内存**里，而普通 PyTorch 分配的张量不满足。

`BucketBuffer`（`deep_ep/buffers/bucket.py`，C++ 侧 `csrc/buffers/bucket.hpp`、内核 `csrc/kernels/bucket/{all_gather,all_reduce,reduce_scatter}.hpp` 与 `impls/bucket/…`）的答案是"**桶（bucket）**"：把多个张量的字节区间打包成桶（`BucketList`，数量上限 `kNumMaxBuckets`，Python 侧 `_pack_tensors` 做打包），一次内核调用搬运/归约一整桶。

### 9.2.2 注册模型：与 BufferAllocator 的天作之合

常规用法（README）：输入必须位于 bucket 自己的存储中——先规划后构造：

```python
allocation_plan = BufferAllocator()
x = allocation_plan.allocate((group.size(), 1024), torch.float32)
buffer = BucketBuffer(group, allocation_plan)
x[group.rank()].fill_(group.rank())          # 写入本地分片（此时 x 已是缓冲区视图）
gathered_x = buffer.all_gather(x[group.rank()]).wait()
```

构造时规划被 `materialize` 到缓冲区对称存储上，meta 张量变成缓冲区视图——**用户拿到的 `x` 从一开始就是"集合通信友好"的内存**：连续、对齐、已注册。三个集合通信方法返回 `EventOverlap`，`.wait()` 后结果是缓冲区视图（all-gather 返回展平视图）；"所有 rank 必须使用相同的分配规划"是对称性的老要求（第 3 章）。

对于"临时想通信一个不在 bucket 里的张量"，`session()` 提供 with 块：

```python
with buffer.session():
    reduced_x = buffer.all_reduce(external_x).wait()
    # Consume or copy reduced_x before reusing the session storage
```

`BucketSession` 的实现（`deep_ep/buffers/bucket.py`）是 Python 动态方法替换的教科书案例：进入 with 块时把实例上的 `all_gather/reduce_scatter/all_reduce` 三个名字替换为 `partial(self._collective_func, name)`（不支持嵌套会话，退出时 `_call_original` 恢复）。会话内部维护：对齐粒度 `_C.get_num_rdma_alignment()`、暂存游标 `num_allocated_bytes`、以及**单通信组约束** `context_idx`（整个会话只允许一个组，`_check_group` 校验）——外部张量先经 `_allocate` 暂存拷贝进对称存储的剩余空间，再走常规集合路径。README 提到的一个特例：*"仅使用 NVLink 的 all-gather 路径也支持外部源，但需要显式提供 bucket 内的 `dsts`；对于一般的 bucket 外输入，请使用会话。"*——因为 NVLink 直达写不需要对端窗口协商，外部源可以直接作为读取源。

### 9.2.3 三个算子的路径矩阵

三个算子各自支持三条路径（NVLink / RDMA / 混合），但实现机制不同：

| 算子 | NVLink 路径 | RDMA 路径 | 混合路径 | 执行器 |
| --- | --- | --- | --- | --- |
| all-gather | **拷贝引擎（CE）直推**，`num_sms=0` | 纯 RDMA 分发 | RDMA 收 + NVLink 转发 | CE 或 SM |
| reduce-scatter | multimem 归约（需 NCCL multimem） | fp32 / **bf16** 两种通信精度 | 混合 | SM |
| all-reduce | multimem 归约 | 纯 RDMA | 混合（需 `buffer_multimem`，≥2 SM） | SM |

**all-gather 的 CE 路径**值得一提：all-gather 无归约、无计数（目的地就是"所有 rank"），是最接近纯搬运的集合通信，因此可以完全交给拷贝引擎——`launch_all_gather`（`csrc/kernels/bucket/all_gather.hpp`）注释："自动选择纯 RDMA/纯 NVLink（CE 直推）/混合（RDMA 收 + NVLink 转发）三条路径"，参数里 `uint64_t& seq_idx` 是拷贝引擎同步用的**跨调用递增序列号**（CE 操作经 `csrc/kernels/driver/api.hpp` 的 push/sync 接口下发），nvl/rdma 带宽参数用于"估算最优 chunk 数"——把一个大桶切成多个 chunk，按两路带宽比例流水。这是第 1 章"SM 墙"论述的镜像：**不需要 SM 的通信就不要用 SM**。

**multimem 归约**：NVLink 4+ 的多播地址空间支持硬件级规约（`multimem.ld_reduce`），reduce-scatter 与 all-reduce 的 NVLink 路径依赖 NCCL 的 multimem 支持（README："NVLink 上的归约需要 NCCL 的 multimem 支持"）；混合 all-reduce 还有硬校验（`csrc/kernels/bucket/all_reduce.hpp`）：

```cpp
EP_HOST_ASSERT(context.buffer_multimem != nullptr);          // 混合 all-reduce 依赖 multimem 缓冲区
EP_HOST_ASSERT(num_sms >= 2 and "Hybrid all-reduce requires at least 2 SMs");
EP_HOST_ASSERT(context.num_rdma_ranks <= 128 and
               context.num_rdma_ranks * context.num_nvl_ranks <= 1024);
```

**reduce-scatter 的通信精度**：`comm_precision` 参数支持 `"fp32"` 与 `"bf16"`——后者"以较低的精度进行通信，同时保持 FP32 的输入与输出张量"（README），即**输入输出 FP32、网络上走 BF16**：先降精度传输、接收端再升回 FP32 归约。这是梯度同步的经典带宽换精度手段，把选择权交给调用方（第 5 章 EPLB 梯度归约坚持全程 FP32，因为参数梯度直接进优化器；而 DP 梯度同步的瞬时误差可被吸收）。

### 9.2.4 信号区与归约的数学

信号区 `BucketSignals`（`layout/bucket/workspace.cuh`）按 RDMA 对齐聚合所有算子的信号：`BarrierSignals barrier` + AllGather/AllReduce/CopyEngine/ReduceScatter 各自的信号结构。数据布局由 `ChunkStorage`（`layout/bucket/chunk.cuh`）承担——桶内数据按 chunk 切分存放，归约内核按 chunk 粒度并行。归约的语义带缩放系数：

$$
\text{result} = \text{scale} \cdot \sum_{p=0}^{P-1} \text{contrib}_p,
$$

`scale` 参数（launch 签名中的 `const float& scale`）允许"求和后平均"一步到位（DP 梯度平均 $= \text{scale} = 1/P$）。reduce-scatter 内核还显式接收 `num_qps`（QP 并行度影响 chunk 到达顺序的并行度）与 `context_idx`（workspace 信号槽数组中本上下文的索引——同一缓冲区可能服务多个通信组，信号区按组隔离）。

### 9.2.5 SM 数与拓扑的解析式决策

`BucketBuffer.get_theoretical_num_sms(op, group=None)`（`deep_ep/buffers/bucket.py`）按算子类型与拓扑估算 SM 数——all-gather 走 CE 时为 0，归约类算子按带宽模型估算。构造前提（README）："BucketBuffer 目前要求 NVLink 与 RDMA 带宽均可被检测到"——三条路径的选择与 chunk 划分都是**启动时解析计算**的（`launch_all_gather` 直接吃 `nvl_bandwidth`/`rdma_bandwidth` 参数），带宽探测缺失就没有规划输入。这与 EP 的哲学完全一致：**能算的不调优，能规划的不同步**。

## 9.3 四个缓冲区谱系的收拢

至此四个子系统全部出场，谱系视角下的统一与差异：

```text
            通信模式谱系（按"同步原语"与"数据流方向"两轴）

 数据流   ▲
 推送向   │  EPBuffer(all-to-all,计数+槽位)        PPBuffer(点对点,环形槽位+到达计数)
 拉取向   │  EngramBuffer(按索引 get,请求句柄+分层流水)
 规整集合 │  BucketBuffer(桶式,chunk+multimem/CE)
          └──────────────────────────────────────────────────────▶ 同步复杂度
```

共享的地基（对称内存 window、`BarrierSignals` 两相位屏障、Gin QP、事件模型）不重复实现；差异面（计数协议、归约布局、chunk 流水、请求句柄）各自封装。V2.5 的拆分正是沿着这些"差异轴"切割的。

## 9.4 本章小结

- PPBuffer 用"按方向的环形槽位 + 累计计数 + 按 QP 分开的到达信号"（`PPSignals`：`send_count[2]`/`recv_count[2]`/`arrival[2][2]`）实现纯设备端的有界缓冲点对点收发，CPU 零参与；32 字节对齐、RDMA-only 组、网格级同步（`this_grid().sync()`）与超时兜底。
- BucketBuffer 以"桶"为单位做批量 all-gather/reduce-scatter/all-reduce：`BufferAllocator` 规划的对称张量即操作数；`BucketSession` 用实例级方法替换支持外部张量的暂存通信（单组、不可嵌套）。
- 路径矩阵：all-gather 可走拷贝引擎（`num_sms=0`、`seq_idx` 序列、`csrc/kernels/driver/` push/sync）；归约类依赖 multimem（混合 all-reduce 强校验 `buffer_multimem`、≥2 SM、RDMA ≤128 / 总 ≤1024）；reduce-scatter 支持 bf16 网络精度保 FP32 进出；`scale` 参数一步平均。
- `BucketSignals` 聚合所有算子信号，`ChunkStorage` 承载数据，chunk 划分由 nvl/rdma 带宽解析估算；`get_theoretical_num_sms(op)` 按算子决策 SM。
- 四缓冲区谱系按"数据流方向 × 同步原语"分野，共享对称内存/屏障/QP/事件地基。

## 9.5 延伸讨论：桶的切分策略与归约的数值分配

**桶切分的自由度与收益**。`BucketList` 与 `kNumMaxBuckets` 把"多少个桶、每个桶多大"的决定权交给调用方的注册方式：`BufferAllocator` 里每个 `allocate` 的张量是一个自然桶，也可以打包（`_pack_tensors`）后作为一个桶。桶数的影响是双向的：桶越多，chunk 划分的粒度越细、三条路径间的负载分配越灵活（all-gather 可以按桶的归属 rank 选择 CE 直推或 RDMA），但信号区占用与内核调度开销随之上升；桶越少，路径选择越粗但开销越低。经验法则与 EP 的通道化一致——**桶数应当与参与 rank 的拓扑多样性同阶**：纯 NVLink 组内桶数不需要多（一条 CE 流就能吃满），混合组则值得为"节点内部分"与"跨节点部分"分桶，让两条路径各取所需。

**chunk 数的带宽配比**。`launch_all_gather` 以 `nvl_bandwidth` 与 `rdma_bandwidth` 为输入估算最优 chunk 数，其原理与第 6 章 SM 估算同构：把总流量按两路带宽的比例切成 chunk，使两路同时完成（$\max$ 最小化）。可以给一个直观的推导：设跨节点流量 $V_o$、节点内流量 $V_i$，理想切分满足 $V_o/B_{\text{rdma}} = V_i/B_{\text{nvl}}$，即 chunk 的跨节点/节点内份额按带宽比反比分配。README 要求 BucketBuffer 的 NVLink 与 RDMA 带宽都必须可探测（`nvidia-smi`/`ibstat`），正是因为这个配比计算在启动时就要完成——探测缺失时库宁可拒绝构造也不猜一个次优比例。

**归约路径的数值分配**。reduce-scatter/all-reduce 的 FP32 与 BF16 两种通信精度，本质上是把第 5 章"多重归约的精度分析"推广到了集合通信：BF16 网络传输把每个贡献的相对误差从 $O(\epsilon_{f32})$ 抬到 $O(\epsilon_{bf16})$（约为 $2^{-8}$ 对 $2^{-24}$），对 $P$ 个贡献求和后总误差 $\sqrt{P} \cdot \epsilon_{bf16} \cdot \max|\text{contrib}|$ 量级。DP 梯度同步对此不敏感（梯度本身带噪声、优化器有动量吸收），这正是 `comm_precision="bf16"` 的适用场景；而第 7 章的 EPLB 梯度归约坚持 FP32，因为其贡献数少但直接进参数更新。两个场景对照给出的原则是：**误差进入"有状态累积"的路径用 FP32，进入"无状态消费"的路径可以用低比特**。

**session 的边界与陷阱**。`BucketSession` 的三处断言值得复述其动机：不支持嵌套（实例方法已被替换，嵌套会破坏恢复顺序）、单通信组（`context_idx` 唯一，避免暂存区在两组间的归属歧义）、退出前消费或拷贝结果（暂存区游标 `num_allocated_bytes` 在下次进入时清零复用）。README 的示例注释 *"Consume or copy reduced_x before reusing the session storage"* 正是对第三点的提示——会话存储是短生命周期的，把它当持久缓冲用必然出错。常规路径（注册张量）没有这些限制，因此注册应当是默认，会话只是补充。

## 9.6 延伸讨论：拷贝引擎路径的原理与边界

第 9.2.3 节指出 all-gather 的纯 NVLink 路径可以完全由拷贝引擎（CE）驱动（`num_sms=0`），本节论证"为什么 all-gather 可以、而其他算子不可以"，并说明 CE 路径的工程边界。CE 是 GPU 上与 SM 并列的异步拷贝硬件，它不执行程序逻辑、只按描述符搬数据；启动与管理 CE 操作需要宿主或内核下发描述符并同步完成——DeepEP 的 `csrc/kernels/driver/`（api.hpp 与 driver.cpp）提供的正是"拷贝引擎 push/sync 接口"（第 2.5 节表格），`launch_all_gather` 的参数 `uint64_t& seq_idx` 是跨调用递增的序列号，用于把多次 CE 推送的完成同步按序组织（第 9 章）。

all-gather 能用 CE 的三个前提：无归约（CE 只会搬不会算）、无动态计数（目的地集合固定为"全部 rank"、每个分片的偏移由注册布局决定）、无条件逻辑（每个 chunk 的源与目标是规划期已知的常量映射）。对照 reduce-scatter（要算）与 dispatch（要计数与路由），CE 的边界清晰。CE 路径的价值是把 all-gather 的 SM 占用降为零——腾出的 SM 全部给计算，这在第 1 章 SM 墙的意义上是"白赚"的并行度。工程边界也有三条：CE 的带宽通常低于 SM 驱动的峰值（描述符驱动的粒度较粗），大块连续数据才能吃满；CE 操作的完成同步要跨调用管理（序列号的递增纪律，第 9 章）；以及 CE 路径与 SM 路径的混合使用要避免争抢同一段存储的读写时序——这也是为什么混合路径由 SM 收尾转发（第 9.2.3 节"RDMA 收 + NVLink 转发"）。README"所有集合通信均支持 NVLink、RDMA 以及混合域"与"all-gather 由拷贝引擎驱动"两句话合起来，就是本节论证的全部。

## 9.7 延伸讨论：PP 槽位数的规划与流水线调度的耦合

`PPBuffer` 的两个资源参数（`num_max_tensor_bytes` 与 `num_max_inflight_tensors`，第 9 章）与流水线调度器（1F1B、交错式等）存在直接耦合，规划时应从调度器推导而不是拍一个数。推导的骨架：在 1F1B 类调度中，一个 stage 在任意时刻的在途激活数量由调度深度决定——深度为 $d$ 的调度最多同时有 $d$ 个未确认的前向激活在两个方向上流动，`num_max_inflight_tensors` 必须不小于两个方向在途数之和，否则发送端会在环形槽位耗尽时等待接收端推进（第 9 章的有界缓冲语义），把流水线调度的并行性人为压扁。`num_max_tensor_bytes` 则由单次 send/recv 的最大张量尺寸决定——即"单个 micro-batch 在该 stage 边界上的激活字节数"的上界，加上对齐余量（32 字节对齐，README）。两个参数都按上限预留显存（第 9 章"按 num_max_tensor_bytes × num_max_inflight_tensors 预留槽位"），因此调度深度越激进，显存预留越大——这与流水线调度的经典权衡（深度换气泡）在显存维度上再次出现。集成方的检查点：更换调度器或调整微批大小时，同步更新这两个参数并重建缓冲区（重建的代价与纪律见第 3 章）；运行时若出现 send 长时间等待，先核对在途数是否超过预留（第 14 章同步类问题的 PP 变体）。

## 9.8 延伸讨论：混合域中的桶级路径决策

第 9.2.3 节给出了三算子的路径矩阵，本节把"混合域里每个桶走哪条路"的决策机制讲透——因为桶是路径选择的实际单位，而桶的归属决定了选项。混合域中，一次 all-gather 的参与方分两类关系：同一节点的 rank（NVLink 可达）与跨节点 rank（仅 RDMA）。桶的"目的地集合"决定可选项：若某桶的全部目的地在本节点，纯 NVLink（拷贝引擎直推）可用且最省 SM；若目的地全在远端节点，纯 RDMA 分发可用；混合目的地则走"RDMA 收 + NVLink 转发"的混合路径——`launch_all_gather` 注释里的三路径自动选择（第 9 章）正是按桶粒度做这个判定。reduce-scatter 与 all-reduce 的桶级决策多一个维度：归约的参与方约束（第 9.2.3 节矩阵中 hybrid all-reduce 对 multimem 缓冲与至少两个 SM 的断言）——混合归约需要 NVLink 侧的硬件归约能力，纯路径则各有自己的前提。这个桶级决策机制解释了注册模型的另一个好处：`BufferAllocator` 的规划期就确定了每个张量的桶归属与字节区间，路径判定所需的全部信息（大小、对齐、归属）在启动时已知——"能规划的不同步"的原则在 bucket 上体现为"路径选择是启动时计算，不是运行时协商"。集成方的调优含义：当混合路径成为瓶颈时，优先调整桶的划分（把同节点目的地的桶拆出来走拷贝引擎，第 13.11 节流量治理），而不是全局切换路径模式——桶的粒度就是路径优化的粒度。

下一章把地基本身挖开：mbarrier、两相位屏障、TMA、原子操作、对称寻址与事件重叠的内核原理。
