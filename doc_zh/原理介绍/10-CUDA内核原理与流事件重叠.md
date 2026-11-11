# 第 10 章 · CUDA 内核原理：PTX、原子操作、对称内存与事件流重叠

前几章反复出现的"notify warp 原子计数""mbarrier 完成通知""两相位屏障""通信流与事件"本章一并拆开。所有设施都集中在两个地方：`deep_ep/include/deep_ep/common/ptx.cuh`（设备端 PTX 封装，`deep_ep::ptx` 命名空间）与 `deep_ep/include/deep_ep/comm/barrier.cuh`（设备端全局屏障）加 `deep_ep/utils/event.py` + `csrc/utils/event.hpp`（事件体系）。

## 10.1 线程组织原语：warp 视角的几个小函数

`ptx.cuh` 的开头是一组 warp 原语（源码）：

```cpp
__forceinline__ __device__ void trap() { asm volatile("trap;"); }   // 设备端致命陷阱

__forceinline__ __device__ int get_warp_idx() {
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);  // 广播 0 号 lane 的 warp 编号
}

__forceinline__ __device__ int get_lane_idx() {
    int lane_idx;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(lane_idx));  // PTX 读取 %laneid
    return lane_idx;
}

__forceinline__ __device__ int elect_one_sync() {
    int pred = 0;
    asm volatile(
        "{\n.reg .b32 %%rx;\n"
        "elect.sync %rx|%p, %membermask;\n"   // 活跃掩码内选举出恰好一个线程
        "@p mov.s32 %0, 1;\n}" : "=r"(pred) …);  // 仅被选中者写 1
    return pred;
}
```

为什么通信内核需要它们？`get_warp_idx`/`get_lane_idx` 让"任务按 warp 分片"的代码可以直接写（第 7 章的"每 warp 私有 chunk"、第 8 章的 issue warp 都以此为前提）；`elect_one_sync` 从活跃 warp 中选举唯一代表线程执行"每 warp 一次"的操作（如推进计数器、发起一次 RDMA 请求）——比起"lane 0 干活 + 全员等待"的写法，`elect.sync` 是硬件级单指令选举，且对 warp 内分支（活跃掩码）天然正确。`trap()` 则是设备端错误的最终兜底：检测到协议损坏（如计数器溢出、非法槽位）时让整个 grid 立即进入错误态，避免静默数据损坏。

## 10.2 mbarrier：异步完成通知的硬件载体

`ptx.cuh` 把 CUDA 标准库的屏障类型直接别名为 DeepEP 的核心同步对象：

```cpp
using mbarrier = cuda::barrier<cuda::thread_scope_block>;  // 块内可见的异步内存屏障对象
using arrival_phase = uint32_t;                            // 屏障相位编号（0/1 交替翻转），用于 parity 等待
```

mbarrier 是 Hopper 架构引入的硬件异步屏障：生产者（TMA 拷贝引擎、DMA、其他 warp）向它"到达"（arrive），消费者以**相位（phase/parity）**为界等待。经典用法：

```text
消费者：mbarrier.wait(parity = p)          # 等"第 p 相"完成
生产者：async copy … ; mbarrier.arrive()   # 拷贝完成后到达
硬件：到达计数满 → 翻转相位 → 等待者放行
```

相位奇偶（parity）翻转是关键：屏障复用时只需交替比较 0/1，无需清零重建——与第 8 章 issue_counters 的"清零即信号"、第 10.3 节两相位屏障是同一思想在不同硬件层的三次出现。在 EP 内核里，mbarrier 承担两类职责：**TMA 拷贝完成通知**（token 槽位数据落定）与**组内生产者-消费者同步**（发送 warp 写完 → 归约 warp 读）。dispatch 的 `TokenLayout` 里可选的 mbarrier 段（`layout/ep/token.cuh`）正是把屏障对象内嵌进数据槽位——每个 token 槽位自带"我到了"的硬件信号灯。

## 10.3 设备端全局屏障与两相位协议

跨 SM、跨 GPU 的同步无法用 mbarrier（它是块内作用域），DeepEP 用共享的信号区（workspace）+ 原子操作实现全局屏障。`layout/common/barrier.cuh`：

```cpp
struct alignas(kNumRDMAAlignmentBytes) BarrierSignals {
    // 实现一个两相位（2-phase）的 NVL 屏障
    unsigned long long nvl_barrier_counter;   // 到达计数器：跨 rank 原子累加，检测全体到达
    int nvl_barrier_signals[2];               // 两相位各自的信号标志：交替使用避免上一相位残留值影响下一相位
    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() { … }
};
```

协议流程（两相位）：

```text
相位 p（p=0,1 交替）：
  每个 rank 的代表线程：atomicAdd(nvl_barrier_counter) → 达到 N 时置 nvl_barrier_signals[p]
  其他线程：自旋读 nvl_barrier_signals[p] == N → 通过
  进入相位 p+1：读/清另一组标志 → 残留值不干扰
```

为什么需要两组标志？若只有一组，屏障复用时上一相位的残留值会让下一相位的等待者**提前放行**（读到了旧值）——这是分布式屏障的经典竞态。两相位把"当前相位的到达信号"与"上一相位的清理"解耦在两个槽位上，等待者只看自己相位的槽位。`comm/barrier.cuh` 在其上补充了工程细节：

- `kNumOneSecCycles = 2000000000`——"An approximation of the GPU clock at 2000 MHz"：超时以**时钟周期**为单位（换算自 `num_gpu_timeout_secs`），使 GPU 侧等待可度量；
- `kDeviceBarrierTag = 0` / `kKernelBarrierTag = 1`——预留标签区分"设备端发起的屏障"与"独立 barrier kernel 发起的屏障"，超时报错时能定位调用来源；
- `comm::timeout_while` 是所有自旋等待的统一入口（Engram、PP 内核都引入它）——把"无限自旋"变成"有限自旋 + 诊断性失败"，这是长运行训练系统对抗死锁/对端掉线的底线机制。

CPU 侧的屏障入口在 `csrc/comm/barrier.hpp`：`barrier(context, signals, stream, num_gpu_timeout_cycles, with_cpu_sync, sequential, flush_stores, wait_comm_stream)`——参数自解释了组合维度（是否 CPU 同步、是否顺序屏障、是否刷新写、是否等通信流），经 `csrc/comm/handle.hpp` 绑定到 Python（`_C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)`）。

## 10.4 TMA 与批量拷贝

TMA（Tensor Memory Accelerator）在 DeepEP 中的角色是"**把数据搬运从 SM 指令流里卸出去**"。SM 逐元素 load/store 每条指令都要占用发射槽与寄存器；TMA 以描述符发起整块拷贝，完成后经 mbarrier 通知，SM 只负责发起与等待。三个落点：

1. **TokenLayout 的对齐设计**（第 4 章）：hidden/SF/metadata 各段按 TMA 对齐（32 字节），使每段可独立用 TMA 搬运；
2. **`use_tma_aligned_col_major_sf`**：scale factor 按 TMA 要求排成列主序，使下游 DeepGEMM 的 TMA 描述符直接可用——通信库为计算库布局"预先对齐"是跨界协同的典型手法；
3. **epilogue 的 `st_bulk`**：`dispatch_copy_epilogue.cuh` 的注释指明其依赖"PTX 内联汇编封装（TMA、mbarrier、st_bulk 等）"——收尾拷贝用批量存储指令写 `recv_x`，比逐元素循环省一个数量级的指令发射。

`utils.cuh` 的 `iterate_ld_st` 模板则是"手工搬运"侧的通用化：`kNumThreadGroups × kNumThreadsPerGroup × kNumElemsPerThread` 的三维分块 + `kThreadStride` 跨步（默认线程总数），同组线程处理的元素在内存中交错跨步——既保证 warp 内合并访存（coalescing），又让不同组处理不同区间、便于按组做"读 A 写 B"的流水。

## 10.5 原子操作：计数器的两个世界

DeepEP 的原子操作分两层：

- **节点内（NVLink）**：对远端显存的原子累加走 NVLink 的原子事务（PTX `red_add`，`ptx.cuh` 注释将其列为封装对象之一）。dispatch 的 notify 计数、`BarrierSignals` 的到达计数都是它。
- **跨节点（RDMA）**：经 NCCL Gin 的设备端原子。能力探测 `check_fast_rdma_atomic_support()` 决定两条路：支持快速原子时（配合网卡 `PCI_ATOMIC_MODE=4`），完成信号可走数据面原子、QP 需求减半（65 vs 129，第 6 章）；不支持时退化为更多消息的显式通知协议。

原子计数器协议有一条隐藏的纪律：**计数器槽位的生命周期必须覆盖所有并发使用者**。`EPSignals` 把 `notify_reduction[kNumMaxRanks + kNumMaxExperts]` 固定在 workspace 里（不随调用分配），`notify warp` 的共享内存镜像按 `kNumNotifyWarps * 32` lane 对齐（`get_num_notify_smem_bytes`）——固定的槽位 + 固定的对齐 + 两相位/清零协议，使计数器可以跨调用复用而不引入竞态。

## 10.6 对称寻址与窗口

第 3 章讲了对称内存的建立，这里是它的消费端。设备内核持有三个地址相关对象：本地指针、NCCL 设备通信器（`deep_jit::NoRefPtr` 包装的 `ncclDevComm_t`——`NoRefPtr` 明确不持引用计数，内核参数传递不改变通信器生命周期）与窗口 `ncclWindow_t`。地址换算在 C++ 侧封装为 `Context::get_sym_ptr(void* ptr, const int& dst_nvl_rank_idx)`（`csrc/kernels/comm/context.cpp`）：给定本地对称区内的指针与对端 rank，返回对端虚拟地址。多分片场景（Engram、LB 区域）则持有窗口**数组**（`rdma_storage_windows_gpu`，数量为 Gin 步长 `gin_min_stride`），设备端按窗口索引访问不同分片。

对称寻址通信的正确性依赖于两条纪律：**写入方确认对端可见性**（内存栅栏 + mbarrier/计数器信号，`ptx.cuh` 的栅栏封装在 Engram 内核注释中被列为依赖）、**读取方等待完成信号**（相位/计数/请求句柄）。任何一侧跳过都是数据竞争——这也解释了为什么 EP 内核里同步代码与搬运代码总是成对出现。

## 10.7 流与事件：重叠的骨架

### 10.7.1 通信流

`csrc/comm/stream.hpp`：

```cpp
static at::cuda::CUDAStream get_comm_stream() {
    static const auto comm_stream = at::cuda::getStreamFromPool(true);  // 高优先级异步流，进程级唯一
    return comm_stream;
}
```

所有通信内核与延迟 epilogue 都在**这一条**进程级通信流上执行。独立流的价值：（1）与计算流天然并行，重叠由硬件队列调度完成；（2）高优先级（`getStreamFromPool(true)`）保证通信不被长计算内核饿死——第 1 章三面墙里"SM 墙"的调度侧解法；（3）全局唯一使事件跨 API 传递有了统一的"时间轴"。

### 10.7.2 EventHandle 与 EventOverlap

C++ 的 `EventHandle`（`csrc/utils/event.hpp`）包装 `torch::Event` 并携带 `tensors_to_record`（`tensor_list_t = vector<optional<torch::Tensor>>`）——对参与跨流传递的张量调用 `record_stream` 登记，防止 PyTorch 缓存分配器在计算流视角过早复用"通信流还在读"的显存。Python 的 `EventOverlap`（`deep_ep/utils/event.py`）在其上提供重叠语义：

| 方法 | 语义 |
| --- | --- |
| `current_stream_wait(release_handle=False)` | 让当前流等待事件；在 `defer_epilogue` 模式下**同时执行延迟的 epilogue**并返回结果元组（dispatch 四元组 / combine 二元组） |
| `wait()` | 单纯等待（供直接消费） |
| `register_hook_after_wait(hook)` | 注册"等待完成后执行的回调"（如登记释放、统计） |
| `__enter__/__exit__` | 上下文管理器形态，块内计算与通信重叠 |

`dispatch`/`combine` 的重叠协议由三个参数编码（第 4 章已述，此处串成时序）：

```python
ev_ready = buffer.capture()          # 在当前流捕获"输入就绪"事件（独立计算入队前）
event = buffer.dispatch(x, …, previous_event=ev_ready,
                        async_with_compute_stream=True,
                        allocate_on_comm_stream=True, defer_epilogue=True)
# 当前流继续做独立计算（不等待通信）
recv_x, recv_topk_idx, recv_topk_weights, handle = event.current_stream_wait()
```

三者的分工：`previous_event` 声明"输入何时就绪"（通信内核先等它再启动）；`allocate_on_comm_stream` 把新分配张量的所有权登记到通信流（否则输出张量在计算流视角可能被提前复用）；`async_with_compute_stream=True` 表示当前流**不**自动等待通信完成——把"何时等待"的决策权交给用户（README："如果没有其他工作可以重叠，请立即调用 `.wait()`；返回的结果由第一次等待产生，因此请保留它供后续计算使用"）。

### 10.7.3 record_stream 的开关

`EP_AVOID_RECORD_STREAM=1`（README）："对于 EP 与 bucket 操作，在完成事件中保留张量，而不是调用 `record_stream`。在通信完成之前，请保持该事件存活。"——record_stream 本身有成本（张量级登记与延迟回收），大批量小张量场景下开销可观；开关允许高级用户用"保证事件存活"的纪律换取零登记开销。这是所有权语义从"框架兜底"上移到"用户契约"的典型权衡，与 `explicitly_destroy`（第 3 章）同一设计谱系。

## 10.8 寄存器、占用与质量门

通信内核的性能对编译质量极其敏感，DeepEP 用三层手段把住关口：

1. **占用声明**：`__global__ void __launch_bounds__(kNumThreads, 1)`——每 SM 恰好一个 block。通信内核按通道划分工作，占用率歧义（编译器为提高占用而压缩寄存器）会破坏手写的寄存器预算（第 5 章 hybrid combine 的 40/216 分组依赖 `kAdjustRegisters` 生效）。
2. **JIT 质量门**：`EP_JIT_CHECK_NO_SPILLS=1` 拒绝寄存器溢出（spill 到 local memory 的内核），`EP_JIT_CHECK_NO_LOCAL_MEMORY=1` 拒绝任何 local memory 使用（第 11 章详述）。通信内核的每次 HBM 往返都直接变成延迟，溢出是性能毒药。
3. **超时与失败语义**：所有自旋等待走 `timeout_while`，屏障超时带标签（`kDeviceBarrierTag`/`kKernelBarrierTag`）诊断；不可恢复错误 `trap()`。

## 10.9 把一次 dispatch 的同步事件全部串起来

```text
计算流：  capture() ────────────── 独立计算 ───────────────── current_stream_wait() ── 消费 recv_x
              │(就绪事件)                                          ▲
              │previous_event                                      │
通信流：      └▶ dispatch 内核（等输入就绪 → 计数 → 搬运）          │
                        │ notify 计数（red_add → workspace）        │
                        │ mbarrier：槽位数据落定                    │
                        │ CPU sync（do_cpu_sync=True 时）           │
                        └▶ epilogue（defer 时在 wait 处执行）───────┘
```

每一条箭头都对应本章的一个原语：捕获/等待是事件体系，计数是原子操作，落定是 mbarrier，冲刷与清零是两相位协议，epilogue 是 TMA/批量存储。

## 10.10 本章小结

- warp 原语（`get_warp_idx`/`get_lane_idx`/`elect_one_sync`）支撑"每 warp 一份任务"的组织；`trap()` 是设备端不可恢复错误的终点。
- mbarrier（块内、相位奇偶等待）承载 TMA 完成通知与组内生产者-消费者同步；`TokenLayout` 甚至把 mbarrier 内嵌进 token 槽位。
- 全局屏障用 `BarrierSignals` 两相位协议（计数器 + 交替标志）实现，超时以 `kNumOneSecCycles` 换算周期、以标签区分调用来源，统一走 `timeout_while`。
- TMA 与 `st_bulk` 把搬运卸出 SM 指令流，对齐纪律（32 字节）与 `use_tma_aligned_col_major_sf` 让通信布局直接服务计算布局；`iterate_ld_st` 是手工搬运的通用分块模板。
- 原子计数两个世界：NVLink `red_add` 与 RDMA Gin 原子（`check_fast_rdma_atomic_support` 决定 QP 减半）；计数器槽位固定、对齐固定、复用靠相位/清零协议。
- 事件体系（通信流单例 + `EventOverlap` + `previous_event`/`allocate_on_comm_stream`/`defer_epilogue`）把"何时等待"变成用户可编程的决策点；`EP_AVOID_RECORD_STREAM` 提供登记开销与用户纪律的交换。

## 10.11 延伸讨论：同步协议的验证视角与占用的定界

**为什么这些协议"看起来"难以验证**。通信内核的同步原语（mbarrier 相位、两相位屏障、issue 计数器清零、环形槽位计数）都属于"无锁并发协议"，直接证明其正确性需要论证不变式、进展性与终止性三类性质。DeepEP 的做法是用**结构**降低验证难度：其一，所有协议都建立在"计数器 + 相位"这一个原语族上（mbarrier 的 parity、`BarrierSignals` 的双标志、Engram 的清零信号、PP 的累计计数）， review 时只需掌握一种推理模式；其二，不变式集中写在信号区结构的字段注释里（例如 `PPSignals` 的"累计发送计数决定当前使用的环形槽位"、`EngramWorkspace` 的"最后一个到达的 warp 负责 flush QP 并把计数清零"），字段语义即协议契约；其三，所有自旋等待统一走 `timeout_while` 并带标签诊断，把"协议损坏"从死循环转化为可报告的错误。对集成方而言，这意味着怀疑同步问题时，第一手证据永远在 workspace 的信号区与超时报错的标签里。

**占用（occupancy）的定界**。`__launch_bounds__(kNumThreads, 1)` 锁定每 SM 一个 block，这不是牺牲灵活性而是通信内核的固有需求：通道到 SM 的映射是静态的（每个通道的 warp 组固定归属一个 SM 或固定一组 SM），如果编译器为了提高占用率而压缩每线程寄存器，通道的寄存器预算（hybrid combine 的 40/216 分组）就会失真，导致隐蔽的性能回退。`EP_JIT_CHECK_NO_SPILLS`/`EP_JIT_CHECK_NO_LOCAL_MEMORY` 两个质量门与占用声明共同构成"寄存器预算的三重保险"：设计期手写预算、编译期锁占用、加载期拒溢出。这个组合可以作为编写任何延迟敏感内核的通用模板。

**通信流单例的一个推论**。`get_comm_stream` 是进程级唯一的高优先级流，这带来一个重要推论：**同一进程内所有 DeepEP 缓冲区的通信天然串行**。若同时持有 EPBuffer 与 BucketBuffer，两者的事件在同一时间轴上排队——这不是缺陷而是特性：它让跨缓冲区的事件推理变得简单（通信流上的顺序即时间顺序），也让"计算流与通信流"的两流模型在整个进程内保持一致。集成方若需要真正的通信并行（例如两个 EP 组同时通信），应当使用多进程（每个 EP 组一个进程），而不是指望单进程内的多流并行——README 的多节点测试约定（每节点启动同一脚本、RANK 对应节点索引）正是多进程模型的体现。

**`trap()` 的位置**。设备端 `trap()` 只应出现在"协议已损坏、继续执行必然产生错误数据"的场景（非法槽位、计数器异常），而不是普通的等待超时——超时应当走 `timeout_while` 的可恢复错误路径，让上层有机会重试或报错重置。区分这两类失败，是把 DeepEP 集成进长运行训练系统时容错设计的分界线：前者意味着进程必须退出（数据完整性已不可信），后者意味着本步操作失败但系统状态仍可诊断。

## 10.12 延伸讨论：可见性与栅栏——同步协议的另一半

前文反复出现"完成通知"，但通知只解决"信号何时可见"，还有一半问题是"数据何时可见"——即内存一致性与栅栏。GPU 的内存模型允许编译器与硬件对普通读写重排，跨线程、跨 SM 乃至跨 GPU 的"先写后通知"必须保证写操作在通知发出前对通知的读者可见。DeepEP 的设备代码为此依赖三类工具：PTX 层的内存栅栏（`ptx.cuh` 的封装清单中包含内存栅栏原语，第 10.6 节 Engram 内核注释明确列出"PTX 工具（warp 索引、mbarrier、内存栅栏）"为依赖）、原子操作的顺序语义（原子读改写自带串行化，`issue_counters` 的清零与 `BarrierSignals` 的标志都借用了这一点）、以及 mbarrier 硬件自带的完成语义（TMA 拷贝完成与 mbarrier 到达的绑定由硬件保证顺序）。三者合起来，让"写数据 → 发通知 → 读通知 → 读数据"这条链在每一跳都有明确的可见性保证。

为什么集成方也要理解这一层？因为任何"绕过库自己写内核"的扩展（例如自定义的数据后处理直接读 `recv_x`）都可能引入可见性假设错误：读 `recv_x` 之前必须先建立与通信流的同步（`current_stream_wait`，第 10.7 节），这既是逻辑依赖也是内存可见性依赖——CUDA 流的事件等待自带全内存栅栏语义，因此正确用法天然安全；但若有人为了"省一次等待"而用非同步方式猜测数据就绪，就会遇到不可复现的数据竞争（第 14 章数据正确性类问题的底层形态）。记住这个判据：**跨流的数据消费必须有事件等待；同流内的事件等待兼管逻辑与可见性**。

## 10.13 延伸讨论：超时设计的分层哲学

第 10.3 节介绍了 `timeout_while` 与超时周期换算，本节讨论超时值的分层设计，因为 `num_cpu_timeout_secs=300` 与 `num_gpu_timeout_secs=100`（构造参数，第 3 章）这一对默认值本身就体现了设计意图。CPU 侧超时（300 秒）覆盖的是"主机同步等待通信完成"的场景，它面对的故障通常是全局性的（某 rank 掉线、网络分区），重启成本高，因此容忍期长；GPU 侧超时（100 秒）覆盖的是内核内自旋等待（屏障、计数、槽位），它面对的故障既可能是全局性的也可能是本地的（参数不匹配导致等待条件永假），且 GPU 自旋占用 SM——等待越久，浪费的算力越多，因此容忍期短、失败要早。两个数值都不是随意的：CPU 侧要覆盖"大集群一轮 NCCL 超时重试"的时长，GPU 侧要在"误报风险"与"算力浪费"之间取衡。`kNumOneSecCycles` 的换算（按 2 GHz 估算，第 10.3 节）则提示超时的度量基准是时钟周期而非挂钟时间——GPU 降频或 MIG 分片会改变换算关系，极端环境下排查"超时误报"时应把这一变量纳入（第 14 章诊断习惯的第二条：读完整断言信息，周期换算常在其中）。

下一章进入让这一切成为可能的编译设施：DeepJIT 运行时编译。
