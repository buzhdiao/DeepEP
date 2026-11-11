# 第 03 章 · BufferBase 生命周期与 BufferAllocator 对称张量布局规划

## 3.1 问题：通信资源为什么难管

一个 DeepEP 缓冲区对象背后压着几类昂贵且危险的资源：

- **NCCL 通信域**（`ncclComm_t`，经 `deep_ep/comm/handle.py` 的 `NCCLCommHandle` 持有，`get()` 返回 int64 伪装的指针）；
- **对称内存**：一块被全组注册的显存（NCCL window，`ncclWindow_t`），以及其下由 CUDA VMM（虚拟内存管理）物理页构成的分配；
- **RDMA 资源**：Gin 上下文（每个 QP 一个，`ginContextCount = num_allocated_qps`）、QP 深度（`ginQueueDepth`）、窗口数组（`rdma_storage_windows_gpu`）；
- **GPU 侧异步状态**：内核在通信流上运行，信号区（workspace）中的计数器可能还有"在途"使用者。

这些资源共同决定了销毁顺序极其敏感：若 GPU 上还有内核在读写窗口，窗口不能释放；若派生类的成员已先析构，`destroy()` 里再访问它们就是未定义行为。V2.5 之前，这些逻辑散在单一的 ElasticBuffer 里；V2.5 把生命周期抽成两层薄基类——Python 的 `BufferBase`（`deep_ep/buffers/base.py`）与 C++ 的 `BufferBase`（`csrc/buffers/base.hpp`）——让四个缓冲区类（EP/Engram/PP/Bucket）共享同一套构造—同步—销毁语义。

## 3.2 C++ 侧：destroy_on_destruction 与析构顺序

`csrc/buffers/base.hpp` 的 C++ `BufferBase` 只有三个成员与一个关键方法：

```cpp
class BufferBase {
protected:
    bool explicitly_destroy;          // 是否要求用户显式调用 destroy()
    bool destroyed = false;           // 防止重复销毁
    explicit BufferBase(const bool& explicitly_destroy)
        : explicitly_destroy(explicitly_destroy) {}

    // Called from the derived destructor, while its members and destroy()
    // implementation are still available.
    void destroy_on_destruction(const char* name) {
        if (not explicitly_destroy and not destroyed) {
            destroy();
            main_context = nullptr;
        }
        if (not destroyed) {
            std::printf("`destroy()` is not called before DeepEP %s buffer destruction, "
                        "which can leak resources.\n", name);
            std::fflush(stdout);
        }
    }
};
```

三个设计点值得逐条论证：

**（1）为什么在派生类析构函数体内调用 `destroy_on_destruction`，而不是在基类析构函数里调 `destroy()`？** C++ 的析构顺序是"派生类成员先析构，然后是派生类体，然后才是基类"。若在基类析构中调用虚函数 `destroy()`，此时派生类对象已经"死"了一半，虚表退化到基类版本，派生类成员不可用——这是经典陷阱。源码注释明确写着 *"Called from the derived destructor, while its members and destroy() implementation are still available"*：每个派生类（如 `csrc/buffers/ep.hpp` 的 `deep_ep::ep::EPBuffer`）在自己的析构函数**体**里调用 `destroy_on_destruction("EP")`，此刻派生类的成员（窗口句柄、workspace 指针等）仍然完好，`destroy()` 的虚调用/普通调用都安全。

**（2）为什么"默认自动销毁 + 显式销毁双模式"？** 训练框架里缓冲区常由全局变量或缓存持有，Python 解释器退出时的析构时机不可控；而 NCCL 通信域、RDMA 窗口的析构要求进程处于合理状态。于是 API 提供 `explicitly_destroy=True`：要求用户在合适的时机（所有在途操作完成后）手动 `destroy()`；若用户声明了显式销毁却忘了调用，析构时打印泄漏警告并 `fflush` 立即输出——宁可留下可见的警告，也不做危险的隐式释放。Python 侧的抽象 `destroy()`（`deep_ep/buffers/base.py`）还有一道 `assert self.explicitly_destroy`，防止误调用。

**（3）`destroyed` 标志与 `main_context` 置空。** `destroy()` 可能被用户显式调用一次、再经析构路径走到一次，`destroyed` 保证幂等；销毁后把主上下文共享指针置空，打破引用环并保证 `main_context` 属性此后返回 `None`（Python 侧 `main_context` property 同样做了 `runtime is not None` 判断）。

## 3.3 Python 侧：抽象基类与 runtime 句柄

```python
class BufferBase(ABC):
    def __init__(self, explicitly_destroy: bool = False):
        self.explicitly_destroy = explicitly_destroy
        self.runtime = None                # C++ 运行时句柄，构造后由子类填充

    @property
    def main_context(self):
        return self.runtime.main_context if self.runtime is not None else None

    def print_memory_usage(self) -> None:
        assert self.runtime is not None
        self.runtime.print_memory_usage()   # workspace / 通信缓冲 / RDMA 存储三行

    @abstractmethod
    def destroy(self) -> None:
        assert self.explicitly_destroy
        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None
```

`print_memory_usage` 委托给 C++ 运行时打印三部分占用：**workspace（信号区）**、**通信缓冲区（GPU buffer）**、**RDMA 存储区**。这正好对应 `Context` 的三个只读属性（`csrc/comm/handle.hpp`：`num_workspace_bytes`、`num_gpu_buffer_bytes`、`num_rdma_storage_bytes`）——Python 与 C++ 对"一块缓冲区由几部分组成"的认知再次对齐。

## 3.4 以 EPBuffer 为例的完整构造时序

`deep_ep/buffers/ep.py` 的 `EPBuffer.__init__` 是四个缓冲区里最复杂的构造，其时序是理解资源模型的最佳标本：

```text
EPBuffer(group, num_max_tokens_per_rank=T, hidden=h, num_topk=k,
         use_fp8_dispatch=fp8, lb_allocation_plan_or_num_bytes=…, …)
 │
 ├─1 self.nccl_comm_handle = comm.get_nccl_comm_handle(group)
 │     EP_REUSE_NCCL_COMM=1（默认）且 PyTorch PG 暴露 _comm_ptr 时直接复用；
 │     否则创建由 DeepEP 管理的 NCCL 通信域（managed 句柄，__del__ 兜底）
 │
 ├─2 num_bytes 未显式给出时：
 │     num_bytes = _C.calculate_ep_buffer_size(nccl_comm, T, h, k, fp8,
 │                       allow_hybrid_mode, allow_multiple_reduction)
 │     —— 按 MoE 配置与域拓扑解析式计算，结果 2MB 对齐
 │
 ├─3 LB 区域解析：BufferAllocator 计划或字节数（必须未 materialize），
 │     与主缓冲区相互独立（详见第 7 章）
 │
 ├─4 check_nvlink_connections(group)：校验节点内 NVLink 连通
 │     （若实际是 PCIe 互联，此处会给出诊断）
 │
 ├─5 QP 预分配数（num_allocated_qps==0 时自动）：
 │     hybrid 模式：支持快速 RDMA 原子 → 65，否则 129
 │       （每个 channel 一个 QP + notify warp 专用 QP；原子快路径减半）
 │     非 hybrid：17（8 数据 QP + 备用）
 │
 ├─6 self.runtime = _C.EPBuffer(rank_idx, num_ranks, nccl_comm, num_bytes,
 │                 num_lb_bytes, allow_hybrid_mode, allow_multiple_reduction,
 │                 prefer_overlap_with_compute, sl_idx, num_allocated_qps,
 │                 num_cpu_timeout_secs=300, num_gpu_timeout_secs=100,
 │                 explicitly_destroy)
 │     —— C++ 侧：分配对称内存并注册 window、建立 Gin 上下文/QP、
 │        构建 workspace 信号区，产出 self.runtime.context
 │
 ├─7 传入 LB 计划则 lb_allocation_plan.materialize(self.runtime.lb_storage)
 │
 ├─8 逻辑/物理域拆分：
 │     num_scaleout_ranks, num_scaleup_ranks = get_logical_domain_size()
 │     scaleout_rank_idx = rank_idx // num_scaleup_ranks
 │     scaleup_rank_idx  = rank_idx %  num_scaleup_ranks
 │     num_rdma_ranks, num_nvlink_ranks = get_physical_domain_size()
 │
 └─9 全局握手：torch.cuda.synchronize() → group.barrier() → torch.cuda.synchronize()
       保证所有 rank 的初始化（窗口注册、QP 建立）互相可见后才放行
```

第 9 步的"双重同步"不是仪式感：NCCL 通信域与 RDMA 连接的建立涉及跨节点握手，任何 rank 提前进入第一次 dispatch 都可能在远端窗口未就绪时发起访问。构造函数以"两侧各一次设备同步 + 一次组屏障"换取后续所有通信的无锁前提。

## 3.5 对称内存：VMM、fd 共享与 NCCL window

DeepEP 所有缓冲区的显存都来自 `csrc/kernels/comm/symmetric.hpp` 的对称内存分配器（`deep_ep::symmetric` 命名空间）。其核心思想是：**每个 rank 按完全相同的顺序、相同的大小分配一段段显存，并把每段的物理页导出为文件描述符在全组交换**，最终每个 rank 拿到"同布局、可互访"的地址空间。

实现要点（源码可见）：

- **VMM 分配**：走 CUDA Driver API（`cuda.h`），`DeviceContext` 结构在构造时查询 `cudaGetDevice`、`lazy_cuDeviceGet` 与设备所属 **NUMA 节点**（`device_idx`、`CUdevice device`、`numa_idx`）——物理页放置靠近网卡所属 NUMA 节点对 RDMA 带宽有直接影响。
- **跨进程共享**：`shared_handle_t = std::pair<int, int>` 即 `(pid, fd)`；`shared_comm_t = std::vector<shared_handle_t>` 是全组句柄列表。进程间传 fd 使用 `pidfd` 系列系统调用（源码包含 `<unistd.h>`、`<sys/syscall.h>` 并注释了 pidfd 用途）——每个 rank 打开对端进程的 fd，把同一段物理页映射进自己的虚拟地址空间。
- **NCCL window**：对称内存交给 NCCL 注册为 `ncclWindow_t`；持有通信域与窗口后，设备端代码可通过 `Context::get_sym_ptr(void* ptr, const int& dst_nvl_rank_idx)`（`csrc/kernels/comm/context.cpp`）把"本地指针 + 对端 rank"换算成对端虚拟地址。多块 RDMA 存储分片则注册成窗口数组 `rdma_storage_windows_gpu`（大小为 `gin_min_stride`，即 Gin 要求的窗口步长），供 Engram 等多分片场景按窗口索引访问（第 8 章）。

**对称性**是 DeepEP 一切"直达寻址"通信的前提：因为所有 rank 的布局一致，"把 token 写到 rank r 的第 s 个槽位"才能表达为"向 `sym_ptr(本地缓冲, r) + s * stride` 写入"这样一次远程写，而不需要任何协商。这也是第 3.7 节 `BufferAllocator` 强制"全组计划一致"的根本原因。

## 3.6 BufferAllocator：先规划，后落实

`deep_ep/buffers/allocator.py` 的 `BufferAllocator` 解决的问题非常具体：**一块缓冲区里要放下若干个形状/类型各异的张量（信号区、数据区、LB 冗余区……），且这些张量的相对布局必须全组一致**。它的 API 只有三个方法，却构成一个两阶段协议：

```python
class BufferAllocator:
    def __init__(self):
        self.alignment = _C.get_num_allocation_alignment()  # 底层对齐粒度
        self.allocations = []        # [(meta 占位张量, 字节偏移), …]
        self.num_bytes = 0           # 累计（含对齐间隙）
        self.materialized = False

    def allocate(self, shape, dtype) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype, device='meta')   # 占位
        offset = align(self.num_bytes, self.alignment)
        self.allocations.append((tensor, offset))
        self.num_bytes = align(offset + tensor.nbytes, self.alignment)
        return tensor

    def materialize(self, storage: torch.Tensor) -> None:
        assert storage.is_cuda and storage.is_contiguous() and storage.dtype == torch.uint8
        assert storage.numel() == self.num_bytes
        for tensor, offset in self.allocations:
            value = storage.narrow(0, offset, tensor.nbytes) \
                        .view(tensor.dtype).view(tensor.shape)
            torch.utils.swap_tensors(tensor, value)   # 原地换成真实存储视图
        self.materialized = True
```

两阶段的本质是**把"声明式布局"与"物理分配"解耦**：

```text
阶段一（规划）：                      阶段二（落实）：
  plan = BufferAllocator()            buffer = BucketBuffer(group, plan)
  x  = plan.allocate((P,1024),f32)  →   materialize(runtime 的 uint8 storage)
  sf = plan.allocate((P,64),f32)        ┌────────────────────────────┐
  （x/sf 是 meta 张量：有形状类型，      │ storage: 一块连续 CUDA u8   │
    无存储，可参与形状推导）            │ [x 1024*4B][pad][sf 64*4B] │
                                       └────────────────────────────┘
                                       x/sf 变成 storage 的 narrow 视图
```

这一设计的四条理由：

1. **全组一致性可校验**。README 对 BucketBuffer 明确要求"所有 rank 必须使用相同的分配规划"、对 EP 的 LB 区域要求"域内每 rank 相同的形状与分配顺序"。计划是纯 Python 对象，容易比对（形状序列 + dtype 序列 + 对齐偏移是确定性函数），比比对"各 rank 各自 malloc 的地址"容易得多。
2. **单次注册**。所有张量最终都落在**一块**连续 CUDA uint8 存储里，对 NCCL window/RDMA 注册而言是**一次**注册成本；若各张量独立分配，注册次数、碎片、页表开销都会放大。
3. **对齐可控**。每次分配的偏移都 `align` 到 `_C.get_num_allocation_alignment()`（缓冲区大小计算同样 2MB 对齐），保证每个子张量的起始地址满足 RDMA/TMA 的对齐要求——TMA 与 RDMA 对不齐的数据会直接报错或降速（第 4 章的 TokenLayout 也遵循同样的对齐纪律）。
4. **meta 张量作为"未来指针"**。`allocate` 返回的 meta 张量在规划期就可以被上游代码持有、切片、比较形状；`materialize` 用 `torch.utils.swap_tensors` 原地替换其存储，使得**先前拿到占位张量的代码在落实后自动指向真实内存**，调用方无需改写任何下游逻辑。这是把"延迟绑定"做进张量语义里的优雅做法。

`materialize` 的三条断言（CUDA、连续、uint8；数量精确相等；只允许一次）把违约场景全部前置成显式错误。落实后再调用 `allocate` 会被 `assert not self.materialized` 拒绝——计划是不可变的，这是对称性的另一面。

## 3.7 缓冲区大小的解析式计算

`EPBuffer.get_buffer_size_hint(group, num_max_tokens_per_rank, hidden, num_topk=…, use_fp8_dispatch=…, allow_hybrid_mode=…, allow_multiple_reduction=…)` 返回推荐缓冲区字节数（2MB 对齐），其实现委托 `_C.calculate_ep_buffer_size`。它的存在解决一个工程问题：**上层框架希望在不真正创建缓冲区的情况下判断"现有缓冲区能否复用"**。README 的示例正是这样写的：

```python
required_bytes = EPBuffer.get_buffer_size_hint(group, num_max_tokens_per_rank, hidden, …)
if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
    return _buffer   # 复用
```

影响缓冲区大小的因子可以从签名直接读出：每 rank 最大 token 数决定槽位总数；hidden 与 dtype（BF16 还是 FP8 + scale factor）决定每槽位字节数——这正是第 4 章 `TokenLayout` 的 `[hidden | SF | metadata]` 结构；hybrid 模式需要为跨节点一级中转留出双份窗口空间；`allow_multiple_reduction` 影响 combine 的归约布局，进而影响接收缓冲的形态（第 5 章）。缓冲区大小**与瞬时批次无关**——它按容量上限分配，这正是"统一接口覆盖训练、prefill、解码"（README：所有 rank 必须就 `num_max_tokens_per_rank` 达成一致）的前提。

## 3.8 与其他章的接口

- 构造时拿到的 `self.context`（`comm::Context`）是后续一切通信的"通行证"：QP 数、窗口、workspace、多播（multimem）缓冲都挂在它上面（第 6、10 章）。
- `EPBuffer` 的 LB 区域（`lb_allocation_plan_or_num_bytes`）在构造阶段由 `BufferAllocator` 落实到 `runtime.lb_storage`，第 7 章的权重预取/梯度归约内核直接在这块对称存储上工作。
- `destroy()` 的幂等语义与"先等在途操作再销毁"的约定（README：*"在替换其缓冲区之前，请先完成所有未完成的操作"*）是第 10 章事件模型的使用纪律在生命周期上的投影。

## 3.10 延伸讨论：页粒度、注册成本与对齐常量的来源

对称内存体系的三个"魔法数字"——分配对齐（`get_num_allocation_alignment`）、RDMA 对齐（`kNumRDMAAlignmentBytes`）、TMA 对齐（32 字节，`deep_ep.get_num_tma_alignment()`）——各自有明确的硬件与协议来源，理解它们有助于在做容量规划时把浪费算清楚。分配对齐与页粒度相关：VMM 分配（`cuMemCreate` 一族）以物理页为单位，2 MB 是常见的 大页 粒度，`get_buffer_size_hint` 返回值"按 2 MB 对齐"（第 3 章）正是为了避免跨页碎片与映射表膨胀——每个对齐缝隙最多浪费一个页粒度减一字节，但换来的是映射项数的最小化。RDMA 对齐源于网卡与协议对传输描述符的要求：RDMA 写的目标地址与长度通常要求满足特定对齐，未对齐要么被拒绝要么退化为低效路径，因此 `TokenLayout` 的槽位总大小、`EPSignals` 等信号区结构都显式 `alignas(kNumRDMAAlignmentBytes)`（第 4 章）。TMA 对齐（32 字节）则源于张量拷贝引擎描述符的约束——`lb_prefetch_weights` 要求每专家字节数是它的倍数（第 7 章）、PP 的输入输出要求 32 字节对齐指针与大小（第 9 章），都是同一约束在不同接口上的投影。

注册成本是容量规划的另一维度。一块显存要成为"对端可直达"的对称内存，需要经历 VMM 分配、页映射、fd 交换、NCCL window 注册多个步骤，其中跨进程的 fd 交换与注册本身是**构造期一次性成本**，这也是"先规划后落实"设计的深层动机：把 N 个张量的 N 次注册压缩成一块存储的一次注册（第 3.6 节理由 2）。运行时频繁分配释放"临时通信张量"的方案之所以被 session 暂存区替代（第 9 章），正是因为每次注册与解注册的代价远大于一次拷贝。规划显存时应当把"注册过的存储"视为不可轻易回收的资产：`destroy()` 的幂等与"先等在途操作"的纪律（第 3 章）本质上都是在保护这份资产的完整性。

## 3.11 延伸讨论：多缓冲区共存时的资源预算

一个进程内同时存在多个缓冲区是常态（不同 EP 组、EP 加 Bucket、EP 加 Engram），此时资源预算要做三笔账。第一笔是**显存**：各缓冲区的主存储、LB 区域、workspace 相加，加上注册的 Engram 分片（若在 GPU）；`print_memory_usage` 按缓冲区逐个打印（第 3 章），乘以缓冲区个数即总账。第二笔是**通信域与 QP**：每个缓冲区持有自己的 Context 与预分配 QP（65/129/17，第 6 章），多缓冲区线性叠加；网卡侧的 QP 总量与内存锚定（pinned memory）有上限，超限会表现为创建失败或性能抖动——规划时按"缓冲区数 × 预分配数"核对网卡配置。第三笔是**流与同步**：`get_comm_stream` 是进程级单例（第 10 章），多缓冲区的通信天然在同一流上排队——这既简化了推理（通信流顺序即时间顺序），也意味着一个缓冲区的长通信会推迟另一个缓冲区的通信启动，容量规划里要把"同时活跃的缓冲区"数量计入时延预算。三笔账合起来，构成了第 19 章容量规划中"显存五项分解"之外的进程级视角。

## 3.13 延伸讨论：构造期失败的自诊断与恢复

构造函数是资源获取最密集的路径（第 3.4 节九步），本节把"构造失败"的诊断与善后讲清——它与运行期故障的处置逻辑不同，因为此时系统尚未进入服务状态。失败点按九步顺序排查：第一步通信域获取失败的报错来自 NCCL bootstrap（unique id 交换超时，`MASTER_ADDR`/`PORT` 与网络可达性，第 12 章约定）；第二步容量计算的失败极罕见（纯计算）；第四步 NVLink 检查的诊断要读完整（PCIe 互联是环境问题不是代码问题，第 6.5 节）；第六步 C++ 运行时创建是失败集中区——对称内存的 VMM 分配失败（显存不足或碎片化）、fd 交换失败（pidfd 受限的容器环境）、window 注册失败（NCCL 版本或 Gin 初始化问题，`EP_DISABLE_GIN` 的适用判断）、QP 建立失败（网卡资源不足，第 3.11 节第二笔账）——每一步的断言都携带具体条件（`EP_HOST_ASSERT` 的文件行号，第 10 章）。善后的关键认知是：**构造是全组协同的（末尾双重同步），单个 rank 的构造失败会让其余 rank 卡在屏障直到 CPU 超时（`num_cpu_timeout_secs`，第 3 章）**——因此失败方要尽快退出并让失败可见（异常传播到框架的全局失败处理，第 14.9 节第三道防线），其余 rank 依赖超时与失败检测结束等待。恢复策略：构造失败不重试原地（资源可能处于半初始化态），正确做法是进程级重启并核对第 19 章容量手册的对应项——把"构造期失败"当作规划错误的信号而不是瞬时故障，是本节与第 14 章"运行期故障可重试"的关键区分。

## 3.14 延伸讨论：从对象模型看 Python 与 C++ 的分工艺术

第 3 章的两层基类结构（Python `BufferBase` 与 C++ `BufferBase`）展示了一种值得细品的对象模型分工：Python 持有"身份"，C++ 持有"实体"。具体而言：Python 侧对象（`EPBuffer` 等）持有 `self.runtime`（C++ 运行时句柄）与各种配置快照（`num_bytes`、`num_allocated_qps`、域拆分结果），它是用户操作的入口与语义校验层；C++ 侧对象持有全部系统资源（对称内存、窗口、QP、上下文），它在自己的析构函数体内调用 `destroy_on_destruction` 保证资源释放顺序（第 3.2 节）。两层之间的引用方向是单向的（Python 指 C++），加上 `destroy()` 的幂等（`destroyed` 标志）与置空（`main_context = None`），构成一个无环的所有权图——这正是垃圾回收语言与手动资源语言协作的标准姿势：**Python 的 GC 永远不会"替你"释放需要精确时序的资源，因为资源的析构权在 C++ 析构链里，而 C++ 析构只在 Python 句柄归零后发生，且被 `explicitly_destroy` 纪律约束在用户选择的时机**。理解这个模型还能解释一个 API 细节：为什么 `destroy()` 在 Python 侧是抽象方法且断言 `explicitly_destroy`（第 3.3 节）——因为 Python 层无法安全地替 C++ 层决定"现在释放是否安全"（GPU 上可能有在途内核），这个判断只能由用户显式做出或由 C++ 析构链兜底并警告。第 3 章的这两个基类加起来不足两百行，却示范了跨语言资源管理的完整答案，值得任何 Python/C++ 混合项目借鉴。



## 3.12 本章小结

- 通信资源的销毁顺序敏感：C++ `BufferBase` 用"派生类析构体内调用 `destroy_on_destruction`"规避虚函数退化陷阱，用 `explicitly_destroy`/`destroyed` 提供"显式销毁 + 幂等 + 泄漏警告"三重语义；Python `BufferBase` 把 runtime 句柄、`main_context`、内存打印、抽象 destroy 统一。
- EPBuffer 构造九步：通信域获取（可复用 `_comm_ptr`）→ 解析式容量计算 → LB 区域 → NVLink 校验 → QP 预分配（65/129/17）→ C++ 运行时（窗口/Gin/QP）→ LB 落实 → 域拆分 → 双重同步握手。
- 对称内存基于 CUDA VMM + 跨进程 fd 共享（`(pid, fd)` 句柄、NUMA 感知、pidfd），经 NCCL 注册为 window，`get_sym_ptr` 完成"本地指针 → 对端地址"的换算；对称性是一切直达寻址通信的前提。
- `BufferAllocator` 用"meta 占位 → 计划 → `swap_tensors` 落实"的两阶段协议，把声明式布局与单块连续存储的物理分配解耦，同时满足全组一致、单次注册、对齐可控、延迟绑定四个目标。
- 缓冲区大小按容量上限解析式计算（`calculate_ep_buffer_size`），与瞬时批次无关，支持 `get_buffer_size_hint` 式的复用判断。

下一章进入 DeepEP 最核心的链路：`EPBuffer.dispatch` 从 top-k 路由到接收端元数据的全过程。
