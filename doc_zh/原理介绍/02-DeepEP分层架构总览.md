# 第 02 章 · DeepEP 分层架构总览

## 2.1 为什么需要分层：一个问题域，四种异构约束

第 1 章结尾列出的八条约束，分别压在四种"异构"的交界面上：

1. **Python 与 C++ 的异构**：MoE 框架（训练/推理引擎）在 Python 侧编排，生命周期管理、参数校验、张量所有权、事件语义最好用 Python 表达；而性能攸关的运行时（通信域创建、内存注册、内核启动）必须在 C++。
2. **宿主与设备的异构**：通信域与对称内存的建立是宿主端（CPU）操作（CUDA 驱动、NCCL 主机 API）；而计数、同步、数据搬运是设备端（GPU）操作（CUDA 内核、PTX 指令）。两边对"布局"的理解必须严格一致——这是 `deep_ep/include/deep_ep/layout/` 目录存在的意义。
3. **配置与编译的异构**：EP 内核的行为被 MoE 配置（hidden、topk、rank 数、FP8 开关、布局开关……）参数化到模板层面，预编译所有组合不现实；因此设备端代码以**头文件模板**形式存在，运行时由 DeepJIT 按当前配置与当前 GPU 编译。
4. **不同并行方式的异构**：EP 要 all-to-all，PP 要点对点，CP/DP 要批量集合通信，Engram 要按索引远程读。四者共享"对称内存 + NCCL 通信域 + 屏障 + 事件"的地基，但上层协议完全不同。

DeepEP 的分层正是对这四组约束的直接回答：**Python 缓冲区层**管生命周期与语义；**C++ 宿主层**管运行时与内核启动；**设备端模板层**管内核与布局；**DeepJIT 层**把后两者在运行时缝合。

```text
┌─────────────────────────────────────────────────────────────────────┐
│  用户代码（MoE 训练/推理框架）                                        │
│      import deep_ep; buffer = EPBuffer(group, …); buffer.dispatch(…) │
├─────────────────────────────────────────────────────────────────────┤
│  L1 Python 层   deep_ep/                                             │
│    buffers/{base,allocator,ep,engram,pp,bucket}.py                   │
│    comm/{handle,stream,barrier}.py   utils/{event,envs,math,…}.py    │
│    职责：生命周期、参数校验、布局规划（BufferAllocator）、事件语义、    │
│         拓扑/带宽探测（envs.py）、参考实现（refs.py，仅测试用）        │
├────────────────────── pybind11 边界（deep_ep._C）────────────────────┤
│  L2 宿主 C++ 层  csrc/            （安装期由 CUDAExtension 编译）     │
│    python_api.cpp → buffers/comm/runtime/utils 的绑定                │
│    buffers/{base,ep,engram,pp,bucket}.hpp  运行时与资源管理           │
│    kernels/{comm,ep,bucket,engram,pp}/   launch_* 内核启动封装        │
│    kernels/comm/{symmetric.hpp,context.cpp} 对称内存 + NCCL 通信域    │
│    runtime/jit.hpp  DeepJIT 入口（LazyInit<Runtime<CUDA>>）           │
├────────────────────── JIT 编译边界（nvcc，运行时）───────────────────┤
│  L3 设备端模板层  deep_ep/include/deep_ep/（纯头文件）                │
│    common/{compiled,math,ptx,utils,exception}.cuh                    │
│    layout/…  TokenLayout、EPSignals、BucketSignals、EngramWorkspace… │
│    impls/…   dispatch/combine/hybrid_*/epilogue/bucket/engram/pp     │
│    comm/{handle.cuh(NCCLGin), barrier.cuh}  设备端通信原语            │
├─────────────────────────────────────────────────────────────────────┤
│  L4 硬件：CUDA Driver（VMM/对称内存）· NCCL Gin（RDMA/多播）·          │
│          NVLink · RDMA 网卡 · 拷贝引擎（CE）                          │
└─────────────────────────────────────────────────────────────────────┘
```

## 2.2 Python 层：包结构与导出面

`deep_ep/__init__.py` 是对外契约，它做的第一件事不是导出 API，而是**环境与依赖体检**：

- 加载 `persistent_envs`（安装期由 `setup.py` 烘焙进 `deep_ep/envs.py` 的默认环境变量，如 `EP_JIT_CACHE_DIR`、`EP_NUM_TOPK_IDX_BITS`），用户显式设置过的变量优先；
- `check_nccl_so()` 读取 `/proc/self/maps`，断言进程内只加载了一份 `libnccl` 且与链接版本一致（可用 `EP_SUPPRESS_NCCL_CHECK=1` 跳过）——因为 DeepEP 与 PyTorch 必须共享同一个 NCCL 运行库，否则设备端 API 的窗口句柄会跨库失效。

随后导出用户可见的全部类型：

```python
from .buffers.allocator import BufferAllocator
from .buffers.base import BufferBase
from .buffers.ep import EPBuffer, EPHandle
from .buffers.engram import EngramBuffer
from .buffers.bucket import BucketBuffer, BucketSession
from .buffers.pp import PPBuffer
from .utils.event import EventOverlap, EventHandle
from deep_ep._C import …   # get_num_allocation_alignment 等底层查询与类型
```

Python 各子模块的职责划分（后续章节逐一展开）：

| 模块 | 关键内容 | 详见 |
| --- | --- | --- |
| `buffers/base.py` | `BufferBase` 抽象基类：`explicitly_destroy`、`runtime`、`main_context`、`print_memory_usage`、抽象 `destroy` | 第 3 章 |
| `buffers/allocator.py` | `BufferAllocator`：meta 占位张量 → 计划 → `materialize` 落实 | 第 3 章 |
| `buffers/ep.py` | `EPBuffer`、`EPHandle`：dispatch/combine、SM/QP 解析式估算、EPLB 原语 | 第 4~7 章 |
| `buffers/engram.py` | `EngramBuffer`：`set_config`/`write`/`fetch`、`get_storage_size_hint`、`get_theoretical_config` | 第 8 章 |
| `buffers/pp.py` | `PPBuffer`：`send`/`recv` | 第 9 章 |
| `buffers/bucket.py` | `BucketBuffer`、`BucketSession`：all_gather/reduce_scatter/all_reduce | 第 9 章 |
| `comm/handle.py` | `NCCLCommHandle`、`get_nccl_comm_handle`（复用 `_comm_ptr`）、物理/逻辑域查询 | 第 6 章 |
| `comm/stream.py` / `comm/barrier.py` | 通信流与组屏障的 Python 门面 | 第 10 章 |
| `utils/event.py` | `EventOverlap`：`current_stream_wait`/`wait`/`register_hook_after_wait` | 第 10 章 |
| `utils/envs.py` | `init_dist` 约定、带宽探测（`get_nvlink_gbs` 等）、`check_nvlink_connections`、`check_fast_rdma_atomic_support`、`EnvOverride` | 第 6/12 章 |
| `utils/refs.py` / `utils/gate.py` / `utils/testing.py` | 参考实现、负载生成器、基准工具（`bench_kineto` 等） | 第 12 章 |

## 2.3 宿主 C++ 层：一个扩展模块，五组绑定

`csrc/python_api.cpp` 定义唯一扩展模块：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {   // TORCH_EXTENSION_NAME 默认 _C
    m.doc() = "DeepEP: an efficient expert-parallel communication library";
    // 注册：buffers/{base,ep,engram,pp,bucket}.hpp 的各 register_apis
    //       comm/api.hpp 的 Context 与 barrier 等
    //       runtime/jit.hpp 的 JIT 初始化
    //       utils/event.hpp 的 EventHandle
}
```

五组绑定的分工：

1. **缓冲区运行时**：`_C.EPBuffer`、`_C.PPBuffer`、`_C.EngramBuffer`、`_C.BucketBuffer`。Python 侧每个缓冲区对象持有一个 `self.runtime` 指向对应 C++ 对象（见 `deep_ep/buffers/ep.py` 构造函数中 `self.runtime = _C.EPBuffer(...)`），资源（对称内存、RDMA 窗口、QP、NCCL 通信域）都在 C++ 对象里，Python 只握引用。
2. **通信上下文**：`_C.Context`（`csrc/comm/handle.hpp` 的 `register_apis` 绑定 `get_physical_domain_size`、`get_logical_domain_size` 及只读属性 `num_workspace_bytes`、`num_gpu_buffer_bytes`、`num_rdma_storage_bytes`）。
3. **通信工具**：`_C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)`、事件句柄 `EventHandle`、各种对齐查询（`get_num_allocation_alignment`、RDMA 对齐等，被 `BufferAllocator` 与 `BucketSession` 调用）。
4. **规模计算**：`_C.calculate_ep_buffer_size(nccl_comm, num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch, allow_hybrid_mode, allow_multiple_reduction)`——`EPBuffer.get_buffer_size_hint` 的后端，返回值按 2 MB 对齐。
5. **JIT 初始化**：`init_jit(library_root_path, nccl_root_path)`（`csrc/runtime/jit.hpp`），把 DeepEP 头文件目录与 NCCL 头文件目录交给 DeepJIT。

`csrc/utils/` 下还有四个小而关键的组件：`event.hpp`（C++ `EventHandle`，包装 `torch::Event` 并登记 `tensors_to_record` 做 record_stream 保护）、`lazy_driver.hpp`（`dlopen("libcuda.so.1")` 懒加载 CUDA 驱动符号，如 `lazy_cuDeviceGet`）、`system.hpp`（`get_env<T>` 环境变量读取）、`tensor.hpp`（`get_shape<kNumDims>` 形状解包与 `tensor_list_t`）。

## 2.4 设备端模板层：layout → impls → comm 的依赖方向

`deep_ep/include/deep_ep/` 是 DeepJIT 编译的"源料"，全部是头文件。依赖方向严格单向：

```text
common/（常量、数学、PTX、断言、通用工具）
   ▲
layout/（纯数据布局：token 槽位、信号区、workspace）      ← 不含通信逻辑
   ▲
comm/（NCCLGin 设备端封装、设备端全局屏障）
   ▲
impls/（各通信协议的内核实现：ep/bucket/engram/pp/comm）
   ▲
csrc/kernels/…（宿主侧 launch_* 封装：拼模板、启动 kernel）
```

**`common/`** 提供全库地基：`compiled.cuh`（架构宏修正、`kNumRDMAAlignmentBytes` 等对齐常量、`topk_idx_t` 位宽、`EP_NUM_TOPK_IDX_BITS`）、`math.cuh`（`ceil_div`、`align`、`advance_ptr` 等 host/device 双端函数）、`ptx.cuh`（`deep_ep::ptx` 命名空间：mbarrier、warp 原语 `get_warp_idx`/`get_lane_idx`/`elect_one_sync`、TMA 与 `red_add` 等 PTX 封装）、`utils.cuh`（`iterate_ld_st` 分块拷贝模板）、`exception.cuh`（`EPException` 与 `EP_STATIC_ASSERT`/`EP_UNIFIED_ASSERT` 等宏族）。

**`layout/`** 定义"数据长什么样"。例如 `layout/ep/token.cuh` 的 `TokenLayout` 描述单个 token 槽位的逐字节布局 `[hidden | SF | metadata | mbarrier]`；`layout/ep/workspace.cuh` 的 `EPSignals` 定义信号区（含 `kNumMaxExperts=2048`、`kNumMaxChannels=1280`、`notify_reduction[]`）；`layout/common/barrier.cuh` 的 `BarrierSignals` 是全子系统共用的两相位屏障信号。layout 层不含任何通信语义，因此宿主 C++（计算缓冲区大小、指针偏移）与设备内核（读写数据）可以共享同一份定义——这是消除"两边布局漂移"这种经典 bug 的结构性手段。

**`comm/`** 是设备端通信原语：`handle.cuh` 的 `NCCLGin` 结构封装 NCCL Gin 的设备 API，用一组 `IS_TEAM_WORLD/LSA/RAIL` 宏按团队（world=全组、LSA=NVLink 域、rail=同机）分发调用；`barrier.cuh` 实现设备端全局屏障（`kNumOneSecCycles=2_000_000_000` 以 2 GHz 估算 1 秒周期数，`kDeviceBarrierTag`/`kKernelBarrierTag` 用于超时诊断）。

**`impls/`** 是协议实现，命名即目录：`impls/ep/dispatch.cuh` 与 `impls/ep/combine.cuh`（纯 scale-up/直连路径）、`impls/ep/hybrid_dispatch.cuh` 与 `impls/ep/hybrid_combine.cuh`（混合两级路径）、`impls/ep/dispatch_copy_epilogue.cuh` / `combine_reduce_epilogue.cuh`（收尾内核）、`impls/ep/prefetch_weights.cuh` / `reduce_grads.cuh`（EPLB）、`impls/bucket/{all_gather,all_reduce,reduce_scatter}/{hybrid,nvlink,rdma}.cuh`（Bucket 三算子 × 三路径）、`impls/engram/engram_fetch.cuh` 与 `engram_fetch_wait.cuh`、`impls/pp/pp_send_recv.cuh`。内核全部形如 `__global__ void __launch_bounds__(kNumThreads, 1) …`，行为由大量 `template <bool k…>` 编译期参数决定（如 dispatch 的 `kIsScaleupNVLink`、combine 的 `kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout`）。

## 2.5 四个缓冲区子系统的横向对照

V2.5 把原 ElasticBuffer 拆成四个类。拆分的判据不是数据结构，而是**通信模式**：

| | EPBuffer | EngramBuffer | PPBuffer | BucketBuffer |
| --- | --- | --- | --- | --- |
| 通信模式 | all-to-all（一对多发送 + 接收端归约） | 按索引的远程读（一对多拉取） | 点对点 send/recv | 批量集合通信 |
| 典型场景 | MoE dispatch/combine | 跨节点专家参数/大表/KV 取数 | 流水线相邻 stage 传激活 | CP/DP 的 gather/scatter/reduce |
| 传输路径 | NVLink 直达 或 hybrid（RDMA+NVLink 两级） | RDMA（GIN，需 `ncclGinOptFlagsWarpGet`） | RDMA（仅 RDMA 组） | NVLink（CE/multimem）、RDMA、混合 |
| SM 使用 | 需要 SM（计数/归约/同步） | 需要 SM（发起与等待） | 需要 SM | all-gather 可 0 SM（CE 驱动） |
| 核心方法 | `dispatch`/`combine`/`lb_prefetch_weights`/`lb_reduce_grads` | `set_config`/`write`/`fetch` | `send`/`recv` | `all_gather`/`reduce_scatter`/`all_reduce`/`session` |
| 独立存储区 | 主缓冲区 + 可选 LB 区域 | 独立 RDMA 注册分片（GPU 或 CPU） | 连续对称显存 | 规划的对称张量 + multimem 缓冲 |

四者共享的地基是 `BufferBase` 生命周期、`comm::Context`（NCCL 通信域 + 对称内存窗口 + QP）与 `layout/common/barrier.cuh` 的屏障信号——这正是"拆类不拆地基"的设计：协议各自演进，资源模型一份。

## 2.6 三条编译时间线

DeepEP 的代码按"何时变成机器码"分三类，理解这一点对第 11 章至关重要：

```text
① 安装期（nvcc/g++，CUDAExtension）
   csrc/*.cpp → deep_ep._C 动态库（pybind11 + ATen + NCCL 主机 API）
   csrc/indexing/main.cu 作为"全量包含"冒烟单元：把所有 impls/layout 头
   include 一遍，确保每个模板头至少能编译通过（回归检查，不含功能逻辑）

② 导入期（纯 Python 检查）
   deep_ep/__init__.py：persistent_envs 写入、check_nccl_so 校验 NCCL 一致性

③ 运行期（DeepJIT，nvcc）
   impls/*.cuh + std::format 拼接的特化参数 → PTX/SASS → 缓存 → cuModule 加载
   缓存根目录 EP_JIT_CACHE_DIR（默认 $HOME/.dj），首编译可在安装后无 GPU 的
   机器上完成源码准备，真正编译发生在有目标 GPU 的运行环境
```

安装期不需要可见的 GPU、也不需要 `TORCH_CUDA_ARCH_LIST`——因为设备端代码根本没参与安装期编译。这个"宿主安装期编译 + 设备运行期 JIT"的组合，同时解决了三件事：目标硬件多样性（Hopper 及更新架构都在运行期适配）、模板组合爆炸（配置进模板参数）、以及集群差异化调优（PTXAS 选项经 `EP_JIT_*` 环境变量注入）。

## 2.7 一次调用穿过的层（以 dispatch 为例）

把一次 `buffer.dispatch(x, topk_idx=…, …)` 的调用栈画出来，作为后文的总纲：

```text
EPBuffer.dispatch (deep_ep/buffers/ep.py)
  ├─ 校验与默认值：check_torch_deterministic()、num_topk 解析、handle 复用判定
  ├─ get_theoretical_num_sms / get_theoretical_num_qps（num_sms=0 时）
  ├─ _C.EPBuffer.dispatch → csrc/buffers/ep.hpp 的 EPBuffer::dispatch
  │    ├─ 计算/查表 TokenLayout（layout/ep/token.cuh）
  │    ├─ csrc/kernels/ep/dispatch.hpp: launch_dispatch(…)
  │    │    ├─ 按 allow_hybrid_mode 与拓扑选择 impls/ep/dispatch.cuh 或 hybrid_dispatch.cuh
  │    │    ├─ deep_jit 编译（缓存命中则加载）：模板特化 + nvcc
  │    │    └─ 在通信流（comm::get_comm_stream）上启动内核
  │    ├─ do_cpu_sync=True 时：cudaStreamSynchronize 等接收计数（读 workspace）
  │    └─ 组装 EPHandle（元数据张量族）
  └─ EventOverlap / defer_epilogue 分支：返回事件，epilogue 推迟到 wait
```

可以看到：Python 层只做"决策与语义"，C++ 宿主层做"资源与启动"，设备端模板层做"协议本体"，DeepJIT 做"缝合"。四层各自的复杂度都被限制在一个可独立演进的边界内。

## 2.8 延伸讨论：层间契约的"反证法"

判断一个分层是否划对了位置，有一个实用的反证法：假设把某一层的职责挪到相邻层，看看会发生什么灾难。用这个方法检验 DeepEP 的四层结构，每一层的存在必要性都能得到独立论证。

假设把"布局定义"从 layout 层挪到宿主 C++ 层（第 2 层）：设备端内核将不得不依赖宿主头文件（ATen、pybind11 的传递依赖），JIT 编译的源料会变得笨重且难以隔离；更糟的是，宿主侧的结构体在编译器、打包方式变化时可能发生布局漂移，而设备内核按自己的假设解释字节——两种"真理"并存的局面正是 layout 作为纯头文件、纯数据定义层要结构性消灭的。假设把"协议逻辑"从 impls 层下沉到宿主 C++ 层：每个行为开关（`kIsScaleupNVLink`、`kDoExpandedSend` 等）都退化为运行时 if 分支，内核失去编译期特化带来的寄存器预算、循环展开与死代码消除，第 10 章的质量门也没有了意义。假设把"资源管理"从 C++ 缓冲区层上移到 Python 层：NCCL 通信域、VMM 页、窗口注册这些需要精确析构顺序的资源将由 Python 的垃圾回收时机决定，`destroy_on_destruction` 所规避的析构顺序陷阱会以更隐蔽的形式回归。假设把"生命周期与校验"从 Python 层下推到 C++ 层：meta 张量规划、参数校验、事件语义这些与 PyTorch 生态深度耦合的东西就要在 C++ 里重复造轮子，且失去 Python 生态的动态性。四次反证全部失败，说明分层线的画法是稳健的。

这个方法也可以用来评估未来的演进：若某一层需要吸收新的职责（例如第 12 章展望的低比特格式），先问"它属于哪一层的契约"——数值格式属于 layout 层（数据长什么样）、解码逻辑属于 impls 层（怎么搬运）、开关暴露属于 Python 层（怎么配置）。职责先归层、再动手，就不会破坏既有结构。

## 2.11 延伸讨论：源码阅读路线图

对准备深入源码的贡献者，本章给出一条由浅入深的阅读路线，每一站都注明"该看什么、为什么"。第一站，入口三件：`deep_ep/__init__.py`（看导出面与环境检查，理解"用户可见什么"）、`csrc/python_api.cpp`（看绑定装配，理解"C++ 暴露什么"）、`deep_ep/buffers/base.py` 与 `csrc/buffers/base.hpp`（看生命周期基类，理解"资源如何进出"）——三站合计不过数百行，却勾勒出整个接口面。第二站，对称内存一条线：`csrc/kernels/comm/symmetric.hpp`（VMM 分配与 fd 共享）到 `csrc/kernels/comm/context.cpp`（Context 创建、Gin 上下文与 QP）再到 `deep_ep/comm/handle.py`（Python 侧通信域管理）——这条线回答"一块可直达的显存从哪来"，是理解所有缓冲区构造的前置。第三站，EP 主干：`deep_ep/buffers/ep.py` 的构造与 `dispatch`（Python 语义决策）、`csrc/buffers/ep.hpp` 与 `csrc/kernels/ep/dispatch.hpp`（宿主启动封装，重点看 `std::format` 拼接与模板名生成）、`deep_ep/include/deep_ep/layout/ep/token.cuh` 与 `workspace.cuh`（布局与信号区，常量集中地）、最后是 `impls/ep/dispatch.cuh` 与 `hybrid_dispatch.cuh`（协议本体，先看骨架再看 warp 分工）。第四站，镜像与周边：`impls/ep/combine.cuh` 与 `hybrid_combine.cuh`（对照 dispatch 看对称性）、`impls/ep/` 的两个 epilogue（收尾逻辑）、EPLB 的两个内核（第 7 章）。第五站，家族扩展：bucket、engram、pp 的同构线（每个都是"Python 缓冲区、宿主封装、layout、impls"四件套）。第六站，基础设施：`common/ptx.cuh`（PTX 封装）、`common/math.cuh` 与 `utils.cuh`（工具）、`runtime/jit.hpp` 与 `third-party/deep_jit`（编译设施）。路线的排序原则与第 18.12 节联调顺序一致：沿依赖方向、每个站都有可独立验证的产出（能回答"这个文件为什么存在"再进下一站）。配合第 2.9 节的命名空间依赖图使用，两周内可以建立对代码库的完整心智地图。

## 2.12 延伸讨论：从分层看代码评审要点

分层结构不仅指导阅读，也指导评审——每一层有自己的"评审问题清单"。Python 层的问题：生命周期纪律是否被遵守（`explicitly_destroy` 的使用、在途操作检查，第 3 章）、参数校验是否前移（断言失败的错误信息是否可行动，第 14 章）、meta 计划的全组一致性是否有校验（第 3.6 节）。宿主 C++ 层的问题：资源获取与释放是否配对（Context、窗口、QP 的所有权，第 3 章）、`launch_*` 封装是否把所有行为开关都表达为模板实参而非运行时分支（第 11.11 节因果链）、启动校验是否覆盖规模与对齐断言（第 6.11 节）。layout 层的问题：新结构是否 `alignas` 正确（RDMA 对齐，第 4 章）、字段注释是否描述了协议契约（第 10.12 节"字段语义即契约"）、宿主与设备是否只有这一份共享定义。impls 层的问题：warp 分工是否用了标准原语（`elect_one_sync`、warp 索引，第 10.1 节）、同步是否走统一入口（`timeout_while` 与屏障标签，第 10.3 节）、寄存器预算是否有声明依据（第 10.8 节三重保险）、共享内存是否静态分配（第 11.11 节第三环）。四份清单合起来的评审哲学只有一条：**在正确的层问正确的问题**——在 impls 里问 Python 语义、在 Python 里问寄存器预算，都是分层被破坏的信号。对新贡献者，这四份清单比编码规范更本质：它们定义了"这个代码库认为什么是好的变更"。

## 2.13 延伸讨论：为什么设备层选择纯头文件

第 2.4 节描述了设备端模板层"全部是头文件"的形态，本节论证这个选择的三个理由——它们共同支撑了 JIT 架构（第 11 章）的可行性。理由一，**源料即接口**：DeepJIT 编译的输入是源码文本（`std::format` 拼接，第 11.3 节），头文件天然是"可拼接的最小单元"——每个 impls 头自带依赖闭包（include 列表），拼接层只需写 include 与实例化声明即可生成完整编译单元；若设备代码以 .cu/.cpp 源文件形式存在，拼接层就要处理多文件工程、链接符号等额外复杂度。理由二，**共享无 ABI 负担**：layout 层的结构（`TokenLayout`、`EPSignals` 等）被宿主 C++ 与设备内核同时包含（第 2.4 节"单一事实来源"），头文件包含是文本级共享——没有动态库的 ABI 兼容问题、没有符号导出清单，两侧"看到"的结构永远是同一份文本。理由三，**编译期校验的统一入口**：`csrc/indexing/main.cu` 的全量包含冒烟（第 11.3 节）之所以可行，正因为所有设备代码都是头文件——一个编译单元 include 全部即可完成"可编译性回归"；如果设备代码分散在多个编译单元，这条冒烟线就不存在，某个头文件的损坏要到运行期 JIT 才暴露。三个理由的合力是：设备层的"纯头文件"不是风格偏好，而是"运行时编译 + 宿主设备共享布局 + 构建期校验"三个架构决策的最小实现形态——第 2.8 节反证法可以再次套用：把任何一部分改成非头文件形态，三个理由中至少一个受损。这也解释了为什么新增子系统（bucket、engram、pp）都自然沿用了同一形态——好的架构约束会自我复制。

## 2.9 延伸讨论：命名空间即依赖图

C++ 侧的命名空间划分（`deep_ep::{ep,pp,bucket,engram,comm,symmetric,ptx,math,layout,utils}`）不只是组织习惯，它直接编码了依赖方向，可以作为阅读源码时的"依赖导航器"。`deep_ep::ptx` 与 `deep_ep::math` 位于最底层：它们只依赖 CUDA 头文件，被所有其他层使用而不使用任何其他层——这也是它们可以放在 `deep_ep/include/deep_ep/common/` 下随 JIT 编译、而不必进宿主扩展的原因。`deep_ep::layout` 依赖 ptx 与 math（TokenLayout 里内嵌 mbarrier 类型），同样位于设备头文件域。`deep_ep::comm`（设备端）依赖 layout 与 ptx，封装 NCCLGin 与屏障。`deep_ep::{ep,pp,bucket,engram}` 的 impls 依赖以上全部，加上 NCCL 设备头。回到宿主侧，`deep_ep::symmetric` 依赖 CUDA 驱动（经 lazy_driver 懒加载）而不依赖任何上层；`deep_ep::comm`（宿主）依赖 symmetric 与 kernels/comm；各缓冲区命名空间依赖 comm、对应 kernels、layout 与 runtime/jit。Python 层则只依赖 `deep_ep._C` 的绑定面。把这张依赖图记在脑子里，读任何源文件的第一步就是看它的 include 列表确认它处在图的哪一层——如果出现"高层 include 低层头文件"之外的边（例如 impls 反过来 include 宿主头文件），那多半是历史遗留或待改进点。第 2.4 节画出的"layout → impls → kernels → buffers"单向链，在这张命名空间图上是可以逐边核验的。



## 2.10 本章小结

- DeepEP 四层结构：Python 缓冲区层（生命周期/语义/规划）、宿主 C++ 层（运行时/启动）、设备端模板层（layout/comm/impls）、DeepJIT（运行时编译）；层间以 pybind11 与 JIT 编译两个边界衔接。
- `layout/` 层是宿主与设备共享的"单一事实来源"，从结构上消灭布局漂移；`impls/` 层以模板参数表达全部行为开关。
- 四个缓冲区类按通信模式拆分（all-to-all / 远程读 / 点对点 / 批量集合通信），共享 BufferBase、Context 与屏障地基。
- 代码按三条编译时间线组织：安装期宿主编译（含 `csrc/indexing/main.cu` 全量包含冒烟）、导入期一致性检查、运行期设备 JIT。
- 一次 dispatch 从 Python 语义决策出发，经 C++ 启动封装，落到 JIT 编译的设备内核，最终在专用通信流上执行——这条主线将贯穿第 4、10、11 章。

下一章先驻留在这条主线的"资源"一端：缓冲区对象如何构造、如何统一销毁，以及 `BufferAllocator` 如何用"先规划后落实"的方式安排对称张量布局。
