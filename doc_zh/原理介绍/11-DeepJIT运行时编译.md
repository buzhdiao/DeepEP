# 第 11 章 · DeepJIT 运行时编译

## 11.1 为什么通信内核必须 JIT

一个直觉的疑问：CUDA 的 C++ 扩展（如 PyTorch 的 CUDAExtension）不也能编译内核吗？DeepEP 为什么选择运行时编译？答案藏在三个组合爆炸里。

**（1）配置空间爆炸。** 回看前面章节出现的编译期参数：dispatch 的 `kIsScaleupNVLink`、`kDoCPUSync`、`kReuseSlotIndices`、`kNumSFPacks`；combine 的 `kAllowMultipleReduction`、`kUseExpandedLayout`、`kDoExpandedSend`、`use_rank_layout()` 谓词；再加上运行参数——`hidden`、`num_topk`、`num_ranks`、`num_experts`、通道数、各 rank/专家规模、FP8 位宽。这些量在 DeepEP 的实现里大多是**模板实参或 `constexpr` 常量**（例如 `constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;`、`constexpr int kNumMaxTokensPerChannel = ceil(kNumMaxTokensPerRank / kNumChannels);`），因为它们参与数组边界、循环展开、共享内存大小与寄存器预算的计算——运行时参数做不到这些。模板实参的笛卡尔积是天文数字，预编译任何"常用子集"都会在某个用户的配置上失效。

**（2）目标硬件多样。** README 环境要求"NVIDIA Hopper 或更新的 GPU"——一个跨越多代架构的区间。PTX/SASS 的最优形态与 SM 数、共享内存容量、TMA 能力相关；安装机器上往往**根本没有 GPU**（README："安装时不需要可见的 GPU，也不需要 `TORCH_CUDA_ARCH_LIST`"），预编译无从谈起。JIT 把"针对当前设备编译"变成运行时自然发生的事。

**（3）集群差异化调优。** 第 10 章的质量门（`EP_JIT_CHECK_NO_SPILLS`）、第 5 章的寄存器预算、PTXAS 的详细输出（`EP_JIT_PTXAS_VERBOSE`）——这些编译器级开关需要按集群环境注入，JIT 的编译命令是运行时拼出来的，天然可配置。

DeepJIT（`third-party/deep_jit` 子模块）正是承载这三点的运行时编译框架：源码模板 + 参数 → nvcc → 缓存 → 设备模块加载。

## 11.2 运行时的初始化：csrc/runtime/jit.hpp

宿主侧入口非常薄：

```cpp
// 全局 JIT 编译运行时：懒初始化（首次使用时才真正创建 CUDA 编译运行时）
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path, const std::string& nccl_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    // 计算 DeepEP 头文件所在目录（根目录下的 include）…
}
```

三个要点：

1. **`LazyInit`**：全局单例直到第一次内核编译才创建——导入 `deep_ep` 而不通信的进程零 JIT 开销；
2. **`Runtime<deep_jit::CUDA>`**：DeepJIT 的 CUDA 后端（nvcc 编译 + 模块加载）；
3. **两个根路径**：DeepEP 的头文件根（`deep_ep/include`，impls/layout/comm 模板所在）与 NCCL 安装根（`EP_NCCL_ROOT_DIR` 或自动发现，设备端要 include `nccl_device.h`）。这两个路径决定了 JIT 能看到什么源料——这也是它们被 `setup.py` 烘焙、被 `init_jit` 接收的原因。

`csrc/utils/lazy_driver.hpp` 的懒加载与之呼应：`dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL)` 延迟加载驱动符号（`lazy_cuDeviceGet` 等）——宿主扩展启动时不强求驱动就绪，首次真正用到时才解析，与 LazyInit 构成同一"按需初始化"哲学。

## 11.3 模板拼接：std::format 生成的内核源码

JIT 的"源码"不是固定文件，而是**运行时拼接的字符串**。以 `csrc/kernels/ep/dispatch.hpp` 的头部为例：

```cpp
#include <format>      // std::format：拼接 JIT 编译模板
…
static void launch_dispatch(void* x, void* sf, /* …30+ 参数… */) { … }
```

`launch_*` 函数的工作流是：

1. **计算编译期常量**：按运行参数推导模板实参（通道数、notify warp 共享内存大小 `get_num_notify_smem_bytes`、TokenLayout 等）；
2. **拼接模板名与源码**：用 `std::format` 把特化参数（`kNumRanks`、`kNumExperts`、`kNumTopk`、`kHidden`、布尔开关……）写进内核模板的实例化声明与 `include` 列表，生成一份完整的 `.cu` 文本；
3. **交给 DeepJIT**：按（源码 + 编译选项 + 目标架构）的键查缓存，未命中则调 nvcc 编译、命中则直接加载；
4. **启动内核**：拿到设备函数句柄后在通信流上 `cuLaunchKernel`（经 DeepJIT 的 CUDA 后端）。

这种"宿主拼字符串、设备端纯头文件"的分工意味着：`deep_ep/include/deep_ep/impls/` 下的每个 `.cuh` 都是**可独立编译的模板**（它们被 `csrc/indexing/main.cu` 这个"全量包含"冒烟单元逐一 include，作为构建期回归检查——该文件注释自称 *"把所有内核实现头文件逐一 include 进来，编译该 .cu 即可对全部模板头做实例化/类型检查"*），而运行时的"语义差异"全部由拼接的实例化代码表达。

## 11.4 编译管线与工具链发现

JIT 编译一个内核的完整管线：

```text
源码文本 + 编译选项
    │  nvcc 发现：CUDA_HOME → CUDA_PATH → PATH 中的 nvcc → /usr/local/cuda
    │  标准版：EP_JIT_CPP_STANDARD（默认 20；DeepEP 要求 C++20 + std::format）
    ▼
nvcc 编译（可注入：EP_JIT_PTXAS_VERBOSE、EP_JIT_WITH_LINEINFO、架构信息）
    ▼
产物：cubin/PTX（可选 EP_JIT_DUMP_PTX / EP_JIT_DUMP_SASS，SASS 需 toolkit 的 cuobjdump）
    ▼
加载：cuModuleLoad（经 DeepJIT 后端；EP_JIT_PRINT_LOAD_TIME 可打印加载耗时）
```

工具链发现顺序（README："DeepJIT 依次通过 `CUDA_HOME`、`CUDA_PATH`、`PATH` 中的 `nvcc`，最后是 `/usr/local/cuda` 来发现 CUDA toolkit"）覆盖了容器、conda 环境与系统安装三种典型布局；`EP_JIT_NVCC_COMPILER` 可强制指定 nvcc 可执行文件（但"仍必须能找到有效的 CUDA toolkit 根目录"）。C++20 是硬性要求——`std::format` 是拼接层的基础设施，`csrc/kernels/` 各头文件对 `<format>` 的 include 是其直接证据。

## 11.5 缓存：EP_JIT_CACHE_DIR 与多根搜索

编译是分钟级操作，缓存是 JIT 可用性的生命线：

- **位置**：`EP_JIT_CACHE_DIR`，默认 `$HOME/.dj`；
- **多根**："或以冒号分隔的多个根目录。按顺序搜索所有根目录，并将新编译的产物写入第一个"（README）——典型用法是把**只读的共享缓存**（NFS 挂载的集群级预热产物）放在前面，本地可写缓存在后：命中即秒加载，未命中才本地编译并写入；
- **键**：源码文本 + 特化参数 + 编译选项 + 目标架构共同决定（同一内核在不同 MoE 配置/不同 SL 环境下是不同条目）；
- **共享维度**：同一集群同型号 GPU 的产物完全可复用——这就是"预热"策略的基础：发布前在代表性节点上跑一遍全部配置，把 `$HOME/.dj` 推到共享存储。

首次编译成本的管理是集成方的必修课：DeepEP 内核数量多（EP 两路径 × dispatch/combine/epilogue、bucket 三算子 × 三路径、engram、pp、barrier），一次冷启动的编译时间不可忽视。实践上配合 `EP_JIT_DEBUG=1`（"启用编译命令与内核加载诊断、PTXAS 输出、源码行信息以及 PTX/SASS 转储"）观察命中情况。

## 11.6 质量门：把坏内核挡在加载之前

两个默认关闭的严格开关（README 环境变量表）：

| 开关 | 拒绝条件 | 动机 |
| --- | --- | --- |
| `EP_JIT_CHECK_NO_SPILLS` | 已编译内核存在寄存器溢出（register spill） | 溢出 = 隐性 HBM 往返，通信内核延迟敏感，第 10 章 |
| `EP_JIT_CHECK_NO_LOCAL_MEMORY` | 内核使用了 local memory | 同上，local memory 访问延迟数倍于寄存器 |

实现上由 PTXAS 的诊断输出判定（配合 `EP_JIT_PTXAS_VERBOSE` 与 `EP_JIT_WITH_LINEINFO` 定位到源码行）。这两个开关把第 10 章的寄存器预算设计（hybrid combine 的 40/216、`__launch_bounds__(…, 1)`）变成**可执行的契约**：任何新内核或新配置若导致溢出，会在部署前被拒绝而不是在运行时以性能退化的形式悄悄发生。

## 11.7 环境变量全景

JIT 相关环境变量分两组：DeepEP 前缀 `EP_JIT_*`（优先）与 DeepJIT 全局回退 `DJ_JIT_*`（README："DeepJIT 会优先读取 `EP_JIT_*`，然后才读取对应的 `DJ_JIT_*` 变量作为全局回退。请在第一次内核编译之前完成这些配置。显式的 `EP_JIT_*` 值（包括打包内置的默认值）优先于 `DJ_JIT_*`"）。全景表：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `EP_JIT_DEBUG` | 0 | 编译命令/加载诊断/PTXAS 输出/源码行/PTX 与 SASS 转储 |
| `EP_JIT_CACHE_DIR` | `$HOME/.dj` | 缓存根（冒号分隔多根，按序搜索、写入第一个） |
| `EP_JIT_NVCC_COMPILER` | 自动检测 | 覆盖 nvcc 可执行文件 |
| `EP_JIT_CPP_STANDARD` | 20 | 传给 nvcc 的 C++ 标准 |
| `EP_JIT_PRINT_COMPILER_COMMAND` | 0 | 打印编译器与反汇编器命令 |
| `EP_JIT_PRINT_LOAD_TIME` | 0 | 打印内核二进制加载时间 |
| `EP_JIT_PTXAS_VERBOSE` | 0 | 启用并打印详细 PTXAS 输出 |
| `EP_JIT_CHECK_NO_SPILLS` | 0 | 拒绝寄存器溢出的内核 |
| `EP_JIT_CHECK_NO_LOCAL_MEMORY` | 0 | 拒绝使用 local memory 的内核 |
| `EP_JIT_WITH_LINEINFO` | 0 | 嵌入剖析用源码行信息 |
| `EP_JIT_DUMP_ASM` / `EP_JIT_DUMP_PTX` / `EP_JIT_DUMP_SASS` | 0 | 缓存未命中时产出汇编产物（SASS 需 cuobjdump） |
| `EP_GIN_GDAKI_DEBUG` | 0 | 编译 JIT 内核时启用 NCCL Gin GDAKI 设备调试 |

其中一部分（`EP_JIT_CACHE_DIR`、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_JIT_CPP_STANDARD`）在包构建期被 `setup.py` 记录为 `persistent_envs` 烘焙进安装产物（第 12 章）——"集群共性"在打包时固化，"节点个性"在运行时覆盖。

## 11.8 JIT 与宿主扩展的边界总结

```text
安装期（setup.py → CUDAExtension）        运行期（DeepJIT）
────────────────────────────             ────────────────────────
编译 csrc/*.cpp → deep_ep._C              拼接 impls/*.cuh 特化源码
（pybind11、ATen、NCCL 主机 API、          nvcc 编译（缓存于 $HOME/.dj）
  launch_* 封装、对称内存、JIT 入口）       cuModuleLoad → 启动内核
csrc/indexing/main.cu 冒烟：              首次成本分钟级，之后零成本
  全部模板头实例化检查
```

边界清晰到可以用一句话概括：**宿主扩展知道"如何启动内核"，设备端模板知道"内核是什么样"，DeepJIT 在两者之间按需生成"针对这台机器的这个配置"的内核二进制**。安装期对 GPU 的零依赖（不需要可见 GPU、不需要 `TORCH_CUDA_ARCH_LIST`）正是这条边界划对了的证明。

## 11.9 本章小结

- JIT 的必要性来自三点：模板参数的配置空间爆炸（行为必须编译进内核）、目标硬件多样且安装机常无 GPU、集群级编译调优需求。
- `csrc/runtime/jit.hpp` 以 `LazyInit<Runtime<deep_jit::CUDA>> jit` 提供懒初始化全局运行时，`init_jit(library_root, nccl_root)` 交付源料路径；`lazy_driver.hpp` 的 dlopen 懒加载与之呼应。
- `launch_*` 封装用 `std::format` 拼接特化源码，impls 头文件全部可独立编译（`csrc/indexing/main.cu` 全量包含冒烟保证）；nvcc 发现链 `CUDA_HOME → CUDA_PATH → PATH → /usr/local/cuda`，C++20 硬性要求。
- 缓存 `EP_JIT_CACHE_DIR`（默认 `$HOME/.dj`）支持冒号分隔多根按序搜索、写入第一个根——共享只读预热 + 本地写回的双层布局；质量门 `EP_JIT_CHECK_NO_SPILLS`/`EP_JIT_CHECK_NO_LOCAL_MEMORY` 把坏内核挡在加载前。
- `EP_JIT_*` 优先于 `DJ_JIT_*` 回退；构建期可烘焙（persistent_envs），运行时用户环境优先。

## 11.10 延伸讨论：JIT 缓存的运维实践与编译失败的排查

**缓存键的稳定性意识**。DeepJIT 的缓存条目由源码文本与特化参数共同决定，这意味着三类日常事件都会导致"新条目"：MoE 配置变化（`hidden`、`num_topk`、rank 数、FP8 开关）、代码版本更新（impls 头文件的任何改动）、编译选项变化（`EP_JIT_CPP_STANDARD`、质量门开关）。运维上应当把"哪些配置在用"当作集群资产来管理：若某个推理服务在滚动更新时改了 topk 或 hidden，它会触发一轮全新的首次编译——正确的做法是滚动前在预热节点上以新配置编译一遍，再把 `EP_JIT_CACHE_DIR` 的共享根同步出去（多根按序搜索、写入第一个根的语义天然支持"共享只读层 + 本地写回层"的缓存布局）。

**首次编译的时序风险**。首次编译发生在第一次内核启动路径上，也就是说它发生在**通信即将开始之前**——若多个 rank 同时触发同一内核的编译，每个进程各自编译一遍再各自写缓存（缓存写入有目录原子性保证，重复产物无害，但时间成本翻倍）。因此预热应当在所有 rank 就绪后、通信开始前串行或并行触发一次（例如调用缓冲区的构造与一次小规模空跑）。`EP_JIT_PRINT_LOAD_TIME=1` 与 `EP_JIT_PRINT_COMPILER_COMMAND=1` 是确认预热覆盖率的两个开关：前者显示每个内核二进制的加载时间（命中缓存时接近零），后者把编译命令打印出来便于比对。

**编译失败的排查顺序**。JIT 编译失败的信息通常表现为内核启动路径上的异常，排查按"三层源料"顺序：先确认 DeepEP 头文件根可见（`init_jit` 的 `library_root_path` 是否指向包含 `deep_ep/include` 的目录，wheel 安装时由 setup.py 把 csrc 源码拷入包内保证）；再确认 NCCL 设备头可见（`EP_NCCL_ROOT_DIR` 指向含 `include/nccl_device.h` 的目录，且版本 ≥ 2.32.3）；最后才是 nvcc 工具链（发现链 `CUDA_HOME → CUDA_PATH → PATH → /usr/local/cuda`，`EP_JIT_NVCC_COMPILER` 强制指定）。`EP_JIT_DEBUG=1` 会打开编译命令、PTXAS 输出与源码行信息——绝大多数编译失败都能在这一层看到直接原因（找不到头文件、架构不支持、C++ 标准不足）。`EP_GIN_GDAKI_DEBUG` 则用于 NCCL Gin 设备侧（GDAKI）的更深一层调试，属于网络后端问题而非编译问题的分界开关。

**JIT 与宿主扩展版本的一致性**。宿主扩展（安装期编译的 `deep_ep._C`）与设备模板（运行期 JIT 编译的 impls 头文件）必须来自同一份源码树：`launch_*` 封装拼接的模板参数、常量与 impls 头文件的结构定义强耦合。setup.py 把 csrc 源码拷入 wheel 正是为了保证这一一致性（安装包内自带模板源码）；若运维手动替换过缓存里的模板或混用了不同版本的扩展与源码树，症状往往是结构体布局断言失败或编译期常量不匹配——`check_nccl_so` 式的"版本一致性防御"思路在这里同样适用：宁可启动时报错，也不要带病运行。

## 11.11 延伸讨论：编译期常量与内核质量的因果链

第 11 章说明了"配置进模板"的现象，本节把"配置进模板为什么能换质量"的因果链论证完整——它解释了第 10 章寄存器预算、第 4 章通道划分与第 11 章 JIT 三者如何互相成就。因果链的第一环是**常量传播与死代码消除**：当 `kNumRanks`、`kNumTopk`、`kIsScaleupNVLink` 等是编译期常量时，编译器可以把所有依赖它们的计算折叠成常量，把 `if constexpr` 的假分支整体删除——内核的指令序列只包含本配置需要的路径，没有任何运行时分支预测的损耗。第二环是**循环展开与向量化**：循环次数依赖常量（如 `kHiddenVec`、`kUnrollFactor`，第 5.4 节），编译器才能做完全展开与寄存器级向量化，这直接决定每次 load/store 的有效带宽（第 10.4 节的访存分析）。第三环是**数组边界与共享内存的静态分配**：`get_num_notify_smem_bytes` 这类函数在宿主侧算出共享内存大小（第 4 章），内核内的数组按编译期大小静态分配——动态尺寸的共享内存会破坏占用声明与寄存器预算（第 10.8 节）。三环相扣的结论是：**配置的每一个编译期化，都同时解锁一层编译器优化**；反之，任何一个"运行时再决定"的量都会沿因果链向上侵蚀内核质量。这也解释了质量门（`EP_JIT_CHECK_NO_SPILLS`）为什么有效：溢出往往是常量化的某个环节被破坏（配置组合超出寄存器预算）的信号，拒绝编译把问题暴露在部署前。

## 11.12 延伸讨论：与常规 CUDA 扩展编译的对照

把 DeepJIT 与 PyTorch 生态最常见的 CUDAExtension 预编译模式逐项对照，可以更准确地判断"什么时候该用哪一种"。预编译模式的流程：安装期以固定架构列表（`TORCH_CUDA_ARCH_LIST`）编译所有内核为二进制，运行时直接加载；它的优点是运行时零编译成本、持续集成可验证；缺点是组合空间必须预先枚举（第 11.1 节的三重爆炸使 EP 内核不可枚举）、且要求安装机可见目标架构或凭空指定。DeepJIT 模式：安装期只编宿主扩展，内核在运行时按"当前配置乘当前设备"生成；优点前面已论证，缺点是首编译延迟（缓存与预热解决，第 11.10 节）与运行环境的工具链依赖。两者的分界判据可以概括为一句话：**内核组合空间小且目标硬件已知时用预编译，组合空间开放或硬件多样时用 JIT**。DeepEP 的另一层巧思在于两种模式并存——宿主扩展预编译（安装期），设备模板 JIT（运行期）：`csrc/indexing/main.cu` 的全量包含冒烟（第 11.3 节）让设备模板在安装期就完成一次"可编译性验证"，等于把预编译模式的持续集成优点借了过来，而把二进制生成的灵活性留给运行时。这个"验证与生成分离"的结构，值得任何带设备代码的项目借鉴。

## 11.13 延伸讨论：模板名、缓存指纹与产物的可读性

JIT 体系有一个容易被忽视的工程细节：内核的"名字"。`csrc/kernels/` 各头文件 include `<string>` 与 `<format>`（第 11.3 节），不仅拼接源码，也拼接**模板实例名**——实例名把配置编码进标识（dispatch 与 combine 的启动封装注释都提到"拼接 JIT 编译模板"与"模板名"）。名字即指纹的价值有三。其一，诊断可读性：编译失败与剖析输出中出现的是带配置的实例名（哪些模板参数、哪些开关），第 14 章排查时能直接定位到配置组合，而不是面对匿名的 `kernel_0x3f2a`。其二，缓存对账：`EP_JIT_CACHE_DIR` 下的产物按内容寻址（第 11.5 节键的定义），配合实例名可以把"哪些配置占用了缓存"列出来——预热覆盖率的核对（第 11.10 节）从猜测变成清点。其三，回归对照：内核版本升级后，新旧实例名的集合差异就是"受影响配置"的清单（第 15.15 节回归触发条件的落地辅助）。与实例名配套的是第 12.1 节的版本溯源（git 哈希进版本元数据）：产物、实例名、源码版本三者可关联，JIT 体系的可追溯性达到与预编译体系相当的水平。这个细节给出的方法论是：**运行时生成的系统尤其需要"生成的对象有可读的名字"**——匿名生成物在排障时的成本会成倍放大，无论生成的是内核、查询计划还是配置快照。



至此，DeepEP 的架构、协议与编译设施都已讲透。最后一章收拢工程面：构建系统、测试体系、昇腾适配与生态，并给出全文总结与展望。
