> 中文版说明文档，对应原文：[README.md](README.md)（结构与章节保持一致，代码、命令、路径、链接不做翻译）

# DeepEP

DeepEP（DeepEveryParallel）是一个面向机器学习训练与推理的高性能通信库。它为 MoE 的 dispatch/combine（分发/聚合）提供高吞吐、低延迟的专家并行（expert parallelism，EP）all-to-all GPU 内核，包括 FP8 dispatch，以及面向冗余专家的 NVLink 权重与梯度交换。它还提供面向流水线并行（pipeline parallelism，PP）、上下文并行（context parallelism，CP）、数据并行（data parallelism，DP）和远程内存访问（Engram）的实验性原语。通信内核在运行时通过 [DeepJIT](https://github.com/deepseek-ai/DeepJIT) 编译，配套的扩展则在安装期间构建。

## 新闻

- **昇腾（Ascend）版本发布**
  - 在华为昇腾 950 NPU 上提供相同的 API 与完整性能
  - 更多细节请查看 [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)

- **V2.5 发布**：
  - 将 `ElasticBuffer` 拆分为 `EPBuffer`、`EngramBuffer`、`PPBuffer` 和 `BucketBuffer`，共享 `BufferBase` 生命周期
  - 新增 `BufferAllocator`，可在构建缓冲区之前规划对称张量（symmetric tensor）的分配
  - 通过 `BucketBuffer` 新增批量的 all-gather、reduce-scatter 与 all-reduce，并为普通 PyTorch 张量提供会话（session）机制
  - 新增 `EPBuffer.lb_prefetch_weights` 和 `EPBuffer.lb_reduce_grads` 以支持动态冗余专家：在专家计算之前通过 NVLink 预取专家权重与量化 scale，然后在反向传播时将冗余专家的 FP32 梯度累加回原始专家。这些原语支持 [MoonEP](https://github.com/MoonshotAI/MoonEP) 和 [UltraEP](https://github.com/Dots-Infra/UltraEP) 所探索的专家复制（expert-replication）方案；API 与集成要求参见[专家负载均衡](#专家负载均衡)
  - 支持延迟执行的 EP epilogue（收尾操作）、缓存的展开布局（expanded layout），以及专家之间的零填充
  - 支持 GPU 或 CPU 上的多层 Engram 存储，每层各有一个等待钩子（wait hook）
  - 完全移除 V1，包括其 API、NVSHMEM 后端及旧版文档。NVSHMEM 不再是依赖项

- **V2 发布**：对专家并行进行了彻底重构，支持更大的 scale-up 与 scale-out 域，并引入轻量级的 **NCCL Gin 后端**。

### 新特性

- 基于 DeepJIT 的 **JIT 编译通信内核**
- **NCCL Gin 后端**
  - 轻量级的设备端通信 API
  - 能够复用已有的 NCCL 通信域（communicator）
- **EPv2**
  - 高吞吐与低延迟 API 统一为单一的 `EPBuffer` 接口，并为分组专家 GEMM 提供展开布局
  - 支持更大的 scale-up 与 scale-out 域
  - 解析式（analytical）SM 与 QP 数量计算——不再需要自动调优
  - 混合（hybrid）模式与直连（direct）模式均继续支持
- **Engram**（实验性，基于 RDMA）
- **PP**（实验性，基于 RDMA）
- 面向 CP 和 DP 的 **Bucket 集合通信**（实验性）（all-gather、reduce-scatter 与 all-reduce）

### 注意事项

- EP 的 dispatch 与 combine 需要 GPU SM；不支持零 SM 的 RDMA EP
- Bucket、Engram 和 PP 为实验性特性
- Engram 需要提供 `ncclGinOptFlagsWarpGet` 的 NCCL 构建

## 快速开始

### 环境要求

- Linux
- Python 3.10 及以上
- NVIDIA Hopper 或更新的 GPU
- 用于 DeepJIT 编译的 CUDA Toolkit 13.1 及以上，且需支持目标 GPU
- 支持 `std::format` 的 C++20 编译器与标准库
- PyTorch 2.10 及以上，需支持 CUDA
- NCCL 2.32.3 及以上
- 节点内（intranode）通信需要 NVLink
- 跨节点（internode）通信需要 RDMA 网络

安装过程会针对 CUDA 与 NCCL 库构建宿主端 C++ 扩展。GPU 内核由 DeepJIT 在运行时针对当前设备编译，因此安装时不需要可见的 GPU，也不需要 `TORCH_CUDA_ARCH_LIST`。运行时请保持 CUDA toolkit 与宿主编译器可用。自动带宽检测使用 `nvidia-smi` 和 `ibstat`；`BucketBuffer` 目前要求 NVLink 与 RDMA 带宽均可被检测到，即使某个通信组只使用其中一种传输方式。

### 安装 NCCL 依赖

安装与你的 CUDA 环境匹配的 NCCL 包，以便 DeepEP 能够找到其头文件和库：

```bash
# CUDA 13.x
python -m pip install "nvidia-nccl-cu13>=2.32.3" --no-deps
# For CUDA 12.x, use nvidia-nccl-cu12 instead
```

对于自定义的 NCCL 安装，请将 `EP_NCCL_ROOT_DIR` 设置为包含 `include/` 和 `lib/` 的目录。PyTorch 与 DeepEP 必须加载同一个 NCCL 共享库。使用低于 2.31 的 NCCL 头文件进行构建时，还额外要求 NCCL 的编译时/运行时版本完全一致。

### 安装

```bash
# Initialize the DeepJIT submodule
git submodule update --init --recursive

# Build a wheel and install it into the current Python environment
bash install.sh
```

然后在你的 Python 项目中导入 `deep_ep`。

### 开发与测试

```bash
# Build and link the extension into the source tree
bash develop.sh

# Run test cases
python tests/ep/test_ep.py
python tests/bucket/test_all_gather.py
python tests/bucket/test_reduce_scatter.py
python tests/bucket/test_all_reduce.py
python tests/buffer/test_allocation.py
python tests/ep/test_prefetch_weights.py
python tests/ep/test_reduce_grads.py
python tests/engram/test_engram.py
python tests/pp/test_pp.py
```

测试脚本依赖 NumPy，并会在本地拉起 GPU worker。对于多节点测试，请在每个节点上以共享的 `MASTER_ADDR` 和 `MASTER_PORT` 启动同一脚本，将 `WORLD_SIZE` 设置为节点数、`RANK` 设置为节点索引。这些是 [deep_ep/utils/envs.py](deep_ep/utils/envs.py) 中 `init_dist` 的约定；如有需要可根据你的集群进行调整。除非已被设置，测试默认使用 `NCCL_IB_SL=1` 和 `EP_OVERRIDE_RDMA_SL=1`。权重预取与梯度归约测试使用单个 NVLink 域；PP 测试需要仅含 RDMA 的通信组。

## 接口与示例

### 缓冲区初始化

高吞吐与低延迟的 EP 操作共享同一个 `EPBuffer` 接口。直接使用 MoE 设置初始化缓冲区；SM 与 QP 数量通过解析式方法估算，并且可以在每次调用时覆盖。

为每个 EP 组创建并复用一个缓冲区。所有 rank 必须就 `num_max_tokens_per_rank` 达成一致；请选择一个能够覆盖预期训练、prefill 和解码批次规模的公共容量。下面的示例管理一个 EP 组。在替换其缓冲区之前，请先完成所有未完成的操作。

```python
import torch.distributed as dist
from typing import Optional

from deep_ep import EPBuffer

# Communication buffer (will allocate at runtime)
_buffer: Optional[EPBuffer] = None

# Number of SMs to use for communication kernels (will be set at buffer creation)
_num_comm_sms: int = 0


def get_buffer(group: dist.ProcessGroup,
               num_max_tokens_per_rank: int,
               hidden: int,
               num_topk: int,
               num_experts: int,
               use_fp8_dispatch: bool = False) -> EPBuffer:
    """Initialize or retrieve the EPBuffer for EP communication."""
    global _buffer, _num_comm_sms

    # Check if we can reuse the existing buffer
    required_bytes = EPBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden,
        num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
        return _buffer

    # Allocate a new buffer with MoE settings
    _buffer = EPBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    # Estimate the SM count from the topology and communication volume
    # You may also specify `num_sms` manually in dispatch/combine calls to override
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)

    return _buffer
```

### 在模型训练与推理中的使用示例

训练、推理 prefill 和推理解码使用相同的 `EPBuffer` dispatch 与 combine API。该示例使用面向分组专家 GEMM 的展开布局，并将前向 epilogue 推迟到框架等待其结果时才执行。将 `expert_alignment` 设置为分组 GEMM 的 token 对齐量；对于 DeepGEMM，请使用 `deep_gemm.get_mk_alignment_for_contiguous_layout()`。

`do_cpu_sync=True` 用于获取精确的输出尺寸和 CPU 侧的专家计数，常用于训练和 prefill。当使用 GPU 侧的接收计数时（如解码场景），将其设为 `False`：此时输出会按配置的容量分配，专家 GEMM 必须使用 `handle.psum_num_recv_tokens_per_expert` 来确定有效的专家范围。新的路由决策需要一次全新的 dispatch；仅靠解码并不会使旧的路由句柄（handle）变为可复用。

```python
import torch
from typing import Optional, Tuple, Union

from deep_ep import EPBuffer, EPHandle, EventHandle, EventOverlap


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int,
                     num_max_tokens_per_rank: int,
                     expert_alignment: int = 1,
                     do_cpu_sync: bool = True,
                     previous_event: Optional[EventHandle] = None) -> EventOverlap:
    """
    MoE dispatch: route tokens to the corresponding experts across all ranks.
    Supports both BF16 and FP8 (x as a tuple of [data, scale_factors]) inputs.
    Wait on the returned event to obtain the expanded tensors and routing handle.
    """
    global _buffer, _num_comm_sms

    return _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        num_sms=_num_comm_sms,
        previous_event=previous_event,
        async_with_compute_stream=True,
        allocate_on_comm_stream=previous_event is not None,
        do_cpu_sync=do_cpu_sync,
        do_expand=True,
        do_zero_padding=True,
        use_tma_aligned_col_major_sf=True,
        defer_epilogue=True,
    )


def dispatch_backward(grad_recv_x: torch.Tensor,
                      grad_recv_topk_weights: torch.Tensor,
                      handle: EPHandle,
                      bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    """The backward pass of MoE dispatch is actually a combine."""
    global _buffer, _num_comm_sms

    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x,
        handle=handle,
        bias=bias,
        topk_weights=grad_recv_topk_weights,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_grad_x, combined_grad_topk_weights, event


def combine_forward(x: torch.Tensor,
                    handle: EPHandle,
                    bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                    previous_event: Optional[EventHandle] = None) -> EventOverlap:
    """MoE combine: reduce expert outputs back to their original ranks."""
    global _buffer, _num_comm_sms

    return _buffer.combine(
        x,
        handle=handle,
        bias=bias,
        num_sms=_num_comm_sms,
        previous_event=previous_event,
        async_with_compute_stream=True,
        allocate_on_comm_stream=previous_event is not None,
        defer_epilogue=True,
    )


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: EPHandle) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    """The backward pass of MoE combine is actually a dispatch."""
    global _buffer, _num_comm_sms

    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
        do_expand=True,
        do_zero_padding=True,
        use_tma_aligned_col_major_sf=True,
    )

    return grad_x, event
```

若要实现通信与计算的重叠，先发起通信，并在专家计算需要其结果的位置等待：

```python
# Use do_cpu_sync=False for decoding with GPU-side expert counts
event = dispatch_forward(...)

# ... do some independent computation here ...

# Run the deferred dispatch epilogue and obtain its results
recv_x, _, recv_topk_weights, handle = event.current_stream_wait()

# ... run the expert GEMMs and apply recv_topk_weights to produce expert_output ...
event = combine_forward(expert_output, handle)

# ... do some independent computation here ...
combined_x, _ = event.current_stream_wait()
```

展开后的 `recv_topk_weights` 是一维的，每个专家行对应一个值。专家计算会在 combine 之前应用这些路由器权重；combine 会对所提供的专家输出进行求和。Combine 的输入必须为 BF16。`bias` 可以在 combine epilogue 期间加入共享专家（shared-expert）的输出或残差。

训练时会保存前向的 `handle` 以供 `combine_backward` 和 `dispatch_backward` 使用。上文的反向辅助函数返回张量和一个事件；在使用这些张量之前请先等待该事件完成。缓存的 dispatch 会重放已保存的展开布局，而无需再做一次接收计数的 CPU 同步。在该句柄的所有使用完成之前，请保持原始 `topk_idx` 不变。

### 延迟 epilogue（Deferred epilogues）

Dispatch 与 combine 接受 `defer_epilogue=True` 配合 `async_with_compute_stream=True` 使用。在这种模式下，调用会直接返回一个 `EventOverlap`。调用 `.wait()` 会在当前流上执行被延迟的 epilogue，并针对 dispatch 返回 `(recv_x, recv_topk_idx, recv_topk_weights, handle)`，或针对 combine 返回 `(combined_x, combined_topk_weights)`。

返回的结果由第一次等待产生，因此请保留它供后续计算使用。如果没有其他工作可以重叠，请立即调用 `.wait()`。`do_zero_padding=True` 会清零专家之间的对齐空隙；在无 CPU 同步模式下，它并不会使未使用的输出容量变成有效数据。

如果独立计算已经在通信调用之前入队，请在该计算入队之前用 `_buffer.capture()` 捕获输入就绪事件，并将其作为 `previous_event` 传入。此时前向辅助函数会按 EP 的要求设置 `allocate_on_comm_stream=True`。被延迟的 epilogue 仍然在 `.wait()` 时于当前流上执行；这也使得由中间计算产生的 combine `bias` 可以在该处被消费。

### Bucket 集合通信（实验性）

`BucketBuffer` 为 CP 和 DP 负载提供批量的 `all_gather`、`reduce_scatter` 和 `all_reduce`。在常规用法中，输入必须位于 bucket 自己的存储中。请在创建缓冲区之前使用 `BufferAllocator` 规划张量。当缓冲区被构建时，规划中的元张量（meta tensor）会成为指向该缓冲区的 CUDA 视图。

```python
import torch

from deep_ep import BufferAllocator, BucketBuffer

# group is an initialized CP or DP process group
allocation_plan = BufferAllocator()
x = allocation_plan.allocate((group.size(), 1024), torch.float32)
buffer = BucketBuffer(group, allocation_plan)

# Gather each rank's local shard into the registered tensor
x[group.rank()].fill_(group.rank())
gathered_x = buffer.all_gather(x[group.rank()]).wait()

# Reduce-scatter and all-reduce use FP32 tensors in the buffer
x.fill_(1)
shard = buffer.reduce_scatter(x).wait()
x.fill_(1)
reduced_x = buffer.all_reduce(x).wait()
```

所有 rank 必须使用相同的分配规划。每个集合通信操作也接受一个张量列表，`.wait()` 会返回对应的张量或张量列表。结果是缓冲区的视图；all-gather 返回展平（flattened）的视图。Reduce-scatter 与 all-reduce 接受 FP32 输入，并默认对贡献求和；可使用 `scale` 对结果进行缩放。Reduce-scatter 还支持 `comm_precision="bf16"`，以较低的精度进行通信，同时保持 FP32 的输入与输出张量。

要使用在 bucket 之外分配的张量，请将集合通信调用包裹在 `buffer.session()` 中。会话会管理临时的 bucket 存储以及所需的暂存拷贝（staging copies）：

```python
with buffer.session():
    reduced_x = buffer.all_reduce(external_x).wait()
    # Consume or copy reduced_x before reusing the session storage
```

仅使用 NVLink 的 all-gather 路径也支持外部源，但需要显式提供 bucket 内的 `dsts`；对于一般的 bucket 外输入，请使用会话。

这三种集合通信均支持 NVLink、RDMA 以及混合域。NVLink 上的归约需要 NCCL 的 multimem 支持。All-gather 由拷贝引擎（copy engine，`num_sms=0`）驱动；reduce-scatter 与 all-reduce 使用 GPU SM。更多示例（包括在多个通信组中的注册）请参见 [bucket 测试](tests/bucket)和[分配测试](tests/buffer/test_allocation.py)。

### 专家负载均衡

动态专家复制（dynamic expert replication）会将负载较重的专家的计算分摊到额外的 GPU 上。[MoonEP](https://github.com/MoonshotAI/MoonEP) 探索了基于动态冗余专家的在线规划、权重预取与梯度归约。[UltraEP](https://github.com/Dots-Infra/UltraEP)（[论文](https://arxiv.org/abs/2606.04101)）根据当前门控后（post-gating）的负载来规划复制与 token 重路由，并将专家复制保持在 NVLink 域内。这些工作启发了 `EPBuffer` 所暴露的冗余专家通信原语。

- `lb_prefetch_weights(redundant_expert_weights, expert_weights, redundancy_mapping)` 会在专家计算之前，将原始专家权重推送到所请求的冗余槽位。它接受单个张量或相互匹配的非空张量序列，因此权重与量化 scale 可以一起传输而无需 dtype 转换。
- `lb_reduce_grads(redundant_expert_grads, expert_grads, redundancy_mapping)` 会通过 NVLink 读取冗余专家的 FP32 梯度，并将其原地累加进原始专家的梯度。请使用与前向传播相同的映射，并在归约之前用本地贡献（或零）初始化 `expert_grads`。

调用方负责提供复制规划、将 token 重路由到所选的专家实例，并管理专家 GEMM。这两个交换操作在每个 NVLink 域内都是集合性的：该域中的每个 rank 都必须调用它们，包括未被分配冗余槽位的 rank。它们在通信流上异步执行并返回一个 `EventOverlap`；在消费其结果之前请调用 `.wait()`。`previous_event` 可以指定输入何时就绪，`num_sms=0` 则选择解析式 SM 估算。

使用 `lb_allocation_plan_or_num_bytes` 为冗余权重与梯度预留存储，可传入 `BufferAllocator` 规划或按对齐的字节数。该 LB 区域与 EP 的 dispatch/combine 缓冲区相互独立。冗余张量必须位于其中；原始专家权重与梯度可以使用普通的 CUDA 分配。在 NVLink 域内的每个 rank 上，规划必须具有相同的形状和分配顺序。

`redundancy_mapping` 必须是一个形状为 `[num_nvlink_ranks, num_redundant_experts]` 的连续 CUDA int32 张量，且在域内每个 rank 上内容完全一致。条目 `[r, c]` 将某个专家分配给域内本地 rank `r` 上的冗余槽位 `c`，若槽位未使用则为 `-1`。专家 ID 的形式为 `owner_rank * num_local_experts + local_expert_idx`，使用域内本地 rank 索引。请将副本分配给其他 peer；某个 rank 自己的专家不得出现在它自己的冗余槽位中。未使用的槽位会被跳过。

所有权重与梯度张量必须是连续的 CUDA 张量。权重张量的最高维为 `num_local_experts`，冗余张量为 `num_redundant_experts`；尾部的形状可以不同，只要每个匹配对的每个专家字节数相同，且是 `deep_ep.get_num_tma_alignment()`（32 字节）的倍数。域内的所有 rank 必须具有相同的 `num_local_experts`。梯度归约接受二维 FP32 张量；请将每个专家的参数展平为一行，并保持相同的字节对齐。

```python
import torch

from deep_ep import BufferAllocator, EPBuffer

# Use the same plan on every rank; each example expert row is 1024 elements
lb_plan = BufferAllocator()
redundant_weights = lb_plan.allocate((num_redundant_experts, 1024), torch.bfloat16)
redundant_grads = lb_plan.allocate((num_redundant_experts, 1024), torch.float32)
buffer = EPBuffer(
    group,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    lb_allocation_plan_or_num_bytes=lb_plan,
)

# expert_weights: contiguous CUDA BF16 [num_local_experts, 1024]
# redundancy_mapping: supplied by the caller's replication planner
weights_ready = buffer.lb_prefetch_weights(
    redundant_weights, expert_weights, redundancy_mapping,
)
# ... independent computation can overlap the exchange ...
weights_ready.wait()
# ... run expert computation using the prefetched weights ...

# After backward has written redundant_grads and the local expert_grads:
grads_ready = buffer.lb_reduce_grads(
    redundant_grads, expert_grads, redundancy_mapping,
)
# ... independent backward computation can overlap the reduction ...
grads_ready.wait()
# expert_grads now includes contributions from all redundant instances
```

在对应的交换完成之前，请保持映射与张量存储有效。如果冗余存储在多个层或微批次间被复用，请在反向计算之前恢复所需的权重，并在复用其输入之前完成梯度归约。归约不会清除冗余梯度；在下一次累加之前请覆盖它们或将它们置零。完整的分配与正确性示例请参见[权重预取测试](tests/ep/test_prefetch_weights.py)和[梯度归约测试](tests/ep/test_reduce_grads.py)。

### Engram（实验性）

`EngramBuffer` 支持基于 RDMA 的 GPU/CPU 存储与多层抓取。使用 `get_storage_size_hint` 和 `get_theoretical_config` 来确定缓冲区及其 QP 的规模，然后使用 `set_config` 和 `write` 填充层表。

`fetch(indices)` 接受一个形状为 `[num_layers, num_tokens, num_entries_per_token]` 的 int32 张量，并为每层返回一个完成钩子。在使用某一层抓取到的数据之前，请调用对应的钩子。BF16、FP8 与 CPU 存储的示例请参见 [Engram 测试](tests/engram/test_engram.py)。

### 流水线并行（实验性）

`PPBuffer` 在仅含 RDMA 的流水线组中的相邻 rank 之间提供 `send(x, dst_rank_idx)` 和 `recv(x, src_rank_idx)`。请在构建时设置 `num_max_tensor_bytes` 和 `num_max_inflight_tensors` 以预留 send/recv 槽位。输入与输出必须是指针和大小均按 32 字节对齐的连续 CUDA 张量。用法请参见 [PP 测试](tests/pp/test_pp.py)。

### 环境变量

请在导入 `deep_ep` 并创建缓冲区之前设置运行时变量。除非另有说明，标志的取值为 `0`/`1`。下列默认值假设打包时未内置任何构建期默认值。

**运行时与网络**

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_BUFFER_DEBUG` | 未设置 | 设为 `1` 以打印初始化、拓扑、缓冲区大小及 SM 估算的诊断信息。保持未设置则禁用所有诊断；Python 侧的检查也会将字符串 `"0"` 视为已启用。 |
| `EP_SUPPRESS_NCCL_CHECK` | `0` | 跳过导入时对重复 NCCL 库以及与所选安装二进制一致性方面的检查。不会绕过 C++ 设备通信域的兼容性检查。 |
| `EP_AVOID_RECORD_STREAM` | `0` | 对于 EP 与 bucket 操作，在完成事件中保留张量，而不是调用 `record_stream`。在通信完成之前，请保持该事件存活。 |
| `EP_REUSE_NCCL_COMM` | `1` | 当 PyTorch 进程组的后端暴露 `_comm_ptr` 时，复用其 NCCL 通信域；否则创建一个由 DeepEP 管理的通信域。 |
| `EP_DEFAULT_RDMA_SL` | 未设置 | 当未提供缓冲区的 `sl_idx` 时，设置 Gin 的流量类别/服务级别。若两者均未设置，则使用 NCCL 的默认值。 |
| `EP_OVERRIDE_RDMA_SL` | 未设置 | 同时覆盖 `EP_DEFAULT_RDMA_SL` 与缓冲区的 `sl_idx`。 |
| `EP_DISABLE_GIN` | `0` | 在仅使用 NVLink 时跳过 Gin 初始化。RDMA 操作需要 Gin；此标志不会选择其他 RDMA 后端。 |
| `EP_NUM_MAX_LOCAL_RANKS` | `16` | 仅限 Engram：在混合模式下为 `NCCL_WIN_STRIDE` 的大小设置估算已注册存储。这只是一个容量估算，并非 rank 数量上限。 |

**JIT 编译**

DeepJIT 会优先读取 `EP_JIT_*`，然后才读取对应的 `DJ_JIT_*` 变量作为全局回退。请在第一次内核编译之前完成这些配置。显式的 `EP_JIT_*` 值（包括打包内置的默认值）优先于 `DJ_JIT_*`。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_JIT_DEBUG` | `0` | 启用编译命令与内核加载诊断、PTXAS 输出、源码行信息以及 PTX/SASS 转储。 |
| `EP_JIT_CACHE_DIR` | `$HOME/.dj` | 缓存根目录，或以冒号分隔的多个根目录。按顺序搜索所有根目录，并将新编译的产物写入第一个。 |
| `EP_JIT_NVCC_COMPILER` | 检测到的 toolkit 的 `bin/nvcc` | 覆盖 NVCC 可执行文件；但仍必须能找到有效的 CUDA toolkit 根目录。 |
| `EP_JIT_CPP_STANDARD` | `20` | 传给 NVCC 的 C++ 标准。DeepEP 要求 C++20 或更新版本。 |
| `EP_JIT_PRINT_COMPILER_COMMAND` | `0` | 打印编译器与反汇编器命令。 |
| `EP_JIT_PRINT_LOAD_TIME` | `0` | 打印内核二进制的加载时间。 |
| `EP_JIT_PTXAS_VERBOSE` | `0` | 启用并打印详细的 PTXAS 输出。 |
| `EP_JIT_CHECK_NO_SPILLS` | `0` | 拒绝存在寄存器溢出（register spill）的已编译内核。 |
| `EP_JIT_CHECK_NO_LOCAL_MEMORY` | `0` | 拒绝使用本地内存（local memory）的已编译内核。 |
| `EP_JIT_WITH_LINEINFO` | `0` | 嵌入用于性能剖析的源码行信息。 |
| `EP_JIT_DUMP_ASM` | `0` | 在缓存未命中时同时生成 PTX 与 SASS 产物。 |
| `EP_JIT_DUMP_PTX` | `0` | 在缓存未命中时生成 PTX 产物。 |
| `EP_JIT_DUMP_SASS` | `0` | 在缓存未命中时生成 SASS 产物；需要 toolkit 中的 `cuobjdump`。 |
| `EP_GIN_GDAKI_DEBUG` | `0` | 编译 JIT 内核时启用 NCCL Gin GDAKI 设备调试。 |

DeepJIT 依次通过 `CUDA_HOME`、`CUDA_PATH`、`PATH` 中的 `nvcc`，最后是 `/usr/local/cuda` 来发现 CUDA toolkit。

**构建与依赖发现**

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_NCCL_ROOT_DIR` | 自动检测 | 包含 `include/` 与 `lib/` 的 NCCL 安装目录，在构建和导入时使用。优先级高于 `NCCL_DIR`，然后是 NVIDIA Python 包发现机制。 |
| `EP_NUM_TOPK_IDX_BITS` | `64` | 构建期的 top-k 索引位宽（`32` 或 `64`）。路由索引请使用导出的 `deep_ep.topk_idx_t`。更改此值需要重新构建扩展。 |

在包构建期间设置时，以下变量会作为导入时默认值被存储：`EP_JIT_CACHE_DIR`、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_JIT_CPP_STANDARD`、`EP_NUM_TOPK_IDX_BITS`、`EP_NCCL_ROOT_DIR`、`EP_DEFAULT_RDMA_SL` 和 `EP_OVERRIDE_RDMA_SL`。导入时，已有的环境变量值优先。在运行时覆盖 `EP_NUM_TOPK_IDX_BITS` 并不会改变已编译的索引类型。

**测试剖析**

这些标志影响 [deep_ep/utils/testing.py](deep_ep/utils/testing.py) 中的 `bench_kineto`，而不影响生产环境通信：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_USE_NVIDIA_TOOLS` | `0` | 在使用 Nsight 或 Compute Sanitizer 时跳过内部剖析器。启用此项时，报告的内部耗时仅为占位值。 |
| `EP_DISABLE_BARRIER_PROFILING` | `0` | 禁用插入在每个被剖析迭代之前的屏障与延迟。 |

## 网络配置

DeepEP 在 InfiniBand 网络上经过了充分测试。不过，理论上它也兼容基于融合以太网的 RDMA（RoCE）。

### 流量隔离

InfiniBand 通过虚拟通道（Virtual Lane，VL）支持流量隔离。

为防止不同类型的流量相互干扰，我们建议按如下方式将负载隔离到不同的虚拟通道：

- 专家并行（expert-parallel）负载
- 其他负载

通过缓冲区的 `sl_idx` 参数选择 RDMA 服务级别（service level）。优先级为 `EP_OVERRIDE_RDMA_SL` > `sl_idx` > `EP_DEFAULT_RDMA_SL` > NCCL 的默认值。网络侧的 SL 到 VL 映射决定了由哪个虚拟通道承载该流量。

### 自适应路由

自适应路由（Adaptive routing）是 InfiniBand 交换机提供的一种高级路由特性，可以将流量均匀分布到多条路径上。尽管自适应路由会引入额外的延迟，我们仍建议在各种网络负载条件下都启用它。

### 拥塞控制

对于追求最大带宽的负载，我们建议在验证网络配置之后禁用拥塞控制。如果拥塞不可避免，请将这些负载放到优先级较低的虚拟通道上。DeepEP 不会在网络侧配置拥塞控制。

### PCI 原子模式

如果硬件支持，我们建议使用以下命令设置网卡的 `PCI_ATOMIC_MODE`，以提升 RDMA 原子操作的性能：

```bash
# Replace mlx5_0 with the target NIC
sudo mlxconfig -y -d mlx5_0 set PCI_ATOMIC_MODE=4
```

## 实验性分支

以下链接描述了独立的实现与研究分支。它们的特性与依赖仅适用于那些分支；在与当前 API 集成之前，请先查阅各分支的说明。

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
    - 移除 PyTorch 张量与通信缓冲区之间的拷贝，从而显著降低 normal 内核的 SM 占用
    - 该 PR 由 **腾讯网络平台部** 编写
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
    - 使用低延迟（low-latency）协议，消除由 RDMA 原子操作引入的额外 RTT 延迟
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
    - 一种新的后端实现，使用 TMA 指令实现最小的 SM 占用，并支持更大的 NVLink 域
    - 面向单批次场景的细粒度通信-计算重叠
    - 为无 NVLink 的环境提供 PCIe 内核支持
    - NVFP4 数据类型支持
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
    - 本优化系列由 **蚂蚁集团网络平台部** 编写
    - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347) 通过将通信内核执行与网卡 token 传输解耦来消除 RDMA 路径上的 SM 占用，将 SM 释放给计算
    - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483) 通过信号机制将 Down GEMM 计算与 Combine Send 通信重叠，以降低端到端延迟
    - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500) 使用轨优化（rail-optimized）转发与数据合并来优化跨节点 LL 算子的通信，以降低延迟
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
    - 基于 [MORI](https://github.com/ROCm/mori) 后端的 ROCm/AMD GPU 支持（低延迟模式）
- [nvDev](https://github.com/deepseek-ai/DeepEP/tree/nvDev)
    - 基于 V2 的分支，包含最新的 CUDA 特性，例如在小 token 规模下带来更低延迟的计算结构传输（Compute Fabric Transport，CFT）。

## 社区分支（Community forks）

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep) - 支持 DeepEP 运行在异构 GPU（如 Nvidia、AMD）和网卡（如 EFA、Broadcom、CX7）上
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) - 在 IBRC 传输中增加多 QP 方案与双端口网卡支持
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace) - 一个诊断分析器，可高效、精准地定位慢 rank
- [ROCm/mori](https://github.com/ROCm/mori) - AMD 面向性能关键型 AI 负载（如 Wide EP、KVCache 传输、集合通信）的下一代通信库

## 致谢

DeepEP 构建在 [NCCL](https://github.com/nvidia/nccl) 的 Gin 后端之上。感谢 [@sjeaugey](https://github.com/sjeaugey)、[@pakmarkthub](https://github.com/pakmarkthub)、[@sb17v](https://github.com/sb17v)、[@xiaofanl-nvidia](https://github.com/xiaofanl-nvidia) 以及 NCCL 团队的支持！

我们也感谢 [MoonEP](https://github.com/MoonshotAI/MoonEP) 和 [UltraEP](https://github.com/Dots-Infra/UltraEP) 在动态专家复制以及支撑专家负载均衡的权重/梯度交换方面的工作。

## 许可证

本代码仓库基于 [the MIT License](LICENSE) 发布。

## 引用

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
