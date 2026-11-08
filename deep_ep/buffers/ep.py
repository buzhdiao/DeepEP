import functools  # 用于 partial：延迟绑定的确定性排序回调
import os  # 读取环境变量（EP_BUFFER_DEBUG 等调试开关）
import math  # 数学工具：comb 组合数、ceil 向上取整
import torch  # PyTorch：张量操作、CUDA 设备属性与同步
import torch.distributed as dist  # 分布式进程组
from typing import Optional, Sequence, Tuple, Union  # 类型注解

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # C++ 扩展：EPBuffer 运行时与缓冲区大小计算
# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle  # C++ 层的 CUDA 事件句柄类型

from .allocator import BufferAllocator  # 分配计划器：在 LB 区域上落实多张量布局
from .base import BufferBase  # 缓冲区抽象基类，统一生命周期管理
from .. import comm  # 通信工具模块
from ..utils.event import EventOverlap  # 事件句柄：支持 wait() 与计算/通信重叠
from ..utils.math import align  # 向上对齐工具函数
from ..utils.semantic import value_or, weak_lru  # 语义工具：参数默认值兜底、弱引用缓存装饰器
from ..utils.envs import (  # 环境探测：RDMA 原子/NVLink 拓扑/确定性开关/各类带宽估计
    check_fast_rdma_atomic_support,
    check_nvlink_connections, check_torch_deterministic,
    get_nvlink_gbs, get_rdma_gbs,
    get_sm_read_gbs, get_sm_write_gbs
)


# dispatch 返回的通信句柄：携带路由与布局元数据，可回传给后续 dispatch 作缓存，
# 并供 combine 使用以逆向还原 token 路由
class EPHandle:
    """
    Communication handle returned by `EPBuffer.dispatch`.
    Can be reused as a cached handle in subsequent `EPBuffer.dispatch` calls to skip layout recomputation,
    and is consumed by `EPBuffer.combine` to reverse the token routing.

    Attributes:
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        num_experts: the number of all experts.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        num_sms: the SM count used during dispatch (reused in combine).
        topk_idx: original top-k expert indices from dispatch, `[num_tokens, num_topk]`.
            Must not be modified while this handle is in use. PyTorch version counters detect ordinary
            in-place changes, including through views, but not writes through `.data` or raw pointers.
            Tensors created in inference mode have no version counter and cannot be checked.
        psum_num_recv_tokens_per_scaleup_rank: inclusive prefix sum of deduplicated received token counts
            per scaleup rank, shape `[num_scaleup_ranks]`. A token is counted once per rank even if
            multiple of its top-k experts land on the same rank. The last element equals the total number
            of received tokens.
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_local_experts]`. Each expert's count is padded to `expert_alignment`.
            In non-expand mode, this is the inclusive prefix sum. In expand mode, `psum[i]` equals
            the aligned cumulative count of experts before `i` plus the actual (unaligned) token count
            of expert `i` — so `psum[i] - align(psum[i-1], expert_alignment)` recovers the real
            count for expert `i`, and `align(psum[i], expert_alignment)` gives expert `i+1`'s
            starting offset.
        num_recv_tokens_per_expert_list: Python list of per-expert received token counts (CPU-side).
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_local_experts]` with `torch.int`. Only populated in expand mode.
        recv_src_metadata: source token indices and buffer slot indices.
        dst_buffer_slot_idx: destination buffer slot indices from dispatch.
        token_metadata_at_forward: per-channel forwarded token metadata (hybrid mode only).
        channel_linked_list: per-channel per-scaleup-peer linked list (hybrid mode only).
        num_recv_tokens: the total number of received tokens.
    """

    # 保存 dispatch 的路由/布局元数据：既可作为缓存句柄传入后续 dispatch 跳过布局重算，
    # 也是 combine 反向还原 token 路由所必需的输入
    def __init__(self,
                 do_expand: bool,  # 是否使用展开布局（每个专家槽一个 token）
                 num_experts: int, expert_alignment: int,  # 专家总数与每个本地专家收到的 token 数对齐粒度
                 num_max_tokens_per_rank: int,  # 每 rank 最大 token 数（各 rank 必须一致）
                 num_sms: int,  # dispatch 使用的 SM 数（combine 时复用）
                 topk_idx: torch.Tensor,  # dispatch 的原始 top-k 专家索引 `[num_tokens, num_topk]`
                 num_recv_tokens: int,  # 接收 token 总数（未做 CPU 同步时可能不准确）
                 num_expanded_tokens: int,  # 展开布局下的 token 总数（含对齐填充）
                 num_recv_tokens_per_expert_list: list,  # 每个专家接收 token 数的 Python 列表（CPU 侧）
                 psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,  # 每 scaleup rank 去重后接收 token 数的前缀和
                 psum_num_recv_tokens_per_expert: torch.Tensor,  # 每个本地专家对齐填充后接收 token 数的前缀和
                 num_unaligned_recv_tokens_per_expert: torch.Tensor,  # 每个本地专家实际（未对齐）接收 token 数，仅展开模式有值
                 recv_src_metadata: torch.Tensor,  # 源 token 索引与缓冲区槽位索引等来源元数据
                 dst_buffer_slot_idx: torch.Tensor,  # dispatch 时目标缓冲区槽位索引
                 token_metadata_at_forward: Optional[torch.Tensor],  # 每 channel 前向转发的 token 元数据（仅混合模式）
                 channel_linked_list: Optional[torch.Tensor]):  # 每 channel 每 scaleup 对端的链表（仅混合模式）
        assert topk_idx is not None  # topk_idx 是句柄的核心路由信息，必须存在

        self.do_expand = do_expand  # 保存布局模式标志
        self.num_experts = num_experts  # 保存专家总数
        self.expert_alignment = expert_alignment  # 保存专家对齐粒度
        self.num_max_tokens_per_rank = num_max_tokens_per_rank  # 保存每 rank 最大 token 数
        self.num_sms = num_sms  # 保存 dispatch 的 SM 数，combine 默认复用
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank  # 保存 scaleup 维度的接收前缀和
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert  # 保存专家维度的接收前缀和
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert  # 保存未对齐的每专家计数
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list  # 保存 CPU 侧每专家计数列表
        self.recv_src_metadata = recv_src_metadata  # 保存来源元数据（源 token 索引 + 缓冲槽位）
        self.dst_buffer_slot_idx = dst_buffer_slot_idx  # 保存目标缓冲槽位索引
        self.token_metadata_at_forward = token_metadata_at_forward  # 保存前向转发元数据（混合模式）
        self.channel_linked_list = channel_linked_list  # 保存 channel 链表（混合模式）

        # May not be accurate without CPU sync
        self.num_recv_tokens = num_recv_tokens  # 接收 token 总数；未做 CPU 同步时该值可能不准确
        self.num_expanded_tokens = num_expanded_tokens  # 展开 token 总数；同样可能未经 CPU 同步

        # For deterministic features
        self.cached_recv_src_metadata_before_sort = None  # 确定性排序前缓存的原始来源元数据

        # `topk_idx` cannot change in the handle
        self._topk_idx = topk_idx  # 内部保存的 topk_idx 引用（句柄存续期间不允许修改）
        self._topk_idx_version = None if topk_idx.is_inference() else topk_idx._version  # 记录版本号以探测原地修改；inference 模式张量无版本号

    # 属性：返回 topk_idx，并在读取时校验其未被原地修改（通过 PyTorch 版本号）
    @property
    def topk_idx(self) -> torch.Tensor:
        assert self._topk_idx_version is None or self._topk_idx._version == self._topk_idx_version, \
            '`topk_idx` must not be modified while the EP handle is in use'  # 版本号变化说明 topk_idx 被修改，句柄失效
        return self._topk_idx  # 返回内部保存的 topk_idx

    # 确定性排序：把接收 token 按来源排序，保证同样的输入得到确定的 dispatch 输出顺序
    def deterministic_sort(self,
                           do_cpu_sync: bool,
                           is_cached_dispatch: bool,
                           recv_x: torch.Tensor,
                           recv_sf: Optional[torch.Tensor],
                           recv_topk_idx: torch.Tensor,
                           recv_topk_weights: torch.Tensor,
                           channel_linked_list: Optional[torch.Tensor]):
        """
        Sort received tokens to guarantee deterministic dispatch output.
        The principle:
          - Non-expand mode: sort everything that depends on the receive order, including
            `recv_x`, `recv_sf`, `recv_topk_weights`, `recv_topk_idx`, and `self.recv_src_metadata`
            (`recv_src_metadata` is sorted only for non-cached dispatch, since it is not regenerated in cached mode).
          - Expand mode: only sort the expanded arrays — `recv_x`, `recv_sf`, and `recv_topk_weights`.
            The slot pointers in `self.recv_src_metadata[:, 2:]` are updated to reflect the new positions, but `self.recv_src_metadata` itself is not permuted.
        """

        # NOTE: `self.recv_src_metadata` is generated once during non-cached dispatch and is not
        # regenerated during cached dispatch (applies to both expand and non-expand mode). So we:
        #  1. Cache it for later sorting
        #  2. Only permute `self.recv_src_metadata` in non-cached mode
        if not is_cached_dispatch:  # 非缓存 dispatch 时 recv_src_metadata 才是新生成的
            self.cached_recv_src_metadata_before_sort = self.recv_src_metadata.clone()  # 缓存排序前的元数据（缓存 dispatch 不再生成它）
        assert self.cached_recv_src_metadata_before_sort is not None  # 必须已有缓存可依据
        sort_keys = self.cached_recv_src_metadata_before_sort[:, 0]  # 排序键：源 token 的全局索引（第 0 列）

        # Ignore trailing tokens by setting their `sort_keys` to max
        num_recv_tokens = self.psum_num_recv_tokens_per_scaleup_rank[-1] if not do_cpu_sync else self.recv_src_metadata.shape[0]  # GPU 侧用前缀和取真实接收数，CPU 同步后则等于元数据行数
        if not do_cpu_sync:  # 未做 CPU 同步时缓冲区可能有"尾部幻影" token
            oob_tokens_mask = torch.arange(0, self.recv_src_metadata.shape[0], device=self.recv_src_metadata.device) >= num_recv_tokens  # 标记超出真实接收数的越界 token
            sort_keys = sort_keys.clone()  # 克隆排序键以便安全写入
            sort_keys[oob_tokens_mask] = torch.iinfo(sort_keys.dtype).max  # 越界 token 的键设为类型最大值，排到末尾被忽略
        orig_indices = torch.sort(sort_keys).indices  # 按源 token 索引排序，得到目标排列

        # 由排列求逆排列：把"新位置 -> 旧位置"转换为"旧位置 -> 新位置"
        def get_reverse_permutation(perm: torch.Tensor) -> torch.Tensor:
            assert perm.dim() == 1  # 仅支持一维排列
            result = torch.empty_like(perm)  # 结果与排列同形状同类型
            result[perm] = torch.arange(0, perm.shape[0], dtype=perm.dtype, device=perm.device)  # 逆映射写入
            return result  # 返回逆排列

        # 按 orig_indices 原地重排张量（None 则跳过）
        def permute(tensor: Optional[torch.Tensor], orig_indices: torch.Tensor):
            if tensor is not None:  # 张量存在时才重排
                tmp = tensor[orig_indices]  # 按排列索引取出
                tensor.copy_(tmp)  # 原地写回，完成重排

        if not self.do_expand:  # 非展开模式：所有依赖接收顺序的内容都要排序
            # Non-expand mode
            # If cached dispatch is enabled, the `dispatch` kernel stores values according to `dst_buffer_slot_idx`, and the `dispatch_copy_epilogue_impl` kernel writes the info of token i into the i-th slot
            permute(recv_x, orig_indices)  # 重排接收 token 数据
            permute(recv_sf, orig_indices)  # 重排 FP8 scale factor
            permute(recv_topk_weights, orig_indices)  # 重排专家权重
            permute(recv_topk_idx, orig_indices)  # 重排专家索引
            if not is_cached_dispatch:  # 缓存 dispatch 不会重新生成该元数据
                permute(self.recv_src_metadata, orig_indices)  # 仅非缓存模式重排来源元数据

            if not is_cached_dispatch and channel_linked_list is not None:  # 混合模式下需同步修正 channel 链表
                valid_mask = (channel_linked_list >= 0) & (channel_linked_list < num_recv_tokens)  # 链表中有效的槽位指向
                to_indices = get_reverse_permutation(orig_indices)  # 求逆排列：旧位置 -> 新位置
                channel_linked_list[valid_mask] = to_indices[channel_linked_list[valid_mask]].to(channel_linked_list.dtype)  # 把链表指针更新到重排后的新位置

        elif not is_cached_dispatch:  # 展开模式且非缓存 dispatch（缓存模式下 epilogue 已按槽位放置，无需重排元数据本身）
            # Expand mode. In cached mode the copy epilogue places tokens according to
            # `self.recv_src_metadata[:, 2:]`, so we only need to permute when `is_cached_dispatch` is `False`.
            # In expand mode, `recv_x`, `recv_sf`, and `recv_topk_weights` are grouped by expert ID, possibly with padding (expert alignment). We permute tokens within each expert and update `self.recv_src_metadata[:, 2:]` accordingly.

            # Now we're going to construct the sorting key, which is:
            #  - `expert_idx*src_token_global_index_max_x2 + (-src_token_global_index_max) + src_token_global_idx`, for valid tokens
            #  - `expert_idx * src_token_global_index_max_x2`, for padding slots
            # This guarantees a two-key sort: first by expert, then by order within each expert.
            # Valid tokens precede padding tokens, and valid tokens are sorted by `src_token_global_idx`.
            src_token_global_index_max_x2 = 10000000000    # 1e10  # 排序键的基数：两倍源 token 全局索引上限，保证两个排序键不串位
            tensor_dim0_after_expand = recv_x.shape[0]  # 展开后的总行数（含各专家对齐填充）

            expert_token_idx_start = self.psum_num_recv_tokens_per_expert - self.num_unaligned_recv_tokens_per_expert  # 每个专家在展开数组中的起始偏移（前缀和减去未对齐计数）
            token_idx2expert_idx = torch.bucketize(torch.arange(tensor_dim0_after_expand, device='cuda'),
                                                   expert_token_idx_start[1:], right=True, out_int32=False)  # 把每个行号映射到所属专家编号
            sort_keys_for_expanded_tensors = token_idx2expert_idx * src_token_global_index_max_x2  # 高位排序键：先按专家分组

            slots = self.cached_recv_src_metadata_before_sort[:, 2:]    # [num_recv_tokens, topk]  # 每个 token 各 topk 项的展开槽位
            src_global_idx = self.cached_recv_src_metadata_before_sort[:, 0]  # 源 token 全局索引
            valid_mask = slots >= 0  # 槽位非负表示有效的 topk 项
            if not do_cpu_sync:  # 未做 CPU 同步时还要剔除越界 token 的槽位
                valid_mask[oob_tokens_mask] = False  # 越界 token 全部标记为无效
            sort_keys_for_expanded_tensors.scatter_add_(0, slots[valid_mask], -src_token_global_index_max_x2//2 + src_global_idx.unsqueeze(1).expand_as(slots)[valid_mask].to(torch.int64))  # 在每个有效槽位的键上叠加"偏置 + 源索引"：负偏置保证有效 token 排在填充槽之前，同专家内按源索引升序

            orig_indices_for_expanded_tensors = torch.sort(sort_keys_for_expanded_tensors, stable=True).indices.to(torch.int32)  # 稳定排序得到展开数组的排列（稳定保证同键时顺序确定）
            permute(recv_x, orig_indices_for_expanded_tensors)  # 重排展开后的 token 数据
            permute(recv_sf, orig_indices_for_expanded_tensors)  # 重排 scale factor
            permute(recv_topk_weights, orig_indices_for_expanded_tensors)  # 重排专家权重

            to_indices_for_expanded_tensors = get_reverse_permutation(orig_indices_for_expanded_tensors)  # 求逆排列：旧槽位 -> 新槽位
            self.recv_src_metadata[:, 2:][valid_mask] = to_indices_for_expanded_tensors[self.recv_src_metadata[:, 2:][valid_mask]]  # 更新元数据中的槽位指针到新位置（元数据行本身不重排）


# EP（专家并行）通信缓冲区：支持高吞吐专家并行 all-to-all（dispatch/combine），
# 节点内走 NVLink、跨节点走 RDMA，混合模式下采用分层的 RDMA+NVLink 通信
class EPBuffer(BufferBase):
    """
    The EP communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)

    Attributes:
        group: the communication group.
        rank_idx: the rank index.
        num_ranks: the number of ranks in the group.
        allow_hybrid_mode: whether to enable hybrid mode for multi-node communication. Hybrid mode uses
            hierarchical RDMA + NVLink communication to achieve higher bandwidth, and is more friendly
            to multi-plane/multi-rail networks.
        allow_multiple_reduction: whether to allow multiple reductions in combine. If disabled,
            only one reduction will be done in the combine epilogue for best precision,
            but it may increase data transfer size.
        prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            If enabled, we tend to use fewer SMs.
        num_bytes: the total buffer size in bytes.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        num_scaleout_ranks: the number of scaleout ranks.
        num_scaleup_ranks: the number of scaleup ranks.
        scaleout_rank_idx: the scaleout rank index of this rank.
        scaleup_rank_idx: the scaleup rank index of this rank.
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
        runtime: the C++ runtime.
    """

    # Common communication functions; refer to `deep_ep.comm` for usage.
    barrier = comm.barrier  # 类级别绑定通用 barrier 函数
    get_comm_stream = comm.get_comm_stream  # 获取专用通信 CUDA 流
    get_physical_domain_size = comm.get_physical_domain_size  # 获取物理域大小（RDMA/NVLink 维度）
    get_logical_domain_size = comm.get_logical_domain_size  # 获取逻辑域大小（scale-out/scale-up 维度）

    # 构造函数：创建 NCCL 通信句柄与 C++ 运行时，按 MoE 配置或显式字节数分配缓冲区
    def __init__(self,
                 group: dist.ProcessGroup,
                 # Provide `num_bytes` (excludes workspace)
                 num_bytes: Optional[int] = None,
                 # Or provide MoE settings (BF16 by default)
                 num_max_tokens_per_rank: int = 0,
                 hidden: int = 0,
                 num_topk: int = 0,
                 use_fp8_dispatch: bool = False,
                 # Load balance configs
                 lb_allocation_plan_or_num_bytes: Union[BufferAllocator, int] = 0,
                 # Configs
                 deterministic: bool = False,
                 allow_hybrid_mode: bool = True,
                 allow_multiple_reduction: bool = True,
                 prefer_overlap_with_compute: bool = True,
                 sl_idx: Optional[int] = None,
                 num_allocated_qps: int = 0,
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False):
        """
        Initialize the EP communication buffer.

        Arguments:
            group: the communication group.
            num_bytes: the total buffer size in bytes (excludes workspace), if set, overrides MoE-based calculation.
                Must be aligned to ``deep_ep.get_num_allocation_alignment()`` bytes.
            num_max_tokens_per_rank: the maximum number of tokens per rank, used for buffer size calculation.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            lb_allocation_plan_or_num_bytes: allocation plan or byte count for the LB region, separate
                from `num_bytes`. Zero disables the LB region. Byte counts must be aligned to
                ``deep_ep.get_num_allocation_alignment()`` bytes. When passing a plan, all ranks in
                an LSA domain must use the same tensor shapes and allocation order.
            deterministic: whether to use deterministic routing algorithms.
            allow_hybrid_mode: whether to enable hybrid mode.
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            sl_idx: the optional RDMA service level index. It overrides `EP_DEFAULT_RDMA_SL` and can be overridden by
                `EP_OVERRIDE_RDMA_SL`.
            num_allocated_qps: the number of QPs to allocate for RDMA (0 for automatic).
            num_cpu_timeout_secs: CPU-side timeout in seconds for CPU sync.
            num_gpu_timeout_secs: GPU-side timeout in seconds for GPU operations.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
        """
        # Some useful utilities
        self.group = group  # 保存进程组引用
        self.rank_idx = group.rank()  # 本 rank 在组内的编号
        self.num_ranks = group.size()  # 组内 rank 总数
        self.allow_hybrid_mode = allow_hybrid_mode  # 保存是否启用混合模式（分层 RDMA+NVLink）
        self.allow_multiple_reduction = allow_multiple_reduction  # 保存 combine 是否允许多次归约（换带宽/精度）
        self.prefer_overlap_with_compute = prefer_overlap_with_compute  # 保存是否倾向与计算重叠（偏少占 SM）
        self.deterministic = deterministic  # 保存是否启用确定性路由算法

        # Create NCCL comm handle
        self.nccl_comm_handle = comm.get_nccl_comm_handle(group)  # 获取该组对应的 NCCL 通信句柄

        # Calculate buffer size (already 2 MB-aligned from hint functions / calculate_ep_buffer_size)
        if num_bytes is None:  # 未显式给字节数时按 MoE 配置计算
            # NOTES: we allow `num_topk == 0`, as the buffer size can also be calculated by number of ranks (maybe bigger though)
            num_bytes = _C.calculate_ep_buffer_size(  # 委托 C++ 按 token/hidden/topk/FP8 及模式计算缓冲区大小
                self.nccl_comm_handle.get(),  # 传入通信器以获取域拓扑
                num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,  # MoE 配置：每 rank 最大 token 数、隐藏维度、topk、是否 FP8
                allow_hybrid_mode, allow_multiple_reduction)  # 混合模式与多次归约也会影响缓冲区大小

        if os.environ.get('EP_BUFFER_DEBUG', 0):  # 调试环境变量开启时打印初始化信息
            print(f'Initializing EP buffer with {num_bytes} bytes at rank EP {group.rank()}/{group.size()}')
        self.num_bytes = num_bytes  # 保存缓冲区总字节数（不含 workspace）

        # Load balance configs
        lb_allocation_plan = lb_allocation_plan_or_num_bytes if isinstance(lb_allocation_plan_or_num_bytes, BufferAllocator) else None  # 传入分配计划则取出
        num_lb_bytes = lb_allocation_plan_or_num_bytes if lb_allocation_plan is None else lb_allocation_plan.num_bytes  # 否则直接使用 LB 字节数
        if lb_allocation_plan is not None:  # 使用分配计划时
            assert not lb_allocation_plan.materialized  # 计划必须尚未落实
        assert isinstance(num_lb_bytes, int) and num_lb_bytes >= 0  # LB 字节数必须为非负整数（0 表示禁用 LB 区域）
        self.num_lb_bytes = num_lb_bytes  # 保存 LB 区域大小

        # Store default values
        self.num_max_tokens_per_rank = num_max_tokens_per_rank  # 保存默认的每 rank 最大 token 数，dispatch 未指定时使用

        # Check PCIe GPUs
        check_nvlink_connections(group)  # 检查组内 NVLink 拓扑（如 PCIe 机型会被指出）

        # Automatic maximum QP count allowed
        if num_allocated_qps == 0:  # 未显式指定时自动决定预分配 QP 数
            # Hybrid mode will consume more QPs
            # The extra QP is for notify warps
            if self.allow_hybrid_mode:  # 混合模式每个 channel 都要独立 QP，且额外一个用于 notify warp
                num_allocated_qps = 65 if check_fast_rdma_atomic_support() else 129  # 支持快速 RDMA 原子时减半
            else:  # 非混合模式：8 个数据 QP + 若干备用
                num_allocated_qps = 17
        self.num_allocated_qps = num_allocated_qps  # 保存预分配 QP 数

        # Create CPP handle
        super().__init__(explicitly_destroy)  # 初始化基类：记录显式销毁标志
        self.runtime = _C.EPBuffer(  # 创建 C++ EPBuffer 运行时：分配缓冲区、注册 RDMA 窗口、建立 QP
            self.rank_idx, self.num_ranks,  # 本 rank 编号与组内 rank 数
            self.nccl_comm_handle.get(), num_bytes, num_lb_bytes,  # 通信器指针、主缓冲区与 LB 区域大小
            allow_hybrid_mode, allow_multiple_reduction, prefer_overlap_with_compute,  # 各行为开关
            sl_idx, num_allocated_qps,  # RDMA 服务等级与预分配 QP 数
            num_cpu_timeout_secs, num_gpu_timeout_secs,  # CPU/GPU 两侧超时
            self.explicitly_destroy)  # 显式销毁标志
        self.context = self.runtime.context  # 保存第一个通信上下文

        # Materialize LB allocation plan
        if lb_allocation_plan is not None:  # 传入了 LB 分配计划时
            lb_allocation_plan.materialize(self.runtime.lb_storage)  # 在 LB 存储上落实计划，使占位张量指向真实内存

        # Logical rank indices
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()  # 逻辑域：跨节点与节点内 rank 数
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks  # 本 rank 的节点间（scale-out）编号
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks  # 本 rank 在节点内的（scale-up）编号

        # Physical rank indices
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()  # 物理域：需走 RDMA 与 NVLink 互联的 rank 数

        # Call a barrier to ensure initialization visibility for all peers
        torch.cuda.synchronize()  # 同步本地 CUDA 流，确保初始化完成
        group.barrier()  # 组内 barrier：等待所有 peer 完成初始化
        torch.cuda.synchronize()  # 再次同步，保证初始化结果全局可见

    # 销毁运行时并释放资源；需构造时传 explicitly_destroy=True
    def destroy(self) -> None:
        """
        Destroy the C++ runtime and release resources. Requires `explicitly_destroy=True` at construction.
        """
        super().destroy()  # 基类销毁：断言标志并销毁 C++ 运行时
        self.context = None  # 释放上下文引用
        self.nccl_comm_handle = None  # 释放 NCCL 通信句柄引用

    # 静态工具：按给定 MoE 配置估算推荐缓冲区大小（2MB 对齐），无需真正构造缓冲区
    @staticmethod
    def get_buffer_size_hint(group: dist.ProcessGroup,
                             num_max_tokens_per_rank: int, hidden: int,
                             num_topk: int = 0, use_fp8_dispatch: bool = False,
                             allow_hybrid_mode: bool = True,
                             allow_multiple_reduction: bool = True) -> int:
        """
        Get a recommended buffer size (in bytes) for the given MoE settings, without constructing the buffer.
        The returned value is aligned to 2 MB.

        Arguments:
            group: the communication group.
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to use FP8 for dispatch.
            allow_hybrid_mode: whether to enable hybrid mode.
            allow_multiple_reduction: whether to allow multiple reductions in combine.

        Returns:
            size: the recommended buffer size in bytes (2 MB-aligned).
        """
        # NOTES: calculate_ep_buffer_size already returns 2 MB-aligned values
        return _C.calculate_ep_buffer_size(  # 委托 C++ 计算推荐缓冲区大小（已 2MB 对齐）
            comm.get_nccl_comm_handle(group).get(),  # 传入通信器以获取域拓扑
            num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,  # MoE 配置
            allow_hybrid_mode, allow_multiple_reduction)  # 模式开关

    # 静态工具：从句柄中拆出缓存 dispatch 所需的全部布局信息（无句柄时返回一串 None）
    @staticmethod
    def _unpack_handle(handle: Optional[EPHandle] = None) \
        -> Tuple[Optional[int], Optional[int], Optional[list],
                 Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor]]:
        if handle is None:  # 无句柄：全新 dispatch，各项缓存均为 None
            return None, None, None, None, None, None, None, None, None, None
        return (handle.num_recv_tokens,  # 缓存的接收 token 总数
                handle.num_expanded_tokens,  # 缓存的展开 token 总数
                handle.num_recv_tokens_per_expert_list,  # 缓存的每专家计数列表
                handle.psum_num_recv_tokens_per_scaleup_rank,  # 缓存的 scaleup 维度前缀和
                handle.psum_num_recv_tokens_per_expert,  # 缓存的专家维度前缀和
                handle.num_unaligned_recv_tokens_per_expert,  # 缓存的未对齐每专家计数
                handle.dst_buffer_slot_idx,  # 缓存的目标缓冲槽位索引
                handle.token_metadata_at_forward,  # 缓存的前向转发元数据
                handle.recv_src_metadata,  # 缓存的来源元数据
                handle.channel_linked_list)  # 缓存的 channel 链表

    # 在当前流（torch.cuda.current_stream()）上捕获一个 CUDA 事件，用于后续跨流等待
    @staticmethod
    def capture() -> EventHandle:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event_handle: the captured event handle.
        """
        return EventHandle()  # 创建并返回捕获了当前流状态的 C++ 事件句柄

    # 按带宽模型估算 dispatch/combine 的最优 SM 数（结果按参数缓存）；假设 gate 均衡分布
    @weak_lru(maxsize=None)
    def get_theoretical_num_sms(self, num_experts: int, num_topk: int,
                                num_scaleout_topk: int = 0,
                                rdma_gbs: float = 0, nvlink_gbs: float = 0,
                                sm_read_gbs: float = 0, sm_write_gbs: float = 0) -> int:
        """
        Estimate the optimal number of SMs for dispatch/combine kernels based on bandwidth modeling.
        The result is cached. This assumes a balanced gate distribution.

        Arguments:
            num_experts: the number of all experts.
            num_topk: the number of top-k experts per token.
            num_scaleout_topk: reserved for balanced gate (must be 0 currently).
            rdma_gbs: the RDMA bandwidth in GB/s (0 for auto-detect).
            nvlink_gbs: the NVLink bandwidth in GB/s (0 for auto-detect).
            sm_read_gbs: the per-SM HBM read bandwidth in GB/s (0 for the default).
            sm_write_gbs: the per-SM HBM write bandwidth in GB/s (0 for the default).

        Returns:
            num_sms: the recommended SM count (even, at least 4).
        """
        # TODO: support `do_expand` and `allow_multiple_reduction`

        # The `1` in this function means scale-up traffic
        # i.e. the HBM read volume of the dispatch copy epilogue, equals to "the number of tokens" * "num_expected_topk" * "data size per token"
        # NOTES: this is for balanced gate
        # For V3.0's group-limited gate, please do not use this function
        # TODO: support this
        assert num_scaleout_topk == 0  # 组限 gate 尚不支持，必须为 0

        # Get bandwidth
        sm_read_gbs = get_sm_read_gbs() if sm_read_gbs == 0 else sm_read_gbs  # 每 SM 的 HBM 读带宽：0 表示自动探测
        sm_write_gbs = get_sm_write_gbs() if sm_write_gbs == 0 else sm_write_gbs  # 每 SM 的 HBM 写带宽：0 表示自动探测
        if rdma_gbs == 0 and self.num_rdma_ranks > 1:  # 未给 RDMA 带宽且确实跨节点时
            rdma_gbs = get_rdma_gbs()  # 自动探测 RDMA 带宽
        if nvlink_gbs == 0:  # 未给 NVLink 带宽时
            nvlink_gbs = get_nvlink_gbs()  # 自动探测 NVLink 带宽

        # Initial count
        # NOTES: we don't count HBM traffic
        sm_read, sm_write = 0, 0  # 归一化的每 token SM 读/写量累加器
        rdma_traffic, nvlink_traffic = 0, 0  # 归一化的 RDMA/NVLink 通信量累加器

        # 期望 topk 计算函数：按组合概率估计 token 命中的不同分组数（均衡 gate 期望）
        def get_expected_topk(num_groups: int) -> float:
            assert num_experts % num_groups == 0  # 要求专家数能被分组数整除
            return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))  # 1 减去"全落在其他组"的概率，乘组数得期望命中组数

        # Expected top-k scale-out ranks
        num_expected_scaleout_topk = get_expected_topk(self.num_scaleout_ranks) if self.num_scaleout_ranks > 1 else 0  # 期望命中的跨节点（scale-out）分组数，单节点为 0

        # Expected top-k scale-up ranks
        num_expected_topk = get_expected_topk(self.num_ranks)  # 期望命中的 scale-up 分组数

        # Read tokens
        sm_read += 1 / num_expected_topk  # dispatch 拷贝 epilogue 读 token：按每 token 期望分发次数归一

        # NOTES: we don't consider the skip-send-buffer cases (all selections fall in the local)
        if self.num_scaleout_ranks > 1:  # 混合模式（分层 scale-out + scale-up）的流量模型
            # Scaleout warps: write send buffer
            sm_write += 1 / num_expected_topk  # scale-out warp 先把 token 写入发送缓冲

            # Scaleout traffic
            sm_write += (1 / num_expected_topk) * (num_expected_scaleout_topk / self.num_scaleout_ranks)  # 本节点命中的部分走本地旁路（仍需写）
            rdma_traffic += (1 / num_expected_topk) * (num_expected_scaleout_topk * (1 - 1 / self.num_scaleout_ranks))  # 其余部分产生 RDMA 流量

            # Forward warps
            sm_read += num_expected_scaleout_topk / num_expected_topk  # forward warp 读回跨节点到达的 token
            sm_write += 1  # Issue scaleup  # 再向节点内发起 scale-up 分发
            nvlink_traffic += 1 - (1 / self.num_scaleup_ranks)  # NVLink 流量：除去本 rank 旁路的部分
        else:  # 非混合模式：直接 NVLink/RDMA 单层分发
            # Write send buffer
            if self.num_rdma_ranks > 1:  # 跨节点时需要写发送缓冲
                sm_write += 1 / num_expected_topk

            # Issue NVLink
            sm_write += self.num_nvlink_ranks / self.num_ranks  # 按比例向各 NVLink 邻居发起分发

            # NVLink and RDMA traffic
            nvlink_traffic += self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)  # NVLink 流量（除本地旁路）
            rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks  # 节点外部分产生 RDMA 流量

        # Found the bounded one
        if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):  # 混合模式下 RDMA 时延更长则 RDMA 是瓶颈
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs  # 记录瓶颈链路的流量与带宽
        else:  # 否则 NVLink 是瓶颈
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

        # Calculate SM count
        # NOTES: will try to use more SMs if not overlap with compute
        num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count  # 设备的 SM 总数
        num_sms = num_device_sms  # No traffic, e.g., EP=1  # 无通信量（如 EP=1）时直接用全部 SM
        if bounded_traffic > 0:  # 存在瓶颈流量时按带宽模型估算
            num_sms = max(
                # Dispatch
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,  # dispatch 读侧所需 SM 数
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,  # dispatch 写侧所需 SM 数
                # Combine
                bounded_gbs / bounded_traffic * (sm_write - nvlink_traffic) / sm_read_gbs,  # combine 读侧所需 SM 数
                bounded_gbs / bounded_traffic * (sm_read + nvlink_traffic) / sm_write_gbs,  # combine 写侧所需 SM 数
            )  # 取四项的最大值
        num_sms = align(max(4, math.ceil(num_sms * 1.3)), 4)  # 加 30% 余量，至少 4 个并对齐到 4 的倍数
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)  # 与计算重叠时保持精简，否则至少 64 个 SM
        num_sms = min(num_sms, num_device_sms // 2 * 2)  # 上限为设备 SM 数向下取偶，留一半给计算

        # Summary
        if os.environ.get('EP_BUFFER_DEBUG', 0):  # 调试模式下打印估算过程各中间量
            print(f'EP SM approximation: '
                  f'{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, '
                  f'{rdma_gbs=}, {nvlink_gbs=}, '
                  f'{num_expected_scaleout_topk=}, {num_expected_topk=}, '
                  f'{bounded_traffic=}, {bounded_gbs=}, {num_sms=}')
        return num_sms  # 返回估算出的推荐 SM 数

    # 按 SM 数与模式估算最优 RDMA QP 数，结果不超过预分配的 num_allocated_qps
    def get_theoretical_num_qps(self, num_sms: int) -> int:
        """
        Estimate the optimal number of RDMA QPs based on SM count and mode.

        Arguments:
            num_sms: the number of SMs used for the dispatch/combine kernel.

        Returns:
            num_qps: the recommended QP count, capped by `num_allocated_qps`.
        """
        # For direct mode, we encourage less QPs to reduce DB ringing overhead
        num_qps = min(num_sms, 8 + 1)  # 直连模式：QP 越少 doorbell 敲铃开销越小，上限 8+1

        # For hybrid mode, we encourage every channel (and notify) to have an independent QP
        if self.allow_hybrid_mode:  # 混合模式：每个 channel 一条独立 QP，外加一条 notify QP
            num_qps = num_sms * 16 + 1

        return min(num_qps, self.num_allocated_qps)  # 不超过预分配上限

    # dispatch：把 token 按路由（topk_idx）分发到持有对应专家的 rank，支持单机与多机、BF16/FP8
    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 num_experts: Optional[int] = None,
                 num_max_tokens_per_rank: Optional[int] = None,
                 expert_alignment: Optional[int] = None,
                 num_sms: int = 0, num_qps: int = 0,
                 previous_event: Optional[EventHandle] = None,
                 async_with_compute_stream: bool = False,
                 allocate_on_comm_stream: bool = False,
                 handle: Optional[EPHandle] = None,
                 do_handle_copy: bool = False,
                 do_cpu_sync: Optional[bool] = None,
                 do_expand: bool = False,
                 do_zero_padding: bool = False,
                 use_tma_aligned_col_major_sf: bool = False,
                 defer_epilogue: bool = False) \
            -> Union[Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           Optional[torch.Tensor], Optional[torch.Tensor],
                           EPHandle, EventOverlap], EventOverlap]:
        """
        Dispatch tokens to different ranks. Supports both single-node and multi-node settings.
            SM and QP counts are automatically determined if not specified.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
                `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
                the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
                the second is the scale factors.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
                Stored by reference in the handle; must not be modified until all uses of the handle complete.
                Must be `None` if `handle` is provided.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
                Must be `None` if `handle` is provided.
            cumulative_local_expert_recv_stats: `[num_local_experts]` with `torch.int`, a cumulative expert count
                tensor for statistics, useful for online EP load balance monitoring.
            num_experts: the number of all experts. Inferred from `handle` if provided.
            num_max_tokens_per_rank: the maximum number of tokens per rank. Inferred from constructor default
                or `handle` if provided.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of SMs to use (0 for automatic via `get_theoretical_num_sms`).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            handle: an optional cached `EPHandle` from a previous dispatch, if set, the CPU will reuse the layout
                information to save some time. `topk_idx` must be `None` (reused from handle).
                `topk_weights` can be optionally provided (e.g. for backward pass with cached expand).
            do_handle_copy: retained for compatibility; must be `False`.
            do_cpu_sync: whether to synchronize with CPU to get exact received token counts.
                `None` defaults to `True` unless `handle` is provided.
            do_expand: whether to use the expanding layout (one slot per expert per token).
            do_zero_padding: whether to zero out the alignment padding slots in the expanded output.
                Only valid when `do_expand` is True. Ensures alignment gaps between experts are zeroed.
            use_tma_aligned_col_major_sf: whether to use TMA-aligned column-major layout for scale factors.
            defer_epilogue: whether to defer the CPU receive-count wait and copy epilogue until
                `event.current_stream_wait()` is called. This requires `async_with_compute_stream=True`.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`.
                Only returned when `defer_epilogue=False`.
            recv_topk_idx: received expert indices. Only returned when `defer_epilogue=False`.
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
                Only returned when `defer_epilogue=False`.
            handle: the returned communication handle. Only returned when `defer_epilogue=False`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
                With `defer_epilogue=True`, this function returns the `EventOverlap` object directly instead
                of the five-item tuple. Call `event.current_stream_wait()` to run the copy epilogue and obtain
                `(recv_x, recv_topk_idx, recv_topk_weights, handle)`.
        """
        assert not do_handle_copy, '`do_handle_copy` must be False; handle copying is no longer supported'  # 句柄拷贝特性已移除，必须为 False
        check_torch_deterministic()  # 检查 torch 确定性模式与本库确定性实现的兼容性

        # Automatic decide SM and QP count
        num_topk = (handle.topk_idx if topk_idx is None else topk_idx).shape[1]  # 从传入或缓存句柄的 topk_idx 取 topk 数
        num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else align(num_sms, 2)  # 未指定时按带宽模型自动选择，否则对齐到偶数
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps  # 未指定时按 SM 数自动选择 QP 数
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'  # 使用的 QP 数不得超过预分配量

        # Unpack SF
        x, sf = x if isinstance(x, tuple) else (x, None)  # FP8 模式 x 为 (数据, scale factor) 元组，否则 sf 为 None

        # Unpack handles
        # Reuse some values if possible
        if handle is not None:  # 传入缓存句柄：复用上次 dispatch 的布局信息，跳过重算
            assert topk_idx is None  # 缓存模式下必须复用句柄内的 topk_idx
            assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'  # 缓存句柄路径不允许 CPU 同步
            topk_idx = handle.topk_idx  # 从句柄取回路由索引
            num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)  # 参数缺省时用句柄内保存值
            num_experts = value_or(num_experts, handle.num_experts)  # 同上：专家数
            expert_alignment = value_or(expert_alignment, handle.expert_alignment)  # 同上：对齐粒度
            do_cpu_sync = False  # 缓存布局已知，无需 CPU 同步确认接收数

            # Should be aligned with the handle context
            assert (num_experts, expert_alignment, num_max_tokens_per_rank) == \
                   (handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank)  # 传入参数必须与句柄上下文一致
        (cached_num_recv_tokens, cached_num_expanded_tokens,
         cached_num_recv_tokens_per_expert_list,
         cached_psum_num_recv_tokens_per_scaleup_rank, cached_psum_num_recv_tokens_per_expert,
         cached_num_unaligned_recv_tokens_per_expert,
         cached_dst_buffer_slot_idx,
         cached_token_metadata_at_forward,
         cached_recv_src_metadata,
         cached_channel_linked_list) = self._unpack_handle(handle)  # 从句柄拆出全部缓存的布局信息（无句柄时全为 None）

        # Some default values
        num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)  # 兜底到构造时的默认值
        expert_alignment = value_or(expert_alignment, 1)  # 未指定对齐粒度时取 1（不对齐）
        do_cpu_sync = value_or(do_cpu_sync, True)  # 默认做 CPU 同步以获得精确接收 token 数

        # Do dispatch
        result, event, deferred_epilogue = self.runtime.dispatch(x, sf, topk_idx, topk_weights,  # 调用 C++ 运行时发起 dispatch，返回结果/事件/延迟 epilogue
                                                                 cumulative_local_expert_recv_stats,
                                                                 cached_num_recv_tokens,
                                                                 cached_num_expanded_tokens,
                                                                 cached_num_recv_tokens_per_expert_list,
                                                                 cached_psum_num_recv_tokens_per_scaleup_rank,
                                                                 cached_psum_num_recv_tokens_per_expert,
                                                                 cached_num_unaligned_recv_tokens_per_expert,
                                                                 cached_dst_buffer_slot_idx,
                                                                 cached_token_metadata_at_forward,
                                                                 cached_recv_src_metadata,
                                                                 cached_channel_linked_list,
                                                                 num_max_tokens_per_rank,
                                                                 num_experts, expert_alignment,
                                                                 num_sms, num_qps,
                                                                 previous_event,
                                                                 async_with_compute_stream, allocate_on_comm_stream,
                                                                 do_cpu_sync, do_expand,
                                                                 do_zero_padding,
                                                                 use_tma_aligned_col_major_sf,
                                                                 defer_epilogue)
        event_overlap = EventOverlap(event)  # 用通信事件创建可重叠句柄

        # dispatch 收尾：从运行时结果构造句柄、做确定性排序、组装返回值
        def finalize_dispatch(dispatch_result: tuple, deterministic_by_hook: bool):
            (recv_x, recv_sf,
             recv_topk_idx, recv_topk_weights,
             num_recv_tokens, num_expanded_tokens,
             num_recv_tokens_per_expert_list,
             psum_num_recv_tokens_per_scaleup_rank,
             psum_num_recv_tokens_per_expert,
             num_unaligned_recv_tokens_per_expert,
             recv_src_metadata,
             dst_buffer_slot_idx,
             token_metadata_at_forward,
             channel_linked_list) = dispatch_result  # 解包运行时返回的接收数据与全部布局元数据

            # Create handle if not cached
            nonlocal handle  # 引用外层的 handle 变量以便复用/更新
            is_cached_dispatch = handle is not None  # 记录本次是否为缓存 dispatch
            handle = EPHandle(do_expand,  # 非缓存 dispatch 时构造新句柄，保存全部元数据
                              num_experts, expert_alignment,  # 专家数与对齐粒度
                              num_max_tokens_per_rank,  # 每 rank 最大 token 数
                              num_sms,  # 本次使用的 SM 数
                              topk_idx,  # 路由索引（句柄存续期间不允许修改）
                              num_recv_tokens, num_expanded_tokens,  # 接收 token 数（未 CPU 同步时可能不准确）
                              num_recv_tokens_per_expert_list,  # CPU 侧每专家计数列表
                              psum_num_recv_tokens_per_scaleup_rank,  # scaleup 维度接收前缀和
                              psum_num_recv_tokens_per_expert,  # 专家维度接收前缀和
                              num_unaligned_recv_tokens_per_expert,  # 未对齐每专家计数
                              recv_src_metadata,  # 来源元数据
                              dst_buffer_slot_idx,  # 目标缓冲槽位索引
                              token_metadata_at_forward,  # 前向转发元数据（混合模式）
                              channel_linked_list) if handle is None else handle  # 缓存 dispatch 时沿用原句柄

            # Do deterministic
            if self.deterministic:  # 启用确定性时对接收结果按来源排序
                deterministic_epilogue = functools.partial(  # 用偏函数绑定确定性排序的全部参数
                    handle.deterministic_sort,
                    do_cpu_sync, is_cached_dispatch,
                    recv_x, recv_sf, recv_topk_idx, recv_topk_weights, channel_linked_list
                )
                if deterministic_by_hook:  # 与计算异步时把排序挂到事件等待后的钩子上
                    event_overlap.register_hook_after_wait(deterministic_epilogue)
                else:  # 同步路径立即执行排序
                    deterministic_epilogue()

            # Return values
            recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x  # FP8 模式把数据与 scale factor 打包成元组
            return recv_x, recv_topk_idx, recv_topk_weights, handle  # 返回接收数据、专家索引、权重与句柄

        # Just launch the dispatch
        if deferred_epilogue is not None:  # defer_epilogue 模式：收尾延迟到 event.current_stream_wait() 时执行
            event_overlap.register_hook_after_wait(
                lambda: finalize_dispatch(deferred_epilogue(), False))  # 等待后先执行运行时 epilogue 再做 finalize
            return event_overlap  # 直接返回事件句柄，由调用方触发

        # Do epilogue ASAP
        assert result is not None  # 非 defer 模式必须已有结果
        return *finalize_dispatch(result, async_with_compute_stream), event_overlap  # 立即执行收尾并连同事件一起返回五元组

    # 静态工具：把 bias 归一化为 (bias_0, bias_1) 二元组，支持单张量、二元组或 None
    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bias_0, bias_1 = None, None  # 默认两个偏置均为 None
        if isinstance(bias, torch.Tensor):  # 单张量：只作为第一个偏置
            bias_0 = bias
        elif isinstance(bias, tuple):  # 元组：必须恰有两个偏置
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1  # 返回拆分后的偏置

    # combine：把分发到各专家 rank 的 token 归约回其原始 rank（dispatch 的逆过程）
    def combine(self,
                x: torch.Tensor,
                handle: EPHandle,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                num_sms: int = 0, num_qps: int = 0,
                previous_event: EventHandle = None,
                async_with_compute_stream: bool = False,
                allocate_on_comm_stream: bool = False,
                defer_epilogue: bool = False) \
            -> Union[Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap], EventOverlap]:
        """
        Combine (reduce) tokens from different ranks back to their original ranks.
        Supports both single-node and multi-node settings.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the `dispatch` function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float` for non-expand mode, or
                `[num_tokens]` 1D for expand mode. The tokens' top-k weights for reducing to
                its original ranks.
            bias: 0, 1 or 2 `[num_combined_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            num_sms: the number of SMs to use (0 to reuse the SM count from the dispatch handle).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            defer_epilogue: whether to defer the reduce epilogue until `event.current_stream_wait()` is called.
                This requires `async_with_compute_stream=True`.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
                Only returned when `defer_epilogue=False`.
            combined_topk_weights: the reduced top-k weights, with shape `[num_combined_tokens, num_topk]`
                and type `torch.float`. Only returned when `defer_epilogue=False`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
                With `defer_epilogue=True`, this function returns the `EventOverlap` object directly instead
                of the three-item tuple. Call `event.current_stream_wait()` to run the reduce epilogue and
                obtain `(combined_x, combined_topk_weights)`.
        """
        check_torch_deterministic()  # 检查 torch 确定性模式兼容性

        # Automatic decide SM and QP count
        num_sms = handle.num_sms if num_sms == 0 else align(num_sms, 2)  # 未指定时复用 dispatch 句柄中的 SM 数，否则对齐到偶数
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps  # 未指定时按 SM 数自动选择 QP 数
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'  # QP 数不得超过预分配量

        bias_0, bias_1 = EPBuffer._unpack_bias(bias)  # 拆分 0/1/2 个偏置张量
        result, event, deferred_epilogue = self.runtime.combine(x, topk_weights,  # 调用 C++ 运行时发起 combine，返回结果/事件/延迟 epilogue
                                                                bias_0, bias_1,  # 两个可选的输出偏置
                                                                handle.recv_src_metadata,  # 来源元数据：据此把结果路由回原 rank
                                                                handle.topk_idx,  # 路由索引：确定每项归约的目标槽位
                                                                handle.psum_num_recv_tokens_per_scaleup_rank,  # scaleup 维度接收前缀和
                                                                handle.token_metadata_at_forward,  # 前向转发元数据（混合模式）
                                                                handle.channel_linked_list,  # channel 链表（混合模式）
                                                                handle.num_experts,  # 专家总数
                                                                handle.num_max_tokens_per_rank,  # 每 rank 最大 token 数
                                                                num_sms, num_qps,  # SM 数与 QP 数
                                                                previous_event,  # 前置等待事件
                                                                async_with_compute_stream,  # 是否与计算流异步
                                                                allocate_on_comm_stream,  # 分配是否归属通信流
                                                                handle.do_expand,  # 是否为展开布局
                                                                defer_epilogue)  # 是否延迟归约 epilogue
        event_overlap = EventOverlap(event)  # 用通信事件创建可重叠句柄
        if deferred_epilogue is not None:  # defer_epilogue 模式：归约结果在等待时才生成
            event_overlap.register_hook_after_wait(deferred_epilogue)  # 把运行时 epilogue 挂到等待后钩子
            return event_overlap  # 直接返回事件句柄，由调用方触发

        assert result is not None  # 非 defer 模式必须已有结果
        combined_x, combined_topk_weights = result  # 解包归约后的 token 与权重
        return combined_x, combined_topk_weights, event_overlap  # 返回归约结果与事件句柄

    # 按 NVLink 带宽估算 LB（负载均衡权重预取）所需 SM 数，结果缓存并向上取整到偶数
    @weak_lru(maxsize=None)
    def lb_get_theoretical_num_sms(self) -> int:
        """Estimate the LB SM count from bandwidths, rounded up to a multiple of 2."""
        # Budget both reading expert weights from HBM and writing them onto NVLink.
        # Use the bandwidth budget independently of the redundancy mapping: ring traffic
        # or rank skew can hit a link bottleneck before using all of this SM capacity.
        nvlink_gbs = get_nvlink_gbs()  # 自动探测 NVLink 带宽
        return align(math.ceil(max(nvlink_gbs / get_sm_read_gbs(), nvlink_gbs / get_sm_write_gbs())), 2)  # 按每 SM 读/写速率分别估算取较大者，对齐到偶数

    # LB 权重预取：把本 rank 的专家权重推送/广播到对端要求的冗余专家槽位（LSA 域内集合操作）
    def lb_prefetch_weights(self,
                            redundant_expert_weights: Sequence[torch.Tensor] | torch.Tensor,
                            expert_weights: Sequence[torch.Tensor] | torch.Tensor,
                            redundancy_mapping: torch.Tensor,
                            num_sms: int = 0,
                            previous_event: Optional[EventHandle] = None) -> EventOverlap:
        """Push this rank's expert weights into the redundant expert slots requested by peers.

        Collective: every rank in the LSA domain must call it, and the fused barriers make the redundant weights
        visible to their users by the time the returned event is waited on.
        Tensor sequences must be nonempty and contain no `None` entries.

        Arguments:
            redundant_expert_weights: each tensor is a contiguous `[num_redundant_experts, *shape]`
                destination and must reside in this buffer's LB region.
            expert_weights: each tensor is a contiguous `[num_local_experts, *shape]` source with at least
                one dimension and a positive `num_local_experts`. Bytes per expert must match its
                destination and be a multiple of `deep_ep.get_num_tma_alignment()`; the trailing shapes may differ.
            redundancy_mapping: `[num_nvlink_ranks, num_redundant_experts]` int32, identical on every rank in the LSA domain;
                entry `[r, c]` is the expert id assigned to redundant slot `c` on LSA rank `r`, or -1 if empty.
                Expert ids are local to the LSA domain, in [0, num_nvlink_ranks * num_local_experts).
            num_sms: the number of SMs to use; 0 uses `lb_get_theoretical_num_sms()`.
            previous_event: the event to wait for before communication; defaults to waiting for the current stream
        """
        redundant_expert_weights = ([redundant_expert_weights] if isinstance(redundant_expert_weights, torch.Tensor)
                                    else list(redundant_expert_weights))  # 归一化冗余槽位目标张量为列表
        expert_weights = [expert_weights] if isinstance(expert_weights, torch.Tensor) else list(expert_weights)  # 归一化本地专家权重源张量为列表

        num_sms = self.lb_get_theoretical_num_sms() if num_sms == 0 else align(num_sms, 2)  # 未指定 SM 数时按带宽模型自动选择，否则对齐到偶数
        return EventOverlap(self.runtime.lb_prefetch_weights(  # 调用 C++ 运行时执行预取，包装为可重叠事件句柄
            redundant_expert_weights, expert_weights, redundancy_mapping, num_sms, previous_event))  # 传入目标/源张量、冗余映射、SM 数与前置事件

    # LB 梯度归约：lb_prefetch_weights 的镜像操作——按同一冗余映射把对端的冗余专家梯度累加回本 rank 的 expert_grads（集合操作，累加而非覆盖）
    def lb_reduce_grads(self,
                        redundant_expert_grads: torch.Tensor,
                        expert_grads: torch.Tensor,
                        redundancy_mapping: torch.Tensor,
                        num_sms: int = 0,
                        previous_event: Optional[EventHandle] = None) -> EventOverlap:
        """Accumulate redundant expert gradients from peers into this rank's `expert_grads`.

        The mirror of `lb_prefetch_weights`: use the same redundancy mapping to return gradients.
        Collective, and it adds rather than overwrites, so seed `expert_grads` before calling.

        Arguments:
            redundant_expert_grads: `[num_redundant_experts, hidden]` fp32 redundant gradients that must reside in this buffer's LB region.
            expert_grads: `[num_local_experts, hidden]` fp32 destination, accumulated into.
            redundancy_mapping: see `lb_prefetch_weights`.
            num_sms: the number of SMs to use; 0 uses `lb_get_theoretical_num_sms()`.
            previous_event: the event to wait for before communication; defaults to waiting for the current stream
        """
        num_sms = self.lb_get_theoretical_num_sms() if num_sms == 0 else align(num_sms, 2)  # 未指定 SM 数时按带宽模型自动选择，否则对齐到偶数
        return EventOverlap(self.runtime.lb_reduce_grads(  # 调用 C++ 运行时执行梯度归约，包装为可重叠事件句柄
            redundant_expert_grads, expert_grads, redundancy_mapping, num_sms, previous_event))  # 传入冗余梯度、目标梯度、冗余映射、SM 数与前置事件
