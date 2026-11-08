import inspect  # 用于解析集合通信函数签名，将 kwargs 绑定到形参
import math  # 数学工具：prod 计算元素总数、ceil 向上取整
import torch  # PyTorch：张量、批量拷贝 _foreach_copy_
import torch.distributed as dist  # 分布式进程组类型
from functools import partial  # 偏函数：将集合通信名预绑定到统一的分发函数
from typing import Any, Optional, Sequence, Tuple, Union  # 类型注解

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # C++ 扩展：BucketBuffer 运行时与对齐参数

from .allocator import BufferAllocator  # 分配计划器：在 storage 上落实多张量的布局
from .base import BufferBase  # 缓冲区抽象基类，统一生命周期管理
from .. import comm  # 通信工具模块
from ..utils.envs import get_nvlink_gbs, get_rdma_gbs, get_sm_read_gbs, get_sm_write_gbs  # 读取 NVLink/RDMA 带宽与 SM 读写速率（GB/s）
from ..utils.event import EventOverlap  # 事件句柄：支持 wait() 与计算/通信重叠
from ..utils.math import align  # 向上对齐工具函数


# 集合通信"会话"：在 with 块内，允许 all_gather/reduce_scatter/all_reduce
# 直接接受未注册进缓冲区的普通张量（自动暂存到对称存储中再通信）
class BucketSession:
    """A session in which collective methods can accept unregistered tensors"""

    # 构造函数：记录宿主缓冲区、对齐粒度等会话状态
    def __init__(self, buffer: 'BucketBuffer'):
        self.buffer = buffer  # 宿主 BucketBuffer
        self.alignment = _C.get_num_rdma_alignment()  # 从 C++ 层获取 RDMA 要求的分配对齐字节数
        self.num_allocated_bytes = 0  # 本会话内已从 storage 暂存区分配的字节数
        self.context_idx: Optional[int] = None  # 本会话使用的通信组上下文索引（要求整个会话只用一个组）
        self.collective_methods = ('all_gather', 'reduce_scatter', 'all_reduce')  # 会话内将被临时替换的集合通信方法名

    # 进入 with 块：清空会话状态，并把三个集合通信方法替换为支持未注册张量的版本
    def __enter__(self) -> 'BucketSession':
        # NOTES: Users must ensure themselves all the allocated bytes in the buffer can be
        # safely reused before entering this session.
        assert not any(name in self.buffer.__dict__ for name in self.collective_methods), \
            'Nested sessions are not supported'  # 缓冲区实例上仍挂着会话方法说明上次会话未退出，不支持嵌套
        self.num_allocated_bytes = 0  # 重置本会话的暂存区分配游标
        self.context_idx = None  # 重置通信组上下文索引
        for name in self.collective_methods:  # 逐个替换集合通信方法
            method = partial(self._collective_func, name)  # 用偏函数将方法名预绑定到统一分发函数
            setattr(self.buffer, name, method)  # 在缓冲区实例上动态设置，覆盖类方法
        return self  # 返回会话自身，支持 as 语法

    # 退出 with 块：删除实例上临时挂载的集合通信方法，恢复原始类方法
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for name in self.collective_methods:  # 逐个删除实例属性
            delattr(self.buffer, name)  # 使 all_gather 等恢复为 BucketBuffer 类上的原始方法

    # 调用缓冲区类上的原始集合通信方法（绕过实例上被会话替换的版本）
    def _call_original(self, name: str, *args, **kwargs):
        return getattr(type(self.buffer), name)(self.buffer, *args, **kwargs)  # 从类上取方法并以 buffer 为 self 调用

    # 将调用参数按原始集合通信方法的签名绑定，解析为规范的 kwargs 字典
    def _parse_args(self, name: str, *args, **kwargs) -> dict:
        """Bind call kwargs to the original collective's signature."""
        signature = inspect.signature(getattr(type(self.buffer), name))  # 获取原始方法的签名
        kwargs = signature.bind(self.buffer, *args, **kwargs).arguments  # 把位置/关键字参数绑定到形参名
        kwargs.pop('self')  # 去掉绑定产生的 self
        return kwargs  # 返回形参名到实参的映射

    # 从 storage 的会话暂存区中按对齐切出一块，返回指定 dtype 的视图
    def _allocate(self, num_bytes: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        assert num_bytes > 0  # 分配大小必须为正
        allocated_bytes = align(num_bytes, self.alignment)  # 实际占用向上对齐到 RDMA 对齐粒度
        offset = self.num_allocated_bytes  # 本会话内的分配偏移（游标）
        if offset + allocated_bytes > self.buffer.num_bytes:  # 超出缓冲区总容量则报错
            raise RuntimeError(f'BucketBuffer session out of storage, which has only {self.buffer.num_bytes}')
        self.num_allocated_bytes += allocated_bytes  # 推进分配游标
        return self.buffer.storage.narrow(0, offset, num_bytes).view(dtype)  # 从对称存储切出 num_bytes 字节并转为目标 dtype

    # 检查本会话内所有集合通信是否使用同一个通信组
    def _check_group(self, group: Optional[dist.ProcessGroup]) -> None:
        # Check only one group is used in one session.
        context_idx = self.buffer.get_context_idx(group)  # 解析该组对应的上下文索引
        assert self.context_idx in (None, context_idx), \
            'All collectives in a session must use the same group'  # 会话一旦定下组就不允许更换
        self.context_idx = context_idx  # 记录/更新上下文索引

    # 包装事件句柄：等待通信完成后，把暂存结果拷贝到用户的外部 dst 张量，并返回结果
    @staticmethod
    def _wrap_handle(handle: EventOverlap,
                     dsts: Sequence[torch.Tensor],
                     result: Any) -> EventOverlap:
        """Copy staged results to their external dsts and return ``result``."""
        hook = handle.hook_after_wait  # 取出句柄上已有的等待后钩子（可能为 None）
        handle.hook_after_wait = None  # 先摘除旧钩子，避免重复执行

        def hook_after_wait():  # 新的等待后钩子：执行原逻辑再做外部拷贝
            dst_buffers = hook()  # 执行原钩子（原 all_gather 的 epilogue），得到缓冲区内的结果视图
            copy_pairs = [(dst, dst_buffer) for dst_buffer, dst in zip(dst_buffers, dsts, strict=True)
                          if dst.data_ptr() != dst_buffer.data_ptr()]  # 找出结果与目标不是同一内存（需真正拷贝）的配对
            if copy_pairs:  # 存在需要拷贝的配对
                torch._foreach_copy_([dst.view(-1) for dst, _ in copy_pairs],  # 批量将缓冲区结果拷入用户 dst
                                     [dst_buffer.view(dst.dtype) for dst, dst_buffer in copy_pairs])
            return result  # 返回用户可见的结果（dsts 本身或打包形式）

        handle.register_hook_after_wait(hook_after_wait)  # 注册新的等待后钩子
        return handle  # 返回被包装的句柄

    # 在已注册缓冲区内按形状分配张量（供会话内使用）
    def allocate(self,
                         shapes: Union[Sequence[int], Sequence[Sequence[int]]],
                         dtype: torch.dtype = torch.float32,
                         group: Optional[dist.ProcessGroup] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Allocate tensors in registered buffer."""
        group = self.buffer.groups[self.buffer.get_context_idx(group)]  # 将 None 组解析为默认组
        self._check_group(group)  # 校验会话内组的一致性

        is_batched = isinstance(shapes[0], (tuple, torch.Size, list))  # 判断是否为批量形状（形状的列表）
        shapes = shapes if is_batched else (shapes, )  # 统一成批量形式便于遍历
        tensors = [self._allocate(math.prod(shape) * dtype.itemsize, dtype).view(shape) for shape in shapes]  # 逐个在暂存区分配并重塑为目标形状
        return tuple(tensors) if is_batched else tensors[0]  # 批量输入返回元组，否则返回单个张量

    # 会话集合通信的统一入口：解析参数、为未注册张量准备暂存缓冲、拷入数据、发起原始集合通信并包装结果
    def _collective_func(self, name, *args, **kwargs) -> EventOverlap:
        """Collective methods accepting unregistered sources."""

        # Parse args
        kwargs = self._parse_args(name, *args, **kwargs)  # 按原方法签名解析参数
        group = kwargs.get('group')  # 取出通信组参数
        group = self.buffer.groups[self.buffer.get_context_idx(group)]  # 解析为实际使用的进程组
        self._check_group(group)  # 校验会话内组一致性
        rank_idx, num_ranks = group.rank(), group.size()  # 本 rank 编号与组内 rank 总数

        # Extract srcs and dsts
        num_shards = {  # 各集合通信的 (src 分片数, dst 分片数) 语义
            'all_gather': (num_ranks, 1),  # all_gather：src 是本 rank 分片（共 num_ranks 份），dst 是聚合结果（1 份）
            'reduce_scatter': (1, num_ranks),  # reduce_scatter：src 是完整张量（1 份），dst 是归约后的本 rank 分片（共 num_ranks 份）
            'all_reduce': (1, 1),  # all_reduce：原位全归约，src/dst 各 1 份
        }[name]
        srcs = kwargs.pop('srcs')  # 源张量列表
        dsts = kwargs.pop('dsts', None)  # 可选的目标张量列表
        dst_exists = dsts is not None  # 是否显式传入了 dst
        if not dst_exists:  # 未传 dst 时
            dsts = srcs  # 视为原位操作（dst 即 src）
        is_packed, srcs, dsts = BucketBuffer._pack_tensors(srcs, dsts)  # 归一化：单张量打包成列表并记录打包标志
        num_rdma_ranks = self.buffer.contexts[self.context_idx].get_physical_domain_size()[0]  # 物理域中需走 RDMA 的 rank 数
        accept_unregistered_src = name == 'all_gather' and num_rdma_ranks == 1  # 仅纯 NVLink 的 all_gather 支持直接读未注册的源张量

        # Construct src & dst buffers
        storage_begin = self.buffer.storage.data_ptr()  # 对称存储的起始地址
        storage_end = storage_begin + self.buffer.num_bytes  # 对称存储的结束地址（用于判断张量是否已注册）
        src_buffers = []  # 每个源张量对应的注册缓冲视图
        dst_buffers = []  # 每个目标张量对应的注册缓冲视图
        copy_tasks = []  # 需要先把用户 src 拷入暂存缓冲的 (缓冲, 源) 任务
        for src, dst in zip(srcs, dsts, strict=True):  # 逐对处理源与目标
            # Check configs
            if dst_exists:  # 显式传入 dst 时校验其与 src 的匹配性
                assert src.dtype == dst.dtype  # 数据类型必须一致
                assert dst.is_contiguous(), f'{name} output must be contiguous'  # 输出必须连续
                assert dst.nbytes * num_shards[1] == src.nbytes * num_shards[0]  # 按 (分片数, 字节数) 折算后 src 与 dst 总量必须相等

            # Find or allocate the registered buffer
            is_src_registered = storage_begin <= src.data_ptr() < storage_end  # src 是否已位于本缓冲区对称存储内
            is_dst_registered = storage_begin <= dst.data_ptr() < storage_end if dst_exists else False  # dst 是否已注册（未传 dst 时视为否）
            num_buffer_bytes = src.nbytes * num_shards[0]  # 该集合通信在此张量上需要的完整缓冲字节数
            if is_src_registered:  # src 已注册：直接在存储上定位其所属的完整缓冲
                assert src.is_contiguous(), f'{name} registered source must be contiguous'  # 注册源必须连续
                offset = src.data_ptr() - storage_begin - (rank_idx % num_shards[0]) * src.nbytes  # 由本 rank 分片地址反推完整缓冲的起始偏移
                buffer = self.buffer.storage.narrow(0, offset, num_buffer_bytes).view(src.dtype)  # 切出完整缓冲并按 src 类型解释
            elif is_dst_registered:  # dst 已注册：由 dst 分片地址反推完整缓冲
                offset = dst.data_ptr() - storage_begin - (rank_idx % num_shards[1]) * dst.nbytes  # 反推缓冲起始偏移
                buffer = self.buffer.storage.narrow(0, offset, num_buffer_bytes).view(dst.dtype)  # 切出完整缓冲并按 dst 类型解释
            else:  # src 与 dst 都未注册：在会话暂存区新分配一块
                buffer = self._allocate(num_buffer_bytes, src.dtype)

            # Calculate src and dst buffer
            src_buffer = buffer.view(num_shards[0], -1)[rank_idx % num_shards[0]].view(src.shape)  # 完整缓冲按 src 分片数切分，取本 rank 分片并重塑为 src 形状
            dst_buffer = buffer.view(num_shards[1], -1)[rank_idx % num_shards[1]]  # 完整缓冲按 dst 分片数切分，取本 rank 分片
            if dst_exists:  # 显式传入 dst 时
                dst_buffer = dst_buffer.view(dst.shape)  # 将本 rank 的 dst 分片重塑为 dst 形状

            # Construct copy tasks
            if not is_src_registered and not accept_unregistered_src:  # 源未注册且底层不支持直接读外部内存
                copy_tasks.append((src_buffer, src))  # 记录拷贝任务：先把用户 src 拷进注册缓冲

            src_buffers.append(src_buffer)  # 记录本张量的 src 缓冲视图
            dst_buffers.append(dst_buffer)  # 记录本张量的 dst 缓冲视图

        # In-place calls output into the registered buffers
        if not dst_exists:  # 原位调用（未显式传 dst）时
            dsts = dst_buffers  # 结果输出到注册缓冲，之后经钩子返回

        # Copy srcs
        if copy_tasks:  # 存在需要拷入的数据
            torch._foreach_copy_(*zip(*copy_tasks))  # 批量执行拷贝：把各用户 src 拷入对应暂存缓冲

        # Launch
        if accept_unregistered_src:  # 纯 NVLink 的 all_gather：可直接传用户 src
            handle = self._call_original(name, srcs, dsts=dst_buffers, **kwargs)  # 调用原始 all_gather，输出到注册的 dst 缓冲
        else:  # 其余情况：传注册缓冲内的 src 视图
            handle = self._call_original(name, src_buffers, **kwargs)  # 调用原始集合通信方法
        return self._wrap_handle(handle, dsts, dsts[0] if is_packed else dsts)  # 包装句柄：等待后在钩子中把结果交给外部 dst，并返回用户可见结果


# Bucket 缓冲区：持有一块连续的对称显存，可注册到多个通信组，
# 为 all_gather / reduce_scatter / all_reduce 提供高性能（含混合 NVLink+RDMA 流水线）实现
class BucketBuffer(BufferBase):
    """Own contiguous symmetric storage registered with one or more communication groups."""

    # 工具函数：把单张量/列表/None 统一归一化为列表，并记录是否为单张量打包调用
    @staticmethod
    def _pack_tensors(*tensors):
        # `None` is an optional list, packed as `[]`
        is_packed = isinstance(tensors[0], torch.Tensor)  # 第一个参数是单张量则视为"打包"调用
        tensors = tuple([] if item is None else [item] if isinstance(item, torch.Tensor) else list(item)
                        for item in tensors)  # None 归一为空列表，单张量包成单元素列表，其余转列表
        assert all(len(item) in (0, len(tensors[0])) for item in tensors)  # 各列表长度必须一致（空列表仅允许是可选参数）
        return is_packed, *tensors  # 返回打包标志与归一化后的各列表

    # 构造函数：创建 C++ 运行时，分配连续对称存储并注册到各通信组
    def __init__(self,
                 group: Union[dist.ProcessGroup, Sequence[dist.ProcessGroup]],  # 单个进程组或进程组列表
                 allocation_plan_or_num_bytes: Union[BufferAllocator, int],  # 分配计划（BufferAllocator）或总字节数
                 sl_idx: Optional[int] = None,  # RDMA 流量的服务等级索引，None 为默认
                 num_gpu_timeout_secs: int = 100,  # GPU 侧等待超时（秒）
                 explicitly_destroy: bool = False):  # 是否要求显式调用 destroy()
        # Provide plan or bytes
        allocation_plan = allocation_plan_or_num_bytes if isinstance(allocation_plan_or_num_bytes, BufferAllocator) else None  # 传入的是分配计划则取出，否则为 None
        num_bytes = allocation_plan_or_num_bytes if allocation_plan is None else allocation_plan.num_bytes  # 无计划时直接使用字节数，有计划时取计划累计大小
        if allocation_plan is not None:  # 使用分配计划时
            assert not allocation_plan.materialized  # 计划必须尚未落实（本构造函数负责落实）
        assert isinstance(num_bytes, int) and num_bytes > 0  # 总字节数必须为正整数

        # Configs
        self.num_bytes = num_bytes  # 保存缓冲区总字节数
        self.groups = [group] if isinstance(group, dist.ProcessGroup) else list(group)  # 统一为进程组列表
        super().__init__(explicitly_destroy)  # 初始化基类：记录显式销毁标志
        self.nccl_comm_handles = [comm.get_nccl_comm_handle(item) for item in self.groups]  # 为每个组获取 NCCL 通信句柄

        # Bandwidths for planning the hybrid pipelines
        nvl_gbs, rdma_gbs = get_nvlink_gbs(), get_rdma_gbs()  # 读取 NVLink 与 RDMA 带宽估计（GB/s）
        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'  # 带宽探测失败则报错

        # Create CPP runtime
        self.runtime = _C.BucketBuffer(  # 创建 C++ 运行时：分配对称存储、注册 RDMA 窗口
            [item.rank() for item in self.groups],  # 各组内本 rank 编号
            [item.size() for item in self.groups],  # 各组的 rank 总数
            [item.get() for item in self.nccl_comm_handles],  # 各组的 NCCL 通信器指针
            num_bytes,  # 对称存储总字节数
            sl_idx,  # RDMA 服务等级
            num_gpu_timeout_secs,  # GPU 侧超时
            nvl_gbs, rdma_gbs,  # NVLink/RDMA 带宽，用于规划混合流水线
            explicitly_destroy)  # 显式销毁标志
        self.contexts = self.runtime.contexts  # 各组的通信上下文列表
        self.storage = self.runtime.storage  # 底层分配的连续对称存储

        # Materialize
        if allocation_plan is not None:  # 传入了分配计划时
            allocation_plan.materialize(self.storage)  # 在该对称存储上落实计划，使各占位张量指向真实内存

    # 估算指定集合通信所需的理论 SM 数（按带宽模型），供调度 num_sms 自动选择
    def get_theoretical_num_sms(self, op: str, group: Optional[dist.ProcessGroup] = None) -> int:
        context_idx = self.get_context_idx(group)  # 解析组对应的上下文索引
        context = self.contexts[context_idx]  # 取出该上下文
        num_rdma_ranks, num_nvl_ranks = context.get_physical_domain_size()  # 物理域：跨 RDMA 的 rank 数与节点内 NVLink rank 数

        if op == 'reduce_scatter':  # reduce-scatter 的 SM 估算
            # NVLink
            if num_rdma_ranks == 1 and num_nvl_ranks > 1:  # 纯节点内 NVLink 场景
                dst_gbs = get_nvlink_gbs() / num_nvl_ranks  # 每 rank 可用的 NVLink 带宽
                num_sms = max(dst_gbs / get_sm_read_gbs(), dst_gbs / get_sm_write_gbs())  # 按每 SM 读/写速率估算所需 SM 数
                return align(max(6, math.ceil(num_sms)), 2)  # 至少 6 个 SM 并对齐到偶数

            # Hybrid
            if num_nvl_ranks > 1 and num_rdma_ranks > 1:  # NVLink+RDMA 混合场景
                dst_gbs = min(get_nvlink_gbs() / num_nvl_ranks,
                              get_rdma_gbs() * num_rdma_ranks / (num_rdma_ranks - 1))  # 目标带宽取 NVLink 与 RDMA（含 rank-1 分摊）的较小者
                # Multimem and local reduction each require one read and one write.
                num_sms = max(2 * dst_gbs / get_sm_read_gbs(),
                              2 * dst_gbs / get_sm_write_gbs())  # Multimem 与本地归约各需一读一写，按两倍带宽估算
                return align(max(8, math.ceil(num_sms * 1.2)), 2)  # 至少 8 个 SM，加 20% 余量并对齐到偶数

            # RDMA
            if num_nvl_ranks == 1 and num_rdma_ranks > 1:  # 纯 RDMA 场景
                algorithm_gbs = get_rdma_gbs() * num_rdma_ranks / (num_rdma_ranks - 1)  # reduce-scatter 的算法带宽（ring/scatter 折算）
                num_sms = max(algorithm_gbs / get_sm_read_gbs(),
                              algorithm_gbs / num_rdma_ranks / get_sm_write_gbs())  # 读带宽全量、写带宽按 rank 数分摊，取较大者
                return align(max(6, math.ceil(num_sms)), 2)  # 至少 6 个 SM 并对齐到偶数

        if op == 'all_reduce':  # all-reduce 的 SM 估算
            # Hybrid
            if num_rdma_ranks > 1 and num_nvl_ranks > 1:  # 混合场景
                dst_gbs = min(get_nvlink_gbs() / num_nvl_ranks,
                              get_rdma_gbs() * num_rdma_ranks / (2 * (num_rdma_ranks - 1)))  # 目标带宽：NVLink 与 RDMA 归约折算的较小者
                # Multimem reduction, RDMA reduction and broadcast each require one read and one write.
                num_sms = max(3 * dst_gbs / get_sm_read_gbs(),
                              3 * dst_gbs / get_sm_write_gbs())  # Multimem 归约、RDMA 归约、广播各需一读一写，按三倍带宽估算
                return align(max(6, math.ceil(num_sms)), 2)  # 至少 6 个 SM 并对齐到偶数

            # NVLink
            if num_rdma_ranks == 1 and num_nvl_ranks > 1:  # 纯 NVLink 场景
                dst_gbs = get_nvlink_gbs() / num_nvl_ranks  # 每 rank 可用带宽
                num_sms = max(dst_gbs / get_sm_read_gbs(), dst_gbs / get_sm_write_gbs())  # 按读/写速率估算
                return align(max(4, math.ceil(num_sms)), 2)  # 至少 4 个 SM 并对齐到偶数

            # RDMA
            if num_nvl_ranks == 1 and num_rdma_ranks > 1:  # 纯 RDMA 场景
                rdma_gbs = get_rdma_gbs()  # RDMA 带宽
                # Local reduction requires one read and one write
                num_sms = max(rdma_gbs / 2 / get_sm_read_gbs(),
                              rdma_gbs / 2 / get_sm_write_gbs())  # 本地归约一读一写，按半带宽分别估算取较大者
                return align(max(6, math.ceil(num_sms)), 2)  # 至少 6 个 SM 并对齐到偶数

            raise NotImplementedError  # 未覆盖的单 rank 场景

        if op == 'all_gather':  # all-gather 由拷贝引擎驱动
            return 0  # 返回 0 表示不占用计算 SM

        raise NotImplementedError  # 未知操作类型

    # 解析通信组对应的上下文索引：单组时可省略 group，多组时必须显式指定
    def get_context_idx(self, group: Optional[dist.ProcessGroup]):
        assert len(self.groups) == 1 or group is not None, '`group` is required when the buffer has multiple groups'  # 多组场景必须指明组
        return 0 if group is None else self.groups.index(group)  # None 且仅一组时返回 0，否则查找该组的索引

    # 打开一个会话：with 块内的集合通信方法可接受未注册张量（自动暂存与回拷）
    def session(self) -> BucketSession:
        """Open a session in which collectives accept unregistered tensors.
        """
        return BucketSession(self)  # 返回绑定本缓冲区的会话对象

    # 在指定通信组上执行 barrier 同步
    def barrier(self,
                wait_comm_stream: bool = True,
                with_cpu_sync: bool = False,
                sequential: bool = True,
                group: Optional[dist.ProcessGroup] = None) -> None:
        """Run a barrier on the selected communication group."""
        context_idx = self.get_context_idx(group)  # 解析组对应的上下文索引
        comm.barrier(self, wait_comm_stream, with_cpu_sync, sequential, context_idx)  # 委托通用 barrier 实现

    # 批量 reduce-scatter：srcs 归约后按 rank 切分，本 rank 得到自己的分片
    def reduce_scatter(self,
                       srcs: Union[torch.Tensor, Sequence[torch.Tensor]],
                       comm_precision: str = 'fp32',
                       scale: float = 1.0,
                       group: Optional[dist.ProcessGroup] = None,
                       num_sms: Optional[int] = None) -> EventOverlap:
        """Launch batched reduce-scatter; ``None`` selects the SM count automatically."""
        is_packed, srcs = self._pack_tensors(srcs)  # 归一化 srcs 为列表并记录打包标志

        # Launch
        context_idx = self.get_context_idx(group)  # 解析上下文索引
        num_sms = self.get_theoretical_num_sms('reduce_scatter', self.groups[context_idx]) if num_sms is None else num_sms  # 未指定 SM 数时按带宽模型自动选择
        dsts, event, num_sms = self.runtime.reduce_scatter(  # 调用 C++ 运行时发起 reduce-scatter，返回结果分片、事件与实际 SM 数
            srcs, context_idx, num_sms, comm_precision, scale)  # 传入源张量、上下文、SM 数、归约精度与缩放系数

        # Create handle
        handle = EventOverlap(event)  # 用事件创建句柄，支持与计算重叠
        handle.register_hook_after_wait(lambda: dsts[0] if is_packed else dsts)  # 等待完成后返回本 rank 的结果分片（打包时为单个）
        setattr(handle, 'num_sms', num_sms)  # 把实际使用的 SM 数附加到句柄上
        return handle  # 返回句柄

    # 批量原位 all-reduce：srcs 必须属于本缓冲区，等待后被归约结果原位覆盖
    def all_reduce(self,
                   srcs: Union[torch.Tensor, Sequence[torch.Tensor]],
                   scale: float = 1.0,
                   group: Optional[dist.ProcessGroup] = None,
                   num_sms: Optional[int] = None) -> EventOverlap:
        """Launch batched in-place all-reduce; ``None`` selects the SM count automatically.

        Arguments:
            srcs: source tensors to reduce; must belong to this ``BucketBuffer`` and are
                overwritten in-place with the reduced results.
            scale: the scaling factor applied to the reduced results.
            group: the communication group to use. Required when this buffer owns multiple groups.
            num_sms: the number of SMs to use. ``None`` selects the SM count automatically.
        """
        is_packed, srcs = self._pack_tensors(srcs)  # 归一化 srcs 为列表并记录打包标志

        # Launch
        context_idx = self.get_context_idx(group)  # 解析上下文索引
        num_sms = self.get_theoretical_num_sms('all_reduce', self.groups[context_idx]) if num_sms is None else num_sms  # 未指定时自动选择 SM 数
        dsts, event, num_sms = self.runtime.all_reduce(srcs, context_idx, num_sms, scale)  # 发起原位 all-reduce，返回结果视图、事件与实际 SM 数

        # Create handle
        handle = EventOverlap(event)  # 创建事件句柄
        handle.register_hook_after_wait(lambda: dsts[0] if is_packed else dsts)  # 等待完成后返回归约结果（打包时为单个）
        setattr(handle, 'num_sms', num_sms)  # 附加实际 SM 数
        return handle  # 返回句柄

    # 批量 all-gather：聚合各 rank 的分片；无 dsts 时原位聚合，有 dsts 时（仅纯 NVLink）可写外部张量
    def all_gather(self,
                   srcs: Union[torch.Tensor, Sequence[torch.Tensor]],
                   dsts: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
                   group: Optional[dist.ProcessGroup] = None,
                   num_sms: Optional[int] = None) -> EventOverlap:
        """Launch batched all-gather.

        Arguments:
            srcs: the local shards. Without ``dsts``, each must belong to this ``BucketBuffer`` and be the
                ``rank_idx``-th shard of the gathered tensor, which is gathered in-place. With ``dsts``, each can be
                any contiguous CUDA tensor (out-of-buffer inputs are only supported by the pure NVLink path).
            dsts: the gathered tensors, each must belong to this ``BucketBuffer`` with
                ``dst.nbytes == group.size() * src.nbytes``.
            group: the communication group to use. Required when this buffer owns multiple groups.
            num_sms: the number of SMs to use. Must be ``0`` or ``None`` (copy-engine driven).

        Returns:
            a handle whose ``wait()`` returns the gathered tensor(s) as 1-D views over the storage with the dtype
            of ``srcs``.
        """
        is_packed, srcs, dsts = self._pack_tensors(srcs, dsts)  # 归一化 srcs/dsts 为列表并记录打包标志

        # Launch
        context_idx = self.get_context_idx(group)  # 解析上下文索引
        num_sms = self.get_theoretical_num_sms('all_gather', self.groups[context_idx]) if num_sms is None else num_sms  # 未指定时取理论值（all-gather 为 0，拷贝引擎驱动）
        event, epilogue = self.runtime.all_gather(srcs, dsts, context_idx, num_sms)  # 发起 all-gather，返回事件与延迟执行的 epilogue 函数（生成结果视图）

        # Create handle
        handle = EventOverlap(event)  # 创建事件句柄
        handle.register_hook_after_wait(lambda: epilogue()[0] if is_packed else epilogue())  # 等待完成后执行 epilogue，返回聚合结果视图（打包时为单个）
        setattr(handle, 'num_sms', num_sms)  # 附加实际 SM 数
        return handle  # 返回句柄

    # 销毁缓冲区：释放运行时、上下文、对称存储与 NCCL 句柄
    def destroy(self) -> None:
        super().destroy()  # 基类销毁：断言标志并销毁 C++ 运行时
        self.contexts = None  # 释放上下文列表引用
        self.storage = None  # 释放对称存储引用
        self.nccl_comm_handles = None  # 释放 NCCL 句柄引用
