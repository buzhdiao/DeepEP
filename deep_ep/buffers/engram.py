import functools  # 用于 wraps 装饰器与 lru_cache 缓存
import os  # 读取/写入环境变量（EP_NUM_MAX_LOCAL_RANKS、NCCL_WIN_STRIDE）
from typing import Callable, List, Optional, Tuple  # 类型注解

import torch  # PyTorch：张量、CUDA 设备属性、同步
import torch.distributed as dist  # 分布式进程组与 all_gather_object

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # C++ 扩展：EngramBuffer 运行时、共享句柄创建、存储大小与配置计算

from .base import BufferBase  # 缓冲区抽象基类，统一生命周期管理
from .. import comm  # 通信工具：barrier、通信流、域大小、NCCL 句柄
from ..utils.envs import EnvOverride, check_nvlink_connections, check_fast_rdma_atomic_support  # 环境变量临时覆盖、NVLink 连接检查、RDMA 原子操作能力探测
from ..utils.math import align  # 向上对齐工具函数


# 兼容旧版参数名的装饰器：允许 init 继续接受旧关键字 num_rdma_storage_size
# TODO: remove this
def _accept_legacy_storage_size(init):
    @functools.wraps(init)  # 保留被装饰 __init__ 的元信息
    def wrapper(self, *args, **kwargs):
        # TODO: Remove the legacy keyword alias once callers migrate to num_rdma_storage_bytes.
        if 'num_rdma_storage_size' in kwargs:  # 调用方使用了旧参数名
            if 'num_rdma_storage_bytes' in kwargs:  # 新旧参数同时出现则报错，避免歧义
                raise TypeError('Specify only one of num_rdma_storage_bytes and num_rdma_storage_size')
            kwargs['num_rdma_storage_bytes'] = kwargs.pop('num_rdma_storage_size')  # 将旧参数名重命名为新参数名
        return init(self, *args, **kwargs)  # 调用原始 __init__

    return wrapper  # 返回包装后的 init


# Engram：带独立 RDMA 注册存储分片（CPU 或 GPU）的 GPU 取数缓冲区。
# 典型用途是跨节点按索引拉取大表/专家参数：write() 写入本 rank 分片，
# fetch() 通过 RDMA read 按索引从远端分片取数，返回逐层的等待钩子。
class EngramBuffer(BufferBase):
    """GPU fetch buffer with separate CPU or GPU storage shards registered for RDMA."""

    # Common communication functions; refer to `deep_ep.comm` for usage.
    barrier = comm.barrier  # 类级别绑定通用 barrier 函数
    get_comm_stream = comm.get_comm_stream  # 获取专用通信 CUDA 流
    get_physical_domain_size = comm.get_physical_domain_size  # 获取物理域大小（RDMA/节点内 NVLink 维度）
    get_logical_domain_size = comm.get_logical_domain_size  # 获取逻辑域大小（scale-out / scale-up 维度）

    # 构造函数：建立 NCCL GIN 通信上下文，注册 GPU 与 RDMA 存储分片
    @_accept_legacy_storage_size  # 兼容旧关键字 num_rdma_storage_size
    def __init__(self,
                 group: dist.ProcessGroup,  # 通信进程组
                 num_gpu_bytes: int,  # GPU 侧取数接收区大小（字节）
                 num_rdma_storage_bytes: int,  # 本 rank 的 RDMA 存储分片大小（字节）
                 use_cpu_rdma_storage: bool,  # RDMA 存储分片放在主机内存（True）还是 GPU 显存（False）
                 num_allocated_qps: int,  # 预分配的 RDMA QP（队列对，GIN 上下文）数量
                 qp_depth: int,  # 每个 QP 的队列深度
                 restrict_rd_atomic: bool = False,  # 是否限制请求端 RD read 原子操作数量
                 allow_hybrid_mode: bool = True,  # 是否允许混合模式：同节点多 GPU 共享 NIC 分片
                 sl_idx: Optional[int] = None,  # RDMA 流量的服务等级索引，None 为默认
                 num_gpu_timeout_secs: int = 100,  # GPU 侧等待超时（秒）
                 explicitly_destroy: bool = False):  # 是否要求显式调用 destroy()
        self.group = group  # 保存进程组引用
        self.rank_idx = group.rank()  # 本 rank 在组内的编号
        self.num_ranks = group.size()  # 组内 rank 总数
        self.num_gpu_bytes = num_gpu_bytes  # 保存 GPU 接收区大小
        self.num_rdma_storage_bytes = num_rdma_storage_bytes  # 保存 RDMA 分片大小
        self.use_cpu_rdma_storage = use_cpu_rdma_storage  # 保存 RDMA 分片所在介质标志
        self.allow_hybrid_mode = allow_hybrid_mode  # 保存是否允许混合模式

        num_max_local_ranks = int(os.getenv('EP_NUM_MAX_LOCAL_RANKS', 16)) if allow_hybrid_mode else 1  # 混合模式下每个节点最多共享分片的本地 rank 数（默认 16），否则仅 1
        num_registered_bytes = num_gpu_bytes + num_rdma_storage_bytes * num_max_local_ranks + (1 << 32)  # 需注册的总字节数：GPU 区 + 各本地 rank 分片 + 4GiB 余量
        num_total_gpu_bytes = torch.cuda.get_device_properties('cuda').total_memory  # 当前 GPU 的总显存容量
        if num_registered_bytes > num_total_gpu_bytes:  # 注册量超过显存时需要 CPU 侧窗口注册
            os.environ['NCCL_WIN_STRIDE'] = str(align(num_registered_bytes, 1 << 32))  # 设置 NCCL 窗口步长（对齐到 4GiB），使 NCCL 能在 CPU 侧注册大窗口

        check_nvlink_connections(group)  # 检查组内 NVLink 全互联拓扑是否满足要求

        assert num_allocated_qps > 0 and qp_depth > 0, \
            'pass `num_allocated_qps` and `qp_depth` from `get_theoretical_config`'  # QP 数与深度必须为正，应取自 get_theoretical_config 的建议值
        self.num_allocated_qps = num_allocated_qps  # 保存预分配 QP 数，fetch 时可指定使用的 QP 个数

        shared_comm = []  # 共享通信句柄列表（混合模式下各 rank 的 pid/fd）
        if allow_hybrid_mode:  # 混合模式：本节点所有 rank 共享同一 NIC 上的 GIN 资源
            pid, fd = _C.EngramBuffer.create_shared_handle(num_rdma_storage_bytes, use_cpu_rdma_storage)  # 创建可跨进程共享的 GDAKI 句柄（进程 pid 与可传递的 fd）
            shared_comm = [None] * self.num_ranks  # 准备 all_gather_object 的接收缓冲
            dist.all_gather_object(shared_comm, (pid, fd), group)  # 全体 rank 交换 (pid, fd)，使各进程能打开彼此的共享句柄

        super().__init__(explicitly_destroy)  # 初始化基类：记录显式销毁标志，runtime 暂置 None
        # register only GDAKI backend to accelerate init
        with EnvOverride([
            ('NCCL_GIN_TYPE', '3', True),  # 强制 NCCL GIN 使用类型 3（GDAKI 后端），加快初始化
            # TODO: Configure RD atomics through NCCL devComm config when supported.
            ('NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC', '8', restrict_rd_atomic),  # 仅在 restrict_rd_atomic 时限制每个 QP 的 RD 原子操作数为 8
        ]):
            self.nccl_comm_handle = comm.get_nccl_comm_handle(group, force_new_comm=True)  # 强制新建 NCCL 通信器（避免共享的默认通信器受环境影响）
            self.runtime = _C.EngramBuffer(  # 创建 C++ EngramBuffer 运行时：注册对称显存与 RDMA 分片、建立 QP
                self.rank_idx, self.num_ranks,  # 本 rank 编号与 rank 总数
                self.nccl_comm_handle.get(), shared_comm,  # NCCL 通信器指针与各 rank 的共享句柄
                num_gpu_bytes, num_rdma_storage_bytes,  # GPU 接收区与 RDMA 分片大小
                use_cpu_rdma_storage, allow_hybrid_mode,  # 分片介质与混合模式开关
                sl_idx, num_allocated_qps, qp_depth,  # 服务等级、QP 数量与深度
                num_gpu_timeout_secs,  # GPU 侧超时
                explicitly_destroy)  # 显式销毁标志
        self.context = self.runtime.context  # 保存第一个通信上下文，供上层获取通信流等

        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()  # 逻辑域：跨节点（scale-out）与节点内（scale-up）rank 数
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks  # 本 rank 的 scale-out（节点间）编号
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks  # 本 rank 在节点内的 scale-up 编号
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()  # 物理域：需走 RDMA 的 rank 数与节点内 NVLink 互联的 rank 数

        torch.cuda.synchronize()  # 同步本地 CUDA 流，确保初始化完成
        group.barrier()  # 组内 barrier：等待所有 peer 完成注册与 QP 建立
        torch.cuda.synchronize()  # 再次同步，保证初始化结果全局可见

    # 销毁运行时，释放上下文与 NCCL 通信句柄；需构造时传 explicitly_destroy=True
    def destroy(self) -> None:
        """Destroy the runtime. Requires ``explicitly_destroy=True`` at construction."""
        super().destroy()  # 基类销毁：断言标志并销毁 C++ 运行时
        self.context = None  # 释放上下文引用
        self.nccl_comm_handle = None  # 释放 NCCL 通信句柄引用

    # 静态工具：按模型配置估算（对齐后的）GPU 主缓冲与每 rank RDMA 分片的建议字节数
    @staticmethod
    def get_storage_size_hint(group: dist.ProcessGroup,
                              num_entries_per_layer: List[int], hidden: int,
                              num_max_tokens: int, num_entries_per_token: int,
                              dtype: torch.dtype = torch.bfloat16,
                              num_sf_packs: int = 0) -> Tuple[int, int]:
        """
        Return the aligned main GPU buffer and per-rank RDMA storage sizes.

        The GPU size covers the fetch receive area plus this rank's shard of the FP8
        scaling-factor table; pass ``num_sf_packs=0`` for BF16 storage.
        """
        return _C.EngramBuffer.calculate_storage_size(  # 委托 C++ 计算精确的存储大小
            comm.get_nccl_comm_handle(group).get(),  # 传入 NCCL 通信器以获取域拓扑
            num_entries_per_layer, hidden, dtype.itemsize,  # 每层条目数、隐藏维度、元素字节数
            num_max_tokens, num_entries_per_token, num_sf_packs)  # 最大 token 数、每 token 条目数、FP8 scale factor 包数（0 表示 BF16 存储）

    # 静态工具：返回推荐的 (QP 数, QP 深度, restrict_rd_atomic)；按进程组缓存结果
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_theoretical_config(group: dist.ProcessGroup,
                               num_layers: int,
                               num_max_tokens: int,
                               num_entries_per_token: int,
                               num_max_read_requests_per_qp: int = 16) -> Tuple[int, int, bool]:
        """
        Return the recommended `(num_qps, qp_depth, restrict_rd_atomic)`.
        Pass `restrict_rd_atomic` to the constructor to limit requester read atomics.
        """
        nic_read_mpps = 90 if check_fast_rdma_atomic_support() else 45  # 网卡 RD read 吞吐估计（百万包/秒）：支持快速 RDMA 原子取 90，否则 45
        return _C.EngramBuffer.get_theoretical_config(  # 委托 C++ 按带宽/深度模型计算推荐 QP 配置
            comm.get_nccl_comm_handle(group).get(),  # NCCL 通信器以获取拓扑信息
            num_layers, num_max_tokens, num_entries_per_token,  # 层数、最大 token 数、每 token 条目数
            nic_read_mpps, num_max_read_requests_per_qp)  # 网卡读吞吐估计与每 QP 最大并发读请求数

    # 配置各层条目布局与 token 容量，供运行时确定索引解析与取数窗口的形状
    def set_config(self, num_entries_per_layer: List[int], hidden: int,
                   num_max_tokens: int, num_entries_per_token: int,
                   dtype: torch.dtype = torch.bfloat16,
                   num_sf_packs: int = 0) -> None:
        self.runtime.set_config(num_entries_per_layer, hidden, dtype.itemsize,  # 将各层条目数、隐藏维度、元素字节数告知运行时
                                num_max_tokens, num_entries_per_token, num_sf_packs)  # 同时传入最大 token 数、每 token 条目数与 SF 包数

    # 写入本 rank 的存储分片：storages 为各层数据张量，sfs 为可选的 FP8 scale factor 张量；
    # 供其他 rank 之后通过 fetch 按 RDMA read 读取
    def write(self, storages: List[torch.Tensor],
              sfs: Optional[List[torch.Tensor]] = None) -> None:
        self.runtime.write(storages, sfs)  # 委托 C++ 运行时写入（含可选的 FP8 scale factor）

    # 对所有层发起 RDMA read 取数，返回每层一个等待钩子（Callable），调用钩子即等待该层完成
    def fetch(self, indices: torch.Tensor, num_qps: int = 0,
              use_tma_aligned_col_major_sf: bool = True) -> List[Callable]:
        """
        Issue the RDMA gets for all layers and return one wait hook per layer.

        Arguments:
            indices: `[num_layers, num_tokens, num_entries_per_token]` entry indices to fetch.
            num_qps: the number of GIN contexts (QPs) to drive, 0 for all of the allocated ones.
            use_tma_aligned_col_major_sf: gather the fetched factors column-major and TMA-aligned.
        """
        num_qps = self.num_allocated_qps if num_qps == 0 else num_qps  # num_qps 为 0 时使用全部预分配 QP
        assert num_qps <= self.num_allocated_qps, 'Allocated QPs are not enough'  # 请求的 QP 数不能超过预分配数量
        return self.runtime.fetch(indices, num_qps, use_tma_aligned_col_major_sf)  # 委托 C++ 发起异步 RDMA get，返回逐层等待钩子列表
