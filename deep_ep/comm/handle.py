import os  # 导入操作系统接口模块，用于读取环境变量
import time  # 导入时间模块，用于生成 force_new_comm 的缓存键
from typing import Any, Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch 主库
import torch.distributed as dist  # 导入分布式通信模块

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # 导入 C++ 扩展模块（NCCL 通信器管理接口）


class NCCLCommHandle:
    """
    A wrapper around a raw NCCL communicator. Manages the lifecycle of the communicator if created by DeepEP,
    or simply wraps an existing one if obtained from PyTorch.

    Attributes:
        nccl_comm: the raw NCCL communicator.
        managed: whether the communicator was created by DeepEP and should be destroyed when this handle is dropped.
    """
    # 原生 NCCL 通信器的封装：若由 DeepEP 创建则负责其生命周期，来自 PyTorch 的则仅包装

    def __init__(self, nccl_comm: int, managed: bool):
        self.nccl_comm = nccl_comm  # 保存原生 NCCL 通信器指针
        self.managed = managed  # 标记是否需要在本句柄销毁时一并销毁通信器
        self.destroy = _C.destroy_nccl_comm  # 绑定 C++ 层的销毁函数

    def __del__(self):
        if self.managed:  # 仅当通信器由 DeepEP 创建时才负责销毁
            self.destroy(self.nccl_comm)  # 调用 C++ 层销毁 NCCL 通信器

    def get(self) -> int:
        """
        Get the raw NCCL communicator.

        Returns:
            nccl_comm: the raw NCCL communicator.
        """
        return self.nccl_comm  # 返回原生通信器指针


_storage = dict()  # 进程组到 NCCL 通信器句柄的全局缓存


def get_nccl_comm_handle(group: dist.ProcessGroup, force_new_comm: bool = False) -> NCCLCommHandle:
    """
    Get or create an NCCL communicator handle for the given process group.
    Results are cached, so subsequent calls with the same group return the same handle.

    Arguments:
        group: the communication group.
        force_new_comm: if set, never reuse PyTorch's communicator and never hit the cache; always
            create a fresh DeepEP-managed comm.

    Returns:
        handle: the NCCL communicator handle.
    """
    # Check cache hit
    # 先查缓存，命中则直接复用
    global _storage  # 声明使用全局缓存字典
    if not force_new_comm and group in _storage:  # 未强制新建且缓存命中
        return _storage[group]  # 返回已缓存的句柄

    # New PyTorch has such API
    # 新版 PyTorch 支持直接取 NCCL 通信器指针
    backend = group._get_backend(torch.device('cuda'))  # 获取进程组的 CUDA 后端（NCCL）
    if not force_new_comm and hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):  # 后端支持且未禁止复用时
        _storage[group] = NCCLCommHandle(backend._comm_ptr(), False)  # 包装 PyTorch 的通信器（不由 DeepEP 管理）
        return _storage[group]  # 返回并缓存该句柄

    # For old PyTorch, we have to recreate a NCCL comm
    # 旧版 PyTorch 需要自行创建新的 NCCL 通信器
    nccl_unique_ids = [None, ] * group.size()  # 准备收集各 rank 的 UniqueID
    dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)  # 全收集本 rank 生成的 UniqueID
    root_unique_id = nccl_unique_ids[0]  # 以 rank 0 的 UniqueID 作为公共引导信息

    # Create a new communicator
    # 创建新的 DeepEP 管理的通信器并缓存
    key = time.time_ns() if force_new_comm else group  # 强制新建时用时间戳作键以免覆盖缓存
    _storage[key] = NCCLCommHandle(
        _C.create_nccl_comm(root_unique_id, group.size(), group.rank()), True)  # 用 UniqueID、组大小和 rank 创建通信器
    return _storage[key]  # 返回新建的句柄


def get_physical_domain_size(group_or_buffer: Any) -> Tuple[int, int]:
    """Return physical RDMA and NVLink domain sizes for a process group or buffer."""
    context = getattr(group_or_buffer, 'context', None)  # 若传入的是缓冲区则取其上下文
    if context is not None:  # 有上下文时直接从缓冲区查询
        return context.get_physical_domain_size()  # 调用上下文方法获取物理域大小
    return _C.get_physical_domain_size(get_nccl_comm_handle(group_or_buffer).get())  # 否则按进程组取通信器后查询


def get_logical_domain_size(group_or_buffer: Any,
                            allow_hybrid_mode: Optional[bool] = None) -> Tuple[int, int]:
    """Return logical scaleout and scaleup domain sizes for a process group or buffer."""
    context = getattr(group_or_buffer, 'context', None)  # 若传入的是缓冲区则取其上下文
    if context is not None:  # 有上下文时直接从缓冲区查询
        return context.get_logical_domain_size()  # 调用上下文方法获取逻辑域大小
    allow_hybrid_mode = True if allow_hybrid_mode is None else allow_hybrid_mode  # 未指定时默认允许混合模式
    return _C.get_logical_domain_size(get_nccl_comm_handle(group_or_buffer).get(), allow_hybrid_mode)  # 按进程组取通信器后查询


def destroy_all_managed_nccl_comm() -> None:
    """
    Destroy all cached NCCL communicator handles and clear the cache.

    """
    _storage.clear()  # 清空缓存；DeepEP 管理的通信器随后由 __del__ 各自销毁
