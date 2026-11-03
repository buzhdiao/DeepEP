from .barrier import barrier  # 导出在当前流上执行栅栏的函数
from .handle import (  # 导出 NCCL 通信器句柄相关接口
    NCCLCommHandle,  # NCCL 通信器封装类
    destroy_all_managed_nccl_comm,  # 销毁全部 DeepEP 管理的通信器
    get_logical_domain_size,  # 查询逻辑扩展域大小
    get_nccl_comm_handle,  # 获取或创建 NCCL 通信器句柄
    get_physical_domain_size,  # 查询物理域大小
)
from .stream import get_comm_stream  # 导出获取通信流的函数


__all__ = [  # 对外公开的符号列表
    'NCCLCommHandle',  # NCCL 通信器封装类
    'barrier',  # 栅栏函数
    'destroy_all_managed_nccl_comm',  # 销毁全部管理通信器
    'get_comm_stream',  # 获取通信流
    'get_logical_domain_size',  # 查询逻辑域大小
    'get_nccl_comm_handle',  # 获取通信器句柄
    'get_physical_domain_size',  # 查询物理域大小
]
