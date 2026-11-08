import torch  # 导入 PyTorch，用于 CUDA 同步与张量传递
import torch.distributed as dist  # 导入分布式模块，使用进程组与 barrier
from typing import Optional  # 类型注解：可选参数

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # 导入 C++ 扩展模块，提供 PPBuffer 的底层运行时实现

from .base import BufferBase  # 导入缓冲区抽象基类，统一生命周期管理
from .. import comm  # 导入通信工具模块，用于获取 NCCL 通信句柄


# 流水线并行（Pipeline Parallelism）发送/接收缓冲区：持有一块连续的对称显存，
# 供相邻 pipeline rank 之间进行 send/recv 通信
class PPBuffer(BufferBase):
    """Own contiguous symmetric storage for pipeline-parallel send/recv."""

    # 构造函数：创建 NCCL 通信句柄与 C++ 运行时，分配对称存储
    def __init__(self,
                 group: dist.ProcessGroup,  # 通信进程组
                 num_max_tensor_bytes: int,  # 单次 send/recv 操作允许的最大张量字节数
                 num_max_inflight_tensors: int,  # 同时在途（in-flight）的最大张量数量
                 sl_idx: Optional[int] = None,  # RDMA 流量的服务等级（service level）索引，None 表示默认
                 num_gpu_timeout_secs: int = 100,  # GPU 侧等待超时时间（秒）
                 explicitly_destroy: bool = False):  # 是否要求用户显式调用 destroy() 释放资源
        """Create the storage and its communication group.

        Arguments:
            group: the communication group.
            num_max_tensor_bytes: the maximum tensor size in bytes per send/recv operation.
            num_max_inflight_tensors: the maximum number of in-flight tensors at once.
            sl_idx: the service level index for RDMA traffic. ``None`` uses the default.
            num_gpu_timeout_secs: the GPU-side timeout in seconds.
            explicitly_destroy: whether the user must call ``destroy()`` explicitly.
        """
        self.group = group  # 保存进程组引用
        super().__init__(explicitly_destroy)  # 初始化基类：记录显式销毁标志，runtime 暂置 None
        self.nccl_comm_handle = comm.get_nccl_comm_handle(group)  # 获取该进程组对应的 NCCL 通信句柄

        # Create CPP runtime
        self.runtime = _C.PPBuffer(  # 创建 C++ PPBuffer 运行时，分配对称存储并建立通信通道
            group.rank(), group.size(), self.nccl_comm_handle.get(),  # 传入本 rank 编号、组内 rank 数与 NCCL 通信器
            num_max_tensor_bytes, num_max_inflight_tensors,  # 传入单张量最大字节数与最大在途张量数
            sl_idx, num_gpu_timeout_secs, explicitly_destroy)  # 传入服务等级、GPU 超时与显式销毁标志
        self.storage = self.runtime.storage  # 暴露底层分配的连续对称存储，供上层写入待发送数据

        # Call a barrier to ensure initialization visibility for all peers
        torch.cuda.synchronize()  # 同步本机 CUDA 流，确保本地初始化完成
        group.barrier()  # 组内 barrier：等待所有 peer 完成初始化，保证初始化结果全局可见
        torch.cuda.synchronize()  # 再次同步 CUDA，确保 barrier 后无未完成操作

    # 向流水线环上的相邻 rank 发送张量（阻塞式，直到数据发出）
    def send(self, x: torch.Tensor, dst_rank_idx: int, num_sms: Optional[int] = None) -> None:
        """Send a tensor to an adjacent rank in the pipeline-parallel ring.

        Arguments:
            x: the contiguous CUDA tensor to send; its size must not exceed ``num_max_tensor_bytes``.
            dst_rank_idx: the destination rank, which must be the previous or next rank in the ring.
            num_sms: the number of SMs to use. ``None`` selects all SMs.
        """
        self.runtime.send(x, dst_rank_idx, 0 if num_sms is None else num_sms)  # 委托 C++ 运行时发送；num_sms 为 None 时传 0 表示用全部 SM

    # 从流水线环上的相邻 rank 接收张量到 x（阻塞式，直到数据到达）
    def recv(self, x: torch.Tensor, src_rank_idx: int, num_sms: Optional[int] = None) -> None:
        """Receive a tensor from an adjacent rank in the pipeline-parallel ring.

        Arguments:
            x: the contiguous CUDA output tensor to receive into; its size must not exceed the
                configured ``num_max_tensor_bytes``.
            src_rank_idx: the source rank, which must be the previous or next rank in the ring.
            num_sms: the number of SMs to use. ``None`` selects all SMs.
        """
        self.runtime.recv(x, src_rank_idx, 0 if num_sms is None else num_sms)  # 委托 C++ 运行时接收；num_sms 为 None 时传 0 表示用全部 SM

    # 销毁缓冲区：释放底层运行时、对称存储与 NCCL 通信句柄
    def destroy(self) -> None:
        super().destroy()  # 调用基类销毁逻辑（断言 explicitly_destroy 并销毁 C++ 运行时）
        self.storage = None  # 释放对对称存储的引用
        self.nccl_comm_handle = None  # 释放对 NCCL 通信句柄的引用
