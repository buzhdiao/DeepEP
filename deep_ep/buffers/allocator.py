import torch  # 导入 PyTorch，用于张量创建与数据类型描述
from typing import List, Tuple  # 类型注解：分配计划中的 (占位张量, 偏移) 列表

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # 导入 C++ 扩展模块，用于查询底层对齐要求

from ..utils.math import align  # 导入对齐工具函数：将数值向上取整到指定倍数


# 缓冲区分配器：先"规划"若干连续、对齐的张量（元占位符），再一次性用一块 CUDA 存储落实
class BufferAllocator:
    """Plan contiguous, aligned tensors before allocating their CUDA storage."""

    # 构造函数：初始化对齐粒度、分配计划列表、总字节数与落实标志
    def __init__(self):
        self.alignment = _C.get_num_allocation_alignment()  # 从 C++ 层获取底层要求的分配对齐字节数
        self.allocations: List[Tuple[torch.Tensor, int]] = []  # 分配计划：每个元素为 (meta 占位张量, 在大存储中的字节偏移)
        self.num_bytes = 0  # 当前所有分配（含对齐间隙）累计占用的总字节数
        self.materialized = False  # 是否已用真实 CUDA 存储落实该计划（落实后不可再修改）

    # 追加一个对齐后的张量分配，返回其 meta 设备上的占位张量（形状/类型真实，无实际存储）
    def allocate(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """Append an aligned tensor allocation and return its meta placeholder."""
        assert not self.materialized, 'A materialized allocation plan cannot be changed'  # 计划一旦落实便不可再追加
        tensor = torch.empty(shape, dtype=dtype, device='meta')  # 在 meta 设备上创建占位张量，仅记录形状与类型
        offset = align(self.num_bytes, self.alignment)  # 将当前偏移向上对齐到 alignment 的整数倍
        self.allocations.append((tensor, offset))  # 记录 (占位张量, 对齐后偏移) 到分配计划
        self.num_bytes = align(offset + tensor.nbytes, self.alignment)  # 更新累计字节数并再次对齐，保证下个分配也对齐
        return tensor  # 返回占位张量，调用方后续可像普通张量一样使用（落实后自动指向真实存储）

    # 用一块真实的连续 CUDA uint8 存储落实计划：把每个占位张量替换为该存储的对应切片视图
    def materialize(self, storage: torch.Tensor) -> None:
        assert not self.materialized, 'An allocation plan can only be materialized once'  # 计划只允许落实一次
        assert storage.is_cuda and storage.is_contiguous() and storage.dtype == torch.uint8  # 要求 CUDA、连续、uint8 的原始字节存储
        assert storage.numel() == self.num_bytes  # 存储大小必须与计划累计字节数完全一致

        for tensor, offset in self.allocations:  # 遍历每个已规划的 (占位张量, 偏移)
            value = storage.narrow(0, offset, tensor.nbytes).view(tensor.dtype).view(tensor.shape)  # 切出该张量的字节区间，重塑为目标类型与形状
            torch.utils.swap_tensors(tensor, value)  # 原地交换张量内容：使占位张量指向真实存储切片
        self.materialized = True  # 标记计划已落实，此后 allocate 会被断言拒绝
