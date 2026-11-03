from typing import Any  # 导入类型注解

import torch  # 导入 PyTorch 主库

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # 导入 C++ 扩展模块


def get_comm_stream(buffer: Any) -> torch.Stream:
    """Return DeepEP's communication stream."""
    ts: torch.Stream = _C.get_comm_stream()  # 从 C++ 层获取 DeepEP 的通信流对象
    return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)  # 用流属性重新包装为 PyTorch 的 CUDA Stream
