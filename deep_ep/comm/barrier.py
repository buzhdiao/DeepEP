from typing import Any, Optional  # 导入类型注解

import deep_ep._C as _C  # 导入 C++ 扩展模块


def barrier(buffer: Any, wait_comm_stream: bool = True,
            with_cpu_sync: bool = False, sequential: bool = True,
            context_idx: Optional[int] = None) -> None:
    """Run a barrier on the current stream, optionally waiting for the communication stream."""
    context = buffer.context if context_idx is None else buffer.contexts[context_idx]  # 按索引取指定上下文，缺省用默认上下文
    _C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)  # 调用 C++ 层在当前流上执行栅栏
