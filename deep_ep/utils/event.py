import torch  # 导入 PyTorch 主库（用于类型注解）
from typing import Any, Optional, Tuple, Callable  # 导入类型注解

# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle  # 导入 C++ 层的 CUDA 事件句柄类型


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """
    # 管理 CUDA 事件的封装类，便于实现计算与通信的流重叠

    def __init__(self, event: Optional[EventHandle] = None, extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event  # 保存捕获的 CUDA 事件（可为空）

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        # TODO: `extra_tensors` is not longer useful for EPv2, as objects are stored in `self.event`
        # 用额外张量模拟 record_stream，否则与 CUDA graph 不兼容
        self.extra_tensors = extra_tensors  # 保存额外张量元组

        # A wrapper for `with event_overlap(release_handle=True)`
        # 记录通过 __call__ 语法设置的释放句柄标志
        self._release_handle_by_call = False  # 默认不在 with 退出时释放事件句柄

        # A hook that will be triggered after `current_stream_wait()`
        # Useful for deterministic dispatch, which requires a sort (on the current stream) after `self.current_stream_wait()` is invoked
        # 在 current_stream_wait 之后触发的钩子（例如确定性 dispatch 需要的排序）
        self.hook_after_wait: Optional[Callable] = None  # 等待后钩子默认为空

    def current_stream_wait(self, release_handle: bool = False) -> Any:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.

        Returns:
            result: the return value of the registered hook, or `None` if no hook is registered.
        """
        assert self.event is not None  # 必须已捕获事件才能等待
        self.event.current_stream_wait()  # 让当前流等待该事件完成

        # Call epilogue hook
        # 调用等待后的尾随钩子
        result = None  # 钩子返回值默认为空
        if self.hook_after_wait is not None:  # 若注册了钩子则执行一次
            result = self.hook_after_wait()  # 调用钩子并记录返回值
            self.hook_after_wait = None  # 钩子只触发一次，随后清除

        # In `self.event`, we also have some V2 APIs storing tensors to record in it,
        # So, after waiting the current stream, those tensors can be released by deleting `self.event`
        # However, you better do it by yourself (to be compatible with multi-stream waits)
        # 事件对象内部还持有需记录的张量，等待完成后可通过删除事件释放它们
        if release_handle:  # 若请求释放句柄
            self.event = None  # 置空以释放事件及其内部张量
        return result  # 返回钩子的返回值

    def wait(self) -> Any:
        """Wait on the current stream and return the registered epilogue result."""
        return self.current_stream_wait()  # 等价于默认参数的 current_stream_wait

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        """
        Register a hook, which will be invoked after `self.current_stream_wait()`
        """
        assert self.hook_after_wait is None, "A hook is already registered on this `EventOverlap`"  # 不允许重复注册钩子
        self.hook_after_wait = hook_after_wait  # 保存等待后钩子

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        """
        Configures the 'release_handle' behavior for the upcoming context manager usage.
        Usage:
            with event_overlap(release_handle=True):
                ...
        Returns `self` to ensure no new wrapper object is created, keeping the reference count of the underlying event unchanged/managed solely by this instance.
        """
        self._release_handle_by_call = release_handle  # 记录 with 退出时是否释放句柄
        return self  # 返回自身，不创建新的封装对象

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap:
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self  # 进入 with 块，返回自身

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:  # 事件仍存在时才需要等待
            self.current_stream_wait(release_handle=self._release_handle_by_call)  # 退出时让当前流等待事件，并按配置决定是否释放句柄
        self._release_handle_by_call = False  # 重置释放标志
