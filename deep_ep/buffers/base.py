from abc import ABC, abstractmethod  # 导入抽象基类与 abstractmethod 装饰器，用于定义缓冲区基类接口


# 所有通信缓冲区（EP/PP 等）的抽象基类，统一管理生命周期：构造、上下文获取、内存打印与销毁
class BufferBase(ABC):
    """Common lifecycle for communication buffers."""

    # 构造函数：explicitly_destroy 控制是否需要显式调用 destroy() 释放底层通信资源
    def __init__(self, explicitly_destroy: bool = False):
        self.explicitly_destroy = explicitly_destroy  # 保存"是否显式销毁"标志
        self.runtime = None  # 底层 C++ 运行时对象句柄，构造时尚未创建，置为 None

    # 属性：返回第一个通信上下文（内含 stream/通信元信息），销毁后返回 None
    @property
    def main_context(self):
        """The first communication context, or ``None`` after destruction."""
        return self.runtime.main_context if self.runtime is not None else None  # 运行时存在则取其主上下文，否则返回 None

    # 打印当前 rank 的显存占用：工作区（workspace）、通信缓冲区与 RDMA 存储大小
    def print_memory_usage(self) -> None:
        """Print this rank's workspace, communication buffer, and RDMA storage sizes."""
        assert self.runtime is not None  # 断言运行时已创建，否则无内存信息可打印
        self.runtime.print_memory_usage()  # 委托给 C++ 运行时打印各部分内存占用

    # 抽象方法：销毁底层运行时，释放工作区与 RDMA 资源；要求构造时显式传 explicitly_destroy=True
    @abstractmethod
    def destroy(self) -> None:
        """Destroy the runtime. Requires ``explicitly_destroy=True`` at construction."""
        assert self.explicitly_destroy  # 仅当构造时声明显式销毁才允许调用，防止意外释放
        if self.runtime is not None:  # 运行时存在时才需要销毁
            self.runtime.destroy()  # 调用 C++ 运行时销毁底层通信资源（含 RDMA 窗口等）
            self.runtime = None  # 置空句柄，避免悬垂引用；此后 main_context 返回 None
