from typing import Any, Optional  # 导入类型注解
import weakref  # 导入弱引用模块，避免缓存导致的对象泄漏
import functools  # 导入函数工具模块，用于 lru_cache

def value_or(value: Optional[Any], default: Any) -> Any:
    # 若 value 为 None 则返回默认值，否则原样返回
    return default if value is None else value  # 简单的空值回退逻辑


def weak_lru(maxsize: Optional[int] = 128, typed: bool = False):
    """
    LRU Cache decorator that keeps a weak reference to `self`
    Useful for caching methods in classes that may cause memory leaks if `functools.lru_cache` is used directly.
    From https://stackoverflow.com/a/68052994/16569836
    """
    # 用弱引用持有 self 的 LRU 缓存装饰器，避免 lru_cache 直接缓存方法导致实例无法释放
    def wrapper(func):  # 装饰器工厂，接收被装饰的方法

        @functools.lru_cache(maxsize, typed)  # 真正的 lru_cache，缓存在 self 的弱引用上
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)  # 解引用弱引用得到实例后调用原方法

        @functools.wraps(func)  # 保留原函数的元信息
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)  # 传入弱引用以避免强引用泄漏

        return inner  # 返回包装后的方法

    return wrapper  # 返回装饰器
