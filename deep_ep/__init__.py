import filecmp  # 导入文件比较模块，用于逐字节比较 NCCL 库文件
import glob  # 导入通配符路径查找模块，用于查找 libnccl.so 文件
import os  # 导入操作系统接口模块，用于读取环境变量和 /proc 信息
import torch  # 导入 PyTorch 主库

from .utils.find_pkgs import find_nccl_root  # 导入查找 NCCL 安装根目录的工具函数

# Set some default environment provided at setup
# 加载安装（setup）时记录的默认环境变量并写入进程环境
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs  # 导入由安装脚本生成的持久化环境变量字典
    for key, value in persistent_envs.items():  # 遍历每个持久化环境变量键值对
        if key not in os.environ:  # 仅当用户未显式设置该变量时才写入，避免覆盖用户配置
            os.environ[key] = value  # 将默认值写入当前进程的环境变量
except ImportError:  # 若 envs 模块不存在（例如源码直接运行）则忽略
    pass  # 静默跳过默认环境变量设置


def check_nccl_so():
    """
    Verify that the NCCL library loaded at runtime matches the linked version.
    Aborts if duplicate NCCL libraries are found or if versions mismatch.
    """
    if int(os.environ.get('EP_SUPPRESS_NCCL_CHECK', 0)):  # 若环境变量开启了抑制检查开关则直接返回
        return  # 跳过 NCCL 运行库一致性检查

    # PyTorch may load another NCCL library, which is different to the linked one
    # 检查当前进程地址空间中是否存在多个不同的 NCCL 动态库
    with open('/proc/self/maps', 'r') as f:  # 读取本进程的内存映射表
        loaded_nccl_so = None  # 记录第一份加载的 NCCL 库路径
        for so in [line.strip().split(' ')[-1] for line in f if 'libnccl' in line]:  # 提取所有含 libnccl 的映射路径
            loaded_nccl_so = so if loaded_nccl_so is None else loaded_nccl_so  # 仅记录首次出现的路径
            assert so == loaded_nccl_so, f'Duplicate NCCL runtime found in the current system: {so} and {loaded_nccl_so}'  # 若发现重复加载的 NCCL 库则断言失败
    linked_nccl_so_candidates = sorted(glob.glob(f'{find_nccl_root()}/lib/libnccl.so*'))  # 查找安装目录下链接期 NCCL 库候选文件
    assert linked_nccl_so_candidates, f'No libnccl.so found in {find_nccl_root()}/lib/'  # 必须至少找到一个候选库文件
    linked_nccl_so = linked_nccl_so_candidates[0]  # 取排序后的第一个作为链接期库

    # So checking binary-level equalness is necessary
    # 因此需要做二进制级别的逐字节比较
    # noinspection PyTypeChecker
    assert filecmp.cmp(loaded_nccl_so, linked_nccl_so, shallow=False), \
        (f'Invalid NCCL versions: {loaded_nccl_so} (loaded) v.s. {linked_nccl_so} (expected), '
         f'please contact Chenggang or Shangyan to upgrade PyTorch NCCL version')  # 运行时加载的库与链接期库必须完全一致


def init_jit():
    """
    Initialize the JIT compilation runtime.
    """
    # noinspection PyUnresolvedReferences
    import deep_ep._C as _C  # 导入 C++ 扩展模块（JIT 编译入口）
    library_root_path = os.path.dirname(os.path.abspath(__file__))  # 计算本包源码的根目录绝对路径
    _C.init_jit(library_root_path, find_nccl_root())  # 用源码根目录与 NCCL 根目录初始化 JIT 编译运行时


# Run initialization
# 模块导入时立即执行初始化流程
check_nccl_so()  # 先校验 NCCL 运行库一致性
init_jit()  # 再初始化 JIT 编译环境


# Import APIs after initialization
# 初始化完成后才导入对外 API
from . import comm  # 导入通信子模块（NCCL 句柄、流、栅栏等）
from .comm import destroy_all_managed_nccl_comm, get_physical_domain_size, get_logical_domain_size  # 从 comm 子模块再导出常用函数
from .buffers.allocator import BufferAllocator  # 导出缓冲区分配器
from .buffers.base import BufferBase  # 导出缓冲区基类
from .buffers.ep import EPBuffer, EPHandle  # 导出专家并行缓冲区及其句柄
from .buffers.engram import EngramBuffer  # 导出 Engram 缓冲区
from .buffers.bucket import BucketBuffer, BucketSession  # 导出桶式缓冲区及其会话
from .buffers.pp import PPBuffer  # 导出流水线并行缓冲区
# noinspection PyUnresolvedReferences
from .utils.event import EventOverlap, EventHandle  # 导出 CUDA 事件封装与事件句柄

# noinspection PyUnresolvedReferences
from deep_ep._C import (  # 从 C++ 扩展中再导出底层查询函数与类型
    get_num_allocation_alignment,  # 查询内存分配对齐字节数
    get_num_rdma_alignment,  # 查询 RDMA 传输对齐字节数
    get_num_tma_alignment,  # 查询 TMA 传输对齐字节数
    topk_idx_t,  # top-k 索引的 C++ 类型别名
)

__version__ = '2.5.0'  # 当前包版本号
