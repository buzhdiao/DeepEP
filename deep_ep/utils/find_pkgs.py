import functools  # 导入函数工具模块，用于 lru_cache 缓存装饰器
import os  # 导入操作系统接口模块，用于读取环境变量
import sys  # 导入 sys 模块，用于读取模块搜索路径优先级
from importlib.metadata import distributions  # 导入已安装包的元数据枚举接口
from pathlib import Path  # 导入路径类，用于定位库文件
from typing import Optional  # 导入类型注解


def find_pkg_root(name: str, lib_name: Optional[str] = None, optional: bool = False):
    """
    Find the root directory of an installed NVIDIA package by inspecting Python package metadata.
    Checks environment variables `EP_{NAME}_ROOT_DIR` and `{NAME}_DIR` first.

    Arguments:
        name: the package name (e.g., `'nccl'`).
        lib_name: the library filename to search for within the package files.
        optional: if ``False``, raises an assertion error when the package is not found.

    Returns:
        root: the package root directory, or `None` if not found and optional.
    """
    upper = name.upper()  # 包名转大写用于拼环境变量名
    for env_name in (f'EP_{upper}_ROOT_DIR', f'{upper}_DIR'):  # 优先检查用户显式指定的环境变量
        if env_name in os.environ:  # 若设置了该环境变量
            return os.environ[env_name]  # 直接采用其值作为包根目录

    path_priority = {p: i for i, p in enumerate(sys.path)}  # 记录 sys.path 中各目录的优先级
    best, best_priority = None, len(sys.path)  # 当前最优匹配及其优先级

    for dist in distributions():  # 遍历所有已安装的 Python 发行包
        dist_name = (dist.metadata['Name'] or '').lower()  # 读取包名（小写化）
        if f'nvidia-{name}' not in dist_name and f'nvidia_{name}' not in dist_name:  # 只关心 nvidia-<name> 系列包
            continue

        dist_site = str(dist._path.parent)  # 该包所在的 site-packages 目录
        priority = path_priority.get(dist_site, len(sys.path))  # 该目录的搜索优先级
        if priority > best_priority:  # 优先级更低的候选直接跳过
            continue

        if lib_name is not None:  # 需要在包文件中查找指定库文件
            for f in (dist.files or []):  # 遍历包内所有文件
                if lib_name in str(f):  # 找到包含目标库的文件
                    lib_dir = os.path.dirname(str(f.locate()))  # 解析出库所在目录
                    root = os.path.dirname(lib_dir) if os.path.basename(lib_dir) == 'lib' else lib_dir  # 若在 lib 子目录则取上一级作为根目录
                    best, best_priority = root, priority  # 更新最优候选
                    break  # 找到即停止
        else:  # 未指定库名时按标准目录布局查找
            pkg_dir = os.path.join(dist_site, 'nvidia', name)  # 拼出 nvidia/<name> 目录
            if os.path.isdir(pkg_dir):  # 目录存在则采用
                best, best_priority = pkg_dir, priority  # 更新最优候选

    # Raise error if not optional
    # 非可选时找不到包必须报错
    if not optional:  # 根据开关决定是否强制
        assert best is not None, f'Cannot find package: {name}'  # 找不到包则断言失败
    return best  # 返回找到的根目录（可为 None）


@functools.lru_cache()  # 结果缓存，避免重复扫描包元数据
def find_nccl_root(optional: bool = False):
    """
    Find the NCCL installation root directory, cached.

    Arguments:
        optional: if `False`, raises an assertion error when NCCL is not found.

    Returns:
        root: the NCCL root directory.
    """
    return find_pkg_root('nccl', lib_name='libnccl.so', optional=optional)  # 在 nvidia-nccl 包中查找 libnccl.so 并返回根目录


def get_nccl_lib_name(root: str) -> str:
    """Resolve the unversioned library or the SONAME-only library shipped in pip wheels."""
    lib_dir = Path(root) / 'lib'  # 拼出 lib 子目录路径
    for path in (lib_dir / 'libnccl.so', *sorted(lib_dir.glob('libnccl.so.*'))):  # 依次检查无版本号库与带版本号库
        if path.is_file():  # 第一个实际存在的文件即为目标
            return path.name  # 返回库文件名
    raise ModuleNotFoundError(f'libnccl.so not found under {lib_dir}')  # 均不存在则抛出找不到模块异常
