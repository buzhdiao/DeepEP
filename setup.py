# 中文说明：DeepEP 的构建脚本。把 C++/CUDA 扩展 deep_ep._C（pybind11 绑定，
# 源码在 csrc/ 下）编译进 deep_ep 包，并定制 wheel 打包行为：
#   - 把集群相关的 JIT 环境变量默认值烘焙进 deep_ep/envs.py；
#   - 把 csrc/ 源码一并拷入 wheel（供 Agent 侧报错时对照源码）。
# 典型用法见 install.sh / build.sh / develop.sh。

# ast：安全地解析 __init__.py 里的版本字面量
import ast
# re：从 __init__.py 中正则匹配 __version__
import re
# os：路径拼接与读取环境变量
import os
# shutil：构建 wheel 时删除/拷贝目录
import shutil
# subprocess：调用 git 获取提交哈希、检查工作区状态
import subprocess
# setuptools：构建入口
import setuptools
# importlib：不触发 deep_ep.__init__ 地加载 find_pkgs 工具模块
import importlib

# Path：跨平台路径拼接
from pathlib import Path
# build_py：setuptools 的打包命令类，下面派生自定义打包逻辑
from setuptools.command.build_py import build_py
# PyTorch 扩展构建设施：CUDAExtension 声明扩展模块，BuildExtension 接管编译
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup.py 所在目录（项目根）的绝对路径
current_dir = os.path.dirname(os.path.realpath(__file__))
# 构建 wheel 时要烘焙进 envs.py 的"持久化"环境变量名单：
# JIT 缓存目录/打印编译命令/C++ 标准、topk 索引位宽、NCCL 路径、RDMA 服务等级
persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND', 'EP_JIT_CPP_STANDARD',
                        'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR', 'EP_DEFAULT_RDMA_SL', 'EP_OVERRIDE_RDMA_SL')

# Load discover module without triggering `deep_ep.__init__`
# 直接按文件路径加载 deep_ep/utils/find_pkgs.py，避免触发 deep_ep 包的
# __init__（那里会 import 编译好的 _C，构建时尚不可用）
find_pkgs_spec = importlib.util.spec_from_file_location('find_pkgs', os.path.join(current_dir, 'deep_ep', 'utils', 'find_pkgs.py'))
find_pkgs = importlib.util.module_from_spec(find_pkgs_spec)
find_pkgs_spec.loader.exec_module(find_pkgs)


def get_package_version():
    # 组装包版本号：<公开版本> + '+' + git 短哈希（本地构建则记 '+local'）
    with open(Path(current_dir) / 'deep_ep' / '__init__.py', 'r') as f:
        # 从 deep_ep/__init__.py 读出 __version__ 字面量
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    # noinspection PyBroadException
    try:
        # 检查 git 工作区是否干净（不干净则直接报错，防止版本不可复现）
        status_cmd = ['git', 'status', '--porcelain']
        status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
        if status_output:
            print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
            assert False, 'Git working directory is not clean'

        # 取 HEAD 的短哈希作为版本后缀
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        # 非 git 环境（如源码包分发）：退化为 '+local' 标记
        revision = '+local'
    return f'{public_version}{revision}'


class CustomBuildPy(build_py):
    # 自定义打包命令：在常规 build_py 之前注入两步额外处理
    def run(self):
        # Make clusters' cache setting default into `envs.py`
        # 第一步：把集群的 JIT 缓存等环境变量默认值生成到 envs.py
        self.generate_default_envs()

        # Then, copy csrc into the wheel for agent-side error lookup
        # 第二步：把 csrc/ 源码拷进 wheel，供 Agent 侧报错时对照
        self.prepare_agent_files()

        # Finally, run the regular build
        # 最后：执行常规 build_py 打包流程
        build_py.run(self)

    def prepare_agent_files(self):
        # Copy csrc into the wheel for agent-side error lookup
        # 把 csrc/ 拷入构建产物的 deep_ep 包目录
        package_dir = os.path.join(self.build_lib, 'deep_ep')
        for name in ('csrc', ):
            dst = os.path.join(package_dir, name)
            # 先清掉目标处的旧拷贝，避免残留
            shutil.rmtree(dst, ignore_errors=True)
            # 拷贝时排除 cmake 构建缓存目录
            ignore = shutil.ignore_patterns('cmake-build-*') if name == 'csrc' else None
            shutil.copytree(os.path.join(current_dir, name), dst, ignore=ignore)

    def generate_default_envs(self):
        # 把构建机上已设置的持久化环境变量烘焙成 deep_ep/envs.py，
        # 安装后这些默认值随包分发（运行时可用真实环境变量覆盖）
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        # noinspection PyShadowingNames
        for name in persistent_env_names:
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        # Create temporary build directory
        # 在构建产物目录中创建 deep_ep/ 并写入 envs.py
        build_include_dir = os.path.join(self.build_lib, 'deep_ep')
        os.makedirs(build_include_dir, exist_ok=True)
        with open(os.path.join(self.build_lib, 'deep_ep', 'envs.py'), 'w') as f:
            f.write(code)


if __name__ == '__main__':
    # 探测 NCCL 安装路径（头文件与库的根目录）
    nccl_root_dir = find_pkgs.find_nccl_root()

    # 宿主 C++ 编译选项：C++20、O3，并屏蔽一批无害告警
    cxx_flags = ['-std=c++20', '-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    # 扩展源文件：pybind11 入口 + 通信上下文实现 + CUDA driver 上下文实现
    sources = ['csrc/python_api.cpp',
               'csrc/kernels/comm/context.cpp',
               'csrc/kernels/driver/driver.cpp']
    # 头文件搜索路径：DeepEP 自带头文件、deep_jit 第三方头、CUDA CCCL
    include_dirs = [f'{current_dir}/deep_ep/include',
                    f'{current_dir}/third-party/deep_jit/include',
                    '/usr/local/cuda/include/cccl']
    # 库搜索路径与链接参数（NCCL 部分稍后补充）
    library_dirs = []
    extra_link_args = []

    # NCCL flags
    # 追加 NCCL 的 include/lib 路径、按探测到的库名链接并固化 rpath
    include_dirs.extend([f'{nccl_root_dir}/include'])
    library_dirs.extend([f'{nccl_root_dir}/lib'])
    extra_link_args.extend([f'-l:{find_pkgs.get_nccl_lib_name(nccl_root_dir)}', f'-Wl,-rpath,{nccl_root_dir}/lib'])

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    # 可选编译期宏：topk_idx 的位宽（32 或 64），影响内核模板特化
    if 'EP_NUM_TOPK_IDX_BITS' in os.environ:
        num_topk_idx_bits = int(os.environ['EP_NUM_TOPK_IDX_BITS'])
        cxx_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')

    # Put them together
    # 汇总编译参数（本扩展无单独的 nvcc 参数段，CUDA 文件由 BuildExtension 处理）
    extra_compile_args = {
        'cxx': cxx_flags,
    }

    # Summary
    # 打印构建配置摘要，便于排查环境问题
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > NCCL path: {nccl_root_dir}')

    # Print persistent env variables
    # 打印当前设置过的持久化环境变量（将被烘焙进 envs.py）
    persistent_envs = []
    for name in persistent_env_names:
        if name in os.environ:
            persistent_envs.append((name, os.environ[name]))
    if len(persistent_envs) > 0:
        print(f' > Persistent envs:')
        for k, v in persistent_envs:
            print(f'   > {k}: {v}')
    print()

    # noinspection bad-argument-type
    # setuptools 入口：包元数据、扩展模块与自定义打包命令
    setuptools.setup(
        name='deep_ep',
        version=get_package_version(),
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        package_data={'deep_ep': ['include/deep_ep/**/*']},
        ext_modules=[
            CUDAExtension(name='deep_ep._C',
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          sources=sources,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args)
        ],
        cmdclass={'build_ext': BuildExtension, 'build_py': CustomBuildPy}
    )
