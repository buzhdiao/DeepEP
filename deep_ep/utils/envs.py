import functools  # 导入函数工具模块，用于 lru_cache 缓存装饰器
import inspect  # 导入自省模块，用于检查函数签名
import os  # 导入操作系统接口模块，用于读写环境变量
import random  # 导入 Python 随机数模块，用于设置随机种子
import re  # 导入正则表达式模块，用于解析命令行工具输出
import subprocess  # 导入子进程模块，用于调用 nvidia-smi / ibstat
import torch  # 导入 PyTorch 主库
import torch.distributed as dist  # 导入 PyTorch 分布式模块
from typing import List, Tuple  # 导入类型注解

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # 导入 C++ 扩展模块

from .. import comm  # 导入上层 comm 子模块（用于查询域大小）


_local_rank = None  # 本节点内的局部 rank（初始化后写入）
_local_seed = 0  # 由全局种子派生出的本地随机种子
_global_seed = 0  # 用户指定的全局随机种子


class EnvOverride:
    # 环境变量临时覆盖的上下文管理器：进入时设置、退出时恢复原值

    def __init__(self, envs: List[Tuple[str, str, bool]]):
        self.envs = {name: value for name, value, enable in envs if enable}  # 仅保留 enable 为真的 (变量名, 值) 对

    def __enter__(self):
        self.original_values = {name: os.environ.get(name) for name in self.envs}  # 进入前记录各变量原值（可能不存在）
        os.environ.update(self.envs)  # 覆盖写入新的环境变量值
        return self  # 返回自身以支持 as 子句

    def __exit__(self, *_):
        for name, value in self.original_values.items():  # 退出时逐个恢复原值
            if value is None:  # 原本不存在的变量直接删除
                os.environ.pop(name)
            else:  # 原本存在的变量恢复为原值
                os.environ[name] = value


def init_seed(global_seed: int) -> None:
    """
    Initialize the random seed for reproducibility. The local seed is derived from the global seed plus rank.

    Arguments:
        global_seed: the global random seed.
    """
    global _local_seed, _global_seed  # 声明修改模块级种子变量
    _local_seed = global_seed + dist.get_rank()  # 本地种子 = 全局种子 + 全局 rank，保证各 rank 不同
    _global_seed = global_seed  # 记录全局种子
    torch.manual_seed(_local_seed)  # 设置 PyTorch（含 CUDA）随机种子
    random.seed(_local_seed)  # 设置 Python 内置随机种子


def get_local_seed() -> int:
    """
    Get the local random seed.

    Returns:
        seed: the local random seed.
    """
    return _local_seed  # 返回本地随机种子


def get_global_seed() -> int:
    """
    Get the global random seed.

    Returns:
        seed: the global random seed.
    """
    return _global_seed  # 返回全局随机种子


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    """
    Print a message from all ranks, or only from rank 0 of each node, followed by a barrier.

    Arguments:
        s: the message to print.
        once_in_node: if `True`, only the first local rank in each node prints.
    """
    global _local_rank  # 声明使用模块级局部 rank 变量
    assert _local_rank is not None  # 必须先调用 init_dist 完成初始化
    if not once_in_node or _local_rank == 0:  # 若开启"每节点仅一次"则仅局部 rank 0 打印
        print(s, flush=True)  # 立即刷新并打印消息
    dist.barrier()  # 打印后同步所有 rank，保证输出顺序


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:
    """
    Initialize the distributed environment with NCCL backend.

    Arguments:
        local_rank: the local rank index.
        num_local_ranks: the number of local ranks.
        seed: the global random seed.

    Returns:
        rank: the global rank index.
        world_size: the total number of ranks.
        group: the communication group.
    """
    # NOTES: you may rewrite this function with your own cluster settings
    # `init_dist` is only used by tests. Default tests to SL 1 to avoid affecting production traffic, which uses SL 3.
    os.environ.setdefault('NCCL_IB_SL', '1')  # 设置 InfiniBand 服务等级默认值为 1
    os.environ.setdefault('EP_OVERRIDE_RDMA_SL', '1')  # 覆盖 RDMA 服务等级为 1（仅测试用）

    ip = os.getenv('MASTER_ADDR', '127.0.0.1')  # 读取主节点地址，默认本机回环
    port = int(os.getenv('MASTER_PORT', '8361'))  # 读取主节点端口，默认 8361
    num_nodes = int(os.getenv('WORLD_SIZE', 1))  # 由 WORLD_SIZE 推断节点数，默认单机
    node_rank = int(os.getenv('RANK', 0))  # 由 RANK 推断节点编号，默认 0

    # Set local rank
    # 记录本进程的局部 rank
    global _local_rank  # 声明修改模块级变量
    _local_rank = local_rank  # 保存局部 rank 供 dist_print 等使用

    sig = inspect.signature(dist.init_process_group)  # 自省 init_process_group 的函数签名
    params = {
        'backend': 'nccl',  # 使用 NCCL 后端
        'init_method': f'tcp://{ip}:{port}',  # 通过 TCP 方式初始化进程组
        'world_size': num_nodes * num_local_ranks,  # 全局进程总数 = 节点数 * 每节点进程数
        'rank': node_rank * num_local_ranks + local_rank,  # 计算全局 rank 编号
    }
    if 'device_id' in sig.parameters:  # 新版 PyTorch 支持 device_id 参数
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')  # 直接绑定本 rank 的 GPU 设备
    dist.init_process_group(**params)  # 正式初始化分布式进程组
    torch.set_default_dtype(torch.bfloat16)  # 将默认浮点类型设为 bfloat16
    torch.set_default_device('cuda')  # 将默认设备设为 CUDA
    torch.cuda.set_device(local_rank)  # 将本进程绑定到对应 GPU

    init_seed(seed)  # 初始化随机种子
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))  # 返回全局 rank、世界大小及新通信组


def get_physical_domain_size(group: dist.ProcessGroup) -> Tuple[int, int]:
    """Return the physical RDMA and NVLink domain sizes."""
    return comm.get_physical_domain_size(group)  # 委托 comm 模块查询物理 RDMA / NVLink 域大小


def get_logical_domain_size(group: dist.ProcessGroup, allow_hybrid_mode: bool = True) -> Tuple[int, int]:
    """Return the logical scaleout and scaleup domain sizes."""
    return comm.get_logical_domain_size(group, allow_hybrid_mode)  # 委托 comm 模块查询逻辑横向扩展 / 纵向扩展域大小


def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if 'PCIE' in torch.cuda.get_device_name():  # 仅 PCIe 版 GPU 需要（SXM 版全互联无需检查）
        assert group.size() <= 2, 'PCIe GPUs only have pairwise NVLink connections'  # PCIe GPU 仅两两互联，组大小不能超过 2

        # noinspection PyUnresolvedReferences
        import pynvml  # 延迟导入 NVML 管理库
        pynvml.nvmlInit()  # 初始化 NVML

        # noinspection PyTypeChecker
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')  # 读取可见的物理设备编号列表
        physical_device_idx = int(devices[torch.cuda.current_device()])  # 将 CUDA 逻辑设备号映射为物理设备号
        physical_device_indices = [0, ] * group.size()  # 准备收集组内所有 rank 的物理设备号
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)  # 全收集各 rank 的物理设备号

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]  # 获取各物理设备的 NVML 句柄
        for i, handle in enumerate(handles):  # 两两遍历设备对
            for j, peer_handle in enumerate(handles):
                if i >= j:  # 跳过自身与已检查过的对称组合
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)  # 查询该设备对的 NVLink P2P 状态
                assert status == pynvml.NVML_P2P_STATUS_OK, \
                    f'GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink'  # 任意设备对未通过 NVLink 连接则报错

        # Close NVML
        # 检查完毕后关闭 NVML
        pynvml.nvmlShutdown()  # 释放 NVML 资源


def check_torch_deterministic() -> None:
    """
    Ensure PyTorch deterministic algorithms and fill_uninitialized_memory are not both enabled.
    When both are on, `torch.empty()` calls an initialization kernel that may overlap with communication streams,
    causing errors.
    """
    assert not (torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory)  # 两者同时开启会引发与通信流重叠的初始化 kernel，需断言禁止


@functools.lru_cache()  # 结果缓存，进程内只计算一次
def get_sm_read_gbs() -> float:
    """Return the default per-SM HBM read bandwidth in GB/s."""
    # TODO: use architecture-specific values.
    return 180  # 返回默认的每 SM HBM 读取带宽（GB/s）


@functools.lru_cache()  # 结果缓存，进程内只计算一次
def get_sm_write_gbs() -> float:
    """Return the default per-SM HBM write bandwidth in GB/s."""
    # TODO: use architecture-specific values.
    return 45  # 返回默认的每 SM HBM 写入带宽（GB/s）


@functools.lru_cache()  # 结果缓存，避免重复调用 nvidia-smi
def get_nvlink_gbs(factor: float = 0.9) -> float:
    """
    Get the total NVLink bandwidth in GB/s, cached.

    Arguments:
        factor: the bandwidth efficiency factor.

    Returns:
        gbs: the total NVLink bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],  # 调用 nvidia-smi 查询 NVLink 连接状态
                                capture_output=True, text=True, check=True)  # 捕获输出并在失败时抛异常
        output = result.stdout  # 取标准输出文本
        pattern = r'GPU \d+:.*?(?=^GPU \d+:|^$)'  # 匹配第一个 GPU 的 NVLink 信息块
        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)  # 执行正则匹配
        assert match  # 必须能匹配到 GPU 信息块

        gpu_block = match.group(0)  # 取出该 GPU 的信息块
        link_pattern = r'Link \d+:\s*([\d\.]+) GB/s'  # 匹配每条链路的带宽数值
        link_matches = re.findall(link_pattern, gpu_block)  # 提取所有链路带宽
        assert link_matches  # 必须至少存在一条链路
        return sum(float(bw) for bw in link_matches) * factor  # 汇总各链路带宽并乘以效率因子
    except Exception as e:  # 检测失败时降级处理
        print(f'Failed to get NVLink connection speed: {e}')  # 打印失败原因
        return 0  # 返回 0 表示未知


@functools.lru_cache()  # 结果缓存，避免重复调用 ibstat
def check_fast_rdma_atomic_support(nic_name: str = 'mlx5_0') -> bool:
    """
    Check whether the NIC supports fast RDMA atomic operations (MT4131 or newer).

    Arguments:
        nic_name: the NIC device name.

    Returns:
        supported: `True` if fast RDMA atomics are supported.
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)  # 调用 ibstat 查询网卡信息
        output = result.stdout  # 取标准输出文本
        pattern = rf"CA '{nic_name}'.*?CA type:\s*(\S+)"  # 匹配指定网卡的 CA 类型字段
        match = re.search(pattern, output, re.DOTALL)  # 执行正则匹配
        assert match  # 必须能找到该网卡
        return match.group(1) == 'MT4131'  # CA 类型为 MT4131（ConnectX-7 及以上）才支持快速 RDMA 原子操作
    except Exception:  # 检测失败时按不支持处理
        return False  # 返回不支持


@functools.lru_cache()  # 结果缓存，避免重复调用 ibstat
def get_rdma_gbs(nic_name: str = 'mlx5_0') -> float:
    """
    Get the RDMA bandwidth in GB/s, cached.

    Arguments:
        nic_name: the NIC device name.

    Returns:
        gbs: the RDMA bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)  # 调用 ibstat 查询网卡速率
        output = result.stdout  # 取标准输出文本

        pattern = rf"CA '{nic_name}'.*?Port \d+:\s*.*?Rate:\s*(\d+)"  # 匹配指定网卡端口的 Rate 字段（Gb/s）
        match = re.search(pattern, output, re.DOTALL)  # 执行正则匹配
        assert match  # 必须能匹配到速率信息
        rate = int(match.group(1))  # 取出速率数值（单位 Gb/s）
        return rate / 8  # 换算为 GB/s（8 比特 = 1 字节）
    except Exception as e:  # 检测失败时降级处理
        print(f'Failed to get RDMA connection speed: {e}')  # 打印失败原因
        return 0  # 返回 0 表示未知
