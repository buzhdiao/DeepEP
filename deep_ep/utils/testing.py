import json  # 导入 JSON 模块，用于解析 Chrome trace
import os  # 导入操作系统接口模块，用于读取环境变量和 devnull
import re  # 导入正则表达式模块，用于解析字节数文本
import sys  # 导入 sys 模块，用于重定向标准输出/错误流
import tempfile  # 导入临时文件模块，用于保存 Chrome trace
from pathlib import Path  # 导入路径类，用于读取临时文件
from typing import Callable, Optional, Union  # 导入类型注解

import numpy as np  # 导入 NumPy，用于统计平均/最小/最大耗时
import torch  # 导入 PyTorch 主库
import torch.distributed as dist  # 导入分布式通信模块


def flush_l2_cache(enabled: bool = True):
    """
    Flush the GPU L2 cache by writing a large zero-initialized tensor.

    Arguments:
        enabled: if `False`, does nothing.
    """
    l2_flush_cache_size = 256e6  # 冲刷缓冲区大小（约 256MB，覆盖常见 L2 容量）
    if enabled:  # 仅在开启时执行冲刷
        torch.empty(int(l2_flush_cache_size // 4), dtype=torch.int, device='cuda').zero_()  # 分配大张量并写零，把 L2 缓存挤出


def bench(fn, num_warmups: int = 30, num_tests: int = 30,
          post_fn: Optional[Callable] = None, flush_l2: bool = True):
    """
    Benchmark a function using CUDA events.

    Arguments:
        fn: the function to benchmark.
        num_warmups: the number of warmup iterations.
        num_tests: the number of measurement iterations.
        post_fn: an optional function to call after each test iteration.
        flush_l2: whether to flush the L2 cache before each iteration.

    Returns:
        avg: the average execution time in seconds.
        min: the minimum execution time in seconds.
        max: the maximum execution time in seconds.
    """
    torch.cuda.synchronize()  # 同步 CUDA，确保基准测试从干净状态开始

    # Warmup
    # 预热阶段，消除首次运行的编译和缓存开销
    for _ in range(num_warmups):  # 执行指定次数的预热迭代
        fn()  # 运行被测函数

    # Testing
    # 正式测量阶段
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]  # 为每次测试创建起始事件
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]  # 为每次测试创建结束事件
    for i in range(num_tests):  # 逐次执行测量
        flush_l2_cache(flush_l2)  # 按需冲刷 L2 缓存
        start_events[i].record()  # 记录起始事件
        fn()  # 运行被测函数
        end_events[i].record()  # 记录结束事件
        if post_fn is not None:  # 若提供了后处理函数则执行
            post_fn()  # 调用每次迭代后的后处理
    torch.cuda.synchronize()  # 等待所有 GPU 工作完成

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]  # 计算各次耗时（秒）并去掉首次
    return np.average(times), np.min(times), np.max(times)  # 返回平均/最小/最大耗时


class empty_suppress:
    # 空的上下文管理器，与 suppress_stdout_stderr 接口一致但不做任何事

    def __enter__(self):
        return self  # 直接返回自身

    def __exit__(self, *_):
        pass  # 退出时不做任何处理


class suppress_stdout_stderr:
    """
    Context manager to suppress stdout and stderr output.
    """

    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')  # 打开指向 devnull 的标准输出重定向文件
        self.errnull_file = open(os.devnull, 'w')  # 打开指向 devnull 的标准错误重定向文件

        self.old_stdout_fileno_undup = sys.stdout.fileno()  # 记录原标准输出的文件描述符
        self.old_stderr_fileno_undup = sys.stderr.fileno()  # 记录原标准错误的文件描述符

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())  # 备份原标准输出描述符
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())  # 备份原标准错误描述符

        self.old_stdout = sys.stdout  # 备份 Python 层的标准输出对象
        self.old_stderr = sys.stderr  # 备份 Python 层的标准错误对象

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)  # 在文件描述符层把 stdout 重定向到 devnull
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)  # 在文件描述符层把 stderr 重定向到 devnull

        sys.stdout = self.outnull_file  # 在 Python 层替换标准输出对象
        sys.stderr = self.errnull_file  # 在 Python 层替换标准错误对象
        return self  # 返回自身

    def __exit__(self, *_):
        sys.stdout = self.old_stdout  # 恢复 Python 层的标准输出对象
        sys.stderr = self.old_stderr  # 恢复 Python 层的标准错误对象

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)  # 在描述符层恢复标准输出
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)  # 在描述符层恢复标准错误

        os.close(self.old_stdout_fileno)  # 关闭备份的标准输出描述符
        os.close(self.old_stderr_fileno)  # 关闭备份的标准错误描述符

        self.outnull_file.close()  # 关闭 devnull 输出文件
        self.errnull_file.close()  # 关闭 devnull 错误文件


def bench_kineto(fn,
                 kernel_names: Union[str, tuple],
                 num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None,
                 flush_l2: bool = True,
                 barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1,
                 barrier: Optional[Callable] = None):
    """
    Benchmark a function using the PyTorch profiler (kineto) to get per-kernel timing.

    Arguments:
        fn: the function to benchmark.
        kernel_names: the CUDA kernel name(s) to profile.
        num_tests: the number of test iterations.
        suppress_kineto_output: whether to suppress profiler output.
        trace_path: the path to save the Chrome trace (`None` to skip).
        flush_l2: whether to flush the L2 cache before each iteration.
        barrier_comm_profiling: whether to insert a barrier before each iteration to reduce
            unbalanced CPU launch overhead.
        num_kernels_per_period: the number of kernels launched per test period.
        barrier: a custom barrier function to use instead of `dist.all_reduce`.

    Returns:
        durations: the average kernel duration(s) in seconds.
    """
    assert isinstance(kernel_names, (str, tuple))  # kernel 名称必须是字符串或元组
    is_tuple = isinstance(kernel_names, tuple)  # 记录是否传入多个 kernel 名称

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    # 与 NVIDIA 性能工具冲突时直接跳过性能分析
    if int(os.environ.get('EP_USE_NVIDIA_TOOLS', 0)):  # 检测是否在 NVIDIA 工具环境下运行
        return (1, ) * len(kernel_names) if is_tuple else 1  # 返回占位值 1

    # For some auto-tuning kernels with prints
    # 先跑一次以触发自动调优 kernel 的输出
    fn()  # 预执行一次被测函数
    torch.cuda.synchronize()  # 同步 CUDA

    # Profile
    # 进入性能分析阶段
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress  # 根据开关选择静默策略
    barrier_comm_profiling &= int(os.environ.get('EP_DISABLE_BARRIER_PROFILING', 0)) == 0  # 环境变量可强制关闭栅栏性能分析
    with suppress():  # 按需静默 profiler 的输出
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)  # 配置 profiler 调度（1 次预热 + 1 次采集）
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True)  # 创建仅采集 CUDA 活动的 profiler
        dummy = torch.ones(1, dtype=torch.float, device='cuda')  # 栅栏用的哑张量
        with profiler:  # 在 profiler 作用域内执行测试
            for i in range(2):  # 两轮：一轮预热，一轮采集
                for _ in range(num_tests):  # 每轮执行指定次数测试
                    # Flush L2 cache
                    # 每次迭代前按需冲刷 L2 缓存
                    flush_l2_cache(flush_l2)  # 冲刷缓存

                    # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                    # 用大 kernel 加栅栏消除各 rank 启动开销不均衡的影响
                    if barrier_comm_profiling:  # 若开启栅栏性能分析
                        torch.cuda._sleep(int(2e7)) # ~10ms  # 用约 10ms 的空转 kernel 对齐各 rank

                        # Some network may have ring-based implement, so be careful to use `all_reduce`
                        # 注意部分网络实现是环形的，慎用 all_reduce
                        if barrier is None:  # 未提供自定义栅栏时用 all_reduce 充当
                            dist.all_reduce(dummy)  # 执行 all_reduce 同步
                        else:  # 提供了自定义栅栏则使用之
                            barrier()  # 调用自定义栅栏
                    fn()  # 运行被测函数
                torch.cuda.synchronize()  # 本轮结束前同步
                profiler.step()  # 推进 profiler 调度阶段

    # Parse the profiling table
    # 解析 profiler 输出的统计表格
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')  # 按总耗时排序并按行拆分
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names  # 统一成元组形式
    assert all([isinstance(name, str) for name in kernel_names])  # 所有 kernel 名称必须是字符串
    for name in kernel_names:  # 逐个校验 kernel 在表格中至多出现一次
        assert sum([name in line for line in prof_lines]) <= 1, f'Errors of the kernel {name} in the profiling table: {prof_lines}'  # 出现多次说明名称有歧义

    # Save chrome traces
    # 按需保存 Chrome trace
    if trace_path is not None:  # 提供了保存路径才导出
        profiler.export_chrome_trace(trace_path)  # 导出 Chrome trace 文件

    # Return average kernel durations
    # 从表格中解析各 kernel 的平均时长
    units = {'ms': 1e3, 'us': 1e6}  # 时间单位到秒的换算比例
    kernel_durations = []  # 收集各 kernel 的平均时长
    for name in kernel_names:  # 逐个 kernel 解析
        total_time = 0  # 该 kernel 的累计总时长（秒）
        total_num = 0  # 该 kernel 的累计调用次数
        for line in prof_lines:  # 遍历表格行
            if name in line:  # 找到包含该 kernel 的行
                time_str = line.split()[-2]  # 取倒数第二列（平均 CUDA 耗时）
                num_str = line.split()[-1]  # 取最后一列（调用次数）
                for unit, scale in units.items():  # 尝试匹配时间单位
                    if unit in time_str:  # 该行使用此单位
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)  # 换算为秒并按调用次数累加
                        total_num += int(num_str)  # 累加调用次数
                        break  # 匹配到单位后退出
        kernel_durations.append(total_time / total_num if total_num > 0 else 0)  # 计算平均时长（无调用时为 0）

    # Expand the kernels by periods
    # 当每个测试周期含多个 kernel 时，按周期展开为逐个时长
    if num_kernels_per_period > 1:  # 仅在需要展开时执行
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:  # 创建临时 JSON 文件
            profiler.export_chrome_trace(tmp.name)  # 导出 Chrome trace
            profile_data = json.loads(Path(tmp.name).read_text())  # 解析 trace 数据

        for i, kernel_name in enumerate(kernel_names):  # 逐个 kernel 展开时长
            events = [event for event in profile_data['traceEvents'] if f'::{kernel_name}' in event['name']]  # 筛出该 kernel 的所有事件
            events = sorted(events, key=lambda event: event['ts'])  # 按时间戳排序
            durations = [event['dur'] / 1e6 for event in events]  # 提取各事件时长（微秒转秒）
            assert len(durations) % num_kernels_per_period == 0  # 事件数必须能被周期数整除
            num_kernel_patterns = len(durations) // num_kernels_per_period  # 计算周期数
            kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) / num_kernel_patterns for j in range(num_kernels_per_period)]  # 按位置取模分组求平均

    # Return execution durations
    # 返回执行时长
    return kernel_durations if is_tuple else kernel_durations[0]  # 多 kernel 返回列表，单 kernel 返回标量


def parse_num_bytes(text: str) -> int:
    """Parse a byte count, optionally with a binary suffix: `1G`, `64M`, `512k`.
    """
    match = re.fullmatch(r'\s*([0-9.]+)\s*([kmgt])?(?:i?b)?\s*', text, re.IGNORECASE)  # 解析数字与可选的二进制后缀
    if match is None:  # 格式不匹配则报错
        raise ValueError(
            f'{text!r} is not a byte count; expected e.g. 1G, 64M, 512K, 1048576')  # 抛出格式错误
    scale = {None: 1, 'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30, 't': 1 << 40}  # 后缀到倍数的映射
    suffix = match.group(2)  # 取出后缀（可能为空）
    try:
        value = int(float(match.group(1)) * scale[suffix.lower() if suffix else None])  # 数字乘以倍数并取整
    except ValueError:  # 数字本身无法解析
        raise ValueError(f'{text!r} has an unparseable number')  # 抛出数值错误
    if value <= 0:  # 字节数必须为正
        raise ValueError(f'byte count must be positive, got {value}')  # 抛出取值错误
    return value  # 返回解析后的字节数
