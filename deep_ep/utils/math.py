import torch  # 导入 PyTorch 主库
from typing import Tuple  # 导入类型注解


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    # 计算两个张量的相对差异（1 - 加权余弦相似度）
    x, y = x.double() + 1, y.double() + 1  # 转为 double 并整体加 1，避免除零且保持符号信息
    denominator = (x * x + y * y).sum()  # 计算归一化分母
    sim = 2 * (x * y).sum() / denominator  # 计算相似度（类似 Dice 系数）
    return (1 - sim).item()  # 返回差异值（浮点）


def safe_div(a, b) -> float:
    # 安全除法：除零且被除数为 0 时返回 0，否则照常抛出异常
    try:
        return a / b  # 正常执行除法
    except ZeroDivisionError as e:  # 捕获除零错误
        if a == 0:  # 0/0 视为 0
            return 0
        else:  # 非 0 除以 0 属于真正错误，重新抛出
            raise


def ceil_div(x: int, y: int) -> int:
    # 向上取整除法
    return (x + y - 1) // y  # 经典的整数上取整公式


def align(x: int, y: int) -> int:
    # 将 x 向上对齐到 y 的整数倍
    return ceil_div(x, y) * y  # 先上取整再乘回 y


@torch.compile(dynamic=True)  # 使用 torch.compile 动态图编译加速
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将 bfloat16 张量按 token 量化为 FP8（e4m3），返回 (量化值, 缩放因子)
    assert x.dim() == 2  # 输入必须是二维 [num_tokens, hidden]
    m, n = x.shape  # 读取行数与列数
    aligned_n = align(n, 128)  # 列数按 128 对齐（FP8 分组量化要求）
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)  # 末尾补零到对齐宽度
    x_padded_view = x_padded.view(m, -1, 128)  # 每 128 列分为一组
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)  # 每组绝对值最大值，下限 1e-4 防止除零
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)  # 按 448 上限缩放量化为 FP8，并返回缩放因子（amax/448）


@torch.compile(dynamic=True)  # 使用 torch.compile 动态图编译加速
def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor) -> torch.Tensor:
    # 将 FP8 量化张量按缩放因子还原为 bfloat16
    if x_fp8.numel() == 0:  # 空张量直接转换返回
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2  # 输入必须是二维
    m, n = x_fp8.shape  # 读取行数与列数
    aligned_n = align(n, 128)  # 计算对齐后的列数
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode='constant', value=0)  # 补零到对齐宽度
    if x_scales.dtype == torch.int:  # 若缩放因子以 int 形式存储（IEEE 754 位模式）
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23  # 左移 23 位还原浮点指数位
        x_scales = x_scales.view(dtype=torch.float)  # 按位重解释为 float32
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.shape[0], -1, 128)  # 转为 float32 并按 128 列分组
    x_scales = x_scales.view(x_fp8.shape[0], -1, 1)  # 缩放因子扩展成与分组对应的形状
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n].contiguous()  # 乘回缩放因子后裁掉补零列并转 bfloat16


def inplace_unique(x: torch.Tensor, num_slots: int) -> None:
    # 就地统计 x 每行的去重取值，并按出现次数降序写入前 num_slots 列，缺失填 -1
    assert x.dim() == 2  # 输入必须是二维
    mask = x < 0  # 找出负值（无效槽位）
    x_padded = x.masked_fill(mask, num_slots)  # 将负值替换为槽位总数（超出统计范围的哨兵）
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)  # 分配直方图缓冲区（含哨兵桶）
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))  # 按取值累加计数
    bin_count = bin_count[:, :num_slots]  # 去掉哨兵桶
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)  # 按出现次数降序排序取值索引
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)  # 出现次数为 0 的索引置为 -1
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values  # 让 -1 沉到末尾
    x[:, :].fill_(-1)  # 先将输出整体填 -1
    valid_len = min(num_slots, x.size(1))  # 有效写入长度受输出列数限制
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]  # 写入按频次排序的去重取值


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    # 将非本组专家的分数清零，得到按组掩码后的分数
    num_tokens, num_experts = scores.shape  # 读取 token 数与专家数
    scores = scores.view(num_tokens, num_groups, -1)  # 按 (token, 组, 组内专家) 重塑
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)  # 初始化组掩码
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)  # 标记每个 token 所属组并扩展到专家维度
    return (scores * mask).view(num_tokens, num_experts)  # 应用掩码并还原原始形状


def hash_tensor(t: torch.Tensor) -> int:
    # 将张量按 int32 视图求和作为简易哈希
    return t.view(torch.int).sum().item()  # 按位重解释为 int 后求和


def hash_tensors(*tensors) -> int:
    # 对任意嵌套的张量集合求异或哈希
    value = 0  # 初始哈希值
    for t in tensors:  # 遍历所有输入
        if isinstance(t, (tuple, list)):  # 嵌套容器则递归
            value ^= hash_tensors(*t)  # 递归计算并异或
        elif t is not None and isinstance(t, torch.Tensor):  # 仅处理有效张量
            value ^= hash_tensor(t)  # 异或该张量的哈希
    return value  # 返回合并后的哈希值


def count_bytes(*tensors) -> int:
    # 统计张量集合占用的总字节数
    total = 0  # 累计字节数
    for t in tensors:  # 遍历所有输入
        if isinstance(t, (tuple, list)):  # 嵌套容器则递归
            total += count_bytes(*t)  # 递归累计
        elif t is not None:  # 忽略空值
            total += t.numel() * t.element_size()  # 元素数乘以单元素字节数
    return total  # 返回总字节数
