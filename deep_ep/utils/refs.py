import math  # 导入数学模块，用于计算 sin 种子
import torch  # 导入 PyTorch 主库
import torch.distributed as dist  # 导入分布式通信模块
from typing import Optional, Tuple, Union  # 导入类型注解

from .envs import get_global_seed  # 导入获取全局种子的函数
from .math import ceil_div  # 导入向上取整除法工具


def dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             topk_idx: torch.Tensor, topk_weights: Optional[torch.Tensor],
             num_max_tokens_per_rank: int, num_experts: int):
    """
    The reference implementation of dispatching tokens to experts across multiple ranks.

    Not expanded. Sorted by rank and then by token within each rank (i.e. sorted by `src_token_global_idx`).

    Arguments:
    - `x`: Input tokens, `[num_tokens, hidden]` or (`[num_tokens, hidden], [num_tokens, hidden_sf]`)
    - `topk_idx`: Top-k expert indices for each token, `[num_tokens, num_topk]`
    - `topk_weights`: Top-k weights for each token, can be None, `[num_tokens, num_topk]`
    - `num_max_tokens_per_rank`: Maximum number of tokens per rank, must >= actual number of tokens per rank and aligned with DeepEP's `num_max_tokens_per_rank` since we're going to calculate `src_token_global_idx = src_rank_idx*num_max_tokens_per_rank + src_token_local_idx`
    - `num_experts`: Total number of experts across all ranks

    Returns:
    - `recv_x`, `recv_topk_idx`, and `recv_topk_weights`: Received tokens, top-k indices, and top-k weights after dispatching. Out of range `recv_topk_idx` (i.e. that expert is not on the current rank) are set to -1.
    - `recv_src_token_idx`: Received `src_token_global_idx` for each received token
    - `num_recv_tokens_per_rank`: Number of received tokens from each rank, `[num_ranks]`
    """
    # TODO: support forwarding
    # TODO: make top-k weight fully optional
    rank_idx = dist.get_rank()  # 获取当前进程的全局 rank 编号
    num_ranks = dist.get_world_size()  # 获取总进程数

    assert num_experts % num_ranks == 0  # 专家总数必须能被 rank 数整除
    num_experts_per_rank = num_experts // num_ranks  # 计算每个 rank 拥有的专家数

    # Unpack SF
    # 解包 scale factor（FP8 量化因子），输入可能是 (token, sf) 元组
    use_fp8 = isinstance(x, tuple)  # 输入为元组则表示启用 FP8 模式
    x, sf = x if use_fp8 else (x, None)  # 解包出 token 张量与 SF 张量

    # TODO: use SF bytes instead of hardcoded recipe
    num_tokens, hidden = x.size()  # 读取 token 数与隐藏维度
    num_tokens_, num_topk = topk_idx.size()  # 读取 top-k 索引张量的形状
    assert num_tokens == num_tokens_  # token 数必须与 top-k 索引一致
    if sf is not None:  # FP8 模式下校验 SF 张量形状
        num_tokens__, hidden_sf = sf.size()  # 读取 SF 的 token 数与 SF 维度
        assert num_tokens == num_tokens__  # SF 的 token 数必须一致
        assert hidden_sf == ceil_div(hidden, 128)  # SF 维度 = 隐藏维度按 128 对齐后的组数
    if topk_weights is not None:  # 若提供了 top-k 权重则校验形状
        num_tokens__, num_topk_ = topk_weights.size()  # 读取权重张量形状
        assert num_tokens == num_tokens__  # token 数必须一致
        assert num_topk == num_topk_  # top-k 数必须一致

    # Prepare per-peer send buffers
    # 为每个目标 rank 准备发送缓冲区
    send_x_list = []  # 各目标 rank 的 token 数据列表
    send_sf_list = []  # 各目标 rank 的 SF 数据列表
    send_topk_idx_list = []  # 各目标 rank 的 top-k 索引列表
    send_topk_weights_list = []  # 各目标 rank 的 top-k 权重列表
    send_src_token_idx_list = []  # 各目标 rank 的源 token 索引列表
    num_send_tokens_per_rank = torch.zeros((num_ranks, ), dtype=torch.int, device=x.device)  # 记录发往各 rank 的 token 数
    for dst_rank_idx in range(num_ranks):  # 逐个目标 rank 构造发送数据
        expert_start_idx = dst_rank_idx * num_experts_per_rank  # 该 rank 上专家的起始编号
        expert_end_idx = expert_start_idx + num_experts_per_rank  # 该 rank 上专家的结束编号

        # Get the indices of tokens
        # 找出 top-k 命中该 rank 专家的 token 索引
        mask_to_send = ((expert_start_idx <= topk_idx) & (topk_idx < expert_end_idx)).any(dim=1)  # 标记每个 token 是否有专家落在该 rank
        indices_to_send = mask_to_send.nonzero(as_tuple=True)[0]  # 取出需要发往该 rank 的 token 索引
        num_send_tokens_per_rank[dst_rank_idx] = indices_to_send.numel()  # 记录发往该 rank 的 token 数量

        # Select the data for tokens
        # 按索引挑选数据并屏蔽不属于该 rank 的专家索引
        x_to_send = x[indices_to_send]  # 挑选对应的 token 数据
        sf_to_send = sf[indices_to_send] if use_fp8 else None  # FP8 模式下挑选对应 SF
        topk_idx_to_send = topk_idx[indices_to_send]  # 挑选对应的 top-k 索引
        topk_weights_to_send = topk_weights[indices_to_send]  # 挑选对应的 top-k 权重
        masked_topk_idx = torch.where((expert_start_idx <= topk_idx_to_send) & (topk_idx_to_send < expert_end_idx),
                                      topk_idx_to_send, torch.full_like(topk_idx_to_send, -1))  # 将不属于该 rank 的专家索引置为 -1

        send_x_list.append(x_to_send)  # 追加 token 数据
        send_sf_list.append(sf_to_send)  # 追加 SF 数据
        send_topk_idx_list.append(masked_topk_idx)  # 追加屏蔽后的 top-k 索引
        send_topk_weights_list.append(topk_weights_to_send)  # 追加 top-k 权重
        send_src_token_idx_list.append(indices_to_send)  # 追加源 token 索引

    send_x = torch.cat(send_x_list, dim=0)  # 拼接所有目标 rank 的 token 数据
    send_sf = torch.cat(send_sf_list, dim=0) if use_fp8 else None  # FP8 模式下拼接 SF 数据
    send_topk_idx = torch.cat(send_topk_idx_list, dim=0)  # 拼接 top-k 索引
    send_topk_weights = torch.cat(send_topk_weights_list, dim=0)  # 拼接 top-k 权重
    send_src_token_idx = torch.cat(send_src_token_idx_list, dim=0).to(torch.int)  # 拼接源 token 索引并转为 int
    send_src_token_idx += rank_idx * num_max_tokens_per_rank  # 将本地索引换算为全局源 token 索引

    # Exchange size
    # 先互相交换各 rank 的收发 token 数量
    num_recv_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device=x.device)  # 预备接收数量缓冲区
    dist.all_to_all_single(num_recv_tokens_per_rank, num_send_tokens_per_rank)  # all-to-all 交换各 rank 的 token 数
    num_recv_tokens = int(num_recv_tokens_per_rank.sum().item())  # 计算本 rank 将接收的总 token 数

    # Exchange main data
    # 交换主要数据（token、SF、索引、权重、源索引）
    num_send_tokens_per_rank = num_send_tokens_per_rank.tolist()  # 发送数量转为 Python 列表
    num_recv_tokens_per_rank = num_recv_tokens_per_rank.tolist()  # 接收数量转为 Python 列表
    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device=x.device)  # 分配接收 token 缓冲区
    recv_sf = torch.empty((num_recv_tokens, hidden_sf), dtype=sf.dtype, device=x.device) if use_fp8 else None  # FP8 模式下分配接收 SF 缓冲区
    recv_topk_idx = torch.empty((num_recv_tokens, num_topk), dtype=topk_idx.dtype, device=x.device)  # 分配接收 top-k 索引缓冲区
    recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=topk_weights.dtype, device=x.device)  # 分配接收 top-k 权重缓冲区
    recv_src_token_idx = torch.empty((num_recv_tokens, ), dtype=torch.int, device=x.device)  # 分配接收源索引缓冲区
    dist.all_to_all_single(recv_x, send_x, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # all-to-all 交换 token 数据
    if use_fp8:  # FP8 模式下额外交换 SF
        dist.all_to_all_single(recv_sf, send_sf, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # all-to-all 交换 SF 数据
    dist.all_to_all_single(recv_topk_idx, send_topk_idx, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # all-to-all 交换 top-k 索引
    dist.all_to_all_single(recv_topk_weights, send_topk_weights, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # all-to-all 交换 top-k 权重
    dist.all_to_all_single(recv_src_token_idx, send_src_token_idx, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # all-to-all 交换源 token 索引

    # Mask top-k indices
    # 将本地专家索引转换为局部编号，越界的置为 -1
    expert_start_idx = rank_idx * num_experts_per_rank  # 本 rank 专家起始编号
    expert_end_idx = expert_start_idx + num_experts_per_rank  # 本 rank 专家结束编号
    mask = (expert_start_idx <= recv_topk_idx) & (recv_topk_idx < expert_end_idx)  # 标记落在本 rank 范围内的专家索引
    recv_topk_idx = recv_topk_idx - expert_start_idx  # 全局专家编号换算为本 rank 局部编号
    recv_topk_idx.masked_fill_(~mask, -1)  # 不属于本 rank 的索引置为 -1

    # Pack SF
    # 将 SF 重新打包进返回值
    recv_x = (recv_x, recv_sf) if use_fp8 else recv_x  # FP8 模式下返回 (token, sf) 元组

    return (recv_x, recv_topk_idx, recv_topk_weights,
            recv_src_token_idx, torch.tensor(num_recv_tokens_per_rank, dtype=torch.int))  # 返回接收数据、索引、权重、源索引及各 rank 接收数量


def generate_pre_combine_data(src_token_global_idx: torch.Tensor,
                              num_max_tokens_per_rank: int, num_topk: int, hidden: int) -> torch.Tensor:
    """
    Generate data needed for combine from `src_token_global_idx`.
    Recall that `src_token_global_idx = src_rank_idx * num_max_tokens_per_rank + src_token_local_idx`.
    The generated data (denoted as `y`) of the i-th token has a shape of [num_topk, hidden], with

    `y[j, k] = sin((token_seeds * P % max_seed + 1) / max_seed * (k + 1) + sin(seed))`

    where `P=131071` is a large prime, `token_seeds` is calculated via `token_seeds = src_token_global_idx[i] * num_topk + j`,
    and `max_seed = num_ranks * num_max_tokens_per_rank * num_topk`.

    Arguments:
    - `src_token_global_idx`: Source token global indices, `[num_tokens]`

    Returns:
    - Generated data, `[num_tokens, num_topk, hidden]`
    """
    num_ranks = dist.get_world_size()  # 获取总 rank 数
    token_seeds = (src_token_global_idx.unsqueeze(1) * num_topk +
                   torch.arange(num_topk, device=src_token_global_idx.device).unsqueeze(0))  # [num_tokens, num_topk]  # 计算每个 (token, topk 位) 的种子
    max_seed = num_ranks * num_max_tokens_per_rank * num_topk  # 种子取模的上界
    result = torch.sin(
        (((token_seeds * 131071 % max_seed).float() + 1) / max_seed).unsqueeze(-1) *
        torch.arange(1, hidden + 1, device=src_token_global_idx.device, dtype=torch.float32).broadcast_to(1, 1, hidden) +
        math.sin(float(get_global_seed()))  # 按文档公式用大素数散列种子生成确定性数据
    )
    return result.to(torch.bfloat16)  # 转为 bfloat16 返回


def ordered_accumulate(data: torch.Tensor, initial_value: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Accumulate `data` in order along the num_topk dimension.

    Arguments:
    - `data`: Data to be accumulated, `[num_tokens, num_topk, hidden]`
    - `initial_value`: Initial value for accumulation, `[num_tokens, hidden]`

    Returns:
    - Result, `[num_tokens, hidden]`
    """
    num_topk = data.shape[1]  # 读取 top-k 维度大小
    if initial_value is None:  # 未提供初始值时以零初始化
        result = torch.zeros((data.shape[0], data.shape[2]), dtype=torch.float32, device=data.device)  # 创建 float32 零张量
    else:  # 提供了初始值则克隆一份避免原地修改
        result = initial_value.clone()  # 克隆初始值
    for i in range(num_topk):  # 依 top-k 顺序逐个累加
        result += data[:, i, :].float()  # 累加第 i 个 topk 位的数据
    return result.to(data.dtype)  # 转回原始 dtype 返回


def combine(y: torch.Tensor, topk_idx: torch.Tensor,
            num_scaleout_ranks: int, num_scaleup_ranks: int, num_experts: int,
            bias: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
            reduce_in_local: bool, reduce_in_scaleup: bool) -> torch.Tensor:
    """
    The reference implementation of (possibly multi-level reduction) combining tokens.

    Arguments:
    - `y`: Input tokens to be combined, `[num_tokens, num_topk, hidden]`
    - `topk_idx`: `[num_tokens, num_topk]`
    - `reduce_in_local` and `reduce_in_scaleup`: Whether to do reduction within rank and within scale-up group.
       - `(True, True)` -> Hybrid combine
       - `(True, False)` -> Non-hybrid combine
       - `(False, False)` -> Equivalent to `allow_multiple_reduction` is False
       Pay attention that `reduce_in_scaleup` = `True` or `False` is NOT equivalent even if `num_scaleout_ranks == 1` due to `bias` handling.

    Returns:
    - Combined result, `[num_tokens, hidden]`
    """
    num_ranks = num_scaleout_ranks * num_scaleup_ranks  # 总 rank 数 = 横向扩展域 * 纵向扩展域
    num_tokens, hidden = y.shape[0], y.shape[2]  # 读取 token 数与隐藏维度
    num_topk = y.shape[1]  # 读取 top-k 维度大小
    assert not (not reduce_in_local and reduce_in_scaleup), 'Invalid reduction configuration'  # 不允许跳过本地归约却做扩展域归约

    def grouped_reduce(data_to_reduce: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        """
        Perform in-place grouped reduction on `data_to_reduce` according to `group_id`.
        The summation within each group are performed in strict order along the `num_topk` dimension.
        The result for each group is stored at the rightmost token of that group, and other tokens are set to zero.

        Arguments:
        - `data_to_reduce`: Data to be reduced, `[num_tokens, num_topk, hidden]`
        - `group_id`: group IDs for each token, `[num_tokens, num_topk]`
        """
        # Shuffle to make tokens with the same group_id contiguous
        # 排序使同组 token 在 topk 维度上连续（稳定排序）
        group_id, src_indices = torch.sort(group_id, dim=-1, stable=True)  # 对组 ID 稳定排序并记录原始位置
        # transformed_src_indices[i, j] = i * num_topk + src_indices[i, j]
        # 计算展平后的原始位置索引
        transformed_src_indices = (
            (src_indices + torch.arange(0, num_tokens, device=y.device).unsqueeze(-1) * num_topk).flatten())  # 展平为一维的源位置索引
        data_to_reduce = data_to_reduce.view(-1, hidden)[transformed_src_indices].view(num_tokens, num_topk, hidden)  # 按排序后的顺序重排数据
        # Perform segmented reduce within each group
        # 在每个组内做分段归约
        cur_accum_buf = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=y.device)  # 当前段的累加缓冲区
        for i in range(num_topk):  # 沿 topk 维度逐列处理
            is_segment_break = torch.full((num_tokens, ), True, dtype=torch.bool, device=y.device) \
                if i == num_topk - 1 else group_id[:, i] != group_id[:, i + 1]  # 判断各 token 在第 i 位是否为段边界
            cur_accum_buf += data_to_reduce[:, i, :].float()  # 将当前列累加进缓冲区
            # For one token, if `is_segment_break` is True,
            # save the accumulated value and clear the buffer, otherwise, clear `data_to_reduce[:, i, :]`
            # 段边界处保存累加结果，非边界处清零当前列
            segment_break_token_indices = torch.where(is_segment_break)[0]  # 取出段边界 token 的索引
            data_to_reduce[segment_break_token_indices, i] = cur_accum_buf[segment_break_token_indices].to(data_to_reduce.dtype)  # 段结果写入边界位置
            cur_accum_buf[segment_break_token_indices] = 0.0  # 清空累加缓冲区
            non_segment_break_token_indices = torch.where(~is_segment_break)[0]  # 取出非边界 token 的索引
            data_to_reduce[non_segment_break_token_indices, i] = 0.0  # 非边界位置清零
        # Unshuffle
        # 按原始顺序还原数据
        # noinspection PyShadowingNames
        result = torch.empty_like(data_to_reduce)  # 分配还原结果缓冲区
        result.view(-1, hidden)[transformed_src_indices] = data_to_reduce.view(-1, hidden)  # 散射回原始位置
        return result.view(num_tokens, num_topk, hidden)  # 返回还原后的张量

    num_experts_per_rank = num_experts // num_ranks  # 每个 rank 的专家数
    if reduce_in_local:  # 第一级：rank 内归约
        y = grouped_reduce(y, topk_idx // num_experts_per_rank)  # 以 rank 编号作为组 ID 归约
    if reduce_in_scaleup:  # 第二级：纵向扩展组内归约
        y = grouped_reduce(y, topk_idx // (num_experts_per_rank * num_scaleup_ranks))  # 以纵向扩展组编号作为组 ID 归约
    bias_sum = bias[0].float() + bias[1].float() if isinstance(bias, tuple) else bias.float() if bias is not None else None  # 计算偏置（可能是双级偏置元组）
    result = ordered_accumulate(y, bias_sum)  # 以偏置为初值按 topk 顺序累加
    return result  # 返回合并结果
