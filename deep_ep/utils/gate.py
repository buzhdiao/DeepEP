import torch  # 导入 PyTorch 主库


def generate_topk_idx(rank_count: torch.Tensor, num_tokens: int, num_experts: int, num_ranks: int, num_topk: int) -> torch.Tensor:
    """
    Map rank count to expert indices
    """
    assert torch.equal(torch.sum(rank_count, dim=1), torch.ones(num_tokens, dtype=torch.int, device='cuda') * num_topk)  # 每个 token 的 rank 计数之和必须等于 num_topk
    assert (num_tokens, num_ranks) == rank_count.shape  # rank_count 形状必须为 (num_tokens, num_ranks)
    num_experts_per_rank = num_experts // num_ranks  # 每个 rank 的专家数

    # Generate base value
    # 生成 0..num_experts-1 的基础专家编号矩阵
    base_vals = torch.arange(num_experts, device='cuda').view(1, num_ranks, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)  # 按 (rank, 组内位置) 展开成三维基础矩阵

    # Randomize the ordering within each row
    # 对每个 rank 组内的专家顺序随机打乱
    rand_vals = torch.rand(num_tokens, num_ranks, num_experts_per_rank, device='cuda')  # 生成随机打分
    perm_indices = torch.argsort(rand_vals, dim=-1)  # 按随机打分得到组内排列索引
    permuted = torch.gather(base_vals, 2, perm_indices)  # 按排列索引打乱组内专家编号

    # Create the mask
    # 根据每个 token 在各 rank 上的专家数构造掩码
    k_idx = torch.arange(num_experts_per_rank, device='cuda').view(1, 1, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)  # 组内位置索引矩阵
    rank_count_expanded = rank_count.unsqueeze(2).expand(num_tokens, num_ranks, num_experts_per_rank)  # 将 rank 计数扩展到组内维度
    mask = k_idx < rank_count_expanded  # 位置索引小于计数的元素被选中

    # Get the final indices by masking and reshaping
    # 用掩码筛选后重排成最终的 top-k 索引
    selected = permuted[mask]  # (num_tokens * num_topk,)  # 取出所有被选中的专家编号
    topk_idx = selected.view(num_tokens, num_topk)  # 重塑为 (num_tokens, num_topk)

    return topk_idx  # 返回生成的 top-k 专家索引


def generate_rank_count(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> torch.Tensor:
    """
    Generate rank count tensor for a given number of tokens, experts, ranks, and top-k.

    This function generates a tensor of shape `(num_tokens, num_ranks)` where each element `[i, j]` represents
    the number of topk experts that token `i` have on rank `j`. The distribution is such that
    one special rank gets `ratio` times more traffic than the others.
    """
    num_experts_per_rank = num_experts // num_ranks  # 每个 rank 的专家数
    num_normal_ranks = num_ranks - 1  # 除特殊 rank 外的普通 rank 数

    assert ratio >= 1.0, 'ratio must be no less than 1.0'  # 特殊 rank 的流量比必须不小于 1

    # Generate rank count of each token from random distribution
    # 用随机分布生成每个 token 命中的 rank 数
    random_scores = torch.rand(num_tokens, num_experts, device='cuda')  # 为每个 (token, 专家) 生成随机分数
    topk_weights_, topk_indices = torch.topk(random_scores, num_topk, dim=1, largest=True, sorted=False)  # 随机选出每个 token 的 top-k 专家
    topk_indices //= num_experts_per_rank  # 专家编号换算为 rank 编号
    sorted_topk_indices = torch.sort(topk_indices, dim=1)[0]  # 对每个 token 的 rank 编号排序
    topk_indices_diff_mask = sorted_topk_indices[:, 1:] != sorted_topk_indices[:, :-1]  # 标记相邻元素不同的位置（去重）
    a = topk_indices_diff_mask.sum(dim=1) + 1  # 每个 token 命中的去重 rank 数

    # Upper bound for this generating algorithm
    # 该生成算法下每个 token 的 rank 数上界
    upper_bound_per_token = int(num_normal_ranks / ratio) + 1  # 由流量比推导的上界

    # Clamp the value in range [1, upper_bound_per_token] for each token
    # 将每个 token 的 rank 数截断到上界以内
    a = torch.clamp(a, None, upper_bound_per_token)  # 截断操作

    # Consider the special rank
    # 计算特殊 rank 需要承担流量的 token 数
    sum_a = torch.sum(a).item()  # 所有 token 的 rank 数总和
    normal_token_count = int(sum_a / (num_normal_ranks + ratio))  # 普通 rank 平均应承担的份额对应的 token 数
    special_token_count = sum_a - normal_token_count * num_normal_ranks  # 特殊 rank 需覆盖的剩余 token 数
    special_token_count = min(special_token_count, int(normal_token_count * ratio) + 1)  # 限制特殊 rank 流量不超过 ratio 倍

    # Tokens that the special rank must be in topk
    # 必须包含特殊 rank 的 token（rank 数达到上限者）
    must_mask = (a == num_ranks)  # a 达到 num_ranks 说明该 token 命中所有 rank（含特殊 rank）
    must_count = int(must_mask.sum().item())  # 统计必须包含特殊 rank 的 token 数
    special_token_count = max(must_count, special_token_count)  # 特殊 token 数至少要覆盖必须项
    assert must_count <= special_token_count, 'Too many tokens with full rank assignment'  # 必须项不能超出配额

    # Tokens that the special rank can optionally be in topk
    # 从可选 token 中随机挑选，凑齐特殊 rank 的 token 集合
    optional_token_indices = torch.where(must_mask == 0)[0]  # 可选 token 的候选索引
    optional_token_indices = optional_token_indices[torch.randperm(num_tokens - must_count, device='cuda')][:special_token_count - must_count]  # 随机挑选补足差额
    must_token_indices = torch.where(must_mask != 0)[0]  # 必须 token 的索引
    special_token_row_index = torch.cat(([must_token_indices, optional_token_indices]))  # 拼出特殊 token 的行索引集合

    # Generate permutations for normal ranks
    # 为普通 rank 生成循环排列（编号从 1 开始，0 留给特殊 rank）
    rank_perm = (torch.randperm(num_normal_ranks, device='cuda') + 1).repeat(num_tokens * num_topk // num_normal_ranks + 1)  # 随机排列并重复到足够长度

    # Compute cumulative sum of a to get starting indices in b for each row
    # 用前缀和计算每行在展平数组中的起始位置
    a_cumsum = torch.cumsum(torch.cat((torch.tensor([0], device='cuda'), a)), dim=0)  # 前缀和（含哨兵 0）
    row_starts = a_cumsum[:-1]  # Starting indices for each row in b, shape (n,)  # 每行的起始索引

    # Insert special rank index into the permutation for special tokens
    # 为特殊 token 在展平数组中预留位置（填 0 表示特殊 rank）
    rank_perm_with_special_rank = torch.zeros(num_tokens * num_topk, dtype=torch.long, device='cuda')  # (n * k,)  # 展平的 rank 序列，0 表示特殊 rank
    special_token_mask = torch.zeros(num_tokens * num_topk, dtype=torch.bool, device='cuda')  # 标记特殊 token 占据的展平位置
    special_token_flattened_row_index = row_starts[special_token_row_index]  # 特殊 token 在展平数组中的起始位置
    special_token_mask[special_token_flattened_row_index] = 1  # 标记这些位置
    all_indices = torch.arange(num_tokens * num_topk, device='cuda')  # 全部展平位置索引
    non_special_indices = all_indices[special_token_mask != True]  # 非特殊 token 的位置索引
    rank_perm_with_special_rank[non_special_indices] = rank_perm[:len(non_special_indices)]  # 非特殊位置填入普通 rank 排列

    # Create column index grids
    # 构造列索引网格
    col_idx = torch.arange(num_topk, device='cuda').view(1, num_topk)  # (1, num_topk)  # 列位置索引

    # Compute modulo indices: col_idx % a[i] for each row
    # torch.max is used to avoid zeros in case a[i] = 0 (which happens when the only topk rank is the special rank)
    # 计算每行内的取模索引
    mod_idx = col_idx % a.view(num_tokens, 1)  # (n, num_topk)  # 列索引对各行 rank 数取模

    # Compute indices in b: row_start + (col % a[i])
    # 由行起始位置加取模索引得到展平数组中的最终索引
    b_indices = row_starts.view(num_tokens, 1) + mod_idx  # (n, k)  # 展平数组中的取值索引

    # Gather values from b using computed indices
    # 按索引从展平 rank 序列中取值
    result = rank_perm_with_special_rank[b_indices]  # 得到 (num_tokens, num_topk) 的 rank 索引

    # Shuffle rows randomly to avoid any pattern
    # 随机打乱行顺序以消除模式
    shuffle_indices = torch.randperm(num_tokens, device='cuda')  # 生成行随机排列
    result = result[shuffle_indices]  # Shuffle rows  # 打乱行

    # Create rank count tensor
    # 用 scatter_add 统计每个 (token, rank) 的命中次数
    rank_count = torch.zeros((num_tokens, num_ranks), dtype=torch.int32, device='cuda')  # 初始化 rank 计数矩阵
    rank_count.scatter_add_(dim=1, index=result, src=torch.ones_like(result, dtype=torch.int32))  # 按 rank 索引累加 1
    return rank_count  # 返回 rank 计数张量


def get_precise_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):
    """
    Generate precise unbalanced scores for testing.

    Note that this function generates a distribution with precise unbalanced distribution,
    which **differs from real distribution**.
    """
    # Generate num topk experts for each rank
    # 先按流量比生成每个 token 的 rank 计数
    rank_count = generate_rank_count(num_tokens, num_experts, num_ranks, num_topk, ratio)  # 生成不均衡的 rank 计数

    # Generate scores in a low distribution
    # 先给全部专家生成低分段的分数
    threshold = 0.9  # 低分段的上界阈值
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')  # 分数矩阵
    scores.uniform_(to=threshold)  # 填充 [0, 0.9) 的均匀分布

    # Generate topk indices and change their scores to a high distribution
    # 生成 top-k 索引并把对应分数替换为高分段
    topk_idx = generate_topk_idx(rank_count, num_tokens, num_experts, num_ranks, num_topk)  # 由 rank 计数生成 top-k 索引
    topk_scores = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')  # top-k 高分矩阵
    topk_scores.uniform_(threshold + 1e-6, 1.0)  # 填充略高于阈值的高分段
    row_idx = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, num_topk)  # 行索引网格
    scores[row_idx, topk_idx] = topk_scores  # 将 top-k 位置的分数替换为高分
    return scores  # 返回构造好的不均衡分数


def get_scores_by_factor(num_tokens: int, num_experts: int, num_ranks: int, factor: float) -> torch.Tensor:
    num_experts_per_rank = num_experts // num_ranks  # 每个 rank 的专家数
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')  # 分数矩阵
    scores[:, :num_experts_per_rank].uniform_(to=factor)  # 第一个 rank 的专家用 factor 倍的分段
    scores[:, num_experts_per_rank:].uniform_(to=1)  # 其余 rank 的专家用正常分段
    return scores  # 返回按 factor 构造的分数


def map_unbalanced_ratio_to_factor(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> float:
    num_iterations = 20  # 二分迭代次数
    factor_l, factor_r = 1.0, 100.0  # factor 的二分搜索区间

    num_experts_per_rank = num_experts // num_ranks  # 每个 rank 的专家数
    for _i in range(num_iterations):  # 二分搜索满足流量比的 factor
        factor_mid = (factor_l + factor_r) / 2  # 取区间中点
        scores = get_scores_by_factor(num_tokens, num_experts, num_ranks, factor_mid)  # 按中点 factor 生成分数
        _, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)  # 按分数选出 top-k 专家
        rank_idx = topk_idx // num_experts_per_rank  # 专家编号换算为 rank 编号
        one_hot = torch.nn.functional.one_hot(rank_idx, num_ranks)  # 转成 one-hot 编码
        counts = one_hot.any(dim=1).to(torch.float).sum(dim=0)  # 统计每个 rank 被（去重后）命中的 token 数
        if counts[0].item() > counts[1:].mean().item() * ratio:  # 若特殊 rank 流量超过目标 ratio
            factor_r = factor_mid  # 缩小 factor 上界
        else:  # 否则提高 factor 下界
            factor_l = factor_mid
    return factor_l  # 返回收敛后的 factor


def get_random_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):
    """Generate unbalanced scores with a given ratio.
    """
    factor = 1.0  # 默认 factor 为 1（无偏置）
    if ratio != 1.0:  # 需要不均衡时才做二分搜索
        factor = map_unbalanced_ratio_to_factor(num_tokens, num_experts, num_ranks, num_topk, ratio)  # 搜索满足 ratio 的 factor
    return get_scores_by_factor(num_tokens, num_experts, num_ranks, factor)  # 按该 factor 生成分数


def get_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float, precise: bool):
    if precise:  # 精确模式使用基于 rank 计数的构造方法
        return get_precise_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)  # 精确不均衡分数
    else:  # 随机模式使用二分搜索 factor 的方法
        return get_random_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)  # 随机不均衡分数
