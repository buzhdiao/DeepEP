# 中文说明：deep_ep.buffers 子包的入口（原文件为空的包标记）。
# 本子包按通信模式组织各 Buffer 的纯 Python 层实现：
#   - base.py      ：Buffer 基类与公共上下文/生命周期逻辑
#   - bucket.py    ：Bucket 式集体通信缓冲（reduce_scatter / all_gather / all_reduce）
#   - ep.py        ：专家并行（EP）的 dispatch / combine / prefetch / grad_reduce 缓冲
#   - pp.py        ：流水线并行（PP）的 send/recv 缓冲
#   - engram.py    ：Engram 取数缓冲
#   - allocator.py ：上述缓冲依赖的分配器工具
# 对外的主要入口（如 deep_ep.Buffer / ElasticBuffer）由上层 deep_ep/__init__.py 转发。
