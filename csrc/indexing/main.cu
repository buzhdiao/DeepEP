// 中文说明：本文件是 DeepEP 的"全量包含"编译冒烟测试单元（compilation unit）：
// 它本身没有任何功能逻辑，只是把所有内核实现头文件逐一 include 进来，
// 编译该 .cu 即可对全部模板头做实例化/类型检查，确保各内核头都能独立编译通过。
// 构建系统（setup.py / CI）会编译此文件作为头文件完整性的回归检查。

// Comm
// 通信原语：跨设备/跨 CTA 屏障（ barrier 内核实现）
#include <deep_ep/impls/comm/barrier.cuh>

// Bucket
// —— Bucket（桶式集体通信）系列内核 ——
// reduce_scatter 的布局计算内核（决定分桶与数据摆放）
#include <deep_ep/layout/bucket/reduce_scatter.cuh>
// 拷贝引擎布局内核（大批量数据搬运的布局规划）
#include <deep_ep/layout/bucket/copy_engine.cuh>
// reduce_scatter 的 NVLink（单机内）实现
#include <deep_ep/impls/bucket/reduce_scatter/nvlink.cuh>
// reduce_scatter 的 RDMA（跨机）实现
#include <deep_ep/impls/bucket/reduce_scatter/rdma.cuh>
// reduce_scatter 的混合（NVLink+RDMA，跨节点同机内组合）实现
#include <deep_ep/impls/bucket/reduce_scatter/hybrid.cuh>
// all_gather 的 RDMA（跨机）实现
#include <deep_ep/impls/bucket/all_gather/rdma.cuh>
// all_gather 的混合（NVLink+RDMA）实现
#include <deep_ep/impls/bucket/all_gather/hybrid.cuh>
// all_reduce 的 NVLink（单机内）实现
#include <deep_ep/impls/bucket/all_reduce/nvlink.cuh>
// all_reduce 的 RDMA（跨机）实现
#include <deep_ep/impls/bucket/all_reduce/rdma.cuh>

// EP
// —— EP（专家并行）dispatch/combine 系列内核 ——
// dispatch：token 按路由结果分发到各专家所在 rank 的内核
#include <deep_ep/impls/ep/dispatch.cuh>
// combine：专家输出按原路由聚合回各 token 的内核
#include <deep_ep/impls/ep/combine.cuh>
// dispatch 的拷贝尾核（copy epilogue）：dispatch 后的重复展开等后处理
#include <deep_ep/impls/ep/dispatch_copy_epilogue.cuh>
// combine 的归约前核（reduce epilogue）：combine 前的重复行累加等预处理
#include <deep_ep/impls/ep/combine_reduce_epilogue.cuh>
// 混合模式（NVLink+RDMA 并存）的 dispatch 内核
#include <deep_ep/impls/ep/hybrid_dispatch.cuh>
// 混合模式的 combine 内核
#include <deep_ep/impls/ep/hybrid_combine.cuh>
// 专家权重前取（prefetch weights）内核：按计划把远端专家权重拉到本地
#include <deep_ep/impls/ep/prefetch_weights.cuh>
// 专家梯度归约（reduce grads）内核：把重复专家的梯度归约回属主
#include <deep_ep/impls/ep/reduce_grads.cuh>

// PP
// 流水线并行（pipeline parallel）的 send/recv 点对点传输内核
#include <deep_ep/impls/pp/pp_send_recv.cuh>

// Engram
// Engram 取数（fetch）内核：按需拉取外部嵌入/缓存数据
#include <deep_ep/impls/engram/engram_fetch.cuh>
// Engram 取数等待（fetch wait）内核：等待异步取数完成
#include <deep_ep/impls/engram/engram_fetch_wait.cuh>

// 空主函数：本文件只为编译期检查，运行时不做任何事
int main() {
    return 0;
}
