#include <algorithm> // std::min
#include <cstddef> // std::size_t
#include <cstdint> // 定宽整数类型
#include <cuda.h> // CUDA Driver API（CUstream、CUresult 等）
#include <cuda_runtime_api.h> // CUDA Runtime API（cudaMemcpyBatchAsync 等）

#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（CUDA_RUNTIME_CHECK 等）

#include "../../utils/lazy_driver.hpp" // lazy_driver：延迟加载的 CUDA Driver API 符号
#include "api.hpp" // driver 命名空间的接口声明

namespace deep_ep::driver {

// 在流上递增序列号：先确认当前流不在 CUDA graph 捕获中，再返回自增后的 seq_idx
uint64_t increase_seq_idx(const CUstream& stream, uint64_t& seq_idx) {
    cudaStreamCaptureStatus status; // 流的捕获状态
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status)); // 查询该流是否正在被 graph 捕获
    EP_HOST_ASSERT(status == cudaStreamCaptureStatusNone and "Sequence indices do not support CUDA graph capture"); // 序列号不支持 graph 捕获
    return ++ seq_idx; // 先自增再返回，作为本轮 CE 操作的同步信号值
}

// 构造一条批量流内存操作参数：64 位写值（WRITE_VALUE_64）或等待值（WAIT_VALUE_64）
static CUstreamBatchMemOpParams create_mem_op(
    uint64_t* signal, // 目标信号量（显存地址）
    const uint64_t& value, // 要写入/等待的值
    const CUstreamBatchMemOpType& type, // 操作类型：写值或等待
    const CUstreamWaitValue_flags& wait_flag = CU_STREAM_WAIT_VALUE_EQ) { // 等待的比较条件（默认等于）
    CUstreamBatchMemOpParams params = {}; // 零初始化操作参数结构体
    if (type == CU_STREAM_MEM_OP_WRITE_VALUE_64) { // 写值操作：由 GPU 硬件直接把 value 写入 signal 地址
        params.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64; // 设置操作类型为 64 位写值
        params.writeValue.address = reinterpret_cast<CUdeviceptr>(signal); // 写值目标地址
        params.writeValue.value64 = value; // 要写入的 64 位值
        params.writeValue.flags = 0; // 写值无附加标志
    } else { // 等待操作：流在此阻塞，直到 signal 处的值满足比较条件
        params.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64; // 设置操作类型为 64 位等待
        params.waitValue.address = reinterpret_cast<CUdeviceptr>(signal); // 等待的信号量地址
        params.waitValue.value64 = value; // 期望比较的值
        params.waitValue.flags = wait_flag; // 比较方式：EQ（等于）或 GEQ（大于等于）
    }
    return params; // 返回填充好的参数
}

// 将批量内存操作按上限分批提交到流上（单批最多 255 条，受驱动 API 限制）
static void launch_batched_mem_ops(CUstream stream, std::vector<CUstreamBatchMemOpParams>& ops) {
    static constexpr int kNumMaxBatchMemOps = 255; // 每次 cuStreamBatchMemOp 调用允许的最大操作数
    for (std::size_t offset = 0; offset < ops.size(); offset += kNumMaxBatchMemOps) { // 以 255 为步长遍历操作列表
        const auto num_ops = static_cast<unsigned int>(
            std::min<std::size_t>(kNumMaxBatchMemOps, ops.size() - offset)); // 本批实际操作数（末批可能不足 255）
        CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp(stream, num_ops, ops.data() + offset, 0)); // 提交本批操作到流
    }
}

// 用拷贝引擎（CE）批量异步推送数据：不占用 SM，专用于 DMA 拷贝的硬件引擎
void copy_engine_push(const CUstream& stream,
                      const std::vector<void*>& dsts, // 目的地址列表
                      const std::vector<void*>& srcs, // 源地址列表
                      const std::vector<size_t>& sizes) { // 每次拷贝的字节数
    EP_HOST_ASSERT(dsts.size() == srcs.size() and srcs.size() == sizes.size()); // 三组列表长度必须一致
    if (dsts.empty()) // 空列表则无事可做
        return;

    cudaMemcpyAttributes attrs = {}; // 批量拷贝属性
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream; // 源访问按流序：保证读到流上之前的写入
    attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute; // 允许 CE 拷贝与计算流并行重叠
    size_t attr_idx = 0; // 属性条目索引（所有拷贝共用第 0 个属性）
    CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync( // 提交批量异步拷贝到流上
        dsts.data(), srcs.data(), sizes.data(), dsts.size(), // 目的/源/大小数组及拷贝条数
        &attrs, &attr_idx, 1, stream)); // 属性数组、索引、属性个数、目标流
}

// 拷贝引擎同步：向各 write_signals 写入 seq_idx，并等待各 wait_signals 达到 seq_idx（GEQ）
void copy_engine_sync(const CUstream& stream, const uint64_t& seq_idx, // 本轮同步使用的序列号
                      const std::vector<uint64_t*>& write_signals, // 要写入的信号量地址（通知对端本侧已就绪）
                      const std::vector<uint64_t*>& wait_signals) { // 要等待的信号量地址（对端通知本侧）
    // Write and wait signals // 一次性批量提交全部写值与等待操作
    std::vector<CUstreamBatchMemOpParams> ops(write_signals.size() + wait_signals.size()); // 操作数组：先写后等
    for (int i = 0; i < write_signals.size(); ++ i) { // 前半段放写值操作
        ops[i] = create_mem_op(write_signals[i], seq_idx, CU_STREAM_MEM_OP_WRITE_VALUE_64); // 向对端信号量写入 seq_idx
    }
    for (int i = 0; i < wait_signals.size(); ++ i) { // 后半段放等待操作
        ops[write_signals.size() + i] = create_mem_op(
            wait_signals[i], seq_idx, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ); // 等待对端信号量 >= seq_idx
    }
    launch_batched_mem_ops(stream, ops); // 分批提交到流上执行
}

// 流上等待一组信号量的值全部达到 target（GEQ），常用于等待对端 CE 拷贝完成
void copy_engine_wait(const CUstream& stream, const std::vector<uint64_t*>& signals, const uint64_t& target) {
    std::vector<CUstreamBatchMemOpParams> ops(signals.size()); // 为每个信号量构造一条等待操作
    for (int i = 0; i < signals.size(); ++ i)
        ops[i] = create_mem_op(signals[i], target, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ); // 等待值 >= target
    launch_batched_mem_ops(stream, ops); // 分批提交到流上执行
}

}  // namespace deep_ep::driver // 结束 driver 命名空间
