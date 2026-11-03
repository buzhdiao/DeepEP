#pragma once // 头文件保护：只被包含一次

#include <cstddef> // std::size_t
#include <cstdint> // 定宽整数类型（uint64_t 等）
#include <cuda.h> // CUDA Driver API（CUstream、CUresult 等）
#include <vector> // std::vector

// driver 命名空间：封装拷贝引擎（Copy Engine, CE）与流级内存操作
namespace deep_ep::driver {

// 在当前流的捕获状态下递增序列号 seq_idx 并返回（禁止在 CUDA graph 捕获中使用）
uint64_t increase_seq_idx(const CUstream& stream, uint64_t& seq_idx);

// 用拷贝引擎批量异步推送数据：将 srcs[i] 处 sizes[i] 字节拷贝到 dsts[i]
void copy_engine_push(const CUstream& stream,
                      const std::vector<void*>& dsts, // 各次拷贝的目的地址列表
                      const std::vector<void*>& srcs, // 各次拷贝的源地址列表
                      const std::vector<size_t>& sizes); // 各次拷贝的字节数列表

// 拷贝引擎同步：先向 write_signals 写入 seq_idx，再阻塞流直到 wait_signals 达到 seq_idx
void copy_engine_sync(const CUstream& stream, const uint64_t& seq_idx, // 本轮同步使用的序列号
                      const std::vector<uint64_t*>& write_signals, // 需要写入信号值的信号量地址
                      const std::vector<uint64_t*>& wait_signals); // 需要等待的信号量地址

// 流上等待一组信号量的值达到 target（>= target 才继续）
void copy_engine_wait(const CUstream& stream, const std::vector<uint64_t*>& signals, const uint64_t& target);

}  // namespace deep_ep::driver // 结束 driver 命名空间
