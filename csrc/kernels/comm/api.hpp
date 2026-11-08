#pragma once // 头文件保护：只被包含一次

#include <memory> // std::shared_ptr
#include <optional> // std::optional

#include <ATen/cuda/CUDAContext.h> // PyTorch 的 CUDA 流封装
#include <torch/python.h> // pybind11 的 Python 绑定支持
#include <nccl.h> // NCCL 主机端 API
#include <nccl_device.h> // NCCL 设备端 API（ncclWindow_t、ncclDevComm_t 等）

#include <deep_ep/layout/bucket/workspace.cuh> // bucket 通信的 workspace 布局
#include <deep_jit/utils/no_ref_ptr.hpp> // NoRefPtr：禁止引用计数的设备指针封装

#include "symmetric.hpp" // symmetric：对称内存分配器（GPU/弹性/混合模式）

namespace deep_ep::comm {

// 获取本 rank 的 NCCL unique id（用于跨进程组建通信器），返回字节数组
pybind11::bytearray get_local_unique_id();

// 用 root 的 unique id 创建 NCCL 主机通信器，返回句柄（int64 伪装的指针）
int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes, // root rank 广播来的 unique id
                         const int& num_ranks, const int& rank_idx); // 通信域总 rank 数与本 rank 编号

// 销毁 NCCL 主机通信器（内部调用 ncclCommAbort）
void destroy_nccl_comm(const int64_t& nccl_comm);

// 查询物理域大小：返回（跨机 RDMA rank 数, 机内 NVLink rank 数）
std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm);

// 查询逻辑域大小：混合模式下即物理域，否则压缩为（1, 总 rank 数）
std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode);

// 查询 GIN 最小步长：返回（gin_min_stride, 总 rank 数 / gin_min_stride）
std::tuple<int, int> get_gin_min_stride(const int64_t& nccl_comm);

// 通信上下文：持有对称内存、NCCL 设备通信器与各层域的拓扑信息
class Context {
    void* raw_window_ptr; // 注册为 NCCL 窗口的原始对称内存起始地址
    std::shared_ptr<symmetric::SymmetricMemory> symmetric_memory; // 对称内存实例（GPU/弹性/混合）
    bool finalized = false; // 是否已调用 finalize()（防止重复释放）

public:
    int rank_idx, num_ranks; // 本 rank 全局编号；通信域总 rank 数
    int num_scaleout_ranks, num_scaleup_ranks; // 逻辑域：跨机（scale-out）rank 数；机内（scale-up）rank 数
    int scaleout_rank_idx, scaleup_rank_idx; // 本 rank 在两层逻辑域中的编号
    int num_rdma_ranks, num_nvl_ranks; // 物理域：RDMA rank 数；NVLink rank 数
    int rdma_rank_idx, nvl_rank_idx; // 本 rank 在两层物理域中的编号
    bool is_scaleup_nvlink; // scale-up 域是否等价于 NVLink 域（走 NVLink 路径）
    int gin_min_stride; // GIN 最小步长（RDMA 存储分片按它分组）

    ncclComm_t comm; // NCCL 主机通信器
    deep_jit::NoRefPtr dev_comm; // NCCL 设备通信器（kernel 内使用的 GIN/multimem 句柄）
    ncclWindow_t window; // 对称内存窗口：kernel 内经它访问任意 NVLink 对端的同偏移地址

    int num_allocated_qps; // 分配的 QP（RDMA 队列对）数量
    int num_cpu_timeout_secs; // CPU 侧等待超时（秒）
    int64_t num_gpu_timeout_cycles; // GPU 侧自旋等待超时（时钟周期数）

    int64_t num_workspace_bytes; // workspace 字节数（信号量/元数据区）
    int64_t num_gpu_buffer_bytes; // GPU 数据缓冲区字节数
    int64_t num_rdma_storage_bytes; // 每 rank 的 RDMA 存储分片字节数
    bool use_cpu_rdma_storage; // RDMA 存储是否落在主机内存（CPU-backed）
    void* workspace; // workspace 起始指针（窗口内偏移 0）
    void* barrier_signals = nullptr; // barrier 信号量缓冲区指针
    void* buffer; // 数据缓冲区指针（紧随 workspace 之后）
    void* buffer_multimem = nullptr; // multimem（组播归约）视角的缓冲区地址
    void* local_rdma_storage_ptr = nullptr; // 本 rank 的 RDMA 存储分片指针
    std::vector<ncclWindow_t> rdma_storage_windows; // 各分片的 GIN 专用窗口（主机侧）
    ncclWindow_t* rdma_storage_windows_gpu = nullptr; // 同上窗口数组的设备侧拷贝

    // 构造函数：基于已有的 NCCL 主机通信器初始化设备通信器、窗口与各层拓扑
    Context(const int64_t& nccl_comm, const symmetric::shared_comm_t& shared_comm, // 主机通信器句柄；各 rank 导出的 (pid, fd) 句柄
            const int& num_ranks, const int& rank_idx, // 总 rank 数与本 rank 编号
            const int64_t& num_workspace_bytes, // workspace 大小（字节）
            const int64_t& num_gpu_buffer_bytes, // GPU 缓冲区大小（字节）
            const int64_t& num_rdma_storage_bytes, // 每 rank RDMA 存储分片大小（字节）
            const bool& use_cpu_rdma_storage, // RDMA 存储是否用主机内存
            const bool& allow_hybrid_mode, // 是否允许混合模式（跨机 RDMA kernel 直发）
            const std::optional<int>& sl_idx, const int& num_allocated_qps, const int& qp_depth, // 服务等级索引；QP 数量与深度
            const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs, // CPU/GPU 两侧超时
            const bool& enable_lsa_multimem = false, // 是否启用 LSA multimem（NVLink 组播归约）
            const std::shared_ptr<Context>& main_context = nullptr); // 主上下文：非空则共享其对称内存

    // 查询物理域大小（RDMA rank 数, NVLink rank 数）
    std::tuple<int, int> get_physical_domain_size() const;

    // 查询逻辑域大小（scale-out rank 数, scale-up rank 数）
    std::tuple<int, int> get_logical_domain_size() const;

    // 取对端 NVLink rank 视角下的对称地址：ptr 必须位于本窗口内，返回同偏移的对端指针
    void* get_sym_ptr(void* ptr, const int& dst_nvl_rank_idx) const;

    // 设置 barrier 信号量缓冲区（由外部分配）
    void set_barrier_signals(void* ptr) {
        barrier_signals = ptr; // 记录信号量缓冲区指针
    }

    // 释放全部资源：注销窗口、销毁设备通信器
    void finalize();
};

}  // namespace deep_ep::comm // 结束 comm 命名空间
