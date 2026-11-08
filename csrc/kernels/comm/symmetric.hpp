#pragma once // 头文件保护：只被包含一次

#include <cuda.h> // CUDA Driver API（虚拟内存管理 VMM）
#include <nccl.h> // NCCL 主机端 API

#include <vector> // std::vector
#include <unistd.h> // POSIX 系统调用（getpid 等）
#include <sys/syscall.h> // syscall/pidfd 相关常量
#include <deep_ep/common/compiled.cuh> // 公共编译配置（kNumAllocationAlignmentBytes 等）
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏
#include <deep_ep/common/math.cuh> // 数学工具

#include "../../utils/lazy_driver.hpp" // lazy_driver：延迟加载的 CUDA Driver API 符号

namespace deep_ep::symmetric { // symmetric 命名空间：对称内存（各 rank 同布局可互访）分配器

// 共享分配类型定义
// Shared allocation types: (pid, fd) handle and list of handles from all ranks
using shared_handle_t = std::pair<int, int>; // 共享句柄类型：(进程 pid, 导出的文件描述符 fd)
using shared_comm_t = std::vector<shared_handle_t>; // 全部 rank 的共享句柄列表（跨进程组建对称内存用）

// 设备上下文：缓存当前 GPU 设备与其 NUMA 节点，并构造各类分配属性
struct DeviceContext {
    int device_idx; // 当前 GPU 设备编号
    CUdevice device; // CUDA driver 设备句柄
    int numa_idx; // 设备所属的 NUMA 节点编号

    DeviceContext() { // 构造时一次性查询设备信息
        CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx)); // 取当前活动的 GPU 设备编号
        CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, device_idx)); // 取 CUDA driver 设备句柄
        CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&numa_idx, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device)); // 查询设备所属的 host NUMA 节点
    }

    CUmemAllocationProp gpu_alloc_prop(const bool& skip_fabric = false) const { // GPU 显存分配属性（可选跳过 fabric 句柄）
        return build_alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, device_idx, skip_fabric); // 位置类型为设备显存，编号为当前设备
    }

    CUmemAllocationProp cpu_alloc_prop(const bool& skip_fabric = false) const { // 主机内存分配属性（绑定设备所在 NUMA 节点，用于 CPU-backed RDMA 存储）
        return build_alloc_prop(CU_MEM_LOCATION_TYPE_HOST_NUMA, numa_idx, skip_fabric); // 位置类型为 host NUMA，编号为设备所在节点
    }

private:
// 构造分配属性：pinned + 指定位置，按需启用 FABRIC 句柄与 GPUDirect RDMA
    CUmemAllocationProp build_alloc_prop(const CUmemLocationType& location_type, const int& location_idx, const bool& skip_fabric = false) const {
        CUmemAllocationProp prop = {}; // 零初始化分配属性
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // 使用物理 pinned 内存分配
        prop.location.type = location_type; // 设置内存位置类型（设备或 host NUMA）
        prop.location.id = location_idx; // 设置内存位置编号

        int requested_handle_types = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // 默认只请求 POSIX 文件描述符共享句柄
        int flag = 0; // 属性查询用临时变量
#if CUDART_VERSION >= 13000
// CUDA 13.0 起支持 fabric 句柄查询
        if (not skip_fabric) { // 未指定跳过 fabric 时尝试启用
            CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device)); // 查询设备是否支持 fabric 句柄（跨节点共享显存）
            if (flag) // 支持则
                requested_handle_types |= CU_MEM_HANDLE_TYPE_FABRIC; // 追加 FABRIC 句柄类型
        }
#endif
        prop.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(requested_handle_types); // 把请求的句柄类型写入分配属性

        if (location_type == CU_MEM_LOCATION_TYPE_DEVICE) { // GPU 显存还需支持 GPUDirect RDMA（网卡直接访问显存）
            flag = 0; // 重新查询支持标志
#if CUDART_VERSION >= 13000
// CUDA 13.0 起查询带 VMM 的支持标志
            CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device)); // CUDA 13.0+：查询带 VMM 的 GPUDirect RDMA 支持
#else
            CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, device)); // 旧版本：查询普通 GPUDirect RDMA 支持
#endif
            EP_HOST_ASSERT(flag and "GPUDirect RDMA with CUDA VMM is not supported on this device"); // 必须支持 GPUDirect RDMA 才能做跨机对称内存
            prop.allocFlags.gpuDirectRDMACapable = 1; // 标记该分配可被 GPUDirect RDMA 使用
        }

        // Check granularity
        size_t num_granularity_bytes = 0; // 查询分配粒度
        CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&num_granularity_bytes, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); // 取推荐分配粒度（字节数）
        EP_HOST_ASSERT((num_granularity_bytes == 0 or kNumAllocationAlignmentBytes % num_granularity_bytes == 0) and // 对齐常量必须是 CUDA 分配粒度的整数倍
                       "Alignment must be a multiple of CUDA allocation granularity");
        return prop; // 返回构造好的分配属性
    }
};
// 带 FABRIC 降级的 cuMemCreate：fabric 不被允许/支持时去掉该句柄类型重试（对齐 ncclMemAlloc 逻辑）

// Try cuMemCreate with FABRIC fallback (mirrors ncclMemAlloc logic)
static void cumem_create_with_fallback(CUmemGenericAllocationHandle* handle,
                                       const int64_t& num_bytes, CUmemAllocationProp* prop) { // 分配字节数与属性
    if (prop->requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) { // 请求了 fabric 句柄时先按原属性尝试
        CUresult err = lazy_cuMemCreate(handle, num_bytes, prop, 0); // 尝试创建物理分配
        if (err == CUDA_ERROR_NOT_PERMITTED or err == CUDA_ERROR_NOT_SUPPORTED) { // 驱动不允许或不支持 fabric 时降级
            prop->requestedHandleTypes = static_cast<CUmemAllocationHandleType>( // 从请求的句柄类型中去掉 FABRIC 位
                prop->requestedHandleTypes & ~CU_MEM_HANDLE_TYPE_FABRIC); // 更新属性后再试
            CUDA_DRIVER_CHECK(lazy_cuMemCreate(handle, num_bytes, prop, 0)); // 以降级后的属性重新创建分配
        } else { // 其他错误
            CUDA_DRIVER_CHECK(err); // 直接抛出检查错误
        }
    } else {
        CUDA_DRIVER_CHECK(lazy_cuMemCreate(handle, num_bytes, prop, 0)); // 无 fabric 请求时直接创建
    }
}
// 为虚拟地址段设置读写访问权：本 GPU 必选，可选再开放给指定 NUMA 节点（CPU-backed RDMA 存储）

// Set read/write access for local GPU, and optionally a NUMA node
static void set_access(const CUdeviceptr& addr, const int64_t& num_bytes,
                       const int& device_idx, const int& numa_idx = -1) {
    const bool with_cpu = (numa_idx >= 0); // 是否同时给 CPU（NUMA 节点）映射访问权
    CUmemAccessDesc desc[2]; // 最多两条访问描述符（GPU + 可选 NUMA 节点）
    desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE; // 第一条：设备显存
    desc[0].location.id = device_idx; // 指定当前 GPU 设备
    desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; // 授予读写权限
    if (with_cpu) {
        desc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; // 第二条：host NUMA 内存
        desc[1].location.id = numa_idx; // 指定 NUMA 节点
        desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; // 授予读写权限
    }
    CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(addr, num_bytes, desc, with_cpu ? 2 : 1)); // 为该段虚拟地址设置访问权限（1 或 2 条描述符）
// 对称内存抽象基类
}

class SymmetricMemory {
public:
    void* ptr = nullptr; // 对称内存的起始虚拟地址（所有 rank 同偏移布局）
    int64_t num_sym_bytes = 0; // 对称段的总字节数

    virtual ~SymmetricMemory() noexcept(0) = default; // 虚析构：noexcept(0) 允许析构中抛出检查异常
};
// 纯 GPU 对称内存：直接封装 ncclMemAlloc/ncclMemFree（当前默认）

// Wraps ncclMemAlloc/ncclMemFree (pure GPU, current default)
class GPUSymmetricMemory final : public SymmetricMemory {
public:
    explicit GPUSymmetricMemory(const int64_t& num_sym_bytes) { // 构造：分配指定大小的纯 GPU 对称内存
        EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0); // 大小为正且满足分配对齐
        NCCL_CHECK(ncclMemAlloc(&ptr, num_sym_bytes)); // ncclMemAlloc 分配（NCCL 内部走对称/多播友好路径）
        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAllocationAlignmentBytes == 0); // 返回地址也必须满足对齐要求
        this->num_sym_bytes = num_sym_bytes; // 记录总字节数
    }

    ~GPUSymmetricMemory() override {
        if (ptr != nullptr) { // 非空才释放
            NCCL_CHECK(ncclMemFree(ptr)); // ncclMemFree 释放
            ptr = nullptr; // 指针置空防重复释放
        }
    }
};

// 弹性对称内存：GPU 主段 + 另配的 CPU/GPU RDMA 存储段，两个物理句柄共享一段连续虚拟地址
// GPU main allocation plus a separately backed CPU/GPU storage via CUDA Driver API.
// The two physical handles share one contiguous VA range.
class ElasticSymmetricMemory : public SymmetricMemory {
    int64_t num_rdma_storage_bytes; // RDMA 存储分片的字节数
    CUmemGenericAllocationHandle gpu_handle = {}; // GPU 主段的物理分配句柄
    CUmemGenericAllocationHandle storage_handle = {}; // 存储段（CPU 或 GPU）的物理分配句柄

public:
    ElasticSymmetricMemory(const int64_t& num_sym_bytes, const int64_t& num_rdma_storage_bytes, // 构造：主段与存储段共两个物理句柄
                           const bool& use_cpu_rdma_storage): // 存储是否落在主机内存
        num_rdma_storage_bytes(num_rdma_storage_bytes) { // 初始化存储分片大小
        EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0); // 主段大小为正且满足分配对齐
        EP_HOST_ASSERT(num_rdma_storage_bytes > 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0); // 存储分片大小为正且满足分配对齐

        DeviceContext ctx; // 查询设备上下文
        auto gpu_prop = ctx.gpu_alloc_prop(); // GPU 主段分配属性
        auto storage_prop = use_cpu_rdma_storage ? ctx.cpu_alloc_prop(true) : ctx.gpu_alloc_prop(true); // 存储段属性：CPU-backed 用 host NUMA（跳过 fabric），否则同 GPU

        this->num_sym_bytes = num_sym_bytes; // 记录对称段大小
        const auto num_allocated_bytes = num_sym_bytes + num_rdma_storage_bytes; // 总预留虚拟地址 = 主段 + 存储分片

        // Reserve VA and map segments
        CUdeviceptr addr;
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, num_allocated_bytes, kNumAllocationAlignmentBytes, 0, 0)); // 预留一整段连续虚拟地址（暂不映射物理内存）
        this->ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr)); // 记录对称内存起始地址

        cumem_create_with_fallback(&gpu_handle, num_sym_bytes, &gpu_prop); // 创建 GPU 主段的物理分配
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, num_sym_bytes, 0, gpu_handle, 0)); // 把 GPU 主段映射到虚拟地址起点

        cumem_create_with_fallback(&storage_handle, num_rdma_storage_bytes, &storage_prop); // 创建存储段（CPU 或 GPU）的物理分配
        CUDA_DRIVER_CHECK(lazy_cuMemMap( // 把存储段映射到主段之后的虚拟地址
            addr + num_sym_bytes, num_rdma_storage_bytes, 0, storage_handle, 0)); // 两段共享一段连续 VA，偏移无缝衔接

        if (use_cpu_rdma_storage) {
            set_access(addr, num_sym_bytes, ctx.device_idx); // CPU-backed：主段只给 GPU 访问权
            set_access(addr + num_sym_bytes, num_rdma_storage_bytes, // 存储段同时给 GPU 与 NUMA 节点访问权（主机可直接读写）
                       ctx.device_idx, ctx.numa_idx); // 绑定设备所在 NUMA 节点
        } else {
            set_access(addr, num_allocated_bytes, ctx.device_idx); // 全 GPU-backed：整段 VA 都只给 GPU 访问权
        }

        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAllocationAlignmentBytes == 0); // 起始地址必须满足对齐要求
    }

    ~ElasticSymmetricMemory() override {
        auto addr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr)); // 还原虚拟地址
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, num_sym_bytes)); // 解除 GPU 主段映射
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(gpu_handle)); // 释放 GPU 主段物理分配
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr + num_sym_bytes, num_rdma_storage_bytes)); // 解除存储段映射
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(storage_handle)); // 释放存储段物理分配
        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree( // 释放整段预留的虚拟地址
            addr, num_sym_bytes + num_rdma_storage_bytes)); // 释放范围为两段总大小
    }
};
// 混合弹性对称内存：把同节点各 rank 导出的 RDMA 存储分片连续映射到主段之后（跨机 kernel 直访对端分片）

// Maps the RDMA storage shards exported by intra-node ranks contiguously after the main symmetric-memory segment.
class HybridElasticSymmetricMemory final : public SymmetricMemory {
    int num_scaleup_ranks; // 机内（scale-up）rank 数
    int64_t num_rdma_storage_bytes; // 每 rank 的 RDMA 存储分片字节数
    CUmemGenericAllocationHandle gpu_handle = {}; // GPU 主段的物理分配句柄
    std::vector<CUmemGenericAllocationHandle> storage_handles; // 各机内 rank 导入的存储段句柄
    int local_export_fd = -1; // 本进程导出的 fd（析构时关闭）

public:
    HybridElasticSymmetricMemory(const shared_comm_t& shared_comm, // 构造：导入全部机内 rank 的存储分片并连续映射
                                 const int64_t& num_sym_bytes, const int64_t& num_rdma_storage_bytes, // 主段与每 rank 分片大小
                                 const bool& use_cpu_rdma_storage, // 存储是否用主机内存
                                 const int& num_scaleup_ranks, const int& scaleout_rank_idx): // 机内 rank 数与本 rank 的跨机编号
        num_scaleup_ranks(num_scaleup_ranks), // 记录机内 rank 数
        num_rdma_storage_bytes(num_rdma_storage_bytes), // 记录每 rank 分片大小
        storage_handles(num_scaleup_ranks) { // 按机内 rank 数预分配句柄数组
        EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0); // 主段大小为正且满足分配对齐
        EP_HOST_ASSERT(num_rdma_storage_bytes > 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0); // 分片大小为正且满足分配对齐

        DeviceContext ctx; // 查询设备上下文
        auto gpu_prop = ctx.gpu_alloc_prop(); // GPU 主段分配属性

        this->num_sym_bytes = num_sym_bytes; // 记录对称段大小
        const auto num_mapped_bytes = num_sym_bytes + num_rdma_storage_bytes * num_scaleup_ranks; // 总映射大小 = 主段 + 机内 rank 数 × 每 rank 分片

        // Reserve VA
        CUdeviceptr addr; // 还原虚拟地址
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, num_mapped_bytes, kNumAllocationAlignmentBytes, 0, 0)); // 预留一整段连续虚拟地址
        ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr)); // 记录对称内存起始地址
        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAllocationAlignmentBytes == 0); // 起始地址必须满足对齐要求

        // Map GPU segment // 映射 GPU 主段
        cumem_create_with_fallback(&gpu_handle, num_sym_bytes, &gpu_prop); // 创建 GPU 主段物理分配
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, num_sym_bytes, 0, gpu_handle, 0)); // 映射到虚拟地址起点

        // Import and map all intra-node storage segments. // 导入并映射所有机内 rank 的存储段（跨进程经 fd 传递）
        const auto local_pid = getpid(); // 本进程 pid
        for (int i = 0; i < num_scaleup_ranks; ++ i) { // 遍历同节点所有 rank 的分片
            auto [pid, fd] = shared_comm[num_scaleup_ranks * scaleout_rank_idx + i]; // 取该 rank 导出的 (pid, fd) 句柄
            int local_fd = fd; // 默认假定 fd 可直接用（同进程时）
            if (pid != local_pid) { // fd 属于其他进程时需要转交
                int pidfd = syscall(SYS_pidfd_open, pid, 0); // pidfd_open 打开目标进程
                EP_HOST_ASSERT(pidfd >= 0 and "`pidfd_open` failed"); // 打开必须成功
                local_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0); // pidfd_getfd 把对方的 fd 复制进本进程
                EP_HOST_ASSERT(local_fd >= 0 and "`pidfd_getfd` failed"); // 复制必须成功
                close(pidfd); // 关闭 pidfd（pidfd 只用于转交）
            }

            const auto offset = num_sym_bytes + i * num_rdma_storage_bytes; // 该分片在连续 VA 中的偏移
            CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle( // 从 POSIX fd 导入物理分配句柄
                &storage_handles[i], // 写入第 i 个句柄槽位
                reinterpret_cast<void*>(static_cast<uintptr_t>(local_fd)), // 以可共享句柄（fd）形式传入
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)); // 句柄类型为 POSIX 文件描述符
            CUDA_DRIVER_CHECK(lazy_cuMemMap(addr + offset, num_rdma_storage_bytes, 0, storage_handles[i], 0)); // 把该分片映射到预留 VA 的对应偏移

            if (pid != local_pid) { // 来自其他进程的 fd
                close(local_fd); // 用完即关（物理内存由导入句柄持有）
            } else { // 本进程自己的导出 fd
                local_export_fd = local_fd; // 记录下来，析构时关闭
            }
        }

        // Enable access // 设置访问权限
        if (use_cpu_rdma_storage) {
            set_access(addr, num_sym_bytes, ctx.device_idx); // CPU-backed：主段只给 GPU 访问权
            set_access(addr + num_sym_bytes, num_rdma_storage_bytes * num_scaleup_ranks, // 全部分片段给 GPU 与 NUMA 节点访问权
                       ctx.device_idx, ctx.numa_idx); // 绑定设备所在 NUMA 节点
        } else {
            set_access(addr, num_mapped_bytes, ctx.device_idx); // 全 GPU-backed：整段 VA 都给 GPU 访问权
        }
    }

    ~HybridElasticSymmetricMemory() override {
        auto addr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr)); // 还原虚拟地址

        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, num_sym_bytes)); // 解除 GPU 主段映射
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(gpu_handle)); // 释放 GPU 主段物理分配

        if (local_export_fd >= 0) // 本进程导出过 fd 时
            close(local_export_fd); // 关闭导出 fd（对端已各自导入）
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr + num_sym_bytes, num_rdma_storage_bytes * num_scaleup_ranks)); // 解除全部分片段的映射
        for (int i = 0; i < num_scaleup_ranks; ++ i) // 逐个释放
            CUDA_DRIVER_CHECK(lazy_cuMemRelease(storage_handles[i])); // 各分片的物理分配句柄

        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(addr, num_sym_bytes + num_rdma_storage_bytes * num_scaleup_ranks)); // 释放整段预留的虚拟地址
    }

// 创建共享存储分片：供同节点其他进程导入
    // Create a CPU/GPU storage segment and export its POSIX FD handle.
    // Returns (pid, fd) handle for cross-process sharing.
    static shared_handle_t create_shared_handle(const int64_t& num_bytes, // 创建一个 CPU/GPU 存储段并导出 POSIX fd 句柄
                                                const bool& use_cpu_rdma_storage) { // 是否用主机内存
        EP_HOST_ASSERT(num_bytes > 0 and num_bytes % kNumAllocationAlignmentBytes == 0); // 大小为正且满足分配对齐

        DeviceContext ctx; // 查询设备上下文
        auto prop = use_cpu_rdma_storage ? ctx.cpu_alloc_prop(true) : ctx.gpu_alloc_prop(true); // CPU-backed 用 host NUMA 属性（跳过 fabric），否则 GPU 属性

        CUmemGenericAllocationHandle handle; // 物理分配句柄
        cumem_create_with_fallback(&handle, num_bytes, &prop); // 创建物理分配

        int fd = -1; // 导出的文件描述符
        CUDA_DRIVER_CHECK(lazy_cuMemExportToShareableHandle( // 导出为 POSIX fd 共享句柄
            &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)); // 供跨进程导入

        // Release the allocation handle -- the POSIX FD keeps the physical memory alive
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle)); // 释放分配句柄——物理内存由 POSIX fd 继续持有

        return {getpid(), fd}; // 返回 (pid, fd) 供其他 rank 导入
    }
// 对称内存分配入口：按是否需要 RDMA 存储、是否混合模式选择三种实现
};

static std::shared_ptr<SymmetricMemory> alloc(const int64_t& num_sym_bytes,
                                              const int64_t& num_rdma_storage_bytes, // RDMA 存储分片字节数
                                              const bool& use_cpu_rdma_storage, // 是否用主机内存做 RDMA 存储
                                              const bool& allow_hybrid_mode = false, // 是否允许混合模式（跨节点导入分片）
                                              const int& num_scaleup_ranks = 0, const int& scaleout_rank_idx = 0, // 机内 rank 数与本 rank 的跨机编号
                                              const shared_comm_t& shared_comm = {}) { // 各 rank 的 (pid, fd) 共享句柄
    EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0); // 对称段大小为正且满足分配对齐
    EP_HOST_ASSERT(num_rdma_storage_bytes >= 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0); // 分片大小非负且满足分配对齐

    std::shared_ptr<SymmetricMemory> result; // 结果对象
    if (num_rdma_storage_bytes > 0) { // 需要 RDMA 存储时分派到弹性/混合分配器
        if (allow_hybrid_mode) { // 混合模式
            result = std::make_shared<HybridElasticSymmetricMemory>( // 创建 HybridElasticSymmetricMemory（导入机内各 rank 分片）
                shared_comm, num_sym_bytes, num_rdma_storage_bytes, use_cpu_rdma_storage, // 传入共享句柄、两段大小与 CPU-backed 标志
                num_scaleup_ranks, scaleout_rank_idx); // 机内 rank 数与跨机编号
        } else { // 非混合但需要存储
            result = std::make_shared<ElasticSymmetricMemory>( // 创建 ElasticSymmetricMemory（本进程内两段拼接）
                num_sym_bytes, num_rdma_storage_bytes, use_cpu_rdma_storage); // 传入两段大小与 CPU-backed 标志
        }
    } else {
        result = std::make_shared<GPUSymmetricMemory>(num_sym_bytes); // 无存储需求：纯 GPU 对称内存
    }

    // TODO: move all variables into NCCL runtime
    // Enable NCCL elastic buffer register if the allocator may produce CPU-backed segments
    if (num_rdma_storage_bytes > 0 and use_cpu_rdma_storage) // 分配器可能产出 CPU-backed 段时
        setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0); // 开启 NCCL 弹性缓冲注册（不覆盖已有设置）
    return result;
}

}  // namespace deep_ep::symmetric
