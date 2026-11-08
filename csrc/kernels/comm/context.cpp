#include <cstring> // std::memcpy
#include <vector> // std::vector
#include <string> // std::string
#include <pybind11/pytypes.h> // pybind11 基础类型（bytearray 等）
#include <pybind11/stl.h> // pybind11 的 STL 转换支持
#include <sstream> // std::stringstream

#include <nccl.h> // NCCL 主机端 API
#include <nccl_device/core.h> // NCCL 设备通信核心（ncclTeam、ncclDevComm 等）

#include <deep_ep/common/compiled.cuh> // 公共编译配置
#include <deep_ep/common/exception.cuh> // 异常/错误检查宏（NCCL_CHECK、EP_HOST_ASSERT 等）
#include <deep_ep/common/math.cuh> // 数学工具（advance_ptr 等）
#include <deep_ep/layout/bucket/workspace.cuh> // bucket 通信 workspace 布局

#include "api.hpp" // 本目录的接口声明（Context 类等）
#include "../../runtime/jit.hpp" // deep_jit：运行时 JIT 编译与 kernel 启动框架
#include "../../utils/system.hpp" // 系统工具（get_env 等）


namespace deep_ep::comm { // comm 命名空间：基于 NCCL 设备 API 的集合通信封装

pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id; // NCCL unique id 结构（跨进程组建通信器的凭证）
    NCCL_CHECK(ncclGetUniqueId(&unique_id)); // 获取本 rank 的 unique id
    std::vector<char> result(sizeof(ncclUniqueId)); // 按结构体大小分配字节缓冲
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId)); // 把 unique id 原样拷入缓冲
    return {result.data(), result.size()}; // 以 pybind11 bytearray 返回给 Python 侧
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx) {
    // Copy unique ID
    ncclUniqueId root_unique_id; // root rank 的 unique id
    const auto root_unique_id_str = root_unique_id_bytes.cast<std::string>(); // 把 Python 传入的字节数组转成字符串视图
    std::memcpy(&root_unique_id, root_unique_id_str.c_str(), sizeof(ncclUniqueId)); // 按字节拷入 unique id 结构体

    // Init
    ncclComm_t comm; // NCCL 主机通信器句柄
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx)); // 以 root 的 unique id 初始化本 rank 的通信器
    if (get_env<int>("EP_BUFFER_DEBUG")) // 调试开关打开时打印
        printf("New NCCL host communicator created (%d/%d)\n", rank_idx, num_ranks); // 打印新创建的通信器（rank/总数）
    return reinterpret_cast<int64_t>(comm); // 把通信器指针伪装成 int64 句柄返回
}

void destroy_nccl_comm(const int64_t& nccl_comm) {
    NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(nccl_comm))); // 中止并销毁通信器（Abort 可处理异常状态）
    if (get_env<int>("EP_BUFFER_DEBUG")) // 调试开关打开时打印
        printf("NCCL host communicator aborted\n"); // 打印通信器已中止
}

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm); // 还原通信器指针
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks; // 物理域：全局 rank 数与机内（LSA/NVLink 域）rank 数
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0); // 全局 rank 数必须能被机内 rank 数整除
    return {num_ranks / num_nvl_ranks, num_nvl_ranks}; // 返回（跨机 rank 数, 机内 rank 数）
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm); // 解构出物理域两层 rank 数
    return {allow_hybrid_mode ? num_rdma_ranks : 1, // 跨机维度：混合模式取真实 RDMA rank 数，否则压成 1
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks}; // 机内维度：混合模式取 NVLink rank 数，否则取全部 rank
}

std::tuple<int, int> get_gin_min_stride(const int64_t& nccl_comm) {
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm); // 还原通信器指针
    const int num_ranks = ncclTeamWorld(comm).nRanks; // 全局 rank 数

    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER; // 通信器属性结构体
    NCCL_CHECK(ncclCommQueryProperties(comm, &props)); // 查询通信器属性（含 ginMinStride）
    EP_HOST_ASSERT(num_ranks % props.ginMinStride == 0); // 全局 rank 数必须能被 GIN 最小步长整除
    return {props.ginMinStride, num_ranks / props.ginMinStride}; // 返回（gin_min_stride, 每 stride 组的 rank 数）
}

// Context 构造函数：复用主机通信器创建设备通信器（含 GIN 配置），分配/共享对称内存并注册 NCCL 窗口
Context::Context(const int64_t& nccl_comm, const symmetric::shared_comm_t& shared_comm,
                 const int& num_ranks, const int& rank_idx,
                 const int64_t& num_workspace_bytes,
                 const int64_t& num_gpu_buffer_bytes,
                 const int64_t& num_rdma_storage_bytes,
                 const bool& use_cpu_rdma_storage,
                 const bool& allow_hybrid_mode,
                 const std::optional<int>& sl_idx, const int& num_allocated_qps, const int& qp_depth,
                 const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
                 const bool& enable_lsa_multimem,
                 const std::shared_ptr<Context>& main_context):
    rank_idx(rank_idx), num_ranks(num_ranks),
    gin_min_stride(1), // 初始化本 rank 编号与总 rank 数
    num_allocated_qps(num_allocated_qps), // GIN 最小步长默认为 1（稍后从属性查询覆盖）
    num_cpu_timeout_secs(num_cpu_timeout_secs), // 记录已分配 QP 数
    num_workspace_bytes(num_workspace_bytes), // 记录 CPU 侧超时（秒）
    num_gpu_buffer_bytes(num_gpu_buffer_bytes), // 记录 workspace 字节数
    num_rdma_storage_bytes(num_rdma_storage_bytes), // 记录 GPU 缓冲字节数
    use_cpu_rdma_storage(use_cpu_rdma_storage) { // 记录 RDMA 存储分片字节数
    EP_HOST_ASSERT(num_ranks > 0 and num_ranks <= kNumMaxRanks); // 记录是否用主机内存做 RDMA 存储
    EP_HOST_ASSERT(num_allocated_qps > 0 and num_allocated_qps <= kNumMaxQPs); // 总 rank 数必须在支持范围内
    EP_HOST_ASSERT(num_workspace_bytes > 0 and num_workspace_bytes % kNumAllocationAlignmentBytes == 0); // QP 数必须在 1 与上限之间
    EP_HOST_ASSERT(num_gpu_buffer_bytes >= 0 and num_gpu_buffer_bytes % kNumAllocationAlignmentBytes == 0); // workspace 大小为正且满足分配对齐
    EP_HOST_ASSERT(num_rdma_storage_bytes >= 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0); // GPU 缓冲大小非负且满足分配对齐
 // RDMA 存储分片大小非负且满足分配对齐
    num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs) * jit->device.get_clock_rate(); // GPU 超时（秒）换算成时钟周期数（自旋等待用）
    int nccl_runtime_version; // NCCL 运行时版本号
    NCCL_CHECK(ncclGetVersion(&nccl_runtime_version)); // 查询实际加载的 NCCL 版本
    if (get_env("EP_BUFFER_DEBUG", 0)) { // 调试开关打开时打印
        printf("DeepEP initialized with NCCL version: %d.%d.%d (loaded library)\n", // 打印 NCCL 版本（主.次.补丁）
               nccl_runtime_version / 10000, (nccl_runtime_version % 10000) / 100, nccl_runtime_version % 100); // 从版本号整数中拆出三段
    }

    // Reuse the NCCL communicator
    comm = reinterpret_cast<ncclComm_t>(nccl_comm); // 复用外部传入的 NCCL 主机通信器

    // Print number of allocated QPs
    if (get_env<int>("EP_BUFFER_DEBUG")) // 调试开关打开时打印
        printf("EP NCCL device communicator has %d allocated QPs\n", num_allocated_qps); // 打印分配的 QP 数量

    // Query props
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER; // 通信器属性结构体（查询 multimem/GIN 支持情况）
    NCCL_CHECK(ncclCommQueryProperties(comm, &props)); // 查询通信器属性
    EP_HOST_ASSERT((not enable_lsa_multimem or props.multimemSupport) and // 启用 multimem 时设备必须支持（NVLink 组播归约）
                   "NCCL multimem is unavailable");

    // Initialize NCCL device communicator
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER; // 设备通信器需求结构体（要求 NCCL 分配的资源）
    reqs.lsaMultimem = enable_lsa_multimem; // 是否需要 LSA multimem 支持
    if (get_env("EP_DISABLE_GIN", 0) == 0) { // GIN 未被环境变量禁用时配置 GIN（RDMA kernel 直发）
        EP_HOST_ASSERT( // 校验 GIN 类型可用
            (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and // 混合模式看 railedGinType，否则看 ginType
            "NCCL GIN is unavailable. This is usually due to a network configuration issue, " // 报错信息（前半）：GIN 不可用通常与网络配置有关
            "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network."); // 报错信息（后半）：如多平面网络下关闭了直发 RDMA kernel
        EP_HOST_ASSERT( // 校验 GDAKI（GIN 的 GDRCopy/AKI 后端）可用
            props.ginSupport[NCCL_GIN_TYPE_GDAKI] and // 该通信器必须支持 GDAKI 类型
            "NCCL GDAKI is unavailable for this communicator."); // 报错信息：GDAKI 不可用

        gin_min_stride = props.ginMinStride; // 记录 GIN 最小步长（RDMA 存储分片按它分组）
        reqs.ginType = NCCL_GIN_TYPE_GDAKI; // 使用 GDAKI 类型（经 GDRCopy 的 kernel 级 RDMA）
        reqs.ginContextCount = num_allocated_qps; // 每个 QP 一个 GIN 上下文
        reqs.ginExclusiveContexts = true; // 独占上下文：避免与其他集合通信竞争
        reqs.ginQueueDepth = qp_depth > 0 ? qp_depth : kDefaultQPDepth; // QP 深度：未指定时用默认值

        // Customized RDMA barrier needs extra signals
        reqs.ginSignalCount = num_ranks + 2 * 2; // 自定义 RDMA barrier 需要额外信号（每 rank 一个 + 2 组备用）
        if (allow_hybrid_mode and num_rdma_storage_bytes > 0) { // 混合模式且有 RDMA 存储时使用自定义步长连接
            reqs.ginCustomStride = gin_min_stride; // GIN 自定义步长 = gin_min_stride
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE; // 连接类型：自定义步长（支持分片直发）
        } else {
            reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL; // 连接类型：混合模式用 RAIL（多轨），否则用 FULL（全连接）
        }

        // SL index
        if (std::getenv("EP_DEFAULT_RDMA_SL") != nullptr) // 环境变量指定默认 RDMA 服务等级（SL）
            reqs.ginTrafficClass = get_env<int>("EP_DEFAULT_RDMA_SL"); // 设置流量类别为该 SL
        if (sl_idx.has_value()) // 显式传入的 SL 优先
            reqs.ginTrafficClass = sl_idx.value(); // 覆盖流量类别
        if (std::getenv("EP_OVERRIDE_RDMA_SL") != nullptr) // 环境变量可最终覆盖 SL
            reqs.ginTrafficClass = get_env<int>("EP_OVERRIDE_RDMA_SL"); // 覆盖流量类别
    }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    reqs.useRuntimeVersion = true; // 要求 NCCL 按运行时版本语义填充设备通信器
    dev_comm.ptr = malloc(props.devCommRuntimeVersionSize); // 按 NCCL 报告的大小分配设备通信器存储
#else
    EP_HOST_ASSERT(NCCL_VERSION_CODE == nccl_runtime_version and "Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. Please re-compile DeepEP."); // NCCL 2.31 之前编译期与运行时版本必须一致
    dev_comm.ptr = malloc(sizeof(ncclDevComm_t)); // 按结构体大小分配设备通信器存储
#endif
    EP_HOST_ASSERT(dev_comm.ptr != nullptr); // 分配必须成功
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, static_cast<ncclDevComm_t*>(dev_comm.ptr))); // 创建 NCCL 设备通信器（kernel 内经它使用 GIN/multimem）

    // Now we know the NVLink domain size
    ncclTeam_t lsaTeam = ncclTeamLsa(comm); // 取机内（LSA）team：同一 NVLink 域内的 rank 集合
    num_nvl_ranks = lsaTeam.nRanks, nvl_rank_idx = lsaTeam.rank; // 记录机内 rank 数与本 rank 的机内编号
    num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks; // 推算跨机 rank 数与本 rank 的跨机编号
    EP_HOST_ASSERT(num_rdma_ranks <= kNumMaxRDMARanks and num_nvl_ranks <= kNumMaxNVLRanks); // 两层域规模都不得超过上限
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0 and nvl_rank_idx == rank_idx % num_nvl_ranks); // 总 rank 数必须能被机内 rank 数整除且编号一致
    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx); // 全局 rank 编号 = 跨机编号 × 机内 rank 数 + 机内编号

    // Calculate scaleout/up domain size
    if (allow_hybrid_mode) { // 混合模式：两层逻辑域即物理域
        num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks; // scale-out（跨机）= RDMA 域；scale-up（机内）= NVLink 域
        scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx; // 对应的两层编号
    } else { // 非混合模式：压缩为单层
        num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks; // scale-out 视为 1，全部 rank 都在 scale-up 域
        scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx; // scale-out 编号为 0，scale-up 编号即全局编号
    }
    EP_HOST_ASSERT(num_scaleout_ranks <= kNumMaxScaleoutRanks and num_scaleup_ranks <= kNumMaxScaleupRanks); // 两层逻辑域规模都不得超过上限
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks; // scale-up 域等价于 NVLink 域时置真（kernel 走 NVLink 路径）

    if (main_context == nullptr) {
        symmetric_memory = symmetric::alloc( // 主上下文：为本上下文分配新的对称内存
            num_workspace_bytes + num_gpu_buffer_bytes, num_rdma_storage_bytes, // workspace + GPU 缓冲合为一个对称段，另加 RDMA 存储分片
            use_cpu_rdma_storage, // 是否用主机内存做 RDMA 存储
            allow_hybrid_mode, gin_min_stride, rank_idx / gin_min_stride, // 混合模式、GIN 步长与本 rank 的分组索引
            shared_comm); // 各 rank 导出的 (pid, fd) 共享句柄
    } else {
        symmetric_memory = main_context->symmetric_memory; // 子上下文：直接共享主上下文的对称内存（不重复分配）
    }

    // Create window
    // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
    // across all ranks, so no explicit barrier is needed after this call.
    raw_window_ptr = this->symmetric_memory->ptr; // 窗口基址 = 对称内存起始
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, symmetric_memory->num_sym_bytes, &window, NCCL_WIN_STRICT_ORDERING)); // 把对称内存注册为 NCCL 窗口（严格序）；内部自带跨 rank barrier
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &workspace)); // 取窗口偏移 0（workspace 起点）在本 rank 的设备指针
    buffer = math::advance_ptr(workspace, num_workspace_bytes); // 数据缓冲区紧随 workspace 之后
    if (enable_lsa_multimem) // 启用 multimem 时
        NCCL_CHECK(ncclGetLsaMultimemDevicePointer(window, num_workspace_bytes, &buffer_multimem)); // 取 multimem 视角的缓冲区地址（组播归约用）
    if (main_context == nullptr) // 主上下文负责初始化
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, num_workspace_bytes)); // workspace 清零（信号量与统计区）

    if (num_rdma_storage_bytes > 0) { // 有 RDMA 存储分片时注册专用窗口
        auto* storage_base_ptr = math::advance_ptr(symmetric_memory->ptr, symmetric_memory->num_sym_bytes); // 存储段紧跟主对称段之后
        local_rdma_storage_ptr = math::advance_ptr(storage_base_ptr, (rank_idx % gin_min_stride) * num_rdma_storage_bytes); // 本 rank 的存储分片指针（按 gin_min_stride 取模分组）

        rdma_storage_windows.resize(gin_min_stride, nullptr); // 为每组分片准备一个窗口槽位
        for (int i = 0; i < gin_min_stride; ++ i) { // 遍历每个分片组
            auto* storage_ptr = math::advance_ptr(storage_base_ptr, i * num_rdma_storage_bytes); // 第 i 组分片的起始地址
            NCCL_CHECK(ncclCommWindowRegister( // 把每组分片注册为 GIN 专用窗口（仅 RDMA kernel 可访问）
                comm, storage_ptr, num_rdma_storage_bytes, // 窗口覆盖该组分片的整段内存
                &rdma_storage_windows[i], NCCL_WIN_GIN_ONLY)); // GIN_ONLY 标志：普通集合通信不可见
        }

        CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&rdma_storage_windows_gpu), sizeof(ncclWindow_t) * gin_min_stride)); // 在设备侧分配窗口数组
        CUDA_RUNTIME_CHECK(cudaMemcpy( // 把窗口数组拷到 GPU
            rdma_storage_windows_gpu, rdma_storage_windows.data(), // 源为主机侧窗口列表
            sizeof(ncclWindow_t) * gin_min_stride, cudaMemcpyHostToDevice)); // 大小为 gin_min_stride 个窗口
    }
}

// 取指定 NVLink 对端视角下的对称地址：ptr 必须位于本窗口内，返回对端同偏移指针（NVLink 域内互访的基础）
void* Context::get_sym_ptr(void* ptr, const int& dst_nvl_rank_idx) const {
    EP_HOST_ASSERT(dst_nvl_rank_idx >= 0 and dst_nvl_rank_idx < num_nvl_ranks); // 校验目标机内 rank 编号合法
    void* dst_ptr = nullptr; // 对端指针占位
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(workspace); // 计算 ptr 相对 workspace 起点的偏移
    NCCL_CHECK(ncclGetLsaDevicePointer(window, offset, dst_nvl_rank_idx, &dst_ptr)); // 按窗口与偏移取对端同位置的设备指针
    return dst_ptr; // 返回对端对称地址
}

std::tuple<int, int> Context::get_physical_domain_size() const {
    EP_HOST_ASSERT(not finalized); // finalize 后不可再查询
    return {num_rdma_ranks, num_nvl_ranks}; // 返回物理域（RDMA rank 数, NVLink rank 数）
}

std::tuple<int, int> Context::get_logical_domain_size() const {
    EP_HOST_ASSERT(not finalized); // finalize 后不可再查询
    return {num_scaleout_ranks, num_scaleup_ranks}; // 返回逻辑域（scale-out rank 数, scale-up rank 数）
}

// 释放全部资源：注销并释放窗口、设备通信器；之后本上下文不可再用
void Context::finalize() {
    EP_HOST_ASSERT(not finalized); // 不允许重复 finalize
    if (rdma_storage_windows_gpu != nullptr) { // 释放设备侧窗口数组
        CUDA_RUNTIME_CHECK(cudaFree(rdma_storage_windows_gpu)); // 释放 GPU 内存
        rdma_storage_windows_gpu = nullptr; // 指针置空防重复释放
    }
    for (const auto storage_window: rdma_storage_windows) // 注销每个 RDMA 存储分片窗口
        NCCL_CHECK(ncclCommWindowDeregister(comm, storage_window)); // 逐个注销
    rdma_storage_windows.clear(); // 清空窗口列表
    local_rdma_storage_ptr = nullptr; // 本地存储分片指针置空

    // Deregister window
    NCCL_CHECK(ncclCommWindowDeregister(comm, window)); // 注销主对称内存窗口
    symmetric_memory.reset(); // 释放对称内存（引用计数归零时触发析构）

    // Barrier can no longer be used
    barrier_signals = nullptr; // barrier 信号量随窗口失效，置空防止误用

    // Destroy device communicator
    NCCL_CHECK(ncclDevCommDestroy(comm, static_cast<ncclDevComm_t*>(dev_comm.ptr))); // 销毁 NCCL 设备通信器
    free(dev_comm.ptr); // 释放设备通信器的宿主存储
    finalized = true; // 标记已终结
}

}  // namespace deep_ep::comm
