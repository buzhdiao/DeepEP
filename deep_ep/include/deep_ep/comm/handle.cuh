#pragma once  // 头文件保护，防止被重复包含

#include <nccl.h>  // NCCL 主头文件
#include <nccl_device.h>  // NCCL 设备端 API（ncclDevComm_t、窗口、Gin 等）

#include <deep_ep/common/exception.cuh>  // 断言与异常宏
#include <deep_ep/common/ptx.cuh>  // PTX 指令封装

// deep_ep::comm 命名空间：基于 NCCL 设备端 API 的通信封装
namespace deep_ep::comm {

struct NCCLGin {  // 封装 NCCL Gin（GPU Initiated Networking：GPU 发起的 RDMA/多播通信接口）
#define IS_TEAM_WORLD(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld>) { code }  // 仅当模板参数 team 为 world 团队时执行 code
#define IS_TEAM_LSA(code) if constexpr (std::is_same_v<team_t, ncclTeamTagLsa>) { code }  // 仅当 team 为 LSA（NVLink 域）团队时执行 code
#define IS_TEAM_RAIL(code) if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) { code }  // 仅当 team 为 rail（同机）团队时执行 code
#define IS_TEAM_WORLD_RAIL(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld> or std::is_same_v<team_t, ncclTeamTagRail>) { code }  // team 为 world 或 rail 时执行 code
#define IS_TEAM_WORLD_LSA(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld> or std::is_same_v<team_t, ncclTeamTagLsa>) { code }  // team 为 world 或 LSA 时执行 code
#define TEAM_WORLD_RAIL() ((std::is_same_v<team_t, ncclTeamTagWorld>) ? team_world : team_rail)  // 按模板参数选择 world 或 rail 团队对象

    const ncclDevComm_t& nccl_dev_comm;  // NCCL 设备端通信对象引用
    const ncclWindow_t& nccl_window;  // 对称内存窗口引用
    ncclGin gin;  // Gin 通信句柄（绑定 QP 等资源）
    ncclTeam team_world, team_lsa, team_rail;  // 三种团队视图：全局 world、节点内 LSA（NVLink）、同机 rail
    uint64_t lsa_base_ptr;  // 本 rank 窗口在 LSA 视图下的基地址（用于对称指针换算）

    // TODO(NCCL): QP index should just be a hint or the users maintain the mapping?  // 保留原注释
    __device__ __forceinline__  // 设备端构造函数
    NCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
            const int& qp_idx = 0,  // 使用的 QP 编号，默认 0
            const ncclGinResourceSharingMode& resource_sharing_mode = NCCL_GIN_RESOURCE_SHARING_GPU):  // Gin 资源共享模式，默认按 GPU 共享
        nccl_dev_comm(nccl_dev_comm), nccl_window(nccl_window),  // 初始化引用成员
        gin(ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode)),  // 用指定 QP 与共享模式创建 Gin 句柄
        team_world(ncclTeamWorld(nccl_dev_comm)), team_lsa(ncclTeamLsa(nccl_dev_comm)), team_rail(ncclTeamRail(nccl_dev_comm)) {  // 获取三种团队视图
        // TODO: what if we only have 1 NVLink rank  // 保留原注释
        lsa_base_ptr = reinterpret_cast<uint64_t>(ncclGetLsaPointer(nccl_window, 0, team_lsa.rank));  // 记录本 rank 在窗口偏移 0 处的 LSA 指针作为基准
    }

    template <typename team_t>
    __device__ __forceinline__ bool is_nvlink_accessible(const int& dst_rank_idx) const {  // 判断目标 rank 是否可经 NVLink（对称指针）直接访问
        IS_TEAM_LSA({  // LSA 团队视图：节点内全部可达
            return true;  // 恒为真
        })

        IS_TEAM_WORLD({  // world 团队视图：按 rail × LSA 的拓扑推算
            // TODO(NCCL): optimize this function's cycles  // 保留原注释
            // return ncclTeamRankIsMember(team_lsa, team_world, dst_rank_idx);  // 保留原注释：备用的官方实现
            return team_rail.rank * team_lsa.nRanks <= dst_rank_idx and  // 目标 rank 落在本 rail 的 rank 区间内
                   dst_rank_idx < (team_rail.rank + 1) * team_lsa.nRanks;  // （下界含、上界不含）即与本 rank 同机
        })

        IS_TEAM_RAIL({  // rail 团队视图：只有本机自身
            // TODO(NCCL): some ranks may be connected by NVLink, e.g., "2 + 2 + 4"  // 保留原注释：跨 rail NVLink 场景未支持
            return team_rail.rank == dst_rank_idx;  // 仅当目标就是本 rank
        })
    }

    // ReSharper disable once CppNotAllPathsReturnValue  // 保留原注释：抑制 IDE 警告
    template <typename dtype_t = void*>
    __device__ __forceinline__  // 计算指针在其窗口内的对称偏移
    uint64_t get_sym_offset(dtype_t* ptr) const {
        return reinterpret_cast<uint64_t>(ptr) - lsa_base_ptr;  // 指针减去 LSA 基地址即窗口内偏移
    }

    // ReSharper disable once CppNotAllPathsReturnValue  // 保留原注释
    template <typename team_t, typename dtype_t = void*>
    __device__ __forceinline__  // 将本 rank 的对称指针翻译为远端 rank 的对应指针
    dtype_t* get_sym_ptr(dtype_t* ptr, const int& dst_rank_idx) const {
        IS_TEAM_RAIL({  // rail 视图：仅本 rank 可达
            return team_rail.rank == dst_rank_idx ? ptr : nullptr;  // 目标是自身返回原指针，否则返回空
        })

        IS_TEAM_WORLD_LSA({  // world 或 LSA 视图
            constexpr bool kIsTeamLSA = (std::is_same_v<team_t, ncclTeamTagLsa>);  // 标记是否为 LSA 视图

            // Team world and not accessible by symmetric pointers  // world 视图下目标不可经 NVLink 访问
            if (not is_nvlink_accessible<team_t>(dst_rank_idx))  // 不可达检查
                return nullptr;  // 返回空指针表示需走 RDMA 路径

            // Translate into NVLink rank index  // 将 rank 编号换算为 NVLink（LSA）域内编号
            const auto dst_nvl_rank_idx = kIsTeamLSA ?  // LSA 视图下本身即 NVLink 编号
                dst_rank_idx : (dst_rank_idx - team_rail.rank * team_lsa.nRanks);  // world 视图需减去本 rail 的起始偏移

            // Local rank bypass  // 目标是本 rank 时直接短路返回
            // TODO(NCCL): support this  // 保留原注释
            if (dst_nvl_rank_idx == team_lsa.rank)  // 本机判断
                return ptr;  // 返回原指针

            // Get base ptr  // 取远端指针
            const auto dst_ptr = ncclGetLsaPointer(  // 按对称偏移在远端 NVLink rank 上取对应指针
                nccl_window, get_sym_offset(ptr), dst_nvl_rank_idx);
            return static_cast<dtype_t*>(dst_ptr);  // 转回目标类型
        });
    }

    // NOTES: take care of this function when `team_t` is not LSA  // 保留原注释：team 非 LSA 时需注意
    // Do not mix atomic add with gin signal into a single position  // 保留原注释：不要把原子加与 gin 信号混入同一位置
    template <typename team_t, typename dtype_t>
    __device__ __forceinline__  // 远端原子加（release 语义）
    void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,
                     const int& extra_options = 0) const {
        const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);  // 先尝试取远端对称指针
        // Use symmetric pointers as much as possible, RDMA otherwise  // 优先走对称指针（NVLink），否则退回 RDMA 信号
        if (dst_ptr != nullptr) {  // 对称指针可用
            if (std::is_same_v<team_t, ncclTeamTagRail> or dst_ptr == sym_ptr) {  // rail 视图或目标即本 rank：GPU 作用域即可
                ptx::red_add_rel_gpu(dst_ptr, value);  // GPU 作用域 release 原子加
            } else {  // 跨 GPU（仍在 LSA/world 域内）
                ptx::red_add_rel_sys(dst_ptr, value);  // 系统作用域 release 原子加
            }
        } else {  // 对称指针不可用：走 Gin 信号（RDMA 原子加）
            EP_DEVICE_ASSERT((not std::is_same_v<team_t, ncclTeamTagLsa>));  // LSA 视图必须可达，不允许走此路径
            EP_DEVICE_ASSERT((std::is_same_v<dtype_t, int64_t>) or (std::is_same_v<dtype_t, uint64_t>));  // 信号路径仅支持 64 位整型
            // TODO(NCCL): support all dtypes  // 保留原注释
            gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx,  // 向目标 rank 发送信号
                       ncclGin_VASignalAdd(nccl_window, reinterpret_cast<int64_t>(sym_ptr) - lsa_base_ptr, static_cast<uint64_t>(value)),  // 动作：对目标窗口偏移处的值做原子加
                       ncclCoopThread(),  // 协作粒度：单线程
                       ncclGin_None(),  // 无附加动作
                       cuda::thread_scope_thread,  // 本地线程作用域
                       cuda::thread_scope_device,  // 远端设备作用域
                       ncclGinOptFlagsDefault | extra_options);  // 默认选项附加调用者选项
        }
    }

    __device__ __forceinline__  // 等待一个 Gin 请求完成
    void wait(ncclGinRequest_t& request,
              const cuda::memory_order order = cuda::memory_order_acquire) const {  // 默认 acquire 内存序
        gin.wait(request, ncclCoopThread(), ncclGin_None(), order);  // 阻塞等待请求完成
    }

    template <typename team_t, typename coop_t = ncclCoopThread, typename segment_t = ncclGin_SegmentDevice>
    __device__ __forceinline__  // 远程读取：从远端窗口拷贝数据到本地窗口
    void get(const ncclWindow_t& remote_window, const int64_t& remote_offset,  // 远端窗口与偏移
             const ncclWindow_t& local_window, const int64_t& local_offset,  // 本地窗口与偏移
             const int64_t& num_bytes, const int& src_rank_idx,  // 拷贝字节数与源 rank
             const int& extra_options = 0) const {  // 额外选项
        IS_TEAM_WORLD_RAIL({  // world/rail 视图下才支持 get
            gin.get(  // 发起 RDMA 读
                TEAM_WORLD_RAIL(),  // 按模板参数选择团队
                src_rank_idx,  // 数据源 rank
                remote_window, remote_offset,  // 远端窗口与偏移
                local_window, local_offset,  // 本地窗口与偏移
                num_bytes,  // 拷贝字节数
                coop_t(),  // 协作粒度
                ncclGin_None(),  // 无附加动作
                ncclGinOptFlagsDefault | extra_options,  // 选项
                segment_t()  // 分段策略（默认整段设备级）
            );
        });
    }

    template <typename coop_t = ncclCoopThread>
    __device__ __forceinline__  // 刷新 Gin 队列，确保此前的远程操作全部发出
    void flush() const {
        gin.flush(coop_t());  // 同步 flush
    }

    template <typename team_t, typename coop_t = ncclCoopThread>
    __device__ __forceinline__  // 异步刷新 Gin 队列，通过请求对象稍后等待
    void flush_async(const int& src_rank_idx, ncclGinRequest_t* request,
                     const int& extra_options = 0) const {
        IS_TEAM_WORLD_RAIL({  // world/rail 视图
            gin.flushAsync(  // 发起异步 flush
                TEAM_WORLD_RAIL(),  // 团队
                src_rank_idx,  // 源 rank
                request,  // 输出的请求对象，供后续 wait
                coop_t(),  // 协作粒度
                ncclGinOptFlagsDefault | extra_options  // 选项
            );
        });
    }

    template <typename team_t, typename remote_action_t>
    __device__ __forceinline__  // 向远端发送信号（携带远程动作，如信号量自增）
    void signal(const int& dst_rank_idx, const remote_action_t& remote_action) const {
        IS_TEAM_WORLD_RAIL({  // world/rail 视图
            gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx, remote_action);  // 直接转发给 Gin
        });
    }

    template <typename team_t, int64_t kNumTimeoutCycles>
    __device__ __forceinline__  // 等待目标 rank 的信号到达（带超时检测）
    void wait_signal(const int& signal_idx, const int& rank_idx) const {
        IS_TEAM_WORLD_RAIL({  // world/rail 视图
            // NOTES: signals are rolling counters, whose width follows the shadow counter type.
            // 注：信号是滚动计数器，位宽与影子计数器类型一致
            const auto shadow_ptr = gin.getSignalShadowPtr(signal_idx);  // 取本 rank 对应信号槽的影子计数指针
            const auto target = ++(*shadow_ptr);  // 先自增影子计数，得到期望到达的目标值
            const auto result = gin.waitSignal(  // 阻塞等待远端信号计数到达 target
                ncclCoopThread(), signal_idx, target, sizeof(*shadow_ptr) * 8,  // 信号槽编号、目标值、计数位宽
                cuda::memory_order_acquire, kNumTimeoutCycles);  // acquire 内存序与超时周期数
            if (result != ncclSuccess) {  // 等待超时或失败
                const auto signal = gin.readSignal(signal_idx, sizeof(*shadow_ptr) * 8, cuda::memory_order_acquire);  // 读取当前信号计数用于诊断
                // This waiting thread owns the shadow entry; reload only for diagnostics
                // to avoid keeping target live into the timeout path.
                // 注：本线程拥有该影子计数项，这里仅为诊断重新读取，避免把 target 保活到超时路径
                const uint64_t timeout_target = *static_cast<const volatile uint64_t*>(shadow_ptr);  // volatile 读取影子计数
                printf("DeepEP Gin signal timeout, rank: %d, signal_idx: %d, signal: %lu, target: %lu\n",  // 打印诊断信息
                       rank_idx, signal_idx, signal, timeout_target);
                ptx::trap();  // 触发致命陷阱终止 kernel
            }
        });
    }

    // TODO: can we always use a single window?  // 保留原注释
    template <typename team_t,
              typename remote_action_t = ncclGin_None>
    __device__ __forceinline__  // put（窗口版）：把本地窗口内数据写到远端 rank 的对称内存
    void put(void* recv_sym_ptr, const ncclWindow_t& send_window, const void* send_window_base_ptr, const void* send_ptr,
             const int64_t& num_bytes, const int& dst_rank_idx, const int& extra_options = 0,
             const remote_action_t& remote_action = remote_action_t()) const {
        IS_TEAM_WORLD_RAIL({  // world/rail 视图
            gin.put(TEAM_WORLD_RAIL(), dst_rank_idx,  // 向目标 rank 发起远程写
                    nccl_window, reinterpret_cast<int64_t>(recv_sym_ptr) - lsa_base_ptr,  // 接收端：对称窗口与本 rank 窗口内的偏移
                    send_window, reinterpret_cast<int64_t>(send_ptr) - reinterpret_cast<int64_t>(send_window_base_ptr),  // 发送端：发送窗口与数据相对窗口基址的偏移
                    num_bytes, remote_action, ncclGin_None(), ncclCoopThread(), ncclGin_None(),  // 字节数、远程动作、本地/远端附加动作与协作粒度
                    cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault | extra_options);  // 本地/远端内存作用域与选项
        });
    }

    template <typename team_t,
              typename remote_action_t = ncclGin_None>
    __device__ __forceinline__  // put（对称指针版）：直接用对称指针完成远程写
    void put(void* recv_sym_ptr, void* send_sym_ptr, const int64_t& num_bytes, const int& dst_rank_idx,
             const int& extra_options = 0,
             const remote_action_t& remote_action = remote_action_t()) const {
        // NOTES: local or NVLink put will also go through NIC via this API.
        // 注：即使是本机或 NVLink 范围的 put，也会经由该 API 走网卡路径
        put<team_t>(recv_sym_ptr, nccl_window, reinterpret_cast<void*>(lsa_base_ptr), send_sym_ptr,  // 以对称窗口与本 rank 基址调用窗口版 put
                    num_bytes, dst_rank_idx, extra_options, remote_action);
    }

    template <typename team_t, typename dtype_t>
    __device__ __forceinline__  // 向远端写入单个值（对称指针优先，否则 Gin putValue）
    void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,
                   const int& extra_options = 0) const {
        const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);  // 尝试取远端对称指针
        if (dst_ptr != nullptr) {  // 可经 NVLink 直接访问
            ptx::st_relaxed_sys(dst_ptr, value);  // 系统作用域弱内存序直接写
        } else {  // 不可达：走 Gin 路径
            EP_DEVICE_ASSERT((not std::is_same_v<team_t, ncclTeamTagLsa>));  // LSA 视图必须可达
            gin.putValue(TEAM_WORLD_RAIL(),  // 远程单值写
                         dst_rank_idx,  // 目标 rank
                         nccl_window, reinterpret_cast<int64_t>(sym_ptr) - lsa_base_ptr,  // 对称窗口与窗口内偏移
                         value,  // 写入的值
                         ncclGin_None(),  // 无附加动作
                         ncclCoopThread(),  // 协作粒度：单线程
                         ncclGin_None(),  // 本地无附加动作
                         cuda::thread_scope_thread,  // 本地线程作用域
                         cuda::thread_scope_device,  // 远端设备作用域
                         ncclGinOptFlagsDefault | extra_options);  // 选项
        }
    }

#undef IS_TEAM_WORLD  // 取消定义团队判断宏，避免泄漏
#undef IS_TEAM_LSA  // 同上
#undef IS_TEAM_RAIL  // 同上
#undef IS_TEAM_WORLD_RAIL  // 同上
#undef IS_TEAM_WORLD_LSA  // 同上
#undef TEAM_WORLD_RAIL  // 同上
};

}  // namespace deep_ep::comm
