#pragma once // 头文件保护：只被包含一次

#include <deep_ep/common/compiled.cuh> // 公共编译配置（nvcc 编译选项、架构宏等）

// EP（专家并行）kernel 层的统一入口：下面四个头文件分别是
#include "dispatch.hpp" // dispatch：token 分发（dispatch + 拷贝 epilogue）
#include "combine.hpp" // combine：token 组合/归约（combine + 归约 epilogue）
#include "prefetch_weights.hpp" // prefetch_weights：EPLB 权重预取 kernel 启动
#include "reduce_grads.hpp" // reduce_grads：EPLB 冗余专家梯度归约 kernel 启动
