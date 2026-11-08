#pragma once

// 引入通信流的获取与流间等待等工具函数
#include "stream.hpp"
// 引入 GPU 端屏障内核的启动封装
#include "barrier.hpp"
// 引入 NCCL 通信上下文 Context 及其对应的 Python 绑定注册函数
#include "handle.hpp"
