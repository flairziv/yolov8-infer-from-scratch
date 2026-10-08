// cuda_api.h —— 设备内存、主机↔设备拷贝与同步的最小接口。
// 不带 CUDA 的构建链接 stub 实现：compiled()/available() 返回 false，其余函数直接抛异常。
#pragma once
#include <cstddef>

namespace yi::cuda {

bool compiled();     // 构建时是否带 CUDA 后端
bool available();    // compiled() 且有可用设备（首次调用会初始化设备）
const char* device_name();   // 设备名；不可用时返回 "unavailable"

void* alloc(size_t bytes);
void release(void* ptr);
void to_device(void* dst, const void* src, size_t bytes);
void to_host(void* dst, const void* src, size_t bytes);
void sync();

}  // namespace yi::cuda
