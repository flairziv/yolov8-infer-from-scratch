// cuda_api.cu —— 设备初始化、设备内存与拷贝。所有 CUDA 错误都转成异常，不静默继续。
#include "cuda/cuda_api.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#include "common.h"

namespace yi::cuda {
namespace {

// 首次使用时初始化设备；失败过一次就不再重试（例如没有设备时反复调用没意义）。
bool init_once(bool& ready, std::string& why) {
    if (ready) return true;
    if (!why.empty()) return false;
    const cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        why = cudaGetErrorString(err);
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        why = "cudaGetDeviceProperties 失败";
        return false;
    }
    ready = true;
    return true;
}

std::string g_error;
bool g_ready = false;

void check(cudaError_t err, const char* what) {
    YI_CHECK(err == cudaSuccess, what << " 失败: " << cudaGetErrorString(err));
}

}  // namespace

bool compiled() { return true; }

bool available() { return init_once(g_ready, g_error); }

const char* device_name() {
    if (!available()) return "unavailable";
    static std::string name = [] {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        return std::string(prop.name);
    }();
    return name.c_str();
}

void* alloc(size_t bytes) {
    YI_CHECK(available(), "没有可用的 CUDA 设备");
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

void release(void* ptr) {
    if (ptr) check(cudaFree(ptr), "cudaFree");
}

void to_device(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
}

void to_host(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
}

void sync() { check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

}  // namespace yi::cuda
