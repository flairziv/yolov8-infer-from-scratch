// cuda_api_stub.cpp —— 不带 CUDA 的构建：后端明确报告不可用，不假装支持。
#include "cuda/cuda_api.h"

#include "common.h"

namespace yi::cuda {

bool compiled() { return false; }
bool available() { return false; }
const char* device_name() { return "unavailable"; }

void* alloc(size_t) {
    YI_CHECK(false, "这次构建没有启用 CUDA（需要 -DYI_ENABLE_CUDA=ON 和 CUDA 工具链）");
    return nullptr;
}
void release(void*) {}
void to_device(void*, const void*, size_t) { YI_CHECK(false, "这次构建没有启用 CUDA"); }
void to_host(void*, const void*, size_t) { YI_CHECK(false, "这次构建没有启用 CUDA"); }
void sync() {}

}  // namespace yi::cuda
