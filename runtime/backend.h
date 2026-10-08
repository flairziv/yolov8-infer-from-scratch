// backend.h —— 后端选择与 CPU/OS 能力检测；本文件不要求 SIMD 指令集。
#pragma once
#include <string>

namespace yi {

enum class Backend { Scalar, SSE, AVX2, CUDA };

Backend parse_backend(const std::string& name);
const char* backend_name(Backend backend);
bool backend_available(Backend backend);
void require_backend(Backend backend);

}  // namespace yi
