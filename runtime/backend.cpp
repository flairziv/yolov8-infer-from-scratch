// backend.cpp —— 不能以 AVX2 编译，否则能力检查之前就可能执行非法指令。
#include "backend.h"

#include "common.h"

namespace yi {

Backend parse_backend(const std::string& name) {
    if (name == "scalar") return Backend::Scalar;
    if (name == "sse") return Backend::SSE;
    if (name == "avx2") return Backend::AVX2;
    YI_CHECK(false, "后端只能是 scalar、sse 或 avx2，实际为 " << name);
    return Backend::Scalar;
}

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::Scalar: return "scalar";
        case Backend::SSE: return "sse";
        case Backend::AVX2: return "avx2";
    }
    return "unknown";
}

bool backend_available(Backend backend) {
    if (backend == Backend::Scalar) return true;
#if defined(YI_X86_SIMD) && (defined(__GNUC__) || defined(__clang__))
    // GCC/Clang 的 CPU 检测同时考虑 OS 保存 AVX 状态的能力，不只看 CPUID 硬件位。
    __builtin_cpu_init();
    if (backend == Backend::SSE) return __builtin_cpu_supports("sse2");
    if (backend == Backend::AVX2)
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    return false;
}

void require_backend(Backend backend) {
    YI_CHECK(backend_available(backend), "后端 " << backend_name(backend)
             << " unavailable：当前构建或 CPU/OS 不支持所需 ISA（avx2 后端还要求 FMA）");
}

}  // namespace yi
