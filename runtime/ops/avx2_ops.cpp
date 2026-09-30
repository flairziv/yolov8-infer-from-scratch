// avx2_ops.cpp —— 仅此翻译单元使用 AVX2+FMA，进入前由 baseline dispatcher 检查 CPU/OS。
#include <immintrin.h>

#include "simd_common.h"

namespace yi {
namespace {
struct AVX2 {
    using F = __m256;
    using I = __m256i;
    static constexpr int width = 8;
    static F zero() { return _mm256_setzero_ps(); }
    static F set(float x) { return _mm256_set1_ps(x); }
    static F load(const float* x) { return _mm256_loadu_ps(x); }
    static void store(float* p, F x) { _mm256_storeu_ps(p, x); }
    static F add(F a, F b) { return _mm256_add_ps(a, b); }
    static F sub(F a, F b) { return _mm256_sub_ps(a, b); }
    static F mul(F a, F b) { return _mm256_mul_ps(a, b); }
    static F div(F a, F b) { return _mm256_div_ps(a, b); }
    static F madd(F a, F b, F c) { return _mm256_fmadd_ps(a, b, c); }
    static F abs(F x) { return _mm256_andnot_ps(set(-0.0f), x); }
    static F gt(F a, F b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
    static F ge(F a, F b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
    static F unordered(F a, F b) { return _mm256_cmp_ps(a, b, _CMP_UNORD_Q); }
    static F bit_or(F a, F b) { return _mm256_or_ps(a, b); }
    static F select(F mask, F yes, F no) { return _mm256_or_ps(_mm256_and_ps(mask, yes), _mm256_andnot_ps(mask, no)); }
    static int mask(F x) { return _mm256_movemask_ps(x); }
    static F to_float(I x) { return _mm256_cvtepi32_ps(x); }
    static I floor_int(F x) {
        const I trunc = _mm256_cvttps_epi32(x);
        const I correction = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(x, to_float(trunc), _CMP_LT_OQ)), _mm256_set1_epi32(1));
        return _mm256_sub_epi32(trunc, correction);
    }
    static F pow2(I n) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23)); }
};
}  // namespace

KernelFn find_avx2_kernel(const std::string& op) {
    if (op == "Add") return detail::add_simd<AVX2>;
    if (op == "SiLU") return detail::silu_simd<AVX2>;
    if (op == "MaxPool") return detail::maxpool_simd<AVX2>;
    if (op == "Conv") return detail::conv_simd<AVX2>;
    return nullptr;
}

}  // namespace yi
