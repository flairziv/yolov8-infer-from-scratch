// sse_ops.cpp —— 仅此翻译单元使用 SSE2，禁止 AVX/FMA 自动混入。
#include <emmintrin.h>

#include "simd_common.h"

namespace yi {
namespace {
struct SSE {
    using F = __m128;
    using I = __m128i;
    static constexpr int width = 4;
    static F zero() { return _mm_setzero_ps(); }
    static F set(float x) { return _mm_set1_ps(x); }
    static F load(const float* x) { return _mm_loadu_ps(x); }
    static void store(float* p, F x) { _mm_storeu_ps(p, x); }
    static F add(F a, F b) { return _mm_add_ps(a, b); }
    static F sub(F a, F b) { return _mm_sub_ps(a, b); }
    static F mul(F a, F b) { return _mm_mul_ps(a, b); }
    static F div(F a, F b) { return _mm_div_ps(a, b); }
    static F madd(F a, F b, F c) { return add(mul(a, b), c); }
    static F abs(F x) { return _mm_andnot_ps(set(-0.0f), x); }
    static F gt(F a, F b) { return _mm_cmpgt_ps(a, b); }
    static F ge(F a, F b) { return _mm_cmpge_ps(a, b); }
    static F unordered(F a, F b) { return _mm_cmpunord_ps(a, b); }
    static F bit_or(F a, F b) { return _mm_or_ps(a, b); }
    static F select(F mask, F yes, F no) { return _mm_or_ps(_mm_and_ps(mask, yes), _mm_andnot_ps(mask, no)); }
    static int mask(F x) { return _mm_movemask_ps(x); }
    static F to_float(I x) { return _mm_cvtepi32_ps(x); }
    static I floor_int(F x) {
        const I trunc = _mm_cvttps_epi32(x);
        const I correction = _mm_and_si128(_mm_castps_si128(_mm_cmplt_ps(x, to_float(trunc))), _mm_set1_epi32(1));
        return _mm_sub_epi32(trunc, correction);
    }
    static F pow2(I n) { return _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(n, _mm_set1_epi32(127)), 23)); }
};
}  // namespace

KernelFn find_sse_kernel(const std::string& op) {
    if (op == "Add") return detail::add_simd<SSE>;
    if (op == "SiLU") return detail::silu_simd<SSE>;
    if (op == "MaxPool") return detail::maxpool_simd<SSE>;
    if (op == "Conv") return detail::conv_simd<SSE>;
    return nullptr;
}

}  // namespace yi
