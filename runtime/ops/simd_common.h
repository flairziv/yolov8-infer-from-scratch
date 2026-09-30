// simd_common.h —— 与 ISA 无关的形状检查、im2col 和 SIMD 算法模板。
// 模板由 sse_ops.cpp / avx2_ops.cpp 各实例化一次，不把 ISA 编译选项传播给执行器。
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "ops.h"

namespace yi {
namespace {
// 每个 ISA 翻译单元各自实例化，避免链接器把不同 ISA 的 inline/模板辅助函数合并。
namespace detail {

template <size_t N>
std::array<int64_t, N> attr(const Node& node, const char* name) {
    const auto it = node.attrs.kv.find(name);
    YI_CHECK(it != node.attrs.kv.end() && it->second.size() == N, node.name << " 的属性 " << name << " 长度错误");
    std::array<int64_t, N> values{};
    for (size_t i = 0; i < N; ++i) {
        const double v = it->second[i];
        YI_CHECK(std::isfinite(v) && v >= -9223372036854775808.0 && v < 9223372036854775808.0 && std::trunc(v) == v,
                 node.name << " 的属性 " << name << " 必须是整数");
        values[i] = static_cast<int64_t>(v);
    }
    return values;
}

inline void nchw(const Tensor& t) {
    YI_CHECK(t.shape.size() == 4, t.name << " 必须是四维 NCHW");
    for (auto d : t.shape) YI_CHECK(d > 0, t.name << " 的维度必须为正数");
}

struct Window {
    std::array<int64_t, 2> k, stride;
    std::array<int64_t, 4> pad;
};

inline Window window(const Node& node, const Tensor& x, const Tensor& y) {
    nchw(x); nchw(y);
    Window w{attr<2>(node, "kernel"), attr<2>(node, "stride"), attr<4>(node, "pad")};
    for (auto v : w.k) YI_CHECK(v > 0, "窗口大小必须为正数");
    for (auto v : w.stride) YI_CHECK(v > 0, "步长必须为正数");
    for (auto v : w.pad) YI_CHECK(v >= 0, "padding 不能为负数");
    for (int d = 0; d < 2; ++d) {
        const auto max = std::numeric_limits<int64_t>::max();
        YI_CHECK(w.pad[d] <= max - x.shape[d + 2] && w.pad[d + 2] <= max - x.shape[d + 2] - w.pad[d], "填充后尺寸溢出");
        const int64_t span = x.shape[d + 2] + w.pad[d] + w.pad[d + 2] - w.k[d];
        YI_CHECK(span >= 0 && y.shape[d + 2] == span / w.stride[d] + 1, "输出空间尺寸错误");
    }
    YI_CHECK(x.shape[0] == y.shape[0], "输入输出 batch 不一致");
    return w;
}

inline size_t mul_size(size_t a, size_t b) {
    YI_CHECK(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "工作区尺寸溢出");
    return a * b;
}

struct ConvShape {
    Window win;
    int64_t M, K, N;
    size_t col_bytes;
};

inline ConvShape conv_shape(const Node& node, const std::vector<Tensor>& t) {
    YI_CHECK(node.inputs.size() == 3 && node.outputs.size() == 1, "Conv 需要 X/W/B 和一个输出");
    const auto& x = t[node.inputs[0]];
    const auto& w = t[node.inputs[1]];
    const auto& b = t[node.inputs[2]];
    const auto& y = t[node.outputs[0]];
    const auto win = window(node, x, y);
    YI_CHECK(w.is_const && b.is_const, "Conv 的权重、偏置必须是常量");
    YI_CHECK(w.shape.size() == 4 && w.shape[0] == y.shape[1] && w.shape[1] == x.shape[1] &&
                 w.shape[2] == win.k[0] && w.shape[3] == win.k[1], "Conv 权重形状错误");
    YI_CHECK(b.shape.size() == 1 && b.shape[0] == y.shape[1], "Conv 偏置形状错误");
    const size_t K = mul_size(mul_size(x.shape[1], win.k[0]), win.k[1]);
    const size_t N = mul_size(y.shape[2], y.shape[3]);
    const size_t bytes = mul_size(mul_size(K, N), sizeof(float));
    YI_CHECK(K <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
                 N <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
                 bytes <= std::numeric_limits<size_t>::max() - kAlign, "Conv 工作区过大");
    return {win, y.shape[1], static_cast<int64_t>(K), static_cast<int64_t>(N), align_up(bytes)};
}

inline size_t conv_workspace(const Node& node, const std::vector<Tensor>& t, int nr) {
    const auto s = conv_shape(node, t);
    const size_t panel = mul_size(mul_size(s.K, nr), sizeof(float));
    YI_CHECK(panel <= std::numeric_limits<size_t>::max() - s.col_bytes, "Conv 面板工作区溢出");
    return s.col_bytes + panel;
}

// 一次展开一个 batch，列坐标是展平的输出空间；batch 间复用同一工作区。
inline void im2col(const Tensor& x, const Tensor& y, const ConvShape& s, int64_t batch, float* col) {
    const int64_t H = x.shape[2], W = x.shape[3], Ho = y.shape[2], Wo = y.shape[3];
    int64_t k = 0;
    for (int64_t ci = 0; ci < x.shape[1]; ++ci) {
        const float* plane = x.data + (batch * x.shape[1] + ci) * H * W;
        for (int64_t kh = 0; kh < s.win.k[0]; ++kh) {
            for (int64_t kw = 0; kw < s.win.k[1]; ++kw, ++k) {
                float* row = col + k * s.N;
                for (int64_t oh = 0; oh < Ho; ++oh) {
                    const int64_t ih = oh * s.win.stride[0] + kh - s.win.pad[0];
                    for (int64_t ow = 0; ow < Wo; ++ow) {
                        const int64_t iw = ow * s.win.stride[1] + kw - s.win.pad[1];
                        row[oh * Wo + ow] = ih >= 0 && ih < H && iw >= 0 && iw < W ? plane[ih * W + iw] : 0.0f;
                    }
                }
            }
        }
    }
}

inline float scalar_silu(float v) {
    const float e = std::exp(-std::fabs(v));
    return v * (v >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e));
}

// 自写 exp 近似：x=n*ln2+r，r∈[-ln2/2,ln2/2]，7阶 Taylor + Horner。
// 调用者只对 x∈[-80,0] 使用，2^n 保持正常 float，超范围/非有限数走标量路径。
template <class V>
typename V::F exp_reduced(typename V::F x) {
    const auto n = V::floor_int(V::add(V::mul(x, V::set(1.4426950408889634f)), V::set(0.5f)));
    const auto nf = V::to_float(n);
    auto r = V::sub(x, V::mul(nf, V::set(0.693359375f)));
    r = V::sub(r, V::mul(nf, V::set(-0.00021219444005469058f)));
    auto p = V::set(1.0f / 5040.0f);
    p = V::madd(p, r, V::set(1.0f / 720.0f));
    p = V::madd(p, r, V::set(1.0f / 120.0f));
    p = V::madd(p, r, V::set(1.0f / 24.0f));
    p = V::madd(p, r, V::set(1.0f / 6.0f));
    p = V::madd(p, r, V::set(0.5f));
    p = V::madd(p, r, V::set(1.0f));
    p = V::madd(p, r, V::set(1.0f));
    return V::mul(p, V::pow2(n));
}

template <class V>
void add_simd(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 2 && node.outputs.size() == 1, "Add 需要两个输入和一个输出");
    const auto& a = t[node.inputs[0]]; const auto& b = t[node.inputs[1]]; auto& y = t[node.outputs[0]];
    YI_CHECK(a.shape == b.shape && a.shape == y.shape, "Add 只支持同形状相加");
    const int64_t count = y.numel();
    int64_t i = 0;
    for (; i + V::width <= count; i += V::width) V::store(y.data + i, V::add(V::load(a.data + i), V::load(b.data + i)));
    for (; i < count; ++i) y.data[i] = a.data[i] + b.data[i];
}

template <class V>
void silu_simd(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "SiLU 需要一个输入和一个输出");
    const auto& x = t[node.inputs[0]]; auto& y = t[node.outputs[0]];
    YI_CHECK(x.shape == y.shape, "SiLU 输入输出形状必须一致");
    const int64_t count = x.numel();
    int64_t i = 0;
    for (; i + V::width <= count; i += V::width) {
        const auto v = V::load(x.data + i), abs = V::abs(v);
        if (V::mask(V::bit_or(V::gt(abs, V::set(80.0f)), V::unordered(v, v)))) {
            for (int j = 0; j < V::width; ++j) y.data[i + j] = scalar_silu(x.data[i + j]);
            continue;
        }
        const auto e = exp_reduced<V>(V::sub(V::zero(), abs));
        const auto numerator = V::select(V::ge(v, V::zero()), V::set(1.0f), e);
        V::store(y.data + i, V::mul(v, V::div(numerator, V::add(V::set(1.0f), e))));
    }
    for (; i < count; ++i) y.data[i] = scalar_silu(x.data[i]);
}

template <class V>
void maxpool_simd(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "MaxPool 需要一个输入和一个输出");
    const auto& x = t[node.inputs[0]]; auto& y = t[node.outputs[0]];
    const auto w = window(node, x, y);
    YI_CHECK(x.shape[1] == y.shape[1], "MaxPool 不能改变通道数");
    const int64_t H = x.shape[2], W = x.shape[3], Ho = y.shape[2], Wo = y.shape[3];
    for (int64_t nc = 0; nc < x.shape[0] * x.shape[1]; ++nc) {
        const float* xp = x.data + nc * H * W;
        float* yp = y.data + nc * Ho * Wo;
        for (int64_t oh = 0; oh < Ho; ++oh) {
            int64_t ow = 0;
            while (ow < Wo) {
                const int64_t left = ow * w.stride[1] - w.pad[1];
                // SIMD 只处理同一输入行内完整且连续的窗口组；边缘/stride>1/尾部按标量取值。
                if (w.stride[1] == 1 && ow + V::width <= Wo && left >= 0 && left + V::width - 1 + w.k[1] <= W) {
                    auto best = V::set(-std::numeric_limits<float>::infinity());
                    for (int64_t kh = 0; kh < w.k[0]; ++kh) {
                        const int64_t ih = oh * w.stride[0] + kh - w.pad[0];
                        if (ih < 0 || ih >= H) continue;
                        for (int64_t kw = 0; kw < w.k[1]; ++kw) {
                            const auto v = V::load(xp + ih * W + left + kw);
                            best = V::select(V::bit_or(V::gt(v, best), V::unordered(v, v)), v, best);
                        }
                    }
                    V::store(yp + oh * Wo + ow, best);
                    ow += V::width;
                } else {
                    float best = -std::numeric_limits<float>::infinity();
                    for (int64_t kh = 0; kh < w.k[0]; ++kh) {
                        const int64_t ih = oh * w.stride[0] + kh - w.pad[0];
                        if (ih < 0 || ih >= H) continue;
                        for (int64_t kw = 0; kw < w.k[1]; ++kw) {
                            const int64_t iw = left + kw;
                            if (iw < 0 || iw >= W) continue;
                            const float v = xp[ih * W + iw];
                            if (v > best || std::isnan(v)) best = v;
                        }
                    }
                    yp[oh * Wo + ow++] = best;
                }
            }
        }
    }
}

template <class V>
void store_gemm_row(float* dst, typename V::F lo, typename V::F hi, float bias, int cols) {
    const auto b = V::set(bias);
    lo = V::add(lo, b); hi = V::add(hi, b);
    if (cols == 2 * V::width) {
        V::store(dst, lo); V::store(dst + V::width, hi);
    } else {
        alignas(64) float tail[2 * V::width];
        V::store(tail, lo); V::store(tail + V::width, hi);
        std::memcpy(dst, tail, cols * sizeof(float));
    }
}

// MR=4、NR=2*向量宽度：8个独立向量累加器，外加2个B向量和1个A广播值。
// Rows 在编译期决定，M尾部不读取不存在的权重行；N尾部只写有效列。
template <class V, int Rows>
void gemm_tile(const float* weights, const float* panel, const float* bias, float* out,
               int64_t K, int64_t N, int cols) {
    auto c00 = V::zero(), c01 = V::zero(), c10 = V::zero(), c11 = V::zero();
    auto c20 = V::zero(), c21 = V::zero(), c30 = V::zero(), c31 = V::zero();
    for (int64_t k = 0; k < K; ++k) {
        const auto b0 = V::load(panel + k * 2 * V::width), b1 = V::load(panel + k * 2 * V::width + V::width);
        auto a = V::set(weights[k]);
        c00 = V::madd(a, b0, c00); c01 = V::madd(a, b1, c01);
        if constexpr (Rows > 1) { a = V::set(weights[K + k]); c10 = V::madd(a, b0, c10); c11 = V::madd(a, b1, c11); }
        if constexpr (Rows > 2) { a = V::set(weights[2 * K + k]); c20 = V::madd(a, b0, c20); c21 = V::madd(a, b1, c21); }
        if constexpr (Rows > 3) { a = V::set(weights[3 * K + k]); c30 = V::madd(a, b0, c30); c31 = V::madd(a, b1, c31); }
    }
    store_gemm_row<V>(out, c00, c01, bias[0], cols);
    if constexpr (Rows > 1) store_gemm_row<V>(out + N, c10, c11, bias[1], cols);
    if constexpr (Rows > 2) store_gemm_row<V>(out + 2 * N, c20, c21, bias[2], cols);
    if constexpr (Rows > 3) store_gemm_row<V>(out + 3 * N, c30, c31, bias[3], cols);
}

template <class V>
void conv_simd(const Node& node, std::vector<Tensor>& t, Workspace& workspace) {
    const auto s = conv_shape(node, t);
    constexpr int NR = 2 * V::width;
    const size_t required = conv_workspace(node, t, NR);
    YI_CHECK(workspace.data && workspace.bytes >= required, "Conv 工作区不足");
    const auto& x = t[node.inputs[0]]; const auto& w = t[node.inputs[1]]; const auto& bias = t[node.inputs[2]];
    auto& y = t[node.outputs[0]];
    float* col = workspace.data;
    float* panel = reinterpret_cast<float*>(reinterpret_cast<char*>(workspace.data) + s.col_bytes);
    for (int64_t batch = 0; batch < x.shape[0]; ++batch) {
        im2col(x, y, s, batch, col);
        float* out = y.data + batch * s.M * s.N;
        for (int64_t j = 0; j < s.N; j += NR) {
            const int cols = static_cast<int>(std::min<int64_t>(NR, s.N - j));
            for (int64_t k = 0; k < s.K; ++k) {
                std::memcpy(panel + k * NR, col + k * s.N + j, cols * sizeof(float));
                std::fill(panel + k * NR + cols, panel + (k + 1) * NR, 0.0f);
            }
            for (int64_t m = 0; m < s.M; m += 4) {
                const float* wm = w.data + m * s.K;
                const float* bm = bias.data + m;
                float* ym = out + m * s.N + j;
                switch (std::min<int64_t>(4, s.M - m)) {
                    case 4: gemm_tile<V, 4>(wm, panel, bm, ym, s.K, s.N, cols); break;
                    case 3: gemm_tile<V, 3>(wm, panel, bm, ym, s.K, s.N, cols); break;
                    case 2: gemm_tile<V, 2>(wm, panel, bm, ym, s.K, s.N, cols); break;
                    case 1: gemm_tile<V, 1>(wm, panel, bm, ym, s.K, s.N, cols); break;
                }
            }
        }
    }
}

}  // namespace detail
}  // namespace
}  // namespace yi
