// cuda_ops.cu —— CUDA 算子实现。每个 KernelFn 与 CPU 版签名一致：读形状/属性，启动 kernel。
// 这一版是"朴素直接卷积"：一个线程算一个输出点，不做 im2col、不做共享内存分块。
// 数值与 CPU 不逐位一致（浮点加法的顺序不同），验证口径是容差而不是 memcmp。
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "common.h"
#include "ops/ops.h"

namespace yi {
namespace {

constexpr int kThreads = 256;

void check(cudaError_t err, const char* what) {
    YI_CHECK(err == cudaSuccess, what << " 失败: " << cudaGetErrorString(err));
}

// 一维 grid-stride 循环的网格大小。
int grid_for(int64_t total) {
    const int64_t blocks = (total + kThreads - 1) / kThreads;
    return static_cast<int>(std::min<int64_t>(blocks, 65535));
}

__device__ float silu_dev(float v) {
    const float e = expf(-fabsf(v));
    return v * (v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
}

struct Window {
    int64_t k[2], s[2], p[4];
};

void check_nchw(const Tensor& t) {
    YI_CHECK(t.shape.size() == 4, t.name << " 必须是四维 NCHW");
    for (int64_t d : t.shape) YI_CHECK(d > 0, t.name << " 的维度必须为正数");
}

Window window_of(const Node& node, const Tensor& x, const Tensor& y) {
    check_nchw(x);
    check_nchw(y);
    const auto k = node.attrs.ints("kernel");
    const auto s = node.attrs.ints("stride");
    const auto p = node.attrs.ints("pad");
    YI_CHECK(k.size() == 2 && s.size() == 2 && p.size() == 4, node.name << " 的 kernel/stride/pad 长度应为 2/2/4");
    Window w{};
    for (int i = 0; i < 2; ++i) {
        w.k[i] = k[i];
        w.s[i] = s[i];
        YI_CHECK(w.k[i] > 0 && w.s[i] > 0, node.name << " 的窗口和步长必须为正数");
        YI_CHECK(x.shape[i + 2] + p[i] + p[i + 2] >= w.k[i], node.name << " 的窗口大于填充后的输入");
        YI_CHECK(y.shape[i + 2] == (x.shape[i + 2] + p[i] + p[i + 2] - w.k[i]) / w.s[i] + 1,
                 node.name << " 的输出空间尺寸错误");
    }
    for (int i = 0; i < 4; ++i) w.p[i] = p[i];
    YI_CHECK(x.shape[0] == y.shape[0], node.name << " 的输入输出 batch 不一致");
    return w;
}

// ---------- 卷积（直接法） ----------

struct ConvArgs {
    const float* x;
    const float* w;
    const float* b;
    float* y;
    int64_t N, Cin, H, W, Cout, Ho, Wo, Kh, Kw, sh, sw, ph, pw;
    int act;
};

__global__ void conv_direct_kernel(ConvArgs a) {
    const int64_t total = a.N * a.Cout * a.Ho * a.Wo;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t r = i;
        const int64_t ow = r % a.Wo; r /= a.Wo;
        const int64_t oh = r % a.Ho; r /= a.Ho;
        const int64_t co = r % a.Cout; r /= a.Cout;
        const int64_t n = r;
        float sum = 0.0f;
        for (int64_t ci = 0; ci < a.Cin; ++ci) {
            const float* xp = a.x + ((n * a.Cin + ci) * a.H) * a.W;
            const float* wp = a.w + ((co * a.Cin + ci) * a.Kh) * a.Kw;
            for (int64_t kh = 0; kh < a.Kh; ++kh) {
                const int64_t ih = oh * a.sh + kh - a.ph;
                if (ih < 0 || ih >= a.H) continue;
                for (int64_t kw = 0; kw < a.Kw; ++kw) {
                    const int64_t iw = ow * a.sw + kw - a.pw;
                    if (iw < 0 || iw >= a.W) continue;
                    sum += xp[ih * a.W + iw] * wp[kh * a.Kw + kw];
                }
            }
        }
        const float v = sum + a.b[co];
        a.y[i] = a.act == 1 ? silu_dev(v) : v;
    }
}

// ---------- 卷积 v2：共享内存分块 + 输出通道寄存器分块 ----------
// 一个 block 负责 16×16 个输出像素 × 8 个输出通道：
//   - 每个输入通道的输入 patch（含 halo）只从全局内存加载一次到共享内存，16×16 个线程、9 个 tap 共用；
//   - 每个线程在寄存器里累加 8 个输出通道，一次读入的输入值复用 8 次；
//   - 本输入通道的 8×taps 个权重也先放进共享内存，block 内广播读取。
// 累加顺序（ci 外、kh/kw 内）与朴素版相同，所以两版输出逐位一致，可以互为对照。
constexpr int kTile = 16;
constexpr int kCoPerThread = 8;
constexpr int kPatchMax = (kTile - 1) * 2 + 3;   // stride<=2、kernel<=3 时 patch 边长最大 33
constexpr int kTapsMax = 9;

__global__ void conv_tiled_kernel(ConvArgs a) {
    __shared__ float patch[kPatchMax * kPatchMax];
    __shared__ float wsm[kCoPerThread * kTapsMax];
    const int co_groups = static_cast<int>((a.Cout + kCoPerThread - 1) / kCoPerThread);
    const int n = static_cast<int>(blockIdx.z) / co_groups;
    const int co_base = (static_cast<int>(blockIdx.z) % co_groups) * kCoPerThread;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * kTile + tx;
    const int oh0 = blockIdx.y * kTile, ow0 = blockIdx.x * kTile;
    const int oh = oh0 + ty, ow = ow0 + tx;
    const int sh = static_cast<int>(a.sh), sw = static_cast<int>(a.sw);
    const int Kh = static_cast<int>(a.Kh), Kw = static_cast<int>(a.Kw);
    const int PH = (kTile - 1) * sh + Kh, PW = (kTile - 1) * sw + Kw;
    const int ih0 = oh0 * sh - static_cast<int>(a.ph), iw0 = ow0 * sw - static_cast<int>(a.pw);
    const int taps = Kh * Kw;
    float acc[kCoPerThread];
#pragma unroll
    for (int c = 0; c < kCoPerThread; ++c) acc[c] = 0.0f;
    for (int64_t ci = 0; ci < a.Cin; ++ci) {
        const float* xp = a.x + ((n * a.Cin + ci) * a.H) * a.W;
        for (int i = tid; i < PH * PW; i += kTile * kTile) {
            const int py = i / PW, px = i % PW;
            const int ih = ih0 + py, iw = iw0 + px;
            patch[i] = (ih >= 0 && ih < a.H && iw >= 0 && iw < a.W) ? xp[static_cast<int64_t>(ih) * a.W + iw] : 0.0f;
        }
        for (int i = tid; i < kCoPerThread * taps; i += kTile * kTile) {
            const int c = i / taps, k = i % taps;
            const int co = co_base + c;
            wsm[i] = co < a.Cout ? a.w[((static_cast<int64_t>(co) * a.Cin + ci) * taps) + k] : 0.0f;
        }
        __syncthreads();
        for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
                const float v = patch[(ty * sh + kh) * PW + tx * sw + kw];
                const int k = kh * Kw + kw;
#pragma unroll
                for (int c = 0; c < kCoPerThread; ++c) acc[c] += v * wsm[c * taps + k];
            }
        }
        __syncthreads();
    }
    if (oh < a.Ho && ow < a.Wo) {
        for (int c = 0; c < kCoPerThread; ++c) {
            const int co = co_base + c;
            if (co >= a.Cout) break;
            const float v = acc[c] + a.b[co];
            a.y[((static_cast<int64_t>(n) * a.Cout + co) * a.Ho + oh) * a.Wo + ow] = a.act == 1 ? silu_dev(v) : v;
        }
    }
}

void conv_cuda(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 3 && node.outputs.size() == 1, "Conv 需要输入、权重、偏置以及一个输出");
    const Tensor& x = t[node.inputs[0]];
    const Tensor& w = t[node.inputs[1]];
    const Tensor& b = t[node.inputs[2]];
    Tensor& y = t[node.outputs[0]];
    const auto win = window_of(node, x, y);
    const int64_t act = node.attrs.has("act") ? node.attrs.i("act") : 0;
    YI_CHECK(act == 0 || act == 1, "Conv 的 act 只支持 0（无）或 1（SiLU）");
    YI_CHECK(w.shape.size() == 4 && w.shape[0] == y.shape[1] && w.shape[1] == x.shape[1] &&
                 w.shape[2] == win.k[0] && w.shape[3] == win.k[1], "Conv 权重形状错误");
    YI_CHECK(b.shape.size() == 1 && b.shape[0] == y.shape[1], "Conv 偏置形状错误");
    ConvArgs a{x.data, w.data, b.data, y.data, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
               y.shape[1], y.shape[2], y.shape[3], win.k[0], win.k[1], win.s[0], win.s[1],
               win.p[0], win.p[1], static_cast<int>(act)};
    // 分块版只覆盖 patch 能放进静态共享内存的形状；其余（以及消融开关 YI_CUDA_CONV=naive）走朴素版。
    const bool fits = (kTile - 1) * win.s[0] + win.k[0] <= kPatchMax && (kTile - 1) * win.s[1] + win.k[1] <= kPatchMax &&
                      win.k[0] * win.k[1] <= kTapsMax;
    const char* mode = std::getenv("YI_CUDA_CONV");
    const bool naive = mode && std::string(mode) == "naive";
    if (fits && !naive) {
        const int co_groups = static_cast<int>((y.shape[1] + kCoPerThread - 1) / kCoPerThread);
        YI_CHECK(x.shape[0] * co_groups <= 65535, "Conv 的 grid.z 超限");
        const dim3 block(kTile, kTile);
        const dim3 grid(static_cast<unsigned>((y.shape[3] + kTile - 1) / kTile),
                        static_cast<unsigned>((y.shape[2] + kTile - 1) / kTile),
                        static_cast<unsigned>(x.shape[0] * co_groups));
        conv_tiled_kernel<<<grid, block>>>(a);
        check(cudaGetLastError(), "conv_tiled_kernel 启动");
        return;
    }
    const int64_t total = x.shape[0] * y.shape[1] * y.shape[2] * y.shape[3];
    conv_direct_kernel<<<grid_for(total), kThreads>>>(a);
    check(cudaGetLastError(), "conv_direct_kernel 启动");
}

// ---------- 逐元素 ----------

__global__ void add_kernel(const float* a, const float* b, float* y, int64_t n) {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        y[i] = a[i] + b[i];
}

__global__ void silu_kernel(const float* x, float* y, int64_t n) {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        y[i] = silu_dev(x[i]);
}

void add_cuda(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 2 && node.outputs.size() == 1, "Add 需要两个输入和一个输出");
    const Tensor& a = t[node.inputs[0]];
    const Tensor& b = t[node.inputs[1]];
    Tensor& y = t[node.outputs[0]];
    YI_CHECK(a.shape == b.shape && a.shape == y.shape, "Add 只支持同形状相加（不做广播）");
    const int64_t n = y.numel();
    add_kernel<<<grid_for(n), kThreads>>>(a.data, b.data, y.data, n);
    check(cudaGetLastError(), "add_kernel 启动");
}

void silu_cuda(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "SiLU 需要一个输入和一个输出");
    const Tensor& x = t[node.inputs[0]];
    Tensor& y = t[node.outputs[0]];
    YI_CHECK(x.shape == y.shape, "SiLU 输入输出形状必须一致");
    const int64_t n = x.numel();
    silu_kernel<<<grid_for(n), kThreads>>>(x.data, y.data, n);
    check(cudaGetLastError(), "silu_kernel 启动");
}

// ---------- 池化 ----------

struct PoolArgs {
    const float* x;
    float* y;
    int64_t NC, H, W, Ho, Wo, Kh, Kw, sh, sw, ph, pw;
};

__global__ void maxpool_kernel(PoolArgs a) {
    const int64_t total = a.NC * a.Ho * a.Wo;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t r = i;
        const int64_t ow = r % a.Wo; r /= a.Wo;
        const int64_t oh = r % a.Ho; r /= a.Ho;
        const int64_t nc = r;
        const float* xp = a.x + nc * a.H * a.W;
        float best = -INFINITY;
        for (int64_t kh = 0; kh < a.Kh; ++kh) {
            const int64_t ih = oh * a.sh + kh - a.ph;
            if (ih < 0 || ih >= a.H) continue;
            for (int64_t kw = 0; kw < a.Kw; ++kw) {
                const int64_t iw = ow * a.sw + kw - a.pw;
                if (iw < 0 || iw >= a.W) continue;
                const float v = xp[ih * a.W + iw];
                // 越界跳过相当于补负无穷；NaN 与 CPU 版一样向外传播。
                if (isnan(v) || v > best) best = v;
            }
        }
        a.y[i] = best;
    }
}

void maxpool_cuda(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "MaxPool 需要一个输入和一个输出");
    const Tensor& x = t[node.inputs[0]];
    Tensor& y = t[node.outputs[0]];
    const auto win = window_of(node, x, y);
    YI_CHECK(x.shape[1] == y.shape[1], "MaxPool 不能改变通道数");
    PoolArgs a{x.data, y.data, x.shape[0] * x.shape[1], x.shape[2], x.shape[3], y.shape[2], y.shape[3],
               win.k[0], win.k[1], win.s[0], win.s[1], win.p[0], win.p[1]};
    const int64_t total = a.NC * a.Ho * a.Wo;
    maxpool_kernel<<<grid_for(total), kThreads>>>(a);
    check(cudaGetLastError(), "maxpool_kernel 启动");
}

// ---------- Concat / Split / Upsample ----------

// 把输入的每一块（x_block 个元素）搬到输出对应块的 pos 偏移处；一次启动搬一个输入。
__global__ void concat_copy_kernel(const float* x, float* y, int64_t outer, int64_t x_block, int64_t y_block, int64_t pos) {
    const int64_t total = outer * x_block;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t o = i / x_block, p = i % x_block;
        y[o * y_block + pos + p] = x[i];
    }
}

// Split 反过来：从输入的块里取出 pos 处的 y_block 个元素。
__global__ void split_copy_kernel(const float* x, float* y, int64_t outer, int64_t x_block, int64_t y_block, int64_t pos) {
    const int64_t total = outer * y_block;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t o = i / y_block, p = i % y_block;
        y[i] = x[o * x_block + pos + p];
    }
}

__global__ void upsample_kernel(const float* x, float* y, int64_t NC, int64_t H, int64_t W, int64_t Ho, int64_t Wo,
                                int64_t sh, int64_t sw) {
    const int64_t total = NC * Ho * Wo;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t r = i;
        const int64_t ow = r % Wo; r /= Wo;
        const int64_t oh = r % Ho; r /= Ho;
        y[i] = x[(r * H + oh / sh) * W + ow / sw];
    }
}

// 按 axis 把形状拆成 outer × mid × inner：行优先存储下张量是 outer 个 mid×inner 的连续块。
struct AxisView {
    int64_t outer = 1, mid = 1, inner = 1;
};

AxisView view_at(const Tensor& t, int64_t axis) {
    const auto rank = static_cast<int64_t>(t.shape.size());
    if (axis < 0) axis += rank;
    YI_CHECK(axis >= 0 && axis < rank, t.name << " 没有第 " << axis << " 维");
    AxisView v;
    for (int64_t d = 0; d < axis; ++d) v.outer *= t.shape[d];
    v.mid = t.shape[axis];
    for (int64_t d = axis + 1; d < rank; ++d) v.inner *= t.shape[d];
    return v;
}

void concat_cuda(const Node& node, std::vector<Tensor>& t, Workspace&) {
    Tensor& y = t[node.outputs[0]];
    const int64_t axis = node.attrs.i("axis");
    const AxisView yv = view_at(y, axis);
    const int64_t y_block = yv.mid * yv.inner;
    int64_t pos = 0;
    for (int idx : node.inputs) {
        const Tensor& x = t[idx];
        const AxisView xv = view_at(x, axis);
        YI_CHECK(xv.outer == yv.outer && xv.inner == yv.inner, "Concat 的输入 " << x.name << " 形状和输出不一致");
        const int64_t x_block = xv.mid * xv.inner;
        concat_copy_kernel<<<grid_for(xv.outer * x_block), kThreads>>>(x.data, y.data, xv.outer, x_block, y_block, pos);
        check(cudaGetLastError(), "concat_copy_kernel 启动");
        pos += x_block;
    }
    YI_CHECK(pos == y_block, "Concat 各输入沿拼接轴的长度之和不等于输出");
}

void split_cuda(const Node& node, std::vector<Tensor>& t, Workspace&) {
    const Tensor& x = t[node.inputs[0]];
    if (t[node.outputs[0]].is_view()) {
        for (int o : node.outputs)
            YI_CHECK(t[o].is_view() && t[o].view_of == t[node.outputs[0]].view_of,
                     "Split 的输出要么全是同一个父张量的视图，要么全不是");
        return;   // 视图：数据已经在父张量的切片里，执行器把指针指好了
    }
    const int64_t axis = node.attrs.i("axis");
    const auto sizes = node.attrs.ints("sizes");
    YI_CHECK(sizes.size() == node.outputs.size(), "Split 的 sizes 个数和输出个数不一致");
    const AxisView xv = view_at(x, axis);
    const int64_t x_block = xv.mid * xv.inner;
    int64_t pos = 0;
    for (size_t k = 0; k < node.outputs.size(); ++k) {
        Tensor& y = t[node.outputs[k]];
        const AxisView yv = view_at(y, axis);
        YI_CHECK(yv.outer == xv.outer && yv.inner == xv.inner && yv.mid == sizes[k],
                 "Split 的输出 " << y.name << " 形状不对");
        const int64_t y_block = yv.mid * yv.inner;
        split_copy_kernel<<<grid_for(xv.outer * y_block), kThreads>>>(x.data, y.data, xv.outer, x_block, y_block, pos);
        check(cudaGetLastError(), "split_copy_kernel 启动");
        pos += y_block;
    }
    YI_CHECK(pos == x_block, "Split 的 sizes 之和不等于输入沿切分轴的长度");
}

void upsample_cuda(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "UpsampleNearest 需要一个输入和一个输出");
    const Tensor& x = t[node.inputs[0]];
    Tensor& y = t[node.outputs[0]];
    check_nchw(x);
    check_nchw(y);
    const auto scale = node.attrs.ints("scale");
    YI_CHECK(scale.size() == 2 && scale[0] > 0 && scale[1] > 0, "上采样倍率必须是两个正整数");
    YI_CHECK(x.shape[0] == y.shape[0] && x.shape[1] == y.shape[1], "上采样不能改变 batch 和通道数");
    for (int d = 0; d < 2; ++d) YI_CHECK(y.shape[d + 2] == x.shape[d + 2] * scale[d], "上采样输出尺寸与倍率不一致");
    const int64_t total = x.shape[0] * x.shape[1] * y.shape[2] * y.shape[3];
    upsample_kernel<<<grid_for(total), kThreads>>>(x.data, y.data, x.shape[0] * x.shape[1], x.shape[2], x.shape[3],
                                                   y.shape[2], y.shape[3], scale[0], scale[1]);
    check(cudaGetLastError(), "upsample_kernel 启动");
}

struct Entry {
    const char* op;
    KernelFn fn;
};

const Entry kCudaKernels[] = {
    {"Conv", conv_cuda},
    {"Concat", concat_cuda},
    {"Split", split_cuda},
    {"Add", add_cuda},
    {"SiLU", silu_cuda},
    {"MaxPool", maxpool_cuda},
    {"UpsampleNearest", upsample_cuda},
};

}  // namespace

KernelFn find_cuda_kernel(const std::string& op) {
    for (const Entry& e : kCudaKernels)
        if (op == e.op) return e.fn;
    return nullptr;
}

}  // namespace yi
