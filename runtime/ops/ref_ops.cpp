// ref_ops.cpp —— CPU 标量参考实现；先保证语义正确，再引入 SIMD、分块和多线程。
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "ops.h"

namespace yi {
namespace {

// 按 axis 把形状拆成三段：outer = axis 之前各维之积，mid = shape[axis]，inner = axis 之后各维之积。
// 行优先存储下，张量就是 outer 个首尾相接的"块"，每块 mid × inner 个连续元素。
// 沿 axis 拼接 / 切分，就是对每个块搬一段连续内存。
struct AxisView {
    int64_t outer = 1, mid = 1, inner = 1;
};

AxisView view_at(const Tensor& t, int64_t axis) {
    const auto rank = static_cast<int64_t>(t.shape.size());
    if (axis < 0) axis += rank;                     // ONNX 允许负轴：-1 表示最后一维
    YI_CHECK(axis >= 0 && axis < rank, t.name << " 没有第 " << axis << " 维");
    AxisView v;
    for (int64_t d = 0; d < axis; ++d) v.outer *= t.shape[d];
    v.mid = t.shape[axis];
    for (int64_t d = axis + 1; d < rank; ++d) v.inner *= t.shape[d];
    return v;
}

// Concat：输出的每个块 = 各输入对应的块依次首尾相接
void concat(const Node& node, std::vector<Tensor>& t, Workspace&) {
    Tensor& y = t[node.outputs[0]];
    const int64_t axis = node.attrs.i("axis");
    const AxisView yv = view_at(y, axis);
    const int64_t y_block = yv.mid * yv.inner;      // 输出每块的元素数
    int64_t pos = 0;                                // 当前输入落在输出块里的起点
    for (int idx : node.inputs) {
        const Tensor& x = t[idx];
        const AxisView xv = view_at(x, axis);
        YI_CHECK(xv.outer == yv.outer && xv.inner == yv.inner, "Concat 的输入 " << x.name << " 除拼接轴以外的形状和输出不一致");
        const int64_t x_block = xv.mid * xv.inner;
        for (int64_t o = 0; o < yv.outer; ++o)
            std::memcpy(y.data + o * y_block + pos, x.data + o * x_block, x_block * sizeof(float));
        pos += x_block;
    }
    YI_CHECK(pos == y_block, "Concat 各输入沿拼接轴的长度之和不等于输出");
}

// Split：Concat 反过来，从输入的每个块里依次切出各输出的那一段
void split(const Node& node, std::vector<Tensor>& t, Workspace&) {
    const Tensor& x = t[node.inputs[0]];
    const int64_t axis = node.attrs.i("axis");
    const auto sizes = node.attrs.ints("sizes");
    YI_CHECK(sizes.size() == node.outputs.size(), "Split 的 sizes 个数和输出个数不一致");
    const AxisView xv = view_at(x, axis);
    const int64_t x_block = xv.mid * xv.inner;
    int64_t pos = 0;
    for (size_t k = 0; k < node.outputs.size(); ++k) {
        Tensor& y = t[node.outputs[k]];
        const AxisView yv = view_at(y, axis);
        YI_CHECK(yv.outer == xv.outer && yv.inner == xv.inner && yv.mid == sizes[k], "Split 的输出 " << y.name << " 形状不对");
        const int64_t y_block = yv.mid * yv.inner;
        for (int64_t o = 0; o < xv.outer; ++o)
            std::memcpy(y.data + o * y_block, x.data + o * x_block + pos, y_block * sizeof(float));
        pos += y_block;
    }
    YI_CHECK(pos == x_block, "Split 的 sizes 之和不等于输入沿切分轴的长度");
}

// Add：逐元素相加。只支持两个输入形状完全相同（YOLOv8 的残差连接正好是这样）
void add(const Node& node, std::vector<Tensor>& t, Workspace&) {
    const Tensor& a = t[node.inputs[0]];
    const Tensor& b = t[node.inputs[1]];
    Tensor& y = t[node.outputs[0]];
    YI_CHECK(a.shape == b.shape && a.shape == y.shape, "Add 只支持同形状相加（不做广播）");
    const int64_t n = y.numel();
    for (int64_t i = 0; i < n; ++i) y.data[i] = a.data[i] + b.data[i];
}

// SiLU(x) = x * sigmoid(x)。负半轴用 exp(x)，避免 exp(-x) 上溢。
// 独立 SiLU 节点和卷积收尾融合共用这一个函数，两条路径才能逐位一致。
float silu_scalar(float v) {
    const float e = std::exp(-std::fabs(v));
    return v * (v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
}

void silu(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "SiLU 需要一个输入和一个输出");
    const Tensor& x = t[node.inputs[0]];
    Tensor& y = t[node.outputs[0]];
    YI_CHECK(x.shape == y.shape, "SiLU 输入输出形状必须一致");
    for (int64_t i = 0; i < x.numel(); ++i) y.data[i] = silu_scalar(x.data[i]);
}

// 空间算子只接受连续的四维 NCHW；检查在任何输出写入之前完成。
void check_nchw(const Tensor& t) {
    YI_CHECK(t.shape.size() == 4, t.name << " 必须是四维 NCHW");
    for (int64_t d : t.shape) YI_CHECK(d > 0, t.name << " 的维度必须为正数");
}

struct Window2D {
    std::vector<int64_t> kernel, stride, pad;
};

// Conv / MaxPool 都用同一个滑动窗口尺寸公式。pad 顺序：上、左、下、右。
Window2D window_for(const Node& node, const Tensor& x, const Tensor& y) {
    check_nchw(x);
    check_nchw(y);
    Window2D w{node.attrs.ints("kernel"), node.attrs.ints("stride"), node.attrs.ints("pad")};
    YI_CHECK(w.kernel.size() == 2 && w.stride.size() == 2 && w.pad.size() == 4,
             node.name << " 的 kernel/stride/pad 长度应为 2/2/4");
    for (int64_t k : w.kernel) YI_CHECK(k > 0, node.name << " 的窗口必须为正数");
    for (int64_t s : w.stride) YI_CHECK(s > 0, node.name << " 的步长必须为正数");
    for (int64_t p : w.pad) YI_CHECK(p >= 0, node.name << " 的填充不能为负数");
    for (int d = 0; d < 2; ++d) {
        const int64_t span = x.shape[d + 2] + w.pad[d] + w.pad[d + 2] - w.kernel[d];
        // 先拒绝负数，避免 C++ 负整数除法向零截断，与 floor 公式不一致。
        YI_CHECK(span >= 0, node.name << " 的窗口大于填充后的输入");
        YI_CHECK(y.shape[d + 2] == span / w.stride[d] + 1, node.name << " 的输出空间尺寸错误");
    }
    YI_CHECK(x.shape[0] == y.shape[0], node.name << " 的输入输出 batch 不一致");
    return w;
}

void maxpool(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "MaxPool 需要一个输入和一个输出");
    const Tensor& x = t[node.inputs[0]];
    Tensor& y = t[node.outputs[0]];
    const auto w = window_for(node, x, y);
    YI_CHECK(x.shape[1] == y.shape[1], "MaxPool 不能改变通道数");
    const int64_t H = x.shape[2], W = x.shape[3], Ho = y.shape[2], Wo = y.shape[3];
    for (int64_t n = 0; n < x.shape[0]; ++n) {
        for (int64_t c = 0; c < x.shape[1]; ++c) {
            const float* xp = x.data + (n * x.shape[1] + c) * H * W;
            float* yp = y.data + (n * y.shape[1] + c) * Ho * Wo;
            for (int64_t oh = 0; oh < Ho; ++oh) {
                for (int64_t ow = 0; ow < Wo; ++ow) {
                    float best = -std::numeric_limits<float>::infinity();
                    for (int64_t kh = 0; kh < w.kernel[0]; ++kh) {
                        const int64_t ih = oh * w.stride[0] + kh - w.pad[0];
                        if (ih < 0 || ih >= H) continue;
                        for (int64_t kw = 0; kw < w.kernel[1]; ++kw) {
                            const int64_t iw = ow * w.stride[1] + kw - w.pad[1];
                            if (iw < 0 || iw >= W) continue;
                            const float v = xp[ih * W + iw];
                            // 越界跳过，相当于补负无穷；不能补 0，否则全负窗口会算错。
                            if (v > best || std::isnan(v)) best = v;
                        }
                    }
                    yp[oh * Wo + ow] = best;
                }
            }
        }
    }
}

void upsample_nearest(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "UpsampleNearest 需要一个输入和一个输出");
    const Tensor& x = t[node.inputs[0]];
    Tensor& y = t[node.outputs[0]];
    check_nchw(x);
    check_nchw(y);
    const auto scale = node.attrs.ints("scale");
    YI_CHECK(scale.size() == 2 && scale[0] > 0 && scale[1] > 0, "上采样倍率必须是两个正整数");
    YI_CHECK(x.shape[0] == y.shape[0] && x.shape[1] == y.shape[1], "上采样不能改变 batch 和通道数");
    for (int d = 0; d < 2; ++d) {
        YI_CHECK(scale[d] <= std::numeric_limits<int64_t>::max() / x.shape[d + 2], "上采样尺寸溢出");
        YI_CHECK(y.shape[d + 2] == x.shape[d + 2] * scale[d], "上采样输出尺寸与倍率不一致");
    }
    const int64_t H = x.shape[2], W = x.shape[3], Ho = y.shape[2], Wo = y.shape[3];
    for (int64_t n = 0; n < x.shape[0]; ++n) {
        for (int64_t c = 0; c < x.shape[1]; ++c) {
            const float* xp = x.data + (n * x.shape[1] + c) * H * W;
            float* yp = y.data + (n * y.shape[1] + c) * Ho * Wo;
            for (int64_t oh = 0; oh < Ho; ++oh)
                for (int64_t ow = 0; ow < Wo; ++ow)
                    // 前端保证 nearest + asymmetric + floor，可直接使用非负整数除法。
                    yp[oh * Wo + ow] = xp[(oh / scale[0]) * W + ow / scale[1]];
        }
    }
}

// 直接卷积（不翻转卷积核）：group=1、dilation=1、权重 OIHW，逐输出点累加。
// 这是正确性基线，不使用 im2col、手写 SIMD、线程或额外工作区。
// act=1 时激活（SiLU）在收尾里直接算：卷积结果不写回内存再读一遍。
void conv(const Node& node, std::vector<Tensor>& t, Workspace&) {
    YI_CHECK(node.inputs.size() == 3 && node.outputs.size() == 1, "Conv 需要输入、权重、偏置以及一个输出");
    const Tensor& x = t[node.inputs[0]];
    const Tensor& weight = t[node.inputs[1]];
    const Tensor& bias = t[node.inputs[2]];
    Tensor& y = t[node.outputs[0]];
    const int64_t act = node.attrs.has("act") ? node.attrs.i("act") : 0;
    YI_CHECK(act == 0 || act == 1, "Conv 的 act 只支持 0（无）或 1（SiLU）");
    const auto win = window_for(node, x, y);
    YI_CHECK(weight.is_const && bias.is_const, "Conv 权重与偏置必须是常量");
    YI_CHECK(weight.shape.size() == 4, "Conv 权重必须是四维 OIHW");
    const int64_t Cin = x.shape[1], Cout = y.shape[1];
    const int64_t Kh = win.kernel[0], Kw = win.kernel[1];
    YI_CHECK(weight.shape[0] == Cout && weight.shape[1] == Cin &&
                 weight.shape[2] == Kh && weight.shape[3] == Kw, "Conv 权重与输入输出通道或窗口不匹配");
    YI_CHECK(bias.shape.size() == 1 && bias.shape[0] == Cout, "Conv 偏置长度必须等于输出通道数");
    const int64_t H = x.shape[2], W = x.shape[3], Ho = y.shape[2], Wo = y.shape[3];
    for (int64_t n = 0; n < x.shape[0]; ++n) {
        for (int64_t co = 0; co < Cout; ++co) {
            float* yp = y.data + (n * Cout + co) * Ho * Wo;
            for (int64_t oh = 0; oh < Ho; ++oh) {
                for (int64_t ow = 0; ow < Wo; ++ow) {
                    float sum = 0.0f;
                    for (int64_t ci = 0; ci < Cin; ++ci) {
                        const float* xp = x.data + (n * Cin + ci) * H * W;
                        const float* wp = weight.data + (co * Cin + ci) * Kh * Kw;
                        for (int64_t kh = 0; kh < Kh; ++kh) {
                            const int64_t ih = oh * win.stride[0] + kh - win.pad[0];
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t kw = 0; kw < Kw; ++kw) {
                                const int64_t iw = ow * win.stride[1] + kw - win.pad[1];
                                if (iw < 0 || iw >= W) continue;
                                sum += xp[ih * W + iw] * wp[kh * Kw + kw];
                            }
                        }
                    }
                    const float v = sum + bias.data[co];
                    yp[oh * Wo + ow] = act == 1 ? silu_scalar(v) : v;
                }
            }
        }
    }
}

struct Entry {
    const char* op;
    KernelFn fn;
};

const Entry kKernels[] = {
    {"Conv", conv},
    {"Concat", concat},
    {"Split", split},
    {"Add", add},
    {"SiLU", silu},
    {"MaxPool", maxpool},
    {"UpsampleNearest", upsample_nearest},
};

}  // namespace

KernelFn find_kernel(const std::string& op) {
    for (const Entry& e : kKernels)
        if (op == e.op) return e.fn;
    return nullptr;
}

}  // namespace yi
