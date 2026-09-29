// ref_ops.cpp —— 标量参考实现
//   阶段 0 只放三个不做乘加的算子：Concat、Split、Add。它们用来验证整条管线 ——
//   结果必须和 ORT 逐位相同，对不上就一定是加载、形状或偏移错了，而不是浮点误差。
#include <cstring>

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
void concat(const Node& node, std::vector<Tensor>& t) {
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
void split(const Node& node, std::vector<Tensor>& t) {
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
void add(const Node& node, std::vector<Tensor>& t) {
    const Tensor& a = t[node.inputs[0]];
    const Tensor& b = t[node.inputs[1]];
    Tensor& y = t[node.outputs[0]];
    YI_CHECK(a.shape == b.shape && a.shape == y.shape, "Add 只支持同形状相加（不做广播）");
    const int64_t n = y.numel();
    for (int64_t i = 0; i < n; ++i) y.data[i] = a.data[i] + b.data[i];
}

struct Entry {
    const char* op;
    KernelFn fn;
};

const Entry kKernels[] = {
    {"Concat", concat},
    {"Split", split},
    {"Add", add},
};

}  // namespace

KernelFn find_kernel(const std::string& op) {
    for (const Entry& e : kKernels)
        if (op == e.op) return e.fn;
    return nullptr;
}

}  // namespace yi
