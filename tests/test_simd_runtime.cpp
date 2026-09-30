// SIMD 后端的重复执行、工作区复用与 naive/reuse 一致性；不依赖 ORT。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "executor.h"

namespace {
using namespace yi;

Model toy_model() {
    Model m;
    auto add = [&](const char* name, std::vector<int64_t> shape, int64_t offset = -1) {
        Tensor t;
        t.name = name;
        t.shape = std::move(shape);
        t.is_const = offset >= 0;
        t.const_offset = offset;
        m.index[t.name] = static_cast<int>(m.tensors.size());
        m.tensors.push_back(std::move(t));
    };
    add("x", {2, 3, 5, 7});
    add("w1", {5, 3, 3, 3}, 0);
    add("b1", {5}, 576);
    add("c1", {2, 5, 5, 7});
    add("s1", {2, 5, 5, 7});
    add("w2", {3, 5, 1, 1}, 640);
    add("b2", {3}, 704);
    add("y", {2, 3, 5, 7});
    m.weights = AlignedBuffer(768);
    for (auto& t : m.tensors) {
        if (!t.is_const) continue;
        t.data = reinterpret_cast<float*>(m.weights.as<char>() + t.const_offset);
        for (int64_t i = 0; i < t.numel(); ++i) t.data[i] = static_cast<float>((i * 7 + 3) % 19 - 9) * 0.02f;
    }
    Node c1;
    c1.name = "conv1"; c1.op = "Conv"; c1.inputs = {0, 1, 2}; c1.outputs = {3};
    c1.attrs.kv = {{"kernel", {3, 3}}, {"stride", {1, 1}}, {"pad", {1, 1, 1, 1}}};
    Node s;
    s.name = "silu"; s.op = "SiLU"; s.inputs = {3}; s.outputs = {4};
    Node c2;
    c2.name = "conv2"; c2.op = "Conv"; c2.inputs = {4, 5, 6}; c2.outputs = {7};
    c2.attrs.kv = {{"kernel", {1, 1}}, {"stride", {1, 1}}, {"pad", {0, 0, 0, 0}}};
    m.nodes = {c1, s, c2};
    m.inputs = {0}; m.outputs = {7};
    m.validate();
    return m;
}

void check_backend(Backend backend) {
    Model scalar_model = toy_model(), naive_model = toy_model(), reused_model = toy_model();
    Executor scalar(scalar_model, false, Backend::Scalar);
    Executor naive(naive_model, false, backend), reused(reused_model, true, backend);
    YI_CHECK(scalar.workspace_bytes() == 0, "标量后端不需要 im2col 工作区");
    const size_t workspace = reused.workspace_bytes();
    YI_CHECK(workspace > 0, "SIMD 卷积没有分配工作区");
    YI_CHECK(naive.workspace_bytes() == workspace, "临时工作区不应依赖激活规划模式");
    std::vector<float> input(2 * 3 * 5 * 7);
    for (int round = 0; round < 2; ++round) {
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = std::sin(static_cast<float>(i) * 0.13f + round) * (round + 1);
        scalar.set_input(0, input.data());
        naive.set_input(0, input.data());
        reused.set_input(0, input.data());
        scalar.run(); naive.run(); reused.run();
        const Tensor& expected = scalar_model.tensors[7];
        const Tensor& a = naive_model.tensors[7];
        const Tensor& b = reused_model.tensors[7];
        YI_CHECK(std::memcmp(a.data, b.data, a.bytes()) == 0, "SIMD 的 naive/reuse 输出不逐位一致");
        for (int64_t i = 0; i < b.numel(); ++i) {
            YI_CHECK(std::isfinite(b.data[i]), "SIMD 产生非有限输出");
            YI_CHECK(std::fabs(b.data[i] - expected.data[i]) <= 1e-5f + 1e-4f * std::fabs(expected.data[i]),
                     "SIMD 与标量逐元素误差超限");
        }
        reused.run(); // 不再 set_input，复用同一输入和 workspace。
        YI_CHECK(std::memcmp(a.data, b.data, a.bytes()) == 0, "重复执行改变了输出");
        YI_CHECK(reused.workspace_bytes() == workspace, "执行期间工作区大小变化");
    }
    std::printf("PASS %s repeated-run workspace=%zu\n", backend_name(backend), workspace);
}
}  // namespace

int main() {
    try {
        for (Backend backend : {Backend::SSE, Backend::AVX2}) {
            if (backend_available(backend)) check_backend(backend);
            else std::printf("SKIP %s unavailable\n", backend_name(backend));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
