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

Model toy_model(bool fused = false) {
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
    if (!fused) add("c1", {2, 5, 5, 7});   // 未融合时 conv1 的中间输出
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
    c1.name = "conv1"; c1.op = "Conv"; c1.inputs = {0, 1, 2};
    c1.attrs.kv = {{"kernel", {3, 3}}, {"stride", {1, 1}}, {"pad", {1, 1, 1, 1}}};
    Node c2;
    c2.name = "conv2"; c2.op = "Conv"; c2.inputs = {m.find("s1"), m.find("w2"), m.find("b2")};
    c2.outputs = {m.find("y")};
    c2.attrs.kv = {{"kernel", {1, 1}}, {"stride", {1, 1}}, {"pad", {0, 0, 0, 0}}};
    if (fused) {
        // 收尾融合：conv1 直接产出激活后的张量 s1，不再有独立的 SiLU 节点。
        c1.outputs = {m.find("s1")};
        c1.attrs.kv["act"] = {1};
        m.nodes = {c1, c2};
    } else {
        Node s;
        s.name = "silu"; s.op = "SiLU"; s.inputs = {m.find("c1")}; s.outputs = {m.find("s1")};
        c1.outputs = {m.find("c1")};
        m.nodes = {c1, s, c2};
    }
    m.inputs = {0}; m.outputs = {m.find("y")};
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
        const Tensor& expected = scalar_model.tensors[scalar_model.find("y")];
        const Tensor& a = naive_model.tensors[naive_model.find("y")];
        const Tensor& b = reused_model.tensors[reused_model.find("y")];
        YI_CHECK(std::memcmp(a.data, b.data, a.bytes()) == 0, "SIMD 的 naive/reuse 输出不逐位一致");
        for (int64_t i = 0; i < b.numel(); ++i) {
            YI_CHECK(std::isfinite(b.data[i]), "SIMD 产生非有限输出");
            YI_CHECK(std::fabs(b.data[i] - expected.data[i]) <= 1e-5f + 1e-4f * std::fabs(expected.data[i]),
                     "SIMD 与标量逐元素误差超限");
        }
        // 契约：图输入的槽位在最后一次被读之后可以被后续激活复用，所以每次 run 之前都要重新 set_input。
        reused.set_input(0, input.data());
        reused.run();
        YI_CHECK(std::memcmp(a.data, b.data, a.bytes()) == 0, "重复执行改变了输出");
        YI_CHECK(reused.workspace_bytes() == workspace, "执行期间工作区大小变化");
    }
    std::printf("PASS %s repeated-run workspace=%zu\n", backend_name(backend), workspace);
}

// 收尾融合：标量下与未融合逐位一致；SIMD 下 naive/reuse 逐位一致、与标量在容差内。
void check_fused(Backend backend) {
    Model fused_scalar_model = toy_model(true), unfused_scalar_model = toy_model(false);
    Model naive_model = toy_model(true), reused_model = toy_model(true);
    Executor fused_scalar(fused_scalar_model, true, Backend::Scalar);
    Executor unfused_scalar(unfused_scalar_model, true, Backend::Scalar);
    Executor naive(naive_model, false, backend), reused(reused_model, true, backend);
    std::vector<float> input(2 * 3 * 5 * 7);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(static_cast<float>(i) * 0.17f) * 3;
    fused_scalar.set_input(0, input.data());
    unfused_scalar.set_input(0, input.data());
    naive.set_input(0, input.data());
    reused.set_input(0, input.data());
    fused_scalar.run(); unfused_scalar.run(); naive.run(); reused.run();
    const Tensor& fused_ref = fused_scalar_model.tensors[fused_scalar_model.find("y")];
    const Tensor& unfused = unfused_scalar_model.tensors[unfused_scalar_model.find("y")];
    YI_CHECK(std::memcmp(fused_ref.data, unfused.data, fused_ref.bytes()) == 0,
             "标量下融合与未融合不逐位一致");
    const Tensor& a = naive_model.tensors[naive_model.find("y")];
    const Tensor& b = reused_model.tensors[reused_model.find("y")];
    YI_CHECK(std::memcmp(a.data, b.data, a.bytes()) == 0, "融合后 SIMD 的 naive/reuse 不逐位一致");
    for (int64_t i = 0; i < b.numel(); ++i) {
        YI_CHECK(std::isfinite(b.data[i]), "融合后 SIMD 产生非有限输出");
        YI_CHECK(std::fabs(b.data[i] - fused_ref.data[i]) <= 1e-5f + 1e-4f * std::fabs(fused_ref.data[i]),
                 "融合后 SIMD 与标量逐元素误差超限");
    }
    std::printf("PASS %s fused-epilogue\n", backend_name(backend));
}
// 多线程检查用的模型：一个足够大的 3×3 卷积（K*N 超过 im2col 并行阈值，列块数远超线程数）。
Model big_conv_model() {
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
    add("x", {1, 3, 64, 64});
    add("w", {8, 3, 3, 3}, 0);
    add("b", {8}, 864);
    add("y", {1, 8, 64, 64});
    m.weights = AlignedBuffer(896);
    for (auto& t : m.tensors) {
        if (!t.is_const) continue;
        t.data = reinterpret_cast<float*>(m.weights.as<char>() + t.const_offset);
        for (int64_t i = 0; i < t.numel(); ++i) t.data[i] = static_cast<float>((i * 5 + 1) % 23 - 11) * 0.03f;
    }
    Node c;
    c.name = "conv"; c.op = "Conv"; c.inputs = {0, 1, 2}; c.outputs = {3};
    c.attrs.kv = {{"kernel", {3, 3}}, {"stride", {1, 1}}, {"pad", {1, 1, 1, 1}}, {"act", {1}}};
    m.nodes = {c};
    m.inputs = {0}; m.outputs = {3};
    m.validate();
    return m;
}

void run_and_copy(Model& model, Executor& ex, const std::vector<float>& input, std::vector<float>& out) {
    ex.set_input(0, input.data());
    ex.run();
    const Tensor& t = model.tensors[model.find("y")];
    out.assign(t.data, t.data + t.numel());
}

// 多线程：任意线程数的输出与单线程逐位一致；线程数超过列块数、naive 规划也不得改变结果。
void check_threads(Backend backend) {
    Model single_model = toy_model(true), quad_model = toy_model(true), many_model = toy_model(true);
    Executor single(single_model, true, backend, 1);
    Executor quad(quad_model, true, backend, 4);
    Executor many(many_model, true, backend, 16);   // 线程数远多于列块数：多余线程领不到工作单元
    YI_CHECK(quad.workspace_bytes() > single.workspace_bytes(), "多线程没有为每个线程留面板");
    YI_CHECK(many.workspace_bytes() > quad.workspace_bytes(), "面板槽位不随线程数增长");
    std::vector<float> input(2 * 3 * 5 * 7), a, b, c;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(static_cast<float>(i) * 0.11f) * 2;
    run_and_copy(single_model, single, input, a);
    run_and_copy(quad_model, quad, input, b);
    run_and_copy(many_model, many, input, c);
    YI_CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "4 线程与单线程输出不逐位一致");
    YI_CHECK(std::memcmp(a.data(), c.data(), a.size() * sizeof(float)) == 0, "16 线程与单线程输出不逐位一致");

    // 大卷积：并行 im2col（K*N 超阈值）与 256 个列块；naive 规划在多线程下同样逐位一致。
    Model big_single = big_conv_model(), big_quad = big_conv_model(), big_naive = big_conv_model();
    Executor bs(big_single, true, backend, 1), bq(big_quad, true, backend, 4), bn(big_naive, false, backend, 4);
    std::vector<float> big_input(3 * 64 * 64), big_a, big_b, big_c;
    for (size_t i = 0; i < big_input.size(); ++i) big_input[i] = std::sin(static_cast<float>(i) * 0.07f);
    run_and_copy(big_single, bs, big_input, big_a);
    run_and_copy(big_quad, bq, big_input, big_b);
    run_and_copy(big_naive, bn, big_input, big_c);
    YI_CHECK(std::memcmp(big_a.data(), big_b.data(), big_a.size() * sizeof(float)) == 0,
             "大卷积多线程输出不逐位一致");
    YI_CHECK(std::memcmp(big_a.data(), big_c.data(), big_a.size() * sizeof(float)) == 0,
             "多线程 naive 规划输出不逐位一致");
    std::printf("PASS %s threads=1/4/16 workspace=%zu/%zu/%zu\n", backend_name(backend),
                single.workspace_bytes(), quad.workspace_bytes(), many.workspace_bytes());
}
}  // namespace

int main() {
    try {
        for (Backend backend : {Backend::SSE, Backend::AVX2}) {
            if (backend_available(backend)) {
                check_backend(backend);
                check_fused(backend);
                check_threads(backend);
            } else {
                std::printf("SKIP %s unavailable\n", backend_name(backend));
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
