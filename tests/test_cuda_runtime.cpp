// test_cuda_runtime.cpp —— CUDA 后端：七种算子、显存上的视图/原地、主机回读，与标量后端容差对照。
// 没有 CUDA 的构建或没有设备时打印 SKIP 并返回 0，不冒充通过。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "executor.h"

namespace {
using namespace yi;

// x → Conv3x3(act) → c → Split(4|4) → a,b（视图）→ Add → s → MaxPool → m → Upsample×2 → u
//                        c → SiLU → t → SiLU → t2（t2 原地接管 t）
// y = Concat(u, u)；输出 y、t2。七种算子全部覆盖。
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
    add("x", {1, 4, 16, 16});
    add("w", {8, 4, 3, 3}, 0);       // 288 floats = 1152 字节
    add("b", {8}, 1152);
    add("c", {1, 8, 16, 16});
    add("a", {1, 4, 16, 16});
    add("bb", {1, 4, 16, 16});
    add("s", {1, 4, 16, 16});
    add("m", {1, 4, 16, 16});
    add("u", {1, 4, 32, 32});
    add("t", {1, 8, 16, 16});
    add("t2", {1, 8, 16, 16});
    add("y", {1, 8, 32, 32});
    m.weights = AlignedBuffer(1216);
    for (auto& t : m.tensors) {
        if (!t.is_const) continue;
        t.data = reinterpret_cast<float*>(m.weights.as<char>() + t.const_offset);
        for (int64_t i = 0; i < t.numel(); ++i) t.data[i] = static_cast<float>((i * 11 + 5) % 17 - 8) * 0.05f;
    }
    auto node = [&](const char* op, const char* name, std::vector<int> ins, std::vector<int> outs,
                    std::map<std::string, std::vector<double>> attrs) {
        Node n;
        n.op = op; n.name = name; n.inputs = std::move(ins); n.outputs = std::move(outs);
        n.attrs.kv = std::move(attrs);
        m.nodes.push_back(std::move(n));
    };
    const auto id = [&](const char* name) { return m.find(name); };
    node("Conv", "conv", {id("x"), id("w"), id("b")}, {id("c")},
         {{"kernel", {3, 3}}, {"stride", {1, 1}}, {"pad", {1, 1, 1, 1}}, {"act", {1}}});
    node("Split", "split", {id("c")}, {id("a"), id("bb")}, {{"axis", {1}}, {"sizes", {4, 4}}});
    node("Add", "add", {id("a"), id("bb")}, {id("s")}, {});
    node("MaxPool", "pool", {id("s")}, {id("m")}, {{"kernel", {3, 3}}, {"stride", {1, 1}}, {"pad", {1, 1, 1, 1}}});
    node("UpsampleNearest", "up", {id("m")}, {id("u")}, {{"scale", {2, 2}}});
    node("SiLU", "silu", {id("c")}, {id("t")}, {});
    node("SiLU", "silu2", {id("t")}, {id("t2")}, {});
    node("Concat", "concat", {id("u"), id("u")}, {id("y")}, {{"axis", {1}}});
    m.inputs = {id("x")};
    m.outputs = {id("y"), id("t2")};
    m.validate();
    return m;
}

void compare(const char* what, const float* got, const float* ref, int64_t n, double tol) {
    double max_abs = 0, ref_max = 0;
    for (int64_t i = 0; i < n; ++i) {
        YI_CHECK(std::isfinite(got[i]), what << " 出现非有限值");
        max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) - ref[i]));
        ref_max = std::fmax(ref_max, std::fabs(ref[i]));
    }
    const double rel = ref_max > 0 ? max_abs / ref_max : max_abs;
    YI_CHECK(rel <= tol, what << " 相对误差 " << rel << " 超过 " << tol);
    std::printf("  %s rel=%.3e\n", what, rel);
}
}  // namespace

int main() {
    try {
        if (!backend_available(Backend::CUDA)) {
            std::printf("SKIP cuda unavailable\n");
            return 0;
        }
        Model scalar_model = toy_model(), cuda_model = toy_model();
        Executor scalar(scalar_model, true, Backend::Scalar);
        Executor cuda(cuda_model, true, Backend::CUDA);
        YI_CHECK(cuda.missing_kernels().empty(), "CUDA 后端缺少算子实现");
        YI_CHECK(cuda.view_tensors() == 2, "Split 输出在 CUDA 后端上没有做成视图");
        YI_CHECK(cuda.in_place_tensors() >= 1, "CUDA 后端上没有产生原地覆盖");
        YI_CHECK(cuda.workspace_bytes() == 0, "CUDA 后端不应申请主机工作区");
        // 视图的设备指针必须指进父张量：a 在 c 开头，bb 在 c 的第 4 个通道。
        const Tensor& c = cuda_model.tensors[cuda_model.find("c")];
        YI_CHECK(cuda_model.tensors[cuda_model.find("a")].data == c.data, "视图 a 没有指进父张量");
        YI_CHECK(cuda_model.tensors[cuda_model.find("bb")].data == c.data + 4 * 16 * 16, "视图 bb 的偏移不对");

        std::vector<float> input(4 * 16 * 16);
        for (int round = 0; round < 2; ++round) {
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(static_cast<float>(i) * 0.05f + round) * 2.0f;
            scalar.set_input(0, input.data());
            cuda.set_input(0, input.data());
            scalar.run();
            cuda.run();
            for (int o : scalar_model.outputs) {
                const Tensor& t = scalar_model.tensors[o];
                compare(t.name.c_str(), cuda.host_ptr(static_cast<size_t>(cuda_model.find(t.name))),
                        scalar.host_ptr(static_cast<size_t>(o)), t.numel(), 1e-4);
            }
            // 中间张量也能回读，并且视图读回的就是父张量的切片。
            const float* host_c = cuda.host_ptr(static_cast<size_t>(cuda_model.find("c")));
            const float* host_bb = cuda.host_ptr(static_cast<size_t>(cuda_model.find("bb")));
            YI_CHECK(std::memcmp(host_bb, host_c + 4 * 16 * 16, 4 * 16 * 16 * sizeof(float)) == 0,
                     "视图回读的内容不等于父张量对应切片");
        }
        // set_tensor / push_tensor 的 H2D/D2H 路径。注意契约：图输入的槽位在最后一次被读之后
        // 可以被后续激活复用，所以对输入的回读必须在 run 之前做。
        const auto x_id = static_cast<size_t>(cuda_model.find("x"));
        std::vector<float> zeros(input.size(), 0.0f);
        cuda.set_tensor(x_id, zeros.data());
        YI_CHECK(cuda.host_ptr(x_id)[0] == 0.0f, "set_tensor 没有写进设备");
        float* host_x = cuda.host_ptr(x_id);
        host_x[0] = 1.0f;
        cuda.push_tensor(x_id);
        YI_CHECK(cuda.host_ptr(x_id)[0] == 1.0f, "push_tensor 没有写进设备");
        cuda.run();   // 改过的输入也能正常算完
        std::printf("PASS cuda runtime\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
