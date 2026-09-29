// yinfer.cpp —— 命令行入口
//   yinfer info   <模型目录>                          模型概况：算子统计、权重和激活内存、卷积计算量
//   yinfer run    <模型目录> <input.bin>              整图执行一遍；有算子没实现会列出来
//   yinfer verify <模型目录> <参考数据目录> [选项]      逐层和 ORT 对拍
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "executor.h"
#include "reference.h"

using namespace yi;

namespace {

void usage() {
    std::printf(
        "用法:\n"
        "  yinfer info   <模型目录>\n"
        "  yinfer run    <模型目录> <input.bin>\n"
        "  yinfer verify <模型目录> <参考数据目录> [--mode isolated|chained] [--tol 1e-4] [--perturb 节点号] [--brief]\n"
        "\n"
        "verify 的两种模式:\n"
        "  isolated  每个节点都从参考数据取输入，只量这一层自己引入的误差（默认）\n"
        "  chained   节点吃前面节点自己算出的结果，误差一路累积，量的是端到端误差\n"
        "还没实现的算子直接用参考数据顶上（状态 REF），所以任何阶段都能把整张图走完\n"
        "--perturb N  执行完第 N 个节点后故意改坏它的输出，用来确认对拍工具真能抓到错\n"
        "--brief      不打印 REF 行\n"
        "--memory naive|reuse  验证时选择内存规划，默认 reuse（不是原地算子）\n");
}

double mib(size_t bytes) { return bytes / 1024.0 / 1024.0; }

void print_missing(const Executor& ex) {
    const auto miss = ex.missing_kernels();
    if (miss.empty()) {
        std::printf("  算子全部已实现\n");
        return;
    }
    std::printf("  还没实现的算子:");
    for (const auto& [op, c] : miss) std::printf(" %s×%d", op.c_str(), c);
    std::printf("\n");
}

int cmd_info(const std::string& model_dir) {
    Model m = Model::load(model_dir);
    Executor ex(m);
    std::map<std::string, int> ops;
    for (const Node& n : m.nodes) ++ops[n.op];
    int n_const = 0, n_act = 0;
    for (const Tensor& t : m.tensors) ++(t.is_const ? n_const : n_act);
    double flops = 0;
    for (const Node& n : m.nodes) {
        if (n.op != "Conv") continue;
        const Tensor& w = m.tensors[n.inputs[1]];
        const Tensor& y = m.tensors[n.outputs[0]];
        flops += 2.0 * w.numel() * y.shape[2] * y.shape[3];   // 每个输出点做 Cin·kh·kw 次乘加，乘和加各算一次
    }
    std::printf("模型 %s\n  节点 %zu 个:", model_dir.c_str(), m.nodes.size());
    for (const auto& [op, c] : ops) std::printf(" %s×%d", op.c_str(), c);
    std::printf("\n  张量 %zu 个：常量 %d 个（权重 %.2f MiB），激活 %d 个（当前内存规划 %.1f MiB）\n",
                m.tensors.size(), n_const, mib(m.weights.bytes()), n_act, mib(ex.arena_bytes()));
    for (int i : m.inputs) std::printf("  输入 %-16s %s\n", m.tensors[i].shape_str().c_str(), m.tensors[i].name.c_str());
    for (int i : m.outputs) std::printf("  输出 %-16s %s\n", m.tensors[i].shape_str().c_str(), m.tensors[i].name.c_str());
    std::printf("  激活规划：reuse %.2f MiB；naive 对照 %.2f MiB（不含权重、参考数据与进程其他内存）\n",
                mib(ex.arena_bytes()), mib(plan_naive(m).total));
    std::printf("  卷积计算量 %.2f GFLOP\n", flops / 1e9);
    print_missing(ex);
    return 0;
}

int cmd_run(const std::string& model_dir, const std::string& input_path) {
    Model m = Model::load(model_dir);
    Executor ex(m);
    if (!ex.missing_kernels().empty()) {
        std::printf("整图还跑不了（可以先用 yinfer verify 对拍已经实现的部分）\n");
        print_missing(ex);
        return 2;
    }
    const Tensor& in = m.tensors[m.inputs[0]];
    const AlignedBuffer x = read_file(input_path, static_cast<int64_t>(in.bytes()));
    ex.set_input(0, x.as<float>());
    const auto t0 = std::chrono::steady_clock::now();
    ex.run();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int i : m.outputs) {
        const Tensor& t = m.tensors[i];
        double lo = INFINITY, hi = -INFINITY, sum = 0;
        for (int64_t k = 0; k < t.numel(); ++k) {
            lo = std::fmin(lo, t.data[k]);
            hi = std::fmax(hi, t.data[k]);
            sum += t.data[k];
        }
        std::printf("  %-14s min %9.4f  max %9.4f  mean %9.4f  %s\n", t.shape_str().c_str(), lo, hi, sum / t.numel(), t.name.c_str());
    }
    std::printf("整图一次 %.2f ms（单次，含冷启动）\n", ms);
    return 0;
}

struct VerifyOptions {
    bool isolated = true;
    double tol = 1e-4;          // 相对误差 = max|d| / max|参考值| 的上限
    long perturb = -1;
    bool brief = false;
    bool reuse = true;
};

// 把输出里绝对值最大的元素加上它自己的 1%：故意制造一个错误，确认对拍工具真的抓得到（验证"验证器"本身）
void perturb(Tensor& t) {
    int64_t k = 0;
    float mx = 0;
    for (int64_t i = 0; i < t.numel(); ++i)
        if (std::fabs(t.data[i]) > mx) {
            mx = std::fabs(t.data[i]);
            k = i;
        }
    t.data[k] += 0.01f * (mx > 0 ? mx : 1.0f);
}

int cmd_verify(const std::string& model_dir, const std::string& ref_dir, const VerifyOptions& opt) {
    Model m = Model::load(model_dir);
    Executor ex(m, opt.reuse);
    const Reference ref = Reference::load(ref_dir);
    for (const Tensor& t : m.tensors) {                     // 每个激活都要有参考数据，并且形状一致
        if (t.is_const) continue;
        const auto& e = ref.at(t.name);
        YI_CHECK(e.shape == t.shape_str(), t.name << " 的形状：模型里 " << t.shape_str() << "，参考数据里 " << e.shape);
    }
    for (size_t k = 0; k < m.inputs.size(); ++k) ex.set_input(k, ref.at(m.tensors[m.inputs[k]].name).data);
    auto load_ref = [&](int ti) {
        Tensor& t = m.tensors[ti];
        std::memcpy(t.data, ref.at(t.name).data, t.bytes());
    };

    std::printf("激活内存：%s %.2f MiB\n", opt.reuse ? "reuse" : "naive", mib(ex.arena_bytes()));
    std::printf("逐层对拍：%s，通过标准 rel = max|d| / max|ref| <= %.0e\n",
                opt.isolated ? "isolated 模式（每层输入都用参考数据）" : "chained 模式（误差逐层累积）", opt.tol);
    std::printf("%4s  %-15s %-14s %-6s %8s %10s %10s %12s  %s\n", "idx", "op", "shape", "status", "exact", "max|d|", "rel", "cos", "tensor");
    std::map<std::string, int> count;
    double worst_rel = -1;
    std::string worst;
    for (size_t i = 0; i < m.nodes.size(); ++i) {
        const Node& n = m.nodes[i];
        if (opt.isolated)
            for (int ti : n.inputs)
                if (!m.tensors[ti].is_const) load_ref(ti);
        const bool computed = ex.has_kernel(i);
        if (computed)
            ex.run_node(i);
        else
            for (int to : n.outputs) load_ref(to);          // 没实现：用参考数据顶上，让后面的节点照常执行
        const bool perturbed = static_cast<long>(i) == opt.perturb;
        if (perturbed) perturb(m.tensors[n.outputs[0]]);

        for (size_t k = 0; k < n.outputs.size(); ++k) {
            const Tensor& t = m.tensors[n.outputs[k]];
            const std::string idx = k == 0 ? std::to_string(i) : "";
            const std::string op = k == 0 ? n.op + (perturbed ? "*" : "") : "";
            if (!computed && !perturbed) {
                ++count["REF"];
                if (!opt.brief)
                    std::printf("%4s  %-15s %-14s %-6s %8s %10s %10s %12s  %s\n", idx.c_str(), op.c_str(), t.shape_str().c_str(),
                                "REF", "-", "-", "-", "-", t.name.c_str());
                continue;
            }
            const Diff d = compare(t.data, ref.at(t.name).data, t.numel());
            // 即使逐位相同，NaN/Inf 也不能作为数值验证通过的依据。
            const char* status = "FAIL";
            if (d.finite) {
                if (d.exact == d.n) status = "EXACT";
                else if (d.rel() <= opt.tol) status = "OK";
            }
            ++count[status];
            if (!d.finite || d.rel() > worst_rel) {
                worst_rel = d.finite ? d.rel() : INFINITY;
                worst = t.name;
            }
            // 40 万个元素里错 1 个是 99.9998%，按 %.2f 会显示成 100.00%。只要不是全对，最多显示 99.99%
            const double exact_pct = d.exact == d.n ? 100.0 : std::fmin(100.0 * d.exact / d.n, 99.99);
            std::printf("%4s  %-15s %-14s %-6s %7.2f%% %10.3e %10.3e %12.9f  %s\n", idx.c_str(), op.c_str(), t.shape_str().c_str(),
                        status, exact_pct, d.max_abs, d.rel(), d.cos, t.name.c_str());
        }
    }
    std::printf("\n%zu 个节点、%d 个输出张量：", m.nodes.size(), count["EXACT"] + count["OK"] + count["FAIL"] + count["REF"]);
    for (const char* s : {"EXACT", "OK", "FAIL", "REF"}) std::printf(" %s %d", s, count[s]);
    std::printf("\n");
    if (worst_rel >= 0) std::printf("相对误差最大的张量: %s（rel %.3e）\n", worst.c_str(), worst_rel);
    print_missing(ex);
    return count["FAIL"] > 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            usage();
            return 1;
        }
        const std::string cmd = argv[1];
        if (cmd == "info") return cmd_info(argv[2]);
        if (cmd == "run" && argc == 4) return cmd_run(argv[2], argv[3]);
        if (cmd == "verify" && argc >= 4) {
            VerifyOptions opt;
            for (int i = 4; i < argc; ++i) {
                const std::string a = argv[i];
                auto value = [&]() -> std::string {
                    YI_CHECK(i + 1 < argc, a << " 后面缺参数");
                    return argv[++i];
                };
                if (a == "--mode") {
                    const std::string v = value();
                    YI_CHECK(v == "isolated" || v == "chained", "--mode 只能是 isolated 或 chained");
                    opt.isolated = v == "isolated";
                } else if (a == "--memory") {
                    const std::string v = value();
                    YI_CHECK(v == "naive" || v == "reuse", "--memory 只能是 naive 或 reuse");
                    opt.reuse = v == "reuse";
                } else if (a == "--tol") {
                    opt.tol = std::stod(value());
                } else if (a == "--perturb") {
                    opt.perturb = std::stol(value());
                } else if (a == "--brief") {
                    opt.brief = true;
                } else {
                    usage();
                    return 1;
                }
            }
            return cmd_verify(argv[2], argv[3], opt);
        }
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "错误: %s\n", e.what());
        return 3;
    }
}
