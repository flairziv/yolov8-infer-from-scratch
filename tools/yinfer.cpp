// yinfer.cpp —— 命令行入口
//   yinfer info   <模型目录>                          模型概况：算子统计、权重和激活内存、卷积计算量
//   yinfer run    <模型目录> <input.bin>              整图执行一遍；有算子没实现会列出来
//   yinfer verify <模型目录> <参考数据目录> [选项]      逐层和 ORT 对拍
//   yinfer detect <模型目录> <input.bin> [选项]        DFL 解码 + 类内 NMS，输出检测框
//   yinfer chain  <chain.txt> <input.bin> [选项]       多段串联执行（split_model.py 切出来的段）
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cuda/cuda_api.h"
#include "executor.h"
#include "postprocess.h"
#include "reference.h"

using namespace yi;

namespace {

void usage() {
    std::printf(
        "用法:\n"
        "  yinfer backends\n"
        "  yinfer info   <模型目录> [--backend scalar|sse|avx2|cuda] [--threads N]\n"
        "  yinfer run    <模型目录> <input.bin> [--backend scalar|sse|avx2|cuda] [--memory naive|reuse] [--threads N] [--repeat N] [--dump-dir 目录]\n"
        "  yinfer verify <模型目录> <参考数据目录> [--backend scalar|sse|avx2|cuda] [--mode isolated|chained] [--tol 1e-4] [--threads N] [--perturb 节点号] [--brief]\n"
        "  yinfer detect <模型目录> <input.bin> [--conf 0.25] [--iou 0.7] [--backend scalar|sse|avx2|cuda] [--threads N] [--letterbox 缩放,左右填充,上下填充] [--out 文件]\n"
        "  yinfer chain  <chain.txt> <input.bin> [--backend b 或 b1,b2,...（每段一个）] [--memory naive|reuse] [--threads N] [--repeat N] [--dump-dir 目录]\n"
        "\n"
        "verify 的两种模式:\n"
        "  isolated  每个节点都从参考数据取输入，只量这一层自己引入的误差（默认）\n"
        "  chained   节点吃前面节点自己算出的结果，误差一路累积，量的是端到端误差\n"
        "还没实现的算子直接用参考数据顶上（状态 REF），所以任何阶段都能把整张图走完\n"
        "--threads N  算子内部数据并行（默认 1）。工作单元彼此独立，任意线程数的输出与单线程逐位一致\n"
        "--perturb N  执行完第 N 个节点后故意改坏它的输出，用来确认对拍工具真能抓到错\n"
        "--brief      不打印 REF 行\n"
        "--memory naive|reuse  验证时选择内存规划，默认 reuse（不是原地算子）\n"
        "\n"
        "detect 输出的是框坐标；给了 --letterbox 就换算回原图坐标，否则是 letterbox 后的输入图坐标\n");
}

double mib(size_t bytes) { return bytes / 1024.0 / 1024.0; }

int parse_threads(const std::string& v) {
    size_t pos = 0;
    int threads = 0;
    try {
        threads = std::stoi(v, &pos);
    } catch (const std::exception&) {
        YI_CHECK(false, "--threads 不是整数: " << v);
    }
    YI_CHECK(pos == v.size() && threads >= 1 && threads <= 1024, "--threads 需要是 1..1024 的整数，收到 " << v);
    return threads;
}

float parse_float(const std::string& v, const char* what) {
    size_t pos = 0;
    float f = 0;
    try {
        f = std::stof(v, &pos);
    } catch (const std::exception&) {
        YI_CHECK(false, what << " 不是数: " << v);
    }
    YI_CHECK(pos == v.size() && std::isfinite(f), what << " 不是合法的数: " << v);
    return f;
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (true) {
        const size_t comma = s.find(',', begin);
        out.push_back(s.substr(begin, comma == std::string::npos ? comma : comma - begin));
        if (comma == std::string::npos) return out;
        begin = comma + 1;
    }
}

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

void print_backend(const Executor& ex) {
    std::printf("backend=%s threads=%d workspace_bytes=%zu simd_nodes=%zu fallback_nodes=%zu views=%zu in_place=%zu\n",
                backend_name(ex.backend()), ex.threads(), ex.workspace_bytes(), ex.simd_nodes(), ex.fallback_nodes(),
                ex.view_tensors(), ex.in_place_tensors());
    if (ex.backend() == Backend::CUDA) std::printf("device=%s arena_bytes=%zu\n", cuda::device_name(), ex.arena_bytes());
}

int cmd_info(const std::string& model_dir, Backend backend, int threads) {
    Model m = Model::load(model_dir);
    Executor ex(m, true, backend, threads);
    print_backend(ex);
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
    std::printf("  激活规划：reuse %.2f MiB（同时存活峰值下界 %.2f MiB）；naive 对照 %.2f MiB（不含权重、参考数据与进程其他内存）\n",
                mib(ex.arena_bytes()), mib(peak_live_bytes(m)), mib(plan_naive(m).total));
    std::printf("  卷积计算量 %.2f GFLOP\n", flops / 1e9);
    print_missing(ex);
    return 0;
}

void dump_outputs(Executor& ex, Model& model, const std::string& dir) {
    std::filesystem::create_directories(dir);
    std::ofstream index(std::filesystem::path(dir) / "outputs.txt");
    YI_CHECK(index, "无法创建输出索引 " << dir);
    for (size_t i = 0; i < model.outputs.size(); ++i) {
        const Tensor& t = model.tensors[model.outputs[i]];
        const float* data = ex.host_ptr(static_cast<size_t>(model.outputs[i]));
        const std::string file = "output" + std::to_string(i) + ".bin";
        std::ofstream out(std::filesystem::path(dir) / file, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(t.bytes()));
        YI_CHECK(out, "输出文件写入失败 " << file);
        index << file << " " << t.name << " ";
        for (size_t d = 0; d < t.shape.size(); ++d) index << (d ? "," : "") << t.shape[d];
        index << "\n";
    }
    YI_CHECK(index, "输出索引写入失败");
}

int cmd_run(const std::string& model_dir, const std::string& input_path,
            Backend backend, bool reuse, int threads, int repeat, const std::string& dump_dir) {
    Model m = Model::load(model_dir);
    Executor ex(m, reuse, backend, threads);
    print_backend(ex);
    if (!ex.missing_kernels().empty()) {
        std::printf("整图还跑不了（可以先用 yinfer verify 对拍已经实现的部分）\n");
        print_missing(ex);
        return 2;
    }
    YI_CHECK(m.inputs.size() == 1, "run 命令当前只接受一个图输入");
    const Tensor& in = m.tensors[m.inputs[0]];
    const AlignedBuffer x = read_file(input_path, static_cast<int64_t>(in.bytes()));
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
        ex.set_input(0, x.as<float>());   // 每次 run 前重新设置输入（输入槽位可能已被复用）
        const auto t0 = std::chrono::steady_clock::now();
        ex.run();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    for (int i : m.outputs) {
        const Tensor& t = m.tensors[i];
        const float* data = ex.host_ptr(static_cast<size_t>(i));
        double lo = INFINITY, hi = -INFINITY, sum = 0;
        for (int64_t k = 0; k < t.numel(); ++k) {
            lo = std::fmin(lo, data[k]);
            hi = std::fmax(hi, data[k]);
            sum += data[k];
        }
        std::printf("  %-14s min %9.4f  max %9.4f  mean %9.4f  %s\n", t.shape_str().c_str(), lo, hi, sum / t.numel(), t.name.c_str());
    }
    std::printf("整图一次 %.6f ms（单次，含冷启动）\n", times[0]);
    if (repeat > 1) {
        // 去掉第一次（冷启动：缓存、页映射、CUDA 模块加载），报告其余各次的中位数与最小值。
        std::vector<double> warm(times.begin() + 1, times.end());
        std::sort(warm.begin(), warm.end());
        const double median = warm.size() % 2 ? warm[warm.size() / 2]
                                              : 0.5 * (warm[warm.size() / 2 - 1] + warm[warm.size() / 2]);
        std::printf("热身后 %zu 次中位数 %.6f ms（最小 %.6f ms）\n", warm.size(), median, warm.front());
    }
    if (!dump_dir.empty()) dump_outputs(ex, m, dump_dir);
    return 0;
}

struct VerifyOptions {
    bool isolated = true;
    double tol = 1e-4;          // 相对误差 = max|d| / max|参考值| 的上限
    long perturb = -1;
    bool brief = false;
    bool reuse = true;
    Backend backend = Backend::Scalar;
    int threads = 1;
};

// 把输出里绝对值最大的元素加上它自己的 1%：故意制造一个错误，确认对拍工具真的抓得到（验证"验证器"本身）。
// 通过执行器读写，CUDA 后端下也会正确写进显存。
void perturb(Executor& ex, Tensor& t, size_t id) {
    float* data = ex.host_ptr(id);
    int64_t k = 0;
    float mx = 0;
    for (int64_t i = 0; i < t.numel(); ++i)
        if (std::fabs(data[i]) > mx) {
            mx = std::fabs(data[i]);
            k = i;
        }
    data[k] += 0.01f * (mx > 0 ? mx : 1.0f);
    ex.push_tensor(id);
}

int cmd_verify(const std::string& model_dir, const std::string& ref_dir, const VerifyOptions& opt) {
    Model m = Model::load(model_dir);
    Executor ex(m, opt.reuse, opt.backend, opt.threads);
    print_backend(ex);
    const Reference ref = Reference::load(ref_dir);
    for (const Tensor& t : m.tensors) {                     // 每个激活都要有参考数据，并且形状一致
        if (t.is_const) continue;
        const auto& e = ref.at(t.name);
        YI_CHECK(e.shape == t.shape_str(), t.name << " 的形状：模型里 " << t.shape_str() << "，参考数据里 " << e.shape);
    }
    for (size_t k = 0; k < m.inputs.size(); ++k) ex.set_input(k, ref.at(m.tensors[m.inputs[k]].name).data);
    auto load_ref = [&](int ti) { ex.set_tensor(static_cast<size_t>(ti), ref.at(m.tensors[ti].name).data); };

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
        if (perturbed) perturb(ex, m.tensors[n.outputs[0]], static_cast<size_t>(n.outputs[0]));

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
            const Diff d = compare(ex.host_ptr(static_cast<size_t>(n.outputs[k])), ref.at(t.name).data, t.numel());
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

struct DetectOptions {
    float conf = 0.25f;
    float iou = 0.7f;
    Backend backend = Backend::Scalar;
    int threads = 1;
    bool letterbox = false;
    float scale = 1.0f, pad_x = 0.0f, pad_y = 0.0f;
    std::string out;
};

int cmd_detect(const std::string& model_dir, const std::string& input_path, const DetectOptions& opt) {
    Model m = Model::load(model_dir);
    Executor ex(m, true, opt.backend, opt.threads);
    print_backend(ex);
    if (!ex.missing_kernels().empty()) {
        std::printf("整图还跑不了（可以先用 yinfer verify 对拍已经实现的部分）\n");
        print_missing(ex);
        return 2;
    }
    YI_CHECK(m.inputs.size() == 1, "detect 命令当前只接受一个图输入");
    const Tensor& in = m.tensors[m.inputs[0]];
    const AlignedBuffer x = read_file(input_path, static_cast<int64_t>(in.bytes()));
    ex.set_input(0, x.as<float>());
    const auto t0 = std::chrono::steady_clock::now();
    ex.run();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const auto decoded = dfl_decode(m, opt.conf, [&](int id) { return ex.host_ptr(static_cast<size_t>(id)); });
    const size_t decoded_count = decoded.size();
    auto boxes = nms(decoded, opt.iou);
    if (opt.letterbox) unletterbox(boxes, opt.scale, opt.pad_x, opt.pad_y);
    std::printf("检测到 %zu 个框（conf>=%.2f 留下 %zu 个，类内 iou>%.2f 抑制 %zu 个；坐标：%s）\n",
                boxes.size(), opt.conf, decoded_count, opt.iou, decoded_count - boxes.size(),
                opt.letterbox ? "原图" : "letterbox 后的输入图");
    std::printf("  %-5s %8s %9s %9s %9s %9s\n", "class", "score", "x1", "y1", "x2", "y2");
    for (const Detection& d : boxes)
        std::printf("  %-5d %8.4f %9.2f %9.2f %9.2f %9.2f\n", d.cls, d.score, d.x1, d.y1, d.x2, d.y2);
    if (!opt.out.empty()) {
        std::ofstream f(opt.out);
        YI_CHECK(f, "无法写入 " << opt.out);
        for (const Detection& d : boxes)
            f << d.cls << " " << d.score << " " << d.x1 << " " << d.y1 << " " << d.x2 << " " << d.y2 << "\n";
        YI_CHECK(f, "写 " << opt.out << " 失败");
        std::printf("框已写入 %s\n", opt.out.c_str());
    }
    std::printf("整图一次 %.6f ms（单次，含冷启动）\n", ms);
    return 0;
}

struct ChainOptions {
    std::vector<Backend> backends;   // 一个：所有段相同；多个：每段一个
    int threads = 1;
    bool reuse = true;
    int repeat = 1;
    std::string dump_dir;
};

// chain.txt：part <目录> ...（相对清单所在目录）；output <名字>（原图输出顺序）；其余行忽略。
struct ChainManifest {
    std::vector<std::string> parts;
    std::vector<std::string> outputs;
};

ChainManifest read_manifest(const std::string& path) {
    std::ifstream f(path);
    YI_CHECK(f, "打不开清单 " << path);
    ChainManifest m;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key, value;
        if (!(ss >> key) || key[0] == '#') continue;
        if (key == "part") {
            YI_CHECK(ss >> value, "清单的 part 行缺少目录");
            m.parts.push_back(value);
        } else if (key == "output") {
            YI_CHECK(ss >> value, "清单的 output 行缺少名字");
            m.outputs.push_back(value);
        }
    }
    YI_CHECK(!m.parts.empty() && !m.outputs.empty(), "清单里没有 part 或 output 行");
    return m;
}

// 多段串联：每段一个模型、一个执行器；段间边界张量按名字从上游段回读、写入下游段的输入。
// 各段内部执行顺序与整体模型相同，所以同后端下串联输出应与整体模型逐位一致。
int cmd_chain(const std::string& manifest_path, const std::string& input_path, const ChainOptions& opt) {
    const ChainManifest manifest = read_manifest(manifest_path);
    const std::filesystem::path base = std::filesystem::path(manifest_path).parent_path();
    YI_CHECK(opt.backends.size() == 1 || opt.backends.size() == manifest.parts.size(),
             "--backend 要么只给一个（各段相同），要么给每段一个，收到 " << opt.backends.size() << " 个、清单有 "
                                                                        << manifest.parts.size() << " 段");
    struct Part {
        Model model;
        std::unique_ptr<Executor> ex;
        Backend backend = Backend::Scalar;
    };
    std::vector<std::unique_ptr<Part>> parts;   // 堆上放置：执行器持有 Model 的引用，不能搬动
    std::map<std::string, std::pair<size_t, int>> produced;   // 名字 → (段号, 张量下标)
    for (size_t i = 0; i < manifest.parts.size(); ++i) {
        auto part = std::make_unique<Part>();
        part->backend = opt.backends.size() == 1 ? opt.backends[0] : opt.backends[i];
        part->model = Model::load((base / manifest.parts[i]).string());
        part->ex = std::make_unique<Executor>(part->model, opt.reuse, part->backend, opt.threads);
        YI_CHECK(part->ex->missing_kernels().empty(), "段 " << i << " 有未实现的算子");
        for (size_t k = 0; k < part->model.inputs.size(); ++k) {
            const Tensor& t = part->model.tensors[part->model.inputs[k]];
            if (i == 0) {
                YI_CHECK(part->model.inputs.size() == 1, "第一段应只有一个图输入");
            } else {
                const auto it = produced.find(t.name);
                YI_CHECK(it != produced.end(), "段 " << i << " 的输入 " << t.name << " 没有更早的段产生它");
                const Tensor& src = parts[it->second.first]->model.tensors[it->second.second];
                YI_CHECK(src.shape == t.shape, "边界张量 " << t.name << " 两端形状不一致");
            }
        }
        for (int o : part->model.outputs) produced[part->model.tensors[o].name] = {i, o};
        std::printf("part%zu: %s 节点 %zu 个 backend=%s arena=%.2f MiB\n", i, manifest.parts[i].c_str(),
                    part->model.nodes.size(), backend_name(part->backend), mib(part->ex->arena_bytes()));
        parts.push_back(std::move(part));
    }
    for (const auto& name : manifest.outputs)
        YI_CHECK(produced.count(name), "清单里的输出 " << name << " 没有任何段产生");

    const Tensor& in = parts[0]->model.tensors[parts[0]->model.inputs[0]];
    const AlignedBuffer x = read_file(input_path, static_cast<int64_t>(in.bytes()));
    std::vector<double> totals;
    std::vector<double> per_part(parts.size(), 0.0);
    for (int r = 0; r < opt.repeat; ++r) {
        double total = 0.0;
        for (size_t i = 0; i < parts.size(); ++i) {
            Part& part = *parts[i];
            const auto t0 = std::chrono::steady_clock::now();
            if (i == 0) {
                part.ex->set_input(0, x.as<float>());
            } else {
                for (size_t k = 0; k < part.model.inputs.size(); ++k) {
                    const auto& src = produced.at(part.model.tensors[part.model.inputs[k]].name);
                    part.ex->set_input(k, parts[src.first]->ex->host_ptr(static_cast<size_t>(src.second)));
                }
            }
            part.ex->run();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            per_part[i] = ms;
            total += ms;
        }
        totals.push_back(total);
    }
    for (size_t i = 0; i < parts.size(); ++i)
        std::printf("  part%zu %s %.6f ms（最后一次，含边界搬运）\n", i, backend_name(parts[i]->backend), per_part[i]);
    for (const auto& name : manifest.outputs) {
        const auto& src = produced.at(name);
        const Tensor& t = parts[src.first]->model.tensors[src.second];
        const float* data = parts[src.first]->ex->host_ptr(static_cast<size_t>(src.second));
        double lo = INFINITY, hi = -INFINITY, sum = 0;
        for (int64_t k = 0; k < t.numel(); ++k) {
            lo = std::fmin(lo, data[k]);
            hi = std::fmax(hi, data[k]);
            sum += data[k];
        }
        std::printf("  %-14s min %9.4f  max %9.4f  mean %9.4f  %s\n", t.shape_str().c_str(), lo, hi, sum / t.numel(), t.name.c_str());
    }
    std::printf("串联一次 %.6f ms（单次，含冷启动与边界搬运）\n", totals[0]);
    if (opt.repeat > 1) {
        std::vector<double> warm(totals.begin() + 1, totals.end());
        std::sort(warm.begin(), warm.end());
        const double median = warm.size() % 2 ? warm[warm.size() / 2]
                                              : 0.5 * (warm[warm.size() / 2 - 1] + warm[warm.size() / 2]);
        std::printf("热身后 %zu 次中位数 %.6f ms（最小 %.6f ms）\n", warm.size(), median, warm.front());
    }
    if (!opt.dump_dir.empty()) {
        std::filesystem::create_directories(opt.dump_dir);
        std::ofstream index(std::filesystem::path(opt.dump_dir) / "outputs.txt");
        YI_CHECK(index, "无法创建输出索引 " << opt.dump_dir);
        for (size_t i = 0; i < manifest.outputs.size(); ++i) {
            const auto& src = produced.at(manifest.outputs[i]);
            const Tensor& t = parts[src.first]->model.tensors[src.second];
            const float* data = parts[src.first]->ex->host_ptr(static_cast<size_t>(src.second));
            const std::string file = "output" + std::to_string(i) + ".bin";
            std::ofstream out(std::filesystem::path(opt.dump_dir) / file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(t.bytes()));
            YI_CHECK(out, "输出文件写入失败 " << file);
            index << file << " " << t.name << " ";
            for (size_t d = 0; d < t.shape.size(); ++d) index << (d ? "," : "") << t.shape[d];
            index << "\n";
        }
        YI_CHECK(index, "输出索引写入失败");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {    try {
        if (argc == 2 && std::string(argv[1]) == "backends") {
            for (Backend b : {Backend::Scalar, Backend::SSE, Backend::AVX2, Backend::CUDA})
                std::printf("%s %s\n", backend_name(b), backend_available(b) ? "available" : "unavailable");
            return 0;
        }
        if (argc < 3) {
            usage();
            return 1;
        }
        const std::string cmd = argv[1];
        if (cmd == "info") {
            Backend backend = Backend::Scalar;
            int threads = 1;
            for (int i = 3; i < argc; ++i) {
                const std::string a = argv[i];
                YI_CHECK(i + 1 < argc, a << " 后面缺参数");
                const std::string v = argv[++i];
                if (a == "--backend") backend = parse_backend(v);
                else if (a == "--threads") threads = parse_threads(v);
                else YI_CHECK(false, "info 只接受 --backend 与 --threads 参数");
            }
            return cmd_info(argv[2], backend, threads);
        }
        if (cmd == "run" && argc >= 4) {
            Backend backend = Backend::Scalar;
            bool reuse = true;
            int threads = 1;
            int repeat = 1;
            std::string dump_dir;
            for (int i = 4; i < argc; ++i) {
                const std::string a = argv[i];
                YI_CHECK(i + 1 < argc, a << " 后面缺参数");
                const std::string v = argv[++i];
                if (a == "--backend") backend = parse_backend(v);
                else if (a == "--dump-dir") dump_dir = v;
                else if (a == "--threads") threads = parse_threads(v);
                else if (a == "--repeat") {
                    repeat = parse_threads(v);
                    YI_CHECK(repeat <= 10000, "--repeat 最多 10000");
                }
                else if (a == "--memory") {
                    YI_CHECK(v == "naive" || v == "reuse", "--memory 只能是 naive 或 reuse");
                    reuse = v == "reuse";
                } else YI_CHECK(false, "run 不认识参数 " << a);
            }
            return cmd_run(argv[2], argv[3], backend, reuse, threads, repeat, dump_dir);
        }
        if (cmd == "verify" && argc >= 4) {
            VerifyOptions opt;
            for (int i = 4; i < argc; ++i) {
                const std::string a = argv[i];
                auto value = [&]() -> std::string {
                    YI_CHECK(i + 1 < argc, a << " 后面缺参数");
                    return argv[++i];
                };
                if (a == "--backend") {
                    opt.backend = parse_backend(value());
                } else if (a == "--mode") {
                    const std::string v = value();
                    YI_CHECK(v == "isolated" || v == "chained", "--mode 只能是 isolated 或 chained");
                    opt.isolated = v == "isolated";
                } else if (a == "--memory") {
                    const std::string v = value();
                    YI_CHECK(v == "naive" || v == "reuse", "--memory 只能是 naive 或 reuse");
                    opt.reuse = v == "reuse";
                } else if (a == "--threads") {
                    opt.threads = parse_threads(value());
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
        if (cmd == "detect" && argc >= 4) {
            DetectOptions opt;
            for (int i = 4; i < argc; ++i) {
                const std::string a = argv[i];
                auto value = [&]() -> std::string {
                    YI_CHECK(i + 1 < argc, a << " 后面缺参数");
                    return argv[++i];
                };
                if (a == "--conf") {
                    opt.conf = parse_float(value(), "--conf");
                } else if (a == "--iou") {
                    opt.iou = parse_float(value(), "--iou");
                } else if (a == "--backend") {
                    opt.backend = parse_backend(value());
                } else if (a == "--threads") {
                    opt.threads = parse_threads(value());
                } else if (a == "--out") {
                    opt.out = value();
                } else if (a == "--letterbox") {
                    const auto parts = split_commas(value());
                    YI_CHECK(parts.size() == 3, "--letterbox 需要 缩放,左右填充,上下填充");
                    opt.scale = parse_float(parts[0], "--letterbox 缩放");
                    opt.pad_x = parse_float(parts[1], "--letterbox 左右填充");
                    opt.pad_y = parse_float(parts[2], "--letterbox 上下填充");
                    opt.letterbox = true;
                } else {
                    usage();
                    return 1;
                }
            }
            return cmd_detect(argv[2], argv[3], opt);
        }
        if (cmd == "chain" && argc >= 4) {
            ChainOptions opt;
            opt.backends = {Backend::Scalar};
            for (int i = 4; i < argc; ++i) {
                const std::string a = argv[i];
                YI_CHECK(i + 1 < argc, a << " 后面缺参数");
                const std::string v = argv[++i];
                if (a == "--backend") {
                    opt.backends.clear();
                    for (const auto& name : split_commas(v)) opt.backends.push_back(parse_backend(name));
                } else if (a == "--threads") {
                    opt.threads = parse_threads(v);
                } else if (a == "--repeat") {
                    opt.repeat = parse_threads(v);
                    YI_CHECK(opt.repeat <= 10000, "--repeat 最多 10000");
                } else if (a == "--memory") {
                    YI_CHECK(v == "naive" || v == "reuse", "--memory 只能是 naive 或 reuse");
                    opt.reuse = v == "reuse";
                } else if (a == "--dump-dir") {
                    opt.dump_dir = v;
                } else {
                    usage();
                    return 1;
                }
            }
            return cmd_chain(argv[2], argv[3], opt);
        }
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "错误: %s\n", e.what());
        return 3;
    }
}
