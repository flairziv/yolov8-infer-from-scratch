// test_memory_plan.cpp —— 生命周期边界与执行一致性；无 gtest，Release 下检查也生效。
// 可选参数：test_memory_plan <模型目录> <input.bin>，逐字节比较真实模型的 naive/reuse 输出。
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "executor.h"
#include "memory_plan.h"

namespace {
using namespace yi;

int tensor(Model& m, const std::string& name, int64_t elements, bool constant = false) {
    Tensor t;
    t.name = name;
    t.shape = {elements};
    t.is_const = constant;
    const int id = static_cast<int>(m.tensors.size());
    m.index[name] = id;
    m.tensors.push_back(std::move(t));
    return id;
}

void node(Model& m, const std::string& op, std::vector<int> ins, std::vector<int> outs,
          std::vector<double> sizes = {}) {
    Node n;
    n.op = op;
    n.name = op + "_" + std::to_string(m.nodes.size());
    n.inputs = std::move(ins);
    n.outputs = std::move(outs);
    if (op == "Split" || op == "Concat") n.attrs.kv["axis"] = {0};
    if (op == "Split") n.attrs.kv["sizes"] = std::move(sizes);
    m.nodes.push_back(std::move(n));
}

bool overlaps(const Model& m, const MemoryPlan& p, int a, int b) {
    const auto ae = p.offset[a] + static_cast<int64_t>(align_up(m.tensors[a].bytes()));
    const auto be = p.offset[b] + static_cast<int64_t>(align_up(m.tensors[b].bytes()));
    return p.offset[a] < be && p.offset[b] < ae;
}

// 独立于生产 validate_plan，按题目契约检查所有同时存活的张量都不共用空间。
void check_contract(const Model& m, const MemoryPlan& p) {
    YI_CHECK(p.offset.size() == m.tensors.size(), "偏移表长度错误");
    const int horizon = static_cast<int>(m.nodes.size());
    std::vector<int> born(m.tensors.size(), -1), last(m.tensors.size(), -1);
    for (int i = 0; i < horizon; ++i) {
        for (int o : m.nodes[i].outputs) born[o] = last[o] = i;
        for (int in : m.nodes[i].inputs) last[in] = std::max(last[in], i);
    }
    for (int in : m.inputs) last[in] = horizon;
    for (int out : m.outputs) last[out] = horizon;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        if (m.tensors[i].is_const) {
            YI_CHECK(p.offset[i] == -1, "常量不应占用激活 arena");
            continue;
        }
        YI_CHECK(p.offset[i] >= 0 && p.offset[i] % kAlign == 0, "激活必须按 64 字节对齐");
        YI_CHECK(static_cast<size_t>(p.offset[i]) + align_up(m.tensors[i].bytes()) <= p.total, "激活超出 arena");
        for (size_t j = 0; j < i; ++j) {
            if (m.tensors[j].is_const) continue;
            if (std::max(born[i], born[j]) <= std::min(last[i], last[j]))
                YI_CHECK(!overlaps(m, p, static_cast<int>(i), static_cast<int>(j)),
                         "同时存活的张量重叠: " << m.tensors[i].name << "/" << m.tensors[j].name);
        }
    }
    validate_plan(m, p);
}

Model chain() {
    Model m;
    for (const char* name : {"x", "a", "b", "c", "y"}) tensor(m, name, 17);
    m.inputs = {0};
    m.outputs = {4};
    for (int i = 0; i < 4; ++i) node(m, "SiLU", {i}, {i + 1});
    m.validate();
    return m;
}

void test_chain_and_closed_intervals() {
    Model m = chain();
    const auto naive = plan_naive(m), reuse = plan_reuse(m);
    check_contract(m, naive);
    check_contract(m, reuse);
    // 17 个 float=68 字节，每段向上对齐至128：输入常驻，再用两个128字节区间轮换。
    YI_CHECK(reuse.total == 3 * 128 && naive.total == 5 * 128, "链式图未有效复用");
    YI_CHECK(overlaps(m, reuse, 1, 3), "a 最后在节点1读取，节点2产生 c 时应可复用");
    for (int i = 0; i < 4; ++i)
        YI_CHECK(!overlaps(m, reuse, i, i + 1), "同一节点不能提前覆盖仍要读取的输入");
}

void test_long_skip_consumer() {
    Model m;
    for (const char* name : {"x", "a", "b", "c", "d", "y"}) tensor(m, name, 16);
    m.inputs = {0};
    m.outputs = {5};
    node(m, "SiLU", {0}, {1});
    node(m, "SiLU", {1}, {2});
    node(m, "SiLU", {2}, {3});
    node(m, "Add", {1, 3}, {4});  // a 跨过两个节点，最后在这里再次消费
    node(m, "SiLU", {4}, {5});
    m.validate();
    const auto p = plan_reuse(m);
    check_contract(m, p);
    for (int i : {2, 3, 4}) YI_CHECK(!overlaps(m, p, 1, i), "跳连输入 a 被过早回收");
    YI_CHECK(p.total < plan_naive(m).total, "长跳连不应阻止所有复用");
}

void test_multiple_outputs_and_coalescing() {
    Model m;
    tensor(m, "x", 32);
    for (const char* name : {"a", "b", "c"}) tensor(m, name, 16);
    tensor(m, "y", 32);
    m.inputs = {0};
    m.outputs = {4};
    node(m, "Split", {0}, {1, 2}, {16, 16});
    node(m, "Add", {1, 2}, {3});
    node(m, "Concat", {3, 3}, {4});
    m.validate();
    const auto p = plan_reuse(m);
    check_contract(m, p);
    YI_CHECK(!overlaps(m, p, 1, 2), "同一 Split 的两个输出不能重叠");
    // x=128，a/b/c各64。最后y=128只能合并相邻空闲a+b，才不必扩展峰值320。
    YI_CHECK(p.total == 320, "相邻空闲块未合并以容纳较大输出");
    YI_CHECK(overlaps(m, p, 4, 1) && overlaps(m, p, 4, 2), "最终输出没有使用合并区域");
}

void test_free_block_splitting() {
    Model m;
    tensor(m, "x", 48);
    tensor(m, "a", 32);
    for (const char* name : {"b", "c", "d", "e", "f"}) tensor(m, name, 16);
    m.inputs = {0};
    m.outputs = {2, 4, 6};  // b、d必须保留；f是最后输出
    node(m, "Split", {0}, {1, 2}, {32, 16});
    node(m, "Split", {1}, {3, 4}, {16, 16});
    node(m, "Add", {3, 4}, {5});
    node(m, "Add", {5, 3}, {6});
    m.validate();
    const auto p = plan_reuse(m);
    check_contract(m, p);
    // a死亡后释放128字节，e用一半，f用另一半；不能丢弃第一轮分配剩下的64字节。
    YI_CHECK(p.total == 512, "空闲块分割余量没有留给后续输出");
    YI_CHECK(overlaps(m, p, 1, 5) && overlaps(m, p, 1, 6), "e/f没有共同复用a的两部分");
    YI_CHECK(!overlaps(m, p, 5, 6), "e是f的输入，两者不得原地覆盖");
}

void test_pinned_input_and_early_output() {
    Model m = chain();
    m.outputs.push_back(1);  // a虽然早已被内部节点消费，但也是外部要读取的图输出
    const auto p = plan_reuse(m);
    check_contract(m, p);
    for (int i = 1; i < 5; ++i) YI_CHECK(!overlaps(m, p, 0, i), "图输入必须保留以支持重复run");
    for (int i = 2; i < 5; ++i) YI_CHECK(!overlaps(m, p, 1, i), "早期图输出被覆盖");
}

template <class F>
void must_reject(F&& f, const char* description) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    YI_CHECK(rejected, "validate_plan 应拒绝: " << description);
}

void test_invalid_plans() {
    Model m = chain();
    const auto good = plan_reuse(m);
    auto bad = good;
    bad.offset[2] = bad.offset[1];
    must_reject([&] { validate_plan(m, bad); }, "同时存活的输入输出重叠");
    bad = good;
    bad.offset[2] = static_cast<int64_t>(good.total);
    must_reject([&] { validate_plan(m, bad); }, "张量超出arena末尾");
    bad = good;
    bad.offset[2] += 1;
    must_reject([&] { validate_plan(m, bad); }, "未对齐偏移");
    bad = good;
    bad.offset[2] = -1;
    must_reject([&] { validate_plan(m, bad); }, "激活缺少地址");
    bad = good;
    bad.offset.pop_back();
    must_reject([&] { validate_plan(m, bad); }, "偏移表长度错误");
    const int k = tensor(m, "constant", 16, true);
    m.validate();
    bad = plan_reuse(m);
    YI_CHECK(bad.offset[k] == -1, "常量应不占arena");
    bad.offset[k] = 0;
    must_reject([&] { validate_plan(m, bad); }, "常量占用激活偏移");
}

Model execution_graph() {
    Model m;
    tensor(m, "x", 32);
    tensor(m, "k", 16, true);
    for (const char* name : {"a", "b", "s", "t", "u", "v", "w", "y"}) tensor(m, name, 16);
    m.inputs = {0};
    m.outputs = {3, 9};
    m.weights = AlignedBuffer(64);
    m.tensors[1].data = m.weights.as<float>();
    m.tensors[1].const_offset = 0;
    for (int i = 0; i < 16; ++i) m.tensors[1].data[i] = (i - 8) * 0.125f;
    node(m, "Split", {0}, {2, 3}, {16, 16});
    node(m, "SiLU", {2}, {4});
    node(m, "Add", {3, 1}, {5});
    node(m, "Add", {4, 5}, {6});
    node(m, "SiLU", {6}, {7});
    node(m, "Add", {7, 4}, {8});
    node(m, "SiLU", {8}, {9});
    m.validate();
    return m;
}

std::vector<std::vector<float>> copy_outputs(const Model& m) {
    std::vector<std::vector<float>> result;
    for (int o : m.outputs) {
        const auto& t = m.tensors[o];
        result.emplace_back(t.data, t.data + t.numel());
    }
    return result;
}

void compare_outputs(const Model& m, const std::vector<std::vector<float>>& expected) {
    YI_CHECK(m.outputs.size() == expected.size(), "输出数量变化");
    for (size_t i = 0; i < m.outputs.size(); ++i) {
        const auto& t = m.tensors[m.outputs[i]];
        YI_CHECK(static_cast<size_t>(t.numel()) == expected[i].size(), "输出形状变化");
        YI_CHECK(std::memcmp(t.data, expected[i].data(), t.bytes()) == 0, "输出不逐位相同: " << t.name);
    }
}

void test_execution_and_repeat_run() {
    Model baseline = execution_graph(), reused = execution_graph();
    Executor naive(baseline, false), reuse(reused, true);
    std::vector<float> input(32);
    for (int i = 0; i < 32; ++i) input[i] = (i - 16) * 0.25f;
    naive.set_input(0, input.data());
    naive.run();
    const auto expected = copy_outputs(baseline);
    reuse.set_input(0, input.data());
    reuse.run();
    compare_outputs(reused, expected);
    // 不再次set_input：若输入被中间激活覆盖，第二轮会用错误输入。
    reuse.run();
    compare_outputs(reused, expected);
    naive.run();
    compare_outputs(baseline, expected);
    YI_CHECK(std::memcmp(reused.tensors[0].data, input.data(), input.size() * sizeof(float)) == 0, "输入内容被改变");
    YI_CHECK(reuse.arena_bytes() < naive.arena_bytes(), "执行器未使用复用规划");
    for (float& x : input) x = -x + 0.375f;
    naive.set_input(0, input.data());
    reuse.set_input(0, input.data());
    naive.run();
    reuse.run();
    compare_outputs(reused, copy_outputs(baseline));
}

void compare_real_model(const std::string& model_dir, const std::string& input_path) {
    Model baseline = Model::load(model_dir), reused = Model::load(model_dir);
    YI_CHECK(baseline.inputs.size() == 1, "真实模型对照入口只接受单输入");
    const auto input = read_file(input_path, static_cast<int64_t>(baseline.tensors[baseline.inputs[0]].bytes()));
    Executor naive(baseline, false), reuse(reused, true);
    naive.set_input(0, input.as<float>());
    naive.run();
    const auto expected = copy_outputs(baseline);
    reuse.set_input(0, input.as<float>());
    reuse.run();
    compare_outputs(reused, expected);
    std::printf("REAL_MODEL_EXACT outputs=%zu naive_bytes=%zu reuse_bytes=%zu\n",
                reused.outputs.size(), naive.arena_bytes(), reuse.arena_bytes());
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<std::pair<const char*, std::function<void()>>> tests = {
            {"chain_and_closed_intervals", test_chain_and_closed_intervals},
            {"long_skip_consumer", test_long_skip_consumer},
            {"multiple_outputs_and_coalescing", test_multiple_outputs_and_coalescing},
            {"free_block_splitting", test_free_block_splitting},
            {"pinned_input_and_early_output", test_pinned_input_and_early_output},
            {"invalid_plans", test_invalid_plans},
            {"execution_and_repeat_run", test_execution_and_repeat_run},
        };
        for (const auto& test : tests) {
            test.second();
            std::printf("PASS %s\n", test.first);
        }
        if (argc == 3) compare_real_model(argv[1], argv[2]);
        else YI_CHECK(argc == 1, "用法: test_memory_plan [模型目录 input.bin]");
        std::printf("MEMORY_PLAN_TESTS_PASSED count=%zu\n", tests.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
