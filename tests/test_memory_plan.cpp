// test_memory_plan.cpp —— 生命周期、Split 视图与原地覆盖的契约；无 gtest，Release 下检查也生效。
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

bool aliased(const MemoryPlan& p, int a, int b) {
    for (const auto& pr : p.in_place)
        if ((pr.first == a && pr.second == b) || (pr.first == b && pr.second == a)) return true;
    return false;
}

// 定义在后面，先声明以便前面的用例调用。
template <class F>
void must_reject(F&& f, const char* description);
std::vector<std::vector<float>> copy_outputs(const Model& m);
void compare_outputs(const Model& m, const std::vector<std::vector<float>>& expected);

// 独立于生产 validate_plan，按题目契约检查：
//  - 同时存活的普通张量不共用空间，显式声明的原地别名对除外（必须共用同一偏移）；
//  - 视图不占槽位，落在父张量槽位内，父张量活到视图最后一次被读；
//  - 图输入按最后消费者释放（每次 run 前重新 set_input），图输出保留到执行结束。
void check_contract(const Model& m, const MemoryPlan& p) {
    YI_CHECK(p.offset.size() == m.tensors.size(), "偏移表长度错误");
    const int horizon = static_cast<int>(m.nodes.size());
    std::vector<int> born(m.tensors.size(), -1), last(m.tensors.size(), -1);
    for (int i = 0; i < horizon; ++i) {
        for (int o : m.nodes[i].outputs) born[o] = last[o] = i;
        for (int in : m.nodes[i].inputs) {
            last[in] = std::max(last[in], i);
            for (int r = m.tensors[in].view_of; r >= 0; r = m.tensors[r].view_of) last[r] = std::max(last[r], i);
        }
    }
    for (int out : m.outputs) {
        last[out] = horizon;
        for (int r = m.tensors[out].view_of; r >= 0; r = m.tensors[r].view_of) last[r] = horizon;
    }
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        if (m.tensors[i].is_const) {
            YI_CHECK(p.offset[i] == -1, "常量不应占用激活 arena");
            continue;
        }
        if (m.tensors[i].is_view()) {
            YI_CHECK(p.offset[i] == -1, "视图不应占用激活 arena");
            const int root = root_of(m, static_cast<int>(i));
            YI_CHECK(m.tensors[i].view_offset + static_cast<int64_t>(m.tensors[i].bytes()) <=
                         static_cast<int64_t>(align_up(m.tensors[root].bytes())), "视图超出父张量");
            YI_CHECK(last[i] <= last[root], "父张量没有活到视图最后一次被读");
            continue;
        }
        YI_CHECK(p.offset[i] >= 0 && p.offset[i] % kAlign == 0, "激活必须按 64 字节对齐");
        YI_CHECK(static_cast<size_t>(p.offset[i]) + align_up(m.tensors[i].bytes()) <= p.total, "激活超出 arena");
        for (size_t j = 0; j < i; ++j) {
            if (m.tensors[j].is_const || m.tensors[j].is_view()) continue;
            if (std::max(born[i], born[j]) <= std::min(last[i], last[j])) {
                if (aliased(p, static_cast<int>(i), static_cast<int>(j))) {
                    YI_CHECK(p.offset[i] == p.offset[j], "原地别名必须共用同一偏移");
                    continue;
                }
                YI_CHECK(!overlaps(m, p, static_cast<int>(i), static_cast<int>(j)),
                         "同时存活的张量重叠: " << m.tensors[i].name << "/" << m.tensors[j].name);
            }
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

// 纯逐元素链：每个输出都能原地接管上一个张量，整条链塌缩成一个槽位。
void test_chain_in_place() {
    Model m = chain();
    const auto naive = plan_naive(m), reuse = plan_reuse(m);
    check_contract(m, naive);
    check_contract(m, reuse);
    YI_CHECK(naive.total == 5 * 128, "对照模式应为每个激活独占空间");
    YI_CHECK(reuse.total == 128, "逐元素链应塌缩到单个槽位");
    for (int i = 0; i < 5; ++i) YI_CHECK(reuse.offset[i] == 0, "原地链上所有张量应共用偏移 0");
    for (int i = 0; i + 1 < 5; ++i) YI_CHECK(aliased(reuse, i, i + 1), "逐元素链缺少原地别名");
    YI_CHECK(reuse.in_place.size() == 4, "原地别名个数不对");
    YI_CHECK(naive.in_place.empty(), "对照模式不应做原地覆盖");
}

// 逐元素之外的算子一律不许原地：MaxPool 的输出必须另占空间。
void test_no_in_place_for_non_elementwise() {
    Model m;
    tensor(m, "x", 32);
    tensor(m, "y", 32);
    m.inputs = {0};
    m.outputs = {1};
    node(m, "MaxPool", {0}, {1});   // 只做规划检查，不执行，所以不需要窗口属性
    m.validate();
    const auto p = plan_reuse(m);
    check_contract(m, p);
    YI_CHECK(p.in_place.empty(), "非逐元素算子不应做原地覆盖");
    YI_CHECK(!overlaps(m, p, 0, 1), "MaxPool 输出不能覆盖它的输入");
    YI_CHECK(p.total == 2 * 128, "两个张量应各占一个槽位");
}

// 输入在后面的节点还要被读时，它的槽位不能被提前覆盖。
void test_input_still_live_blocks_reuse() {
    Model m;
    for (const char* name : {"x", "a", "b", "y"}) tensor(m, name, 16);
    m.inputs = {0};
    m.outputs = {3};
    node(m, "SiLU", {0}, {1});    // a = SiLU(x)
    node(m, "Add", {1, 0}, {2});  // b = a + x：x 到这里才死
    node(m, "SiLU", {2}, {3});    // y
    m.validate();
    const auto p = plan_reuse(m);
    check_contract(m, p);
    YI_CHECK(!overlaps(m, p, 0, 1), "x 在节点 1 还要被读，不能被 a 覆盖");
    YI_CHECK(aliased(p, 1, 2), "b 应原地接管已经死亡的 a");
    YI_CHECK(aliased(p, 2, 3), "y 应原地接管 b");
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
    for (int i : {2, 3}) YI_CHECK(!overlaps(m, p, 1, i), "跳连输入 a 被过早回收");
    YI_CHECK(aliased(p, 1, 4), "a 死亡后应由 Add 的输出原地接管");
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
    YI_CHECK(aliased(p, 1, 3), "Add 的输出应原地接管已死亡的 a");
    // x=128、a/b=64：a 被 c 原地接管，b 在节点 2 后释放；y=128 直接复用 x 的槽位。
    YI_CHECK(p.total == 256, "空闲块没有正确复用");
    YI_CHECK(overlaps(m, p, 4, 0), "最终输出没有复用输入释放出的槽位");
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
    // x 死后 192 字节的空间由 c/d 各取一半、剩下的 64 字节再给 e；f 原地接管 e。
    // 总占用 = x 的 192 + a 的 128 + b 的 64 = 384 字节。
    YI_CHECK(overlaps(m, p, 0, 3) && overlaps(m, p, 0, 4), "c/d 没有复用 x 释放出的空间");
    YI_CHECK(!overlaps(m, p, 3, 4), "c/d 必须各占一段，不能重叠");
    YI_CHECK(aliased(p, 5, 6), "f 应原地接管 e");
    YI_CHECK(p.total == 384, "空闲块分割余量没有留给后续输出");
}

// 图输入不再钉到结束：最后一次被读之后槽位即可复用；图输出仍保留到执行结束。
void test_released_input_and_pinned_output() {
    Model m;
    tensor(m, "x", 32);
    for (const char* name : {"m", "y"}) tensor(m, name, 16);
    m.inputs = {0};
    m.outputs = {1, 2};   // m 是早期图输出，必须保留到结束
    node(m, "MaxPool", {0}, {1});   // 非逐元素：不会原地接管 x
    node(m, "SiLU", {1}, {2});
    m.validate();
    const auto p = plan_reuse(m);
    check_contract(m, p);
    YI_CHECK(overlaps(m, p, 0, 2), "输入槽位在最后一次读取后没有释放");
    YI_CHECK(!overlaps(m, p, 1, 2), "早期图输出被覆盖");
    YI_CHECK(p.total == 192, "输入释放后 y 应塞进 x 的槽位（x 的 128 字节里只用掉 64）");
}

// Split 视图：标记、规划（不占槽位）、校验（越界要拒绝）。
Model split_graph() {
    Model m;
    tensor(m, "x", 32);
    tensor(m, "a", 16);
    tensor(m, "b", 16);
    tensor(m, "y", 32);   // Concat(a, b) 拼回 32 个元素
    m.inputs = {0};
    m.outputs = {3};
    node(m, "Split", {0}, {1, 2}, {16, 16});
    node(m, "Concat", {1, 2}, {3});
    m.validate();
    return m;
}

void test_split_views() {
    Model m = split_graph();
    mark_split_views(m);
    YI_CHECK(m.tensors[1].is_view() && m.tensors[2].is_view(), "Split 输出没有标记成视图");
    YI_CHECK(m.tensors[1].view_of == 0 && m.tensors[1].view_offset == 0, "第一段视图偏移不对");
    YI_CHECK(m.tensors[2].view_of == 0 && m.tensors[2].view_offset == 16 * 4, "第二段视图偏移不对");
    const auto p = plan_reuse(m);
    check_contract(m, p);
    YI_CHECK(p.offset[1] == -1 && p.offset[2] == -1, "视图不应占用 arena");
    // x 在 Concat 处还要被读，y 不能复用它的槽位；总占用与"不标记视图"时相同，
    // 差别在同时存活的缓冲：视图化后 a/b 不再各占 64 字节。
    YI_CHECK(p.total == 256, "视图化后的总占用不对: " << p.total);
    Model materialized = split_graph();
    const auto pm = plan_reuse(materialized);
    YI_CHECK(pm.offset[1] >= 0 && pm.offset[2] >= 0, "未标记视图时应为 a/b 分配槽位");
    // 视图落在父张量之外、或视图占了自己的槽位，都必须被拒绝。
    auto bad = p;
    bad.offset[1] = 0;
    must_reject([&] { validate_plan(m, bad); }, "视图占用激活空间");
    Model out_of_range = split_graph();
    mark_split_views(out_of_range);
    out_of_range.tensors[2].view_offset = 1 << 20;
    must_reject([&] { validate_plan(out_of_range, plan_reuse(out_of_range)); }, "视图超出父张量");
    // 多 batch 时切片不连续，必须保持拷贝。
    Model batched;
    tensor(batched, "x", 64);
    for (const char* name : {"a", "b", "y"}) tensor(batched, name, 32);
    batched.tensors[0].shape = {2, 32};
    for (const char* name : {"a", "b"}) batched.tensors[batched.find(name)].shape = {2, 16};
    batched.tensors[batched.find("y")].shape = {2, 32};
    batched.inputs = {0};
    batched.outputs = {3};
    node(batched, "Split", {0}, {1, 2}, {16, 16});
    node(batched, "Concat", {1, 2}, {3});
    batched.validate();
    mark_split_views(batched);
    YI_CHECK(!batched.tensors[1].is_view(), "多 batch 的切片不连续，不能做成视图");
}

// 视图路径的执行：Concat(Split(x)) 就是 x 本身，输出必须原样复现输入，且视图确实指进父张量。
void test_views_execution_reproduces_input() {
    Model views = split_graph();
    Executor with_views(views, true, Backend::Scalar);
    YI_CHECK(with_views.view_tensors() == 2, "执行器没有把 Split 输出当成视图");
    std::vector<float> input(32);
    for (int i = 0; i < 32; ++i) input[i] = (i - 16) * 0.3f;
    with_views.set_input(0, input.data());
    with_views.run();
    const Tensor& y = views.tensors[views.find("y")];
    YI_CHECK(std::memcmp(y.data, input.data(), input.size() * sizeof(float)) == 0, "视图路径没有原样复现输入");
    YI_CHECK(views.tensors[views.find("a")].data == views.tensors[0].data, "第一段视图没有指进父张量");
    YI_CHECK(views.tensors[views.find("b")].data == views.tensors[0].data + 16, "第二段视图偏移不对");
    // naive 规划同样使用视图，输出必须逐位一致。
    Model naive_model = split_graph();
    Executor naive(naive_model, false, Backend::Scalar);
    naive.set_input(0, input.data());
    naive.run();
    compare_outputs(naive_model, copy_outputs(views));
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
    bad.offset[2] = 128;                       // 与原地别名对 (1,2) 的偏移不一致
    must_reject([&] { validate_plan(m, bad); }, "原地别名偏移不一致");
    bad = good;
    bad.in_place.push_back({0, 3});            // x 与 c 不在同一节点交接
    must_reject([&] { validate_plan(m, bad); }, "原地别名不在同一节点交接");
    bad = good;
    bad.offset[2] = static_cast<int64_t>(good.total);
    bad.in_place.clear();                      // 去掉别名后必须各自独立
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

// 原地覆盖与视图都不许改变数值：与 naive 独占空间的执行逐位对照。
void test_execution_and_repeat_run() {
    Model baseline = execution_graph(), reused = execution_graph();
    Executor naive(baseline, false), reuse(reused, true);
    YI_CHECK(reuse.in_place_tensors() > 0, "复用规划里没有产生原地覆盖");
    std::vector<float> input(32);
    for (int i = 0; i < 32; ++i) input[i] = (i - 16) * 0.25f;
    naive.set_input(0, input.data());
    naive.run();
    const auto expected = copy_outputs(baseline);
    reuse.set_input(0, input.data());
    reuse.run();
    compare_outputs(reused, expected);
    // 契约：图输入的槽位可以被后续激活复用，所以每次 run 之前都要重新 set_input。
    reuse.set_input(0, input.data());
    reuse.run();
    compare_outputs(reused, expected);
    naive.set_input(0, input.data());
    naive.run();
    compare_outputs(baseline, expected);
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
    std::printf("REAL_MODEL_EXACT outputs=%zu naive_bytes=%zu reuse_bytes=%zu views=%zu in_place=%zu\n",
                reused.outputs.size(), naive.arena_bytes(), reuse.arena_bytes(),
                reuse.view_tensors(), reuse.in_place_tensors());
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<std::pair<const char*, std::function<void()>>> tests = {
            {"chain_in_place", test_chain_in_place},
            {"no_in_place_for_non_elementwise", test_no_in_place_for_non_elementwise},
            {"input_still_live_blocks_reuse", test_input_still_live_blocks_reuse},
            {"long_skip_consumer", test_long_skip_consumer},
            {"multiple_outputs_and_coalescing", test_multiple_outputs_and_coalescing},
            {"free_block_splitting", test_free_block_splitting},
            {"released_input_and_pinned_output", test_released_input_and_pinned_output},
            {"split_views", test_split_views},
            {"views_execution_reproduces_input", test_views_execution_reproduces_input},
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
