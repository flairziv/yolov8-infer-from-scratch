// test_postprocess.cpp —— DFL 解码、置信度过滤、类内 NMS 与 letterbox 反算的单元测试。
// 直接搭一个只有张量、没有节点的模型：被测函数只读图输出与图输入的形状，不需要执行图。
// 张量数据放在测试自己的缓冲池里，再把 data 指过去（模型不拥有这些数据）。
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "postprocess.h"

namespace {
using namespace yi;

constexpr int64_t kClasses = 80;

struct Head {
    Model model;
    std::vector<std::vector<float>> pool;   // 每个张量一块数据；vector 扩容不影响已取出的指针
    std::vector<int> box, cls;              // 各尺度的框分支、类别分支下标

    int add(const std::string& name, std::vector<int64_t> shape) {
        Tensor t;
        t.name = name;
        t.shape = std::move(shape);
        const int id = static_cast<int>(model.tensors.size());
        model.index[name] = id;
        model.tensors.push_back(std::move(t));
        return id;
    }

    void fill(int id, float value) {
        Tensor& t = model.tensors[id];
        pool.emplace_back(static_cast<size_t>(t.numel()), value);
        t.data = pool.back().data();
    }

    // 第 7 类的 logit 抬高，其余压低。
    void fill_classes(int id, float high, float low) {
        Tensor& t = model.tensors[id];
        pool.emplace_back(static_cast<size_t>(t.numel()), low);
        const int64_t plane = t.shape[2] * t.shape[3];
        for (int64_t pos = 0; pos < plane; ++pos) pool.back()[7 * plane + pos] = high;
        t.data = pool.back().data();
    }

    // 每个锚点把 4 个方向各自的期望钉在指定 bin 上。
    void fill_dfl(int id, const std::array<int64_t, 4>& bins) {
        Tensor& t = model.tensors[id];
        pool.emplace_back(static_cast<size_t>(t.numel()), -20.0f);
        const int64_t plane = t.shape[2] * t.shape[3];
        for (int64_t pos = 0; pos < plane; ++pos)
            for (int64_t d = 0; d < 4; ++d) pool.back()[(d * 16 + bins[static_cast<size_t>(d)]) * plane + pos] = 20.0f;
        t.data = pool.back().data();
    }

    void finish() {
        model.inputs = {0};
        for (size_t i = 0; i < box.size(); ++i) {
            model.outputs.push_back(box[i]);
            model.outputs.push_back(cls[i]);
        }
    }
};

// 三个尺度：side/8、side/16、side/32（side=64 时对应步长 8/16/32）。
Head make_head(int64_t side) {
    Head h;
    h.add("x", {1, 3, side, side});
    for (int64_t f : {side / 8, side / 16, side / 32}) {
        const std::string tag = std::to_string(f);
        h.box.push_back(h.add("box" + tag, {1, 64, f, f}));
        h.cls.push_back(h.add("cls" + tag, {1, kClasses, f, f}));
    }
    h.finish();
    return h;
}

// 所有 DFL logit 相同 ⇒ softmax 均匀 ⇒ 期望 = (reg_max-1)/2 = 7.5 个 bin。
void test_uniform_dfl() {
    Head h = make_head(64);
    for (int id : h.box) h.fill(id, 0.0f);
    for (int id : h.cls) h.fill_classes(id, 10.0f, -10.0f);
    const auto boxes = dfl_decode(h.model, 0.9f);
    // 每个尺度 8*8 + 4*4 + 2*2 = 84 个锚点，每个锚点一个框。
    YI_CHECK(boxes.size() == 84, "框数量不对: " << boxes.size());
    YI_CHECK(boxes[0].cls == 7, "类别下标不对");
    YI_CHECK(std::fabs(boxes[0].score - 1.0f / (1.0f + std::exp(-10.0f))) < 1e-6f, "分数不是 sigmoid(logit)");
    // 分数完全相同，稳定排序后第一项仍是第一个锚点（尺度 8×8 的 (0,0)）。
    const float st = 8.0f;   // 64 / 8
    YI_CHECK(std::fabs(boxes[0].x1 - (0.5f - 7.5f) * st) < 1e-4f, "x1 不对: " << boxes[0].x1);
    YI_CHECK(std::fabs(boxes[0].y1 - (0.5f - 7.5f) * st) < 1e-4f, "y1 不对: " << boxes[0].y1);
    YI_CHECK(std::fabs(boxes[0].x2 - (0.5f + 7.5f) * st) < 1e-4f, "x2 不对: " << boxes[0].x2);
    YI_CHECK(std::fabs(boxes[0].y2 - (0.5f + 7.5f) * st) < 1e-4f, "y2 不对: " << boxes[0].y2);
    std::printf("PASS uniform_dfl\n");
}

// DFL 期望：把某个 bin 的 logit 抬高，期望就趋近那个 bin。
void test_dfl_expectation_peaked() {
    Head h = make_head(64);
    for (int id : h.box) h.fill_dfl(id, {3, 0, 15, 8});   // left/top/right/bottom
    for (int id : h.cls) h.fill_classes(id, 10.0f, -10.0f);
    const auto boxes = dfl_decode(h.model, 0.5f);
    YI_CHECK(!boxes.empty(), "没有解出框");
    const float st = 8.0f;
    YI_CHECK(std::fabs(boxes[0].x1 - (0.5f - 3.0f) * st) < 1e-3f, "left 距离不对: " << boxes[0].x1);
    YI_CHECK(std::fabs(boxes[0].y1 - (0.5f - 0.0f) * st) < 1e-3f, "top 距离不对: " << boxes[0].y1);
    YI_CHECK(std::fabs(boxes[0].x2 - (0.5f + 15.0f) * st) < 1e-3f, "right 距离不对: " << boxes[0].x2);
    YI_CHECK(std::fabs(boxes[0].y2 - (0.5f + 8.0f) * st) < 1e-3f, "bottom 距离不对: " << boxes[0].y2);
    std::printf("PASS dfl_expectation_peaked\n");
}

void test_conf_filter() {
    Head h = make_head(64);
    for (int id : h.box) h.fill(id, 0.0f);
    for (int id : h.cls) h.fill(id, 0.0f);   // sigmoid(0) = 0.5
    YI_CHECK(dfl_decode(h.model, 0.5f).size() == 84, "0.5 阈值应全部留下");
    YI_CHECK(dfl_decode(h.model, 0.6f).empty(), "0.6 阈值应全部滤掉");
    std::printf("PASS conf_filter\n");
}

Detection det(int cls, float score, float x1, float y1, float x2, float y2) {
    return {cls, score, x1, y1, x2, y2};
}

void test_nms() {
    // 同类重叠：低分的被抑制。
    auto kept = nms({det(1, 0.9f, 0, 0, 10, 10), det(1, 0.8f, 1, 1, 11, 11)}, 0.5f);
    YI_CHECK(kept.size() == 1 && std::fabs(kept[0].score - 0.9f) < 1e-6f, "同类重叠没有抑制低分框");
    // 不同类别即使完全重叠也都要留下。
    kept = nms({det(1, 0.9f, 0, 0, 10, 10), det(2, 0.8f, 0, 0, 10, 10)}, 0.5f);
    YI_CHECK(kept.size() == 2, "不同类别不应互相抑制");
    // 两个 10×10 的框错开 5 像素：交 5×10=50，并 200-50=150，IoU=1/3。判定是严格大于，等于阈值不抑制。
    kept = nms({det(1, 0.9f, 0, 0, 10, 10), det(1, 0.8f, 5, 0, 15, 10)}, 1.0f / 3.0f);
    YI_CHECK(kept.size() == 2, "IoU 等于阈值时不应抑制");
    kept = nms({det(1, 0.9f, 0, 0, 10, 10), det(1, 0.8f, 5, 0, 15, 10)}, 0.3f);
    YI_CHECK(kept.size() == 1, "IoU 超过阈值应抑制");
    // 不重叠：都留下；空输入：空输出。
    kept = nms({det(1, 0.9f, 0, 0, 10, 10), det(1, 0.8f, 20, 20, 30, 30)}, 0.5f);
    YI_CHECK(kept.size() == 2, "不重叠的框不应被抑制");
    YI_CHECK(nms({}, 0.5f).empty(), "空输入应返回空");
    // 传递性：三个框里第一个同时压制后两个。
    kept = nms({det(1, 0.9f, 0, 0, 10, 10), det(1, 0.8f, 1, 0, 11, 10), det(1, 0.7f, 2, 0, 12, 10)}, 0.5f);
    YI_CHECK(kept.size() == 1, "重叠链应只留最高分");
    // 退化框（零面积）：IoU 视为 0，不应误抑制。
    kept = nms({det(1, 0.9f, 5, 5, 5, 5), det(1, 0.8f, 5, 5, 5, 5)}, 0.5f);
    YI_CHECK(kept.size() == 2, "零面积框不应被抑制");
    std::printf("PASS nms\n");
}

void test_unletterbox() {
    std::vector<Detection> boxes = {det(0, 1.0f, 100.0f, 150.0f, 300.0f, 350.0f)};
    unletterbox(boxes, 0.5f, 100.0f, 150.0f);
    YI_CHECK(std::fabs(boxes[0].x1) < 1e-5f && std::fabs(boxes[0].y1) < 1e-5f, "反算 x1/y1 不对");
    YI_CHECK(std::fabs(boxes[0].x2 - 400.0f) < 1e-5f && std::fabs(boxes[0].y2 - 400.0f) < 1e-5f, "反算 x2/y2 不对");
    std::printf("PASS unletterbox\n");
}

template <class F>
void must_reject(F&& f, const char* description) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    YI_CHECK(rejected, "应拒绝: " << description);
}

void test_rejects_bad_input() {
    Head h = make_head(64);
    for (int id : h.box) h.fill(id, 0.0f);
    for (int id : h.cls) h.fill(id, 0.0f);
    must_reject([&] { dfl_decode(h.model, -0.1f); }, "负的 conf");
    must_reject([&] { dfl_decode(h.model, 1.1f); }, "大于 1 的 conf");
    must_reject([&] { nms({}, -0.5f); }, "负的 iou");
    // 少一个类别分支：配不成对。
    Head broken = make_head(64);
    for (int id : broken.box) broken.fill(id, 0.0f);
    for (int id : broken.cls) broken.fill(id, 0.0f);
    broken.model.outputs.pop_back();
    must_reject([&] { dfl_decode(broken.model, 0.5f); }, "框/类别分支配不成对");
    // 某个尺度的类别数与其他尺度不一致。
    Head mixed = make_head(64);
    for (int id : mixed.box) mixed.fill(id, 0.0f);
    for (int id : mixed.cls) mixed.fill(id, 0.0f);
    const int small = mixed.add("cls_small", {1, 20, 2, 2});
    mixed.fill(small, 0.0f);
    mixed.model.outputs[mixed.model.outputs.size() - 1] = small;
    must_reject([&] { dfl_decode(mixed.model, 0.5f); }, "各尺度类别数不一致");
    std::printf("PASS rejects_bad_input\n");
}
}  // namespace

int main() {
    try {
        test_uniform_dfl();
        test_dfl_expectation_peaked();
        test_conf_filter();
        test_nms();
        test_unletterbox();
        test_rejects_bad_input();
        std::printf("POSTPROCESS_TESTS_PASSED\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
