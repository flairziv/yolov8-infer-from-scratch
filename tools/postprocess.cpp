// postprocess.cpp —— DFL 解码 + 类内 NMS。只做数学，不碰执行器。
#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace yi {
namespace {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// DFL：对 reg_max 个 bin 做 softmax，再取加权期望（等价于 Σ b·p_b）。
// p 指向 (d, 0, h, w)，同一 (h,w) 上相邻 bin 相差 elem_stride 个元素。
float dfl_expectation(const float* p, int64_t bins, int64_t elem_stride) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t b = 0; b < bins; ++b) mx = std::max(mx, p[b * elem_stride]);
    YI_CHECK(std::isfinite(mx), "DFL 的输入出现非有限值");
    float sum = 0.0f, acc = 0.0f;
    for (int64_t b = 0; b < bins; ++b) {
        const float e = std::exp(p[b * elem_stride] - mx);
        sum += e;
        acc += e * static_cast<float>(b);
    }
    return acc / sum;
}

struct Branch {
    int64_t height = 0, width = 0;
    int box = -1, cls = -1;
};

// 把图输出按 (H, W) 配对成"框分支 + 类别分支"。框分支通道数 = 4*reg_max，类别分支 = 类别数。
std::vector<Branch> pair_branches(const Model& model, int64_t& reg_max, int64_t& classes) {
    std::vector<Branch> branches;
    for (int id : model.outputs) {
        const Tensor& t = model.tensors[id];
        YI_CHECK(t.shape.size() == 4 && t.shape[0] == 1, "检测头输出必须是 1×C×H×W: " << t.name);
        const int64_t channels = t.shape[1];
        auto it = std::find_if(branches.begin(), branches.end(), [&](const Branch& b) {
            return b.height == t.shape[2] && b.width == t.shape[3];
        });
        if (it == branches.end()) {
            branches.push_back({t.shape[2], t.shape[3], -1, -1});
            it = branches.end() - 1;
        }
        if (channels % 4 == 0 && channels == 64) {
            YI_CHECK(it->box < 0, "同一尺度出现了两个框分支");
            it->box = id;
        } else {
            YI_CHECK(it->cls < 0, "同一尺度出现了两个类别分支");
            it->cls = id;
            classes = channels;
        }
    }
    YI_CHECK(!branches.empty(), "模型没有检测头输出");
    for (const Branch& b : branches) {
        YI_CHECK(b.box >= 0 && b.cls >= 0, "检测头输出没有配成框/类别分支对（尺度 " << b.height << "x" << b.width << "）");
        YI_CHECK(model.tensors[b.cls].shape[1] == classes, "各尺度的类别数不一致");
    }
    reg_max = model.tensors[branches[0].box].shape[1] / 4;
    YI_CHECK(reg_max > 0 && model.tensors[branches[0].box].shape[1] == 4 * reg_max, "框分支通道数不是 4 的倍数");
    return branches;
}

float iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

std::vector<Detection> dfl_decode(const Model& model, float conf, const HostTensorPtr& host) {
    YI_CHECK(conf >= 0.0f && conf <= 1.0f, "--conf 需要在 0..1 内");
    int64_t reg_max = 0, classes = 0;
    const auto branches = pair_branches(model, reg_max, classes);
    const auto data_of = [&](int id) { return host ? host(id) : model.tensors[id].data; };
    // 输入图边长决定步长：stride = 输入边长 / 该尺度的特征图边长。
    YI_CHECK(model.inputs.size() == 1, "检测头解码目前只接受单个图输入");
    const Tensor& in = model.tensors[model.inputs[0]];
    YI_CHECK(in.shape.size() == 4 && in.shape[2] == in.shape[3], "检测头解码要求方形输入");
    const int64_t input_side = in.shape[2];

    std::vector<Detection> boxes;
    for (const Branch& b : branches) {
        const float* box_data = data_of(b.box);
        const float* cls_data = data_of(b.cls);
        YI_CHECK(input_side % b.height == 0 && input_side % b.width == 0, "特征图边长不整除输入边长");
        const float st_h = static_cast<float>(input_side) / static_cast<float>(b.height);
        const float st_w = static_cast<float>(input_side) / static_cast<float>(b.width);
        YI_CHECK(st_h == st_w, "特征图的横纵步长不一致");
        const int64_t plane = b.height * b.width;
        for (int64_t h = 0; h < b.height; ++h) {
            for (int64_t w = 0; w < b.width; ++w) {
                const int64_t pos = h * b.width + w;
                // 类别分支：取最大 sigmoid 分数（YOLOv8 没有 objectness 通道）。
                int best = 0;
                float best_score = -1.0f;
                for (int64_t c = 0; c < classes; ++c) {
                    const float s = sigmoid(cls_data[c * plane + pos]);
                    if (s > best_score) { best_score = s; best = static_cast<int>(c); }
                }
                if (best_score < conf) continue;
                // 框分支：4 组 reg_max 个 bin，DFL 期望得到 left/top/right/bottom 距离。
                const float left = dfl_expectation(box_data + 0 * reg_max * plane + pos, reg_max, plane);
                const float top = dfl_expectation(box_data + 1 * reg_max * plane + pos, reg_max, plane);
                const float right = dfl_expectation(box_data + 2 * reg_max * plane + pos, reg_max, plane);
                const float bottom = dfl_expectation(box_data + 3 * reg_max * plane + pos, reg_max, plane);
                const float cx = (static_cast<float>(w) + 0.5f) * st_w;
                const float cy = (static_cast<float>(h) + 0.5f) * st_h;
                boxes.push_back({best, best_score, cx - left * st_w, cy - top * st_h,
                                 cx + right * st_w, cy + bottom * st_h});
            }
        }
    }
    std::stable_sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    return boxes;
}

std::vector<Detection> nms(std::vector<Detection> boxes, float iou_threshold) {
    YI_CHECK(iou_threshold >= 0.0f && iou_threshold <= 1.0f, "--iou 需要在 0..1 内");
    std::stable_sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<char> removed(boxes.size(), 0);
    std::vector<Detection> kept;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j] || boxes[j].cls != boxes[i].cls) continue;   // 类内抑制
            if (iou(boxes[i], boxes[j]) > iou_threshold) removed[j] = 1;
        }
    }
    return kept;
}

void unletterbox(std::vector<Detection>& boxes, float scale, float pad_x, float pad_y) {
    YI_CHECK(scale > 0.0f, "letterbox 缩放必须是正数");
    for (Detection& d : boxes) {
        d.x1 = (d.x1 - pad_x) / scale;
        d.x2 = (d.x2 - pad_x) / scale;
        d.y1 = (d.y1 - pad_y) / scale;
        d.y2 = (d.y2 - pad_y) / scale;
    }
}

}  // namespace yi
