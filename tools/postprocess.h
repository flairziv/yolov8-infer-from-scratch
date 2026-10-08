// postprocess.h —— 检测头输出的 DFL 解码、置信度过滤、类内 NMS 与 letterbox 反算。
// 运行时的子图停在检测头卷积之后：框分支还没做 softmax，类别分支还没做 sigmoid。
#pragma once
#include <functional>
#include <vector>

#include "model.h"

namespace yi {

struct Detection {
    int cls = -1;
    float score = 0.0f;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;   // 输入图（letterbox 后）像素坐标
};

// 取张量的主机可读指针。CUDA 后端的数据在显存里，必须通过执行器回读；
// 传空表示直接用 Tensor::data（CPU 后端或测试里手工搭的模型）。
using HostTensorPtr = std::function<const float*(int)>;

// 解码检测头：每个尺度一对输出——框分支 4*reg_max 通道，类别分支 nc 通道（按 (H,W) 配对）。
// 做 DFL 期望（softmax 加权）、sigmoid、按 conf 过滤；输出按分数降序，不做 NMS。
std::vector<Detection> dfl_decode(const Model& model, float conf, const HostTensorPtr& host = {});

// 类内贪心 NMS：按分数降序处理，IoU > iou 的同类框被抑制；不同类别互不影响。
std::vector<Detection> nms(std::vector<Detection> boxes, float iou);

// letterbox 反算：x_orig = (x_lb - pad) / scale。
void unletterbox(std::vector<Detection>& boxes, float scale, float pad_x, float pad_y);

}  // namespace yi
