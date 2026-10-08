// model.h —— 运行时看到的模型：张量表 + 拓扑序节点表 + 一整块权重
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.h"

namespace yi {

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;     // NCHW。数据按行优先连续存放：最后一维相邻的元素，内存地址也相邻
    bool is_const = false;          // 常量（权重、偏置）还是激活
    int64_t const_offset = -1;      // 常量在 weights.bin 里的字节偏移
    float* data = nullptr;          // 常量：指向模型的权重缓冲；激活：由执行器分配内存后填入
    // 零拷贝视图：本张量是 view_of 那个张量的一段连续切片（Split 在 batch=1 时成立）。
    // 视图不占 arena，data 指向父张量缓冲 + view_offset；生命周期由父张量覆盖。
    int view_of = -1;
    int64_t view_offset = -1;       // 相对父张量起始的字节偏移

    bool is_view() const { return view_of >= 0; }
    int64_t numel() const;
    size_t bytes() const { return static_cast<size_t>(numel()) * sizeof(float); }
    std::string shape_str() const;  // 例如 "1x16x320x320"
};

// 算子属性。model.txt 里写成 key=1,2,3；整数属性也存成 double（double 能精确表示 2^53 以内的整数），读的时候再转
struct Attrs {
    std::map<std::string, std::vector<double>> kv;
    bool has(const std::string& key) const { return kv.count(key) > 0; }
    std::vector<int64_t> ints(const std::string& key) const;   // 缺失或不是整数就抛异常
    int64_t i(const std::string& key) const;                  // 单个整数
};

struct Node {
    std::string op;                 // 运行时算子名：Conv / SiLU / MaxPool / UpsampleNearest / Concat / Split / Add
    std::string name;               // 沿用 ONNX 节点名，方便和原图对照
    std::vector<int> inputs;        // 张量下标
    std::vector<int> outputs;
    Attrs attrs;
};

struct Model {
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;                // 导出器保证已按拓扑序排好，load 时再检查一遍
    std::vector<int> inputs, outputs;       // 图输入、图输出的张量下标
    std::map<std::string, int> index;       // 张量名 → 下标
    AlignedBuffer weights;                  // weights.bin 整个读进来，常量张量的 data 直接指进这里

    static Model load(const std::string& dir);   // 读 dir/model.txt 和它引用的权重文件
    int find(const std::string& name) const;     // 找不到返回 -1
    void validate() const;                       // 检查节点表是拓扑序、每个激活恰好产生一次
};

}  // namespace yi
