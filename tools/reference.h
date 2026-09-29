// reference.h —— 读 make_reference.py 生成的逐层参考数据；以及对拍用的误差统计
#pragma once
#include <cstdint>
#include <map>
#include <string>

#include "common.h"

namespace yi {

struct Reference {
    struct Entry {
        const float* data = nullptr;
        int64_t numel = 0;
        std::string shape;          // "1x16x320x320"，和 Tensor::shape_str() 同一种写法
    };
    AlignedBuffer blob;             // ref.bin 整个读进来，各条目的 data 指进这里
    std::map<std::string, Entry> entries;

    static Reference load(const std::string& dir);    // 读 dir/ref.txt 和 dir/ref.bin
    const Entry& at(const std::string& name) const;   // 没有这个张量就抛异常
};

// 一个张量的对拍结果
struct Diff {
    int64_t n = 0;          // 元素数
    int64_t exact = 0;      // 和参考逐位相同的元素数
    double max_abs = 0;     // 最大绝对误差
    double ref_max = 0;     // 参考值的最大绝对值，用来把绝对误差换算成相对误差
    double cos = 1;         // 余弦相似度（把整个张量看成一个向量）
    bool finite = true;     // 两边都没有 NaN / Inf
    double rel() const { return ref_max > 0 ? max_abs / ref_max : max_abs; }
};

Diff compare(const float* out, const float* ref, int64_t n);

}  // namespace yi
