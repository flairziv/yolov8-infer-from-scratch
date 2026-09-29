// reference.cpp
#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace yi {

Reference Reference::load(const std::string& dir) {
    Reference r;
    r.blob = read_file(dir + "/ref.bin");
    std::ifstream f(dir + "/ref.txt");
    YI_CHECK(f, "打不开 " << dir << "/ref.txt");
    std::string name, shape;
    int64_t offset = 0, numel = 0;
    while (f >> name >> offset >> numel >> shape) {       // 每行：名字 字节偏移 元素数 形状
        YI_CHECK(offset % static_cast<int64_t>(kAlign) == 0 &&
                     offset + numel * static_cast<int64_t>(sizeof(float)) <= static_cast<int64_t>(r.blob.bytes()),
                 "ref.txt 里 " << name << " 的偏移或长度不对");
        std::replace(shape.begin(), shape.end(), ',', 'x');
        r.entries[name] = {reinterpret_cast<const float*>(r.blob.as<char>() + offset), numel, shape};
    }
    YI_CHECK(!r.entries.empty(), dir << "/ref.txt 是空的");
    return r;
}

const Reference::Entry& Reference::at(const std::string& name) const {
    const auto it = entries.find(name);
    YI_CHECK(it != entries.end(), "参考数据里没有张量 " << name);
    return it->second;
}

Diff compare(const float* out, const float* ref, int64_t n) {
    Diff d;
    d.n = n;
    double dot = 0, sq_out = 0, sq_ref = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t a, b;                       // 按位比较：用 == 判断的话 +0 和 -0 算相等、NaN 和 NaN 算不等，都不是"逐位相同"
        std::memcpy(&a, out + i, sizeof a);
        std::memcpy(&b, ref + i, sizeof b);
        d.exact += (a == b);
        if (!std::isfinite(out[i]) || !std::isfinite(ref[i])) {
            d.finite = false;
            continue;
        }
        const double x = out[i], y = ref[i];     // 统计量用 double 累加，免得统计本身引入误差
        d.max_abs = std::max(d.max_abs, std::fabs(x - y));
        d.ref_max = std::max(d.ref_max, std::fabs(y));
        dot += x * y;
        sq_out += x * x;
        sq_ref += y * y;
    }
    if (sq_out > 0 && sq_ref > 0)
        d.cos = dot / std::sqrt(sq_out * sq_ref);
    else
        d.cos = (sq_out == sq_ref) ? 1.0 : 0.0;   // 两边都全零算一致；只有一边全零算完全不相关
    return d;
}

}  // namespace yi
