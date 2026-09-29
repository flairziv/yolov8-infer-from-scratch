// memory_plan.cpp —— 顺序执行图的激活内存规划
#include "memory_plan.h"

#include <algorithm>
#include <limits>

namespace yi {
namespace {

struct Lifetime {
    int64_t begin = -2;  // -2 未产生，-1 图输入，>=0 节点序号
    int64_t end = -2;    // 闭区间：最后读取的节点执行完，才允许释放
};

std::vector<Lifetime> lifetimes(const Model& m) {
    m.validate();
    std::vector<Lifetime> life(m.tensors.size());
    const auto finish = static_cast<int64_t>(m.nodes.size());
    // 图输入也保留：set_input 一次后可重复 run，不会读到上轮覆盖后的数据。
    for (int id : m.inputs) life[id] = {-1, finish};
    for (size_t i = 0; i < m.nodes.size(); ++i) {
        const auto step = static_cast<int64_t>(i);
        for (int id : m.nodes[i].inputs) {
            if (m.tensors[id].is_const) continue;
            YI_CHECK(life[id].begin >= -1, "规划时遇到尚未产生的输入 " << m.tensors[id].name);
            life[id].end = std::max(life[id].end, step);
        }
        for (int id : m.nodes[i].outputs) life[id] = {step, step};
    }
    // 某个图输出可能在中途就产生，但调用者要等整次执行后再读它。
    for (int id : m.outputs)
        if (!m.tensors[id].is_const) life[id].end = finish;
    return life;
}

struct Block {
    size_t offset, bytes;
};

void coalesce(std::vector<Block>& free) {
    std::sort(free.begin(), free.end(), [](const Block& a, const Block& b) { return a.offset < b.offset; });
    size_t count = 0;
    for (const Block block : free) {
        if (count && free[count - 1].offset + free[count - 1].bytes == block.offset)
            free[count - 1].bytes += block.bytes;
        else
            free[count++] = block;
    }
    free.resize(count);
}

}  // namespace

MemoryPlan plan_naive(const Model& m) {
    MemoryPlan p;
    p.offset.assign(m.tensors.size(), -1);
    size_t off = 0;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const Tensor& t = m.tensors[i];
        if (t.is_const) continue;
        p.offset[i] = static_cast<int64_t>(off);
        off += align_up(t.bytes());
    }
    p.total = off;
    return p;
}

MemoryPlan plan_reuse(const Model& m) {
    const auto life = lifetimes(m);
    std::vector<size_t> order;
    for (size_t i = 0; i < m.tensors.size(); ++i)
        if (!m.tensors[i].is_const) order.push_back(i);
    // 张量声明顺序不一定等于产生顺序；同节点的多个输出用张量编号稳定排序。
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return life[a].begin < life[b].begin; });

    MemoryPlan p;
    p.offset.assign(m.tensors.size(), -1);
    struct Active { size_t id; Block block; };
    std::vector<Active> active;
    std::vector<Block> free;
    for (size_t id : order) {
        for (auto it = active.begin(); it != active.end();) {
            // 不能用 <=：当前节点正在读取的输入，不能提前给它的输出覆盖。
            if (life[it->id].end < life[id].begin) {
                free.push_back(it->block);
                it = active.erase(it);
            } else {
                ++it;
            }
        }
        coalesce(free);
        const size_t need = align_up(m.tensors[id].bytes());
        size_t best = free.size();
        for (size_t i = 0; i < free.size(); ++i)
            if (free[i].bytes >= need && (best == free.size() || free[i].bytes < free[best].bytes)) best = i;
        size_t offset;
        if (best == free.size()) {
            offset = p.total;
            YI_CHECK(need <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) - p.total, "激活空间过大");
            p.total += need;
        } else {
            offset = free[best].offset;
            free[best].offset += need;
            free[best].bytes -= need;
            if (free[best].bytes == 0) free.erase(free.begin() + best);
        }
        p.offset[id] = static_cast<int64_t>(offset);
        active.push_back({id, {offset, need}});
    }
    validate_plan(m, p);
    return p;
}

void validate_plan(const Model& m, const MemoryPlan& p) {
    const auto life = lifetimes(m);
    YI_CHECK(p.offset.size() == m.tensors.size(), "内存规划的张量数不匹配");
    YI_CHECK(p.total % kAlign == 0, "arena 大小没有对齐");
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        if (m.tensors[i].is_const) {
            YI_CHECK(p.offset[i] == -1, "常量不能占用激活空间");
            continue;
        }
        YI_CHECK(p.offset[i] >= 0 && p.offset[i] % kAlign == 0, "激活偏移非法: " << m.tensors[i].name);
        const auto off = static_cast<size_t>(p.offset[i]);
        const auto bytes = align_up(m.tensors[i].bytes());
        YI_CHECK(off <= p.total && bytes <= p.total - off, "激活超出 arena: " << m.tensors[i].name);
    }
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        if (m.tensors[i].is_const) continue;
        for (size_t j = i + 1; j < m.tensors.size(); ++j) {
            if (m.tensors[j].is_const || life[i].end < life[j].begin || life[j].end < life[i].begin) continue;
            const auto a = static_cast<size_t>(p.offset[i]), b = static_cast<size_t>(p.offset[j]);
            const bool disjoint = a + align_up(m.tensors[i].bytes()) <= b || b + align_up(m.tensors[j].bytes()) <= a;
            YI_CHECK(disjoint, "同时存活的张量空间重叠: " << m.tensors[i].name << " / " << m.tensors[j].name);
        }
    }
}

}  // namespace yi
