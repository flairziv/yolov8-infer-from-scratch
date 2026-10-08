// memory_plan.cpp —— 顺序执行图的内存规划：生命周期复用、Split 零拷贝视图与原地算子
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
    // 图输入只活到最后一次被读，之后它的槽位可以给别人用。
    // 契约：每次 run 之前必须重新 set_input（否则会读到被覆盖的输入）。
    for (int id : m.inputs) life[id] = {-1, -1};
    for (size_t i = 0; i < m.nodes.size(); ++i) {
        const auto step = static_cast<int64_t>(i);
        for (int id : m.nodes[i].inputs) {
            if (m.tensors[id].is_const) continue;
            YI_CHECK(life[id].begin >= -1, "规划时遇到尚未产生的输入 " << m.tensors[id].name);
            life[id].end = std::max(life[id].end, step);
            // 视图和父张量共用内存：读视图等于读父张量的一段，父张量必须活到最晚的视图消费者。
            for (int root = m.tensors[id].view_of, guard = 0; root >= 0; root = m.tensors[root].view_of) {
                YI_CHECK(root < static_cast<int>(m.tensors.size()) && ++guard <= static_cast<int>(m.tensors.size()),
                         "视图链非法: " << m.tensors[id].name);
                life[root].end = std::max(life[root].end, step);
            }
        }
        for (int id : m.nodes[i].outputs) life[id] = {step, step};
    }
    // 某个图输出可能在中途就产生，但调用者要等整次执行后再读它。
    // 输出本身是视图时，父张量的缓冲也必须保留到执行结束。
    for (int id : m.outputs) {
        if (m.tensors[id].is_const) continue;
        life[id].end = finish;
        for (int root = m.tensors[id].view_of, guard = 0; root >= 0; root = m.tensors[root].view_of) {
            YI_CHECK(root < static_cast<int>(m.tensors.size()) && ++guard <= static_cast<int>(m.tensors.size()),
                     "视图链非法: " << m.tensors[id].name);
            life[root].end = finish;
        }
    }
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

// 原地算子的候选目标：本节点是它最后一个读者、不是常量/视图/图输出、没有被任何视图引用、
// 且和输出同样大。逐元素算子读写同一批元素，原地覆盖安全；其它算子一律不允许。
int in_place_target(const Model& m, const std::vector<Lifetime>& life, const std::vector<char>& aliased,
                    const Node& n, int64_t step) {
    if (n.op != "Add" && n.op != "SiLU") return -1;
    if (n.outputs.size() != 1) return -1;
    const int64_t out_bytes = static_cast<int64_t>(m.tensors[n.outputs[0]].bytes());
    for (int id : n.inputs) {
        const Tensor& t = m.tensors[id];
        if (t.is_const || t.is_view() || aliased[id]) continue;
        if (std::find(m.outputs.begin(), m.outputs.end(), id) != m.outputs.end()) continue;
        if (life[id].end != step) continue;
        if (static_cast<int64_t>(t.bytes()) != out_bytes) continue;
        return id;
    }
    return -1;
}

}  // namespace

int root_of(const Model& m, int tensor_id) {
    YI_CHECK(tensor_id >= 0 && tensor_id < static_cast<int>(m.tensors.size()), "张量下标越界");
    int root = tensor_id;
    for (size_t guard = 0; m.tensors[root].is_view(); ++guard) {
        YI_CHECK(guard < m.tensors.size(), "视图链成环: " << m.tensors[tensor_id].name);
        root = m.tensors[root].view_of;
        YI_CHECK(root >= 0 && root < static_cast<int>(m.tensors.size()), "视图指向不存在的张量");
    }
    return root;
}

void mark_split_views(Model& m) {
    for (Node& n : m.nodes) {
        if (n.op != "Split" || n.inputs.size() != 1 || n.outputs.empty()) continue;
        if (!n.attrs.has("axis") || !n.attrs.has("sizes")) continue;
        const int64_t axis = n.attrs.i("axis");
        const auto sizes = n.attrs.ints("sizes");
        if (sizes.size() != n.outputs.size()) continue;
        const Tensor& x = m.tensors[n.inputs[0]];
        if (x.is_const) continue;
        // 沿 axis 把形状拆成 outer × mid × inner；只有 outer==1（batch=1）时切片才在内存里连续。
        const auto rank = static_cast<int64_t>(x.shape.size());
        const int64_t ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank) continue;
        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < ax; ++d) outer *= x.shape[d];
        for (int64_t d = ax + 1; d < rank; ++d) inner *= x.shape[d];
        if (outer != 1) continue;   // 多 batch：切片不连续，保持拷贝路径
        // 形状对不上就整体退回拷贝路径，交给 Split 内核按原逻辑报错。
        bool ok = true;
        int64_t total = 0;
        for (size_t k = 0; k < n.outputs.size() && ok; ++k) {
            const Tensor& y = m.tensors[n.outputs[k]];
            ok = static_cast<int64_t>(y.shape.size()) == rank && y.shape[ax] == sizes[k];
            for (int64_t d = 0; d < rank && ok; ++d)
                if (d != ax) ok = y.shape[d] == x.shape[d];
            total += sizes[k];
        }
        if (!ok || total != x.shape[ax]) continue;
        // 父张量自己可能也是视图：偏移要累加到最外层。
        int64_t base = 0;
        for (const Tensor* t = &x; t->is_view(); t = &m.tensors[t->view_of]) base += t->view_offset;
        const int root = root_of(m, n.inputs[0]);
        int64_t offset = 0;
        for (size_t k = 0; k < n.outputs.size(); ++k) {
            Tensor& y = m.tensors[n.outputs[k]];
            y.view_of = root;
            y.view_offset = base + offset * inner * static_cast<int64_t>(sizeof(float));
            offset += sizes[k];
        }
    }
}

MemoryPlan plan_naive(const Model& m) {
    MemoryPlan p;
    p.offset.assign(m.tensors.size(), -1);
    size_t off = 0;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const Tensor& t = m.tensors[i];
        if (t.is_const || t.is_view()) continue;   // 视图与父张量共用内存，不单独占位
        p.offset[i] = static_cast<int64_t>(off);
        off += align_up(t.bytes());
    }
    p.total = off;
    return p;
}

MemoryPlan plan_reuse(const Model& m) {
    const auto life = lifetimes(m);
    std::vector<char> aliased(m.tensors.size(), 0);
    for (const Tensor& t : m.tensors)
        if (t.is_view()) aliased[root_of(m, t.view_of)] = 1;   // 被视图引用的张量不做原地目标
    std::vector<size_t> order;
    for (size_t i = 0; i < m.tensors.size(); ++i)
        if (!m.tensors[i].is_const && !m.tensors[i].is_view()) order.push_back(i);
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
        // 原地：输出接管一个本节点之后没人再读的输入槽位（块大小相同，直接换所有者）。
        if (life[id].begin >= 0) {
            const int target = in_place_target(m, life, aliased, m.nodes[life[id].begin], life[id].begin);
            if (target >= 0) {
                const auto it = std::find_if(active.begin(), active.end(),
                                             [&](const Active& a) { return a.id == static_cast<size_t>(target); });
                YI_CHECK(it != active.end(), "原地算子的目标槽位不在活跃表里: " << m.tensors[target].name);
                p.offset[id] = p.offset[target];
                p.in_place.push_back({target, static_cast<int>(id)});
                it->id = id;   // 块的所有权转给输出，之后的释放按输出的生命周期走
                continue;
            }
        }
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

size_t peak_live_bytes(const Model& m) {
    const auto life = lifetimes(m);
    std::vector<char> counted(m.tensors.size(), 0);
    size_t peak = 0;
    for (int64_t step = 0; step <= static_cast<int64_t>(m.nodes.size()); ++step) {
        std::fill(counted.begin(), counted.end(), 0);
        size_t bytes = 0;
        for (size_t i = 0; i < m.tensors.size(); ++i) {
            if (m.tensors[i].is_const || life[i].end < 0) continue;
            if (step < life[i].begin || step > life[i].end) continue;
            const int root = m.tensors[i].is_view() ? root_of(m, i) : static_cast<int>(i);
            if (counted[root]) continue;   // 视图和父张量共用一块内存，只算一次
            counted[root] = 1;
            bytes += align_up(m.tensors[root].bytes());
        }
        peak = std::max(peak, bytes);
    }
    return peak;
}

void validate_plan(const Model& m, const MemoryPlan& p) {
    const auto life = lifetimes(m);
    YI_CHECK(p.offset.size() == m.tensors.size(), "内存规划的张量数不匹配");
    YI_CHECK(p.total % kAlign == 0, "arena 大小没有对齐");
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const Tensor& t = m.tensors[i];
        if (t.is_const) {
            YI_CHECK(p.offset[i] == -1, "常量不能占用激活空间");
            continue;
        }
        if (t.is_view()) {
            // 视图不占槽位，但要落在父张量槽位内，而且父张量必须活到视图最后一次被读。
            YI_CHECK(p.offset[i] == -1, "视图不能占用激活空间: " << t.name);
            const int root = root_of(m, i);
            YI_CHECK(p.offset[root] >= 0, "视图的父张量没有地址: " << t.name);
            YI_CHECK(t.view_offset >= 0 && t.view_offset % static_cast<int64_t>(sizeof(float)) == 0,
                     "视图偏移非法: " << t.name);
            YI_CHECK(t.view_offset + static_cast<int64_t>(t.bytes()) <=
                         static_cast<int64_t>(align_up(m.tensors[root].bytes())), "视图超出父张量: " << t.name);
            YI_CHECK(life[i].end <= life[root].end, "父张量没有活到视图最后一次被读: " << t.name);
            continue;
        }
        YI_CHECK(p.offset[i] >= 0 && p.offset[i] % kAlign == 0, "激活偏移非法: " << t.name);
        const auto off = static_cast<size_t>(p.offset[i]);
        const auto bytes = align_up(t.bytes());
        YI_CHECK(off <= p.total && bytes <= p.total - off, "激活超出 arena: " << t.name);
    }
    auto alias_pair = [&](size_t a, size_t b) {
        for (const auto& pr : p.in_place) {
            const auto in = static_cast<size_t>(pr.first), out = static_cast<size_t>(pr.second);
            if ((in == a && out == b) || (in == b && out == a)) return true;
        }
        return false;
    };
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        if (m.tensors[i].is_const || m.tensors[i].is_view()) continue;
        for (size_t j = i + 1; j < m.tensors.size(); ++j) {
            if (m.tensors[j].is_const || m.tensors[j].is_view()) continue;
            if (life[i].end < life[j].begin || life[j].end < life[i].begin) continue;
            if (alias_pair(i, j)) {
                YI_CHECK(p.offset[i] == p.offset[j], "原地别名的两个张量必须共用同一偏移: " << m.tensors[i].name);
                continue;
            }
            const auto a = static_cast<size_t>(p.offset[i]), b = static_cast<size_t>(p.offset[j]);
            const bool disjoint = a + align_up(m.tensors[i].bytes()) <= b || b + align_up(m.tensors[j].bytes()) <= a;
            YI_CHECK(disjoint, "同时存活的张量空间重叠: " << m.tensors[i].name << " / " << m.tensors[j].name);
        }
    }
    // 原地别名必须真是"输入在本节点死亡、输出在同节点接管"。
    for (const auto& pr : p.in_place) {
        const auto in = static_cast<size_t>(pr.first), out = static_cast<size_t>(pr.second);
        YI_CHECK(in < m.tensors.size() && out < m.tensors.size(), "原地别名下标越界");
        YI_CHECK(!m.tensors[in].is_const && !m.tensors[in].is_view() && !m.tensors[out].is_view(),
                 "原地别名指向了常量或视图");
        YI_CHECK(life[in].end == life[out].begin, "原地别名不是在同一节点交接: " << m.tensors[in].name);
    }
}

}  // namespace yi
