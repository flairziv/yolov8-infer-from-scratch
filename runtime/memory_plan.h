// memory_plan.h —— 激活内存规划：决定每个激活张量放在 arena（一大块连续内存）的哪个偏移
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

#include "model.h"

namespace yi {

struct MemoryPlan {
    std::vector<int64_t> offset;    // offset[t]：激活张量 t 在 arena 里的字节偏移；常量与视图是 -1
    size_t total = 0;               // arena 一共要多少字节
    // 原地别名对 (输入张量, 输出张量)：输出直接写进一个"本节点之后再没人读"的输入槽位。
    // 逐元素算子（Add/SiLU）读写同一批元素，原地覆盖是安全的；其余算子不允许。
    std::vector<std::pair<int, int>> in_place;
};

// 对照模式：每个激活独占空间（不做复用，也不做原地别名）。
MemoryPlan plan_naive(const Model& m);

// 顺序执行：按闭区间生命周期复用；Split 视图不占空间；逐元素算子可原地覆盖已死亡的输入。
MemoryPlan plan_reuse(const Model& m);

// 检查对齐、范围、视图落点和活跃张量之间的重叠；ASan 无法代替 arena 内部的此项检查。
void validate_plan(const Model& m, const MemoryPlan& plan);

// 独立于具体规划的同时存活峰值：按生命周期统计每一步必须同时驻留的字节数（视图记在父张量上）。
// 这是 arena 大小的理论下界；贪心复用可能高于它。
size_t peak_live_bytes(const Model& m);

// 把 batch=1 时 Split 的输出标记为父张量的零拷贝切片。多 batch 时切片在内存里不连续，保持拷贝。
// 必须在规划之前调用（执行器负责）；重复调用是幂等的。
void mark_split_views(Model& m);

// 视图的最外层父张量（沿 view_of 链一直走到非视图）。
int root_of(const Model& m, int tensor_id);

}  // namespace yi
