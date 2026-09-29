// memory_plan.h —— 激活内存规划：决定每个激活张量放在 arena（一大块连续内存）的哪个偏移
#pragma once
#include <cstdint>
#include <vector>

#include "model.h"

namespace yi {

struct MemoryPlan {
    std::vector<int64_t> offset;    // offset[t]：激活张量 t 在 arena 里的字节偏移；常量是 -1
    size_t total = 0;               // arena 一共要多少字节
};

// 对照模式：每个激活独占空间。
MemoryPlan plan_naive(const Model& m);

// 顺序执行、无视图别名：按闭区间生命周期复用；图输入/输出保留到执行结束。
MemoryPlan plan_reuse(const Model& m);

// 检查对齐、范围和活跃张量之间的重叠；ASan 无法代替 arena 内部的此项检查。
void validate_plan(const Model& m, const MemoryPlan& plan);

}  // namespace yi
