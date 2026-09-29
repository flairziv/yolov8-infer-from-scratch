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

// 阶段 0：每个激活独占一段，谁也不复用，总量 = 所有激活之和。
// 阶段 1 会加 plan_reuse：一个张量最后一次被读之后，它那段内存就可以给后面产生的张量用。
MemoryPlan plan_naive(const Model& m);

}  // namespace yi
