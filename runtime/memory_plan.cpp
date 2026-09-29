// memory_plan.cpp
#include "memory_plan.h"

namespace yi {

MemoryPlan plan_naive(const Model& m) {
    MemoryPlan p;
    p.offset.assign(m.tensors.size(), -1);
    size_t off = 0;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const Tensor& t = m.tensors[i];
        if (t.is_const) continue;
        p.offset[i] = static_cast<int64_t>(off);
        off += align_up(t.bytes());     // 下一段的起点也保持 64 字节对齐
    }
    p.total = off;
    return p;
}

}  // namespace yi
