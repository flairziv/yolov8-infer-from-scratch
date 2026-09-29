// executor.h —— 执行器：按内存规划给激活分配内存，再按拓扑序逐个节点调用算子函数
#pragma once
#include <map>
#include <string>
#include <vector>

#include "memory_plan.h"
#include "model.h"
#include "ops/ops.h"

namespace yi {

class Executor {
public:
    // 构造时做完所有"一次性"的工作：规划并分配激活内存、给每个节点查好算子函数。
    // 注意：执行器把激活的地址写进 model.tensors[*].data，所以一个 Model 同一时间只能配一个 Executor
    explicit Executor(Model& model, bool reuse = true);

    size_t arena_bytes() const { return arena_.bytes(); }
    void set_input(size_t k, const float* data);        // 把第 k 个图输入的数据拷进它的激活内存
    void run();                                          // 整图执行；有算子没实现就抛异常并列出来
    void run_node(size_t i);                             // 复用模式下必须按拓扑序执行；过期中间结果可能被覆盖
    bool has_kernel(size_t i) const { return kernels_[i] != nullptr; }
    std::map<std::string, int> missing_kernels() const;  // 还没实现的算子名 → 节点个数

private:
    Model& model_;
    AlignedBuffer arena_;            // 所有激活共用的一大块内存，按内存规划的偏移切给各个张量
    std::vector<KernelFn> kernels_;  // kernels_[i] 是第 i 个节点的算子函数；执行时不再查表
};

}  // namespace yi
