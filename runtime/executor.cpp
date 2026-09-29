// executor.cpp
#include "executor.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace yi {

Executor::Executor(Model& model, bool reuse) : model_(model) {
    const MemoryPlan plan = reuse ? plan_reuse(model_) : plan_naive(model_);
    arena_ = AlignedBuffer(plan.total);
    for (size_t i = 0; i < model_.tensors.size(); ++i) {
        Tensor& t = model_.tensors[i];
        if (!t.is_const) t.data = reinterpret_cast<float*>(arena_.as<char>() + plan.offset[i]);
    }
    for (const Node& n : model_.nodes) kernels_.push_back(find_kernel(n.op));
}

void Executor::set_input(size_t k, const float* data) {
    YI_CHECK(k < model_.inputs.size(), "模型没有第 " << k << " 个图输入");
    Tensor& t = model_.tensors[model_.inputs[k]];
    std::memcpy(t.data, data, t.bytes());
}

std::map<std::string, int> Executor::missing_kernels() const {
    std::map<std::string, int> miss;
    for (size_t i = 0; i < kernels_.size(); ++i)
        if (!kernels_[i]) ++miss[model_.nodes[i].op];
    return miss;
}

void Executor::run_node(size_t i) {
    const Node& n = model_.nodes[i];
    YI_CHECK(kernels_[i], "算子 " << n.op << " 还没有实现（节点 " << n.name << "）");
    kernels_[i](n, model_.tensors);
}

void Executor::run() {
    const auto miss = missing_kernels();
    if (!miss.empty()) {
        std::ostringstream oss;
        oss << "还没实现的算子:";
        for (const auto& [op, cnt] : miss) oss << " " << op << "×" << cnt;
        throw std::runtime_error(oss.str());
    }
    for (size_t i = 0; i < model_.nodes.size(); ++i) kernels_[i](model_.nodes[i], model_.tensors);
}

}  // namespace yi
