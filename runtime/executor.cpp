// executor.cpp
#include "executor.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace yi {

Executor::Executor(Model& model, bool reuse, Backend backend, int threads)
    : model_(model), backend_(backend), threads_(threads) {
    require_backend(backend_);
    YI_CHECK(threads_ >= 1, "线程数至少是 1");
    if (threads_ > 1) pool_ = std::make_unique<ThreadPool>(threads_);
    mark_split_views(model_);   // batch=1 的 Split 输出做成零拷贝视图，规划与执行都按视图处理
    const MemoryPlan plan = reuse ? plan_reuse(model_) : plan_naive(model_);
    arena_ = AlignedBuffer(plan.total);
    // 先给普通激活分配地址，再让视图指进父张量的切片（视图的父张量一定不是视图）。
    for (size_t i = 0; i < model_.tensors.size(); ++i) {
        Tensor& t = model_.tensors[i];
        if (!t.is_const && !t.is_view()) t.data = reinterpret_cast<float*>(arena_.as<char>() + plan.offset[i]);
    }
    for (Tensor& t : model_.tensors)
        if (t.is_view())
            t.data = reinterpret_cast<float*>(reinterpret_cast<char*>(model_.tensors[t.view_of].data) + t.view_offset);
    for (const Tensor& t : model_.tensors)
        if (t.is_view()) ++view_tensors_;
    in_place_tensors_ = plan.in_place.size();
    size_t workspace_bytes = 0;
    for (const Node& n : model_.nodes) {
        const auto selected = select_kernel(n, model_.tensors, backend_);
        kernels_.push_back(selected.fn);
        if (selected.simd) ++simd_nodes_;
        else if (backend_ != Backend::Scalar && selected.fn) ++fallback_nodes_;
        workspace_bytes = std::max(workspace_bytes, kernel_workspace_bytes(n, model_.tensors, backend_, threads_));
    }
    workspace_storage_ = AlignedBuffer(workspace_bytes);
    workspace_ = {workspace_storage_.as<float>(), workspace_bytes, pool_.get()};
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
    kernels_[i](n, model_.tensors, workspace_);
}

void Executor::run() {
    const auto miss = missing_kernels();
    if (!miss.empty()) {
        std::ostringstream oss;
        oss << "还没实现的算子:";
        for (const auto& [op, cnt] : miss) oss << " " << op << "×" << cnt;
        throw std::runtime_error(oss.str());
    }
    for (size_t i = 0; i < model_.nodes.size(); ++i) kernels_[i](model_.nodes[i], model_.tensors, workspace_);
}

}  // namespace yi
