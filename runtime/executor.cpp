// executor.cpp
#include "executor.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "cuda/cuda_api.h"

namespace yi {

Executor::Executor(Model& model, bool reuse, Backend backend, int threads)
    : model_(model), backend_(backend), threads_(threads) {
    require_backend(backend_);
    YI_CHECK(threads_ >= 1, "线程数至少是 1");
    if (threads_ > 1) pool_ = std::make_unique<ThreadPool>(threads_);
    mark_split_views(model_);   // batch=1 的 Split 输出做成零拷贝视图，规划与执行都按视图处理
    const MemoryPlan plan = reuse ? plan_reuse(model_) : plan_naive(model_);
    offsets_ = plan.offset;
    arena_bytes_ = plan.total;
    if (backend_ == Backend::CUDA) {
        // 激活与权重都放在显存里；主机侧保留同样布局的影子缓冲，读的时候按张量拷回来。
        host_shadow_ = AlignedBuffer(plan.total);
        device_arena_ = cuda::alloc(plan.total);
        device_weights_ = cuda::alloc(model_.weights.bytes());
        cuda::to_device(device_weights_, model_.weights.as<char>(), model_.weights.bytes());
        for (size_t i = 0; i < model_.tensors.size(); ++i) {
            Tensor& t = model_.tensors[i];
            if (t.is_const)
                t.data = reinterpret_cast<float*>(static_cast<char*>(device_weights_) + t.const_offset);
            else if (!t.is_view())
                t.data = reinterpret_cast<float*>(static_cast<char*>(device_arena_) + plan.offset[i]);
        }
        for (Tensor& t : model_.tensors)
            if (t.is_view())
                t.data = reinterpret_cast<float*>(reinterpret_cast<char*>(model_.tensors[t.view_of].data) + t.view_offset);
    } else {
        arena_ = AlignedBuffer(plan.total);
        // 先给普通激活分配地址，再让视图指进父张量的切片（视图的父张量一定不是视图）。
        for (size_t i = 0; i < model_.tensors.size(); ++i) {
            Tensor& t = model_.tensors[i];
            if (!t.is_const && !t.is_view()) t.data = reinterpret_cast<float*>(arena_.as<char>() + plan.offset[i]);
        }
        for (Tensor& t : model_.tensors)
            if (t.is_view())
                t.data = reinterpret_cast<float*>(reinterpret_cast<char*>(model_.tensors[t.view_of].data) + t.view_offset);
    }
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

Executor::~Executor() {
    if (device_arena_) cuda::release(device_arena_);
    if (device_weights_) cuda::release(device_weights_);
}

size_t Executor::shadow_offset(size_t id) const {
    const Tensor& t = model_.tensors[id];
    if (!t.is_view()) return static_cast<size_t>(offsets_[id]);
    return static_cast<size_t>(offsets_[root_of(model_, static_cast<int>(id))]) + static_cast<size_t>(t.view_offset);
}

void Executor::set_input(size_t k, const float* data) {
    YI_CHECK(k < model_.inputs.size(), "模型没有第 " << k << " 个图输入");
    set_tensor(static_cast<size_t>(model_.inputs[k]), data);
}

void Executor::set_tensor(size_t id, const float* data) {
    YI_CHECK(id < model_.tensors.size(), "模型没有下标 " << id << " 的张量");
    Tensor& t = model_.tensors[id];
    if (backend_ == Backend::CUDA) {
        std::memcpy(host_shadow_.as<char>() + shadow_offset(id), data, t.bytes());
        cuda::to_device(t.data, data, t.bytes());
    } else {
        std::memcpy(t.data, data, t.bytes());
    }
}

float* Executor::host_ptr(size_t id) {
    YI_CHECK(id < model_.tensors.size(), "模型没有下标 " << id << " 的张量");
    Tensor& t = model_.tensors[id];
    if (backend_ != Backend::CUDA) return t.data;
    float* dst = reinterpret_cast<float*>(host_shadow_.as<char>() + shadow_offset(id));
    cuda::to_host(dst, t.data, t.bytes());
    return dst;
}

void Executor::push_tensor(size_t id) {
    if (backend_ != Backend::CUDA) return;
    Tensor& t = model_.tensors[id];
    cuda::to_device(t.data, host_shadow_.as<char>() + shadow_offset(id), t.bytes());
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
    // CUDA kernel 是异步的：不等它做完，调用者测到的时间就只是"提交时间"。
    if (backend_ == Backend::CUDA) cuda::sync();
}

}  // namespace yi
