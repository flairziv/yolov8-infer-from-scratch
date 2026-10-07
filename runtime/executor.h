// executor.h —— 顺序执行器：激活 arena 与算子临时 workspace 各自一次分配。
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "memory_plan.h"
#include "model.h"
#include "ops/ops.h"
#include "thread_pool.h"

namespace yi {

class Executor {
public:
    // 一个 Model 同时只能绑定一个 Executor；已有两参数调用仍默认使用标量后端。
    // threads > 1 时 SIMD 算子内部按数据并行；标量后端始终单线程，作为逐位参考。
    explicit Executor(Model& model, bool reuse = true, Backend backend = Backend::Scalar, int threads = 1);

    size_t arena_bytes() const { return arena_.bytes(); }
    size_t workspace_bytes() const { return workspace_storage_.bytes(); }
    Backend backend() const { return backend_; }
    int threads() const { return threads_; }
    size_t simd_nodes() const { return simd_nodes_; }
    size_t fallback_nodes() const { return fallback_nodes_; }
    void set_input(size_t k, const float* data);
    void run();
    void run_node(size_t i); // 复用模式下按拓扑序执行；中间值过期后可以被覆盖
    bool has_kernel(size_t i) const { return kernels_[i] != nullptr; }
    std::map<std::string, int> missing_kernels() const;

private:
    Model& model_;
    Backend backend_;
    int threads_ = 1;
    std::unique_ptr<ThreadPool> pool_;   // threads_ == 1 时为 nullptr，算子走串行路径
    AlignedBuffer arena_;
    AlignedBuffer workspace_storage_;
    Workspace workspace_;
    std::vector<KernelFn> kernels_;
    size_t simd_nodes_ = 0;
    size_t fallback_nodes_ = 0;
};

}  // namespace yi
