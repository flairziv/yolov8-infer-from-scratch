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
    // CUDA 后端把激活 arena 和权重放在显存里，主机侧保留一份影子缓冲用于读取。
    explicit Executor(Model& model, bool reuse = true, Backend backend = Backend::Scalar, int threads = 1);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    size_t arena_bytes() const { return arena_bytes_; }
    size_t workspace_bytes() const { return workspace_storage_.bytes(); }
    Backend backend() const { return backend_; }
    int threads() const { return threads_; }
    size_t simd_nodes() const { return simd_nodes_; }
    size_t fallback_nodes() const { return fallback_nodes_; }
    size_t view_tensors() const { return view_tensors_; }      // 零拷贝视图张量个数
    size_t in_place_tensors() const { return in_place_tensors_; }  // 原地覆盖的输出个数
    void set_input(size_t k, const float* data);
    void set_tensor(size_t id, const float* data);   // 按张量下标写数据（对拍时加载参考值用）
    // 读一个张量：CPU 后端直接返回 data；CUDA 后端先从显存拷回主机影子再返回。
    float* host_ptr(size_t id);
    void push_tensor(size_t id);                     // 主机影子 → 设备（改过 host_ptr 之后调用）
    void run();
    void run_node(size_t i); // 复用模式下按拓扑序执行；中间值过期后可以被覆盖
    bool has_kernel(size_t i) const { return kernels_[i] != nullptr; }
    std::map<std::string, int> missing_kernels() const;

private:
    size_t shadow_offset(size_t id) const;
    Model& model_;
    Backend backend_;
    int threads_ = 1;
    std::unique_ptr<ThreadPool> pool_;   // threads_ == 1 时为 nullptr，算子走串行路径
    AlignedBuffer arena_;
    AlignedBuffer workspace_storage_;
    AlignedBuffer host_shadow_;          // 只给 CUDA 用：显存激活的读取镜像
    void* device_arena_ = nullptr;
    void* device_weights_ = nullptr;
    std::vector<int64_t> offsets_;       // 内存规划的字节偏移（CUDA 回读时用来定位影子缓冲）
    size_t arena_bytes_ = 0;
    Workspace workspace_;
    std::vector<KernelFn> kernels_;
    size_t simd_nodes_ = 0;
    size_t fallback_nodes_ = 0;
    size_t view_tensors_ = 0;
    size_t in_place_tensors_ = 0;
};

}  // namespace yi
