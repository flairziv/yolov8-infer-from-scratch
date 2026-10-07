// ops.h —— 统一算子接口、执行器所有的临时工作区，以及后端分派。
#pragma once
#include <string>
#include <vector>

#include "backend.h"
#include "model.h"

namespace yi {

class ThreadPool;

// 只借用执行器持有的内存，不负责分配/释放。各顺序执行节点复用同一块临时空间。
// pool 由执行器按线程数注入：nullptr 表示单线程，算子内部一律走串行路径。
struct Workspace {
    float* data = nullptr;
    size_t bytes = 0;
    ThreadPool* pool = nullptr;
};

using KernelFn = void (*)(const Node& node, std::vector<Tensor>& tensors, Workspace& workspace);

KernelFn find_kernel(const std::string& op);  // 标量参考实现；未实现返回 nullptr
struct KernelSelection { KernelFn fn = nullptr; bool simd = false; };
KernelSelection select_kernel(const Node& node, const std::vector<Tensor>& tensors, Backend backend);
size_t kernel_workspace_bytes(const Node& node, const std::vector<Tensor>& tensors, Backend backend, int threads);

}  // namespace yi
