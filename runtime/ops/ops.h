// ops.h —— 算子注册表：算子名 → 实现函数
#pragma once
#include <string>
#include <vector>

#include "model.h"

namespace yi {

// 所有算子函数的统一签名：从 node.inputs 指向的张量读，往 node.outputs 指向的张量写。
// 输出的内存由执行器事先分配好，算子只负责填数，不分配内存
using KernelFn = void (*)(const Node& node, std::vector<Tensor>& tensors);

KernelFn find_kernel(const std::string& op);   // 还没实现的算子返回 nullptr

}  // namespace yi
