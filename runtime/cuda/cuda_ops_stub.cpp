// cuda_ops_stub.cpp —— 不带 CUDA 的构建：没有任何 CUDA 算子，分派会把它记为缺失算子。
#include "ops/ops.h"

namespace yi {

KernelFn find_cuda_kernel(const std::string&) { return nullptr; }

}  // namespace yi
