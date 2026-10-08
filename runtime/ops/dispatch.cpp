// dispatch.cpp —— baseline ISA 分派；共享标量实现不是“未实现”的 REF 占位。
#include "ops.h"
#include "simd_common.h"

namespace yi {
#if defined(YI_X86_SIMD)
KernelFn find_sse_kernel(const std::string& op);
KernelFn find_avx2_kernel(const std::string& op);
#endif
// CUDA 未启用时由 stub 提供，永远返回 nullptr。
KernelFn find_cuda_kernel(const std::string& op);

KernelSelection select_kernel(const Node& node, const std::vector<Tensor>& tensors, Backend backend) {
    require_backend(backend);
    const KernelFn scalar = find_kernel(node.op);
    if (backend == Backend::Scalar) return {scalar, false};
    if (backend == Backend::CUDA) {
        const KernelFn cuda = find_cuda_kernel(node.op);
        // 没有 CUDA 实现就是真的缺失（不像 CPU 那样退回标量：数据在显存里，CPU 算子读不到）。
        return {cuda, cuda != nullptr};
    }
#if defined(YI_X86_SIMD)
    const int lanes = backend == Backend::SSE ? 4 : 8;
    // 没有完整向量组的简单算子直接使用标量版，诊断里计为回退。
    if ((node.op == "Add" || node.op == "SiLU") && node.outputs.size() == 1 &&
        tensors[node.outputs[0]].numel() < lanes) return {scalar, false};
    if (node.op == "MaxPool") {
        YI_CHECK(node.inputs.size() == 1 && node.outputs.size() == 1, "MaxPool 输入输出数量错误");
        const auto& x = tensors[node.inputs[0]];
        const auto& y = tensors[node.outputs[0]];
        const auto win = detail::window(node, x, y);
        const int64_t first = win.pad[1];
        const int64_t last = std::min(y.shape[3], win.pad[1] + x.shape[3] - win.k[1] + 1);
        if (win.stride[1] != 1 || last - first < lanes) return {scalar, false};
    }
    const auto simd = backend == Backend::SSE ? find_sse_kernel(node.op) : find_avx2_kernel(node.op);
    if (simd) return {simd, true};
#else
    (void)tensors;
#endif
    return {scalar, false};
}

size_t kernel_workspace_bytes(const Node& node, const std::vector<Tensor>& tensors, Backend backend, int threads) {
    require_backend(backend);
    YI_CHECK(threads >= 1, "线程数至少是 1");
    if (backend == Backend::Scalar || backend == Backend::CUDA || node.op != "Conv") return 0;
    return detail::conv_workspace(node, tensors, backend == Backend::SSE ? 8 : 16, threads);
}

}  // namespace yi
