"""SIMD 后端回归：复用原算子测试，再覆盖向量尾部、边界窗口和微内核尾块。"""
import subprocess
import unittest

import numpy as np
from onnx import helper

import test_ref_ops as ref_tests


class SimdCases:
    @classmethod
    def setUpClass(cls):
        result = subprocess.run([str(ref_tests.BINARY), "backends"], capture_output=True,
                                encoding="utf-8", check=False)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        states = dict(line.split() for line in result.stdout.splitlines() if len(line.split()) == 2)
        if states.get(cls.backend) == "unavailable":
            raise unittest.SkipTest(f"当前构建或硬件不支持 {cls.backend}")
        if states.get(cls.backend) != "available":
            raise RuntimeError(f"未能确认后端状态: {result.stdout}")

    def test_add_vector_and_scalar_tails(self):
        rng = np.random.default_rng(21)
        for width in (1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33):
            with self.subTest(width=width):
                a = rng.normal(size=(1, 1, 1, width)).astype(np.float32)
                b = rng.normal(size=a.shape).astype(np.float32)
                self.run_kernel("Add", {"a": a, "b": b}, {}, {}, a + b, exact=True)

    def test_pool_vector_interior_and_borders(self):
        rng = np.random.default_rng(22)
        for width in (1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33):
            with self.subTest(width=width):
                x = -np.abs(rng.normal(size=(2, 2, 5, width)).astype(np.float32))
                y = ref_tests.reference(
                    [helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3],
                                      strides=[1, 1], pads=[1, 1, 1, 1])],
                    {"x": x}, {}, list(x.shape))
                self.run_kernel("MaxPool", {"x": x}, {},
                                {"kernel": [3, 3], "stride": [1, 1], "pad": [1, 1, 1, 1]}, y, exact=True)

    def test_silu_dense_pointwise_accuracy(self):
        # 不只看整张量最大值归一化，独立导出输出并逐元素对比 float64 数学参考。
        dense = np.linspace(-18, 18, 4097, dtype=np.float32)
        transitions = (np.arange(-20, 21) * np.log(2)).astype(np.float32)
        x = np.concatenate([dense, transitions - 1e-5, transitions + 1e-5,
                            np.array([-80, -100, -1e30, -0.0, 0.0, 80, 100, 1e30], dtype=np.float32)])
        x = x.reshape(1, 1, 1, -1)
        y = ref_tests.reference([helper.make_node("Sigmoid", ["x"], ["s"]),
                                 helper.make_node("Mul", ["x", "s"], ["y"])], {"x": x}, {}, list(x.shape))
        actual = self.run_kernel("SiLU", {"x": x}, {}, {}, y, return_output=True)
        v = x.astype(np.float64)
        e = np.exp(-np.abs(v))
        expected = v * np.where(v >= 0, 1 / (1 + e), e / (1 + e))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
        zero = (x == 0)
        np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(x[zero]))

    def test_conv_microkernel_tails(self):
        rng = np.random.default_rng(23)
        # M=Cout、N=Ho*Wo、K=Cin*Kh*Kw 都不能假设整除分块。
        for cout, width, cin, kernel in ((1, 1, 1, 1), (3, 7, 3, 1), (5, 9, 5, 1),
                                         (6, 17, 3, 3), (7, 31, 2, 3), (9, 33, 5, 1)):
            with self.subTest(cout=cout, width=width, cin=cin, kernel=kernel):
                x = rng.normal(size=(2, cin, 3, width)).astype(np.float32)
                w = (rng.normal(size=(cout, cin, kernel, kernel)) * 0.1).astype(np.float32)
                b = rng.normal(size=(cout,)).astype(np.float32)
                pad = [kernel // 2] * 4
                y = ref_tests.reference([helper.make_node("Conv", ["x", "w", "b"], ["y"],
                                          kernel_shape=[kernel, kernel], strides=[1, 1], pads=pad)],
                                        {"x": x}, {"w": w, "b": b}, [2, cout, 3, width])
                attrs = {"kernel": [kernel, kernel], "stride": [1, 1], "pad": pad}
                actual = self.run_kernel("Conv", {"x": x}, {"w": w, "b": b}, attrs, y, return_output=True)
                np.testing.assert_allclose(actual, y, rtol=2e-5, atol=2e-5)
                # 同一形状的 naive 规划也走相同后端，防止默认复用掩盖输入输出地址假设。
                self.run_kernel("Conv", {"x": x}, {"w": w, "b": b}, attrs, y, memory="naive")


class SseOpsTest(SimdCases, ref_tests.RefOpsTest):
    backend = "sse"


class Avx2OpsTest(SimdCases, ref_tests.RefOpsTest):
    backend = "avx2"


if __name__ == "__main__":
    unittest.main()
