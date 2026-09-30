"""用小型 ONNX 模型验证 CPU 标量算子；参考由 ORT 生成，不接受 REF 占位通过。"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("YINFER_BIN", ROOT / "build" / "yinfer")).resolve()


def reference(nodes, inputs, constants, output_shape):
    graph = helper.make_graph(
        nodes, "operator_test",
        [helper.make_tensor_value_info(k, TensorProto.FLOAT, list(v.shape)) for k, v in inputs.items()],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(v, k) for k, v in constants.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"]).run(
        ["y"], inputs
    )[0]


class RefOpsTest(unittest.TestCase):
    backend = os.environ.get("YINFER_BACKEND", "scalar")

    def run_kernel(self, op, inputs, constants, attrs, expected, *, exact=False, rejected=False, memory="reuse", return_output=False):
        # 每个测试拥有独立模型、参考文件；不会污染用户的 artifacts。
        with tempfile.TemporaryDirectory(prefix="yinfer-op-") as tmp:
            root = Path(tmp)
            lines, weights, ref_blob, index = [], bytearray(), bytearray(), []
            for name, value in {**inputs, **constants, "y": expected}.items():
                v = np.ascontiguousarray(value, dtype="<f4")
                shape = ",".join(map(str, v.shape))
                if name in constants:
                    weights.extend(b"\0" * (-len(weights) % 64))
                    lines.append(f"tensor {name} {shape} const {len(weights)}")
                    weights.extend(v.tobytes())
                else:
                    lines.append(f"tensor {name} {shape} act")
                    ref_blob.extend(b"\0" * (-len(ref_blob) % 64))
                    index.append(f"{name} {len(ref_blob)} {v.size} {shape}")
                    ref_blob.extend(v.tobytes())
            header = ["format yolov8-infer 1", f"weights weights.bin {len(weights)}"]
            header += [f"input {name}" for name in inputs] + ["output y"]
            args = " ".join(f"{key}=" + ",".join(map(str, value)) for key, value in attrs.items())
            names = ",".join([*inputs, *constants])
            lines.append(f"node {op} tested in={names} out=y {args}")
            (root / "model.txt").write_text("\n".join(header + lines) + "\n", encoding="utf-8")
            (root / "weights.bin").write_bytes(weights)
            (root / "ref.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
            (root / "ref.bin").write_bytes(ref_blob)
            result = subprocess.run(
                [str(BINARY), "verify", str(root), str(root), "--tol", "0" if exact else "1e-5", "--brief",
                 "--backend", self.backend, "--memory", memory],
                capture_output=True, encoding="utf-8", timeout=60, check=False,
            )
            detail = result.stdout + result.stderr
            if rejected:
                self.assertEqual(result.returncode, 3, detail)
                return
            self.assertEqual(result.returncode, 0, detail)
            counts = re.search(r"EXACT (\d+) OK (\d+) FAIL (\d+) REF (\d+)", result.stdout)
            self.assertIsNotNone(counts, detail)
            matched, close, failed, ref = map(int, counts.groups())
            self.assertEqual((matched + close, failed, ref), (1, 0, 0), detail)
            if exact:
                self.assertEqual(matched, 1, detail)
            if return_output:
                self.assertEqual(len(inputs), 1, "dump 测试入口只接收单输入")
                np.ascontiguousarray(next(iter(inputs.values())), dtype="<f4").tofile(root / "input.bin")
                run = subprocess.run(
                    [str(BINARY), "run", str(root), str(root / "input.bin"), "--backend", self.backend,
                     "--dump-dir", str(root / "dump")],
                    capture_output=True, encoding="utf-8", timeout=60, check=False,
                )
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                return np.fromfile(root / "dump" / "output0.bin", dtype="<f4").reshape(expected.shape)

    def test_maxpool_negative_padding(self):
        x = -np.arange(1, 13, dtype=np.float32).reshape(1, 1, 3, 4)
        shape = [1, 1, 3, 4]
        y = reference([helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3],
                                        strides=[1, 1], pads=[1, 1, 1, 1])], {"x": x}, {}, shape)
        self.assertTrue(np.all(y < 0))
        self.run_kernel("MaxPool", {"x": x}, {}, {"kernel": [3, 3], "stride": [1, 1], "pad": [1, 1, 1, 1]}, y, exact=True)

    def test_maxpool_sppf_multibatch(self):
        x = np.random.default_rng(11).normal(size=(2, 3, 6, 7)).astype(np.float32)
        y = reference([helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[5, 5],
                                        strides=[1, 1], pads=[2, 2, 2, 2])], {"x": x}, {}, list(x.shape))
        self.run_kernel("MaxPool", {"x": x}, {}, {"kernel": [5, 5], "stride": [1, 1], "pad": [2, 2, 2, 2]}, y, exact=True)

    def test_maxpool_asymmetric_stride(self):
        x = np.random.default_rng(12).normal(size=(1, 2, 5, 7)).astype(np.float32)
        y = reference([helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 3],
                                        strides=[2, 2], pads=[1, 0, 0, 1])], {"x": x}, {}, [1, 2, 3, 3])
        self.run_kernel("MaxPool", {"x": x}, {}, {"kernel": [2, 3], "stride": [2, 2], "pad": [1, 0, 0, 1]}, y, exact=True)

    def test_upsample_multibatch_different_scales(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
        for sh, sw in ((2, 3), (1, 1), (3, 1)):
            with self.subTest(scale=(sh, sw)):
                scales = np.array([1, 1, sh, sw], dtype=np.float32)
                y = reference([helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="nearest",
                                                coordinate_transformation_mode="asymmetric", nearest_mode="floor")],
                              {"x": x}, {"scales": scales}, [2, 2, 2 * sh, 3 * sw])
                # scales 已降级为属性，不作为运行时数据输入。
                self.run_kernel("UpsampleNearest", {"x": x}, {}, {"scale": [sh, sw]}, y, exact=True)

    def test_silu_range(self):
        x = np.array([-1e30, -100, -20, -5, -1, -0.0, 0, 0.1, 1, 5, 20, 100], dtype=np.float32).reshape(1, 1, 3, 4)
        y = reference([helper.make_node("Sigmoid", ["x"], ["sig"]),
                       helper.make_node("Mul", ["x", "sig"], ["y"])], {"x": x}, {}, list(x.shape))
        self.run_kernel("SiLU", {"x": x}, {}, {}, y)

    def test_silu_negative_tail(self):
        # 不让大正值把小负值的错误稀释掉，另测负半轴。
        x = np.linspace(-8, 0, 257, dtype=np.float32).reshape(1, 1, 1, 257)
        y = reference([helper.make_node("Sigmoid", ["x"], ["sig"]),
                       helper.make_node("Mul", ["x", "sig"], ["y"])], {"x": x}, {}, list(x.shape))
        self.run_kernel("SiLU", {"x": x}, {}, {}, y)

    def test_conv_variants(self):
        cases = [
            ((1, 2, 3, 5), 3, (1, 1), (1, 1), (0, 0, 0, 0)),
            ((2, 3, 5, 7), 4, (3, 3), (1, 1), (1, 1, 1, 1)),
            ((1, 2, 6, 7), 3, (3, 3), (2, 2), (1, 1, 1, 1)),
            ((2, 2, 4, 6), 3, (2, 3), (1, 2), (0, 1, 1, 0)),
        ]
        rng = np.random.default_rng(13)
        for shape, cout, kernel, stride, pad in cases:
            with self.subTest(shape=shape, kernel=kernel, stride=stride, pad=pad):
                x = rng.normal(size=shape).astype(np.float32)
                w = rng.normal(size=(cout, shape[1], *kernel)).astype(np.float32)
                b = rng.normal(size=(cout,)).astype(np.float32)
                ho = (shape[2] + pad[0] + pad[2] - kernel[0]) // stride[0] + 1
                wo = (shape[3] + pad[1] + pad[3] - kernel[1]) // stride[1] + 1
                attrs = {"kernel": list(kernel), "stride": list(stride), "pad": list(pad)}
                y = reference([helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=kernel,
                                                strides=stride, pads=pad)], {"x": x}, {"w": w, "b": b}, [shape[0], cout, ho, wo])
                self.run_kernel("Conv", {"x": x}, {"w": w, "b": b}, attrs, y)

    def test_conv_known_cross_correlation(self):
        x = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        w = np.array([1, 2, 3, 4], dtype=np.float32).reshape(1, 1, 2, 2)
        b = np.array([1], dtype=np.float32)
        # 不翻转卷积核：左上角 1*1 + 2*2 + 4*3 + 5*4 + bias = 38。
        y = np.array([38, 48, 68, 78], dtype=np.float32).reshape(1, 1, 2, 2)
        self.run_kernel("Conv", {"x": x}, {"w": w, "b": b},
                        {"kernel": [2, 2], "stride": [1, 1], "pad": [0, 0, 0, 0]}, y, exact=True)

    def test_spatial_rejects_invalid_parameters(self):
        x = np.ones((1, 1, 3, 4), dtype=np.float32)
        cases = [
            ("MaxPool", {"kernel": [3, 3], "stride": [0, 1], "pad": [1, 1, 1, 1]}, x),
            ("MaxPool", {"kernel": [3], "stride": [1, 1], "pad": [1, 1, 1, 1]}, x),
            ("MaxPool", {"kernel": [3, 3], "stride": [1, 1], "pad": [1, 1, 1, 1]}, np.zeros((1, 1, 2, 4), np.float32)),
            ("UpsampleNearest", {"scale": [0, 2]}, x),
            ("UpsampleNearest", {"scale": [2, 2]}, x),
        ]
        for op, attrs, y in cases:
            with self.subTest(op=op, attrs=attrs, shape=y.shape):
                self.run_kernel(op, {"x": x}, {}, attrs, y, rejected=True)

    def test_silu_rejects_wrong_shape(self):
        self.run_kernel("SiLU", {"x": np.ones((1, 1, 2, 3), np.float32)}, {}, {},
                        np.zeros((1, 1, 3, 2), np.float32), rejected=True)

    def test_conv_rejects_wrong_bias_and_channels(self):
        x = np.ones((1, 2, 3, 4), np.float32)
        attrs = {"kernel": [1, 1], "stride": [1, 1], "pad": [0, 0, 0, 0]}
        for wshape, bshape in (((3, 2, 1, 1), (2,)), ((3, 1, 1, 1), (3,))):
            with self.subTest(weight=wshape, bias=bshape):
                self.run_kernel("Conv", {"x": x}, {"w": np.ones(wshape, np.float32), "b": np.zeros(bshape, np.float32)},
                                attrs, np.zeros((1, 3, 3, 4), np.float32), rejected=True)


if __name__ == "__main__":
    unittest.main()
