"""Conv+SiLU 收尾融合的消融验证：同一后端、同一权重下，融合与未融合必须逐位一致。

原理：融合只把"卷积结果写回内存、SiLU 再读回来算"改成"在寄存器里直接算"，
数值路径不变；中间那次 f32 存储-读取是精确的，所以两条路径应产生相同比特。
前提：输出空间 N=Ho*Wo 能被向量宽度整除，且没有向量组触发 |x|>80 的标量回退；
测试形状按这个前提选取。
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

import test_ref_ops as ref_tests


def build_model(root, fused, x, w, b, y_shape, kernel, stride, pad):
    """写一个最小模型：融合版是单节点 Conv(act=1)，未融合版是 Conv → SiLU 两节点。"""
    root.mkdir()
    weights = bytearray()
    tensor_lines = []

    def add_const(name, value):
        v = np.ascontiguousarray(value, dtype="<f4")
        weights.extend(b"\0" * (-len(weights) % 64))
        tensor_lines.append(f"tensor {name} {','.join(map(str, v.shape))} const {len(weights)}")
        weights.extend(v.tobytes())

    add_const("w", w)
    add_const("b", b)
    tensor_lines.append(f"tensor x {','.join(map(str, x.shape))} act")
    y_shape_str = ",".join(map(str, y_shape))
    attrs = f"kernel={kernel[0]},{kernel[1]} stride={stride[0]},{stride[1]} pad={','.join(map(str, pad))}"
    if fused:
        tensor_lines.append(f"tensor y {y_shape_str} act")
        node_lines = [f"node Conv fused in=x,w,b out=y {attrs} act=1"]
    else:
        tensor_lines.append(f"tensor c {y_shape_str} act")
        tensor_lines.append(f"tensor y {y_shape_str} act")
        node_lines = [f"node Conv conv in=x,w,b out=c {attrs}", "node SiLU silu in=c out=y"]
    header = ["format yolov8-infer 2", f"weights weights.bin {len(weights)}", "input x", "output y"]
    (root / "model.txt").write_text("\n".join(header + tensor_lines + node_lines) + "\n", encoding="utf-8")
    (root / "weights.bin").write_bytes(weights)
    (root / "input.bin").write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def run_and_read(root, backend):
    dump = root / "dump"
    result = subprocess.run(
        [str(ref_tests.BINARY), "run", str(root), str(root / "input.bin"), "--backend", backend,
         "--dump-dir", str(dump)],
        capture_output=True, encoding="utf-8", timeout=60, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return np.fromfile(dump / "output0.bin", dtype="<f4")


class FusionTest(unittest.TestCase):
    def available_backends(self):
        out = subprocess.run([str(ref_tests.BINARY), "backends"], capture_output=True,
                             encoding="utf-8", check=True).stdout
        return dict(line.split() for line in out.splitlines() if len(line.split()) == 2)

    def test_fused_matches_unfused_bitwise(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
        w = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
        # N = 4*8 = 32，能被 SSE 与 AVX2 的向量宽度整除，两组实现的 lane 分组一致。
        for bias in (rng.normal(size=(3,)).astype(np.float32), np.full(3, -3.0, dtype=np.float32)):
            for backend, state in self.available_backends().items():
                if state != "available":
                    continue
                with self.subTest(backend=backend, bias=float(bias[0])):
                    with tempfile.TemporaryDirectory(prefix="yinfer-fusion-") as tmp:
                        root = Path(tmp)
                        build_model(root / "fused", True, x, w, b=bias, y_shape=(1, 3, 4, 8),
                                    kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1))
                        build_model(root / "unfused", False, x, w, b=bias, y_shape=(1, 3, 4, 8),
                                    kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1))
                        fused = run_and_read(root / "fused", backend)
                        unfused = run_and_read(root / "unfused", backend)
                        np.testing.assert_array_equal(
                            fused.view(np.uint32), unfused.view(np.uint32),
                            err_msg=f"{backend} 下融合与未融合输出不逐位一致")

    def test_fused_1x1_matches_unfused_bitwise(self):
        # 1×1 直通路径与未融合的 im2col 路径也应逐位一致。
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 3, 4, 8)).astype(np.float32)
        w = rng.normal(size=(5, 3, 1, 1)).astype(np.float32)
        b = rng.normal(size=(5,)).astype(np.float32)
        for backend, state in self.available_backends().items():
            if state != "available":
                continue
            with self.subTest(backend=backend):
                with tempfile.TemporaryDirectory(prefix="yinfer-fusion-") as tmp:
                    root = Path(tmp)
                    for name, fused in (("fused", True), ("unfused", False)):
                        build_model(root / name, fused, x, w, b, y_shape=(2, 5, 4, 8),
                                    kernel=(1, 1), stride=(1, 1), pad=(0, 0, 0, 0))
                    fused = run_and_read(root / "fused", backend)
                    unfused = run_and_read(root / "unfused", backend)
                    np.testing.assert_array_equal(fused.view(np.uint32), unfused.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
