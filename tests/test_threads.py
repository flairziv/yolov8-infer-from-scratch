"""多线程执行的一致性回归：任意线程数的输出必须与单线程逐位一致。

用一个足够大的小型模型覆盖并行 im2col、卷积列块、逐元素分块与按平面池化；
对照对象是同一二进制的单线程输出，不需要 ORT。工作单元彼此独立，
调度顺序不允许影响任何元素的值——这是多线程版本的硬性契约。
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("YINFER_BIN", ROOT / "build" / "yinfer")).resolve()


def available_backends():
    result = subprocess.run([str(BINARY), "backends"], capture_output=True, encoding="utf-8", timeout=30)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return [line.split()[0] for line in result.stdout.splitlines() if line.endswith(" available")]


def write_model(root: Path):
    # Conv 1×4×128×128 → 8×128×128：K*N=589824 超过 im2col 并行阈值，列块数 1024；
    # 后续 SiLU/Add 各 131072 个元素、MaxPool 8 个平面，都会走各自的并行路径。
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 4, 128, 128)).astype("<f4")
    w = (rng.normal(size=(8, 4, 3, 3)) * 0.2).astype("<f4")
    b = rng.normal(size=(8,)).astype("<f4")
    weights, decls = bytearray(), []
    for name, value in (("w", w), ("b", b)):
        weights.extend(b"\0" * (-len(weights) % 64))
        decls.append(f"tensor {name} {','.join(map(str, value.shape))} const {len(weights)}")
        weights.extend(value.tobytes())
    lines = [
        "format yolov8-infer 2",
        f"weights weights.bin {len(weights)}",
        "input x",
        "output y",
        "tensor x 1,4,128,128 act",
        *decls,
        "tensor c 1,8,128,128 act",
        "tensor s 1,8,128,128 act",
        "tensor a 1,8,128,128 act",
        "tensor m 1,8,128,128 act",
        "tensor y 1,16,128,128 act",
        "node Conv conv in=x,w,b out=c kernel=3,3 stride=1,1 pad=1,1,1,1 act=1",
        "node SiLU silu in=c out=s",
        "node Add add in=s,c out=a",
        "node MaxPool pool in=a out=m kernel=3,3 stride=1,1 pad=1,1,1,1",
        "node Concat concat in=m,m out=y axis=1",
    ]
    (root / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "weights.bin").write_bytes(weights)
    x.tofile(root / "input.bin")


def run_yinfer(*args, expect=0):
    result = subprocess.run([str(BINARY), *map(str, args)], capture_output=True, encoding="utf-8", timeout=300)
    if expect is not None:
        assert result.returncode == expect, result.stdout + result.stderr
    return result


class ThreadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = available_backends()

    def dump(self, root, backend, threads, label):
        out = root / f"dump-{label}"
        run_yinfer("run", root, root / "input.bin", "--backend", backend, "--threads", threads, "--dump-dir", out)
        return np.fromfile(out / "output0.bin", dtype="<f4")

    def test_threads_bitwise_identical(self):
        with tempfile.TemporaryDirectory(prefix="yinfer-threads-") as tmp:
            root = Path(tmp)
            write_model(root)
            for backend in self.backends:
                with self.subTest(backend=backend):
                    single = self.dump(root, backend, 1, "t1")
                    quad = self.dump(root, backend, 4, "t4")
                    many = self.dump(root, backend, 16, "t16")   # 线程数远超列块粒度之外也不会出错
                    np.testing.assert_array_equal(single.view(np.uint32), quad.view(np.uint32))
                    np.testing.assert_array_equal(single.view(np.uint32), many.view(np.uint32))

    def test_workspace_grows_with_threads(self):
        simd = [b for b in self.backends if b != "scalar"]
        if not simd:
            self.skipTest("没有可用的 SIMD 后端")
        with tempfile.TemporaryDirectory(prefix="yinfer-threads-") as tmp:
            root = Path(tmp)
            write_model(root)

            def workspace(threads):
                text = run_yinfer("info", root, "--backend", simd[-1], "--threads", threads).stdout
                match = re.search(r"workspace_bytes=(\d+)", text)
                self.assertIsNotNone(match, text)
                return int(match.group(1))

            # 每个线程一块打包面板，工作区随线程数增长（列展开缓冲只有一份）。
            self.assertGreater(workspace(4), workspace(1))

    def test_invalid_threads_rejected(self):
        with tempfile.TemporaryDirectory(prefix="yinfer-threads-") as tmp:
            root = Path(tmp)
            write_model(root)
            for value in ("0", "-1", "x", "1x", "1e2"):
                with self.subTest(threads=value):
                    run_yinfer("run", root, root / "input.bin", "--threads", value, expect=3)


if __name__ == "__main__":
    unittest.main()
