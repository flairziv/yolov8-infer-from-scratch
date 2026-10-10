"""模型切分与串联执行的回归：合成模型按 --at / --parts 切开，yinfer chain 的输出必须与整体 run 逐位相同。

不依赖 artifacts；覆盖多个跨段边界张量（一个张量既被下游段消费、又在本段内使用）、
每段独立加载、以及切分工具和 chain 命令对非法用法的拒绝。
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("YINFER_BIN", ROOT / "build" / "yinfer")).resolve()
SPLIT = ROOT / "frontend" / "split_model.py"


def write_model(root: Path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1, 4, 32, 32)).astype("<f4")
    w = (rng.normal(size=(8, 4, 3, 3)) * 0.2).astype("<f4")
    b = rng.normal(size=(8,)).astype("<f4")
    weights, decls = bytearray(), []
    for name, value in (("w", w), ("b", b)):
        weights.extend(b"\0" * (-len(weights) % 64))
        decls.append(f"tensor {name} {','.join(map(str, value.shape))} const {len(weights)}")
        weights.extend(value.tobytes())
    lines = [
        "format yolov8-infer 2", f"weights weights.bin {len(weights)}", "input x", "output y", "output s",
        "tensor x 1,4,32,32 act", *decls,
        "tensor c 1,8,32,32 act", "tensor s 1,8,32,32 act", "tensor a 1,8,32,32 act",
        "tensor m 1,8,32,32 act", "tensor y 1,16,32,32 act",
        "node Conv conv in=x,w,b out=c kernel=3,3 stride=1,1 pad=1,1,1,1 act=1",
        "node SiLU silu in=c out=s",
        "node Add add in=s,c out=a",          # c 跨段再次被消费
        "node MaxPool pool in=a out=m kernel=3,3 stride=1,1 pad=1,1,1,1",
        "node Concat concat in=m,c out=y axis=1",
    ]
    (root / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "weights.bin").write_bytes(weights)
    x.tofile(root / "input.bin")


def run(argv, expect=0, cwd=None):
    result = subprocess.run([str(a) for a in argv], capture_output=True, encoding="utf-8", timeout=300, cwd=cwd)
    if expect is not None:
        assert result.returncode == expect, f"{argv}\n{result.stdout}\n{result.stderr}"
    return result


def dump_of(d: Path):
    files = sorted(p for p in d.iterdir() if p.suffix == ".bin")
    return [np.fromfile(p, dtype="<f4").view(np.uint32) for p in files]


class SplitChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="yinfer-split-")
        cls.root = Path(cls.tmp.name)
        write_model(cls.root)
        run([BINARY, "run", cls.root, cls.root / "input.bin", "--backend", "scalar", "--dump-dir", cls.root / "ref"])
        cls.ref = dump_of(cls.root / "ref")
        assert len(cls.ref) == 2, "合成模型应有两个输出"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_chain(self, out: Path, backend="scalar"):
        manifest = out / "chain.txt"
        self.assertTrue(manifest.exists())
        text = manifest.read_text(encoding="utf-8")
        self.assertIn("output y", text)
        self.assertIn("output s", text)
        for part in sorted(out.glob("part*")):
            run([BINARY, "info", part])   # 每段都是合法模型
        run([BINARY, "chain", manifest, self.root / "input.bin", "--backend", backend, "--dump-dir", out / "dump"])
        got = dump_of(out / "dump")
        self.assertEqual(len(got), len(self.ref))
        for a, b in zip(self.ref, got):
            np.testing.assert_array_equal(a, b)

    def test_split_at_tensor_is_bitwise_identical(self):
        out = self.root / "at"
        result = run([sys.executable, SPLIT, "--model", self.root, "--out", out, "--at", "s"])
        self.assertIn("2 段", result.stdout)
        # s 是图输出也是边界；c 在 part0 产生、part1 的 Add 和 Concat 还要用 → 也是边界。
        self.assertIn("边界 → c", result.stdout)
        self.check_chain(out)

    def test_split_parts_is_bitwise_identical(self):
        out = self.root / "parts"
        run([sys.executable, SPLIT, "--model", self.root, "--out", out, "--parts", "3"])
        self.assertEqual(len(list(out.glob("part*"))), 3)
        self.check_chain(out)

    def test_split_copy_weights(self):
        out = self.root / "copy"
        run([sys.executable, SPLIT, "--model", self.root, "--out", out, "--parts", "2", "--copy-weights"])
        self.assertTrue((out / "weights.bin").exists())
        self.check_chain(out)

    def test_rejects_bad_usage(self):
        run([sys.executable, SPLIT, "--model", self.root, "--out", self.root / "bad1", "--at", "x"], expect=1)
        run([sys.executable, SPLIT, "--model", self.root, "--out", self.root / "bad2", "--at", "nope"], expect=1)
        run([sys.executable, SPLIT, "--model", self.root, "--out", self.root / "bad3", "--at", "y"], expect=1)  # 最后一个节点
        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "x").write_text("x")
        run([sys.executable, SPLIT, "--model", self.root, "--out", occupied, "--parts", "2"], expect=1)
        out = self.root / "ok"
        run([sys.executable, SPLIT, "--model", self.root, "--out", out, "--parts", "2"])
        # 后端个数既不是 1 也不等于段数 → 退出码 3
        run([BINARY, "chain", out / "chain.txt", self.root / "input.bin", "--backend", "scalar,scalar,scalar"], expect=3)
        run([BINARY, "chain", self.root / "missing.txt", self.root / "input.bin"], expect=3)


if __name__ == "__main__":
    unittest.main()
