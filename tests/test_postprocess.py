"""检测后处理集成测试：合成一个检测头模型，用独立实现的 numpy 参考（DFL+sigmoid+NMS）对照 yinfer detect。

不依赖 ORT、不依赖 artifacts：模型权重和输入都在测试里现造。
参考实现刻意用另一套写法（矩阵运算、逐尺度循环），避免和 C++ 实现共享同一个错误。
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

SIDE = 64
CLASSES = 80
REG_MAX = 16
SCALES = (8, 4, 2)   # 特征图边长：步长 8/16/32


def conv1x1(x, w, b, stride):
    """1×1 卷积（stride 采样）：x (3,64,64) → (C, 64/stride, 64/stride)。"""
    y = np.tensordot(w[:, :, 0, 0], x, axes=([1], [0])) + b[:, None, None]
    return y[:, ::stride, ::stride]


def write_model(root: Path, rng):
    x = rng.normal(size=(3, SIDE, SIDE)).astype("<f4")
    weights = bytearray()
    decls, convs = [], {}
    for name, channels in (("box", 4 * REG_MAX), ("cls", CLASSES)):
        for f in SCALES:
            w = (rng.normal(size=(channels, 3, 1, 1)) * 0.5).astype("<f4")
            b = (rng.normal(size=(channels,)) * 0.5).astype("<f4")
            for suffix, value in ((".w", w), (".b", b)):
                weights.extend(b"\0" * (-len(weights) % 64))
                tag = f"{name}{f}{suffix}"
                decls.append(f"tensor {tag} {','.join(map(str, value.shape))} const {len(weights)}")
                weights.extend(value.tobytes())
            convs[(name, f)] = (w, b)
    lines = ["format yolov8-infer 2", f"weights weights.bin {len(weights)}", "input x"]
    for f in SCALES:
        lines += [f"output box{f}", f"output cls{f}"]
    lines += ["tensor x 1,3,%d,%d act" % (SIDE, SIDE), *decls]
    for f in SCALES:
        lines += [f"tensor box{f} 1,{4 * REG_MAX},{f},{f} act", f"tensor cls{f} 1,{CLASSES},{f},{f} act"]
    for f in SCALES:
        stride = SIDE // f
        for name in ("box", "cls"):
            lines.append(f"node Conv {name}{f} in=x,{name}{f}.w,{name}{f}.b out={name}{f} "
                         f"kernel=1,1 stride={stride},{stride} pad=0,0,0,0")
    (root / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "weights.bin").write_bytes(weights)
    x.astype("<f4").tofile(root / "input.bin")
    return x, convs


def reference(x, convs, conf, iou):
    """独立的 numpy 后处理：DFL 期望 + sigmoid + 置信度过滤 + 类内贪心 NMS。"""
    boxes = []
    for f in SCALES:
        stride = SIDE // f
        bw, bb = convs[("box", f)]
        cw, cb = convs[("cls", f)]
        box_logits = conv1x1(x, bw, bb, stride).reshape(4, REG_MAX, f * f)
        cls_logits = conv1x1(x, cw, cb, stride).reshape(CLASSES, f * f)
        shifted = box_logits - box_logits.max(axis=1, keepdims=True)
        prob = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        dist = (prob * np.arange(REG_MAX)[None, :, None]).sum(axis=1)          # (4, f*f)
        score = 1.0 / (1.0 + np.exp(-cls_logits))
        best = score.argmax(axis=0)
        top = score.max(axis=0)
        for pos in range(f * f):
            if top[pos] < conf:
                continue
            h, w = divmod(pos, f)
            cx, cy = (w + 0.5) * stride, (h + 0.5) * stride
            left, up, right, down = (float(v) * stride for v in dist[:, pos])
            boxes.append({"cls": int(best[pos]), "score": float(top[pos]),
                          "box": [cx - left, cy - up, cx + right, cy + down]})
    boxes.sort(key=lambda d: -d["score"])

    def iou_of(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    kept, dropped = [], set()
    for i, box in enumerate(boxes):
        if i in dropped:
            continue
        kept.append(box)
        for j in range(i + 1, len(boxes)):
            if j in dropped or boxes[j]["cls"] != box["cls"]:
                continue
            if iou_of(box["box"], boxes[j]["box"]) > iou:
                dropped.add(j)
    return kept


def run_detect(model, input_path, out_txt, *args, expect=0):
    result = subprocess.run([str(BINARY), "detect", str(model), str(input_path), "--out", str(out_txt), *args],
                            capture_output=True, encoding="utf-8", timeout=120)
    if expect is not None:
        assert result.returncode == expect, result.stdout + result.stderr
    boxes = []
    if out_txt.exists():
        for line in out_txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls, score, x1, y1, x2, y2 = line.split()
                boxes.append({"cls": int(cls), "score": float(score),
                              "box": [float(x1), float(y1), float(x2), float(y2)]})
    return boxes, result.stdout


class DetectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="yinfer-detect-")
        cls.root = Path(cls.tmp.name)
        cls.x, cls.convs = write_model(cls.root, np.random.default_rng(20261008))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_matches_reference(self):
        for conf, iou in ((0.25, 0.7), (0.5, 0.5), (0.9, 0.7)):
            with self.subTest(conf=conf, iou=iou):
                boxes, _ = run_detect(self.root, self.root / "input.bin", self.root / "boxes.txt",
                                      "--conf", str(conf), "--iou", str(iou), "--backend", "avx2", "--threads", "4")
                expected = reference(self.x, self.convs, conf, iou)
                self.assertEqual(len(boxes), len(expected),
                                 f"框数量不一致: 引擎 {len(boxes)}，参考 {len(expected)}")
                for mine in boxes:
                    best = None
                    for ref in expected:
                        if ref["cls"] != mine["cls"]:
                            continue
                        inter = max(0.0, min(mine["box"][2], ref["box"][2]) - max(mine["box"][0], ref["box"][0])) * \
                                max(0.0, min(mine["box"][3], ref["box"][3]) - max(mine["box"][1], ref["box"][1]))
                        union = (mine["box"][2] - mine["box"][0]) * (mine["box"][3] - mine["box"][1]) + \
                                (ref["box"][2] - ref["box"][0]) * (ref["box"][3] - ref["box"][1]) - inter
                        value = inter / union if union > 0 else 0.0
                        if best is None or value > best:
                            best = value
                    self.assertIsNotNone(best, f"没有同类的参考框可配对: {mine}")
                    self.assertGreater(best, 0.999, f"框对不上: {mine}")
                for mine, ref in zip(sorted(boxes, key=lambda d: -d["score"]),
                                     sorted(expected, key=lambda d: -d["score"])):
                    self.assertAlmostEqual(mine["score"], ref["score"], delta=2e-4)

    def test_letterbox_unmap(self):
        scale, pad_x, pad_y = 0.5, 100.0, 150.0
        raw, _ = run_detect(self.root, self.root / "input.bin", self.root / "raw.txt",
                            "--conf", "0.25", "--iou", "0.7")
        moved, _ = run_detect(self.root, self.root / "input.bin", self.root / "moved.txt",
                              "--conf", "0.25", "--iou", "0.7", "--letterbox", f"{scale},{pad_x},{pad_y}")
        self.assertEqual(len(raw), len(moved))
        for a, b in zip(raw, moved):
            for i in range(4):
                expected = (a["box"][i] - (pad_x if i % 2 == 0 else pad_y)) / scale
                self.assertAlmostEqual(b["box"][i], expected, places=2)

    def test_rejects_bad_arguments(self):
        # 非法取值：参数解析抛异常 → 退出码 3。
        for args in (("--conf", "2"), ("--conf", "-1"), ("--iou", "1.5"), ("--threads", "0"),
                     ("--letterbox", "0.5,0"), ("--letterbox", "0.5,x,0")):
            with self.subTest(args=args):
                run_detect(self.root, self.root / "input.bin", self.root / "bad.txt", *args, expect=3)
        # 不认识的参数：打印用法 → 退出码 1（与其它子命令一致）。
        run_detect(self.root, self.root / "input.bin", self.root / "bad.txt", "--unknown", "1", expect=1)

    def test_stdout_reports_counts(self):
        _, stdout = run_detect(self.root, self.root / "input.bin", self.root / "counts.txt",
                               "--conf", "0.5", "--iou", "0.7")
        match = re.search(r"检测到 (\d+) 个框", stdout)
        self.assertIsNotNone(match, stdout)
        self.assertGreater(int(match.group(1)), 0, "合成模型应该至少检出一个框")


if __name__ == "__main__":
    unittest.main()
