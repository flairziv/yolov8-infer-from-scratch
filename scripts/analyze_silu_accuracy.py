#!/usr/bin/env python3
"""在固定 [-18,18] 区间测 SiLU 相对 float64 数学参考的绝对误差与 ULP。

复用小模型测试的导出/运行入口；不是分类或检测精度评估，不能用来推断 mAP。
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from onnx import helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_ref_ops


def ordered_bits(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    ordered = np.where(bits & np.uint32(0x80000000), ~bits, bits ^ np.uint32(0x80000000))
    return ordered.astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/stage2-silu-accuracy.json")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        ap.error("输出已存在，请指定新路径")
    x = np.linspace(-18, 18, 20001, dtype=np.float32).reshape(1, 1, 1, -1)
    v = x.astype(np.float64)
    e = np.exp(-np.abs(v))
    gold = v * np.where(v >= 0, 1 / (1 + e), e / (1 + e))
    rounded = gold.astype(np.float32)
    y = test_ref_ops.reference([helper.make_node("Sigmoid", ["x"], ["s"]),
                                helper.make_node("Mul", ["x", "s"], ["y"])], {"x": x}, {}, list(x.shape))
    query = subprocess.run([str(test_ref_ops.BINARY), "backends"], check=True, capture_output=True, encoding="utf-8")
    available = dict(line.split() for line in query.stdout.splitlines() if len(line.split()) == 2)
    report = {"range": [-18, 18], "samples": int(x.size), "reference": "float64 stable SiLU, rounded to float32 for ULP",
              "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(), "backends": {}}
    for backend in ("scalar", "sse", "avx2"):
        if available.get(backend) != "available":
            report["backends"][backend] = {"skipped": "unavailable"}
            continue
        case = test_ref_ops.RefOpsTest()
        case.backend = backend
        actual = case.run_kernel("SiLU", {"x": x}, {}, {}, y, return_output=True)
        np.testing.assert_allclose(actual, gold, rtol=3e-6, atol=3e-7)
        ulp = np.abs(ordered_bits(actual) - ordered_bits(rounded))
        stats = {"max_abs_to_float64": float(np.abs(actual.astype(np.float64) - gold).max()),
                 "max_ulp_to_rounded_float64": int(ulp.max()), "p99_ulp": float(np.percentile(ulp, 99)),
                 "mean_ulp": float(ulp.mean()), "exact_to_rounded_float64": int(np.count_nonzero(ulp == 0))}
        report["backends"][backend] = stats
        print(backend, stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("report:", out)


if __name__ == "__main__":
    main()
