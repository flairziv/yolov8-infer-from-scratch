#!/usr/bin/env python3
"""比较两个 --dump-dir 目录：逐文件报告是否逐位相同、最大绝对误差与相对误差（max|d| / max|ref|）。

用法：python3 scripts/compare_dumps.py <参考目录> <待比目录> [--tol 1e-4]
退出码：0 全部在容差内；1 有超容差或非有限值；2 文件/形状不匹配。
"""
import argparse
from pathlib import Path
import sys

import numpy as np


def read_index(d: Path):
    entries = []
    for line in (d / "outputs.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            file, name, shape = line.split()
            entries.append((file, name, tuple(int(v) for v in shape.split(","))))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref")
    ap.add_argument("got")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    ref_dir, got_dir = Path(args.ref), Path(args.got)
    ref_idx, got_idx = read_index(ref_dir), read_index(got_dir)
    if [e[1:] for e in ref_idx] != [e[1:] for e in got_idx]:
        print("输出名字或形状不一致:", ref_idx, got_idx)
        return 2
    worst, all_exact, failed = 0.0, True, False
    for (file, name, shape), (gfile, _, _) in zip(ref_idx, got_idx):
        a = np.fromfile(ref_dir / file, dtype="<f4")
        b = np.fromfile(got_dir / gfile, dtype="<f4")
        if a.size != b.size or a.size != int(np.prod(shape)):
            print(f"{name}: 元素数不一致 {a.size} vs {b.size}")
            return 2
        exact = np.array_equal(a.view(np.uint32), b.view(np.uint32))
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if finite else float("inf")
        ref_max = float(np.max(np.abs(a))) if finite else 0.0
        rel = max_abs / ref_max if ref_max > 0 else max_abs
        status = "EXACT" if exact and finite else ("OK" if finite and rel <= args.tol else "FAIL")
        all_exact &= exact and finite
        failed |= status == "FAIL"
        worst = max(worst, rel)
        print(f"{status:5s} {name} max|d|={max_abs:.3e} rel={rel:.3e}")
    print(f"汇总: {'全部逐位相同' if all_exact else f'最大 rel {worst:.3e}（容差 {args.tol:g}）'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
