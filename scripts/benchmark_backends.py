#!/usr/bin/env python3
"""同机交错对照：默认比较三种后端；给 --alt-model 时改为比较两个模型的消融实验。

只记录 C++ run 内部计时，不含进程启动与模型加载。每组两块：ABBA、BAAB，每轮都是 fresh process。
先逐层验证（REF=0、无 FAIL），再预热、采样。后端对照比较的是完整实现差异（算法、im2col、SIMD
一起变化），不能把结果归因于 SIMD 指令宽度；模型消融用于隔离单个优化（例如收尾融合）。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(argv):
    result = subprocess.run(list(map(str, argv)), capture_output=True, encoding="utf-8", timeout=300)
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode}: {' '.join(map(str, argv))}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def variant(label, binary, model, ref, backend):
    binary, model, ref = Path(binary).resolve(), Path(model).resolve(), Path(ref).resolve()
    weights = next(line.split()[1] for line in (model / "model.txt").read_text(encoding="utf-8").splitlines()
                   if line.startswith("weights "))
    return {"label": label, "binary": binary, "model": model, "ref": ref, "backend": backend,
            "sha256": {"binary": sha(binary), "model.txt": sha(model / "model.txt"), "weights": sha(model / weights),
                       "ref.txt": sha(ref / "ref.txt"), "ref.bin": sha(ref / "ref.bin")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default="build/yinfer")
    ap.add_argument("--alt-binary", default=None, help="消融模式：对照可执行文件；缺省与 --binary 相同")
    ap.add_argument("--model", default="artifacts/model")
    ap.add_argument("--input", default="artifacts/ref/input.bin")
    ap.add_argument("--ref", default="artifacts/ref")
    ap.add_argument("--candidates", nargs="+", choices=["sse", "avx2"], default=["sse", "avx2"])
    ap.add_argument("--alt-model", default=None, help="消融模式：对照模型目录；与 --alt-ref 一起给出")
    ap.add_argument("--alt-ref", default=None)
    ap.add_argument("--alt-label", default="alt")
    ap.add_argument("--backend", choices=["scalar", "sse", "avx2"], default="avx2",
                    help="消融模式下两个变体共用的后端")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--out", default="artifacts/stage2-benchmark.json")
    args = ap.parse_args()
    if args.blocks < 2 or args.blocks % 2:
        ap.error("--blocks 必须是至少2的偶数，交替使用ABBA/BAAB")
    if bool(args.alt_model) != bool(args.alt_ref):
        ap.error("--alt-model 与 --alt-ref 必须同时给出")
    out = Path(args.out)
    if out.exists():
        ap.error("报告已存在，请用 --out 指定新文件，避免覆盖旧证据")
    image = Path(args.input).resolve()
    alt_binary = args.alt_binary or args.binary

    if args.alt_model:
        baseline = variant("baseline", args.binary, args.model, args.ref, args.backend)
        candidate = variant(args.alt_label, alt_binary, args.alt_model, args.alt_ref, args.backend)
        scope = f"single-machine ablation at backend={args.backend} (binary and/or model), not detector accuracy"
    else:
        baseline = variant("scalar", args.binary, args.model, args.ref, "scalar")
        scope = "single-machine CPU backend comparison, not pure SIMD width or detector accuracy"
    variants = [baseline] if not args.alt_model else [baseline, candidate]
    if not args.alt_model:
        variants += [variant(name, args.binary, args.model, args.ref, name) for name in args.candidates]

    report = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
              "passed": False, "platform": platform.platform(), "timing": "C++ run internal elapsed milliseconds",
              "scope": scope,
              "ordering": "alternating ABBA/BAAB blocks, fresh process per sample, one warm-up per pair/variant",
              "limitations": ["host scheduling and frequency are not controlled", "first run inside each fresh process",
                              "load averages are observations, not a resource-isolation gate"],
              "verification": {}, "variants": {v["label"]: {"binary": str(v["binary"]), "model": str(v["model"]),
                                                            "ref": str(v["ref"]), "backend": v["backend"],
                                                            "sha256": v["sha256"]}
                                               for v in variants},
              "pairs": []}
    try:
        if sha(image) != sha(baseline["ref"] / "input.bin"):
            raise ValueError("计时输入与基线参考输入不是同一份文件")
        if args.alt_model and sha(image) != sha(candidate["ref"] / "input.bin"):
            raise ValueError("两个变体的输入不是同一份文件，无法配对比较")
        report["sha256"] = {"input.bin": sha(image)}
        root = Path(__file__).resolve().parents[1]
        report["source_sha256"] = {str(p.relative_to(root)): sha(p) for sub in ("runtime", "tools")
                                   for p in sorted((root / sub).rglob("*")) if p.suffix in (".cpp", ".h")}
        report["source_sha256"]["CMakeLists.txt"] = sha(root / "CMakeLists.txt")
        if Path("/proc/cpuinfo").exists():
            report["cpu_model"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                         if line.startswith("model name")), "unknown")
        if Path("/proc/loadavg").exists():
            report["loadavg_before"] = Path("/proc/loadavg").read_text().strip()
        states = dict(line.split() for line in command([baseline["binary"], "backends"]).splitlines() if len(line.split()) == 2)
        for v in variants:
            if states.get(v["backend"]) != "available":
                raise ValueError(f"后端不可用: {v['backend']}（变体 {v['label']}）")
            text = command([v["binary"], "verify", v["model"], v["ref"], "--backend", v["backend"], "--mode", "chained", "--brief"])
            counts = re.search(r"EXACT (\d+) OK (\d+) FAIL (\d+) REF (\d+)", text)
            if not counts or int(counts[3]) or int(counts[4]) or int(counts[1]) + int(counts[2]) == 0:
                raise ValueError(f"{v['label']} 验证未全部实际执行并通过:\n{text}")
            report["verification"][v["label"]] = text

        def sample(v):
            text = command([v["binary"], "run", v["model"], image, "--backend", v["backend"]])
            match = re.search(r"整图一次\s+([0-9.]+)\s+ms", text)
            if not match or float(match[1]) <= 0:
                raise ValueError(f"无法提取 C++ 计时:\n{text}")
            return {"label": v["label"], "milliseconds": float(match[1]), "stdout": text}

        pairs = [(baseline, v) for v in variants[1:]]
        for base, cand in pairs:
            sample(base)
            sample(cand)
            pair = {"baseline": base["label"], "candidate": cand["label"], "samples": []}
            report["pairs"].append(pair)
            for block in range(args.blocks):
                order = [base, cand, cand, base] if block % 2 == 0 else [cand, base, base, cand]
                for slot, v in enumerate(order):
                    record = sample(v)
                    record.update(block=block, slot=slot)
                    pair["samples"].append(record)
                    print(f"{cand['label']} block={block} slot={slot} {v['label']}: {record['milliseconds']:.3f} ms", flush=True)
            for v in (base, cand):
                values = [s["milliseconds"] for s in pair["samples"] if s["label"] == v["label"]]
                pair[v["label"]] = {"median_ms": statistics.median(values), "min_ms": min(values),
                                    "max_ms": max(values), "count": len(values)}
            pair["ratio_of_medians"] = pair[base["label"]]["median_ms"] / pair[cand["label"]]["median_ms"]
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    if Path("/proc/loadavg").exists():
        report["loadavg_after"] = Path("/proc/loadavg").read_text().strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"report: {out} sha256={sha(out)}")
    if not report["passed"]:
        print(report.get("error", "验证失败"), file=sys.stderr)
        return 1
    for pair in report["pairs"]:
        print(f"{pair['baseline']} / {pair['candidate']} median ratio: {pair['ratio_of_medians']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
