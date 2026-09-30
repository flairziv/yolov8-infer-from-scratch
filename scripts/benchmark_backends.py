#!/usr/bin/env python3
"""同机同输入的 CPU 后端交错对照；只记录 C++ run 内部计时，不含进程启动与模型加载。

默认每组两块：ABBA、BAAB，每轮都是 fresh process。先验证 REF=0，再预热、采样。
这比较完整后端（算法、im2col、SIMD一起变化），不能把结果归因于 SIMD 指令宽度。
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default="build/yinfer")
    ap.add_argument("--model", default="artifacts/model")
    ap.add_argument("--input", default="artifacts/ref/input.bin")
    ap.add_argument("--ref", default="artifacts/ref")
    ap.add_argument("--candidates", nargs="+", choices=["sse", "avx2"], default=["sse", "avx2"])
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--out", default="artifacts/stage2-benchmark.json")
    args = ap.parse_args()
    if args.blocks < 2 or args.blocks % 2:
        ap.error("--blocks 必须是至少2的偶数，交替使用ABBA/BAAB")
    out = Path(args.out)
    if out.exists():
        ap.error("报告已存在，请用 --out 指定新文件，避免覆盖旧证据")
    binary, model, ref, image = map(lambda p: Path(p).resolve(), (args.binary, args.model, args.ref, args.input))
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "passed": False, "platform": platform.platform(), "timing": "C++ run internal elapsed milliseconds",
              "scope": "single-machine CPU backend comparison, not pure SIMD width or detector accuracy",
              "ordering": "alternating ABBA/BAAB blocks, fresh process per sample, one warm-up per pair/backend",
              "limitations": ["host scheduling and frequency are not controlled", "first run inside each fresh process",
                              "load averages are observations, not a resource-isolation gate"],
              "verification": {}, "pairs": []}
    try:
        if sha(image) != sha(ref / "input.bin"):
            raise ValueError("计时输入与参考输入不是同一份文件")
        report["sha256"] = {"binary": sha(binary), "model.txt": sha(model / "model.txt"),
                            "input.bin": sha(image), "ref.bin": sha(ref / "ref.bin"), "ref.txt": sha(ref / "ref.txt")}
        weights = next(line.split()[1] for line in (model / "model.txt").read_text(encoding="utf-8").splitlines()
                       if line.startswith("weights "))
        report["sha256"]["weights"] = sha(model / weights)
        root = Path(__file__).resolve().parents[1]
        report["source_sha256"] = {str(p.relative_to(root)): sha(p) for sub in ("runtime", "tools")
                                   for p in sorted((root / sub).rglob("*")) if p.suffix in (".cpp", ".h")}
        report["source_sha256"]["CMakeLists.txt"] = sha(root / "CMakeLists.txt")
        if Path("/proc/cpuinfo").exists():
            report["cpu_model"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                         if line.startswith("model name")), "unknown")
        if Path("/proc/loadavg").exists():
            report["loadavg_before"] = Path("/proc/loadavg").read_text().strip()
        states = dict(line.split() for line in command([binary, "backends"]).splitlines() if len(line.split()) == 2)
        for backend in ["scalar", *args.candidates]:
            if states.get(backend) != "available":
                raise ValueError(f"后端不可用: {backend}")
            text = command([binary, "verify", model, ref, "--backend", backend, "--mode", "chained", "--brief"])
            counts = re.search(r"EXACT (\d+) OK (\d+) FAIL (\d+) REF (\d+)", text)
            if not counts or int(counts[3]) or int(counts[4]) or int(counts[1]) + int(counts[2]) == 0:
                raise ValueError(f"{backend} 验证未全部实际执行并通过:\n{text}")
            report["verification"][backend] = text

        def sample(backend):
            text = command([binary, "run", model, image, "--backend", backend])
            match = re.search(r"整图一次\s+([0-9.]+)\s+ms", text)
            if not match or float(match[1]) <= 0:
                raise ValueError(f"无法提取 C++ 计时:\n{text}")
            return {"backend": backend, "milliseconds": float(match[1]), "stdout": text}

        for candidate in args.candidates:
            sample("scalar")
            sample(candidate)
            pair = {"baseline": "scalar", "candidate": candidate, "samples": []}
            report["pairs"].append(pair)
            for block in range(args.blocks):
                order = ["scalar", candidate, candidate, "scalar"] if block % 2 == 0 else [candidate, "scalar", "scalar", candidate]
                for slot, backend in enumerate(order):
                    record = sample(backend)
                    record.update(block=block, slot=slot)
                    pair["samples"].append(record)
                    print(f"{candidate} block={block} slot={slot} {backend}: {record['milliseconds']:.3f} ms", flush=True)
            for backend in ("scalar", candidate):
                values = [s["milliseconds"] for s in pair["samples"] if s["backend"] == backend]
                pair[backend] = {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values), "count": len(values)}
            pair["ratio_of_medians"] = pair["scalar"]["median_ms"] / pair[candidate]["median_ms"]
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
        print(f"scalar / {pair['candidate']} median ratio: {pair['ratio_of_medians']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
