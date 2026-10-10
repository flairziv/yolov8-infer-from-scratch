#!/usr/bin/env python3
"""把编译好的模型（model.txt + weights.bin）切成多段，供 yinfer chain 串联执行。

部署里常见的切法：主干放加速器，颈部/头放 CPU；或者一个大模型切成几段流水。
这里只做"切"：每段都是一个完整合法的 model.txt（自己的 input/output/tensor/node 行），
共享同一份 weights.bin（常量偏移不变，不复制权重）。段之间的边界张量由工具自动推出并写进 chain.txt。

用法：
    python3 frontend/split_model.py --model artifacts/model --out artifacts/model-parts \
        --at /model.9/cv2/act/Mul_output_0            # 在产生该张量的节点之后切一刀
    python3 frontend/split_model.py --model artifacts/model --out artifacts/model-parts3 --parts 3

切分不改任何节点、不改权重、不改执行顺序：每段内部按原拓扑序执行，所以串联结果应与整体模型逐位一致。
"""
import argparse
import os
from pathlib import Path
import shutil
import sys


def parse_model(model_dir: Path):
    """按行保留原文：切分时节点行与张量声明原样写回，避免重新序列化属性带来的格式差异。"""
    lines = (model_dir / "model.txt").read_text(encoding="utf-8").splitlines()
    fmt, weights = None, None
    inputs, outputs = [], []
    tensors = {}          # 名 → (原始行, 是否常量)
    nodes = []            # (原始行, op, name, ins, outs)
    for raw in lines:
        parts = raw.split()
        if not parts or parts[0].startswith("#"):
            continue
        key = parts[0]
        if key == "format":
            fmt = raw
        elif key == "weights":
            weights = (parts[1], int(parts[2]))
        elif key == "input":
            inputs.append(parts[1])
        elif key == "output":
            outputs.append(parts[1])
        elif key == "tensor":
            tensors[parts[1]] = (raw, parts[3] == "const")
        elif key == "node":
            ins, outs = [], []
            for tok in parts[3:]:
                k, _, v = tok.partition("=")
                if k == "in":
                    ins = v.split(",")
                elif k == "out":
                    outs = v.split(",")
            nodes.append((raw, parts[1], parts[2], ins, outs))
        else:
            raise SystemExit(f"不认识的行: {raw}")
    if fmt is None or weights is None:
        raise SystemExit("model.txt 缺少 format 或 weights 行")
    return fmt, weights, inputs, outputs, tensors, nodes


def cut_points(nodes, at_names, parts):
    """返回每段最后一个节点的下标（升序）。--at 在产生该张量的节点后切；--parts 按节点数均分。"""
    last = len(nodes) - 1
    if parts:
        if parts < 2 or parts > len(nodes):
            raise SystemExit(f"--parts 需要在 2..{len(nodes)} 内")
        ends = [((i + 1) * len(nodes)) // parts - 1 for i in range(parts)]
    else:
        producer = {}
        for idx, (_, _, _, _, outs) in enumerate(nodes):
            for name in outs:
                producer[name] = idx
        ends = []
        for name in at_names:
            if name not in producer:
                raise SystemExit(f"--at {name}: 没有节点产生这个张量（图输入不能作为切点）")
            ends.append(producer[name])
        ends = sorted(set(ends))
        if ends and ends[-1] == last:
            raise SystemExit(f"--at {at_names[-1]}: 切在最后一个节点之后，不会产生新的段")
        ends.append(last)
    if len(set(ends)) != len(ends):
        raise SystemExit("切点重复")
    return ends


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="artifacts/model", help="编译好的模型目录")
    ap.add_argument("--out", required=True, help="输出目录；里面生成 part0/ part1/ ... 与 chain.txt")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--at", action="append", default=None, help="在产生该张量的节点之后切一刀，可重复")
    group.add_argument("--parts", type=int, default=None, help="按节点数均分成 N 段")
    ap.add_argument("--copy-weights", action="store_true",
                    help="把 weights.bin 复制到输出目录（默认各段用相对路径引用原文件，不复制）")
    args = ap.parse_args()

    model_dir = Path(args.model).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"输出目录 {out} 非空，换一个目录，避免覆盖旧产物")
    out.mkdir(parents=True, exist_ok=True)
    fmt, (weights_file, weights_bytes), graph_inputs, graph_outputs, tensors, nodes = parse_model(model_dir)

    ends = cut_points(nodes, args.at or [], args.parts)
    starts = [0] + [e + 1 for e in ends[:-1]]
    ranges = list(zip(starts, ends))

    # 每个激活的生产段、消费段
    produced_in, consumed_in = {}, {}
    for p, (lo, hi) in enumerate(ranges):
        for _, _, _, ins, outs in nodes[lo:hi + 1]:
            for name in outs:
                produced_in[name] = p
            for name in ins:
                if not tensors[name][1]:
                    consumed_in.setdefault(name, set()).add(p)

    if args.copy_weights:
        shutil.copy2(model_dir / weights_file, out / weights_file)
        weights_ref = f"../{weights_file}"
    else:
        # 加载器把 weights 行拼在模型目录后面，所以这里必须是相对各段目录的路径（各段目录同级，相对路径相同）。
        weights_ref = os.path.relpath(model_dir / weights_file, out / "part0").replace("\\", "/")
    graph_output_set = set(graph_outputs)

    manifest = [f"# 由 split_model.py 从 {model_dir} 切出，{len(ranges)} 段", f"model {model_dir}"]
    summary = []
    for p, (lo, hi) in enumerate(ranges):
        part_nodes = nodes[lo:hi + 1]
        produced = {n for _, _, _, _, outs in part_nodes for n in outs}
        consumed = {n for _, _, _, ins, _ in part_nodes for n in ins}
        part_inputs = [n for n in sorted(consumed - produced, key=lambda n: (n not in graph_inputs, n))
                       if not tensors[n][1]]
        # 输出：被后面的段消费的，或者本身就是图输出；顺序按原图输出顺序优先
        part_outputs = [n for n in graph_outputs if n in produced]
        for n in sorted(produced):
            if n in part_outputs:
                continue
            if any(q > p for q in consumed_in.get(n, ())):
                part_outputs.append(n)
        for n in part_inputs:
            if n in graph_inputs:
                continue
            if produced_in.get(n, p) >= p:
                raise SystemExit(f"段 {p} 的输入 {n} 不是由更早的段产生的，切分不合法")
        used = sorted(consumed | produced, key=lambda n: list(tensors).index(n))
        part_dir = out / f"part{p}"
        part_dir.mkdir()
        lines = [fmt, f"weights {weights_ref} {weights_bytes}"]
        lines += [f"input {n}" for n in part_inputs]
        lines += [f"output {n}" for n in part_outputs]
        lines += [tensors[n][0] for n in used]
        lines += [raw for raw, *_ in part_nodes]
        (part_dir / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append(f"part part{p} nodes={lo}..{hi} inputs={','.join(part_inputs)} outputs={','.join(part_outputs)}")
        boundary = [n for n in part_outputs if n not in graph_output_set]
        summary.append((p, hi - lo + 1, part_inputs, part_outputs, boundary))
    manifest += [f"output {n}" for n in graph_outputs]
    (out / "chain.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"[split] {model_dir} → {out}：{len(ranges)} 段，共 {len(nodes)} 个节点")
    for p, count, ins, outs, boundary in summary:
        print(f"  part{p}: {count} 个节点，输入 {len(ins)} 个，输出 {len(outs)} 个（其中跨段边界 {len(boundary)} 个）")
        for n in boundary:
            shape = tensors[n][0].split()[2]
            print(f"    边界 → {n} [{shape}]")
    print(f"  清单: {out / 'chain.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
