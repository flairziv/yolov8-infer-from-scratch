"""编译：yolov8n_bn.onnx → 图优化 → 切出运行时要跑的子图 → artifacts/model/{model.txt, weights.bin}

流程（每一步都是 passes.py 里的一个 pass）：
  BN 折叠 → 常量折叠（和形状推导交替，直到不动点）→ Slice 组合成 Split → SiLU 融合 → 按运行时支持的算子切图 → 降级成运行时算子并写文件

另存 artifacts/yolov8n_opt.onnx：做完 BN / 常量折叠和 Slice→Split、还没做 SiLU 融合的图。
它全是标准 ONNX 算子，张量名、数值都和运行时子图一致；make_reference.py 用 ORT 跑它拿逐层参考数据。
"""
import argparse
import collections
import os
import re
import warnings

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from passes import fold_bn, fold_constants, fuse_silu, fuse_slices_to_split, partition_graph

warnings.filterwarnings("ignore")

FORMAT = "yolov8-infer 1"
ALIGN = 64      # weights.bin 里每个常量的起点按 64 字节对齐：一条缓存行，也满足 SSE/AVX/AVX-512 的对齐读
# 运行时实现的 ONNX 算子（SiLU 是融合后的名字）。不在这里的算子都留给后处理
SUPPORTED = {"Conv", "SiLU", "MaxPool", "Resize", "Concat", "Split", "Add"}


def optimize(src, opt_path):
    """跑 pass 链，返回融合完 SiLU 的 gs 图；途中把标准算子版的等价图存到 opt_path。"""
    m = onnx.load(src)
    n_src = len(m.graph.node)
    g = gs.import_onnx(m)
    n_bn = fold_bn(g)["folded"]
    m = gs.export_onnx(g)
    del m.graph.value_info[:]                   # 旧的形状标注里还有 BN 删掉的中间张量，清空让形状推导重来

    rounds = 0
    while True:                                 # 常量折叠和形状推导互相解锁：推形状 → 折叠 → 再推，直到一轮什么都没折
        m = onnx.shape_inference.infer_shapes(m)
        g = gs.import_onnx(m)
        rounds += 1
        replaced = fold_constants(g)["replaced"]
        m = gs.export_onnx(g)
        if replaced == 0:
            break
    g = gs.import_onnx(onnx.shape_inference.infer_shapes(m))
    n_fold = len(g.nodes)
    n_split = fuse_slices_to_split(g)["fused"]

    m_opt = onnx.shape_inference.infer_shapes(gs.export_onnx(g))
    onnx.checker.check_model(m_opt)
    onnx.save(m_opt, opt_path)

    silu = fuse_silu(g, "SiLU")
    print(f"[compile] 图优化 {src}")
    print(f"  原图 {n_src} 节点 → BN 折叠 {n_bn} 个 → 常量折叠 {rounds} 轮后 {n_fold} 节点"
          f" → Slice 合成 Split {n_split} 个 → SiLU 融合 {silu['fused']} 个（跳过 {len(silu['skipped'])} 个）→ {len(g.nodes)} 节点")
    print(f"  标准算子版等价图 → {opt_path}")
    return g


def head_order(t):
    """运行时子图的 6 个输出排成 [P3 框, P3 类, P4 框, P4 类, P5 框, P5 类]。
    cv2.i 是框分支（64 = 4 条边 × 16 个距离桶，给 DFL 用），cv3.i 是类别分支（80 类），i = 0/1/2 对应步长 8/16/32。"""
    mm = re.search(r"/model\.22/cv([23])\.(\d)/", t.name)
    assert mm, f"意料之外的切割张量 {t.name}"
    return int(mm.group(2)), int(mm.group(1))


def ints(v):
    return [int(x) for x in v]


def lower(node):
    """ONNX 节点 → (运行时算子名, 属性, 数据输入)。

    只接受运行时实现了的那几种写法，其余在编译期直接报错：不支持的情况在这里暴露，而不是到运行时悄悄算错。
    参数型的常量输入（Split 的切分长度、Resize 的缩放倍数）在这里变成属性，运行时只看到真正参与计算的张量。
    """
    op, a, ins = node.op, node.attrs, node.inputs

    def need(cond, why):
        if not cond:
            raise NotImplementedError(f"{node.name}（{op}）: {why}")

    if op == "Conv":
        need(len(ins) == 3 and all(isinstance(t, gs.Constant) for t in ins[1:]), "权重和偏置必须都是常量")
        need(a.get("group", 1) == 1, "只支持 group=1")
        need(ints(a.get("dilations", [1, 1])) == [1, 1], "只支持 dilation=1")
        need(a.get("auto_pad", "NOTSET") == "NOTSET", "不支持 auto_pad")
        k = ints(ins[1].values.shape[2:])
        need(ints(a.get("kernel_shape", k)) == k, "kernel_shape 和权重形状不一致")
        # pad 的顺序和 ONNX 一样：[上, 左, 下, 右]
        return "Conv", {"kernel": k, "stride": ints(a.get("strides", [1, 1])), "pad": ints(a.get("pads", [0, 0, 0, 0]))}, list(ins)
    if op == "SiLU":
        return "SiLU", {}, [ins[0]]
    if op == "MaxPool":
        need(a.get("ceil_mode", 0) == 0, "只支持 ceil_mode=0")
        need(ints(a.get("dilations", [1, 1])) == [1, 1], "只支持 dilation=1")
        need(a.get("storage_order", 0) == 0 and len(node.outputs) == 1, "不支持 Indices 输出")
        need(a.get("auto_pad", "NOTSET") == "NOTSET", "不支持 auto_pad")
        return "MaxPool", {"kernel": ints(a["kernel_shape"]), "stride": ints(a.get("strides", [1, 1])),
                           "pad": ints(a.get("pads", [0, 0, 0, 0]))}, [ins[0]]
    if op == "Resize":
        need(a.get("mode") == "nearest" and a.get("coordinate_transformation_mode") == "asymmetric"
             and a.get("nearest_mode") == "floor",
             "只支持 nearest + asymmetric + floor，也就是 PyTorch 的 nn.Upsample(mode='nearest')")
        scales = ins[2] if len(ins) > 2 else None
        need(isinstance(scales, gs.Constant) and scales.values.size == 4, "缩放倍数必须是 4 个元素的常量 scales（不支持 sizes）")
        s = scales.values.astype(np.float64).tolist()
        need(s[:2] == [1.0, 1.0] and all(v.is_integer() and v >= 1 for v in s[2:]), f"只支持 H、W 整数倍放大，实际 {s}")
        # 这三个条件合起来，输出 (y, x) 就等于输入 (y / s, x / s) 整除 —— 运行时不用再管坐标变换
        return "UpsampleNearest", {"scale": [int(s[2]), int(s[3])]}, [ins[0]]
    if op == "Concat":
        return "Concat", {"axis": int(a["axis"])}, list(ins)
    if op == "Split":
        need(len(ins) == 2 and isinstance(ins[1], gs.Constant), "切分长度必须是常量输入")
        return "Split", {"axis": int(a.get("axis", 0)), "sizes": ints(ins[1].values)}, [ins[0]]
    if op == "Add":
        need(len(ins) == 2 and list(ins[0].shape) == list(ins[1].shape), "只支持同形状相加（不做广播）")
        return "Add", {}, list(ins)
    raise NotImplementedError(f"{node.name}: 运行时没有 {op}")


def check_name(s):
    # model.txt 按空白分词、按逗号分列表、按第一个等号分键值，名字里不能有这些字符
    assert s and not re.search(r"[\s,=]", s), f"名字里不能有空白、逗号、等号: {s!r}"
    return s


def write_model(nodes, graph_in, graph_out, out_dir):
    """把子图写成 model.txt（文本：张量表 + 拓扑序节点表）和 weights.bin（所有常量拼在一起）。"""
    os.makedirs(out_dir, exist_ok=True)
    blob = bytearray()
    decl = {}                                   # 张量名 → 声明行，按第一次出现的顺序
    act_bytes = 0

    def declare(t):
        nonlocal act_bytes
        if t.name in decl:
            return
        shape = list(t.values.shape) if isinstance(t, gs.Constant) else list(t.shape)
        assert all(isinstance(d, int) and d > 0 for d in shape), f"{t.name} 形状不是全静态: {shape}"
        s = ",".join(map(str, shape))
        if isinstance(t, gs.Constant):
            v = np.ascontiguousarray(t.values, dtype="<f4")      # 小端 float32；x86 和 ARM 都是小端，可以直接按内存读
            blob.extend(b"\0" * (-len(blob) % ALIGN))
            decl[t.name] = f"tensor {check_name(t.name)} {s} const {len(blob)}"
            blob.extend(v.tobytes())
        else:
            assert t.dtype == np.float32, f"{t.name} 不是 float32"
            decl[t.name] = f"tensor {check_name(t.name)} {s} act"
            act_bytes += int(np.prod(shape)) * 4

    for t in graph_in:
        declare(t)
    node_lines, ops, gflop = [], collections.Counter(), 0.0
    for n in nodes:
        op, attrs, data_in = lower(n)
        for t in data_in + list(n.outputs):
            declare(t)
        fields = [f"node {op} {check_name(n.name)}",
                  "in=" + ",".join(t.name for t in data_in),
                  "out=" + ",".join(t.name for t in n.outputs)]
        fields += [f"{k}=" + ",".join(map(str, v if isinstance(v, list) else [v])) for k, v in attrs.items()]
        node_lines.append(" ".join(fields))
        ops[op] += 1
        if op == "Conv":
            y = n.outputs[0].shape
            gflop += 2 * n.inputs[1].values.size * y[2] * y[3] / 1e9    # 每个输出点 Cin·kh·kw 次乘加

    header = ["# 由 frontend/compile_model.py 生成，不要手改。格式说明见 README「模型文件格式」",
              f"format {FORMAT}",
              f"weights weights.bin {len(blob)}"]
    header += [f"input {t.name}" for t in graph_in] + [f"output {t.name}" for t in graph_out]
    with open(os.path.join(out_dir, "model.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header + list(decl.values()) + node_lines) + "\n")
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(blob)
    return ops, len(blob), act_bytes, gflop, len(decl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="artifacts/yolov8n_bn.onnx", help="export_onnx.py 导出的、还带 BN 的 ONNX")
    ap.add_argument("--opt", default="artifacts/yolov8n_opt.onnx", help="另存的标准算子版等价图（生成参考数据用）")
    ap.add_argument("--out", default="artifacts/model")
    args = ap.parse_args()

    g = optimize(args.onnx, args.opt)
    r = partition_graph(g, SUPPORTED, single_cut=True)
    outs = sorted(r["cut"], key=head_order)
    print(f"  切图：运行时 {len(r['dev'])} 节点，后处理 {len(r['host'])} 节点"
          f"（{dict(collections.Counter(n.op for n in r['host']))}），交界 {len(outs)} 个张量 {r['cut_bytes'] / 2**20:.2f} MiB")

    ops, w_bytes, act_bytes, gflop, n_tensors = write_model(r["dev"], g.inputs, outs, args.out)
    print(f"[compile] 写出 {args.out}/model.txt + weights.bin")
    print(f"  运行时算子 {sum(ops.values())} 个：{dict(ops.most_common())}")
    print(f"  张量 {n_tensors} 个；权重 {w_bytes / 2**20:.2f} MiB；激活不复用共 {act_bytes / 2**20:.1f} MiB；卷积 {gflop:.2f} GFLOP")
    for t in outs:
        print(f"  输出 {t.name:45s} {t.shape}")


if __name__ == "__main__":
    main()
