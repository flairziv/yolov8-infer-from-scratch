# passes.py —— 图变换 pass（基于 onnx-graphsurgeon），来自编译器课 compiler/passes.py，只保留本项目用到的 5 个
#   每个 pass 的约定：输入 gs.Graph，原地修改，返回一个 dict 统计；找不到可改的模式就什么都不做。
#   graphsurgeon 的四个引用：
#     tensor.inputs  = 生产它的节点列表（0 或 1 个）      tensor.outputs = 消费它的节点列表
#     node.inputs    = 输入张量列表                        node.outputs   = 输出张量列表
#   改 node.inputs / node.outputs 时张量那边的反向引用自动同步。
#   所有 pass 共同的改图原则：新节点沿用旧节点的输出张量（同一个对象、同一个名字），下游完全不用改。
import numpy as np
import onnx_graphsurgeon as gs


def fold_bn(graph: gs.Graph) -> dict:
    """Conv → BatchNormalization 折叠成一个 Conv。

    数学：BN(x) = s·x + t，s = γ/√(σ²+ε)，t = β − s·μ（逐输出通道）
          Conv 线性 ⇒ BN(W⊛in + b) = (s·W)⊛in + (s·b + t)
    前提：BN 推理模式；Conv 输出只被这个 BN 消费；Conv 输出不是图输出。
    """
    stats = {"folded": 0, "conv_had_bias": 0, "skipped": []}
    for bn in [n for n in graph.nodes if n.op == "BatchNormalization"]:
        x = bn.inputs[0]
        if bn.attrs.get("training_mode", 0):
            stats["skipped"].append((bn.name, "training_mode=1")); continue
        if len(x.inputs) != 1 or x.inputs[0].op not in ("Conv", "ConvTranspose"):
            stats["skipped"].append((bn.name, "上游不是 Conv")); continue
        conv = x.inputs[0]
        if len(x.outputs) != 1:
            stats["skipped"].append((bn.name, f"Conv 输出有 {len(x.outputs)} 个消费者")); continue
        if x in graph.outputs:
            stats["skipped"].append((bn.name, "Conv 输出是图输出")); continue
        if not all(isinstance(t, gs.Constant) for t in bn.inputs[1:5]) or not isinstance(conv.inputs[1], gs.Constant):
            stats["skipped"].append((bn.name, "参数不是常量")); continue

        # 用 float64 算 s、t 和新权重，最后只转回 float32 一次，少一次舍入
        gamma, beta, mean, var = (t.values.astype(np.float64) for t in bn.inputs[1:5])
        eps = float(bn.attrs.get("epsilon", 1e-5))
        s = gamma / np.sqrt(var + eps)
        t = beta - s * mean

        W = conv.inputs[1]
        out_axis = 1 if conv.op == "ConvTranspose" else 0     # ConvTranspose 权重是 [Cin, Cout/g, kH, kW]
        shape = [1] * W.values.ndim; shape[out_axis] = -1
        W_new = (W.values.astype(np.float64) * s.reshape(shape)).astype(W.values.dtype)
        if len(conv.inputs) > 2 and isinstance(conv.inputs[2], gs.Constant):
            b_old = conv.inputs[2].values.astype(np.float64); stats["conv_had_bias"] += 1
        else:
            b_old = np.zeros_like(s)
        b_new = (s * b_old + t).astype(W.values.dtype)
        conv.inputs[1] = gs.Constant(W.name + "_bnfold", W_new)
        if len(conv.inputs) > 2:
            conv.inputs[2] = gs.Constant(conv.inputs[2].name + "_bnfold", b_new)
        else:
            conv.inputs.append(gs.Constant(conv.name + "_bias_bnfold", b_new))

        # 改边：Conv 直接生产 BN 原来的输出张量；BN 和旧的中间张量变成孤儿，由 cleanup 删掉
        y = bn.outputs[0]
        bn.outputs.clear()
        bn.inputs.clear()
        conv.outputs = [y]
        stats["folded"] += 1
    graph.cleanup().toposort()
    return stats


NO_FOLD_OPS = {"RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike",
               "Multinomial", "Bernoulli", "Dropout", "If", "Loop", "Scan"}     # 有随机性 / 控制流的不折


def fold_constants(graph: gs.Graph, size_limit_bytes: int = 1 << 20) -> dict:
    """所有输入都是常量的节点，离线算出输出、替换成 Constant，节点删掉。

    步骤：① 按拓扑序标记"可折叠"节点（Shape 节点只要输入形状全静态也算）
          ② 把可折叠节点拼成子图，用 onnxruntime 跑一次拿到数值
          ③ 被非折叠节点消费的那些张量替换成 Constant；cleanup 删掉孤儿节点
    保护：单个折叠结果超过 size_limit_bytes 就不替换（防 Expand/ConstantOfShape 撑大模型）。
    前提：graph 是从跑过 onnx.shape_inference 的模型导入的，否则 Shape 节点折不掉。
    """
    import onnxruntime as ort
    stats = {"foldable_nodes": 0, "shape_nodes": 0, "replaced": 0, "skipped_big": [], "skipped_ops": []}
    graph.toposort()

    def is_const(t):
        return isinstance(t, gs.Constant) or getattr(t, "_fold_ok", False)

    foldable = []
    for node in graph.nodes:
        if node.op in NO_FOLD_OPS:
            stats["skipped_ops"].append(node.name); continue
        if node.op == "Constant":
            ok = True                   # Constant 节点本身就是常量的来源（导入时不会自动变成 Constant 张量）
        elif node.op == "Shape":
            shp = node.inputs[0].shape
            ok = shp is not None and all(isinstance(d, int) for d in shp)
            if ok: stats["shape_nodes"] += 1
        else:
            ok = len(node.inputs) > 0 and all(is_const(t) for t in node.inputs if t.name)   # 空名张量是可选输入占位
        if ok:
            foldable.append(node)
            for o in node.outputs:
                o._fold_ok = True
    stats["foldable_nodes"] = len(foldable)
    if not foldable:
        return stats

    # 只需要求那些"被非折叠节点消费"或"是图输出"的张量
    fold_set = set(id(n) for n in foldable)
    need = []
    for n in foldable:
        for o in n.outputs:
            if o in graph.outputs or any(id(c) not in fold_set for c in o.outputs):
                need.append(o)
    sub = gs.Graph(nodes=foldable, inputs=[], outputs=list(need),
                   opset=graph.opset, import_domains=graph.import_domains)
    # Shape 节点的输入是激活：子图里把它当成一个只有形状的输入，喂一个同形状的假张量
    fake_inputs = {}
    for n in foldable:
        if n.op == "Shape":
            x = n.inputs[0]
            if isinstance(x, gs.Constant) or getattr(x, "_fold_ok", False):
                continue                # x 由子图内部产生，不能再当子图输入（否则重复定义）
            if x.name not in fake_inputs:
                sub.inputs.append(x); fake_inputs[x.name] = np.zeros(x.shape, dtype=x.dtype or np.float32)
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(gs.export_onnx(sub).SerializeToString(), so, providers=["CPUExecutionProvider"])
    values = sess.run([t.name for t in need], fake_inputs)

    # 替换：新建 Constant，把每个消费者里指向旧张量的位置换成它（消费者持有的是对象引用，只改类型没用）
    for t, v in zip(need, values):
        v = np.asarray(v)
        if v.nbytes > size_limit_bytes:
            stats["skipped_big"].append((t.name, v.nbytes)); continue
        const = gs.Constant(t.name, values=v)
        t.name = t.name + "__prefold"          # 旧张量改名，避免 cleanup 前图里有两个同名张量
        for c in list(t.outputs):
            c.inputs = [const if i is t else i for i in c.inputs]
        if t in graph.outputs:
            graph.outputs = [const if o is t else o for o in graph.outputs]
        stats["replaced"] += 1
    for n in graph.nodes:
        for o in n.outputs:
            if hasattr(o, "_fold_ok"): del o._fold_ok
    graph.cleanup().toposort()
    return stats


def fuse_silu(graph: gs.Graph, target_op: str = "SiLU", domain: str = "", attrs: dict | None = None) -> dict:
    """Sigmoid(x) → Mul(x, ·) 两个节点换成一个 target_op(x)。

    模式：s = Sigmoid(x)；y = Mul(x, s) 或 Mul(s, x)；s 只被这个 Mul 消费；s 不是图输出。
    改边：新节点 inputs=[x]、outputs=[y]（沿用 Mul 的输出张量）；Sigmoid/Mul 断开后由 cleanup 删。
    target_op 由后端决定：本项目运行时叫 SiLU；RKNN 叫 exSwish；ORT 用 com.microsoft::QuickGelu(alpha=1)。
    """
    stats = {"fused": 0, "skipped": []}
    for sig in [n for n in graph.nodes if n.op == "Sigmoid"]:
        x, s = sig.inputs[0], sig.outputs[0]
        if s in graph.outputs or len(s.outputs) != 1 or s.outputs[0].op != "Mul":
            stats["skipped"].append((sig.name, "sigmoid 输出不是恰好被一个 Mul 消费")); continue
        mul = s.outputs[0]
        if set(map(id, mul.inputs)) != {id(x), id(s)}:
            stats["skipped"].append((sig.name, "Mul 的另一个输入不是 sigmoid 的输入 x")); continue
        y = mul.outputs[0]
        mul.outputs.clear(); mul.inputs.clear()
        sig.outputs.clear(); sig.inputs.clear()
        new = gs.Node(op=target_op, name=sig.name + "_silu", inputs=[x], outputs=[y],
                      attrs=dict(attrs or {}), domain=domain or None)
        graph.nodes.append(new)
        stats["fused"] += 1
    graph.cleanup().toposort()
    return stats


def fuse_slices_to_split(graph: gs.Graph) -> dict:
    """同一张量、同一轴上首尾相接、恰好覆盖整个轴的一组 Slice，换成一个 Split。

    条件：starts/ends/axes 都是常量；axes 长度 1；steps 缺省或 1；按 start 排序后 [0,e1),[e1,e2),...,[·,C) 无缝无重叠；
          C 取自张量的静态形状（需要先跑 shape inference）。
    改边：Split(axis) 的 inputs=[data, split 常量]，outputs=原各 Slice 的输出张量（按 start 排序）。
    """
    stats = {"fused": 0, "slices_removed": 0, "skipped": []}
    groups: dict = {}
    for n in graph.nodes:
        if n.op != "Slice": continue
        ins = n.inputs
        if len(ins) < 4 or not all(isinstance(t, gs.Constant) for t in ins[1:]):
            stats["skipped"].append((n.name, "starts/ends/axes 不是常量")); continue
        starts, ends, axes = (int(ins[i].values.reshape(-1)[0]) for i in (1, 2, 3))
        if ins[3].values.size != 1 or (len(ins) > 4 and int(ins[4].values.reshape(-1)[0]) != 1):
            stats["skipped"].append((n.name, "多轴或 step≠1")); continue
        groups.setdefault((id(ins[0]), axes), []).append((starts, ends, n))
    for (_, axis), items in groups.items():
        items.sort(key=lambda it: it[0])
        data = items[0][2].inputs[0]
        shp = data.shape
        if shp is None or not isinstance(shp[axis], int):
            stats["skipped"].append((data.name, "轴长度未知，无法确认覆盖完整")); continue
        C = shp[axis]
        ok = (len(items) >= 2 and items[0][0] == 0 and items[-1][1] >= C
              and all(items[i][1] == items[i + 1][0] for i in range(len(items) - 1)))
        if not ok:
            stats["skipped"].append((data.name, f"{[(s, e) for s, e, _ in items]} 未无缝覆盖 [0,{C})")); continue
        sizes = [min(e, C) - s for s, e, _ in items]
        outs = [n.outputs[0] for _, _, n in items]
        split_c = gs.Constant(items[0][2].name + "_split_sizes", values=np.asarray(sizes, dtype=np.int64))
        for _, _, n in items:
            n.outputs.clear(); n.inputs.clear()
        new = gs.Node(op="Split", name=items[0][2].name + "_split", inputs=[data, split_c], outputs=outs, attrs={"axis": axis})
        graph.nodes.append(new)
        stats["fused"] += 1; stats["slices_removed"] += len(items)
    graph.cleanup().toposort()
    return stats


def partition_graph(graph: gs.Graph, supported_ops: set, single_cut: bool = True) -> dict:
    """给每个节点打标签：运行时能跑（dev）/ 交给后处理（host），并找出切割边。不改图，只返回决策。

    规则：算子不在 supported_ops 里 → host。
          single_cut=True 时，host 节点的所有下游也 → host（只切一刀，数据不来回搬）。
    返回：{"dev": [...], "host": [...], "cut": [...张量], "cut_bytes": int, "unsupported": {op: 次数}}
      cut = 由 dev 节点生产、被 host 节点消费（或本身是图输出）的张量 = 运行时子图的输出。
    """
    graph.toposort()
    label, unsupported = {}, {}
    for n in graph.nodes:
        if n.op in supported_ops:
            label[id(n)] = "dev"
        else:
            label[id(n)] = "host"; unsupported[n.op] = unsupported.get(n.op, 0) + 1
    if single_cut:
        for n in graph.nodes:           # 已按拓扑序，一遍就能把 host 标签传到所有下游
            if label[id(n)] == "host": continue
            for t in n.inputs:
                if isinstance(t, gs.Variable) and t.inputs and label[id(t.inputs[0])] == "host":
                    label[id(n)] = "host"; break
    dev = [n for n in graph.nodes if label[id(n)] == "dev"]
    host = [n for n in graph.nodes if label[id(n)] == "host"]
    cut = []
    for n in dev:
        for t in n.outputs:
            if t in graph.outputs or any(label[id(c)] == "host" for c in t.outputs):
                cut.append(t)
    cut_bytes = sum(int(np.prod(t.shape)) * np.dtype(t.dtype or np.float32).itemsize for t in cut)
    return {"dev": dev, "host": host, "cut": cut, "cut_bytes": cut_bytes, "unsupported": unsupported}
