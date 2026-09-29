"""参考数据：ORT 在 yolov8n_opt.onnx 上跑一张真实图片，把运行时子图里每个激活张量的值存下来，供 C++ 逐层对拍。

输出到 artifacts/ref/：
  input.bin           预处理后的输入 [1,3,640,640] float32（letterbox → BGR 转 RGB → HWC 转 CHW → 除以 255）
  ref.bin + ref.txt   每个激活张量的参考值（64 字节对齐首尾相接）和索引；ref.txt 每行：名字 字节偏移 元素数 形状
  meta.txt            原图尺寸和 letterbox 参数（后处理把框映射回原图时用）
  manifest.txt        生成环境（库版本、ORT 设置）和各文件的 SHA256，保证参考数据可追溯
"""
import argparse
import hashlib
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ALIGN = 64


def read_model(model_dir):
    """从 model.txt 读出图输入名，以及全部激活张量（名字 → 形状，保持声明顺序）。"""
    inputs, acts = [], {}
    with open(os.path.join(model_dir, "model.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p or p[0].startswith("#"):
                continue
            if p[0] == "input":
                inputs.append(p[1])
            elif p[0] == "tensor" and p[3] == "act":
                acts[p[1]] = [int(d) for d in p[2].split(",")]
    return inputs, acts


def letterbox(img, size=640, color=114):
    """和 Ultralytics 的 LetterBox(auto=False, center=True) 逐像素一致：等比缩放到能放进 size×size，两侧对称填灰。"""
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return img, r, left, top


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_image():
    import ultralytics
    return os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="artifacts/model", help="compile_model.py 的输出目录")
    ap.add_argument("--onnx", default="artifacts/yolov8n_opt.onnx", help="标准算子版等价图")
    ap.add_argument("--orig", default="artifacts/yolov8n_bn.onnx", help="优化前的原图，用来核对前端 pass 没改变结果")
    ap.add_argument("--image", default=None, help="测试图片，默认 Ultralytics 自带的 bus.jpg")
    ap.add_argument("--out", default="artifacts/ref")
    args = ap.parse_args()

    import cv2
    import onnx
    import onnxruntime as ort

    image = args.image or default_image()
    inputs, acts = read_model(args.model)
    assert inputs == ["images"], inputs
    size = acts["images"][2]
    img = cv2.imread(image)
    assert img is not None, f"读不了图片 {image}"
    lb, r, left, top = letterbox(img, size)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    # 把子图里的每个激活张量都加成图输出，ORT 才会把中间结果交出来
    m = onnx.load(args.onnx)
    value_info = {v.name: v for v in m.graph.value_info}
    have = {o.name for o in m.graph.output}
    names = [n for n in acts if n not in inputs]
    for n in names:
        if n not in have:
            assert n in value_info, f"{n} 在 {args.onnx} 里没有形状信息"
            m.graph.output.append(value_info[n])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # 不让 ORT 自己融合，中间张量才和我们一一对应
    so.intra_op_num_threads = 1                                                # 单线程：参考数据可复现
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    outs = sess.run(names + ["output0"], {"images": x})
    vals = dict(zip(names + ["output0"], outs))
    vals["images"] = x

    os.makedirs(args.out, exist_ok=True)
    x.tofile(os.path.join(args.out, "input.bin"))
    index, off = [], 0
    with open(os.path.join(args.out, "ref.bin"), "wb") as f:
        for name, shape in acts.items():
            v = np.ascontiguousarray(vals[name], dtype="<f4")
            assert list(v.shape) == shape, (name, v.shape, shape)
            pad = -off % ALIGN
            f.write(b"\0" * pad)
            off += pad
            f.write(v.tobytes())
            index.append(f"{name} {off} {v.size} {','.join(map(str, shape))}")
            off += v.nbytes
    with open(os.path.join(args.out, "ref.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(index) + "\n")
    h, w = img.shape[:2]
    with open(os.path.join(args.out, "meta.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"image {os.path.abspath(image)} {w} {h}\nletterbox {size} {r!r} {left} {top}\n")

    # 核对：前端 pass 前后，ORT 跑两张图的整网输出差多少（BN 折叠改变了浮点舍入，不会逐位相同）
    base = ort.InferenceSession(args.orig, so, providers=["CPUExecutionProvider"]).run(["output0"], {"images": x})[0]
    d = np.abs(base.astype(np.float64) - vals["output0"])
    print(f"[reference] {image}（{w}x{h}）letterbox 缩放 {r:.4f}，左右各填 {left}、上下各填 {top}")
    print(f"  {len(acts)} 个激活张量（含输入）→ {args.out}/ref.bin {off / 2**20:.1f} MiB")
    print(f"  前端 pass 前后整网输出 output0：max|d| = {d.max():.3e}（输出最大值 {np.abs(base).max():.1f}），"
          f"逐位相同 {np.mean(base == vals['output0']) * 100:.1f}%")

    files = [os.path.join(args.model, "model.txt"), os.path.join(args.model, "weights.bin")] + \
            [os.path.join(args.out, n) for n in ("input.bin", "ref.bin", "ref.txt")]
    lines = [f"image {os.path.abspath(image)}",
             f"onnx {onnx.__version__}", f"onnxruntime {ort.__version__}", f"numpy {np.__version__}", f"opencv {cv2.__version__}",
             "ort_graph_optimization disable_all", "ort_threads 1"]
    lines += [f"sha256 {sha256(p)} {os.path.relpath(p, args.out)}" for p in files]
    with open(os.path.join(args.out, "manifest.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  manifest → {args.out}/manifest.txt")


if __name__ == "__main__":
    main()
