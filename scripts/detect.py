#!/usr/bin/env python3
"""检测后处理端到端脚本：跑 yinfer detect，画框，并与 Ultralytics 在同一几何下逐框对照。

关键点：Ultralytics 的 predict 对 .pt 模型默认用 auto letterbox（只把边长补齐到 stride 的整数倍），
和本项目的固定方形 letterbox（auto=False）不是同一种几何，直接对照会差几个像素。
所以这里把**同一张 letterbox 后的 640×640 图**喂给两边，框都在 letterbox 图坐标里比较；
画图时再统一反算回原图坐标。

用法：
    python3 scripts/detect.py --ref artifacts/ref --out artifacts/detect-bus.jpg
    python3 scripts/detect.py --ref artifacts/stage1-zidane --out artifacts/detect-zidane.jpg
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))
from make_reference import letterbox as our_letterbox  # noqa: E402


def read_meta(ref_dir):
    """meta.txt：image <路径> <宽> <高>；letterbox <size> <缩放> <左右填充> <上下填充>。"""
    image = size = scale = pad_x = pad_y = width = height = None
    for line in (Path(ref_dir) / "meta.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == "image":
            image, width, height = parts[1], int(parts[2]), int(parts[3])
        elif parts and parts[0] == "letterbox":
            size, scale, pad_x, pad_y = int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    if image is None or size is None:
        raise SystemExit(f"{ref_dir}/meta.txt 里缺少 image 或 letterbox 行")
    return {"image": image, "size": size, "scale": scale, "pad_x": pad_x, "pad_y": pad_y,
            "width": width, "height": height}


def run_detect(binary, model, ref_dir, conf, iou, backend, threads, out_txt):
    """不带 --letterbox：输出的框在 letterbox 后的输入图坐标里。"""
    result = subprocess.run(
        [str(binary), "detect", str(model), str(Path(ref_dir) / "input.bin"),
         "--conf", str(conf), "--iou", str(iou), "--backend", backend, "--threads", str(threads),
         "--out", str(out_txt)],
        capture_output=True, encoding="utf-8", timeout=300)
    if result.returncode:
        raise SystemExit(f"yinfer detect 失败（exit={result.returncode}）:\n{result.stdout}\n{result.stderr}")
    boxes = []
    for line in Path(out_txt).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, score, x1, y1, x2, y2 = line.split()
        boxes.append({"cls": int(cls), "score": float(score),
                      "box": [float(x1), float(y1), float(x2), float(y2)]})
    return boxes, result.stdout


def ultralytics_boxes(weights, image_bgr, conf, iou, imgsz):
    """在已经 letterbox 好的方形图上跑官方实现；auto letterbox 对 640×640 是空操作，几何与输入一致。"""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    result = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    return [{"cls": int(box.cls.item()), "score": float(box.conf.item()),
             "box": [float(v) for v in box.xyxy[0].tolist()]} for box in result.boxes]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def match(mine, theirs):
    """同类框按 IoU 贪心配对，返回 (配对, 我方未配对, 对方未配对)。"""
    used, pairs = set(), []
    for m in sorted(mine, key=lambda d: -d["score"]):
        best, best_iou = None, 0.0
        for j, t in enumerate(theirs):
            if j in used or t["cls"] != m["cls"]:
                continue
            value = iou(m["box"], t["box"])
            if value > best_iou:
                best, best_iou = j, value
        if best is not None:
            used.add(best)
            pairs.append((m, theirs[best], best_iou))
    unmatched_mine = [m for m in mine if not any(m is p[0] for p in pairs)]
    unmatched_theirs = [t for j, t in enumerate(theirs) if j not in used]
    return pairs, unmatched_mine, unmatched_theirs


def unmap(boxes, scale, pad_x, pad_y):
    out = []
    for d in boxes:
        x1, y1, x2, y2 = d["box"]
        out.append({**d, "box": [(x1 - pad_x) / scale, (y1 - pad_y) / scale,
                                 (x2 - pad_x) / scale, (y2 - pad_y) / scale]})
    return out


def draw(image_path, mine, theirs, out_path):
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"读不到图片 {image_path}")
    for d in theirs:
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(img, f"ref cls{d['cls']}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)
    for d in mine:
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(img, f"ours cls{d['cls']} {d['score']:.2f}", (x1, min(img.shape[0] - 4, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    print(f"画框图已保存（绿=本引擎，红=Ultralytics 参考）: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default="build/yinfer")
    ap.add_argument("--model", default="artifacts/model")
    ap.add_argument("--ref", default="artifacts/ref", help="参考目录（提供 input.bin 与 meta.txt）")
    ap.add_argument("--weights", default="artifacts/yolov8n.pt", help="官方权重（与导出 ONNX 用的同一份）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--backend", default="avx2")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="artifacts/detect-bus.jpg")
    ap.add_argument("--report", default=None, help="把对照结果写成 JSON")
    ap.add_argument("--skip-ultralytics", action="store_true", help="只画本引擎的框，不做对照")
    args = ap.parse_args()

    binary = Path(args.binary) if Path(args.binary).is_absolute() else (ROOT / args.binary)
    meta = read_meta(args.ref)
    out_txt = ROOT / "artifacts" / (Path(args.out).stem + "-boxes.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    boxes, stdout = run_detect(binary, args.model, args.ref, args.conf, args.iou,
                               args.backend, args.threads, out_txt)
    print(stdout.strip().splitlines()[0])
    print(f"  图片 {meta['image']}（{meta['width']}x{meta['height']}），letterbox 缩放 {meta['scale']:.6f}，"
          f"左右填 {meta['pad_x']}、上下填 {meta['pad_y']}")

    report = {"image": meta["image"], "conf": args.conf, "iou": args.iou, "backend": args.backend,
              "threads": args.threads, "coords": "letterbox 640x640", "mine": boxes}
    theirs_lb = []
    if not args.skip_ultralytics:
        img = cv2.imread(meta["image"])
        lb_img, r, left, top = our_letterbox(img, meta["size"])
        # 复核 meta.txt 记的参数确实就是这张图的 letterbox 参数。
        assert abs(r - meta["scale"]) < 1e-9 and left == meta["pad_x"] and top == meta["pad_y"], \
            "meta.txt 的 letterbox 参数与重新计算的参数不一致"
        theirs_lb = ultralytics_boxes(args.weights, lb_img, args.conf, args.iou, meta["size"])
        pairs, only_mine, only_theirs = match(boxes, theirs_lb)
        print(f"对照 Ultralytics（同一张 letterbox 图）：我方 {len(boxes)} 个框，参考 {len(theirs_lb)} 个框，"
              f"配对 {len(pairs)} 对")
        for m, t, value in pairs:
            delta = max(abs(a - b) for a, b in zip(m["box"], t["box"]))
            print(f"  cls {m['cls']:>2}  分数 {m['score']:.4f} / {t['score']:.4f}  IoU {value:.4f}  "
                  f"坐标最大偏差 {delta:5.2f} px")
        for m in only_mine:
            print(f"  只有我方: cls {m['cls']} score {m['score']:.4f} box {[round(v, 1) for v in m['box']]}")
        for t in only_theirs:
            print(f"  只有参考: cls {t['cls']} score {t['score']:.4f} box {[round(v, 1) for v in t['box']]}")
        worst = min((p[2] for p in pairs), default=1.0)
        report.update({"ultralytics": theirs_lb,
                       "pairs": [{"mine": p[0], "theirs": p[1], "iou": p[2]} for p in pairs],
                       "only_mine": only_mine, "only_theirs": only_theirs,
                       "min_pair_iou": worst,
                       "passed": not only_mine and not only_theirs and worst >= 0.99})
        print(f"最小配对 IoU {worst:.4f}；结果 {'一致' if report['passed'] else '不一致'}")

    mine_orig = unmap(boxes, meta["scale"], meta["pad_x"], meta["pad_y"])
    print("反算回原图的框：")
    for d in mine_orig:
        print(f"  cls {d['cls']:>2}  {d['score']:.4f}  " + " ".join(f"{v:8.2f}" for v in d["box"]))
    theirs_orig = unmap(theirs_lb, meta["scale"], meta["pad_x"], meta["pad_y"])
    draw(meta["image"], mine_orig, theirs_orig, ROOT / args.out)
    report["mine_original"] = mine_orig
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"对照报告: {args.report}")
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
