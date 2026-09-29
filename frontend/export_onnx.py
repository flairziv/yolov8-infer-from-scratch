"""导出：yolov8n.pt → artifacts/yolov8n_bn.onnx，保留 BatchNormalization 节点，交给我们自己的前端去折叠。

Ultralytics 自带的 export 会先 model.fuse() 把 BN 合进卷积、再让导出器做常量折叠，
那样导出来的 ONNX 里已经没有 BN，前端的 BN 折叠在真实模型上就是空操作。所以这里自己调 torch.onnx.export：
  · 不调 fuse()，BN 层原样保留
  · do_constant_folding=False，常量计算也留给我们的前端
"""
import argparse
import collections
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default="artifacts/yolov8n.pt", help="Ultralytics 的 yolov8n.pt（COCO 80 类）")
    ap.add_argument("--out", default="artifacts/yolov8n_bn.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if not os.path.exists(args.pt):
        sys.exit(f"找不到 {args.pt}：把 yolov8n.pt 放到这里，或用 --pt 指定路径")

    import onnx
    import torch
    from ultralytics import YOLO

    model = YOLO(args.pt).model.float().eval()       # DetectionModel（普通 nn.Module），没有 fuse
    head = model.model[-1]
    head.export, head.format = True, "onnx"          # 检测头直接返回拼好的 [1, 84, 8400]，不返回训练用的中间列表
    n_bn = sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    x = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(model, x, args.out, opset_version=17, input_names=["images"], output_names=["output0"],
                      do_constant_folding=False, dynamo=False)

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print(f"[export] {args.pt} → {args.out}")
    print(f"  PyTorch 里 BatchNorm2d {n_bn} 层；ONNX 里 BatchNormalization {ops['BatchNormalization']} 个")
    print(f"  节点 {len(m.graph.node)} 个：{dict(ops.most_common())}")
    print(f"  输出 {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")


if __name__ == "__main__":
    main()
