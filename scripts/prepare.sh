#!/usr/bin/env bash
# 准备模型和参考数据（Python 前端）：
#   yolov8n.pt → ONNX（保留 BN）→ 图优化 + 切图 + 写成运行时格式 → ORT 生成逐层参考数据
# 用法：bash scripts/prepare.sh [yolov8n.pt 路径] [测试图片]
#   环境变量 PYTHON 指定解释器（需要 torch、ultralytics、onnx、onnx-graphsurgeon、onnxruntime、opencv-python）
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
PT=${1:-artifacts/yolov8n.pt}
mkdir -p artifacts
"$PY" frontend/export_onnx.py --pt "$PT" --out artifacts/yolov8n_bn.onnx
"$PY" frontend/compile_model.py --onnx artifacts/yolov8n_bn.onnx --opt artifacts/yolov8n_opt.onnx --out artifacts/model
"$PY" frontend/make_reference.py --model artifacts/model --onnx artifacts/yolov8n_opt.onnx \
    --orig artifacts/yolov8n_bn.onnx --out artifacts/ref ${2:+--image "$2"}
