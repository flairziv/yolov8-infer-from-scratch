#!/usr/bin/env bash
# 构建带 CUDA 后端的版本。CUDA 工具链在 /usr/local/cuda-12.6，不在默认 PATH 上，
# 所以要显式告诉 CMake 用哪个 nvcc（CUDACXX）。
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
BUILD_DIR=${BUILD_DIR:-build-cuda} bash scripts/build.sh \
    -DYI_ENABLE_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 "$@"
