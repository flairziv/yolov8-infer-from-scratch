#!/usr/bin/env bash
# 编译 C++ 运行时。额外参数原样传给 cmake 配置步骤，例如：
#   bash scripts/build.sh                                        # Release
#   BUILD_DIR=build-asan bash scripts/build.sh -DYI_SANITIZE=ON   # 带越界检查的调试版
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD_DIR:-build}
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" "$@"
cmake --build "$BUILD" -j"$(nproc)"
