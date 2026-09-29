# yolov8-infer-from-scratch

从零手写 YOLOv8 推理引擎的学习项目：Python 图优化前端 + 不依赖推理库的 C++ 运行时，使用 ONNX Runtime 的参考输出做逐层验证。

**当前完成阶段 1 的算子部分**：七类 CPU 标量算子均已实现，可以独立执行 152 节点子图并产生六路检测头张量。两张测试图片逐层对拍均为 `REF=0`。生命周期内存复用尚未实现，激活仍独占空间；SSE/AVX2、CUDA、DFL 解码和 NMS 属于后续计划，目前还不能输出最终检测框。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 前端导出与模型格式、C++ 加载与执行骨架、逐层对拍工具 | 完成 |
| 1 | 七个算子的标量实现，整图对齐 ORT；按生命周期复用激活内存 | 算子完成，内存复用待实现 |
| 2 | SSE 重写逐元素算子、池化、SiLU 和卷积微内核 | 待实现 |
| 3 | 卷积覆盖全部形状，偏置与 SiLU 融进卷积，多线程 | 待实现 |
| 4 | CUDA 后端：隐式 GEMM、共享内存与寄存器分块 | 待实现 |
| 5 | 可选：INT8 卷积 / ARM NEON 移植 | 待实现 |

## 整体结构

```
Python 前端 frontend/（离线跑一次）
  yolov8n.pt ──export_onnx.py──▶ yolov8n_bn.onnx（保留 BatchNormalization）
             ──compile_model.py──▶ BN 折叠 → 常量折叠 → Slice 合成 Split → SiLU 融合 → 按运行时支持的算子切图
                                 ▶ artifacts/model/model.txt + weights.bin
             ──make_reference.py──▶ artifacts/ref/input.bin + ref.bin + ref.txt

C++ 运行时 runtime/（不依赖推理库）
  Model::load 解析模型、权重整块读入 → 内存规划 → Executor 按拓扑序逐个调用算子
  tools/yinfer：info（模型概况）/ run（子图独立执行）/ verify（逐层对拍）
```

前端导出的目标子图位于检测头 DFL 解码之前：152 个节点、7 种算子，输出 3 个尺度各一个框分支（64 通道）和类别分支（80 通道）。DFL 解码、坐标变换和 NMS 计划放在 C++ 后处理里；这种拆分思路也常用于 NPU 部署，但具体支持范围取决于工具链和版本。

| 项目 | 值 |
|---|---|
| 节点 | Conv 63、SiLU 57、Concat 13、Split 8、Add 6、MaxPool 3、UpsampleNearest 2 |
| 卷积形状 | 3×3 步长 1（32 个）、1×1 步长 1（24 个）、3×3 步长 2（7 个） |
| 计算量 | 卷积 8.74 GFLOP（batch=1，乘和加各算一次） |
| 权重 | 12.02 MiB（float32） |
| 激活 | 161 个张量，不复用时共 158.8 MiB |

## 快速开始（WSL / Linux）

依赖：g++ ≥ 9、CMake ≥ 3.16；Python 前端需要 torch、ultralytics、numpy、onnx、onnx-graphsurgeon、onnxruntime、opencv-python。下列 `python3` 应指向已安装依赖的环境。

```bash
# 1. 创建产物目录，并把自备的 yolov8n.pt 放进去（权重不随仓库分发）
mkdir -p artifacts
# 2. 导出、编译模型、生成参考数据（默认 ultralytics 自带的 bus.jpg）
PYTHON=python3 bash scripts/prepare.sh
# 3. 编译运行时
bash scripts/build.sh
# 4. 查看模型、逐层对拍
./build/yinfer info artifacts/model
./build/yinfer verify artifacts/model artifacts/ref --mode isolated
./build/yinfer verify artifacts/model artifacts/ref --mode chained
# 5. 独立执行：只读取模型、权重和输入，不读取 ref.bin
./build/yinfer run artifacts/model artifacts/ref/input.bin
# 6. 算子对拍与验证器回归测试
python3 -m unittest discover -s tests -v
```

独立执行命令目前打印六路输出的形状和值域，不进行检测框解码。打印的单次冷启动耗时不是正式性能基准，不用于宣称加速比。

AddressSanitizer + UBSan 检查：

```bash
BUILD_DIR=build-asan bash scripts/build.sh -DYI_SANITIZE=ON
YINFER_BIN=build-asan/yinfer python3 -m unittest discover -s tests -v
./build-asan/yinfer verify artifacts/model artifacts/ref --mode chained
```

`tests/test_verify_nonfinite.py` 只依赖 Python 标准库；`tests/test_ref_ops.py` 额外需要 numpy、onnx、onnxruntime。测试覆盖有限值逐位相同，以及相同比特的 NaN、正无穷和负无穷；非有限值必须判为失败，不能因为比特相同就判通过。

开发环境：Ubuntu 24.04（WSL2）、g++ 13.3、CMake 3.28、Python 3.11、onnx 1.21、onnxruntime 1.24.4、ultralytics 8.4.155。

## 模型文件格式

`model.txt` 是文本，描述图；`weights.bin` 是所有常量首尾相接的 float32 小端数据，每个常量的起点按 64 字节对齐。思路和 ncnn 的 `.param` + `.bin` 一样：图结构人能直接读、能 diff，权重按内存布局直接读入。

```text
format yolov8-infer 1
weights weights.bin 12607552
input images
output /model.22/cv2.0/cv2.0.2/Conv_output_0
tensor images 1,3,640,640 act
tensor model.0.conv.weight_bnfold 16,3,3,3 const 0
node Conv /model.0/conv/Conv in=images,model.0.conv.weight_bnfold,/model.0/conv/Conv_bias_bnfold out=/model.0/bn/BatchNormalization_output_0 kernel=3,3 stride=2,2 pad=1,1,1,1
```

上面只是格式片段，省略了其余声明。`weights` 行记录文件名和字节数；`const` 后是权重文件中的字节偏移。节点按拓扑序排列，加载时检查每个输入先于使用产生、每个激活只产生一次。张量名沿用 ONNX，便于查询参考值。

| 运行时算子 | 属性 | 支持范围 |
|---|---|---|
| Conv | kernel、stride、pad（上 左 下 右） | FP32 NCHW，OIHW 常量权重和偏置，group=1、dilation=1；支持多 batch |
| SiLU | 无 | 同形状逐元素计算，指数参数取非正数，避免有限输入导致 exp 上溢 |
| MaxPool | kernel、stride、pad | FP32 NCHW，单输出，ceil_mode=0、dilation=1；越界跳过，相当于补负无穷 |
| UpsampleNearest | scale | nearest + asymmetric + floor，H/W 可用不同的正整数倍率 |
| Concat | axis | 连续行优先张量的拼接 |
| Split | axis、sizes | 连续行优先张量的切分 |
| Add | 无 | 两个输入同形状，不支持广播 |

前端负责确认 ONNX 属性属于上述支持范围，并将参数型常量输入降级成运行时属性。当前不是通用 ONNX 解释器，也不宣称能够安全读取任意不可信模型文件。

## 逐层对拍

`make_reference.py` 让 ORT 在只做了 BN 折叠、常量折叠和 Slice→Split 的标准算子图上运行图片，并关闭 ORT 图优化、使用单线程。它保存每个激活张量作为参考，文件版本和 SHA256 记录在生成目录的 `manifest.txt` 中。

| 指标 | 含义 |
|---|---|
| exact | 和参考逐位相同的元素比例 |
| max\|d\| | 最大绝对误差 |
| rel | max\|d\| / max\|参考值\|，参考全零时使用最大绝对误差；默认阈值 1e-4 |
| cos | 整个张量作为向量的余弦相似度，只作辅助指标 |

- **isolated**：每个节点都使用参考输入，只量这一层引入的误差。
- **chained**：使用前面节点的计算结果，观察误差累计。
- 未实现算子会标记 REF 并使用参考输出，**不能据此宣称整网通过**；误差传播也会被 REF 节点截断。
- 有 FAIL 时退出码为 1；没有 FAIL 时为 0，即使仍有 REF。独立执行遇到未实现算子时返回 2。
- 当前七种算子已实现，两张测试图片的结果均不含 REF。

### 阶段 1：标量算子验证

固定 batch=1、640×640 letterbox、FP32，沿用阈值 `rel <= 1e-4`，没有因新增算子而放宽。表格统计的是 160 个节点输出张量，不包含图输入。

| 测试图片 | 模式 | EXACT | OK | FAIL | REF | 最大 rel |
|---|---|---:|---:|---:|---:|---:|
| bus.jpg | isolated | 40 | 120 | 0 | 0 | 1.554e-6 |
| bus.jpg | chained | 0 | 160 | 0 | 0 | 5.166e-6 |
| zidane.jpg | isolated | 40 | 120 | 0 | 0 | 1.897e-6 |
| zidane.jpg | chained | 0 | 160 | 0 | 0 | 8.875e-6 |

isolated 中，Concat/Split/Add 的 35 个输出，加上 MaxPool/UpsampleNearest 的 5 个输出逐位相同。chained 中它们继承上游卷积和 SiLU 的差异，因此不再要求与 ORT 的原始参考逐位相同。

验证范围与限制：

- 13 个 unittest 测试方法通过，包含多个子用例；普通构建和 ASan/UBSan 构建均通过。
- 小模型覆盖负数池化、5×5 SPPF、多 batch、非正方形、非对称 padding、不同放大倍率，以及 1×1/3×3 卷积、stride=2、通道累加和偏置。
- 非法倍率、步长、输出形状以及错误权重/偏置形状会被拒绝。
- bus.jpg 的 ASan/UBSan chained 验证通过；真实子图独立执行成功。
- `--mode chained --perturb 10` 故意破坏 bus.jpg 中一个节点的输出，得到 88 个 FAIL、退出码 1，确认错误可以被检测。
- 只说明这些测试样例的数值一致性，不等于检测 mAP 或任意模型精度认证。
- 卷积是逐输出直接累加的正确性基线，未做 im2col、手写 SIMD、多线程或内存复用。

第二张图片复现方式：

```bash
IMAGE=$(python3 -c 'from pathlib import Path; import ultralytics; print(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg")')
python3 frontend/make_reference.py --image "$IMAGE" --out artifacts/stage1-zidane
./build/yinfer verify artifacts/model artifacts/stage1-zidane --mode isolated
./build/yinfer verify artifacts/model artifacts/stage1-zidane --mode chained
```

### 阶段 0：保留的历史记录

阶段 0 仅实现 Concat、Split、Add：两种模式均为 35 个 EXACT、125 个 REF。当时破坏节点 10，isolated 只出现一层 FAIL，chained 出现三层 FAIL；后面的 REF 会截断误差传播。阶段 1 已不再用参考值占位，不能沿用旧的失败层数。

余弦相似度对单点大错可能不敏感：阶段 0 故意改坏一个元素后 rel 已到 1e-2，cos 仍约为 0.99999999，不能只看 cos 判断正确性。

## 目录

```
frontend/   Python 前端：导出、图优化、编译成运行时格式、生成参考数据
runtime/    C++ 运行时：模型加载、内存规划、执行器、七类标量算子（ops/）
tools/      命令行工具 yinfer 与参考数据读取
tests/      小型 ONNX 算子对拍与验证器回归测试
scripts/    prepare.sh（Python 前端）、build.sh（CMake 编译）
artifacts/  权重、ONNX、模型文件、参考数据和本地验证日志，不进版本库
```

## 许可

代码以 MIT 许可发布。YOLOv8 的结构与权重来自 [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）；本仓库不包含其代码和权重，由脚本从用户自备的 `yolov8n.pt` 导出。依赖和模型的许可证不因本仓库采用 MIT 而改变。
