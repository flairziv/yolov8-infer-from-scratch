# yolov8-infer-from-scratch

从零手写 YOLOv8 推理引擎的学习项目：Python 图优化前端 + 不依赖推理库的 C++ 运行时，使用 ONNX Runtime 的参考输出做逐层验证。

**当前完成阶段 0**：图优化与导出、模型加载、执行骨架，以及 Concat / Split / Add 的标量实现和对拍。Conv、SiLU、MaxPool、上采样尚未实现；SSE/AVX2、CUDA 和 C++ 检测后处理属于后续计划。目前不能独立完成整网推理，也不能输出最终检测框。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 前端导出与模型格式、C++ 加载与执行骨架、逐层对拍工具 | 完成 |
| 1 | 七个算子的标量实现，整图对齐 ORT；按生命周期复用激活内存 | |
| 2 | SSE 重写逐元素算子、池化、SiLU 和卷积微内核 | |
| 3 | 卷积覆盖全部形状，偏置与 SiLU 融进卷积，多线程 | |
| 4 | CUDA 后端：隐式 GEMM、共享内存与寄存器分块 | |
| 5 | 可选：INT8 卷积 / ARM NEON 移植 | |

## 整体结构

```
Python 前端 frontend/（离线跑一次）
  yolov8n.pt ──export_onnx.py──▶ yolov8n_bn.onnx（保留 BatchNormalization）
             ──compile_model.py──▶ BN 折叠 → 常量折叠 → Slice 合成 Split → SiLU 融合 → 按运行时支持的算子切图
                                 ▶ artifacts/model/model.txt + weights.bin        运行时要执行的子图
             ──make_reference.py──▶ artifacts/ref/input.bin + ref.bin + ref.txt   ORT 逐层参考数据

C++ 运行时 runtime/（不依赖任何推理库）
  Model::load 解析模型、权重整块读入 → 内存规划 → Executor 按拓扑序逐个调用算子
  tools/yinfer：info（模型概况）/ run（整图执行）/ verify（逐层对拍）
```

前端导出的目标子图位于检测头 DFL 解码之前：152 个节点、7 种算子，输出 3 个尺度各一个框分支（64 通道）和类别分支（80 通道）。DFL 解码、坐标变换和 NMS 计划放在 C++ 后处理里；这种拆分思路也常用于 NPU 部署，但具体支持范围取决于工具链和版本。

| | |
|---|---|
| 节点 | Conv 63、SiLU 57、Concat 13、Split 8、Add 6、MaxPool 3、UpsampleNearest 2 |
| 卷积形状 | 3×3 步长 1（32 个）、1×1 步长 1（24 个）、3×3 步长 2（7 个） |
| 计算量 | 卷积 8.74 GFLOP（乘和加各算一次） |
| 权重 | 12.02 MiB（float32） |
| 激活 | 161 个张量，不复用时共 158.8 MiB |

## 快速开始（WSL / Linux）

依赖：g++ ≥ 9、CMake ≥ 3.16；Python 需要 torch、ultralytics、onnx、onnx-graphsurgeon、onnxruntime、opencv-python。

```bash
# 1. 把 Ultralytics 的 yolov8n.pt 放到 artifacts/（权重是 AGPL-3.0 许可，不随仓库分发）
# 2. 导出、编译模型、生成参考数据（默认测试图是 ultralytics 自带的 bus.jpg）
PYTHON=python3 bash scripts/prepare.sh
# 3. 编译运行时
bash scripts/build.sh
# 4. 查看模型、逐层对拍
./build/yinfer info artifacts/model
./build/yinfer verify artifacts/model artifacts/ref
./build/yinfer verify artifacts/model artifacts/ref --mode chained --brief
```

调算子时可以编一份带越界检查的版本：`BUILD_DIR=build-asan bash scripts/build.sh -DYI_SANITIZE=ON`。

验证器回归测试只依赖 Python 标准库和已编译的 `yinfer`：

```bash
python3 -m unittest discover -s tests -v
# 检查带 ASan/UBSan 的可执行文件
YINFER_BIN=build-asan/yinfer python3 -m unittest discover -s tests -v
```

测试覆盖有限值逐位相同，以及相同比特的 NaN、正无穷和负无穷；非有限值必须判为失败，不能因为比特相同就判通过。

开发环境：Ubuntu 24.04（WSL2）、g++ 13.3、CMake 3.28、Python 3.11、onnx 1.21、onnxruntime 1.24.4、ultralytics 8.4.155。

## 模型文件格式

`model.txt` 是文本，描述图；`weights.bin` 是所有常量首尾相接的 float32 小端数据，每个常量的起点按 64 字节对齐。思路和 ncnn 的 `.param` + `.bin` 一样：图结构人能直接读、能 diff，权重按内存布局直接读入。

```
format yolov8-infer 1
weights weights.bin 12607552                         # 权重文件名与字节数，加载时核对
input images                                         # 图输入
output /model.22/cv2.0/cv2.0.2/Conv_output_0         # 图输出（共 6 个）
tensor images 1,3,640,640 act                        # 激活：名字 形状
tensor model.0.conv.weight_bnfold 16,3,3,3 const 0   # 常量：名字 形状 在 weights.bin 中的字节偏移
node Conv /model.0/conv/Conv in=images,model.0.conv.weight_bnfold,/model.0/conv/Conv_bias_bnfold out=/model.0/bn/BatchNormalization_output_0 kernel=3,3 stride=2,2 pad=1,1,1,1
```

- 节点行按拓扑序排列，加载时会再检查一遍：每个输入都必须在该节点之前产生，每个激活恰好产生一次。
- 张量名沿用 ONNX 里的名字，这样每一层都能按名字找到 ORT 的参考值。
- 前端把 ONNX 算子收窄成运行时实现的写法，不支持的写法在编译期直接报错。参数型的常量输入会变成属性，比如 Split 的切分长度、Resize 的缩放倍数：

| 运行时算子 | 属性 | 来自 |
|---|---|---|
| Conv | kernel、stride、pad（上 左 下 右） | ONNX Conv，group = 1、dilation = 1 |
| SiLU | 无 | Sigmoid + Mul 融合 |
| MaxPool | kernel、stride、pad | ONNX MaxPool，ceil_mode = 0 |
| UpsampleNearest | scale | ONNX Resize：nearest + asymmetric + floor + 整数倍 |
| Concat | axis | ONNX Concat |
| Split | axis、sizes | 由成组的 Slice 合并而来 |
| Add | 无 | ONNX Add，两个输入同形状、不广播 |

## 逐层对拍

`make_reference.py` 让 ORT 在"只做了 BN 折叠、常量折叠和 Slice→Split"的标准算子图上跑一张真实图片，并关闭 ORT 自己的图优化、只用单线程，保证可复现。它把运行时子图里每个激活张量的值都存下来。`yinfer verify` 每执行完一个节点，就把输出和同名参考值比较：

| 指标 | 含义 |
|---|---|
| exact | 和参考逐位相同的元素比例 |
| max\|d\| | 最大绝对误差 |
| rel | max\|d\| / max\|参考值\|，通过标准默认 ≤ 1e-4 |
| cos | 把整个张量当成一个向量算余弦相似度。它对少数元素的大错很不敏感，只作参考 |

- **isolated** 模式：每个节点的输入都换成参考值，只量这一层自己引入的误差。
- **chained** 模式：节点吃前面节点自己算出的结果，量的是一路累积下来的误差。
- 还没实现的算子直接用参考数据顶上，状态标为 REF，所以任何阶段都能把整张图走完。
- **存在 REF 时不代表整网推理通过**：这些节点没有执行自写实现；chained 模式的误差传播也会在 REF 节点处被参考输出截断。当前的结果只用于检查已实现部分。
- 有 FAIL 时退出码为 1；没有 FAIL 时为 0，即使仍有 REF。独立整图执行 `yinfer run` 遇到未实现算子会返回 2。

阶段 0 的结果：Concat、Split、Add 的 35 个输出张量在两种模式下都和 ORT 逐位相同，其余 125 个为 REF。

`--perturb N` 会在第 N 个节点执行后故意改坏它的输出，用来确认对拍工具本身能抓到错误：

| 故意改坏节点 10（SiLU） | 节点 10 | 下游 Add（节点 11） | 下游 Concat（节点 12） |
|---|---|---|---|
| isolated | FAIL，rel 1.0e-2 | EXACT | EXACT |
| chained | FAIL，rel 1.0e-2 | FAIL，rel 8.6e-3 | FAIL，rel 8.6e-3 |

这时 rel 已到 1e-2，cos 却仍有 0.99999999。所以通过标准看 rel，不看 cos。

## 目录

```
frontend/   Python 前端：导出、图优化 pass、编译成运行时格式、生成参考数据
runtime/    C++ 运行时：模型加载、内存规划、执行器、算子（ops/）
tools/      命令行工具 yinfer 与参考数据读取
scripts/    prepare.sh（Python 前端一条龙）、build.sh（CMake 编译）
artifacts/  生成物（权重、ONNX、模型文件、参考数据），不进版本库
```

## 许可

代码以 MIT 许可发布。YOLOv8 的结构与权重来自 [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）；本仓库不包含其代码和权重，运行时由脚本从用户自备的 `yolov8n.pt` 导出。
