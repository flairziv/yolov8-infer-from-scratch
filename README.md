# yolov8-infer-from-scratch

从零手写 YOLOv8 推理引擎的学习项目：Python 图优化前端 + 不依赖推理库的 C++ 运行时，使用 ONNX Runtime 的参考输出做逐层验证。

**当前完成阶段 5**：CPU 侧完成卷积形状特化、Conv+SiLU 收尾融合、多线程（任意线程数输出与单线程逐位一致）、第二次内存优化（arena 18.75 → 10.94 MiB）与检测后处理（与 Ultralytics 同几何逐框 IoU ≥ 0.9998）；CUDA 后端朴素直接卷积 → 共享内存分块 2.43×，热身后比 AVX2 4 线程快 2.94×；模型可按张量切成多段串联执行（每段可不同后端），串联输出与整体模型逐位相同。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 前端导出与模型格式、C++ 加载与执行骨架、逐层对拍工具 | 完成 |
| 1 | 七个算子的标量实现，整图对齐 ORT；按生命周期复用激活内存 | 完成 |
| 2 | SSE/AVX2 逐元素算子、池化、SiLU、卷积微内核与工作区复用 | 完成 |
| 3 | 卷积形状特化、Conv+SiLU 收尾融合、多线程、第二次内存优化、检测后处理 | 完成 |
| 4 | CUDA 后端：显存 arena、七种算子、直接卷积的共享内存与寄存器分块 | 完成 |
| 5 | 算子切分与串联部署：按张量切段、共享权重、逐段执行、每段可选后端 | **完成**（INT8 / ARM NEON 未做） |

## 整体结构

```
Python 前端 frontend/（离线跑一次）
  yolov8n.pt ──export_onnx.py──▶ yolov8n_bn.onnx（保留 BatchNormalization）
             ──compile_model.py──▶ BN 折叠 → 常量折叠 → Slice 合成 Split → SiLU 融合 → Conv 收尾融合
                                   → 按运行时支持的算子切图
                                 ▶ artifacts/model/model.txt + weights.bin
             ──make_reference.py──▶ artifacts/ref/input.bin + ref.bin + ref.txt

C++ 运行时 runtime/（不依赖推理库）
  Model::load 解析模型、权重整块读入 → 内存规划（生命周期复用、Split 视图、原地覆盖）→ Executor 按拓扑序逐个调用算子
  后端：scalar / sse / avx2（可多线程）/ cuda（激活与权重在显存，主机影子回读）
  tools/yinfer：info（模型概况）/ run（整图执行）/ verify（逐层对拍）/ detect（DFL + NMS，输出检测框）/ chain（多段串联）
```

前端导出的目标子图位于检测头 DFL 解码之前：95 个节点、7 种算子，输出 3 个尺度各一个框分支（64 通道）和类别分支（80 通道）。57 个卷积带 `act=1`，SiLU 收尾已融进卷积。DFL 解码、sigmoid、坐标变换和 NMS 在 C++ 后处理里实现（`tools/postprocess.cpp` + `yinfer detect`，见阶段 3.5）；这种拆分思路也常用于 NPU 部署，但具体支持范围取决于工具链和版本。

| 项目 | 值 |
|---|---|
| 节点 | Conv 63（其中 57 个带 SiLU 收尾）、Concat 13、Split 8、Add 6、MaxPool 3、UpsampleNearest 2 |
| 卷积形状 | 3×3 步长 1（32 个）、1×1 步长 1（24 个）、3×3 步长 2（7 个） |
| 计算量 | 卷积 8.74 GFLOP（batch=1，乘和加各算一次） |
| 权重 | 12.02 MiB（float32） |
| 激活 | 104 个张量，不复用时共 105.8 MiB（融合前为 161 个、158.8 MiB） |

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

CUDA 后端（可选，需要 NVIDIA 设备与 CUDA 工具链；默认构建不带 CUDA，`yinfer backends` 会报 `cuda unavailable`）：

```bash
bash scripts/build_cuda.sh                       # -DYI_ENABLE_CUDA=ON，nvcc 取 /usr/local/cuda-12.6，目标 sm_89
./build-cuda/yinfer verify artifacts/model artifacts/ref --backend cuda --mode chained
./build-cuda/yinfer run artifacts/model artifacts/ref/input.bin --backend cuda --repeat 10
```

`tests/test_verify_nonfinite.py` 只依赖 Python 标准库；`tests/test_ref_ops.py` 额外需要 numpy、onnx、onnxruntime。测试覆盖有限值逐位相同，以及相同比特的 NaN、正无穷和负无穷；非有限值必须判为失败，不能因为比特相同就判通过。

开发环境：Ubuntu 24.04（WSL2）、g++ 13.3、CMake 3.28、Python 3.11、onnx 1.21、onnxruntime 1.24.4、ultralytics 8.4.155。

## 模型文件格式

`model.txt` 是文本，描述图；`weights.bin` 是所有常量首尾相接的 float32 小端数据，每个常量的起点按 64 字节对齐。思路和 ncnn 的 `.param` + `.bin` 一样：图结构人能直接读、能 diff，权重按内存布局直接读入。

```text
format yolov8-infer 2
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
| Conv | kernel、stride、pad（上 左 下 右）、act（可选） | FP32 NCHW，OIHW 常量权重和偏置，group=1、dilation=1；支持多 batch。`act=1` 表示 SiLU 收尾融合 |
| SiLU | 无 | 同形状逐元素计算，指数参数取非正数，避免有限输入导致 exp 上溢 |
| MaxPool | kernel、stride、pad | FP32 NCHW，单输出，ceil_mode=0、dilation=1；越界跳过，相当于补负无穷 |
| UpsampleNearest | scale | nearest + asymmetric + floor，H/W 可用不同的正整数倍率 |
| Concat | axis | 连续行优先张量的拼接 |
| Split | axis、sizes | 连续行优先张量的切分 |
| Add | 无 | 两个输入同形状，不支持广播 |

前端负责确认 ONNX 属性属于上述支持范围，并将参数型常量输入降级成运行时属性。版本 2 增加了 Conv 的可选 `act` 属性；旧运行时读到版本 2 会明确拒绝（见 `runtime/model.cpp` 的版本检查），不会把 `act` 当未知属性忽略后静默算出没有激活的错误结果。Split 视图与原地覆盖都是**运行时内存规划层面**的属性，不改模型文件格式：`model.txt` 里 Split 仍是普通节点，执行器在规划前决定它的输出是指进父张量的切片还是独立缓冲。当前不是通用 ONNX 解释器，也不宣称能够安全读取任意不可信模型文件。

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

### 阶段 5：算子切分与串联部署

部署里常见的切法是"主干放加速器、颈部和头放 CPU"，或者把一个大模型切成几段分别跑。这一阶段做的是"切"和"串"两件事，不改任何节点、权重和执行顺序：

- **`frontend/split_model.py`**：把编译好的模型按 `--at 张量名`（在产生该张量的节点之后切一刀，可重复）或 `--parts N`（按节点数均分）切成多段。每段都是完整合法的 `model.txt`（自己的 input/output/tensor 声明，节点行原样写回），**共享同一份 `weights.bin`**（相对路径引用，常量偏移不变、不复制权重；`--copy-weights` 可选复制）。边界张量由工具推出：在本段产生、被后面的段消费或本身是图输出的激活。最后写出 `chain.txt` 清单（段目录、每段输入/输出、原图输出顺序）。
- **`yinfer chain <chain.txt> <input.bin> [--backend b 或 b1,b2,...] [--threads] [--repeat] [--dump-dir]`**：逐段加载、各建一个执行器（**每段可指定不同后端**），段间按名字把上游段的边界张量回读后写进下游段的输入；最终输出按原图顺序打印/dump，与 `run --dump-dir` 可直接比较。
- **真实模型的切法**：在 SPPF 输出 `/model.9/cv2/act/Mul_output_0` 后切 → part0 主干 45 节点、part1 颈部+头 50 节点，边界正好是 P3/P4/P5 三张特征图（1×64×80×80、1×128×40×40、1×256×20×20，共 4.4 MiB）；`--parts 3` 均分则有 5 个边界张量。
- `scripts/compare_dumps.py` 比较两个 dump 目录（逐位 / 容差两种口径）。

验证：

| 对照 | 结果 |
|---|---|
| 2 段串联（avx2×4）vs 整体 `run` | 六路输出**逐位相同** |
| 3 段串联（avx2×4）vs 整体 `run` | 逐位相同 |
| 异构串联（part0=cuda，part1=avx2）vs 整体 avx2 | 最大 rel 2.55e-6 |
| 同上 vs 整体 cuda | 最大 rel 9.4e-7 |
| `tests/test_split_chain.py` | 合成模型 `--at` / `--parts` / `--copy-weights` 三种切法串联逐位一致（含"一个张量既是边界又在本段内继续使用"的情形）；非法切点、切在最后一个节点后、非空输出目录、后端个数不匹配、清单缺失都被拒绝 |
| 测试矩阵 | 62 个 Python 测试在 CPU 与 CUDA 二进制下通过，4 个 CTest 目标在两种构建下通过 |

串联开销（单进程 `--repeat 10` 热身后中位数，**观测值，不是交错协议**）：整体 avx2×4 49.4 ms vs 2 段 43.3 ms / 3 段 43.7 ms（噪声范围内，边界只是 4.4 MiB 的 memcpy）；整体 cuda 17.0 ms vs 2 段 cuda,cuda 18.0 ms（约 +1 ms，边界张量 D2H+H2D）；异构 cuda,avx2 30.9 ms = part0 cuda 6.7 + part1 avx2 25.9——比全 CUDA 慢，因为颈部+头本来就是 CPU 上的大头。这个例子证明的是**机制**（跨后端切分并逐位/容差可验），不是加速；真实的异构部署会只把加速器不支持的尾巴放 CPU。

边界：只支持顺序串联（没有流水并行与重叠）；切点必须落在节点边界；只支持第一段单输入；没有做 INT8 与 ARM NEON（阶段 5 的另两个可选项）。

```bash
python3 frontend/split_model.py --model artifacts/model --out artifacts/model-parts2 --at /model.9/cv2/act/Mul_output_0
./build/yinfer run artifacts/model artifacts/ref/input.bin --backend avx2 --threads 4 --dump-dir artifacts/chain-ref-avx2
./build/yinfer chain artifacts/model-parts2/chain.txt artifacts/ref/input.bin --backend avx2 --threads 4 --dump-dir artifacts/chain-2-avx2
python3 scripts/compare_dumps.py artifacts/chain-ref-avx2 artifacts/chain-2-avx2      # 全部逐位相同
./build-cuda/yinfer chain artifacts/model-parts2/chain.txt artifacts/ref/input.bin --backend cuda,avx2 --threads 4 --repeat 10
python3 tests/test_split_chain.py
```

### 阶段 4：CUDA 后端

- **构建**：`bash scripts/build_cuda.sh`（`-DYI_ENABLE_CUDA=ON`，nvcc 取 `/usr/local/cuda-12.6`，目标架构 `sm_89`，换卡改 `CMAKE_CUDA_ARCHITECTURES`）。不带 CUDA 的构建链接 stub：`backend_available(CUDA)` 返回 false，`yinfer backends` 打印 `cuda unavailable`，请求 `--backend cuda` 直接报错，不冒充支持。
- **结构**：`runtime/cuda/cuda_api.{h,cu}`（设备初始化、显存分配、H2D/D2H 拷贝、同步，所有 CUDA 错误转异常）、`runtime/cuda/cuda_ops.cu`（七种算子的 kernel 与注册表，`YI_ENABLE_CUDA=OFF` 时由 stub 提供空注册表）。
- **执行器**：CUDA 路径下**激活 arena 与权重整体放显存**（沿用同一份内存规划，Split 视图与原地覆盖照旧生效——视图的设备指针就是父张量的切片地址）；主机侧保留同布局的影子缓冲，`host_ptr(id)` 按张量 D2H 回读、`set_tensor/push_tensor` 写回；`run()` 末尾 `cudaDeviceSynchronize`，所以计时是真实的 GPU 时间而不是提交时间。
- **卷积两版**（累加顺序相同，输出逐位一致；`YI_CUDA_CONV=naive` 切换做消融）：
  - 朴素直接卷积：一线程一个输出点，逐 `ci/kh/kw` 累加，无 im2col、无共享内存。
  - 分块版：一个 block 负责 16×16 输出像素 × 8 个输出通道——输入 patch（含 halo）每个输入通道只从全局内存读一次进共享内存，8 个通道的累加器在寄存器里（输入值复用 8 次），本通道的权重也先放共享内存做块内广播。patch 放不进静态共享内存（stride>2 或 kernel>3）时自动回退朴素版。

验证（`rel <= 1e-4` 沿用，未放宽；GPU 与 CPU 的加法顺序不同，**不承诺逐位一致**）：

| 门禁 | 结果 |
|---|---|
| 4 组对拍（bus/zidane × isolated/chained） | 全部 `FAIL=0、REF=0`，最大 rel 1.22e-5；isolated EXACT/OK 46/57（与 avx2 相同） |
| 分块 vs 朴素卷积六路输出 | `cmp` 逐位相同（12/12 文件） |
| 检测框（CUDA vs CPU，同输入同阈值） | 5 个框全同，分数一致到 6 位小数 |
| `test_cuda_runtime` | 七种算子、显存上的视图与原地、主机回读，与标量后端 rel ~1e-7；没有 CUDA 的构建打印 SKIP |
| 测试矩阵 | CPU 构建、ASan/UBSan 构建、CUDA 构建下：4 个 CTest 目标与 58 个 Python 测试全部通过 |

同机交错计时（ABBA/BAAB × 4 块，每变体 8 个采样，fresh process，`--repeat 10` 取热身后 9 次的中位数）：

| 对照 | 基线中位数 | 候选中位数 | 之比 |
|---|---:|---:|---:|
| avx2×4 线程 → cuda（分块） | 43.09 ms | 14.66 ms | **2.94×** |
| cuda 朴素 → cuda 分块 | 35.39 ms | 14.56 ms | **2.43×** |
| avx2×4 线程 → cuda 朴素（每进程首次执行，冷） | 42.78 ms | 41.70 ms | 1.02 |

- **冷启动值得单列**：每进程第一次执行包含 CUDA 模块加载等一次性开销，朴素 GPU 版冷启动时与 CPU 打平（1.02）——只看冷启动会得出"GPU 没用"的错误结论。这也是 `run` 加 `--repeat` 的原因：报告同时记录 `cold_ms` 与热身后的中位数。
- 分块的收益来自访存：朴素版每个输出点都要重新读 `Cin×Kh×Kw` 个输入，分块版把输入 patch 在共享内存里复用、并用 8 个寄存器累加器把每次读入的输入值复用 8 次。
- 报告：`artifacts/stage4-cuda-vs-avx2-warm.json`（SHA256 `4be88363f9f5acb0b6361343dc2aab75566816001ea4576c2b9fcbd80dc8992c`）、`artifacts/stage4-conv-tiled-vs-naive.json`（`6ebc6c39678de879396f692c6d7c301c0407dcb0a1881b8e243460fc6cf0d9bc`）、冷启动对照 `artifacts/stage4-cuda-vs-avx2.json`（`4d8225ac0de753b74c13b0ebe739aca4d6d8a4362f4de9d3e7b3af1d5588d732`）。

边界与限制：单卡 RTX 4060 Laptop 8GB（sm_89）、WSL2、CUDA 12.6；FP32、单流顺序执行、每节点一次 kernel 启动；Conv 是直接卷积（朴素/共享内存分块），没有用 im2col+cuBLAS，也没有 Tensor Core、INT8 或 cuDNN；Concat/Split/Upsample 是朴素 kernel。14.7 ms 这个数字**不是** TensorRT 级别的性能，不与 TensorRT/ORT-GPU 做比较。

```bash
bash scripts/build_cuda.sh
./build-cuda/yinfer backends
./build-cuda/yinfer verify artifacts/model artifacts/ref --backend cuda --mode chained
./build-cuda/yinfer detect artifacts/model artifacts/ref/input.bin --backend cuda --conf 0.25 --iou 0.7
ctest --test-dir build-cuda --output-on-failure
# 消融：基线 = 分块卷积，候选 = 朴素卷积（YI_CUDA_CONV=naive）
python3 scripts/benchmark_backends.py --binary build-cuda/yinfer --backend cuda \
    --alt-env YI_CUDA_CONV=naive --alt-label naive --blocks 4 --repeat 10 \
    --out artifacts/stage4-conv-tiled-vs-naive.json
```

### 阶段 3.5：检测后处理（能画框）

子图停在检测头卷积之后，所以框分支还没做 DFL softmax、类别分支还没做 sigmoid，这些都在后处理里补：

- `tools/postprocess.cpp`：**DFL 期望**（对 16 个 bin 做 softmax 再取加权期望）、**sigmoid**、置信度过滤、**类内贪心 NMS**、letterbox 反算。尺度配对不写死顺序——按 (H,W) 把图输出两两配成框/类别分支，步长由输入边长推出；通道数不匹配、配不成对会明确报错。
- `yinfer detect <模型目录> <input.bin> [--conf 0.25] [--iou 0.7] [--backend] [--threads] [--letterbox 缩放,左右填充,上下填充] [--out 文件]`：不给 `--letterbox` 时框坐标在 letterbox 后的输入图里，给了就反算回原图坐标。
- `scripts/detect.py`：从参考目录的 `meta.txt` 读 letterbox 参数，跑二进制、画框、并与官方实现逐框对照。

对照口径（这一条很关键）：Ultralytics 的 `predict` 对 `.pt` 模型默认用 **auto letterbox**（只把边长补齐到 stride 的整数倍，不填满方形），与本项目的固定方形 letterbox 不是同一种几何——直接比会差几个像素。所以脚本把**同一张 letterbox 后的 640×640 图**喂给两边，框都在同一坐标系里比较；参考权重用 `artifacts/yolov8n.pt`（与导出 ONNX 的同一份）。

| 图片 | 框数（本引擎 / 参考） | 最小配对 IoU | 坐标最大偏差 | 分数最大偏差 |
|---|---|---:|---:|---:|
| bus.jpg（810×1080） | 5 / 5 | 0.9998 | 0.04 px | 5e-4 |
| zidane.jpg（1280×720） | 3 / 3 | 0.9999 | 0.02 px | 7e-4 |

- bus.jpg 检到 4 个人 + 1 辆巴士，zidane.jpg 检到 2 个人 + 1 条领带（cls 27），与参考一致。
- 画框图 `artifacts/detect-bus.jpg`、`artifacts/detect-zidane.jpg`（绿=本引擎，红=参考，不提交）；对照报告 `artifacts/detect-bus.json`、`artifacts/detect-zidane.json`。
- 验证：C++ `test_postprocess`（DFL 均匀分布的手算期望、尖峰分布趋近对应 bin、sigmoid 分数、阈值过滤、NMS 的同类抑制/跨类保留/IoU 等于阈值的边界/零面积框/重叠链、letterbox 反算、非法输入拒绝）；Python `tests/test_postprocess.py`（合成检测头模型 + 独立 numpy 参考实现，三组 conf/iou 下逐框 IoU>0.999 且分数一致到 2e-4）。58 个 Python 测试、3 个 CTest 目标在 Release 与 ASan/UBSan 下通过。
- 边界：只支持这个子图的输出布局（3 尺度、DFL reg_max=16、无 objectness、类内 NMS）；不支持 agnostic NMS、多标签、旋转框、分割/姿态头。对照只说明这两张图的框与官方实现一致，不是 mAP 评测。

```bash
./build/yinfer detect artifacts/model artifacts/ref/input.bin --backend avx2 --threads 4 --conf 0.25 --iou 0.7
python3 scripts/detect.py --ref artifacts/ref --out artifacts/detect-bus.jpg --report artifacts/detect-bus.json
python3 scripts/detect.py --ref artifacts/stage1-zidane --out artifacts/detect-zidane.jpg
python3 tests/test_postprocess.py
```

### 阶段 3.4：Split 视图、输入解钉与原地覆盖

三项都在内存侧，数值必须逐位不变：

1. **Split 零拷贝视图**（`memory_plan.cpp: mark_split_views`）：batch=1 时 Split 的每个输出就是父张量的一段连续切片，把它标记成视图（`Tensor::view_of` / `view_offset`）——不占 arena、不拷贝，`data` 直接指进父张量的对应偏移，Split 内核退化成空操作。多 batch 时切片在内存里不连续，保持原来的拷贝路径。规划器把父张量的生命周期延到最晚的视图消费者，独立校验器检查视图落在父槽位内、父张量活得够久。8 个 Split 的 16 个输出全部命中。
2. **图输入解钉**：输入的槽位在最后一次被读之后即可被后续激活复用（此前保留到执行结束，白占 4.9 MiB）。**契约变化**：每次 `run()` 之前必须重新 `set_input`；CLI、测试与基准工具都按此调用。
3. **原地覆盖**（`plan_reuse` 的 `in_place_target`）：Add/SiLU 读写同一批元素，当某个输入在本节点死亡、不是常量/视图/图输出、没有被任何视图引用、且与输出同样大时，输出直接写进它的槽位。6 个 Add 全部命中。非逐元素算子（Conv/MaxPool/Concat/Split/UpsampleNearest）一律不做原地——池化的窗口读和写会互相踩。

| 指标 | 阶段 3.3 | 阶段 3.4 |
|---|---:|---:|
| 激活 arena（reuse） | 18.75 MiB | **10.94 MiB** |
| 同时存活峰值（独立统计的下界） | 10.94 MiB | 10.94 MiB |
| 激活 arena（naive 对照） | 105.83 MiB | 96.46 MiB |
| 视图张量 / 原地输出 | 0 / 0 | 16 / 6 |

- arena 降低 42%，并且**正好等于按生命周期独立算出的同时存活峰值**：贪心规划已经没有碎片损失。
- 这三个数字不是同一个量：arena 是激活缓冲，权重 12.02 MiB 和卷积工作区 18.5 MiB 在进程里另外占着，别把它们相加后当成 RSS。

验证：

- 12 组对拍（scalar/sse/avx2 × bus/zidane × isolated/chained，threads=4）全部 `FAIL=0、REF=0`，EXACT/OK 分布与阶段 3.3 完全一致。
- **新旧二进制逐位对照**：阶段 3.3 与 3.4 的可执行文件、同一模型、同一输入，六路输出 12/12 文件 `cmp` 逐位相同（sse/avx2 × threads=4）。内存布局变化没有改变任何一个数值。
- 新增 11 组 C++ 内存契约测试：原地链塌缩成单个槽位、非逐元素不原地、输入仍被后续节点读取时禁止覆盖、视图标记与越界拒绝、多 batch 不做视图、视图执行原样复现输入、原地与 naive 逐位一致。54 个 Python 测试、2 个 CTest 目标在 Release 与 ASan/UBSan 下通过；TSan 下 `test_memory_plan`、`test_simd_runtime` 与真实模型 verify/run 无竞争报告。
- 真实模型 `./build/test_memory_plan artifacts/model artifacts/ref/input.bin`：naive 与 reuse 六路输出逐位相同（`REAL_MODEL_EXACT outputs=6 naive_bytes=101145600 reuse_bytes=11468800 views=16 in_place=6`）。

计时（同机交错 ABBA/BAAB，fresh process，avx2）：

| 配置 | 阶段 3.3 中位数 | 阶段 3.4 中位数 | 之比 |
|---|---:|---:|---:|
| threads=1（8 采样/变体） | 124.59 ms | 121.66 ms | 0.98（范围完全重叠，视为噪声） |
| threads=4（8 采样/变体） | 48.05 ms | 44.64 ms | 0.93 |
| threads=4（16 采样/变体） | 47.48 ms | 44.28 ms | 0.93 |

- 单线程测不出差异。4 线程下两次独立交错都落在 0.93，但样本范围仍有重叠，只能作为方向性观察（Split 不再拷贝；原地覆盖让写回落在刚读过的缓存行上）。**这一阶段的主要收益是内存，不是速度。**
- 报告：`artifacts/stage34-memory-avx2-t1.json`（SHA256 `e420bc5287d5c69d1faf05beff784cd921be91898bcefac11ff8bd5109ddeada`）、`artifacts/stage34-memory-avx2-t4.json`（`1324360492e238a232dcee3a474789089b860c4840462424ad4028ea631ef89d`）、`artifacts/stage34-memory-avx2-t4-blocks8.json`（`93b211941bf02392091cc5dae160e8cb399af21fde51b4ca11a3ab67e2ff5e92`）。

```bash
./build/yinfer info artifacts/model --backend avx2        # views=16 in_place=6，arena 与峰值下界
./build/test_memory_plan artifacts/model artifacts/ref/input.bin
./build/yinfer verify artifacts/model artifacts/ref --backend avx2 --mode chained --threads 4 --brief
# 内存消融：基线 = 阶段 3.4，候选 = 阶段 3.3 的二进制
python3 scripts/benchmark_backends.py --backend avx2 --threads 4 \
    --alt-binary artifacts/yinfer-stage33 --alt-label stage33 --blocks 8 --out artifacts/stage34-memory-avx2-t4-blocks8.json
```

### 阶段 3.3：多线程

- **线程池**（`runtime/thread_pool.{h,cpp}`）：固定线程数，`parallel_for` 用原子计数器动态领取工作单元（先做完的线程接着领下一块）；工作单元抛异常时其余单元尽快停下，异常在调用线程重抛；`worker_slot()` 给每个线程一个私有槽位。线程池随 `Executor` 创建一次，节点之间复用，不在每个节点上反复建线程。
- **并行分解**（只在 SIMD 后端；标量后端保持单线程，继续充当逐位参考）：
  - Conv 按输出列块（j 方向）切分；每个工作单元把 K×NR 面板打包进**自己线程的槽位**，写出的输出列互不重叠。
  - im2col 按展开矩阵的行（ci,kh,kw）并行，K*N ≥ 65536 才并行（小层不值得付唤醒开销）。
  - Add/SiLU 按 32K 元素分块；MaxPool 按 (n,c) 平面。Concat/Split/UpsampleNearest 仍是共享标量实现，不参与并行。
- **工作区**：打包面板按线程数复制，`workspace_bytes` 从 18478080（1 线程）增到 18616320（4 线程），+138240 字节 = 每线程 45 KiB（来自 Cin=80 的 3×3 层）；激活 arena 不变（18.75 MiB）——线程数只影响临时空间，不影响内存规划。
- **默认线程数 1**，`--threads N` 显式开启（`info/run/verify` 都支持）。工作单元彼此独立 ⇒ **任意线程数的输出与单线程逐位一致**，这不是容差内的近似。

验证（全部 `FAIL=0、REF=0`，阈值 `rel<=1e-4` 未放宽）：

| 门禁 | 结果 |
|---|---|
| 12 组真实模型（scalar/sse/avx2 × bus/zidane × isolated/chained，threads=4） | 全部通过，EXACT/OK 分布与单线程完全一致（如 avx2 isolated 46/57） |
| 六路输出 threads=1 vs 4（sse/avx2 各 6 个 dump，`cmp`） | 12/12 逐位相同 |
| 54 个 Python 测试 + 2 个 CTest 目标（Release 与 ASan/UBSan） | 通过；新增 `tests/test_threads.py`（1/4/16 线程 dump 逐位、工作区随线程数增长、非法线程数拒绝）与 C++ `check_threads`（含线程数多于列块数的"空工作单元"路径） |

TSan：本机内核 `mmap_rnd_bits` 偏高，TSan 启动时约一半概率报 `unexpected memory mapping`；**没有修改系统 ASLR 设置**（`setarch -R` 与 `-no-pie` 两个方案都未采用），改为重试到 TSan 正常启动——成功启动的运行（单元测试 + 真实模型 run + verify，`halt_on_error=1`）没有竞争报告。

同机交错计时（ABBA/BAAB × 4 块，每变体 8 个采样，fresh process，C++ run 内部计时，WSL 4 vCPU）：

| 后端 | 1 线程中位数 | 4 线程中位数 | 中位数之比 |
|---|---:|---:|---:|
| avx2 | 116.72 ms | 53.24 ms | **2.18×** |
| sse | 267.69 ms | 92.44 ms | **2.90×** |

- 这是**本机 1→4 线程**的对照：不是"4 核理想 4×"，也不能外推到其它核数或设备。VM 只有 4 个 vCPU（宿主 24 线程），更多线程数未测。
- SSE 的扩展性优于 AVX2，与 AVX2 更受内存带宽限制一致；剩余串行部分包括阈值以下的 im2col、Concat/Split/Upsample 的标量拷贝，以及每层两次并行区之间的同步开销。
- 报告：`artifacts/stage3-threads-avx2.json`（SHA256 `865ca2fe1f9aaf0154f16064de4a298a07b0c35896b23591a9982b348196364f`）、`artifacts/stage3-threads-sse.json`（`a9441a10fd3ede203d56ea9132a26554645aa5bc6bd6b13a245ffefd58027152`）。

```bash
bash scripts/build.sh
./build/yinfer info artifacts/model --backend avx2 --threads 4
./build/yinfer verify artifacts/model artifacts/ref --backend avx2 --mode chained --threads 4 --brief
./build/yinfer run artifacts/model artifacts/ref/input.bin --backend avx2 --threads 4 --dump-dir artifacts/t4
# 线程数消融（基线 1 线程，候选 4 线程）
python3 scripts/benchmark_backends.py --backend avx2 --threads 1 --alt-threads 4 \
    --blocks 4 --out artifacts/stage3-threads-avx2.json
# 并发检查：TSan 需要能正常启动（本机内核需重试）；ASan/UBSan 回归
BUILD_DIR=build-tsan bash scripts/build.sh -DYI_TSAN=ON
BUILD_DIR=build-asan bash scripts/build.sh -DYI_SANITIZE=ON
```

### 阶段 3.1–3.2：1×1 直通与收尾融合

两项优化，都可独立关闭做消融对照：

- **Conv+SiLU 收尾融合**（`frontend/passes.py: fuse_conv_silu`）：Conv 的输出只被一个 SiLU 消费时，两个节点合成一个带 `act=1` 的卷积；激活在微内核收尾里对寄存器里的累加结果直接计算，中间张量不再写回内存再读一遍。57 个 SiLU 全部融合，运行时子图 152 → 95 节点，参考数据 158.8 → 105.8 MiB。`compile_model.py --no-fuse-act` 生成未融合对照模型。
- **1×1 直通**（`runtime/ops/simd_common.h: im2col_identity`）：1×1、stride=1、无填充时展开矩阵就是输入本身，`col` 直接指向输入张量，跳过对整张输入的标量 im2col 拷贝。63 个卷积里 24 个命中。

数值验证（`rel <= 1e-4` 沿用，未放宽；两张图片均 `FAIL=0、REF=0`）：

| 图片 | 后端 | isolated EXACT/OK | isolated 最大 rel | chained 最大 rel |
|---|---|---|---:|---:|
| bus.jpg | scalar | 40 / 63 | 1.781e-6 | 5.166e-6 |
| bus.jpg | sse | 40 / 63 | 1.781e-6 | 6.252e-6 |
| bus.jpg | avx2 | 46 / 57 | 1.908e-6 | 4.376e-6 |
| zidane.jpg | scalar | 40 / 63 | 1.999e-6 | 8.875e-6 |
| zidane.jpg | sse | 40 / 63 | 1.999e-6 | 1.565e-5 |
| zidane.jpg | avx2 | 46 / 57 | 1.999e-6 | 1.403e-5 |

- 融合 vs 未融合：同一后端、同一输入的**六路最终输出逐位一致**（`memcmp`）；新增 `tests/test_fusion.py` 对每个可用后端做逐位对照。
- 51 个 Python 测试、2 个 CTest 目标在 Release 和 ASan/UBSan 构建下通过。测试覆盖收尾激活的正负半轴、1×1 带 stride/填充（不能直通）的路径、非法 `act` 拒绝。
- 融合后的模型可以直接用融合前的参考数据对拍（张量集合是原来的子集）；zidane 的验证就复用了阶段 1 生成的参考目录。

同机交错消融（ABBA/BAAB，每变体 8 个采样，fresh process，C++ run 内部计时）：

| 消融 | 后端 | 基线中位数 | 候选中位数 | 中位数之比 |
|---|---|---:|---:|---:|
| 1×1 直通（关 → 开） | avx2 | 205.78 ms | 136.88 ms | **1.50×** |
| 收尾融合（未融合 → 融合） | avx2 | 125.79 ms | 127.32 ms | 0.99 |
| 收尾融合（未融合 → 融合） | sse | 257.38 ms | 258.41 ms | 1.00 |

- 1×1 直通的收益来自去掉了一个逐元素带边界判断的标量拷贝循环，不只是"省一次内存读"。
- **收尾融合在这台机器上测不出差异**（两个后端的中位数之比都在 0.99～1.00，样本范围完全重叠）。合理解释：中间张量大多命中 L3（本机 36 MiB），省下的写回-读取流量有限；激活计算量与独立节点相同，只是换了位置。它在本机的价值是结构性的（模型更小、节点更少），省流量的收益在缓存更小的设备上才会显现。这不是"融合无用"，也不能拿它宣称加速。
- 报告：`artifacts/stage3-bypass-avx2.json`（SHA256 `088f539118c4ca5ced9a6d757c48fd55410857c7a692e7683847ee258f07de06`）、`artifacts/stage3-fusion-avx2-blocks4.json`（`867b6a4a59161a6d5dd29eca4383ba4d2849f1b8f6d26afc43a576235d4dda19`）、`artifacts/stage3-fusion-sse-blocks4.json`（`2d2ebe83dd98ba711e7bd97754163a582a7f9da2296cb3b775fc8e3ac68fca7f`）。

内存侧记录（供阶段 3.4 参考）：融合后同时存活峰值从 12.50 降到 10.94 MiB，但贪心复用规划出的 arena 从 17.19 略升到 18.75 MiB——分配序列改变后贪心装箱的碎片不同。**峰值是理论下界，arena 是启发式结果，两者不要混为一谈。**

```bash
# 融合模型（默认）与未融合对照
PYTHON=python3 bash scripts/prepare.sh
python3 frontend/compile_model.py --onnx artifacts/yolov8n_bn.onnx --opt artifacts/yolov8n_opt.onnx \
    --out artifacts/model-nofuse --no-fuse-act
python3 frontend/make_reference.py --model artifacts/model-nofuse --onnx artifacts/yolov8n_opt.onnx \
    --orig artifacts/yolov8n_bn.onnx --out artifacts/ref-nofuse
./build/yinfer info artifacts/model
./build/yinfer verify artifacts/model artifacts/ref --mode chained --backend avx2
# 消融：基线 = 未融合模型，候选 = 融合模型（ratio > 1 表示融合更快）
python3 scripts/benchmark_backends.py --model artifacts/model-nofuse --ref artifacts/ref-nofuse \
    --alt-model artifacts/model --alt-ref artifacts/ref --alt-label fused --backend avx2 \
    --blocks 4 --out artifacts/fusion-ablation.json
```

### 阶段 2：SIMD 后端、卷积微内核与工作区

```bash
bash scripts/build.sh
./build/yinfer backends
./build/yinfer info artifacts/model --backend avx2
./build/yinfer verify artifacts/model artifacts/ref --backend sse --mode isolated
./build/yinfer verify artifacts/model artifacts/ref --backend avx2 --mode chained
./build/yinfer run artifacts/model artifacts/ref/input.bin --backend avx2
# 可选：导出六路原始张量，不包含 DFL/NMS；文件与时间日志分开
./build/yinfer run artifacts/model artifacts/ref/input.bin --backend avx2 --dump-dir artifacts/my-output
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -v
```

`Executor(model, reuse, Backend::AVX2)` 保留旧构造调用的兼容性，默认 `Backend::Scalar`。`info/run/verify` 可显式选择 `--backend scalar|sse|avx2`；`run/verify` 可选择 `--memory naive|reuse`。dump 目录中为 `output0.bin` 等小端 FP32 张量，以及描述文件名、张量名和形状的 `outputs.txt`。

实现范围：

- SSE 后端实际要求 SSE2；AVX2 后端同时要求 FMA。启动时检查 CPU/OS 能力，不支持时明确报错，不在能力检查前执行 AVX 指令。
- Add 使用非对齐向量读写加标量尾部。MaxPool 的连续完整窗口组向量化，边缘标量处理，strideW 不为 1 或无完整向量组时整节点回退。
- SiLU 自写 7 阶 Taylor/Horner exp 近似，范围缩减到 `[-ln2/2, ln2/2]`；完整向量组中 `|x|<=80` 时走 SIMD，极值、非有限数组和尾部走稳定标量公式。
- Conv 将单个 batch 展开为 `K×N`，再打包 `K×NR` 面板，复用于所有输出通道行。SSE 微内核 `MR=4, NR=8`；AVX2/FMA 为 `MR=4, NR=16`，均有 8 个向量累加器。M/N 尾块显式处理，不假设通道数或输出尺寸整除分块。
- Concat、Split、UpsampleNearest 共享既有实现，不声称所有节点都实现了 SIMD。当前真实模型选择 129 个 SIMD 节点、23 个共享/回退节点；节点内部仍可能处理标量边界。回退是真实计算，不是 REF 占位。
- 工作区在 Executor 初始化时按所有卷积的最大需求分配：`max(align64(K*N*4) + K*NR*4)`，不同层、不同 batch 共享；执行卷积时不申请大型数值缓冲。

| 后端 | 激活 arena | 额外 workspace 字节数 | 额外 workspace MiB |
|---|---:|---:|---:|
| scalar | 17.19 MiB | 0 | 0 |
| sse | 17.19 MiB | 18455040 | 17.60 |
| avx2 | 17.19 MiB | 18478080 | 17.62 |

不能只报 arena 并忽略 workspace：显式展开是本阶段以临时空间换取规则访存的取舍。上表仍不含权重和进程其他内存。

#### 数值与边界验证

以下均为 `FAIL=0、REF=0`，沿用全图 `rel<=1e-4`，未放宽阈值：

| 图片 | 后端 | isolated 最大 rel | chained 最大 rel |
|---|---|---:|---:|
| bus.jpg | sse | 1.554e-6 | 6.252e-6 |
| zidane.jpg | sse | 1.897e-6 | 1.565e-5 |
| bus.jpg | avx2 | 1.484e-6 | 4.376e-6 |
| zidane.jpg | avx2 | 1.644e-6 | 1.403e-5 |

- 43 个 Python 测试、2 个 CTest 目标在 Release 和 ASan/UBSan 构建下通过。SIMD 后端不可用时明确跳过对应测试类，不冒充通过。
- 测试覆盖向量边界、全负池化、非正方形、多 batch、卷积 M/N/K 尾部、错误属性、重复执行与输入更换。
- 两张图片、两种 SIMD 后端分别比较 naive/reuse 六路输出，逐位相同；两种后端的 bus.jpg sanitizer chained 检查通过。
- SiLU 在固定 `[-18,18]` 区间的 20001 个点上，对 float64 数学参考四舍五入为 float32 后的最大距离为 3 ULP、99 分位为 2 ULP（scalar/SSE/AVX2 均如此）。这不是全实数域误差证明。另有密集点、范围缩减边界与极值的逐元素容差测试。
- 禁用 SIMD 的构建可以运行 scalar，请求 avx2 返回错误码 3。这里只测试了编译开关路径，未声称在所有旧 CPU 上实测。
- ISA 标志仅用于对应源文件；SSE 对象未出现 YMM/FMA，AVX2 对象包含 YMM/FMA 指令。公共模板采用内部链接，避免不同 ISA 实例被链接器合并。未启用 fast-math；MSVC 当前明确报不支持，推荐 WSL GCC/Clang。

```bash
python3 scripts/analyze_silu_accuracy.py --out artifacts/silu-new-run.json
BUILD_DIR=build-asan bash scripts/build.sh -DYI_SANITIZE=ON
ctest --test-dir build-asan --output-on-failure
YINFER_BIN=build-asan/yinfer python3 -m unittest discover -s tests -v
BUILD_DIR=build-scalar-only bash scripts/build.sh -DYI_ENABLE_X86_SIMD=OFF
./build-scalar-only/yinfer backends
```

#### 同次交错计时

CPU：i9-14900HX，WSL；同一模型、同一份 bus.jpg 预处理输入。每组 ABBA/BAAB 两块，每个后端 4 次采样，另有预热。每轮 fresh process，取 C++ `run` 内部计时，不包含进程启动、模型加载、输出打印和 dump；包含 im2col、打包与算子执行。

| 配对组 | scalar 中位数 ms | 候选中位数 ms | 候选范围 ms | 中位数之比 |
|---|---:|---:|---:|---:|
| scalar / sse | 6819.99 | 457.66 | 423.12～461.55 | 14.90× |
| scalar / avx2 | 6731.88 | 214.18 | 206.98～227.25 | 31.43× |

这些是**完整后端变化**的同次对照，不是单条 SIMD 指令的加速比，也不是与 ORT/TensorRT 的比较。宿主机调度和频率未固定；loadavg 是观测，不是资源隔离门禁。每进程首次执行不等于常驻服务吞吐，不能与旧阶段的单次耗时相除，也不能泛化为其他设备性能。

```bash
python3 scripts/benchmark_backends.py --blocks 2 --out artifacts/stage2-benchmark.json
```

本次原始报告：`artifacts/stage2-benchmark.json`，SHA256：`e5f36ac2c52460a42bbb6c0f550f5dfe035dc70d84cec717565774ade5304dd2`。报告包含逐轮计时、正确性输出、二进制/输入/模型/权重哈希与运行时代码哈希。为保留旧记录，脚本拒绝覆盖已存在的报告；复跑应指定新文件名。大产物与原始日志不提交。

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
- 卷积是逐输出直接累加的正确性基线，未做 im2col、手写 SIMD 或多线程；内存复用不改变算子计算代码。

第二张图片复现方式：

```bash
IMAGE=$(python3 -c 'from pathlib import Path; import ultralytics; print(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg")')
python3 frontend/make_reference.py --image "$IMAGE" --out artifacts/stage1-zidane
./build/yinfer verify artifacts/model artifacts/stage1-zidane --mode isolated
./build/yinfer verify artifacts/model artifacts/stage1-zidane --mode chained
```

### 阶段 1.4：生命周期内存复用

`Executor(model, true)` 默认使用复用规划，传 `false` 保留独占空间对照。`yinfer verify` 支持 `--memory naive|reuse`，默认 reuse；`info` 同时列出两种规划大小，`run` 使用默认复用规划。

| 规划 | 激活 arena 字节数 | MiB |
|---|---:|---:|
| naive | 166528000 | 158.81 |
| reuse | 18022400 | 17.19 |

激活 arena 减少约 89.2%。这不是进程 RSS 或整个推理链路的内存降幅：权重、参考数据、解析结构和运行时开销不在此数值内，也不由此宣称性能加速。

策略与边界：

- 按节点顺序计算每个激活的闭区间生命周期；只有 `旧张量最后使用节点 < 新张量生产节点` 才能复用。
- 同一节点的输入、输出以及多个输出互不覆盖；不做原地算子或 Split 视图别名。**（阶段 3.4 已放宽：Add/SiLU 允许原地覆盖已死亡的输入，batch=1 的 Split 输出改成零拷贝视图；同一节点内部的覆盖规则仍然禁止，见阶段 3.4 一节。）**
- 图输入和图输出保留到执行结束：支持只 `set_input` 一次后连续 `run`，早期图输出不会被后续节点覆盖。**（阶段 3.4 起图输入改为按最后消费者释放：每次 `run` 之前必须重新 `set_input`；图输出仍然保留到结束。）**
- 选择最小的足够大空闲块，拆分剩余空间，并合并相邻空闲块；所有偏移保持 64 字节对齐。这是启发式方案，不保证全局最优。
- 独立检查范围、对齐和活跃区间重叠；ASan 不一定能发现同一 arena 内部的数据覆盖。
- 只适用于顺序执行、连续张量和无视图别名的当前模型契约。中间张量过了生命周期后不再保证保留，逐层对拍必须在执行当前节点后立即进行。

验证：7 组 C++ 内存测试、13 个 Python 算子/验证器测试在 Release 和 ASan/UBSan 构建下通过。两张图片分别执行 naive/reuse，六路最终输出 `memcmp` 完全一致；两图 × 两种输入模式 × 两种内存规划的 ORT 对拍都通过，结果与上表阶段 1 数值一致。bus.jpg 的 sanitizer 真实模型对照和 chained 验证通过；错误注入仍返回 1。

```bash
bash scripts/build.sh
ctest --test-dir build --output-on-failure
./build/yinfer info artifacts/model
./build/yinfer verify artifacts/model artifacts/ref --mode chained --memory naive
./build/yinfer verify artifacts/model artifacts/ref --mode chained --memory reuse
# 直接比较自写算子在两种规划下的六路输出，而不是只比较各自对 ORT 的容差
./build/test_memory_plan artifacts/model artifacts/ref/input.bin
./build/test_memory_plan artifacts/model artifacts/stage1-zidane/input.bin
# 内存测试的 sanitizer 版本
BUILD_DIR=build-asan bash scripts/build.sh -DYI_SANITIZE=ON
ctest --test-dir build-asan --output-on-failure
```

### 阶段 0：保留的历史记录

阶段 0 仅实现 Concat、Split、Add：两种模式均为 35 个 EXACT、125 个 REF。当时破坏节点 10，isolated 只出现一层 FAIL，chained 出现三层 FAIL；后面的 REF 会截断误差传播。阶段 1 已不再用参考值占位，不能沿用旧的失败层数。

余弦相似度对单点大错可能不敏感：阶段 0 故意改坏一个元素后 rel 已到 1e-2，cos 仍约为 0.99999999，不能只看 cos 判断正确性。

## 目录

```
frontend/   Python 前端：导出、图优化、编译成运行时格式、生成参考数据、按张量切分模型（split_model.py）
runtime/    C++ 运行时：模型加载、内存规划、线程池、执行器、标量与 SSE/AVX2 算子；cuda/ 为 CUDA 后端（含无 CUDA 时的 stub）
tools/      命令行工具 yinfer、参考数据读取、检测后处理（DFL/NMS）
tests/      小型 ONNX 算子对拍、验证器回归、多线程/内存/后处理/CUDA/切分串联的 C++ 与 Python 测试
scripts/    prepare.sh（Python 前端）、build.sh / build_cuda.sh（CMake 编译）、benchmark_backends.py（同机交错基准）、detect.py（画框与对照）、compare_dumps.py（dump 对比）
artifacts/  权重、ONNX、模型文件、参考数据、基准报告和本地验证日志，不进版本库
```

## 许可

代码以 MIT 许可发布。YOLOv8 的结构与权重来自 [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）；本仓库不包含其代码和权重，由脚本从用户自备的 `yolov8n.pt` 导出。依赖和模型的许可证不因本仓库采用 MIT 而改变。
