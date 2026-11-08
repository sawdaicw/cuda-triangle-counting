# CUDA Triangle Counting

P09 并行计算课程项目，最终形成课程论文。当前先建立正确的串行实现，后续并行方案通过实验探索。

## 当前实现

- Python 小图参考程序和独立枚举 oracle。
- C++ CPU 串行程序 `tc_cpu_serial`。
- CUDA GPU 串行程序 `tc_gpu_serial`：仅启动 **1 个 block、1 个线程**，该线程完成全部计数；不是并行加速版本。
- CPU 负责文件读取、无向图清洗、定向和 CSR 构建，GPU 负责双指针交集计数。
- 使用 32 位稠密顶点索引、64 位 CSR 偏移和无符号 64 位计数。
- 尚无 GPU 实测性能结论；CPU CI 不代表 CUDA 编译或运行通过。

## A100 上编译与验证

需要 CUDA Toolkit、兼容的驱动、C++ 编译器、CMake 和 Python 3。

```bash
export PATH="/usr/local/cuda-12.8/bin:$PATH"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4
./build/tc_gpu_serial tests/triangle.txt --verify
python3 tests/check_serial.py ./build/tc_gpu_serial
```

示例图应输出 `triangles: 1` 和 `verified: true`（实际输出为 JSON）。`--verify` 额外在 CPU 执行相同计数逻辑并核对；测试脚本另用独立三重枚举核验 37 个小图，包含空图、自环、重复边、负数/稀疏编号、完全图和随机图。

```bash
./build/tc_cpu_serial tests/triangle.txt
python3 tests/check_serial.py ./build/tc_cpu_serial
python3 -m unittest discover -s tests -v
```

无 CUDA 环境仅构建 CPU：`cmake -S . -B build-cpu -DTC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`，再执行 `cmake --build build-cpu`。

## 输入和算法

C++/CUDA 输入为**未压缩纯文本边列表**，每行两个有符号 64 位整数，支持空行和以 `#`、`%` 开头的注释。不接受 `n m` 元信息头或第三列权重；`.gz` 先解压。Python 参考 CLI 另支持 `.gz`。

1. 收集端点编号，去除自环，将边规范为 `(min(u,v),max(u,v))` 后去重。
2. 原始编号排序后映射为连续下标；仅在自环中出现的顶点仍保留。没有出现在输入中的孤立点无法从边列表恢复。
3. 根据清洗后的无向度数，按 `(degree,id)` 全序给每条边定向，建立向前邻居 CSR。
4. 每个邻居表按编号排序，对**每条已有定向边** `(u,v)` 双指针求 `F(u)` 与 `F(v)` 的交集并累加。

每个三角形恰好被其顺序最靠前的两个顶点之间的边计入一次，不需要除以 3。对有向网页数据，当前语义是忽略原始方向后统计无向简单图三角形。

## 输出和计时

GPU 输出包含 `preprocess_ms`、`h2d_ms`、`kernel_ms`、`d2h_ms`、`end_to_end_ms` 和 `device_buffer_bytes`。内核计时使用 CUDA events，包含一次内核启动，不含预处理和传输。端到端计时从读取文件前到结果回传后，含 CUDA 初始化和分配，不含可选 CPU 核验、输出和资源析构。每次调用仅运行一次，尚不是正式重复测量协议。

`device_buffer_bytes` 只统计程序申请的 CSR 和结果缓冲区，不包含驱动、上下文等额外显存。整张 CSR 常驻显存，目前没有分块和限额机制。单线程 GPU 版只用于算法验证，大图可能很慢；先完成小图验证，再决定大图测试方式。CPU 单线程仍应作为后续加速比的正式基准。

## 后续研究

并行任务划分、交集算法选择与负载均衡方式尚待探索。[研究计划](docs/research_plan.md)、[实验协议](docs/experiment_protocol.md)、[论文提纲](paper/outline.md) 和配置文件是早期研究草案，里面的 G0/G1/G2 为候选方案，不是已确定的实现或实测结果。

原始图放 `data/`，构建产物放 `build/`，均不提交。代码与实验记录及时 commit 并 push，释放临时实例前检查远端已保存。未指定开源许可证，第三方代码和数据按各自条款处理。
