# CUDA Triangle Counting：单 GPU 精确三角形计数

P09 并行计算课程项目。输入无向简单图，输出精确三角形总数。当前主线是 **OpenMP 图预处理 + CUDA 边并行计数**，并比较固定 warp 和轻重分流。

## 当前进展（2026-11-13）

- **已有实现**：CPU 串行、GPU 单线程、GPU 边并行 baseline、每边一个 warp、轻重边队列分流。
- **历史预处理探索**：A800 上并行解析、映射、排序组合显著改善完整时间；历史汇总与本次 A100 数据分开保存。
- **最新 A100 评测**：14 图 × 5 配置 × 20 次独立进程，1400 份原始结果全部 `verified=true`。固定 warp 在多数图上更慢；split16 在 S6/S7 的 GPU 计数阶段分别约 1.37×/1.19×加速。尚无全面胜出的调度方案。
- **下一步探索**：warp 内虚拟查询池，将短表元素查询分配给线程。CPU 映射模型在 204 图、816 组配置上通过；**尚未实现 CUDA 原型，没有新 GPU 性能结论**。

| 现在想看什么 | 入口 |
|---|---|
| A100 实验配置、结果与限制 | [A100 第一轮结果](docs/a100_v1_results.md) |
| 完整原始记录、CSV、图与复算 | [results/a100_v1](results/a100_v1/README.md) |
| 查询级并行方案、正确性与风险 | [下一版设计](docs/query_parallel_exploration.md) |
| 源码阅读顺序与文件职责 | [代码导航](docs/code_map.md) |
| 输入、编号、定向、CSR、唯一计数原理 | [输入与算法](docs/input_and_algorithm.md) |
| CUDA baseline 参数、验证和计时字段 | [基线测试指南](docs/baseline_testing.md) |
| warp/split 实现及运行 | [调度指南](docs/warp_scheduling.md) |
| OpenMP 开关与验证 | [预处理指南](docs/openmp_preprocessing.md) |
| 14 图下载与合成 | [benchmark 清单](data/benchmark.txt) |
| 论文 | [LaTeX 初稿](paper/README.md) |

## 程序结构

统一流程：**读边表 → 去自环/重边 → 编号映射 → 无向度数 → 全序定向 → 排序向前 CSR → GPU 计数与归约 → 返回总数**。

| 可执行文件 | 计数方式 | 保留用途 |
|---|---|---|
| `tc_cpu_serial` | CPU 双指针交集 | CPU 参考、预处理诊断；可选 OpenMP 预处理，计数仍串行 |
| `tc_gpu_serial` | GPU 一个线程，双指针交集 | 环境与串行验证，非主要性能基线 |
| `tc_gpu_edge_baseline` | 每线程处理边，短表元素在长表二分 | 同 GPU 主对照 |
| `tc_gpu_warp` | 一个 warp 协作处理一条边 | 固定粒度对照 |
| `tc_gpu_split` | 轻边线程计算，重边入队后 warp 计算 | 两阶段分流对照 |

GPU 并行入口共用 runner，块内 CUB BlockReduce、全局 DeviceReduce，计数为 64 位。当前没有自动策略选择、查询池 CUDA kernel、逐顶点计数或多 GPU 计数。

## 输入与计数原理

输入为未压缩两列整数边表；`#`/`%` 开头的整行是注释，无数据表头，无第三列或行尾注释。例如：

```text
# 一个三角形
10 20
20 100
10 100
```

有向数据忽略方向：任一方向存在就保留无向边，删除自环、合并重边。原始有符号 64 位编号按数值映射为连续内部编号，**不是按读取顺序编号**。自环端点保留为顶点，因此可以用自环记录声明孤立点；没有端点记录的孤立点无法恢复。

边按 `(无向度数, 编号)` 从小到大定向；每张向前邻接表**内部按编号排序**。两种排序用途不同。令 F(u) 为向前邻居：

```text
T = 对所有定向边 u→v 求和 |F(u) ∩ F(v)|
```

三角形 a≺b≺c 只在边 a→b 上由共同邻居 c 计入一次，不再除以 3 或 6。CPU/单线程 GPU 用双指针，三个并行 GPU 入口用短表遍历加长表二分。详细例子见[输入与算法](docs/input_and_algorithm.md)。

## 编译与正确性检查

需要 C++17、CMake、OpenMP；GPU 构建还需要 CUDA Toolkit。以下架构 80 适用于本项目使用的 A100/A800，其他卡按实际架构配置。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4

python3 -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
python3 tests/check_baseline.py ./build/tc_gpu_edge_baseline
python3 tests/check_warp.py ./build/tc_gpu_warp ./build/tc_gpu_split
```

实验室可显式指定 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`，路径以实际安装为准。没有 CUDA 时使用：

```bash
cmake -S . -B build-cpu -DTC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4
python3 tests/check_serial.py ./build-cpu/tc_cpu_serial
```

`check_serial.py` 使用独立枚举核对 76 个图并拒绝 3 个非法输入。大图的 `--verify` 使用同一 CSR 上的 CPU 双指针参考；它不能替代小图独立穷举。A100 原始结果包没有包含完整 GPU 单元测试日志，应区分核验记录与独立测试证据。

## 运行与输出

```bash
CUDA_VISIBLE_DEVICES=0 ./build/tc_gpu_edge_baseline \
  data/com-amazon.ungraph.txt \
  --parser omp --mapping omp --sort omp --threads 12 \
  --verify --warmup 0 --repeat 1
```

标准输出为一行 JSON：图规模、`triangles`、设备/启动配置、显式缓冲区字节数和计时；错误写入 stderr 并返回非零。12 线程是本轮云机器统一设置，不是所有机器的默认最优值。代码默认仍保留 legacy/fused/serial 预处理。

| 指标 | 用途与范围 |
|---|---|
| `first_total_ms` | **主指标**：从读边表前到首次结果回传；包括必要预处理、元数据、CUDA 初始化、分配、传输、计数和归约 |
| `first_count_reduce_ms` | GPU 辅助指标；包括所需队列清零、分类/入队、计数及归约 |
| `preprocess_ms` / `cuda_setup_ms` | 解释完整时间，不把各阶段中位数直接相加 |
| `count_reduce_ms_samples` | 图已加载后的额外 GPU 重复，与首次完整样本分开 |
| `verified` / `verify_ms` | 额外 CPU 核验及其耗时，位于主计时之外 |

`--warmup 0 --repeat 1` 表示首次完整执行之外还有一次 GPU-only 重复，**不代表一个进程产生两次完整样本**。下载、解压、编译、日志、额外核验与最后析构不计入主指标。计时细节见[实验协议](docs/experiment_protocol.md)。

## 准备与评测数据

```bash
python3 scripts/prepare_benchmark.py download --suite core
python3 scripts/prepare_benchmark.py generate --suite core
```

主集包括 7 张真实图和 7 张确定性合成图；原始大图不提交 Git。`data/benchmark.txt` 是说明文档，不能作为图输入，不要用 `data/*.txt` 盲目运行。下载地址已修正 SNAP 的 `wiki-Vote.txt.gz` 大小写，保存名仍为 `Wiki-Vote.txt`。

统一比较入口为 `scripts/benchmark_scheduling.py`。A100 这轮固定 12 线程预处理、20 次独立进程、阈值 16/32/64、256 线程/块，逐轮打乱顺序。复算已有结果无需 GPU，见[归档说明](results/a100_v1/README.md)。

## 最新结果应怎样理解

| 图 | baseline GPU计数/ms | warp/ms | split16/ms |
|---|---:|---:|---:|
| roadNet-CA | 0.1173 | 1.0752 | 0.1349 |
| S2 网格 | 0.0963 | 1.1577 | 0.1118 |
| S5 大团链 | 22.7226 | 22.6665 | 26.6398 |
| S6 混合图 | 0.8289 | 1.1653 | 0.6048 |
| S7 混合图 | 3.5839 | 3.2515 | 3.0122 |

以上均为 20 次首次 GPU 计数阶段中位数，包含分流和归约。完整时间的变化较小，预处理与 CUDA 准备波动不能归功于 GPU 调度。Wiki-Vote baseline 使用 394 blocks，其余配置 864 blocks，需要补充相同启动网格的对照。全部 14 图和不确定性见[完整报告](results/a100_v1/summary/analysis.md)。

历史 A800 的 1.65×/3.31×/3.54×是**预处理优化的完整任务加速**，不是本轮 GPU 调度收益，不与 A100 数值混算。

## 后续研究与边界

先验证批内查询平铺能否减少拖尾、避免全局队列，再研究任务定位摊薄和低成本选择；不预先认定新方案优于 baseline。查询级思想与 WeTriC 等已有研究相关，不作为首次提出的概念。见[研究计划](docs/research_plan.md)。

当前整张 CSR 常驻显存，没有 out-of-core、显存限额或自动超时。内部顶点32位、偏移和计数64位；计数溢出尚未检测。CPU临时数组也占主存。仓库尚未指定开源许可证，第三方资料遵循各自条款。

## 保存与更新

代码更新后重新编译；本地源码有 Git 元数据时才可使用 `git pull --ff-only`。下载 ZIP 解压的目录不能直接 pull，可另下载新版本并复制需要保留的数据和结果。GitHub 网络阻塞时，优先在本地下载后通过 SCP 上传，不依赖第三方镜像。

实验原始记录需在释放实例前另行备份；源码、文档和特选结果提交到 Git。自动生成的构建目录、大图、解压后的实验工作目录不提交。原始 A100 结果包已作为小型复现归档保存于 `results/a100_v1/raw/`。
