# GPU 边并行基线：实现、计时与服务器测试

## 当前状态

更新：边并行基线已在用户实验室 A800 上通过小图、边界与三个真实图核验，并完成每图 20 次完整流程记录。新增可选 `--profile-preprocess`，见[预处理诊断指南](preprocess_profiling.md)。下面的 CUDA 编译/待验证说明记录初次提交时状态；新诊断路径仍需服务器确认。

目标程序为 `tc_gpu_edge_baseline`。这是自行实现的基础 CUDA 边并行二分交集，作为后续 warp 内局部调度的同 GPU 对照；本次未实现优化版。

本地已验证共享二分交集、CSR 边源点元数据以及跨步分配的 CPU 模拟。CUDA 编译交由新增的 GitHub Actions 编译检查，**其实际结果以 Actions 为准；GPU 运行和性能仍待服务器验证**。不能把原有 GPU 单线程程序通过的测试，算作本程序已经通过。

## 1. 算法与分工

1. CPU 使用原有 `read_graph` 清洗输入、按 `(degree,id)` 定向并构建向前 CSR；输入语义和原有串行版相同。
2. CPU 构建 `sources[e]`，记录 CSR 中第 e 条边的源点；`neighbors[e]` 本身就是终点，因此只增加一个长度 m 的 32 位数组，不再复制终点。
3. GPU 每块固定 256 个线程，线程执行 64 位下标的 grid-stride 循环：`e = global_tid; e < m; e += total_threads`。
4. 对每条边，检查两端向前表是否为空；否则遍历较短表的每个元素，在较长表中二分搜索。找到共同邻居时线程局部 64 位计数加一。
5. 每个线程完成全部分配任务后，使用 CUB BlockReduce 将局部计数加到块部分和；包括没有任务的线程在内，所有线程都参与该归约。
6. 使用 CUB DeviceReduce 汇总块部分和，复制最终 64 位结果回 CPU。

没有全局边排序、筛选队列或每个三角形的全局原子累加。空任务直接在执行时跳过。边交集与任务分配是自己的实现，归约使用课程允许的 CUB（CUDA Toolkit 自带）。

默认 blocks 为 `min(ceil(m/256), SM数量×8)`，至少为 1；这是一个初始启动配置，不声称最优。`--blocks N` 可覆盖，以便后续在固定配置下比较。空图也会启动一块，并将所有零部分和正确归约。

核心文件：

- `src/common/edge_binary.hpp`：共享的二分交集与边源点构建。
- `src/cuda/edge_baseline.cu`：任务分配、块归约、全局归约、错误检查与计时。
- `tests/test_edge_binary.cpp`：CPU 侧独立三元组核验和任务覆盖检查。
- `tests/check_baseline.py`：GPU 运行测试入口。

## 2. 晚上在 A100 上的操作顺序

新租服务器后，使用实例页面的新 SSH 地址。先克隆（已有项目也可以 `git pull --ff-only`；若 pull 无响应，可以克隆到一个新的空目录）：

```bash
cd /root
git clone https://github.com/sawdaicw/cuda-triangle-counting.git cuda-triangle-baseline
cd cuda-triangle-baseline
```

确认实际环境，若 nvcc 路径不同，替换后面编译命令中的路径：

```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
g++ --version
cmake --version
python3 --version
```

编译 A100 版本（80 是 A100 架构，其他型号需调整）：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

先跑一个三角形，再跑完整基线验证：

```bash
./build/tc_gpu_edge_baseline tests/triangle.txt --verify --warmup 0 --repeat 1
python3 tests/check_baseline.py ./build/tc_gpu_edge_baseline
```

第一个应得到 `triangles=1`、`verified=true`。第二个复用课程 D1～D4 等 76 图和 3 种非法输入，再检查边数为 0、1、31、32、33、255、256、257、513 的任务边界，以及 K64（2016 条边、41664 个三角形）；强制 1/2 个块检查跨步循环与多块归约。重复执行会检查计数稳定性。脚本运行期间会反复初始化 CUDA 上下文，比单次内核慢是正常的。

也可仅复用原有 76 图测试：

```bash
python3 tests/check_serial.py ./build/tc_gpu_edge_baseline --verify --warmup 0 --repeat 1
```

正确性通过后再测试真实图。原实例已释放，需要重新上传/下载并解压三个数据文件到 `data/`。路径中不应带 `.gz`：

```bash
mkdir -p data results/raw
./build/tc_gpu_edge_baseline data/com-amazon.ungraph.txt --verify --warmup 2 --repeat 20 > results/raw/amazon_edge_baseline.json
```

程序成功后才读取结果；报错或超时可能留下空文件，不能算有效记录：

```bash
python3 -m json.tool results/raw/amazon_edge_baseline.json
```

Amazon 通过后，再运行：

```bash
./build/tc_gpu_edge_baseline data/roadNet-CA.txt --verify --warmup 2 --repeat 20 > results/raw/roadNet-CA_edge_baseline.json
./build/tc_gpu_edge_baseline data/web-Google.txt --verify --warmup 2 --repeat 20 > results/raw/web-Google_edge_baseline.json
```

根据此前串行测试，三角形数依次应为 667129、120676、13391903。新程序仍须自行验证，不能预先填写结果。

原始计时样本会保存在 JSON 中；结果出来前请不要据此声称加速。

## 3. 参数

```text
tc_gpu_edge_baseline INPUT.txt [--verify] [--profile-preprocess] [--repeat N] [--warmup N] [--blocks N]
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| INPUT.txt | 必填 | 原有两列整数文本边表，支持整行 #/% 注释 |
| --profile-preprocess | 关闭 | 增加预处理分项时间对象，不改变图和计数语义 |
| --verify | 关闭 | 额外用 CPU 双指针串行逻辑核验，与 GPU 二分算法不同 |
| --repeat N | 10 | 计时重复次数，1～10000 |
| --warmup N | 2 | 首次完整运行之后、重复计时之前的额外预热次数，0～10000 |
| --blocks N | 自动 | 线程块数，正整数且不能超过设备限制 |

使用可见设备中的逻辑 GPU 0；可以通过 `CUDA_VISIBLE_DEVICES` 选择物理设备。线程块大小固定为 256，本版不提供修改线程块大小的选项。

## 4. 两种计时范围，不能混用

### 一次完整计数：first_total_ms

从读取文件之前开始，到**首次完整结果复制回 CPU 后**结束。包括：

- 文本读取、去自环、合并重复边、编号映射、度数定向、排序向前 CSR；
- `sources` 任务元数据构建；
- CUDA 初始化、设备信息查询、显存分配、CUB 临时存储查询与分配、event 创建；
- CSR 和任务元数据上传；
- 首次边交集计算、块内归约、设备级最终归约；
- 同步等待和最终结果回传。

**这是本版用于观察一次性完整计数成本的指标。** 不含后续重复实验、可选 CPU 核验、JSON 格式化输出与程序结束时的资源释放。没有隐藏筛选或归约成本。

`first_count_reduce_ms` 是首次运行中 CUDA events 包围的“边计算＋块内归约＋最终归约”时间，不是完整程序时间。

### CSR / 元数据 / 显存复用：count_reduce_ms_samples

首次运行之后，额外预热 `warmup` 次，再计时 `repeat` 次。每次包含完整计数和必要归约，每次结果都回传并与首次计数核对；**GPU event 样本不包含回传时间**。

输出全部 `count_reduce_ms_samples`，并计算 min / median / mean。偶数样本的中位数取中间两项平均。`repeated_execution_wall_ms_sum` 是这 repeat 次从启动到结果回传的 CPU 墙钟耗时之和，含同步、回传及相关调用成本。

这些执行共享一次预处理和上传，不能把其内核中位数作为“单次完整计数总时间”。首次运行与额外 warmup 是基准测试准备，未混进计时样本；实际执行次数为 `1 + warmup + repeat`。

题目要求报告准备和复用次数。分析真实复用场景时，应按 `Tprep + Σ Texec` 统计并说明边界；单次任务使用 `first_total_ms`。若要对完整总时间给出中位数，需要多次独立运行程序，而不是对内核重复样本取中位数。warmup 方案是我们自定的测试方法，不是题目规定。

旧 GPU 单线程版的 `kernel_ms`、本版的 `count_reduce_ms_median`、CPU 的 `count_ms` 和 `first_total_ms` 范围不同，不可直接混用计算加速比。未来优化版应复用本版的准备、传输和汇总路径，分别比较同 GPU 完整成本与分项成本。

## 5. 主要输出字段

成功输出一行 JSON，失败向 stderr 输出错误且返回非零退出码。

| 字段 | 含义 |
|---|---|
| backend | 固定 `gpu_edge_binary` |
| input | 输入路径 |
| device / compute_capability | GPU 型号与计算能力 |
| cuda_runtime_version / cuda_driver_version | CUDA API 整数编码，如 12080 表示 12.8；驱动 API 版本不是 nvidia-smi 中的驱动包号 |
| vertices / undirected_edges / triangles | 清洗后图规模和三角形数 |
| blocks / threads_per_block | 实际启动配置 |
| preprocess_ms | 文件读取和原有图预处理 |
| task_metadata_ms | 构建边源点数组耗时 |
| cuda_setup_ms | CUDA 初始化、查询、分配和 event 创建 |
| h2d_ms | CSR 与源点元数据上传 |
| first_count_reduce_ms | 首次计数和两级归约的 event 时间 |
| first_d2h_ms | 首次结果回传时间 |
| first_execution_wall_ms | 首次启动到结果回传的墙钟时间 |
| first_total_ms | 上述首次完整计数总时间 |
| warmup / repeat | 额外预热次数与计时样本数 |
| count_reduce_ms_samples / min / median / mean | 原始样本及统计，实际字段名均以 `count_reduce_ms_` 为前缀 |
| repeated_execution_wall_ms_sum | 计时重复运行的含回传墙钟耗时之和 |
| verified / verify_ms | 是否执行且通过 CPU 核验、核验耗时；false 表示未请求核验 |
| device_buffer_bytes | 程序显式申请的设备缓冲区总字节数 |
| cub_scratch_bytes | CUB 实际分配的临时缓冲区字节数，包含于上一字段 |

所有 `_ms` 字段单位为毫秒。device_buffer_bytes 包含 CSR、sources、块部分和、最终结果、CUB scratch；零长度数组分配至少一个元素。它不包含 CUDA 上下文、运行时额外显存或寄存器/共享内存，不能等同于 nvidia-smi 进程显存。

## 6. 保存可复现实验记录

运行前记录代码版本和输入文件哈希，机器环境也一起保存：

```bash
git rev-parse HEAD > results/raw/code_commit.txt
sha256sum data/com-amazon.ungraph.txt data/roadNet-CA.txt data/web-Google.txt > results/raw/input_sha256.txt
nvidia-smi > results/raw/nvidia-smi.txt
/usr/local/cuda/bin/nvcc --version > results/raw/nvcc-version.txt
g++ --version > results/raw/compiler-version.txt
```

把运行命令、JSON、错误和超时记录一并保留，释放实例前下载或 push。不要提交 data/ 和 build/。如果工作区有未提交源码改动，单独保存差异，提交号本身不足以描述修改后的程序。

## 7. 当前限制

尚未实现 warp 内局部调度、S1～S3 生成器和外部论文实现对照。整图常驻显存，额外有 4m 字节左右的源点元数据；未实现显存限额、分块和计数溢出检测。当前正确性用例没有覆盖超过 2^32 的三角形总数，64 位类型本身不等同于完成超大计数压力测试。正式性能比较仍待真实 GPU 测量。
