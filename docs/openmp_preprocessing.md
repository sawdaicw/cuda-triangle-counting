# OpenMP 预处理：解析、编号映射、排序去重

## 实现范围和状态

本轮完成三个可独立开启的优化：快速/多线程解析、拆分后的并行编号映射、并行排序去重。度数累加、定向、CSR 构建和邻居段排序保持串行；GPU 边并行二分内核及归约不变。没有实现 warp 局部调度，也没有声称所有预处理步骤都已并行。

默认仍选择原实现 `legacy / fused / serial`。后续比较应在同一可执行文件上切换选项，保留输入、GPU 和计时边界一致，不把旧提交上的历史数据直接当作新版本的同期对照。

本地已完成 CPU 构建、30 种输入×54 种配置的精确 CSR 核对、12 种非法输入的行号检查、组合模式下 76 图及计时测试，以及较大合成输入检查。CUDA 编译状态以本提交 Actions 为准，新增预处理模式的真实 GPU 运行和服务器性能仍待验证。

## 1. 选项

CPU 入口 `tc_cpu_serial` 和 GPU 入口 `tc_gpu_edge_baseline` 接受相同的预处理选项：

| 参数 | 可选值 / 默认值 | 含义 |
|---|---|---|
| `--parser` | `legacy`（默认）/ `fast` / `omp` | 原流解析；缓冲区快速单线程；缓冲区 OpenMP 解析 |
| `--mapping` | `fused`（默认）/ `serial` / `omp` | 原串行映射度数融合循环；拆分单线程；拆分并行映射 |
| `--sort` | `serial`（默认）/ `omp` | 顶点编号及规范化边的排序去重；不改变最终邻居段排序 |
| `--threads N` | 默认 1，范围 1～256 | 仅供选择了 omp 的阶段使用；不是 GPU 线程数 |
| `--check-csr` | 默认关闭 | 计时结束后再用原版构图，逐项比较 offsets、neighbors；失败则报错 |
| `--profile-preprocess` | 默认关闭 | 输出原有 8 阶段时间；快速解析额外提供内部明细 |

只指定 `--threads 8` 不会自动启用优化。`fast` 始终单线程，`serial` 映射也是单线程。OpenMP 运行时可能受 `OMP_DYNAMIC` / `OMP_THREAD_LIMIT` 等影响，输出同时记录请求和实际使用的线程数。

## 2. 更新和编译

需要 C++17、支持 OpenMP 的 C++ 编译器及运行时；实验室 g++-13 已具备。OpenMP 主机代码独立编译为静态库，再供 CPU/GPU 入口调用，避免把 OpenMP 编译指令混入 CUDA 内核。

```bash
cmake -S . -B build-lab \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-lab -j 4
ctest --test-dir build-lab --output-on-failure
```

若 pull 仍有网络问题，可普通 clone 到新的空目录。原 data/ 和 results/ 保留，需要时将旧 data/ 链接到新目录；不要删除旧实验记录。

## 3. 先检查正确性

```bash
python3 tests/check_preprocess_modes.py ./build-lab/tc_cpu_serial ./build-lab/tc_preprocess_equivalence
python3 tests/check_serial.py ./build-lab/tc_cpu_serial --parser omp --mapping omp --sort omp --threads 4 --check-csr
```

`tc_preprocess_equivalence` 对一个输入检查 3 种解析×3 种映射×2 种排序×线程数 1/3/8，共 54 配置的 CSR 完全相同。Python 测试包括：空文件、CRLF、无末尾换行、正负 64 位边界、超长注释/数据行、跨块重复、自环、随机图、奇数长度排序归并，以及错误行号。旧解析器的十进制符号语义保持兼容，推荐输入仍为两列空白分隔整数。

真实图上先对拟使用的组合检查一次：

```bash
./build-lab/tc_cpu_serial data/com-amazon.ungraph.txt \
  --parser omp --mapping omp --sort omp --threads 8 \
  --profile-preprocess --check-csr
```

预期 `preprocess_options.csr_verified=true`、三角形数 667129。`--check-csr` 的额外原版构图不计入本次预处理/完整任务时间；它可能较耗时，不必每次性能采样都开启。

GPU 小图确认新增主机预处理与 CUDA 入口的衔接：

```bash
CUDA_VISIBLE_DEVICES=0 ./build-lab/tc_gpu_edge_baseline tests/triangle.txt \
  --parser omp --mapping omp --sort omp --threads 8 \
  --check-csr --verify --warmup 0 --repeat 1
```

然后可以复用整个 76 图（含独立枚举）测试：

```bash
CUDA_VISIBLE_DEVICES=0 python3 tests/check_serial.py ./build-lab/tc_gpu_edge_baseline \
  --parser omp --mapping omp --sort omp --threads 4 --check-csr --verify --warmup 0 --repeat 1
```

## 4. 算法细节

### 解析

legacy 原入口完整保留；fast/omp 首先顺序读取文件到连续缓冲区。切点由字节位置估计后移动到完整行边界，允许空分块；所有区间首尾相接，行不重不漏。整数使用 `from_chars`，显式兼容正号，检查溢出和额外字段。

每个任务写自己的边/端点数组。各数组长度做前缀和后，复制到互不重叠的输出区间。同一个顶点在多个线程重复出现没有冲突，之后统一排序去重。错误发生时报告原文件中最早错误的行号。并行区内的异常先记录、退出并行区后再抛出，不跨 OpenMP 区域传播异常。

快速路径需要可 seek 的稳定文件，读取期间不要修改文件。缓冲区读取不是并行磁盘 I/O。

### 映射

原 fused 路径仍在一个串行循环里完成二分映射、写稠密边、累加度数。serial/omp 路径预先 resize 稠密边数组，每个任务只写 `dense[i]`；线程共享只读有序原编号数组，没有哈希表写冲突。之后单独串行扫描稠密边统计度数，避免高连接度顶点上的原子竞争。

serial 是拆分循环本身的对照，多一遍扫描及其成本都包含在 `remap_degree_ms`，不能只比较映射循环。未实现原子度数统计。

### 排序去重

1. 将顶点编号/边数组划成连续段，各段用 std::sort 排序。
2. 多轮两两归并；每一对归并也按输出位置分成多个 tile，通过稳定 co-rank 找到两输入的区间，使最后一轮大归并仍能多线程执行。
3. 排序完，各线程统计自己区间内与**全局前一个元素**不同的项数，因此跨块重复也能正确删除。
4. 对线程计数做 O(线程数) 前缀和，再将不同元素复制到互不重叠的输出位置。

算法会分配 O(输入元素数) 的归并缓冲区；分配、归并分界、线程同步、去重统计和压缩都包含在对应排序阶段时间。排序函数退出时释放临时存储的成本也被阶段计时覆盖。算法是标准并行归并/压缩思路，不作为新算法声称原创。

## 5. 诊断输出与内存

原 `preprocess_profile` 的 8 项和残余项语义保留。在 `preprocess_options` 里额外记录：

- parser / mapping / sort：实际选项。
- threads_requested：请求的 OpenMP 线程数。
- parser_threads_actual / mapping_threads_actual / sorting_threads_max_actual：各阶段实际线程数，排序记录多次并行区中的最大值；未启用并行时为 1。
- input_buffer_bytes：快速路径文件缓冲区大小，legacy 为 0；**不是总主存或峰值主存**。
- csr_verified：是否执行并通过完整 CSR 核对；false 表示未请求检查。
- parser_detail：仅在快速路径且开启 profile 时存在，包含 file_read_ms / partition_ms / parse_collect_ms / merge_collect_ms。

`read_parse_normalize_ms` 仍覆盖整个读取解析过程。parser_detail 是它的内部明细，不要重复加进总时间。缓冲区/线程局部数组释放和合并成本包含在该阶段；函数边界等微小残余可能不在内部四项之和中。

新路径需要额外主存：文件缓冲区、线程局部端点/边、合并后的数组，以及排序 scratch 可能使峰值明显高于原版。各大阶段结束会释放内部临时数组，但没有实现主存预算/外存分块。正式实验建议用 `/usr/bin/time -v` 另外记录 Maximum resident set size；原 first_total_ms 计时边界不因该外部工具改变。

## 6. 先探索，再测 20 次完整 GPU 总时间

脚本每个样本都是**新进程的一次完整读取/预处理/计数**，不是同一 CSR 的内核重复。版本与图的顺序在每轮使用固定种子打乱；每版原始 JSON、错误、命令、输入哈希、源码版本/脏状态和环境配置都会保存。

探索全部 15 种对照，先每图每版 3 次（运行次数较多，可先选子集）：

```bash
python3 scripts/benchmark_preprocess.py \
  --exe ./build-lab/tc_cpu_serial --runs 3 --profile \
  --out results/raw/preprocess-explore
```

此时 metric 是 **preprocess_ms**，只是预处理诊断收益，不是主 GPU 加速比。脚本同时会执行 CPU 计数并检查各版图规模和计数一致；先通过上面的独立正确性/CSR 检查再采样。

配置名称：

| 名称 | 解析 | 映射 | 排序 |
|---|---|---|---|
| legacy | 原版 | 原融合 | 串行 |
| fast | 快速单线程 | 原融合 | 串行 |
| parse2/4/8 | OpenMP 对应线程数 | 原融合 | 串行 |
| map1 | 原版 | 拆分单线程 | 串行 |
| map2/4/8 | 原版 | 拆分 OpenMP | 串行 |
| sort2/4/8 | 原版 | 原融合 | OpenMP |
| all2/4/8 | OpenMP | 拆分 OpenMP | OpenMP |

单个输入和部分配置可以这样选，其他图相同：

```bash
python3 scripts/benchmark_preprocess.py --exe ./build-lab/tc_cpu_serial \
  --inputs data/com-amazon.ungraph.txt --variants legacy fast parse4 map1 map4 sort4 all4 \
  --runs 3 --profile --out results/raw/amazon-explore
```

确认后选择一个统一多线程配置做完整总时间比较，例如暂选 all4（仅是命令示例，不预设它最快），每个图各版 20 次：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/benchmark_preprocess.py \
  --exe ./build-lab/tc_gpu_edge_baseline --gpu \
  --variants legacy fast all4 --runs 20 \
  --out results/raw/preprocess-full20
```

这时 metric 为 **first_total_ms**：从读文件之前到首次 GPU 结果回传，包含预处理、OpenMP 启动/合并/准备/清理、CUDA 初始化、元数据、传输、计数与归约。`--gpu` 内部传入 `--verify --warmup 0 --repeat 1`，仍会有一遍额外内核采样，但主指标在那之前结束；额外 CPU 核验也在计时之外。不把内部样本当完整流程次数。

正式轮不加 --profile，降低诊断开销。汇总文件 summary.json 给出中位数/min/max 和相对 legacy 的加速比；CPU 探索轮的加速比仅针对预处理。若道路图在 all4 上反而变慢，也应保留，而非事后每图选择最有利线程数。

注意：脚本为校验哈希，会在测量前读一遍输入；不清除文件缓存，因此是普通缓存条件下的完整流程，不是冷磁盘测试。它不控制其他用户负载、不保证 CPU 绑核和 GPU 独占。固定物理 GPU，记录共享负载，必要时重测。输出目录必须是新的，避免覆盖历史结果。
