# Warp 协作与轻重分流：探索版

本轮只改变 CUDA 的任务划分，二分交集、图清洗、度数优先定向、向前 CSR 和 64 位计数保持一致。它是用于发现问题的实验实现，不预设一定比 baseline 快；不是动态抢任务，也不是已经完成的 warp 内自适应调度。

## 1. 三个入口怎样分任务

| 程序 | 分配方式 | 额外成本 |
|---|---|---|
| `tc_gpu_edge_baseline` | 一个线程独立处理一条边，grid-stride 循环 | 原有源点元数据、块内及全局归约 |
| `tc_gpu_warp` | 一个 warp 的 32 个线程合作处理一条边，warp-stride 循环 | 短表不足 32 项时部分 lane 无查询 |
| `tc_gpu_split` | 轻边单线程计数；重边入队，再由一个 warp 处理一条重边 | 队列分配、计数器清零、分类判断、原子入队、第二次核函数启动、更多部分和 |

所有入口每块固定 256 个线程。baseline 的默认块数上限为 `SM数×8`，小图按每块 256 条初始任务缩减；warp/split 按每块 8 个 warp 的容量缩减，也设相同上限。可以通过 `--blocks N` 为三个入口指定相同块数。SM 实际驻留多少块由资源和硬件决定，`SM数×8` 并不是每个 SM 同时驻留 8 块，也不是已调优配置。

固定 warp 版中，第 `w` 个 warp 处理边 `w, w+总warp数, ...`。对一条边，先选较短的向前邻居表，lane `k` 查询其中下标 `k, k+32, k+64, ...` 的元素，每个元素仍在长表中二分查找。各 lane 先累加自己的结果，再通过 CUB BlockReduce 按块求和，最后 DeviceReduce 得到总数；不必为每条边都额外做一次 warp 归约。没有任务的线程也必须到达块内归约点。

## 2. 分流规则与队列

对每条定向边 `(u,v)`，令 `s=min(|F(u)|,|F(v)|)`，只看最终 CSR 的向前度数，不看原始图度数。

- `s=0`：跳过，贡献为零。
- `0<s<threshold`：当前线程直接二分计数。
- `s>=threshold`：以 64 位原子计数器获得队列位置，存入该边的 64 位 CSR 下标。

默认 `--threshold 32`，可先探索 16、32、64。阈值等号属于重边。参数必须是正的有符号 32 位整数；阈值很大时可构造无重边对照，阈值 1 则把所有非空查询任务入队。空交集任务不会入队。

第一核函数输出轻边部分和与重边队列。第二核函数在同一默认 CUDA stream 中启动，因此可直接读取已经完成的队列，无需把队列长度传回 CPU 再决定启动。即使队列为空仍启动第二个核函数，成本照常计入。队列顺序可能因原子操作变化，但每条边只入队一次，整数总数不受顺序影响。

队列预留最多 `|E+|` 个 64 位边下标，额外约 `8|E+|+8` 字节（空图也分配一个占位槽），另有额外块部分和。三个参考图队列大约占 7.4、22.1、34.6 MB（十进制）。所有额外设备缓冲区都计入 `device_buffer_bytes`，本轮没有复制第二份图，也没有构造轻边队列。

每次首次运行、预热、重复运行都重新清零并构建队列，不能复用第一次分类结果来宣称单次任务更快。

## 3. 获取和编译

已在仓库中的用户可先 `git pull --ff-only`。如果该服务器的 pull 仍卡住，可在一个新的空目录克隆：

```bash
git clone https://github.com/sawdaicw/cuda-triangle-counting.git cuda-triangle-warp
cd cuda-triangle-warp
```

实验室 A800 的编译命令：

```bash
cmake -S . -B build-lab \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-lab -j 4
```

换机器时按实际 CUDA/GCC 路径和显卡架构调整；不要把 A800 的 `80` 直接用于所有显卡。

## 4. 先验证正确性

先选择你可以使用的 GPU。`CUDA_VISIBLE_DEVICES=0` 中的 0 是示例，按实际环境修改：

```bash
CUDA_VISIBLE_DEVICES=0 ./build-lab/tc_gpu_warp \
  tests/triangle.txt --verify --warmup 0 --repeat 1

CUDA_VISIBLE_DEVICES=0 ./build-lab/tc_gpu_split \
  tests/triangle.txt --threshold 32 --verify --warmup 0 --repeat 1

ctest --test-dir build-lab --output-on-failure
CUDA_VISIBLE_DEVICES=0 python3 tests/check_baseline.py ./build-lab/tc_gpu_edge_baseline
CUDA_VISIBLE_DEVICES=0 python3 tests/check_warp.py \
  ./build-lab/tc_gpu_warp ./build-lab/tc_gpu_split
```

`check_warp.py` 为两个入口各跑 76 个小图的独立枚举核对和 3 类非法输入，再检查部分 warp、跨步循环、K18/K34/K35/K66/K67/K96、混合团与星形、空队列、阈值两侧、重复执行重置和开启分项计时的结果。CPU 侧 `warp_tasks_host` 检查相同的 lane 查询函数、任务覆盖及分流规则；它不能代替真实 GPU 的同步、原子队列和运行时测试。

实现提交时，CPU 侧测试可在开发环境运行；CUDA 编译由仓库 CI 检查（没有 GPU），GPU 正确性与性能仍须在服务器实测。请以本次提交的 CI 状态和服务器输出为准，不把“代码已实现”写成“加速已验证”。

## 5. 小规模对比：相同预处理，三种调度

数据仍使用原来的解压文本，放入或链接到 `data/`。默认先做探索：3 个图，每个版本 3 次独立完整进程；版本为 edge、warp、split16、split32、split64，共 45 次完整样本。

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/benchmark_scheduling.py \
  --build build-lab --threads 32 --runs 3 \
  --thresholds 16 32 64 \
  --out results/raw/warp_explore_01
```

也可以只跑一张图：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/benchmark_scheduling.py \
  --build build-lab --threads 32 --runs 3 \
  --inputs data/web-Google.txt \
  --thresholds 16 32 64 \
  --out results/raw/warp_google_01
```

目录必须不存在，避免覆盖原始结果。脚本使用相同 OpenMP 解析/映射/排序参数，打乱版本顺序（seed 42），保存命令、输入与可执行文件 SHA256、Git 状态、环境、每次 JSON、stderr 和汇总。每次都会 `--verify`。计数、图规模或设备/运行时信息不一致即停止；脚本不替你保证独占 GPU、CPU 亲和性或 NUMA 一致。

默认每进程 `--warmup 0 --repeat 1`，即一次首次完整流程，之后还有一次额外 GPU-only 运行。完整任务样本只取首次 `first_total_ms`；不能把内部 repeat 算成完整任务重复。可用脚本参数 `--repeat 20 --warmup 2` 辅助观察缓存复用后的 GPU 波动，但它们与首次完整任务仍是不同指标。

## 6. 输出和评价

三个入口保留原有 JSON 口径：

- **主指标 `first_total_ms`**：输入打开前到首次 GPU 总数回到 CPU，包含预处理、元数据、CUDA 准备、显存分配和传输、全部计数与归约。完整加速比为同一批次 edge 中位数除以待比较版中位数。
- **次指标 `first_count_reduce_ms`**：首次 GPU 全计数阶段。split 包含计数器清零、轻边计数和入队、重边计数、两部分全局归约；不能只拿重边核函数时间计算加速比。
- `count_reduce_ms_samples`：额外 GPU-only 重复样本，每次 split 仍包含完整队列重建。
- `heavy_edges`、`heavy_edge_fraction`：分流重边数量、占全部定向边的比例。
- `light_edges_including_empty`：其余边，包含空交集任务，不能解释成实际发生查询的轻边数。
- `queue_capacity_edges`、`queue_buffer_bytes`：预留容量及队列+计数器实际分配字节。
- `stats_readback_ms`：最后一次队列长度诊断读回，放在首次完整计时及重复样本外；得到正确总数无需这次诊断。

warp/split 可额外开启 `--profile-gpu`，定位阶段成本：

```bash
CUDA_VISIBLE_DEVICES=0 ./build-lab/tc_gpu_split data/web-Google.txt \
  --parser omp --mapping omp --sort omp --threads 32 \
  --threshold 32 --verify --warmup 2 --repeat 20 --profile-gpu
```

此时 `gpu_stage_profiled=true`，输出 `first_stages_ms` 和三组重复样本：

| 阶段 | 包含内容 |
|---|---|
| `light_count_queue_ms` | 计数器清零、分类、原子入队、轻边交集和轻边块内归约，无法从该融合核函数中单独分出“纯筛选时间” |
| `warp_count_ms` | 固定版全部边或 split 重边的 warp 协作与块内归约 |
| `global_reduce_ms` | CUB 全局求和 |

固定 warp 版没有轻边阶段，数值为 0。事件跨度还可能包含同一 stream 的启动间隙和共享设备干扰，并非纯指令耗时。分项诊断默认关闭，防止附加 CUDA event 对很短的核函数对比造成额外扰动；正式总时间比较不要混用开关状态。

## 7. 如何解读探索结果

先看重边比例，再看完整 GPU 计数阶段，再看完整任务总时间。原子入队、稀疏任务和第二次启动可能使 split 更慢；固定 warp 也可能因短邻居表导致大部分 lane 闲置。阈值按短表长度只是一种廉价估计，不代表真正比较次数或访存代价。

A800 很快不意味着必须换慢卡才能研究。先在同一 GPU 上比较三个版本，必要时加入课程 S1/S2/S3 或更大、度分布不同的图。其他 GPU 可作为硬件适用性探索；每张卡都要重跑自己的 baseline，不能用 A800 baseline 时间除以另一张卡的优化时间，也不能预设消费卡一定更能体现收益。

后续如果“两张表都很长且接近”的任务确实占主要成本，再考虑并行归并；长短悬殊时保留二分。动态任务领取、subwarp、队列压缩和交集算法自适应均留作后续，不把它们算成本轮已实现的功能。
