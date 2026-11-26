# Warp 内共享查询目标与哈希复用

本轮只推进一个改动：保留现有批内查询池，让查询复用共同的长邻接表。借鉴 TRUST 的“建表后复用查询”思想；这是我们对现有边中心 kernel 的改造，不是 TRUST 复现。

## 四个固定对照

| 名称 | 行为 |
|---|---|
| edge | 原始每边一线程、短表查询长表 |
| pool | 已有 K=0 批内查询池，重分配查询，但每次仍定位所属边、二分搜索 |
| reuse | 非空任务的实际长表范围完全相同时，共享长表上下文，省去每查询两次目标范围 shuffle；其他批次走原 pool |
| hash | 同 reuse；共同长表长度 16..512、总查询数 Q >= 2b 时，warp 协作建哈希表，池中的查询改为 shared-memory 精确成员查询；其他批次二分/原 pool |

同源顶点不一定代表查询同一张长表：选择短表后两端可能交换，所以检查实际 `(bb,be)`。零查询和尾部无效 lane 不参与共同目标判定。

哈希表固定 1024 个 32 位槽/warp，最多装 512 个键；乘法散列、atomicCAS 插入、线性探测、完整键比较。空值 UINT32_MAX 不与合法 dense ID 重合。每批重新清空/建表，warp 同步保证读写顺序。8 warps/block 共占 32 KiB 哈希共享内存，另加 CUB 临时空间，可能影响占用率；`reuse` 编译实例不预留这张表。CPU/GPU 外部缓冲无需新增全图队列。

本版本仍保留查询归属定位；没有宣称哈希消除了该成本。首次门限为固定探索值，不逐图挑选，也不称为最优参数。

## 服务器执行

```bash
cd /root/cuda-triangle-counting
git pull --ff-only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target tc_gpu_edge_baseline tc_gpu_query_v2 -j 4
python3 tests/check_reuse.py ./build/tc_gpu_query_v2
```

先对 Google 与 S5/S7 取结果，再扩展七张真实图。若合成图不存在，用已有 prepare_benchmark.py generate 生成。四配置 × 3 个独立进程，每进程 warmup 5 次，额外 GPU 计时 30 次；每个进程同时保存首次完整时间。用同轮 baseline 比较，不和历史绝对时间混用。

```bash
python3 scripts/benchmark_query_v2.py --suite reuse --build build \
  --inputs data/web-Google.txt data/generated/S5.txt data/generated/S7.txt \
  --threads 12 --blocks 864 --runs 3 --warmup 5 --repeat 30 \
  --out results/reuse_first
```

完整真实图：

```bash
python3 scripts/benchmark_query_v2.py --suite reuse --build build \
  --inputs data/com-amazon.ungraph.txt data/roadNet-CA.txt data/web-Google.txt \
  data/amazon0302.txt data/Wiki-Vote.txt data/soc-Epinions1.txt data/cit-Patents.txt \
  --threads 12 --blocks 864 --runs 3 --warmup 5 --repeat 30 \
  --out results/reuse_real
```

沿用 runner：顺序随机化、保存输入/二进制 SHA256、命令、各次 JSON、summary.json/CSV。`count_reduce_ms_median` 为预热后 GPU 计数+归约（包括本次建表、同步）；`first_count_reduce_ms` 为首次 GPU 计数+归约；`first_total_ms` 包含预处理/初始化/传输/首次计数，分别解释。

额外诊断（单独 pass，不进入计时样本）：

```bash
./build/tc_gpu_query_v2 data/generated/S5.txt --query-mode hash \
  --parser omp --mapping omp --sort omp --threads 12 --blocks 864 \
  --verify --warmup 2 --repeat 10 --query-diagnostics
```

诊断含 `common_target_batches`、`hash_batches`、`hash_queries`、`hash_inserted_keys`。先看是否确实启用哈希，再判断查找收益能否抵消建表及共享内存成本。

## 验证状态

按本轮要求，代码检查后直接提交，未执行本地测试或编译。真实 CUDA 编译、GPU 正确性与性能由上面的服务器命令确认；当前没有新版加速结论。
