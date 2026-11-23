# 先执行 + 剩余线程数门控（gated）

沿用 `tc_gpu_query_v2`，增加 `--query-mode gated`，原 budget/score/scalar/pool 保留。
**这是新增 CUDA 源码，当前编辑环境无 nvcc/GPU，门控版本尚未编译或 GPU 验证。**
用户已通过的 v2 验证不覆盖这个新模式。

## 机制

每 warp 领取 32 条边，每 lane 先独立完成最多 K 次短表二分查询。
用原本就需要的 ballot 得到剩余 lane 掩码，gated 额外 popc 得到 r：

| r | 行为 |
|---|---|
| 0 | 本批已完成，进入下一批 |
| 1..R | 建立局部前缀和，warp 协作处理剩余查询 |
| R+1..32 | 不建立池，各线程独立做完自己的剩余查询 |

只判断一次，不反复执行 K 次/投票。进入下一批前当前批次全部完成。
K 用 `--budget`，默认 8；R 用 `--remaining-limit`，默认 16，范围 [0,32]。
R=0 永不启用池；R=32 的数学分配与原 budget 相同，是边界参考，不保证指令或耗时相同。
保留 full-mask shuffle、尾批无效 lane、64 位索引和统一块归约。

规则只看活跃线程数，不额外读取邻居长度或归约剩余工作量。
它可能错过多数线程都忙但极不均衡的批次，也可能给少量很短的剩余任务建池。
这是先验证“空闲线程多时才协作”的简单版本，没有承诺减少 S5 退化或保留已有加速。

## 更新、编译与验证

```bash
git pull --ff-only
cmake -S . -B build
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 tests/check_query_v2.py ./build/tc_gpu_query_v2
```

检查已扩展到 gated：三模式各76图参考计数，另查五模式、尾批/跨步、R=0/1/8/16/31/32、
参数错误和诊断直方图；host 模型检查84图/4536组配置以及逐查询无重复/遗漏。
这些 host/Python 检查不能替代 CUDA 编译、同步/内存检查和服务器实测。

## 初步性能实验

先固定 K=8，比 edge、原 budget 和 gated R=8/16/32，共5配置；14图×3次为210完整样本。
脚本沿用必要预处理、CUDA准备、传输和计数归约的完整计时边界；固定相同blocks和CPU线程。

```bash
mapfile -t INPUTS < <(python3 scripts/prepare_benchmark.py paths --suite core)
if [ "${#INPUTS[@]}" -eq 14 ]; then
  python3 -u scripts/benchmark_query_gate.py --build build \
    --inputs "${INPUTS[@]}" --threads 12 --blocks 864 \
    --budget 8 --remaining-limits 8 16 32 --runs 3 \
    --out results/raw/a100_query_gate_pilot
fi
```

默认 warmup=0/repeat=1，首次结果可与今日探索协议对照。
也可**另开目录**用 `--warmup 5 --repeat 30` 做预热后GPU诊断：主字段 first_total_ms 与
first_count_reduce_ms 仍然是预热前的首次结果；额外字段 count_reduce_ms_median 汇总
每进程预热后重复的中位数，再跨独立进程求中位数，不将30次GPU-only执行当成30个完整样本。
两种计时分开解读，不替换或混算今天的首次计数时间。

## 剩余线程分布

```bash
./build/tc_gpu_query_v2 data/generated/S5.txt \
  --query-mode gated --budget 8 --remaining-limit 16 \
  --blocks 864 --parser omp --mapping omp --sort omp --threads 12 \
  --warmup 0 --repeat 1 --verify --query-diagnostics
```

诊断另跑一轮、不进入正式计时。gated 使用44个64位计数（352B）而非原模式的88B。
增加 `remaining_threads_histogram[0..32]`：索引r表示前缀后还剩r线程的批次数。
应满足：直方图和=batches；hist[0]=fast_batches；sum(hist[1:R+1])=pool_batches。
其余 scalar_batches 表示拒绝协作并独立处理剩余任务的批次。
all_queries=prefix_queries+remaining_queries；pooled_queries≤remaining_queries，
两者之差由各线程独立完成。pass_ms 含统计/回传，不能作为正常性能时间。

## 今日记录

[两轮初步结果与可复算表格](../results/query_pilot_20261123/README.md)。
只有用户回传的聚合数据，逐次原始JSON/输入哈希未收到；不能补造样本或置信区间。
