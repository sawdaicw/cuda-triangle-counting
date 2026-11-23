# 查询池 v2：先执行后协作，以及成本评分后协作

本轮目标：普通任务接近原 baseline，困难任务获得收益；不要求每图都更快。
本版是新 CUDA 实验源码，**当前编辑环境无 nvcc/GPU，尚未编译 CUDA 或测量 v2 性能**。
用户此前已验证的是 v1，不能将该结果当作 v2 正确性或性能证据。

## 两条研究路线

统一保留原无向清洗、(度数,编号)定向、CSR 和短表元素在长表二分的查询集合。
每 warp 每轮领取 32 条连续边，以 blocks×256 跨步；各 lane 先读取本边两个表范围。
计数为 64 位，块内/全局 CUB 归约与原 runner 相同。

### budget：先执行，再协作

每 lane 先执行 min(a_i,K) 次二分查询，然后将短表起点前移该数量。
所有 lane 完成这段工作后，用一次 ballot 判断是否还存在未完成查询。
全完成则进入下一批；否则只对剩余段做局部前缀和，由 32 lane 平铺分担查询。
不做难度评分，不构建全局队列；pool 中的 query index 从剩余短表起点计算。

K 由 `--budget` 指定，默认 8，首轮探索 4/8/16，也可设 10。
K=0 是边界检查/实验入口；K 很大则可能全部在独立阶段完成。
二分次数相同不等于耗时相同；进入协作前仍需等待慢 lane。
均匀重任务也会进入 pool，可能因额外开销而退化。本轮不叠加新门控。

### score：先比较两条路径的预计成本

若所有短表长度 a_i≤small，直接一边一线程。否则设
l_i=bit_width(b_i)，空短表对应有效深度 0，Q=sum(a_i)，计算：

```text
edge_cost = max(a_i * l_i)
pool_cost = setup + ceil(Q/32) * (locate + max_effective_depth)
启用 pool 当且仅当：100*edge_cost > (100+margin_percent)*pool_cost
```

默认 small=4、setup=16、locate=8、margin_percent=20。
setup/locate 是比较工作量代理，**不是 GPU 周期/纳秒，默认值尚未校准**。
最大深度项较保守；edge_cost 也未精确描述 warp 访存和分支运行时间，因此这是待检验的启发式。
与 v1 的 max-minus-mean 门控不同，本版显式比较两条执行路径的预计成本。
均匀重任务通常不会通过成本门控，少量重边混合轻边时可能通过，但不保证实际净收益。

参数范围 small/setup/locate/budget 为 [0,65535]，margin_percent 为 [0,1000]。
有效简单图的 32 位点号约束保证局部 query 前缀、分数和百分比比较乘积可放入 64 位。
全图诊断累计和、三角形总数仍须可由 64 位容纳。

## 编译期独立内核与共用查询池

新增 `tc_gpu_query_v2`，由 `TC_EDGE_MODE=4` 构建，原 `tc_gpu_query` v1 保留。
`--query-mode budget|score` 是两条主路线；scalar/pool 是可选基础对照。
模式在 host dispatch，device kernel 使用模板 Mode 分别编译，scalar 不携带 pool/评分分支。
两条路线调用相同的虚拟查询池 helper；没有显式 query 数组，没有 O(m)/O(Q) 新持久缓冲。
模板分开不保证寄存器更少或实际更快，应查看编译报告/设备实测。

所有 lane（含尾批无效边）参加 full-mask warp 操作。查询池最后不足 32 项的一轮，
无效 query lane 仍执行全部 metadata shuffle，但不访问 neighbors；定位前缀 owner 使用
固定六轮 upper_bound，跳过零长度段。所有 block 线程最终统一到达 BlockReduce。

## 先在服务器验证

已有 build 使用缓存中的 CUDA 路径/架构，重新配置以生成新 target：

```bash
git pull --ff-only
cmake -S . -B build
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 tests/check_query_v2.py ./build/tc_gpu_query_v2
```

全新环境的完整配置：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4
```

check_query_v2 对 budget/score 各跑 76 图/3 非法输入；另查四模式、尾批、32/256 边界、
跨步、K=0/1/4/8/10/16/65535、极端评分参数、空图/大团/混合图、诊断独立参考、重复运行和 CSR。
它是待执行的 GPU 检查，不包含在这里的 host 测试完成声明中。
可选进一步检查同步和越界（需要已准备数据）：

```bash
compute-sanitizer --tool memcheck ./build/tc_gpu_query_v2 tests/triangle.txt --query-mode pool --warmup 0 --repeat 1 --verify
compute-sanitizer --tool synccheck ./build/tc_gpu_query_v2 data/generated/S1.txt --query-mode budget --budget 4 --warmup 0 --repeat 1 --verify
```

若要研究编译资源，可在另一 build 目录配置 `-DCMAKE_CUDA_FLAGS=-Xptxas=-v`，保留 stderr 中
各模板内核的寄存器和 spill 报告。不要只根据源代码长度推断资源开销。

## 首轮评测（不展开复杂消融）

脚本默认比较 edge、budget4、budget8、budget16、score 五种配置。
每张图每种配置 3 次独立完整进程；14 图共 210 次。不开 profile/diagnostics，
统一预处理、blocks、固定评分参数，逐轮随机顺序，保留原始 JSON、哈希、版本、命令与 summary。
`--include-controls` 才加入编译期 scalar/pool；默认不加。旧评分版通过旧脚本单独保留。

```bash
mapfile -t INPUTS < <(python3 scripts/prepare_benchmark.py paths --suite core)
if [ "${#INPUTS[@]}" -eq 14 ]; then
  python3 -u scripts/benchmark_query_v2.py --build build \
    --inputs "${INPUTS[@]}" --threads 12 --blocks 864 \
    --budgets 4 8 16 --runs 3 --out results/raw/a100_query_v2_pilot
fi
```

若只想快速检查两条路线，设 `--budgets 8`。每轮统一参数，不按图名挑选，不自动运行后选冠军。
正式规则在开发集确定后冻结，再对保留图评估，必要时增至 20 次独立完整运行。
主指标 `first_total_ms` 含必要预处理/元数据/CUDA准备/H2D/计数归约/首次D2H；
辅指标 `first_count_reduce_ms` 含 budget 前缀或 score 判断、pool scan/owner定位以及 CUB 归约。
首次时间、额外 GPU-only 重复和诊断时间分开；不把分项中位数相加。

## 查看协作覆盖范围

```bash
./build/tc_gpu_query_v2 data/generated/S7.txt \
  --query-mode budget --budget 8 --blocks 864 \
  --parser omp --mapping omp --sort omp --threads 12 \
  --verify --warmup 0 --repeat 1 --query-diagnostics
```

诊断在首次完整执行和 GPU-only 重复之后，另跑带统计的实例并重新归约核对计数。
88 字节统计缓冲单列；计时 kernel 不带统计 atomics。
`all_queries = prefix_queries + remaining_queries`；budget 下
`pooled_queries = remaining_queries`。pool 模式 prefix=0；score/scalar 模式 prefix=0。
`fast_batches` 在 budget 中指全部在前缀阶段完成的批次，在 score 中指 all-short 快速路径。
`scalar_batches` 不含 fast；三类批次之和等于 batches。
`pass_ms` 包含额外诊断、分配/回传等，不能拿来比较正常计数性能。

## 本次已完成与未完成

- g++ host 模型：84 图、1008 组模式/预算/评分参数，独立三元组计数和逐查询覆盖均通过。
- host 侧成本边界、64 位范围、prefix+remaining 恒等式及参数边界检查。
- Python 评测编排测试使用明确的假设备，仅证明脚本流程，不是 GPU 数据。
- 五种 TC_EDGE_MODE 的预处理隔离检查，避免旧后端看到新内核名称。
- 既有 host/Python 回归检查。
- **这里没有 CUDA 编译器/GPU；v2 CUDA 编译、GPU 正确性、sanitizer 和性能仍待服务器验证。**
