> 当前新一轮源码见 [query v2](query_v2.md)：先执行 K 次后协作、成本评分后协作；尚待 GPU 验证。下文保留 v1/早期设计的历史状态。

# 自适应 warp 查询池 v1：待 GPU 验证的实验实现

目标是普通任务接近 baseline、困难任务获得收益；不是每张图都必须更快。
本次加入 CUDA 源码、host 策略/映射检查、GPU 验证入口和独立进程评测脚本。
当前编辑环境没有 nvcc、CMake 或可用 GPU。用户已在 A100 成功编译链接 `tc_gpu_query`；本次修复其他三个后端对 query 诊断调用的条件编译隔离，需重新构建全部目标。**尚无新 GPU 正确性和性能结果**。
现有 A100 v1 历史结果不变。请先完成下述 GPU 验证，再做性能测试。

## 1. 算法与两条执行路径

保持现有无向简单图规范化、(度数,编号)定向、CSR 顺序、短表遍历/长表二分不变。
每 warp 每轮领取连续 32 条定向边；网格跨步为 blocks×256，与 edge baseline
的边分配一致。无效尾 lane 使用空表，不提前退出 warp 集体操作。

1. 每 lane 读取一条边，得到短/长表范围 a、b。
2. adaptive 下用一次 ballot 检查是否所有 a≤small；满足则直接一边一线程。
3. 否则计算 S=a×bit_width(b)，并做 warp 内 sum/max。
4. 当 32×max(S) > sum(S)+32×[overhead+margin+per_round×ceil(Q/32)] 时启用查询池。
   Q=sum(a)。这是固定整数启发式，单位是比较工作量代理，**不是 GPU 周期或纳秒**。
5. scalar 路径各 lane 自行完成该边；pool 路径做 32-lane inclusive scan 获得前缀终点。
6. lane r 处理 q=r,r+32,...；对前缀终点做 upper_bound 找到 owner 边和局部查询索引，
   shuffle 获取该边范围，执行原二分。重复终点（零查询边）自然被 upper_bound 跳过。
7. 所有 block 线程统一到达 CUB BlockReduce，再沿用 DeviceReduce 和 64 位总数。

pool 路径采用 warp-uniform 查询轮数；最后一轮不足 32 个查询时，全部 lane 仍执行
所有 metadata shuffle，只有有效 q 才访问 neighbors。owner 搜索固定六轮，以覆盖
32 项上界查找；所有 lane 均以 full mask 参与，避免 Volta+ 独立线程调度下错误使用掩码。
局部前缀、query 编号、score 与计数均为 64 位。简单图的 32 位点号约束使每行长度
最多 2^32−1；在 CLI 参数 ≤65535 时，单批 score 和决策乘积不会溢出 64 位。
诊断全图累计和与三角形总数仍须可由 64 位容纳。

没有全局重边队列、没有显式查询表、不增加按边/查询数增长的持久显存。
更细分会增加 scan、owner 定位、shuffle 和寄存器需求，不能据此假设性能更优。
批内均衡但批间不均衡的问题没有在本版解决。

## 2. 实验模式与暂定参数

新增可执行文件 `tc_gpu_query`，共用原 runner 与预处理。三种模式：

| 参数 | 用途 |
|---|---|
| `--query-mode scalar` | 不评分、不启用查询池；相同批次循环的结构对照，并非原 baseline 的替代 |
| `--query-mode pool` | 所有批次使用虚拟查询池，包括零查询批次；测分配收益与固定开销 |
| `--query-mode adaptive` | 默认，短表快速路径 + 工作量差异/成本门控 |

默认 small=4、overhead=64、per_round=2、margin=32，版本标记
`warp-query-v1-uncalibrated`。**这些是可运行的暂定参数，没有经 GPU 校准，不能称为最优规则。**
可分别用 `--small`、`--pool-overhead`、`--pool-per-round`、`--pool-margin` 修改，范围 [0,65535]。
CPU 短/长表数学模型不预测实际 GPU 加速比；评分忽略缓存、访存合并、不同二分路径等因素。
参数在开发集校准后冻结，以未参与调参的图验证，不能每图跑完各版本再事后选最快者。
不重排边、不改变定向；退化度排序/2-core/桥分解属于未来单独消融。

## 3. 编译与 GPU 正确性（服务器执行）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 tests/check_query.py ./build/tc_gpu_query
```

实验室 nvcc/host 编译器路径按实际替换。check_query 针对三模式各跑 76 图与 3 种非法输入，
另查空图、32/256 边界和尾批、跨步、K18/K40/K67、混合图、重复计数、策略参数强制分支、
独立 Python 诊断统计、GPU 阶段计时与 OpenMP CSR 等价。host 测试不能替代这些 CUDA 检查。

可进一步在 GPU 上运行（这些也是待执行项目）：

```bash
compute-sanitizer --tool memcheck ./build/tc_gpu_query tests/triangle.txt --query-mode pool --warmup 0 --repeat 1 --verify
compute-sanitizer --tool synccheck ./build/tc_gpu_query data/generated/S6.txt --query-mode adaptive --small 0 --pool-overhead 0 --pool-per-round 0 --pool-margin 0 --warmup 0 --repeat 1 --verify
```

## 4. 诊断与计时

```bash
./build/tc_gpu_query data/generated/S6.txt \
  --parser omp --mapping omp --sort omp --threads 12 \
  --blocks 864 --query-mode adaptive --warmup 0 --repeat 1 \
  --verify --query-diagnostics --profile-gpu
```

`--query-diagnostics` 在首次完整执行和所有 GPU-only 重复之后另跑一轮，记录：
总批次、快速路径批次、其他 scalar 批次、pool 批次、非空任务数、池内 query 数、
全图 query 数、score 总和、各批最大 score 之和。与独立参考统计对照。
诊断轮重新归约并检查计数一致；它有 atomic 统计，**不用于性能比较**。
72 字节计数缓冲在测量之后才分配；单独列入 `extra_device_buffer_bytes`，
不包含在 timed-pass 的 `device_buffer_bytes`。诊断实际时间记录为 pass_ms。
正式 kernel 用模板去掉统计原子操作。

主指标仍是 `first_total_ms`（读文本前至首次结果回传），辅指标为
`first_count_reduce_ms`（CUDA Event，计数含评分/scan/定位 + 全局归约，不含 D2H）。
没有隐藏正常调度开销。`--profile-gpu` 用额外 events 分开 count 和 reduce；历史
`warp_count_ms` 字段在 query 模式表示整个 query kernel，另提供明确的
`first_query_count_ms` 和 `query_count_ms_samples` 别名。正式比较不开 profile/diagnostics。
`--repeat 1 --warmup 0` 仍有一次额外 GPU-only 执行；它不是第二次完整样本。

## 5. 独立完整运行对照

A100/A800 SM80 示例，blocks=864 与现有 A100 实验相同。其他设备自行选择一个共同值，
不要对每种方法分别选择最有利的 blocks。线程数也按 CPU 配额统一设置。

```bash
python3 scripts/benchmark_query.py --build build \
  --inputs data/com-amazon.ungraph.txt data/roadNet-CA.txt data/web-Google.txt \
           data/generated/S2.txt data/generated/S5.txt data/generated/S6.txt data/generated/S7.txt \
  --threads 12 --blocks 864 --runs 3 --out results/raw/query_v1_smoke
```

先小规模探索，正确性和规则确定后，将 runs 改为 20，换新的输出目录。
脚本比较原 edge、query scalar、强制 pool、adaptive，逐轮固定种子打乱顺序；固定
policy、blocks 和预处理，记录源码版本、输入/二进制 SHA256、命令和环境，检查图/计数一致，
同时汇总完整时间和 GPU 时间，报告相对 edge 的变化百分比。它不会自动搜参或选冠军。
文件缓存不清空，哈希输入会预读；这不是冷盘 I/O 测试。OMP_DYNAMIC=FALSE、OMP_WAIT_POLICY=PASSIVE。

关注道路/网格的退化幅度、大团上的固定开销、混合结构是否获益，并保留无收益结果。
后续可将 Orkut/Friendster 作为保留图，但先检查 CPU 峰值内存和预处理承载能力。

## 6. 本次实际验证范围

- g++ C++17 host 策略/边界/scan/映射检查；544 组查询映射、大于 32 位的前缀边界，82 图 × 9 模式/参数组合对照独立三元组枚举。
- Python 独立参考与查询映射模型；新评测脚本用明确的假设备测试编排和汇总，非 GPU 数据。
- 原 CPU 计数和预处理回归检查。
- 增加四种 TC_EDGE_MODE 的 host 预处理回归检查，确认旧后端完全排除 query 内核名称；这不是 CUDA 编译测试。
- **全部 CUDA 目标重编译、GPU 正确性、sanitizer 和性能测量尚待服务器执行；没有承诺 5%/10% 退化上限或任何加速比。**
