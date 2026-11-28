# 中期海报冻结评测 v1

本轮主线 5 个方案，另加一个简单队列分流参考。所有 14 图，各方案独立完整进程 10 次；每进程先产生首次完整结果，额外预热 5 次，再记录 30 次 GPU 计数与归约。GPU 主指标：先对 30 次取中位数，再对 10 个进程取中位数，同时报告进程间四分位区间。首次 GPU 时间和完整时间单列；原始样本、进程均值仍保留。

| 方案 | 固定规则 |
|---|---|
| edge | 原每边一线程二分 baseline |
| pool | 32 条边的查询放入批内虚拟池，无前缀预算 |
| reuse | pool 上复用共同长表范围，保持二分 |
| gate_reuse | m < 262144 直接启动原 baseline kernel；否则批内所有查询数 <=8 时线程独立完成，其余 reuse |
| gate_hash | 相同图级/短批门控；其余批次共用长表、64<=b<=512、Q>=8b 时启用 shared hash，否则 reuse |
| split_gt32 | 查询次数 min(|F(u)|,|F(v)|)>32 入队，第二个 kernel 每边一个 warp；含清零、入队、双 kernel 和归约成本 |

这里查询次数表示当前算法必须处理的短表元素数，不是实际二分比较次数。所有定向/CSR/图清洗一致；gated 两版相对旧探索新增相同图级绕过，方便控制 hash 的增量效果。固定 1024 槽/warp，hash kernel 的共享内存预留会影响未建表批次的资源占用，此成本不排除。

实验每轮打乱图顺序，每张图内部打乱方案顺序，无逐图挑参数。连续记录 GPU 快照以辅助解释频率/负载变化，但快照本身不能证明没有外部干扰。输入哈希提前读取图文件，因此属于不清理文件缓存的测量，不声称冷盘 I/O 性能。

运行：

```bash
cmake --build build --target tc_gpu_edge_baseline tc_gpu_query_v2 tc_gpu_split -j 4
python3 tests/check_poster.py ./build/tc_gpu_query_v2
python3 scripts/benchmark_poster.py --build build --out results/poster_v1 --runs 10 --threads 12 --blocks 864 --warmup 5 --repeat 30
```

默认六版共 840 个独立进程。`--without-split` 只跑五版，共700进程。输出 metadata.json（数据/源码/二进制哈希、冻结规则、CPU/GPU环境）、commands.json、每次 stdout JSON/stderr、status.json、summary.json 和 summary.csv。运行中不要修改源码或重新编译。

代码本轮按用户要求检查后交付，不在本地编译/跑测试；服务器检查通过后开始正式评测。当前没有这版组合方案的性能结论。
