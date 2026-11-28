# gated-reuse 主路径 + 独立 hash 队列

本次在用户上传且已实测的 poster_patch_v1 上增加 `tc_gpu_query_v2 --query-mode hash-queue`。旧版 `reuse-gated`、融合版 `hash-gated` 及六方案评测保留。

## 执行规则

1. 预处理后无向简单图边数 m<262144：直接启动原始count_edges kernel。不分配batch队列、没有hash阶段，沿用原baseline归约。
2. 其余图先启动不预留hash表的gated-reuse producer。各warp领取连续32条CSR边；短表长度即该边的查询次数。全批<=8时线程独立完成。
3. 其他批次检查实际长表的CSR范围。共同长表、64<=b<=512、总查询数Q>=8b时，lane0把批次起始边编号入队，整个warp本阶段返回0。其他任务执行原reuse/虚拟查询池。
4. 同一CUDA stream启动独立consumer，只处理队列中的批次。每个warp重读自己的32条边，合作重建1024槽shared-memory哈希表并查询。普通producer不承担consumer的32KiB哈希表预留。
5. CUB归约两阶段的block sums，返回精确总数。每个batch只由一个阶段完成，没有跨阶段重复计数。

阈值仍为固定经验值，未声称成本最优。这次不同时改阈值，也未实现全图持久哈希或跨批次共享同一张表。consumer重读任务并再次确认共同目标的成本计入时间。

## 显存与资源

队列每批一个64位起始编号，容量ceil(m/32)，另加8字节计数器。非空图约m/4字节，不显式展开Q个查询。producer每个batch只访问一次，最多入队一次，因此容量覆盖最坏情况。队列每次run（包括首次、预热、每个计时样本）重新清零和构建。空队列仍启动consumer，并写零partial sums，这笔启动和双partial归约成本照常计入。

`hash_queue_buffer_bytes`报告实际分配，含最小分配和计数器；`device_buffer_bytes`还含CSR、sources、部分和及CUB scratch，不是峰值显存。`queue_primary_static_shared_bytes`、`queue_consumer_static_shared_bytes`和寄存器数来自cudaFuncGetAttributes。producer仍有CUB归约storage和极小的模板占位，不能称为零共享内存，但没有1024×8槽hash预留。

## 服务器：编译与正确性

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target tc_gpu_edge_baseline tc_gpu_query_v2 -j 4
python3 tests/check_hash_queue.py ./build/tc_gpu_query_v2
```

检查包含76个独立小图oracle、错误输入、精确批次归属、空队列、零查询、部分warp、grid-stride、长表容量边界、碰撞、重复执行及小图baseline绕过。代码按用户要求检查后上传，未执行本地编译/测试；服务器检查是新增CUDA正确性与性能验证，不把历史poster正确性当作新代码已验证。

## 全14图统一评测

保留已有data目录即可。数据下载/生成见data/benchmark.txt及prepare_benchmark.py。

```bash
mkdir -p results
nohup python3 -u scripts/benchmark_hash_queue.py \
  --build build --out results/hash_queue_v1 \
  --runs 10 --threads 12 --blocks 864 --warmup 5 --repeat 30 \
  > results/hash_queue_v1.log 2>&1 &
tail -f results/hash_queue_v1.log
```

默认baseline、gated-reuse、新hash-queue三版，14×3×10=420独立完整进程。`--include-fused`增加旧hash-gated，变为560进程，可直接评估资源分离的增量效果。不默认重跑pool等历史版本。可用`--inputs data/...txt ...`限制输入，但默认是全部14图；本轮不扫参数。

顺序随机交错，同图同轮baseline比较。预处理、数据传输、初始化和队列分配包括在first_total_ms；GPU计时包括队列清零、producer路由/计数、consumer建表/计数以及全部归约。队列大小不在计时路径读回CPU。GPU主指标固定为进程内30次中位数，再取10进程中位数；同时输出IQR、均值和首次完整时间。源码、二进制、输入哈希和CPU/GPU状态保存。

## 独立诊断

正式计时结束后脚本自动每图多跑一个profiler/diagnostic进程，保存在diagnostics/，不混入正式统计。包含：入队批次数、共同目标数、实际hash查询量/建表元素数、显存分配和两kernel资源信息。各阶段profile时间可用于解释开销，但来自独立插event的运行，不冒充正式样本的精确拆分。

`reuse_route_ms_samples`含清零和producer；`queued_hash_ms_samples`含consumer；`global_reduce_ms_samples`含全局归约。小图绕过时route字段记录baseline计数，hash为0，标明diagnostics_skipped_for_bypass，不执行被绕过候选的诊断。其他历史模式仍保留其原诊断语义。

若主评测结束而诊断中断，可独立补跑（diagnostics目录必须尚不存在，已有部分结果请先改名保留）：

```bash
python3 scripts/hash_queue_diagnostics.py --results results/hash_queue_v1
```

打包时包括results/hash_queue_v1及同名.log，并保存git commit。若用新服务器，硬件/驱动环境变化可能影响绝对时间，不直接用旧poster时间作为本轮分母。
