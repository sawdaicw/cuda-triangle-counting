# 预处理分项计时：先定位，再并行优化

更新：本页介绍最初的计时功能；现在已有可选 OpenMP 优化及输入细分计时，见 [OpenMP 预处理指南](openmp_preprocessing.md)。不加新选项仍走原版，原阶段定义保持。

本次增加诊断计时，**没有实现 OpenMP/GPU 并行预处理，也没有修改交集算法、边顺序或线程任务分配**。

## 1. 更新与重新编译

在实验室服务器项目根目录执行：

```bash
git pull --ff-only
cmake --build build-lab -j 4
ctest --test-dir build-lab --output-on-failure
```

如果 `build-lab` 尚未配置，使用实验室已验证的 CUDA 12.9 / g++-13：

```bash
cmake -S . -B build-lab \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-lab -j 4
```

## 2. 推荐先用 CPU 入口诊断，不需要占用 GPU

CPU 串行和 GPU 基线调用相同的 `read_graph`。为了看预处理，可以直接运行 CPU 入口，避免 CUDA 初始化和 GPU 争用：

```bash
mkdir -p results/raw/preprocess_profile
./build-lab/tc_cpu_serial data/com-amazon.ungraph.txt --profile-preprocess > results/raw/preprocess_profile/amazon_cpu.json
./build-lab/tc_cpu_serial data/roadNet-CA.txt --profile-preprocess > results/raw/preprocess_profile/roadNet-CA_cpu.json
./build-lab/tc_cpu_serial data/web-Google.txt --profile-preprocess > results/raw/preprocess_profile/web-Google_cpu.json
```

确认各命令成功后查看：

```bash
python3 -m json.tool results/raw/preprocess_profile/amazon_cpu.json
python3 -m json.tool results/raw/preprocess_profile/roadNet-CA_cpu.json
python3 -m json.tool results/raw/preprocess_profile/web-Google_cpu.json
```

CPU 入口仍会在预处理后执行串行计数，并输出 `count_ms`。不要将 CPU 的 `count_ms` 当作 GPU 计数时间；本步骤只用其 `preprocess_ms` 和 `preprocess_profile` 定位热点。

先跑每图一次初步观察，必要时独立重复几次检查热点是否稳定。没有设置文件缓存清理；缓存状态、CPU 竞争及主存带宽会影响测量，不能把重复读取的时间说成冷磁盘读取时间。

## 3. 可选：在 GPU 基线里同时记录

```bash
CUDA_VISIBLE_DEVICES=0 ./build-lab/tc_gpu_edge_baseline \
  data/com-amazon.ungraph.txt --verify --warmup 0 --repeat 1 \
  --profile-preprocess > results/raw/preprocess_profile/amazon_gpu.json
```

原有字段和 `first_total_ms` 的起止位置不变，仅额外输出 `preprocess_profile` 对象。阶段计时发生在首次完整流程中，诊断开销也会包含在它的总时间内。

该命令仍是首次 GPU 运行加 1 次额外计时运行，不是两次完整预处理；`first_total_ms` 在额外运行前已经结束。此次未修改 `--repeat` 的原有语义。

## 4. 阶段字段

单位均为毫秒。阶段按原有程序的实际执行顺序划分，**没有把原来融合的循环拆成新的多遍扫描**。

| 字段 | 计时范围 |
|---|---|
| `read_parse_normalize_ms` | 打开文件、逐行读取和整数解析、收集端点、排除自环边、将每条边规范成 `(min,max)`；此时还没合并重边 |
| `vertex_sort_unique_ms` | 原始顶点编号排序、去重、检查内部编号范围 |
| `edge_sort_unique_ms` | 规范化边排序、合并重复边 |
| `remap_degree_ms` | 分配稠密边和度数数组、两端点二分映射、同时累加无向度数；这两项在原代码中融合，因此合并计时 |
| `orient_row_counts_ms` | 分配 offsets、按 `(degree,id)` 定向，并统计每行向前边数 |
| `csr_scan_ms` | 对行长度执行串行前缀和，得到 offsets |
| `csr_fill_ms` | 分配 neighbors、复制填充游标、填充 CSR 邻居数组 |
| `neighbor_sort_ms` | 对各顶点的向前邻居段排序 |
| `accounted_ms` | 上述 8 个阶段耗时之和 |
| `unattributed_ms` | 外层 `preprocess_ms` 减去阶段和，包括函数边界、局部对象清理等未单列成本；不是额外加在 preprocess_ms 之外的时间 |

满足近似关系（JSON 小数舍入可能产生微小误差）：

```text
preprocess_ms = accounted_ms + unattributed_ms
```

`task_metadata_ms` 是构建边源点任务数组的时间，仍在 GPU 基线顶层单独输出，不属于 `read_graph` 的这 8 个阶段。`cuda_setup_ms`、传输、计数和归约也不属于此对象。GPU 完整总时间继续包括这些必要成本。

## 5. 如何使用结果

- 先看各阶段占 `preprocess_ms` 的比例，定位主要热点。
- 如果 `remap_degree_ms` 最高，后续再细分或分析映射与度数累加；当前不能把它全部归因于其中一项。
- 如果文件解析最高，应区分快速解析实现和多线程解析的收益。
- 如果排序最高，再考虑并行排序/去重或 GPU Thrust/CUB 路线。
- 如果 CSR 阶段占比很小，不必优先为它引入复杂并行结构。

这是诊断功能。**正式主加速比仍使用完整流程中位数，不把分项之和当作新的分母，也不能只比较最快阶段。** 正式基线/优化对照时应统一开启或关闭诊断；建议关闭诊断，以避免计时插桩对微小收益的干扰。默认不开启分项计时时，不调用这些阶段时钟、不增加阶段 JSON。

此前 20 次基础并行版的完整时间中位数（用户在 A800 上运行的记录）为 Amazon 931.228 ms、roadNet-CA 3209.273 ms、web-Google 3634.905 ms。这些是历史版本测量，不应冒充本次诊断版本的新结果；原始 JSON、提交号和环境记录继续保留。

## 6. 验证

```bash
python3 tests/check_preprocess_profile.py ./build-lab/tc_cpu_serial
python3 tests/check_serial.py ./build-lab/tc_cpu_serial --profile-preprocess
```

前者比较开启/关闭诊断的 40 个图（课程案例、随机图和稀疏负数 ID），检查精确结果、字段非负有限以及阶段时间核算；后者在诊断开启时复用 76 图与非法输入测试。CTest 另外核对开启/关闭诊断得到的 CSR 数组完全一致，并验证计时对象重用不累计旧值。

可选在 GPU 上检查新增参数和输出：

```bash
CUDA_VISIBLE_DEVICES=0 python3 tests/check_preprocess_profile.py \
  ./build-lab/tc_gpu_edge_baseline --verify --warmup 0 --repeat 1
```

GPU 测试需要逐项初始化 CUDA，运行时间会比 CPU 诊断测试长。没有 GPU 的 CI 只能检查 CPU 运行及 CUDA 编译，不能证明设备执行正确。
