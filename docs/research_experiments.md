# 下一轮实验：局部协作与全局查询（2026-11-26）

目标是在一般真实图上接近或优于边并行 baseline，在困难图上争取明显收益；不要求所有图都赢。合成图解释机制，真实图是主评测，Orkut 检验规模。暂不引入新评分模型、多 GPU 或逐顶点计数。

## 实现与假设

1. **局部协作**：一个 warp 处理连续 32 条边；每线程先独立完成 K 次二分查询。budget 模式将剩余查询汇入 warp 内虚拟池；gated 模式只在剩余线程数为 1..R 时协作，否则继续独立完成。K=0、R=0、R=32 是有用的边界对照。门控只反映剩余线程数，不能保证准确预测代价。
2. **全局查询**：对每条定向边计算短邻接表长度 q_e，通过 64 位前缀和给每次二分查询一个全局编号。线程通过前缀索引找到所属边，再连续处理 B 次查询，跨边时推进索引。不显式物化全部查询。B=1 是查询级基线，B>1 尝试摊薄所属边定位成本，但可能影响访存合并及负载均衡。
3. **小图绕过**：CPU 已知清洗后边数 m，如果 m<M，直接启动现有一边一线程 kernel，不执行局部判断/全局索引准备。M=0 关闭。阈值是实验候选，边数并不能完整表示查询量或图难度。

所有方法用相同 OpenMP 预处理、CSR、线程块与归约方式。全局版本每次 GPU 执行都重建索引并计时；不会把索引视作免费缓存。新增前缀存储约 8(m+1) 字节，CUB scratch 另计。目前扫描项数受 m+1<=INT_MAX 限制，查询编号/偏移/结果仍为 64 位。Orkut 满足此范围。

新全局 CUDA 源码在提交环境没有 nvcc/GPU；host 查询覆盖、穷举及流程回归通过不等于已经通过 GPU 验证。租机后的第一步必须编译并执行下面的正确性检查。

## 1. 编译、检查与固定环境

在仓库根目录执行，A100 使用 sm_80：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 tests/check_serial.py ./build/tc_gpu_edge_baseline --verify --warmup 0 --repeat 1
python3 tests/check_query_v2.py ./build/tc_gpu_query_v2
python3 tests/check_global_queries.py ./build/tc_gpu_query_global
```

全局检查包括空图、零查询、重复前缀、块尾、B 扫描、绕过开关及阶段计时。任何失败先停止性能实验。

同一 GPU、不并行跑多个实验进程，统一 `--threads 12 --blocks 864`、每块 256 线程。12 是此前租用实例的 CPU 预算；换机器先确认 `nproc` 和 cgroup CPU 配额，若调整必须对本轮所有方法一致。runner 设置 OMP_DYNAMIC=FALSE、OMP_WAIT_POLICY=PASSIVE，并记录线程亲和环境、CPU/GPU 与驱动信息，不自动修改 GPU 时钟。

## 2. 数据准备

```bash
python3 scripts/prepare_benchmark.py download --suite core
python3 scripts/prepare_benchmark.py generate --suite core
python3 scripts/prepare_benchmark.py download --suite extended --ids X1
```

如果下载慢，可本地下载并上传同名解压文件：

- Orkut 官方页面：<https://snap.stanford.edu/data/com-Orkut.html>
- 压缩文件：<https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz>
- 默认路径：`data/com-orkut.ungraph.txt`。

参考规模约 307 万顶点、1.172 亿无向边；清洗后的数目必须以本程序记录为准。基础图 GPU 数组约 0.96 GB，全局前缀额外约 0.94 GB，均未含 scratch/上下文。CPU 文本、边表、映射、并行排序临时缓冲会占用更多内存，不能用显存估算主存需求。建议优先使用此前约 90 GB 主存的实例，仍先测峰值。

先只运行一次 edge，确认清洗、计数和内存都可承受；这是资源预检，不参与挑选参数：

```bash
mkdir -p results/research_v1
/usr/bin/time -v -o results/research_v1/orkut_smoke.resource.txt \
  timeout 30m ./build/tc_gpu_edge_baseline data/com-orkut.ungraph.txt \
  --parser omp --mapping omp --sort omp --threads 12 --blocks 864 \
  --verify --warmup 0 --repeat 1 \
  > results/research_v1/orkut_smoke.json 2> results/research_v1/orkut_smoke.stderr
```

`--verify` 的 CPU 核对在计时区间之外，但消耗实际租机时间。不要在 Orkut 上做全部参数组合扫描。

## 3. 开发集与参数曲线

先用以下代表性开发集，涵盖短表、长尾、均匀大团和混合团：Wiki-Vote、web-Google、cit-Patents、S5、S7。其余已有真实图可随后作为固定参数复测；这些图也已经用于前期探索，不能宣称是从未见过的测试集。

```bash
DEV=(data/Wiki-Vote.txt data/web-Google.txt data/cit-Patents.txt \
     data/generated/S5.txt data/generated/S7.txt)

# 先检查执行清单，不占 GPU
python3 scripts/run_research.py --stage k --inputs "${DEV[@]}" \
  --out results/research_v1/k --dry-run

python3 scripts/run_research.py --stage k --inputs "${DEV[@]}" \
  --out results/research_v1/k
python3 scripts/plot_research.py results/research_v1/k
```

| 阶段 | 固定项 | 扫描项 | 要回答的问题 |
|---|---|---|---|
| `k` | budget 模式，不用剩余线程门控 | K=0,2,4,8,16,32,64 | 先执行多少，能否减少协作准备成本？ |
| `r` | 从 K 曲线选出的统一 K | R=0,4,8,12,16,24,32，另有 budget 对照 | 剩余多少线程时值得协作？ |
| `b` | 每次重建全局索引，无绕过 | B=1,2,4,8,16,32 | 查询定位成本能否被连续查询摊薄？ |
| `bypass` | 统一 K/R/B | M=0,1万,10万,100万条边 | 小输入直接用 baseline 是否值得？ |
| `profile` | 固定 K/R/B | 不扫描 | 索引、查询、归约各占多少？ |
| `final` | 冻结参数 | 所有既有真实图，20 次 | 固定方法的收益、退化与完整耗时如何？ |
| `stress` | 相同冻结参数 | Orkut，先 3 次 | 更大真实图上的规模、内存和收益如何？ |

探索默认每配置 3 个独立进程；每进程先完成首次执行，再 warmup=5、repeat=30 次 GPU 重复。各轮方法/输入顺序用固定 seed 随机打乱。`--runs` 控制完整进程次数，`--repeat` 只重复 GPU 阶段，二者不能混称。

例如 K 曲线之后，先人工选一个折中 K（下面的 8 只是候选）：

```bash
python3 scripts/run_research.py --stage r --k 8 --inputs "${DEV[@]}" \
  --out results/research_v1/r
python3 scripts/run_research.py --stage b --inputs "${DEV[@]}" \
  --out results/research_v1/b
python3 scripts/plot_research.py results/research_v1/r
python3 scripts/plot_research.py results/research_v1/b
```

选择时看跨图曲线的稳定区间、收益和退化的绝对毫秒量，不逐图取最优，不根据 Orkut 成绩倒改参数。3 次探索不能可靠区分几微秒差异；需要时仅在接近的候选之间追加统一次数。

## 4. 小图绕过与阶段诊断

用固定 seed 的嵌套顶点诱导子图补齐小输入尺度，保留原始编号。仅对 Wiki-Vote 这种开发图生成，不对 Orkut 用 Python 大集合处理。

```bash
python3 scripts/make_scale_graphs.py data/Wiki-Vote.txt \
  --out data/generated/wiki_scale
python3 scripts/run_research.py --stage bypass --k 8 --r 16 --b 8 \
  --inputs data/generated/wiki_scale/*.txt data/Wiki-Vote.txt \
  --out results/research_v1/bypass
python3 scripts/plot_research.py results/research_v1/bypass

python3 scripts/run_research.py --stage profile --k 8 --r 16 --b 8 \
  --inputs data/web-Google.txt data/generated/S5.txt data/generated/S7.txt \
  --runs 3 --out results/research_v1/profile
python3 scripts/plot_research.py results/research_v1/profile
```

诱导子图不是对任意真实小图的无偏代表，删除顶点也会改变度数与定向；它只解释规模趋势，不能凭它证明阈值对所有图有效。需要增加其他真实开发图验证阈值。

profile 增加 CUDA event，结果单独保存，不能代替无额外阶段事件的正式计时。堆叠图使用同批运行的阶段**均值**，不相加独立中位数。现有 `light_count_queue_ms` 在全局版本表示索引构建与扫描，`warp_count_ms` 表示查询 kernel，不代表该 kernel 按每边一个 warp 执行。

## 5. 冻结后完整评测与压力测试

复制配置，填写选择依据、开发输入文件名和两个方法的统一参数，最后将 frozen 改成 true。示例不会自动冒充已选择的最优参数：

```bash
cp configs/research_freeze.example.json configs/research_freeze.json
# 编辑 research_freeze.json，保留开发集名单并补全选择依据，然后 frozen=true

REAL=(data/com-amazon.ungraph.txt data/roadNet-CA.txt data/web-Google.txt \
      data/amazon0302.txt data/Wiki-Vote.txt data/soc-Epinions1.txt data/cit-Patents.txt)
python3 scripts/run_research.py --stage final --frozen configs/research_freeze.json \
  --allow-development-overlap --inputs "${REAL[@]}" \
  --out results/research_v1/final
python3 scripts/plot_research.py results/research_v1/final

# Orkut 是留出的压力图。先 3 次、较少 GPU 重复，确认可行后再追加独立目录。
python3 scripts/run_research.py --stage stress --frozen configs/research_freeze.json \
  --inputs data/com-orkut.ungraph.txt --runs 3 --warmup 2 --repeat 5 \
  --timeout 1800 --out results/research_v1/stress
python3 scripts/plot_research.py results/research_v1/stress
```

`--allow-development-overlap` 明确标记开发图复测；真正未参与调参的图不使用此开关。需要正式压力成绩时，在新输出目录统一 `--runs 20 --warmup 5 --repeat 30`，如果成本太高可统一减少重复并单列协议，不能拼接不同协议的样本。

## 指标、保存与绘图

- **主指标** `first_total_ms`：读入前开始，经过预处理、任务元数据、CUDA 初始化与分配、H2D、首次 GPU 工作及 D2H 后结束；不包含后续 warmup/repeat 和正确性参考运行。加速比=同轮 edge 的完整时间中位数 / 候选方法中位数。
- **次指标** `first_count_reduce_ms`：CUDA event 测量首次 GPU 完整工作。全局查询包括长度 kernel、前缀扫描、查询 kernel 和归约；局部版本包括执行/判断/协作及归约。按同一边界计算 GPU 加速比。
- **补充指标** `count_reduce_ms_median`：每个进程预热后 GPU 重复的中位数，再跨独立进程取中位数。全局版本仍每次重建索引。它与首次执行分别展示，不能混为一次完整任务加速。
- 内存：JSON 的 `device_buffer_bytes` 是显式申请的缓冲字节，不是 CUDA 上下文/驱动总峰值；GNU time 的 Maximum resident set size 是进程 CPU 主存峰值，单位依 Linux 工具说明。原始 resource 文件保留。

每次实验目录包含：

| 文件 | 内容 |
|---|---|
| `metadata.json` | 精确命令顺序、参数、seed、源码/归档版本、二进制/输入/冻结配置 SHA256、缓存及环境说明 |
| `*.json` | 每次进程完整原始结果，包括 GPU 样本数组、图规模、计数、显存申请量、verify |
| `*.stderr`, `*.resource.txt` | 错误日志、GNU time 资源记录；超时会停止同组子进程，保留失败记录 |
| `status.json`, `summary.json` | 运行状态和分方法汇总，失败立即停止，不忽略错误继续拼完整成绩 |
| `nvidia-smi.txt`, `cpu.txt`, `memory.txt` | 环境快照 |
| `figures/plot_data.csv` | 曲线/柱图的全部精确数值，可重新用 LaTeX/PGFPlots 绘图 |
| `figures/*.png`, `*.pdf` | K/R/B/M 参数曲线，正式/压力测试加速比柱图，三个计时口径分别绘制 |
| `figures/phase_samples.csv`, `gpu_phase_means.*` | profile 阶段样本和索引/计数/归约均值堆叠图 |
| `figures/overview.json` | 每种固定配置的快于 baseline 图数、几何平均加速比、最大收益/退化毫秒量；只是点估计 |
| `figures/provenance.json` | 绘图输入哈希与口径说明 |

runner 不清操作系统文件缓存，输入哈希会预读文件，所以这不是冷磁盘测试。所有方法采用同样策略；缓存和 CUDA 初始化对完整时间的影响需要在论文中说明。先保存绝对耗时、退化量及内存，再解释加速比，不仅展示最好看的比值。

绘图只需 `python3 -m pip install matplotlib`；使用无桌面的 Agg 后端。建议实验后台运行并备份整个目录：

```bash
# 将选定的 python3 scripts/run_research.py ... 命令置于 nohup 后即可
# nohup python3 scripts/run_research.py ... > results/research_v1/run.log 2>&1 &
tar -czf research_v1_results.tar.gz results/research_v1 configs/research_freeze.json
```

释放实例前把该压缩包下载到本地。大型 SNAP 数据和运行产物不自动上传 GitHub；仓库提供下载清单、构造脚本与结果格式。正式结论以回收的原始 JSON 为依据，不能由探索聚合表补造样本。
