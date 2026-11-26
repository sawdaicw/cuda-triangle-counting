# 自动任务选择原型 v1（2026-11-26）

目标：低工作量图尽量绕过准备机制，均衡图保留 baseline，只有足够大的可利用不均衡才启用重排或分段。不是保证每图都优于 baseline。

当前代码提供两个 CUDA runner、36 图生成器、单进程驻留标定、分组拟合/冻结与独立进程评测。仓库没有虚构的 A100 模型文件；必须在目标 GPU 上实测生成。新增 CUDA 在提交环境无 nvcc/GPU，host 语义及逻辑检查不能代替下面的服务器验证。

## 三个候选和两个入口

| 候选 | 实现 | 每次执行计时包括 |
|---|---|---|
| baseline | 自然 CSR 边顺序，一边一线程 | 二分查询、块内/全局归约 |
| bucket | 64 桶的工作量直方图、前缀扫描、原子散射边编号，再一边一线程 | 清零、直方图、scan、散射、计数、归约；不改 CSR |
| segment | 准确查询数与表长参与边级成本门控；只拆值得拆的边 | 清零、门控分类与普通边计数、段任务写入、段计数、归约 |

- `tc_schedule_calibrate`：一进程一图，预处理/H2D 一次；每轮随机交错测试 baseline、bucket、segment 的三套门控参数。三套门控是标定候选，不是最终部署时轮流试跑。
- `tc_schedule_auto`：可强制执行某候选，用于核对与评价；`--strategy auto --model FILE` 时只执行预测选中的候选，不现场跑完三种再选择。

原有 `tc_gpu_edge_baseline`、warp、split、gated、global-query 均保留。新 baseline 使用相同的短表二分计数与 grid-stride 思路；完整程序的初始化/事件等有新 runner 开销。可在同一卡上额外与旧 baseline 做核对，不能假设新旧程序完整耗时完全相同。

## 分段、尾段和队列

固定初始段长 L=128，队列预算默认64MiB。q 是准确查询次数，d=ceil(log2(b+1)) 是二分深度代理，找到目标提前退出和缓存效应不由该代理精确表达。

比较 q*d 与 `kappa*最大段长度*d + 边开销 + 每段开销*段数`，并留余量。三组代理单位系数由标定实测选择，不称为拟合出的逐边纳秒/周期。当前还不是连续拟合每条边的真实成本。

段数为 max(1,floor(q/L))，末段合并余数，通常长度在[L,2L)；q=L+1 不拆。每条段任务16字节：边号、短表内起始位置。段队列受预算限制；原子预留跨过容量时，只将可容纳的段写入，其余由原线程完成，不丢失或重复计数。每次运行清零计数器、重新建队列，不复用免费准备结果。

该边级模型并不假设拆成s段就真实快s倍。是否在整张图启用segment，还受图级实测成本模型控制。

## 两级选择与模型

1. 已有边数的粗门控。阈值来自合成验证集。
2. 最多256条固定种子的分层边抽样，在CPU已有CSR上读准确q/b，预测轻任务；抽样可能漏掉极少数重边，不保证完美识别。
3. 未绕过时，在GPU扫描CSR长度，计算准确总查询Q、工作量代理总和、批次最大值之和U、逻辑warp累积工作的最大值M，以及各门控参数下的段数。特征缓冲只在需要时申请。
4. 分别预测三候选成本；只有预测节省超过实测特征成本和余量才启用优化。没有值得拆的段时不选择segment。

主模型每候选5个非负系数，输入 `[1,m/1e6,work/1e6,U/1e6,M/1e3]`；便宜模型4个系数。M只是逻辑任务分配的拖尾代理，不是物理GPU makespan；模型也不将6912逻辑warp当成可无代价同时执行的6912台机器。

拟合使用非负小型岭正则最小二乘。训练子集拟合时间系数；合成验证子集选择门控参数、粗门控、便宜预测阈值与保守余量。目标惩罚平均regret及相对baseline退化，尤其最差退化。模型/验证报告不等于最终真实图成绩。默认搜索108个低成本离线选择器组合，不重新执行108轮GPU标定。

## 服务器执行顺序

在仓库根目录，A100 CUDA路径按实际环境调整：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 tests/check_schedule.py build
```

GPU检查覆盖三个强制候选的76图独立穷举、三种自动分支、特征Q、空图/尾块、重复运行和1MiB队列溢出的精确计数。失败先停止，保留编译第一处error或检查traceback。

拟合需要numpy，绘图需要matplotlib；CUDA程序不依赖它们：

```bash
python3 -m pip install numpy matplotlib
# 需要主存峰值时，先确认 /usr/bin/time 存在；Ubuntu可安装 time 软件包。
```

### 1. 生成训练数据

```bash
python3 scripts/generate_schedule_training.py --out data/schedule_train_v1
```

默认36基础图，约24训练、12合成验证，输出manifest包含seed、文件SHA256、参数、解析图已知计数和分组。训练/验证是基础图分组后的参数留出，不是全图家族留出。

| 生成意图 | 构造 | 数量 |
|---|---|---:|
| low_work | 小团链、小网格、小星形 | 12 |
| normal | 不同团大小/边数的均匀团链、网格 | 12 |
| skew | 大小团混合、偏斜随机图 | 12 |

部分正常团链和混合团保持约相同的边规模。偏斜随机图未必在定向后仍有重查询，必须看实际特征和计时；regime不是最优策略标签。

如需编号变体，改用新目录，例如：

```bash
python3 scripts/generate_schedule_training.py --out data/schedule_train_pairs_v1 \
  --relabel-ids g00 g18 g27
```

变体与原图同组、同一split。`--tiny`只用于流程正确性，拟合/评测默认拒绝把它当正式训练数据。

### 2. 单进程多策略标定

```bash
python3 -u scripts/schedule_experiments.py calibrate \
  --manifest data/schedule_train_v1/manifest.json \
  --build build --threads 12 --blocks 864 \
  --warmup 3 --repeat 15 --rounds 3 \
  --out results/schedule_v1/calibration
```

36进程，不是每策略每图新启动。每进程测试5个配置×3轮，每配置预热3次、测15次。baseline/bucket仅各一配置，segment有3个门控候选。CUDA事件包含每轮重建桶/队列和归约。公共驻留分配与一次特征提取不在候选GPU样本内，另外保存实际wall时间。

每次样本保留GPU时间数组和准备/计数/归约阶段均值。segment 的分类、门控和队列写入与普通边计数融合在一个 kernel，因此其count阶段包含这些成本，不能解读为纯二分时间。

原始JSON、stderr、精确命令、哈希、硬件快照、状态及可用时GNU time资源文件全部保存。失败立即停止，不能用不完整目录拟合。可用nohup包装该命令；脚本使用不缓冲进度输出。

### 3. 拟合并冻结

```bash
python3 scripts/fit_schedule_model.py results/schedule_v1/calibration \
  --out configs/local-schedule-v1/model.txt
```

输出：

- model.txt：CUDA自动runner直接读取的参数；GPU名称、架构、SM数、runtime/driver、blocks和段/队列配置绑定，换配置需要重新标定。
- model.json：模型哈希、开发输入哈希、训练/验证规模、搜索结果、估计验证regret与选择结果。
- model.calibration.csv：特征、实测候选耗时与预测耗时。
- model.validation.csv：验证集的估计选择结果、regret和退化。
- model.fit.png/pdf：有matplotlib时生成实测/预测图。

如果结果主要选择baseline，不自动把它包装成成功的优化；先看候选收益、拟合误差和特征税。若在验证集表现不好，可保留模型用于诊断，不能宣称可靠部署。

### 4. 冻结模型后评测真实图

已有真实图用于固定方法复测；这些图已在前期研究中使用，不应声称它们全部是从未见过的最终留出图。先小规模检查执行协议：

```bash
python3 -u scripts/schedule_experiments.py evaluate \
  --model configs/local-schedule-v1/model.txt \
  --inputs data/roadNet-CA.txt data/web-Google.txt data/cit-Patents.txt \
  --build build --threads 12 --blocks 864 \
  --runs 3 --warmup 3 --repeat 15 \
  --out results/schedule_v1/real_pilot
python3 scripts/summarize_schedule.py results/schedule_v1/real_pilot
```

评价脚本额外跑强制baseline/bucket/segment以获得候选oracle；自动runner自身仍只运行一种。正式结果用新目录统一`--runs 20`，不能混合不同协议目录。

Orkut先确认主存/显存和单次运行可承受，再用相同冻结模型单独跑压力测试；不在Orkut上搜索参数：

```bash
python3 -u scripts/schedule_experiments.py evaluate \
  --model configs/local-schedule-v1/model.txt \
  --inputs data/com-orkut.ungraph.txt \
  --threads 12 --blocks 864 --runs 3 --warmup 2 --repeat 5 \
  --timeout 1800 --out results/schedule_v1/orkut_pilot
python3 scripts/summarize_schedule.py results/schedule_v1/orkut_pilot
```

分桶额外存8m字节任务编号，队列受预算限制，CPU预处理仍需较多内存。运行前确认机器可用资源，不根据显存大小推断主存够用。

## 计时与评价边界

- **first_total_ms 主指标**：读入前开始，包含模型加载、预处理、元数据、CUDA初始化/分配、H2D、抽样、完整特征/决策、候选分配、必要准备、首次计数/归约、D2H。CPU参考核对和后续warmup/repeat排除。
- **first_execution_with_selection_wall_ms**：H2D完成后到首次结果返回，包括选择、特征缓冲/候选分配和主机决策等待。用于避免完整时间被共有预处理掩盖。
- **first_candidate_gpu_ms**：CUDA事件的候选准备+计数+归约，排除选择。其速度/regret仅用于机制诊断，不能替代自动方法成绩。
- **candidate_gpu_ms_samples**：后续预热后的候选重复；每次重排/建队列，但不重新选择。这不是多次完整自动流程。
- regret以同图三个强制候选的中位数最小值为参照；自动方法保留自身选择成本。候选oracle是评价参照，不是免费的可部署方法。噪声可能产生负的采样regret，原样保留。
- 输出每图/策略CSV、自动选择次数、平均regret、最差相对退化及加速图。用绝对毫秒数一起解释，低/正常/偏斜结构分开看。

一次图级选择不可能无成本；该版本的目标是测清楚何时能回本，不能承诺所有正常图严格零退化。模型不是查表按图名选策略，也不现场穷举候选。


释放实例前备份结果、冻结模型与生成清单：

```bash
tar -czf schedule_v1_results.tar.gz results/schedule_v1 configs/local-schedule-v1 \
  data/schedule_train_v1/manifest.json
```

训练图可按清单重建，不必把数百MB文本放入GitHub。正式报告使用回收的原始JSON；假的流程测试数据不属于GPU结果。
