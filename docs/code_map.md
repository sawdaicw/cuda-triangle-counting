# 代码导航与保留理由

| 路径 | 用途 |
|---|---|
| `src/common/graph.hpp` | Graph/CSR、原版读图、CPU双指针参考 |
| `src/common/preprocess.*` | legacy/fast/OpenMP解析、编号映射、排序去重；共享预处理入口 |
| `src/common/edge_binary.hpp` | GPU/CPU可共用二分交集、定向边源点元数据 |
| `src/common/query_v2_policy.hpp` | v2 两路径成本模型、共用单查询二分 |
| `src/cuda/query_v2_kernels.cuh` | 编译期独立模式、前缀执行/剩余协作、成本评分 |
| `scripts/benchmark_query_gate.py` | 固定K、扫描R的门控对照，分开汇总首次与额外GPU重复 |
| `results/query_pilot_20261123/` | 今日两轮用户汇总日志、表格及复算，不含逐次原始样本 |
| `scripts/benchmark_query_v2.py` | 两条主路线独立进程探索；可选基础对照 |
| `src/common/query_policy.hpp` | host/device 共用整数评分与门控，默认参数未校准 |
| `src/cuda/query_kernels.cuh` | 虚拟查询池、快速路径、自适应批内调度与可选诊断 |
| `scripts/benchmark_query.py` | 固定参数、相同网格的四模式完整进程对照 |
| `src/common/warp_tasks.hpp` | 短表长度、lane不重叠查询切片 |
| `src/common/profile_json.hpp` | 可选预处理诊断JSON |
| `src/cpu/serial.cpp` | CPU串行计数入口；预处理可并行 |
| `src/cuda/serial.cu` | GPU单线程验证；与主要并行性能结果分开 |
| `src/cuda/edge_baseline.cu` | edge/warp/split/query-v1/query-v2共用CLI、分配、传输、归约与计时runner |
| `src/cuda/warp_kernels.cuh` | 固定warp及轻边计算/重边入队kernel |
| `scripts/prepare_benchmark.py` | 数据清单下载、解压、确定性合成 |
| `scripts/benchmark_preprocess.py` | 预处理消融、线程扫描、完整预处理对照 |
| `scripts/benchmark_scheduling.py` | 调度对照，独立进程、逐次记录 |
| `scripts/explore_queries.py` | 查询池CPU映射与理想化轮数分析，非CUDA程序 |
| `python/reference.py` | Python参考与独立小图三元组枚举 |
| `tests/` | 小图、非法输入、CSR一致性、任务覆盖、GPU入口验证 |
| `results/reported/` | 历史A800聚合记录，不伪造逐次样本 |
| `results/a100_v1/` | A100原始包、重算统计、图及复现脚本 |
| `results/query_exploration_v1/` | CPU模型结果，非GPU评测 |
| `paper/` | LaTeX正文与从CSV生成的表格 |

## 本轮清理

保留全部当前计数入口、预处理模式及测试：它们分别承担正确性参考、性能基线或消融对照，尚不能视作无用代码。`graph.hpp`的原始预处理也是legacy模式和CSR一致性检查的参考，不删除。

删除未被代码读取的`configs/experiments.json`早期模板（G0/G1/G2、预热5次/重复30次）与空`results/raw/template.csv`。真实配置由CLI、实验metadata和原始JSON记录，避免模板被误当执行参数。

早期研究计划移到`docs/archive/initial_research_plan.md`；当前路线在`docs/research_plan.md`。README去掉重复的阶段更新，输入/算法详解保留到`docs/input_and_algorithm.md`。清理未修改已测GPU计算代码。


## 2026-11-26：统一研究实验与全局查询

- `src/common/global_queries.hpp`：CPU/GPU 共用查询所属边定位与连续查询段执行，可独立验证查询覆盖。
- `src/cuda/global_query_kernels.cuh`：短表查询数、CUB 前缀扫描、查询段分配与块内归约；由 shared runner 的 mode 5 调用。
- `scripts/run_research.py`：K/R/B/M 扫描、固定参数真实图评测与压力测试，失败保留记录并停止。
- `scripts/make_scale_graphs.py`：开发图嵌套顶点诱导子图，辅助小图阈值探索。
- `scripts/plot_research.py`：精确 CSV、PNG/PDF 参数曲线、固定方法加速比与阶段均值。
- `configs/research_freeze.example.json`：参数冻结模板，不代表已经选出的最优参数。
- `tests/test_global_queries.cpp` / `tests/check_global_queries.py`：前者 host 独立穷举/查询覆盖，后者服务器 GPU 编译后运行。
- `tests/test_research_workflow.py`：假设备测试执行协议、冻结检查、错误保存与绘图；其数据不属于性能结果。

完整使用顺序见 [实验指南](research_experiments.md)。


## 自动调度原型

- `src/common/schedule_policy.hpp`：段划分/尾段合并、查询代理成本门控、特征归约结构、便宜抽样、模型读取与保守选择。
- `src/cuda/schedule_kernels.cuh`：GPU特征、64桶重排、普通边计数、带容量回退的段队列。
- `src/cuda/schedule_runner.cu`：标定/自动执行两个入口共用的预处理、资源和计时逻辑。
- `scripts/generate_schedule_training.py`：36基础图与可选编号变体，按基础图分组划分。
- `scripts/schedule_experiments.py`：标定/冻结评测，保存命令、哈希、状态与原始结果。
- `scripts/fit_schedule_model.py`：小型非负成本模型、验证选择器、导出参数及诊断CSV/图。
- `scripts/summarize_schedule.py`：完整成本与选择阶段成本的regret、退化和加速比。
- `tests/test_schedule_policy.cpp`：段覆盖、容量溢出、独立穷举核对与选择逻辑。
- `tests/check_schedule.py`：服务器上三候选及自动分支的GPU验证。
- `tests/test_schedule_cuda_syntax.py`：去掉launch语法后的host C++语义检查，**不是nvcc/device编译**。

运行顺序与限制见 [自动调度指南](automatic_scheduling.md)。


## 2026-11-28：独立hash批次队列

- `src/cuda/query_v2_kernels.cuh`：Mode9 producer，借助原reuse共同目标检查在计数前延期符合hash条件的批次；独立`count_queued_hash_batches`消费队列。旧Mode7/8保留。
- `src/cuda/edge_baseline.cu`：hash-queue缓冲、两阶段计时与归约、小图绕过和CUDA资源属性。
- `scripts/benchmark_hash_queue.py`：默认三版全14图；正式计时后独立诊断。
- `scripts/hash_queue_diagnostics.py`：实际入队/查询量、资源与阶段时间，不与主样本混合。
- `tests/check_hash_queue.py`：服务器oracle、路由、空队列、碰撞、容量及绕过检查。
- `results/poster_v1_20261126`：此前用户回传840次统一评测，勿与新版本混用。
