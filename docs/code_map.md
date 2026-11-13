# 代码导航与保留理由

| 路径 | 用途 |
|---|---|
| `src/common/graph.hpp` | Graph/CSR、原版读图、CPU双指针参考 |
| `src/common/preprocess.*` | legacy/fast/OpenMP解析、编号映射、排序去重；共享预处理入口 |
| `src/common/edge_binary.hpp` | GPU/CPU可共用二分交集、定向边源点元数据 |
| `src/common/warp_tasks.hpp` | 短表长度、lane不重叠查询切片 |
| `src/common/profile_json.hpp` | 可选预处理诊断JSON |
| `src/cpu/serial.cpp` | CPU串行计数入口；预处理可并行 |
| `src/cuda/serial.cu` | GPU单线程验证；与主要并行性能结果分开 |
| `src/cuda/edge_baseline.cu` | edge/warp/split共用CLI、分配、传输、归约与计时runner |
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
