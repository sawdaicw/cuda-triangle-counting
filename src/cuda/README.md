# CUDA 实现

- `serial.cu`：单线程双指针计数，保留作环境与串行验证。
- `edge_baseline.cu`：三个并行入口共用runner，负责CLI、共享预处理、显存、传输、计时与CUB两级归约。默认编译为`tc_gpu_edge_baseline`；`TC_EDGE_MODE=1/2`分别构建warp/split。
- `warp_kernels.cuh`：每边一个warp，以及轻边计算/重边入队kernel。

A100已有1400次调度评测，结果和限制见[报告](../../docs/a100_v1_results.md)。查询级虚拟任务池仍是[设计与CPU模型](../../docs/query_parallel_exploration.md)，没有对应CUDA可执行文件。

共享二分见`../common/edge_binary.hpp`，lane切片见`../common/warp_tasks.hpp`。编译、参数和验证见[baseline指南](../../docs/baseline_testing.md)与[warp指南](../../docs/warp_scheduling.md)。
