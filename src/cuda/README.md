# CUDA 实现

- `serial.cu`：单线程双指针计数，保留作环境与串行验证。
- `edge_baseline.cu`：五个并行入口共用runner，负责CLI、共享预处理、显存、传输、计时与CUB两级归约。默认编译为`tc_gpu_edge_baseline`；`TC_EDGE_MODE=1/2/3/4`分别构建warp/split/query-v1/query-v2。
- `query_v2_kernels.cuh`：编译期独立 scalar/pool/budget/score/gated 内核，共用剩余查询池；[v2 指南](../../docs/query_v2.md)。
- `query_kernels.cuh`：warp 内 scalar/pool/adaptive 三模式，保留原边查询集合；诊断单独运行。
- `warp_kernels.cuh`：每边一个warp，以及轻边计算/重边入队kernel。

A100已有1400次调度评测，结果和限制见[报告](../../docs/a100_v1_results.md)。查询池新增 `tc_gpu_query` 实验源码及[运行指南](../../docs/adaptive_query_v1.md)，v1 已由用户在 A100 验证并探索。v2 budget/score 已由用户完成验证和探索；新增 gated 模式仍待服务器编译验证，见[门控指南](../../docs/query_gate.md)。

共享二分见`../common/edge_binary.hpp`，lane切片见`../common/warp_tasks.hpp`。编译、参数和验证见[baseline指南](../../docs/baseline_testing.md)与[warp指南](../../docs/warp_scheduling.md)。
