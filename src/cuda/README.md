# CUDA 实现

- `serial.cu`：GPU 单线程串行验证入口，已有 A100 测试记录。
- `edge_baseline.cu`：新增边并行二分基线，一个线程循环处理多条边；块内 CUB BlockReduce，最终 CUB DeviceReduce，全部使用 64 位计数。待服务器验证。

共享图预处理见 `../common/graph.hpp`，共享二分交集与任务元数据见 `../common/edge_binary.hpp`。

编译、运行、计时边界、已完成和待完成验证见 [基线测试指南](../../docs/baseline_testing.md)。

本次没有实现 warp 局部调度优化版。
