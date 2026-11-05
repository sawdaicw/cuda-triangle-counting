# CUDA implementation roadmap

当前不含 CUDA 内核，不构成可运行 GPU 实现。

计划接口：输入排序向前 CSR、边端点数组、可选任务索引队列；输出每任务 uint64 计数和最终总数。

- G0: one thread per edge, binary search intersection.
- G1: one full warp per edge, strided short-list traversal and warp reduction.
- G2: disjoint thread/warp task queues selected from forward-list lengths.
- Shared: error checking, empty input, 64-bit offsets/counts, identical reduction accounting.

CUDA 内核完成后增加独立构建配置和 GPU 验证；CPU CI 不能证明 CUDA 正确。
