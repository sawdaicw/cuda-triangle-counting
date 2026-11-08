# GPU 单线程串行实现

`serial.cu` 的 `serial_kernel<<<1, 1>>>` 在一个 GPU 线程上处理全部定向边。
CPU 清洗输入、按 `(degree,id)` 定向并构建排序向前 CSR，GPU 用双指针交集累加 64 位三角形数量。
共享算法见 `../common/graph.hpp`；C++ CPU 入口见 `../cpu/serial.cpp`。

构建、测试、输入约定及计时边界见仓库根 README。
该版本用于正确性和环境验证；后续 CUDA 并行策略尚未确定。
