# CUDA Triangle Counting

P09 并行计算课程项目。研究方向：基于任务规模的线程与 warp 分级调度，最终形成课程论文。

## 当前状态

- 已完成研究计划、论文提纲、实验配置模板及小图 CPU 参考程序。
- 参考程序测试已在本次初始化环境运行；后续 CI 状态以 Actions 实际结果为准。
- CUDA G0 / G1 / G2 尚未实现，尚无 GPU 计时或加速结论。
- 仓库初始为私有；未指定开源许可证，第三方代码和数据按各自条款处理。

## 研究设计

输入统一转换为无向简单图，按 `(degree, id)` 定向，计数 `sum |F(u) intersect F(v)|`。
G0 一线程一条边；G1 一 warp 一条边；G2 按定向后交集任务规模分队列。
先固定二分交集算法比较粒度，再考虑算法选择扩展。对照和优化均包含必要归约成本。

完整计划见 [研究计划](docs/research_plan.md)，实验记录约定见 [实验协议](docs/experiment_protocol.md)，论文提纲见 [paper/outline.md](paper/outline.md)。

## 运行已有参考程序

Python 3.10+，仅使用标准库。该程序用于核验，Python 对象开销较高，不作为大图高性能基准。

```bash
python -m unittest discover -s tests -v
python python/reference.py tests/triangle.txt --oracle
```

输入为两列整数边端点，允许 `#` 或 `%` 注释，支持 `.gz`。孤立点可通过 Python API 的 `vertices` 参数显式提供，CLI 仅从端点识别顶点。

后续 CUDA 开发需 NVIDIA GPU、兼容驱动和 CUDA Toolkit。先核验 `nvidia-smi` 与 `nvcc --version`，再确定编译架构和工具版本。当前没有虚构的 CUDA 构建命令。

## 实施顺序

1. 固定图清洗、CSR 与 64 位计数语义。
2. 实现并验证 G0，记录环境。
3. 实现 G1，再实现队列互斥且覆盖完整的 G2。
4. 冻结阈值，执行同 GPU 主比较、消融和完整成本实验。
5. 整理原始数据与可重绘图表，撰写课程论文。

原始图放 `data/` 并保持忽略；实验每次记录输入哈希和提交 SHA。不得把尚未测量的数据填入结果文件。
