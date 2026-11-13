# 当前研究计划（2026-11-13）

## 已完成

统一无向简单图规范、度数/编号定向、排序向前CSR、CPU与GPU串行验证；CUDA边并行baseline；OpenMP解析/映射/排序；固定warp与队列分流；14图A100第一轮1400次原始结果归档。

历史预处理收益见[历史结果](experiment_results.md)，最新调度结论见[A100结果](a100_v1_results.md)。二者机器和比较对象不同，不混算。

## 当前问题

边的短表长度差异会产生不同查询数量，但不能默认负载不均衡是所有图上的唯一瓶颈。现有baseline已快；重边分类、原子队列、第二kernel与协作本身均可能产生额外成本。本轮未测GPU分项，因此先提出假设，再用诊断检验。

## 下一步：warp内虚拟查询池

保持baseline的短表查询集合及二分算法，每warp取32条连续边，以局部前缀和描述查询池，lane分担查询，不显式生成全部查询记录、不建全局重边队列。完整设计和CPU验证见[查询级探索](query_parallel_exploration.md)。

这是已有细粒度查询/wedge思想上的实现探索，不能将查询级并行或前缀映射宣称首次提出。CUDA原型尚未实现。

先核验每个(e,j)恰好一次、空表/尾批、64位索引和归约；然后检查道路/网格退化、S5长任务和S6/S7混合图。若定位开销占主导，再试起点定位摊薄或baseline快速路径；不同时引入复杂评分、随机重排和专用分析warp。

## 方法选择与文献

- Fox等，Fast and Adaptive List Intersections on the GPU（HPEC2018）：工作量分桶与不同线程数；打分并非新概念。
- Pearson等，Update on Triangle Counting on GPU（HPEC2019）：warp按32边取批，现场成本估计选择执行方式。
- Almasri等，HyKernel（HPEC2021）：单阶段/双阶段及交集算法/粒度选择。
- Spaan等，WeTriC（Euro-Par2025）：细粒度wedge、轻量索引及定位摊薄。
- Pandey等，TRUST（TPDS2021）：顶点中心哈希复用与负载均衡。

不预设单一方法全图最优；在候选实现有稳定收益后再研究低成本选择，保留调参/验证边界，选择成本计入总时间。

## 评价与论文

主指标first_total_ms、辅指标first_count_reduce_ms，包含必要准备和归约。CPU模型轮数比不是GPU加速比。原始样本保留，固定版本/输入/设备/线程配置；新实验不覆盖历史数据。

论文目前记录预处理与调度探索，不把未实现的查询池写成已完成贡献。优先补充同blocks对照、阶段诊断和结果解释，其他GPU/外部复现属扩展。早期候选方案见[历史计划](archive/initial_research_plan.md)。
