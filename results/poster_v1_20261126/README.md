# A100 门控与哈希：2026-11-26 中期冻结评测

用户回传 `poster_v1_results.tar.gz`。14图×6方案×10独立进程，共840份JSON，全部verified；同图规模与精确三角形数一致。重新计算原始样本得到的统计与所存summary.json完全一致。

GPU主指标是每进程30次count+reduce样本的中位数，再对10进程取中位数。每进程额外预热5次。完整时间为首次端到端结果，另列，不把GPU加速比解释为端到端加速。IQR为10个进程统计值的25/75分位。

A100 PCIe40GB，CUDA12.8，864×256，OpenMP预处理12线程。主比较使用同轮baseline，图及方案随机交错。固定small=8、m<262144绕过、hash长度64..512且Q>=8b，split查询数严格>32（threshold33）。数据与二进制哈希在metadata.json；回传源码匹配先前poster_patch_v1，基底763670d加本地补丁，不伪装为该基底本身已经包含门控。

阶段结论：

- Patents：reuse 4.515096→3.225864ms，1.400×；gated-reuse 3.345224ms，1.350×。
- S2：reuse相对baseline额外66.720μs，gated-reuse额外10.936μs。
- S5：fused gated-hash 15.012792→9.471808ms，1.585×；S7约1.531×。
- S6：split_gt32约1.582×。
- fused gated-hash在7张真实图上均无中位数加速；Patents为5.343552ms。静态共享内存、建表成本与任务特点是待分辨的原因，没有诊断证据时不指定唯一原因。
- 当前组合方案不满足跨图普遍加速，不把不同方案逐图最佳值当成已实现自动方案。

此目录保存用户回传汇总及测量元数据；完整原始archive由用户保留，不重复提交840份JSON。新hash-queue代码及实验与这一历史批次分开，见docs/hash_queue.md。
