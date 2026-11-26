# A100 门控探索回传（2026-11-23）

`reported.txt` 原样保留用户提供的聚合文本，14 图 × 5 配置（edge、budget K8、gate K8/R8、R16、R32），每配置 n=3。`summary.json`、`summary.csv` 可由 `python3 results/query_gate_pilot_20261123/rebuild.py` 复算。

没有逐次原始 JSON，因此无法补出样本、置信区间、资源峰值或严格确认源码 SHA/运行环境。关联仓库门控版本 e47a1c7，关联操作流程为 warmup=0/repeat=1；这些不是从原始运行元数据验证的事实。`count_reduce_ms_median` 在该流程下只是第一次完整运行之后的单次 GPU 重复，不能称作稳定热运行中位数。

按首次 GPU 计数归约中位数比较，7 个真实图中 budget、gate8、gate16、gate32 分别有 4、4、2、2 图快于本轮 edge；这些差异包含微秒级变化，n=3 不足以断言稳定收益。Google、Patents 有值得继续测的信号，均匀大团 S5 明显退化。不同配置逐图挑最好值不能作为固定方法的最终成绩。

这是探索记录；下一轮统一协议见 [实验指南](../../docs/research_experiments.md)。
