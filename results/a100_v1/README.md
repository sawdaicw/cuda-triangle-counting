# A100 第一轮调度评测（2026-11-12）

入口报告：[完整分析](summary/analysis.md) · [项目结果说明](../../docs/a100_v1_results.md)。

## 内容与来源

- `raw/a100_results_v1.tar.gz`：用户从实际A100实例下载的原始包，按字节保留；包含1400次GPU记录、命令、元数据、空stderr、线程校准与输入检查记录。包内历史文件不静默改写。
- `raw/archive.sha256`：原始包SHA-256。包内输入和程序哈希是运行时记录；图数据和二进制本体不在包内，无法在此重新验其哈希。
- `summary/all_metrics.csv`：70组重算统计，每组20个完整样本，GPU-only额外重复单列。
- `summary/aggregate.json`：14图的几何平均比值。
- `summary/baseline_phase_means.csv`：可加的均值分解，不把各阶段中位数相加。
- `summary/integrity_check.json`：核验记录、配置及限制。
- `summary/speedup_overview.svg/.png/.pdf`：相同色标的GPU与完整任务加速比图，不表示统计显著性。
- `reproduce/`：只使用记录复算，不调用GPU或下载图。

原始源码为`77a42904ac6724e2bed77697c3c9b30ba957afb5`，另有包内记录的Wiki-Vote下载URL大小写修补。当前归档提交不冒充实验源码版本。

## 复算（仓库根目录）

```bash
# 安装分析依赖；计数程序本身不依赖numpy/matplotlib
python3 -m pip install numpy matplotlib
python3 results/a100_v1/reproduce/extract.py
python3 results/a100_v1/reproduce/analyze.py
python3 results/a100_v1/reproduce/make_report.py
python3 paper/generate_tables.py
```

解压目录`work/`忽略提交。统计/报告写回`summary/`，不改原始包。Bootstrap使用10000次按外层轮次成对重采样，seed20261112；区间为探索性且无多重比较校正，不替代独立实验确认。未删除离群值。

中文图形默认使用Linux Noto CJK字体；如系统无此字体，设置`TC_PLOT_FONT`为本机支持中文的字体文件完整路径，再运行绘图。SVG使用字形路径，提交后的图不依赖浏览器安装中文字体。

## 评价边界

主指标为`first_total_ms`，含读入/预处理/必要任务准备/初始化/传输/计数/归约/结果回传；CPU额外核验与之后GPU-only重复不计入主指标。次指标`first_count_reduce_ms`包含必要的分流和归约。

12线程OpenMP在三张真实图上校准，统一用于全部版本，不逐图挑线程。所有版本256线程/块；Wiki-Vote baseline394块，其余方案864块，是待补充控制的因素。CPU型号、频率时间序列和硬件独占性未完整建立。

1400次均通过CPU参考核验，但包内没有完整GPU单元测试日志。固定warp多数图更慢，split16在S6/S7有条件性GPU收益；完整时间的小幅变化不能直接归因GPU调度。本轮仍是探索评测，不宣称得到普遍最优策略。

历史A800预处理结果仍在`results/reported/`，其逐次原始样本尚未归档；不得与本批混算。
