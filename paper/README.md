# LaTeX论文初稿

入口为 `main.tex`，建议使用 XeLaTeX（ctexart + Fandol字体）。当前结构为：

1. 问题定义。
2. 图预处理与精确计数流程。
3. OpenMP并行预处理优化。
4. 实验设计与评价指标。
5. 实验结果与分析。
6. 总结与后续工作。

这是基于当前已完成工作的阶段性初稿，不是定稿。作者、相关工作、BibTeX、原始样本统计与主存测量仍需补充；未实现的warp局部调度只放在后续工作里。

## 生成表格和编译

从仓库根目录执行：

```bash
python3 paper/generate_tables.py
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

也可在Overleaf上传整个paper目录，选择XeLaTeX，以main.tex为主文件。生成好的tables已提交，上传paper目录即可编译；更新CSV后应在本地先重新运行生成脚本再上传。若使用大学正式模板，替换导言区/入口，sections与tables可继续使用。

历史表格来源是 `results/reported/` 中用户提供的聚合结果，新增A100表格来源是 `results/a100_v1/summary/all_metrics.csv` 的原始记录重算结果，脚本不推测逐次样本。不要手改表格数值而不更新CSV。编译产物在paper/build/，由Git忽略；仓库保存源码。

原始120份逐次完整流程JSON尚未上传，不能据当前摘要生成箱线图或显著性检验。详细来源和限制见 `docs/experiment_results.md` 与 `results/reported/experiment_manifest.json`。

新增A100调度评测已写入结果节，查询级方案仍仅为后续设计与CPU映射验证。A100原始数据已归档，历史A800的120份逐次记录仍未归档。两者不得混淆。
