from pathlib import Path
import os
import json,csv,math,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager
r=Path(__file__).resolve().parents[1];o=r/'summary'
rows=list(csv.DictReader((o/'all_metrics.csv').open(encoding='utf-8-sig')))
for x in rows:
 for k in list(x):
  if k not in ['id','graph','variant'] and x[k]!='':x[k]=float(x[k])
lookup={(x['id'],x['variant']):x for x in rows}
ids=['C1','C2','C3','W1','W2','W3','W4','S1','S2','S3','S4','S5','S6','S7']
names={'C1':'com-Amazon','C2':'roadNet-CA','C3':'web-Google','W1':'Amazon0302','W2':'Wiki-Vote','W3':'soc-Epinions1','W4':'cit-Patents','S1':'团链 K32','S2':'网格','S3':'共享枢纽','S4':'团链 K128','S5':'团链 K512','S6':'混合图：较少大团','S7':'混合图：较多大团'}
vs=['warp','split16','split32','split64'];aggregate=json.loads((o/'aggregate.json').read_text());check=json.loads((o/'integrity_check.json').read_text())
report='''# A100 第一轮完整评测分析

数据来源：用户上传 `a100_results_v1.tar.gz`，源代码基于 `77a4290`，另有 W2 下载URL修补（不改变计数代码）。本报告从原始逐次JSON重新计算，没有删除离群值。

## 主要结论

1. **结果完整且计数一致**：14图×5配置×20次=1400份原始结果，70组每组恰好20次；1400份均verified=true，无非空stderr。图规模、三角形数在版本与重复间一致，并与清单中的参考计数/合成公式相符。脚本汇总值已重新核算。
2. **固定warp没有普遍收益**：GPU计数阶段在14图中的10图点估计更慢。道路图约慢9.17倍，网格约慢12.02倍。S5虽有长交集，固定warp仅约1.002倍，尚无稳定改善证据。
3. **分流对轻重混合结构有价值**：S6的split16计数阶段约1.371倍加速，S7约1.190倍。它们支持继续研究自适应分配，但不支持当前版本普遍优于baseline。
4. **完整任务差异很小，不能直接归因GPU任务划分**：完整时间几何平均比约1.008～1.024倍，且同一预处理代码的耗时、CUDA准备时间存在变化。主指标仍应保留完整任务时间，同时分别分析GPU阶段。
5. **发现需要单独控制的实验因素**：Wiki-Vote的edge使用394块，其他方案864块。其结果比较的是当前默认配置，不能严格解释为固定相同网格后的纯任务粒度效果。

## 实验配置与计时

- NVIDIA A100-PCIE-40GB，compute capability8.0，CUDA runtime/driver均12080。
- 统一OpenMP解析、映射、排序，requested/actual均12线程；OMP_DYNAMIC=FALSE，OMP_WAIT_POLICY=PASSIVE。
- 每块256线程；常见默认864块，Wiki-Vote baseline例外。
- 首次完整任务：输入读取到最终计数回到CPU，含预处理、任务元数据、CUDA准备、分配/传输、计数/分流与归约。
- 主指标：20次独立进程的first_total_ms中位数。次指标：first_count_reduce_ms中位数。
- 每进程warmup0、repeat1，除首次执行外还有1次GPU-only重复。重复样本没有当成第21次完整测量。
- GPU分项profile关闭。队列诊断读回和CPU验证位于首次完整计时之外。

## 全部图的GPU阶段结果

加速比=同图edge中位数/该方案中位数。大于1为更快，小于1为更慢。GPU阶段包含分流及全部归约。

| 图 | edge GPU时间/ms | warp | split16 | split32 | split64 |
|---|---:|---:|---:|---:|---:|
'''
for id in ids:
 b=lookup[id,'edge']
 report+=f"| {id} {names[id]} | {b['first_count_reduce_ms_median']:.4f} | "+' | '.join(f"{lookup[id,v]['first_count_reduce_ms_speedup']:.3f}×" for v in vs)+' |\n'
report+='''
## 全部图的完整任务结果

| 图 | edge完整时间/ms | warp | split16 | split32 | split64 |
|---|---:|---:|---:|---:|---:|
'''
for id in ids:
 b=lookup[id,'edge']
 report+=f"| {id} {names[id]} | {b['first_total_ms_median']:.3f} | "+' | '.join(f"{lookup[id,v]['first_total_ms_speedup']:.3f}×" for v in vs)+' |\n'
report+='''
## 跨图几何平均

每张图先计算中位数比值，再对14个比值取几何平均；不是把所有图的毫秒时间混在一起平均。它只总结本次14图集合，不代表任意图的期望加速。

| 配置 | 完整任务 | GPU计数阶段 | GPU阶段更快的图数（点估计） |
|---|---:|---:|---:|
'''
for v in vs:
 report+=f"| {v} | {aggregate['first_total_ms'][v]['geomean_speedup']:.3f}× | {aggregate['first_count_reduce_ms'][v]['geomean_speedup']:.3f}× | {aggregate['first_count_reduce_ms'][v]['point_estimate_wins']}/14 |\n"
report+='''
## 如何解释结果

### 短任务使用整个warp，可能降低吞吐

道路图与网格中，warp分别约慢9.17倍和12.02倍，而分流的重边数在16/32/64阈值下均为0。固定warp的结果与短任务线程闲置、每个warp处理更少边的解释相符，但本批没有采集硬件利用率，尚不能仅凭时间定量确认原因。

com-Amazon、Amazon0302、共享枢纽也没有达到16阈值的重边。这进一步说明原始高度数并不等于定向后的长交集。

### 轻重混合图支持继续探索分流

| 图 | split16重边占比 | GPU edge/ms | GPU split16/ms | GPU加速比 |
|---|---:|---:|---:|---:|
'''
for id in ['S6','S7']:
 b=lookup[id,'edge'];x=lookup[id,'split16']
 report+=f"| {id} | {x['heavy_edge_fraction']*100:.2f}% | {b['first_count_reduce_ms_median']:.4f} | {x['first_count_reduce_ms_median']:.4f} | {x['first_count_reduce_ms_speedup']:.3f}× |\n"
report+='''
S6和S7具有相同边数而团构成不同，但同时也改变了顶点数和三角形数，不应将它们描述成除“重边比例”外完全相同的输入。warp、分流的具体收益由任务分布、访存和额外开销共同决定。

S5固定warp几乎持平，split16/32/64均更慢，说明“交集更长”本身还不足以保证收益。下一步需要分项计时与硬件分析，而不是仅继续增大阈值。

### web-Google的split64更快，不能归因于重边warp协作

该组重边数为0，GPU阶段点估计约1.113倍，但bootstrap区间覆盖1。即使以后确认收益，也不能归因于当前没有发生的重边协作；需要考察提前跳过空交集的代码路径、编译差异和运行波动。

### 完整时间主要由预处理和CUDA准备构成

'''
report+=f"在280个edge首次运行中，cuda_setup_ms整体中位数约 **{check['cuda_setup_median_all_edge_samples_ms']:.2f} ms**。逐次GPU计数时间占完整时间比例的中位数约 **{check['gpu_fraction_per_run_median_percent']:.3f}%**，最大约 **{check['gpu_fraction_per_run_max_percent']:.2f}%**。因此计数阶段的收益很容易被其他阶段掩盖。\n\n"
b=lookup['S6','edge'];x=lookup['S6','split16']
report+=f"例如S6的split16将GPU计数中位数减少约 **{b['first_count_reduce_ms_median']-x['first_count_reduce_ms_median']:.3f} ms**，而完整任务中位数减少约 **{b['first_total_ms_median']-x['first_total_ms_median']:.3f} ms**。这两个差值不能直接作为可加的成本分解，且后者的大部分差异无法由这约0.224 ms的GPU改善解释。\n\n"
report+='''完整任务是课程要求的主指标，应继续报告。可以另设“图已加载、CUDA上下文已建立”的重复查询场景作为补充，但必须明确其计时边界，不能替代或混入本轮主指标。

## 不确定性与需要补充的对照

CSV给出每组min/median/max，以及对20个外层轮次成对重采样得到的比值区间。采用10000次percentile bootstrap，固定seed20261112；同图各配置在同一轮次中交错执行。区间仅描述本批采样的不确定性，没有多重比较校正，也不覆盖未记录的跨时段、跨机器变化，不作为确认性显著性结论。

- S6 split16 GPU加速比约1.371，探索性95%区间约[1.248,1.583]。
- S7 split16 GPU加速比约1.190，探索性95%区间约[1.081,1.294]。
- S5 warp GPU加速比约1.002，区间约[0.862,1.084]。
- 多数组合的完整时间区间包含1；少数组合不包含1，也不能据此将其变化全部归因于GPU分配策略，因为共享预处理与初始化耗时同样变化。

建议后续分开做三类补充：

1. **诊断而非重跑全部**：挑C2/S2/S5/S6/S7，打开GPU分项profile，观察融合的轻边计数+入队、重边计数、全局归约。融合阶段不能直接拆成纯筛选时间。
2. **控制实验因素**：Wiki-Vote各版本固定相同blocks追加对照；无重边图用于测队列与额外启动成本。必要时用Nsight检查利用率和访存，未测前不指定瓶颈。
3. **改进后再扩展**：评估无队列的局部分配、subwarp或廉价工作量估计；先冻结新版本和阈值选择规则，再用保留图或新的图评测，避免在同一14图上选最快参数又将其当独立验证。

本轮所有原始样本都应保留。当前结果是一个有价值的负面与条件性结果：固定warp未普遍优于线程边并行，轻重分流在部分混合结构上显示收益。

## 文件说明

- all_metrics.csv：70组原始数据重算汇总，包含计时、加速比、探索性区间、重边数量和队列比例。
- aggregate.json：跨图几何平均。
- integrity_check.json：核验结果、配置、已知限制。
- speedup_overview.png/.pdf/.svg：完整任务与GPU阶段的加速比图，同一色标；1为无变化。
- baseline_phase_means.csv：可加的均值分解。使用预处理、CUDA准备、其余时间三部分，三者均值之和等于总时间均值；不把各阶段中位数相加。
- analyze.py、make_report.py：分析与绘图脚本；运行需原始包及numpy/matplotlib。本次分析不包含任何新的GPU运行。

原始包未包含原始图字节和可执行文件，因此只能核对记录的哈希和版本信息，不能从包内重新计算输入/二进制哈希。CPU型号、频率、温度时间序列和运行独占性也未由包内信息完整建立。完整GPU正确性测试日志不在包中，不能把1400次CPU参考核验等同于额外独立穷举测试。
'''
(o/'analysis.md').write_text(report,encoding='utf-8')
# Static scientific figure with exact point estimates, not a dashboard.
font_path=Path(os.environ.get('TC_PLOT_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if font_path.exists(): font_manager.fontManager.addfont(str(font_path))
font_name=font_manager.FontProperties(fname=str(font_path)).get_name() if font_path.exists() else 'DejaVu Sans'
plt.rcParams.update({'font.family':[font_name,'DejaVu Sans'],'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
fig,axes=plt.subplots(1,2,figsize=(12.3,7.7),sharey=True,layout='constrained')
norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=1.5)
for ax,metric,title in zip(axes,['first_count_reduce_ms','first_total_ms'],['GPU计数阶段（含分流与归约）','完整任务（含预处理与CUDA准备）']):
 data=np.array([[lookup[id,v][metric+'_speedup'] for v in vs] for id in ids])
 im=ax.imshow(data,aspect='auto',norm=norm,cmap='RdYlGn')
 ax.set_xticks(range(4),['warp','split16','split32','split64']);ax.set_title(title,fontsize=12,pad=14)
 ax.set_yticks(range(14),[f'{id}  {names[id]}' for id in ids],fontsize=10)
 for i in range(14):
  for j in range(4):ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=10,color='#17222D')
 ax.tick_params(length=0);ax.spines[:].set_visible(False)
fig.colorbar(im,ax=axes,shrink=.65,label='加速比（大于1表示更快）',ticks=[0,.5,1,1.5])
fig.suptitle('A100第一轮：GPU计数改善具有结构依赖，完整时间差异较小',fontsize=15)
fig.supxlabel('每格为20次独立进程的中位数比值：edge / 对应方案；不表示统计显著性',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(o/f'speedup_overview.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print('Report and figures written')
