"""Generate paper tables from reported aggregate CSVs; no raw samples inferred."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/reported'
OUT = ROOT / 'paper/tables'
NAMES = {'com-amazon.ungraph':'Amazon', 'roadNet-CA':'roadNet-CA', 'web-Google':'web-Google'}


def load(name):
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        if 'median_ms' in r:
            low, mid, high = (float(r[k]) for k in ('min_ms','median_ms','max_ms'))
            if not all(math.isfinite(x) and x >= 0 for x in (low,mid,high)) or not low <= mid <= high:
                raise ValueError(('invalid aggregate',r))
    return rows


def table(name, caption, label, columns, header, rows, note):
    text = '\\begin{table}[htbp]\n\\centering\n\\small\n'
    text += f'\\caption{{{caption}}}\\label{{{label}}}\n'
    text += f'\\begin{{tabular}}{{{columns}}}\n\\toprule\n'
    text += ' & '.join(header) + ' \\\\\n\\midrule\n'
    for row in rows: text += ' & '.join(row) + ' \\\\\n'
    text += '\\bottomrule\n\\end{tabular}\n'
    text += '\\par\\smallskip{\\footnotesize ' + note + '}\n\\end{table}\n'
    (OUT/name).write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    full=load('full_gpu_20_summary.csv')
    keyed={(r['dataset'],r['variant']):r for r in full}
    if len(keyed)!=6 or len(full)!=6 or any(int(r['n'])!=20 for r in full): raise ValueError('full sample count mismatch')
    rows=[]
    for graph,name in NAMES.items():
        base,opt=keyed[graph,'legacy'],keyed[graph,'omp32']
        b,o=float(base['median_ms']),float(opt['median_ms'])
        if not math.isclose(b/o,float(opt['speedup_vs_legacy']),rel_tol=1e-12): raise ValueError('speedup mismatch')
        rows.append([name,f'{b:.3f}',f'{o:.3f}',f'{b/o:.2f}'+r'$\times$',f'{100*(1-o/b):.1f}'+r'\%'])
    table('full_pipeline.tex','完整计数时间与优化加速比','tab:full','lrrrr',
          ['数据集','基础版/ms','优化版/ms','加速比','耗时降低'],rows,
          '每版每图独立完整运行20次；表中时间为中位数。优化版为32线程组合预处理，GPU边并行计数相同。')
    ablation=load('preprocess_ablation_3_summary.csv')
    keyed={(r['dataset'],r['variant']):r for r in ablation}
    if len(keyed)!=21 or len(ablation)!=21: raise ValueError('ablation rows missing')
    rows=[]
    for variant in ['legacy','fast','parse8','map1','map8','sort8','all8']:
        rows.append([r'\texttt{'+variant+'}']+[f"{float(keyed[g,variant]['median_ms']):.3f}" for g in NAMES])
    table('ablation.tex','预处理单项与组合消融（中位数，ms）','tab:ablation','lrrr',
          ['配置','Amazon','roadNet-CA','web-Google'],rows,
          '每配置每图3次，仅计预处理；不同于完整GPU流程主指标。配置定义见正文。')
    scaling=load('thread_scaling_3_summary.csv')
    high={(r['dataset'],int(r['omp_threads'])):r for r in scaling if r['experiment_id']=='thread_scaling_high_3'}
    if len(high)!=12 or len(scaling)!=24: raise ValueError('thread sweep rows missing')
    rows=[[str(t)]+[f"{float(high[g,t]['median_ms']):.3f}" for g in NAMES] for t in [8,16,32,64]]
    table('thread_scaling.tex','组合预处理的高线程数扫描（中位数，ms）','tab:threads','rrrr',
          ['线程数','Amazon','roadNet-CA','web-Google'],rows,
          '每配置每图3次；仅使用同一轮8/16/32/64线程扫描，不与早期2/4/8轮次混合。')
    a100 = ROOT / 'results/a100_v1/summary/all_metrics.csv'
    with a100.open(encoding='utf-8-sig', newline='') as stream:
        measured = list(csv.DictReader(stream))
    indexed = {(r['id'], r['variant']): r for r in measured}
    ids = ['C1','C2','C3','W1','W2','W3','W4','S1','S2','S3','S4','S5','S6','S7']
    variants = ['edge','warp','split16','split32','split64']
    if len(measured) != 70 or len(indexed) != 70 or any(int(r['n']) != 20 for r in measured):
        raise ValueError('A100 group/sample count mismatch')
    for metric, suffix, caption in [
        ('first_count_reduce_ms', 'gpu', 'A100调度对照：GPU计数阶段'),
        ('first_total_ms', 'full', 'A100调度对照：完整任务')]:
        rows = []
        for graph in ids:
            base = float(indexed[graph, 'edge'][metric + '_median'])
            ratios = []
            for variant in variants[1:]:
                row = indexed[graph, variant]
                ratio = base / float(row[metric + '_median'])
                if not math.isclose(ratio, float(row[metric + '_speedup']), rel_tol=1e-12):
                    raise ValueError('A100 speedup mismatch')
                ratios.append(f'{ratio:.3f}')
            rows.append([graph, f'{base:.4f}' if suffix == 'gpu' else f'{base:.3f}', *ratios])
        table(f'a100_{suffix}.tex', caption, f'tab:a100_{suffix}', 'lrrrrr',
              ['图','edge/ms','warp比值','split16比值','split32比值','split64比值'], rows,
              '每图每配置20次独立进程；时间为中位数，比值为edge中位数除以对应配置中位数。大于1为更快。Wiki-Vote默认blocks不同。')
    print('Validated historical and A100 aggregates; generated 5 LaTeX tables')


if __name__=='__main__': main()
