"""Rebuild aggregates from user-reported logs. These are NOT per-run raw samples."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
METHODS={'v1':['edge','scalar','pool','adaptive'],'v2':['edge','budget4','budget8','budget16','score']}


def load():
    result={}
    for cohort,methods in METHODS.items():
        graphs={}
        for line in (ROOT/'input_logs'/f'{cohort}_reported.txt').read_text().splitlines():
            graph,mode,payload=line.split(' ',2)
            row=json.loads(payload)
            if mode not in methods or mode in graphs.setdefault(graph,{}): raise ValueError('unexpected/duplicate variant')
            if row['n']!=3: raise ValueError('unexpected sample count')
            for metric in ['first_count_reduce_ms','first_total_ms']:
                item=row[metric]
                if not all(math.isfinite(item[k]) and item[k]>0 for k in ['median_ms','min_ms','max_ms']): raise ValueError('bad timing')
                if not item['min_ms']<=item['median_ms']<=item['max_ms']: raise ValueError('bad range')
            graphs[graph][mode]=row
        if len(graphs)!=14 or any(set(v)!=set(methods) for v in graphs.values()): raise ValueError('incomplete cohort')
        for variants in graphs.values():
            for mode,row in variants.items():
                for metric in ['first_count_reduce_ms','first_total_ms']:
                    ratio=variants['edge'][metric]['median_ms']/row[metric]['median_ms']
                    if not math.isclose(ratio,row[metric]['speedup_vs_edge'],rel_tol=1e-9): raise ValueError('ratio mismatch')
        result[cohort]=graphs
    return result


def main():
    data=load()
    (ROOT/'summary.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    stream=io.StringIO();writer=csv.writer(stream,lineterminator="\n")
    writer.writerow(['cohort','graph','variant','n','gpu_median_ms','gpu_min_ms','gpu_max_ms','gpu_speedup_vs_edge',
                     'full_median_ms','full_min_ms','full_max_ms','full_speedup_vs_edge'])
    for cohort,graphs in data.items():
        for graph,variants in graphs.items():
            for mode in METHODS[cohort]:
                row=variants[mode];gpu=row['first_count_reduce_ms'];full=row['first_total_ms']
                writer.writerow([cohort,graph,mode,row['n'],gpu['median_ms'],gpu['min_ms'],gpu['max_ms'],gpu['speedup_vs_edge'],
                                 full['median_ms'],full['min_ms'],full['max_ms'],full['speedup_vs_edge']])
    (ROOT/'summary.csv').write_text(stream.getvalue())
    metadata={'date':'2026-11-23','device_reported':'NVIDIA A100-PCIE-40GB','preprocessing_threads_reported':12,
              'blocks_reported':864,'warmup_reported':0,'repeat_reported':1,
              'status':'user-reported aggregate pilot results; not raw per-process samples',
              'cohort_source_commits':{'v1':'a787faffc04295f3a4124728662c61804f9e270c','v2':'f67bea4cdf331895321953a0b4b11dd1da07073e'},
              'cohort_sizes':{'v1':'14 graphs x 4 variants x 3 processes (reported)','v2':'14 graphs x 5 variants x 3 processes (reported)'},
              'limits':['No original per-process JSON, hashes or machine metadata received; commit/config association follows conversation and supplied commands.',
                        'first GPU execution includes initial execution effects; no warmups before it.',
                        'Do not combine cohorts or interpret min/median/max as a confidence interval.',
                        'Additional repeat=1 GPU-only sample is not another full-task sample.'],
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'input_logs').glob('*.txt'))}}
    (ROOT/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    lines=['# 2026-11-23 A100 查询池初步探索','','来源是用户回传的两份汇总日志，原文保存在 input_logs/。**没有收到逐次原始 JSON，不能由中位数/min/max 还原样本或计算可靠置信区间。**',
           '','按当时命令：统一 OpenMP 12 线程、864 blocks、每配置 3 次独立完整进程、warmup=0、repeat=1。输入哈希与完整机器元数据未收到。',
           '','以下为首次 GPU 计数+归约时间中位数（ms）；完整时间及范围在 CSV/JSON 中。两轮 baseline 有波动，不混合分母、不逐图选最优拼成一个方法。','']
    for cohort,graphs in data.items():
        methods=METHODS[cohort];lines += [f'## {cohort}','','| 图 | '+' | '.join(methods)+' |','|---|'+'---:|'*len(methods)]
        for graph,variants in graphs.items():
            lines.append('| '+graph+' | '+' | '.join(f"{variants[mode]['first_count_reduce_ms']['median_ms']:.4f}" for mode in methods)+' |')
        lines.append('')
    snapshots=json.loads((ROOT/'diagnostics_v1_reported.json').read_text())['snapshots']
    lines += ['## v1 额外诊断快照','','来源为用户粘贴的六份诊断，pass_ms含原子统计/回传，不作正常性能时间。',
              '','| 图 | 快速批次比例 | 池批次比例 | 池查询比例 |','|---|---:|---:|---:|']
    for graph,row in snapshots.items():
        if row['fast_batches']+row['scalar_batches']+row['pool_batches']!=row['batches']: raise ValueError('bad diagnostic partition')
        lines.append(f"| {graph} | {row['fast_batches']/row['batches']:.2%} | {row['pool_batches']/row['batches']:.2%} | {row['pooled_queries']/row['all_queries']:.2%} |")
    lines.append('')
    lines += ['## 观察与下一步','','- v1：pool 在 Google/专利图有收益信号，adaptive 可能付出评分却很少启用 pool；S7 有初步收益。',
              '- v2：budget 三个 K 在 Google、专利、S1 的点估计均优于本轮 baseline；S5 均匀重任务仍退化。',
              '- 道路图短表均≤4，K=4/8/16 应均在独立前缀完成，但时间仍有波动，不能直接把 K 的差异解释为机制差异。',
              '- 新门控先独立 K 次，仅当剩余线程数 1..R 时共享，否则独立完成；本目录不包含新门控性能数据。',
              '- 后续保留完整流程时间，并补充分开标记的预热 GPU 重复诊断；规则冻结后再做正式多次运行与保留图验证。',
              '','## 复算','','```bash','python3 results/query_pilot_20261123/rebuild_summary.py','```','',
              'metadata.json 明确来源与缺失信息；summary.csv 共 126 个图/配置聚合行。不要将这些聚合行当作 378 个独立样本。','']
    (ROOT/'README.md').write_text('\n'.join(lines))
    print('PASS: 56 v1 + 70 v2 aggregate rows; supplied ratios checked; tables rebuilt')


if __name__=='__main__': main()
