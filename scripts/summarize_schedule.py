"""Export frozen evaluation metrics, regret, regressions and exact CSV."""
import argparse,csv,json,statistics as st
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('evaluation');a=p.parse_args();root=Path(a.evaluation)
 meta=json.loads((root/'metadata.json').read_text())
 if not (root/'COMPLETE').is_file() or meta['arguments']['action']!='evaluate':p.error('complete evaluation required')
 rows=[];summary={}
 for graph in meta['graphs']:
  methods={s:[] for s in ['baseline','bucket','segment','auto']}
  for job in meta['jobs']:
   if job['graph']['id']==graph['id']:methods[job['strategy']].append(json.loads((root/(job['id']+'.json')).read_text()))
  s={}
  for method,records in methods.items():
   entry=dict(n=len(records),vertices=records[0]['vertices'],undirected_edges=records[0]['undirected_edges'],triangles=records[0]['triangles'])
   for key in ['first_total_ms','first_execution_with_selection_wall_ms','first_candidate_gpu_ms']:
    vals=[r[key] for r in records];entry[key]=dict(median_ms=st.median(vals),min_ms=min(vals),max_ms=max(vals))
   entry['selections']={k:sum(r['selected_strategy']==k for r in records) for k in methods if k!='auto'}
   s[method]=entry
  for key in ['first_total_ms','first_execution_with_selection_wall_ms','first_candidate_gpu_ms']:
   base=s['baseline'][key]['median_ms'];auto=s['auto'][key]['median_ms'];oracle=min(s[m][key]['median_ms'] for m in ['baseline','bucket','segment'])
   s['auto'][key].update(speedup_vs_baseline=base/auto,regret=auto/oracle-1,regression_vs_baseline=auto/base-1,oracle_ms=oracle)
   for method in methods:rows.append(dict(graph=graph['id'],strategy=method,metric=key,**s[method][key]))
  summary[graph['id']]=s
 (root/'summary.json').write_text(json.dumps(summary,indent=2))
 fields=['graph','strategy','metric','median_ms','min_ms','max_ms','speedup_vs_baseline','regret','regression_vs_baseline','oracle_ms']
 with (root/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 overview={}
 for key in ['first_total_ms','first_execution_with_selection_wall_ms']:
  vals=[s['auto'][key] for s in summary.values()];overview[key]=dict(graphs=len(vals),mean_regret=st.mean(r['regret'] for r in vals),worst_regression=max(r['regression_vs_baseline'] for r in vals),faster_graphs=sum(r['speedup_vs_baseline']>1 for r in vals))
 overview['note']='Primary first_total includes all selection/setup. Selection-wall is post-H2D completion including features, decision and allocation. Candidate-only GPU time excludes selection, so its regret is diagnostic only. Oracle is minimum of candidate medians, not a realizable free selector; negative sampled regret may arise from noise.'
 (root/'overview.json').write_text(json.dumps(overview,indent=2));print(json.dumps(overview,indent=2))
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
 except ImportError:return
 names=list(summary);fig,axes=plt.subplots(2,1,figsize=(max(7,len(names)*.8),7))
 for ax,key in zip(axes,['first_total_ms','first_execution_with_selection_wall_ms']):
  ax.bar(names,[summary[g]['auto'][key]['speedup_vs_baseline'] for g in names]);ax.axhline(1,color='black',ls='--');ax.set_title(key);ax.set_ylabel('Speedup over baseline');ax.tick_params(axis='x',rotation=30)
 fig.tight_layout();fig.savefig(root/'speedup.png',dpi=180);fig.savefig(root/'speedup.pdf');plt.close(fig)
if __name__=='__main__':main()
