"""Export exact plot data, PDF/PNG curves and paired speedup plots; no winner selection."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');a=p.parse_args()
 root=Path(a.experiment);meta=json.loads((root/'metadata.json').read_text());summary=json.loads((root/'summary.json').read_text())
 states=json.loads((root/'status.json').read_text())
 if len(states)!=len(meta['jobs']) or any(s['status']!='ok' for s in states):p.error('refusing incomplete/failed experiment')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=root/'figures';out.mkdir(exist_ok=True)
 metrics={'first_total_ms':'Full task','first_count_reduce_ms':'First GPU execution (index + count + reduce)', 'count_reduce_ms_median':'Post-warmup GPU execution (index + count + reduce)'}
 fields=['graph','method','vertices','undirected_edges','triangles','n','metric','median_ms','min_ms','max_ms','speedup_vs_edge','max_device_buffer_bytes'];rows=[]
 for graph,methods in summary.items():
  for method,s in methods.items():
   for key in metrics:rows.append(dict(graph=graph,method=method,metric=key,**{k:s[k] for k in fields if k in s},**s[key]))
 with (out/'plot_data.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 def save(fig,name):
  fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
 overview={}
 for method in meta['variants']:
  if method=='edge':continue
  overview[method]={}
  for metric in metrics:
   ss=[methods[method][metric] for methods in summary.values()]
   deltas=[methods['edge'][metric]['median_ms']-methods[method][metric]['median_ms'] for methods in summary.values()]
   overview[method][metric]=dict(graphs=len(ss),faster_graphs=sum(d>0 for d in deltas),geomean_speedup=math.exp(statistics.mean(math.log(s['speedup_vs_edge']) for s in ss)),max_gain_ms=max([0,*deltas]),max_regression_ms=max([0,*[-d for d in deltas]]))
 (out/'overview.json').write_text(json.dumps(dict(methods=overview,note='Point estimates only; faster_graphs does not establish statistical significance. Do not mix development, real and synthetic cohorts when drawing final conclusions.'),indent=2))
 stage=meta['stage'];graphs=list(summary)
 for metric,title in metrics.items():
  fig,axes=plt.subplots(len(graphs),1,figsize=(8,max(3,2.8*len(graphs))),squeeze=False)
  for ax,graph in zip(axes[:,0],graphs):
   methods=summary[graph];ax.axhline(1,color='gray',ls='--',lw=1)
   if stage in ['k','r','b','bypass']:
    groups=[('local_m','Local'),('global_m','Global')] if stage=='bypass' else [({'k':'k','r':'r','b':'b'}[stage],stage.upper())]
    for prefix,label in groups:
     points=sorted((int(m[len(prefix):]),s[metric]['speedup_vs_edge']) for m,s in methods.items() if m.startswith(prefix) and m[len(prefix):].isdigit())
     ax.plot([x for x,y in points],[y for x,y in points],'o-',label=label)
    if stage=='r':ax.axhline(methods['budget'][metric]['speedup_vs_edge'],color='orange',ls=':',label='Budget (always pool remainder)')
    ax.set_xlabel({'k':'Prefix budget K','r':'Remaining-lane limit R','b':'Queries per thread B','bypass':'Bypass threshold M (edges)'}[stage])
   else:
    names=[m for m in methods if m!='edge'];ax.bar(names,[methods[m][metric]['speedup_vs_edge'] for m in names])
   ax.set_title(graph);ax.set_ylabel('Speedup over edge');ax.grid(axis='y',alpha=.2)
   if stage in ['k','r','bypass']:ax.legend()
  fig.suptitle(title+' — ratio of process medians',y=1.001);save(fig,metric+'_speedup')
 if stage=='profile':
  # Means are additive; do not stack separately computed phase medians.
  phase_rows=[]
  for job in meta['jobs']:
   row=json.loads((root/(job['stem']+'.json')).read_text());s=row['first_stages_ms']
   phase_rows.append(dict(graph=Path(job['input']).name,method=job['method'],run=job['run'],index_ms=s['light_count_queue_ms'],count_ms=s['warp_count_ms'],reduce_ms=s['global_reduce_ms']))
  with (out/'phase_samples.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(phase_rows[0]));w.writeheader();w.writerows(phase_rows)
  fig,axes=plt.subplots(len(graphs),1,figsize=(8,max(3,len(graphs)*3)),squeeze=False)
  for ax,graph in zip(axes[:,0],graphs):
   methods=list(summary[graph]);bottom=[0.]*len(methods)
   for phase in ['index_ms','count_ms','reduce_ms']:
    vals=[statistics.mean(r[phase] for r in phase_rows if r['graph']==graph and r['method']==m) for m in methods]
    ax.bar(methods,vals,bottom=bottom,label=phase);bottom=[x+y for x,y in zip(bottom,vals)]
   ax.set_title(graph);ax.set_ylabel('First GPU execution, mean ms');ax.legend()
  save(fig,'gpu_phase_means')
 (out/'provenance.json').write_text(json.dumps({'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256((root/'metadata.json').read_bytes()).hexdigest(),'stage':stage,'note':'Curves are exploratory point estimates; no confidence intervals. Profile uses additional CUDA events. No per-graph best-parameter envelope.'},indent=2))
 print(out)
if __name__=='__main__':main()
