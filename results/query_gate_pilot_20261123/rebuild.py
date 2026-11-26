"""Rebuild CSV/JSON from reported aggregates; individual runs were not supplied."""
import csv,json
from pathlib import Path
root=Path(__file__).resolve().parent;summary={};rows=[]
for line in (root/'reported.txt').read_text().splitlines():
 if not line.strip():continue
 graph,method,blob=line.split(' ',2);record=json.loads(blob)
 if method in summary.setdefault(graph,{}):raise ValueError('duplicate row')
 summary[graph][method]=record
for graph,methods in summary.items():
 if set(methods)!={'edge','budget','gate8','gate16','gate32'}:raise ValueError('missing methods')
 for method,record in methods.items():
  for key in ['first_total_ms','first_count_reduce_ms','count_reduce_ms_median']:
   s=record[key];ratio=methods['edge'][key]['median_ms']/s['median_ms']
   if abs(ratio-s['speedup_vs_edge'])>1e-8:raise ValueError('ratio mismatch')
   rows.append(dict(graph=graph,method=method,metric=key,n=record['n'],**s))
if len(summary)!=14:raise ValueError('expected 14 graphs')
(root/'summary.json').write_text(json.dumps(summary,indent=2))
with (root/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Rebuilt 70 reported aggregate rows (14 graphs x 5 methods).')
