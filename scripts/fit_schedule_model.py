"""Small nonnegative timing models with grouped synthetic validation; no real-test tuning."""
import argparse,csv,itertools,json,math,statistics
from pathlib import Path
import numpy as np
from schedule_experiments import sha

def nnls_small(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 # Relative-error weighting; ridge only regularizes the tiny coefficient fit.
 weight=1/np.maximum(y,.03);z=x*weight[:,None];target=y*weight
 scale=np.maximum(np.linalg.norm(z,axis=0),1e-12);z=z/scale
 best=None
 for mask in itertools.product([False,True],repeat=x.shape[1]):
  ids=np.flatnonzero(mask);coef=np.zeros(x.shape[1])
  if len(ids):
   zz=z[:,ids];v=np.linalg.lstsq(np.vstack([zz,np.eye(len(ids))*1e-3]),np.r_[target,np.zeros(len(ids))],rcond=None)[0]
   if np.any(v < -1e-10):continue
   coef[ids]=np.maximum(v,0)
  loss=np.sum((z@coef-target)**2)+1e-6*np.sum(coef**2)
  if best is None or loss<best[0]:best=(loss,coef/scale)
 return best[1]

def basis(row):
 f=row['features'];return np.array([1,row['undirected_edges']/1e6,f['work']/1e6,f['warp_work']/1e6,f['tail_work']/1e3])
def cheap_basis(row):return np.array([1,row['undirected_edges']/1e6,row['cheap_work']/1e6,row['cheap_max_cost']/1e3])
def truth(row,policy):
 return np.array([statistics.median(x['gpu_ms'] for x in row['measurements'] if x['strategy']==s and (s!='segment' or x['policy']==policy)) for s in ['baseline','bucket','segment']])
def choose(pred,feature_wall,margin):
 best=int(np.argmin(pred));return best if pred[0]-pred[best]>feature_wall+margin*pred[0] else 0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('calibration');p.add_argument('--out',required=True);p.add_argument('--allow-debug',action='store_true');a=p.parse_args()
 root=Path(a.calibration);meta=json.loads((root/'metadata.json').read_text())
 if meta['arguments']['action']!='calibrate' or not (root/'COMPLETE').is_file():p.error('complete calibration required')
 if meta['debug_tiny'] and not a.allow_debug:p.error('tiny data is for correctness only')
 records=[(g,json.loads((root/(g['id']+'.json')).read_text())) for g in meta['graphs']]
 train=[r for g,r in records if g['split']=='train'];validation=[(g,r) for g,r in records if g['split']=='validation']
 if len(train)<6 or len(validation)<3:p.error('need >=6 train and >=3 validation graphs')
 train_groups={g['group'] for g,r in records if g['split']=='train'}
 if train_groups.intersection(g['group'] for g,r in validation):p.error('base-graph group leakage')
 fingerprint_keys=['schema','device','major','minor','sms','cuda_runtime_version','cuda_driver_version','blocks','segment_length','queue_mib']
 fingerprint={k:train[0][k] for k in fingerprint_keys}
 if any(any(r[k]!=v for k,v in fingerprint.items()) for g,r in records):p.error('hardware/configuration mismatch')
 if any(r['verified'] is not True for g,r in records):p.error('unverified calibration data')
 cheap=nnls_small([cheap_basis(r) for r in train],[truth(r,0)[0] for r in train]);candidates=[]
 for policy in range(3):
  coef=np.stack([nnls_small([basis(r) for r in train],[truth(r,policy)[s] for r in train]) for s in range(3)])
  for coarse,gate,margin in itertools.product([0,1024,8192],[0,.05,.1,.3],[.05,.1,.2]):
   rows=[]
   for g,r in validation:
    t=truth(r,policy);small=r['undirected_edges']<coarse
    sampled=not small
    if not small:small=gate>0 and float(cheap_basis(r)@cheap)<=gate
    pred=coef@basis(r)
    if not r['features']['segment_tasks'][policy]:pred[2]=max(pred[0],pred[1])+1
    selected=0 if small else choose(pred,r['feature_wall_ms']+r['cheap_wall_ms'],margin)
    if selected==2 and not r['features']['segment_tasks'][policy]:selected=0
    overhead=(r['cheap_wall_ms'] if sampled else 0)+(0 if small else r['feature_wall_ms'])
    auto=t[selected]+overhead;oracle=min(t)
    rows.append(dict(id=g['id'],regime=g['regime'],selected=['baseline','bucket','segment'][selected],estimated_auto_ms=float(auto),oracle_ms=float(oracle),baseline_ms=float(t[0]),regret=float(auto/oracle-1),regression_vs_baseline=float(auto/t[0]-1)))
   regrets=[r['regret'] for r in rows];degradations=[max(0,r['regression_vs_baseline']) for r in rows]
   # Penalize regressions and tail errors, rather than maximizing average wins.
   objective=float(np.mean(np.log1p(regrets))+2*np.mean(degradations)+max(degradations))
   candidates.append(dict(objective=objective,policy=policy,coarse_edges=coarse,cheap_gate_ms=gate,margin=margin,coefficients=coef.tolist(),validation=rows))
 best=min(candidates,key=lambda c:c['objective']);out=Path(a.out)
 if out.exists() or out.with_suffix('.json').exists():p.error('model output already exists')
 out.parent.mkdir(parents=True,exist_ok=True)
 r=train[0]
 lines=['TC_SCHEDULE_MODEL_V1',json.dumps(r['device']),f"{r['major']} {r['minor']} {r['sms']} {r['cuda_runtime_version']} {r['cuda_driver_version']} {r['blocks']}",
        f"{r['segment_length']} {r['queue_mib']} {best['policy']} {best['coarse_edges']} {best['margin']} {best['cheap_gate_ms']}",
        ' '.join(format(v,'.17g') for v in cheap)]
 lines+=[' '.join(format(v,'.17g') for v in row) for row in best['coefficients']];out.write_text('\n'.join(lines)+'\n')
 info=dict(schema='schedule-fit-v1',model_sha256=sha(out),fingerprint=fingerprint,debug_tiny=meta['debug_tiny'],train_graphs=len(train),validation_graphs=len(validation),
           development_input_sha256=[g['actual_sha256'] for g,r in records],development_groups=[g['group'] for g,r in records],
           calibration_metadata_sha256=sha(root/'metadata.json'),calibration_output=str(root.resolve()),
           note='5 nonnegative coefficients per strategy; 4 cheap-model coefficients. Edge gate uses exact query count and binary-depth proxy; policy constants are selected from three candidates, not fitted per-edge nanoseconds. Validation regret is an estimate from resident calibration timings plus measured feature wall cost, not a final end-to-end evaluation.',selected=best,
           search=[{k:v for k,v in c.items() if k not in ['coefficients','validation']} for c in candidates])
 out.with_suffix('.json').write_text(json.dumps(info,indent=2))
 calibration_rows=[]
 for g,r in records:
  y=truth(r,best['policy']);pred=np.asarray(best['coefficients'])@basis(r)
  for s,strategy in enumerate(['baseline','bucket','segment']):
   calibration_rows.append(dict(id=g['id'],group=g['group'],split=g['split'],regime=g['regime'],family=g['family'],strategy=strategy,m=r['undirected_edges'],queries=r['features']['queries'],lane_eff=r['features']['lane_eff'],warp_eff=r['features']['warp_eff'],actual_gpu_ms=float(y[s]),predicted_gpu_ms=float(pred[s])))
 for suffix,rows in [('.calibration.csv',calibration_rows),('.validation.csv',best['validation'])]:
  with out.with_suffix(suffix).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
 except ImportError:plt=None
 if plt is not None:
  fig,axes=plt.subplots(1,3,figsize=(12,4))
  for ax,strategy in zip(axes,['baseline','bucket','segment']):
   for split,marker in [('train','o'),('validation','x')]:
    rr=[r for r in calibration_rows if r['strategy']==strategy and r['split']==split]
    ax.scatter([r['actual_gpu_ms'] for r in rr],[max(1e-6,r['predicted_gpu_ms']) for r in rr],marker=marker,label=split)
   lo=min(r['actual_gpu_ms'] for r in calibration_rows);hi=max(r['actual_gpu_ms'] for r in calibration_rows)
   ax.plot([lo,hi],[lo,hi],'k--');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Measured resident GPU ms');ax.set_ylabel('Predicted GPU ms');ax.set_title(strategy);ax.legend()
  fig.tight_layout();fig.savefig(out.with_suffix('.fit.png'),dpi=180);fig.savefig(out.with_suffix('.fit.pdf'));plt.close(fig)
 print('MODEL',out);print('SELECTED',json.dumps({k:best[k] for k in ['policy','coarse_edges','cheap_gate_ms','margin','objective']}))
if __name__=='__main__':main()
