from pathlib import Path
import json,collections,statistics as st,math,csv,hashlib
import numpy as np
root=Path(__file__).resolve().parents[1]
src=root/'work/input/results/raw/a100_full_v1'
out=root/'summary';out.mkdir(exist_ok=True)
meta=json.loads((src/'metadata.json').read_text());catalog=json.loads((root/'work/input/data/benchmark.json').read_text())
by_name={Path(x['path']).name:x for x in catalog['datasets'] if x['tier']=='core'}
variants=['edge','warp','split16','split32','split64'];groups=collections.defaultdict(list)
for p in src.glob('*.json'):
 if p.name in ['metadata.json','commands.json','summary.json']:continue
 name,v,index=p.stem.rsplit('.',2);row=json.loads(p.read_text());groups[name,v].append((int(index),row))
assert len(groups)==70
for key,values in groups.items():
 assert sorted(i for i,_ in values)==list(range(1,21)),key
 values.sort()
commands=json.loads((src/'commands.json').read_text());assert len(commands)==1400 and len({x['sample'] for x in commands})==1400
stderr=[p.name for p in src.glob('*.stderr') if p.stat().st_size];assert not stderr
summary=json.loads((src/'summary.json').read_text())
assert len(summary)==14
identities={};blocks={};startup=[];shares=[];bootstrap={};rng=np.random.default_rng(20261112)
fields=['first_total_ms','first_count_reduce_ms','preprocess_ms','task_metadata_ms','cuda_setup_ms','h2d_ms','first_execution_wall_ms','first_d2h_ms','verify_ms']
records=[];phase=[]
for name,case in by_name.items():
 expected={'triangles':case.get('reference_triangles')}
 if case['kind']=='generate':
  p=case['parameters'];family=case['family']
  if family=='clique_chain':
   c,k=p['cliques'],p['size'];expected=dict(vertices=c*k,undirected_edges=c*math.comb(k,2)+c-1,triangles=c*math.comb(k,3))
  elif family=='grid':
   a,b=p['rows'],p['cols'];expected=dict(vertices=a*b,undirected_edges=a*(b-1)+(a-1)*b,triangles=0)
  elif family=='shared_hub':
   g,k=p['groups'],p['size'];expected=dict(vertices=1+g*k,undirected_edges=g*(math.comb(k,2)+k),triangles=g*math.comb(k+1,3))
  elif family=='mixed_cliques':
   a,b,k,l=p['small_count'],p['large_count'],p['small_size'],p['large_size'];expected=dict(vertices=a*k+b*l,undirected_edges=a*math.comb(k,2)+b*math.comb(l,2),triangles=a*math.comb(k,3)+b*math.comb(l,3))
  else:raise ValueError(family)
 else:expected['undirected_edges']=case['reference_clean_edges']
 all_identity=set()
 indices=rng.integers(0,20,size=(10000,20))
 for v in variants:
  rows=[r for _,r in groups[name,v]]
  for row in rows:
   assert row['verified'] is True
   assert row['device']=='NVIDIA A100-PCIE-40GB' and row['compute_capability']=='8.0'
   assert row['cuda_runtime_version']==12080 and row['cuda_driver_version']==12080
   assert row['warmup']==0 and row['repeat']==1 and len(row['count_reduce_ms_samples'])==1
   assert row['threads_per_block']==256
   pp=row['preprocess_options']
   assert [pp[x] for x in ['parser','mapping','sort']]==['omp']*3
   assert pp['threads_requested']==12
   assert all(pp[x]==12 for x in ['parser_threads_actual','mapping_threads_actual','sorting_threads_max_actual'])
   assert not row.get('gpu_stage_profiled',False)
   assert all(math.isfinite(row[f]) and row[f]>=0 for f in fields)
   assert all(row[k]==value for k,value in expected.items()),(name,v,expected,row)
   all_identity.add(tuple(row[k] for k in ['vertices','undirected_edges','triangles']))
  blocks[name,v]=sorted({r['blocks'] for r in rows})
  if v.startswith('split'):
   assert len({r['heavy_edges'] for r in rows})==1
   for row in rows:
    assert row['threshold']==int(v[5:])
    assert row['heavy_edges']+row['light_edges_including_empty']==row['undirected_edges']
    assert row['queue_buffer_bytes']==8*max(row['undirected_edges'],1)+8
  base=[r for _,r in groups[name,'edge']]
  record={'id':case['id'],'graph':name,'variant':v,'n':20,'vertices':rows[0]['vertices'],'undirected_edges':rows[0]['undirected_edges'],'triangles':rows[0]['triangles'],'blocks':rows[0]['blocks']}
  for f in fields:
   vals=[r[f] for r in rows]
   record[f+'_median']=st.median(vals);record[f+'_min']=min(vals);record[f+'_max']=max(vals)
  for metric in ['first_total_ms','first_count_reduce_ms']:
   vals=np.array([r[metric] for r in rows]);b=np.array([r[metric] for r in base]);ratio=float(np.median(b)/np.median(vals))
   assert np.isclose(st.median(vals),summary[name][v][metric]['median_ms'],rtol=1e-10)
   assert np.isclose(ratio,summary[name][v][metric]['speedup_vs_edge'],rtol=1e-10)
   samples=np.median(b[indices],axis=1)/np.median(vals[indices],axis=1)
   low,high=np.quantile(samples,[.025,.975])
   record[metric+'_speedup']=ratio;record[metric+'_speedup_ci_low']=float(low);record[metric+'_speedup_ci_high']=float(high)
  record['heavy_edges']=rows[0].get('heavy_edges','');record['heavy_edge_fraction']=rows[0].get('heavy_edge_fraction','')
  record['repeat_gpu_median_ms']=st.median(r['count_reduce_ms_samples'][0] for r in rows)
  records.append(record)
  if v=='edge':
   startup.extend(r['cuda_setup_ms'] for r in rows);shares.extend(r['first_count_reduce_ms']/r['first_total_ms']*100 for r in rows)
   phase.append({'id':case['id'],'graph':name,'preprocess_mean':st.mean(r['preprocess_ms'] for r in rows),'cuda_setup_mean':st.mean(r['cuda_setup_ms'] for r in rows),'other_mean':st.mean(r['first_total_ms']-r['preprocess_ms']-r['cuda_setup_ms'] for r in rows),'total_mean':st.mean(r['first_total_ms'] for r in rows)})
 assert len(all_identity)==1
 identities[name]=next(iter(all_identity))
with (out/'all_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
with (out/'baseline_phase_means.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(phase[0]));w.writeheader();w.writerows(phase)
aggregate={}
for metric in ['first_total_ms','first_count_reduce_ms']:
 aggregate[metric]={}
 for v in variants[1:]:
  vals=[r[metric+'_speedup'] for r in records if r['variant']==v]
  aggregate[metric][v]={'geomean_speedup':math.exp(st.mean(math.log(x) for x in vals)),'point_estimate_wins':sum(x>1 for x in vals)}
check={'raw_samples':1400,'groups':70,'samples_per_group':20,'commands':len(commands),'verified_true':1400,'nonempty_stderr':stderr,'identity_consistent':True,'matches_reference_counts':True,'summary_recomputed_matches':True,'preprocess':'omp/omp/omp, requested and actual 12 threads','environment':meta['environment'],'source':meta['archive_source_commit'],'patch':(root/'work/input/BENCHMARK_PATCHES.txt').read_text(),'gpu':'A100-PCIE-40GB, CC8.0, CUDA runtime/driver12080','gpu_stage_profiled':False,'cuda_setup_median_all_edge_samples_ms':st.median(startup),'gpu_fraction_per_run_median_percent':st.median(shares),'gpu_fraction_per_run_max_percent':max(shares),'block_counts':{name:{v:blocks[name,v] for v in variants} for name in by_name},'bootstrap':'10000 paired-by-round percentile resamples, ratio of medians, seed20261112, no multiplicity correction; exploratory intervals, not confirmatory testing','limitations':['Input graph bytes not included; hashes are recorded by runner but cannot be rehashed from this archive.','Executable hashes recorded; binaries and full build logs not included.','CPU model, clocks, temperature time series and exclusivity were not established by this archive.','No separate GPU stage profile in this formal batch.','W2 uses394 blocks for edge and864 for warp/split; launch geometry is not held fixed on W2.','One post-first GPU-only repeat per process is separate from20 complete samples.']}
(out/'integrity_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
(out/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
print(json.dumps({'integrity':check,'aggregate':aggregate},ensure_ascii=False,indent=2))
# Compact interval diagnostics; no outliers removed.
for r in records:
 if r['variant'] in ['warp','split16']:
  print(r['id'],r['variant'],'total',round(r['first_total_ms_speedup'],3),tuple(round(r['first_total_ms_speedup_ci_'+k],3) for k in ['low','high']),'gpu',round(r['first_count_reduce_ms_speedup'],3),tuple(round(r['first_count_reduce_ms_speedup_ci_'+k],3) for k in ['low','high']))
