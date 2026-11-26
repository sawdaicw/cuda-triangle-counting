"""Reproducible experiments for local cooperation and global query allocation."""
import datetime
import argparse,hashlib,json,math,os,random,signal,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def variants(a):
 edge=('tc_gpu_edge_baseline',[]);out={'edge':edge}
 local=lambda mode,k,r,cut=0:('tc_gpu_query_v2',['--query-mode',mode,'--budget',str(k),'--remaining-limit',str(r),'--bypass-edges',str(cut)])
 glob=lambda b,cut=0:('tc_gpu_query_global',['--chunk',str(b),'--bypass-edges',str(cut)])
 if a.stage=='k':out.update({f'k{k}':local('budget',k,32) for k in a.ks})
 elif a.stage=='r':
  out['budget']=local('budget',a.k,32);out.update({f'r{r}':local('gated',a.k,r) for r in a.rs})
 elif a.stage=='b':out.update({f'b{b}':glob(b) for b in a.bs})
 elif a.stage=='bypass':
  for cut in a.cutoffs:out[f'local_m{cut}']=local('gated',a.k,a.r,cut);out[f'global_m{cut}']=glob(a.b,cut)
 elif a.stage in ['final','stress']:
  if not a.frozen:raise ValueError('final/stress requires --frozen, selected before evaluation')
  f=json.loads(Path(a.frozen).read_text())
  if f.get('frozen') is not True:raise ValueError('example config is not a frozen selection')
  l=f['local'];g=f['global']
  if l['mode'] not in ['budget','gated'] or not 0<=l['k']<=65535 or not 0<=l['r']<=32 or not 1<=g['chunk']<=65535:raise ValueError('bad frozen parameters')
  if any(not 0<=v<=2147483647 for v in [l['bypass_edges'],g['bypass_edges']]):raise ValueError('bad cutoff')
  out['local']=local(l['mode'],l['k'],l['r'],l['bypass_edges']);out['global']=glob(g['chunk'],g['bypass_edges'])
 elif a.stage=='profile':out.update(local=local('gated',a.k,a.r),global_query=glob(a.b))
 return out

def summarize(records):
 out={}
 for graph,methods in records.items():
  out[graph]={}
  for method,rows in methods.items():
   if not rows:continue
   entry={'n':len(rows),'vertices':rows[0]['vertices'],'undirected_edges':rows[0]['undirected_edges'],'triangles':rows[0]['triangles']}
   for metric in ['first_total_ms','first_count_reduce_ms','count_reduce_ms_median']:
    vals=[r[metric] for r in rows];base=[r[metric] for r in methods['edge']]
    entry[metric]={'median_ms':statistics.median(vals),'min_ms':min(vals),'max_ms':max(vals),
                   'speedup_vs_edge':statistics.median(base)/statistics.median(vals) if base else None}
   entry['max_device_buffer_bytes']=max(r['device_buffer_bytes'] for r in rows)
   out[graph][method]=entry
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=['k','r','b','bypass','final','stress','profile'],required=True)
 p.add_argument('--build',default='build');p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
 p.add_argument('--threads',type=int,default=12);p.add_argument('--blocks',type=int,default=864)
 p.add_argument('--runs',type=int);p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeat',type=int,default=30)
 p.add_argument('--timeout',type=int,default=1800);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--ks',type=int,nargs='+',default=[0,2,4,8,16,32,64]);p.add_argument('--rs',type=int,nargs='+',default=[0,4,8,12,16,24,32]);p.add_argument('--bs',type=int,nargs='+',default=[1,2,4,8,16,32])
 p.add_argument('--cutoffs',type=int,nargs='+',default=[0,10000,100000,1000000])
 p.add_argument('--k',type=int,default=8);p.add_argument('--r',type=int,default=16);p.add_argument('--b',type=int,default=8)
 p.add_argument('--frozen');p.add_argument('--allow-development-overlap',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();a.runs=a.runs if a.runs is not None else (20 if a.stage=='final' else 3)
 if not (1<=a.runs and 0<=a.warmup<=10000 and 1<=a.repeat<=10000 and 1<=a.threads<=256 and 1<=a.blocks<=2147483647 and a.timeout>0):p.error('invalid run/thread/grid controls')
 for values,lo,hi in [(a.ks,0,65535),(a.rs,0,32),(a.bs,1,65535),(a.cutoffs,0,2147483647)]:
  if len(set(values))!=len(values) or any(not lo<=v<=hi for v in values):p.error('invalid parameter sweep')
 if not(0<=a.k<=65535 and 0<=a.r<=32 and 1<=a.b<=65535):p.error('invalid fixed parameters')
 try:v=variants(a)
 except (ValueError,KeyError,OSError) as e:p.error(str(e))
 inputs=[Path(x).resolve() for x in a.inputs];build=Path(a.build).resolve()
 if len({x.name for x in inputs})!=len(inputs) or any(not x.is_file() for x in inputs):p.error('missing input or duplicate basename')
 if a.frozen and a.stage in ['final','stress']:
  dev=set(json.loads(Path(a.frozen).read_text()).get('development_inputs',[]))
  if dev.intersection(x.name for x in inputs) and not a.allow_development_overlap:p.error('development inputs overlap; tag as retest with --allow-development-overlap')
 jobs=[];rng=random.Random(a.seed)
 common=['--parser','omp','--mapping','omp','--sort','omp','--threads',str(a.threads),'--blocks',str(a.blocks),'--verify','--warmup',str(a.warmup),'--repeat',str(a.repeat)]
 if a.stage=='profile':common+=['--profile-gpu']
 for run in range(a.runs):
  work=[(x,m) for x in inputs for m in v];rng.shuffle(work)
  for path,method in work:
   exe,flags=v[method];jobs.append({'input':str(path),'method':method,'run':run+1,'stem':f'{path.name}.{method}.{run+1:03d}',
                                  'command':[str(build/exe),str(path),*common,*flags]})
 if a.dry_run:print(json.dumps({'stage':a.stage,'jobs':jobs,'processes':len(jobs)},indent=2));return
 if any(not (build/exe).is_file() for exe,flags in v.values()):p.error('build required executables first')
 output=Path(a.out);output.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy();env['OMP_DYNAMIC']='FALSE';env['OMP_WAIT_POLICY']='PASSIVE'
 def capture(command):
  try:return subprocess.run(command,capture_output=True,text=True,timeout=20).stdout
  except (OSError,subprocess.TimeoutExpired):return 'unavailable'
 manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':a.stage,'arguments':vars(a),'variants':v,'jobs':jobs,'commit':capture(['git','-C',str(ROOT),'rev-parse','HEAD']).strip(),
           'worktree':capture(['git','-C',str(ROOT),'status','--porcelain']),
           'source_archive_commit':(ROOT/'SOURCE_COMMIT.txt').read_text().strip() if (ROOT/'SOURCE_COMMIT.txt').exists() else None,
           'input_sha256':{str(x):sha(x) for x in inputs},'executable_sha256':{exe:sha(build/exe) for exe,flags in v.values()},
           'frozen_sha256':sha(a.frozen) if a.frozen else None,
           'cache_policy':'not flushed; hashing pre-reads inputs; not cold-disk measurements',
           'resource_tool':'/usr/bin/time -v' if Path('/usr/bin/time').exists() else 'unavailable: no peak RSS collected',
           'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
           'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
           'metrics':{'primary':'first_total_ms','secondary':'first_count_reduce_ms includes global index rebuild',
                      'supplemental':'count_reduce_ms_median: within-process post-warmup median; then median across processes'},
           'environment':{k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_DYNAMIC','OMP_WAIT_POLICY','OMP_PROC_BIND','OMP_PLACES']}}
 (output/'metadata.json').write_text(json.dumps(manifest,indent=2))
 for name,cmd in [('nvidia-smi.txt',['nvidia-smi','-q']),('cpu.txt',['lscpu']),('memory.txt',['free','-h'])]:(output/name).write_text(capture(cmd))
 records={x.name:{m:[] for m in v} for x in inputs};states=[];identities={};devices=set()
 for i,job in enumerate(jobs):
  print(f"{i+1}/{len(jobs)} {Path(job['input']).name} {job['method']}",flush=True)
  command=job['command'];resource=output/(job['stem']+'.resource.txt')
  if Path('/usr/bin/time').exists():command=['/usr/bin/time','-v','-o',str(resource),*command]
  try:
   with (output/(job['stem']+'.json')).open('w') as stdout,(output/(job['stem']+'.stderr')).open('w') as stderr:
    process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
    try:process.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGKILL);process.wait();raise
    result=process
   if result.returncode:raise RuntimeError(f'exit {result.returncode}')
   row=json.loads((output/(job['stem']+'.json')).read_text());name=Path(job['input']).name
   ident=tuple(row[k] for k in ['vertices','undirected_edges','triangles'])
   if name in identities and identities[name]!=ident:raise RuntimeError('graph/count mismatch')
   identities[name]=ident
   if row['verified'] is not True or row['blocks']!=a.blocks or row['threads_per_block']!=256:raise RuntimeError('verification/grid mismatch')
   if row['warmup']!=a.warmup or row['repeat']!=a.repeat or len(row['count_reduce_ms_samples'])!=a.repeat:raise RuntimeError('repeat protocol mismatch')
   if job['method']!='edge':
    flags=v[job['method']][1]
    for flag,key in [('--chunk','query_chunk'),('--budget','query_budget'),('--remaining-limit','remaining_limit'),('--bypass-edges','bypass_threshold_edges')]:
     if flag in flags and row.get(key)!=int(flags[flags.index(flag)+1]):raise RuntimeError(f'{flag} mismatch')
    if '--query-mode' in flags and row.get('query_mode')!=flags[flags.index('--query-mode')+1]:raise RuntimeError('mode mismatch')
   for key in ['first_total_ms','first_count_reduce_ms','count_reduce_ms_median']:
    if not math.isfinite(row[key]) or row[key]<=0:raise RuntimeError('invalid timing')
   devices.add(tuple(row[k] for k in ['device','compute_capability','cuda_runtime_version','cuda_driver_version']))
   if len(devices)>1:raise RuntimeError('device/runtime changed')
   records[name][job['method']].append(row);states.append(dict(stem=job['stem'],status='ok'))
  except Exception as e:
   states.append(dict(stem=job['stem'],status='failed',error=str(e)))
   (output/'status.json').write_text(json.dumps(states,indent=2));(output/'summary.json').write_text(json.dumps(summarize(records),indent=2));raise
  (output/'status.json').write_text(json.dumps(states,indent=2));(output/'summary.json').write_text(json.dumps(summarize(records),indent=2))
 print(f'COMPLETE: {output}/summary.json',flush=True)
if __name__=='__main__':main()
