"""Collect single-process calibration or independent-process frozen evaluation."""
import argparse,hashlib,json,math,os,random,signal,statistics,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def capture(cmd):
 try:
  r=subprocess.run(cmd,text=True,capture_output=True,timeout=15);return r.stdout if r.returncode==0 else 'unavailable'
 except (OSError,subprocess.TimeoutExpired):return 'unavailable'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['calibrate','evaluate']);p.add_argument('--manifest');p.add_argument('--inputs',nargs='+');p.add_argument('--model');p.add_argument('--out',required=True);p.add_argument('--build',default='build')
 p.add_argument('--threads',type=int,default=12);p.add_argument('--blocks',type=int,default=864);p.add_argument('--warmup',type=int,default=3);p.add_argument('--repeat',type=int,default=15);p.add_argument('--rounds',type=int,default=3);p.add_argument('--runs',type=int,default=10);p.add_argument('--timeout',type=int,default=1800);p.add_argument('--seed',type=int,default=42);p.add_argument('--allow-development-overlap',action='store_true');p.add_argument('--allow-debug',action='store_true');a=p.parse_args()
 if min(a.threads,a.blocks,a.repeat,a.rounds,a.runs,a.timeout)<1 or a.warmup<0:p.error('invalid run controls')
 graphs=[];manifest=None
 if a.action=='calibrate':
  if not a.manifest or a.inputs or a.model:p.error('calibrate needs --manifest only')
  source=Path(a.manifest).resolve();manifest=json.loads(source.read_text())
  for row in manifest['graphs']:
   graph=dict(row);graph['input']=str((source.parent/row['path']).resolve());graphs.append(graph)
  exe=Path(a.build).resolve()/'tc_schedule_calibrate'
 else:
  if not a.inputs or not a.model or a.manifest:p.error('evaluate needs --inputs and --model')
  model=Path(a.model).resolve();info_path=model.with_suffix('.json')
  if not model.is_file() or not info_path.is_file():p.error('need model and matching .json provenance')
  info=json.loads(info_path.read_text())
  if info.get('debug_tiny') and not a.allow_debug:p.error('debug model cannot be used for reported evaluation')
  if info['model_sha256']!=sha(model):p.error('model provenance hash mismatch')
  hashes={sha(Path(x)) for x in a.inputs};overlap=hashes.intersection(info['development_input_sha256'])
  if overlap and not a.allow_development_overlap:p.error('development data overlap; use --allow-development-overlap for a tagged retest')
  graphs=[dict(id=Path(x).stem,input=str(Path(x).resolve())) for x in a.inputs];exe=Path(a.build).resolve()/'tc_schedule_auto'
 if len({g['id'] for g in graphs})!=len(graphs):p.error('duplicate graph IDs')
 if not exe.is_file():p.error('compile the runner first')
 for g in graphs:
  if not Path(g['input']).is_file():p.error('missing '+g['input'])
  g['actual_sha256']=sha(g['input'])
  if g.get('sha256',g['actual_sha256'])!=g['actual_sha256']:p.error('input hash mismatch')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);jobs=[];rng=random.Random(a.seed)
 common=['--parser','omp','--mapping','omp','--sort','omp','--threads',str(a.threads),'--blocks',str(a.blocks),'--verify','--warmup',str(a.warmup),'--repeat',str(a.repeat),'--seed',str(a.seed)]
 if a.action=='calibrate':
  shuffled=graphs[:];rng.shuffle(shuffled)
  jobs=[dict(id=g['id'],graph=g,command=[str(exe),g['input'],*common,'--rounds',str(a.rounds)]) for g in shuffled]
 else:
  # Auto always executes ONE candidate. Forced candidates below are evaluation oracles only.
  for run in range(a.runs):
   work=[(g,s) for g in graphs for s in ['baseline','bucket','segment','auto']];rng.shuffle(work)
   for g,s in work:jobs.append(dict(id=f'{g["id"]}.{s}.{run:03d}',graph=g,strategy=s,run=run,command=[str(exe),g['input'],*common,'--strategy',s,'--model',str(model)]))
 metadata=dict(schema='schedule-experiments-v1',arguments=vars(a),graphs=graphs,jobs=jobs,debug_tiny=bool(manifest and manifest.get('debug_tiny')),
               commit=capture(['git','-C',str(ROOT),'rev-parse','HEAD']).strip(),worktree=capture(['git','-C',str(ROOT),'status','--porcelain']),
               executable_sha256=sha(exe),model_sha256=sha(a.model) if a.model else None,
               cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
               resource_tool='/usr/bin/time -v' if Path('/usr/bin/time').is_file() else 'unavailable',
               cache_policy='Inputs hashed and pre-read; OS page cache not flushed.',
               metric_note='Calibration GPU events include per-run bucket/queue preparation and reduction, resident allocations excluded. Evaluation first_total_ms includes preprocessing, setup, features, selection, candidate allocation, preparation and first result transfer. Steady samples do not repeat selection.')
 (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 for name,cmd in [('nvidia-smi.txt',['nvidia-smi','-q']),('cpu.txt',['lscpu']),('memory.txt',['free','-h'])]:(out/name).write_text(capture(cmd))
 env=os.environ.copy();env.update(OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE');states=[];identities={};fingerprint=None
 for i,job in enumerate(jobs):
  print(f'{i+1}/{len(jobs)} {job["id"]}',flush=True);cmd=job['command']
  if Path('/usr/bin/time').is_file():cmd=['/usr/bin/time','-v','-o',str(out/(job['id']+'.resource.txt')),*cmd]
  try:
   with (out/(job['id']+'.json')).open('w') as stdout,(out/(job['id']+'.stderr')).open('w') as stderr:
    proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
    try:proc.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise
   if proc.returncode:raise RuntimeError(f'exit {proc.returncode}')
   row=json.loads((out/(job['id']+'.json')).read_text())
   if not row['verified'] or row['blocks']!=a.blocks or row['threads_per_block']!=256 or row['preprocess_options']['threads_requested']!=a.threads:raise ValueError('verification/config mismatch')
   if row['warmup']!=a.warmup or row['repeat']!=a.repeat:raise ValueError('timing protocol mismatch')
   fp=tuple(row[k] for k in ['schema','device','major','minor','sms','cuda_runtime_version','cuda_driver_version','blocks','segment_length','queue_mib'])
   if fingerprint is not None and fp!=fingerprint:raise ValueError('hardware/configuration changed')
   fingerprint=fp;identity=tuple(row[k] for k in ['vertices','undirected_edges','triangles']);gid=job['graph']['id']
   if gid in identities and identities[gid]!=identity:raise ValueError('candidate count mismatch')
   identities[gid]=identity
   if 'expected_clean_graph' in job['graph']:
    e=job['graph']['expected_clean_graph']
    if identity!=tuple(e[k] for k in ['vertices','undirected_edges','triangles']):raise ValueError('analytic counts mismatch')
   if a.action=='calibrate':
    expected={(s,p,r) for s,p in [('baseline',0),('bucket',0),('segment',0),('segment',1),('segment',2)] for r in range(a.rounds)}
    actual={(x['strategy'],x['policy'],x['round']) for x in row['measurements']}
    if len(row['measurements'])!=len(expected) or actual!=expected:raise ValueError('missing calibration measurements')
    if any(len(x['gpu_ms_samples'])!=a.repeat or any(not math.isfinite(v) or v<=0 for v in x['gpu_ms_samples']) or abs(statistics.median(x['gpu_ms_samples'])-x['gpu_ms'])>1e-6 for x in row['measurements']):raise ValueError('invalid calibration timing samples')
   elif row['requested_strategy']!=job['strategy'] or len(row['candidate_gpu_ms_samples'])!=a.repeat:raise ValueError('wrong execution variant')
   states.append(dict(id=job['id'],status='ok'))
  except Exception as error:
   states.append(dict(id=job['id'],status='failed',error=str(error)));(out/'status.json').write_text(json.dumps(states,indent=2));raise
  (out/'status.json').write_text(json.dumps(states,indent=2))
 (out/'COMPLETE').write_text('All recorded jobs passed verification.\n');print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
