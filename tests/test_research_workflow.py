"""Fake-device orchestration/plot tests. These are not CUDA performance measurements."""
import csv,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FAKE='''#!/usr/bin/env python3
import json,sys
args=sys.argv[2:]
def v(flag,default):return args[args.index(flag)+1] if flag in args else default
mode=v('--query-mode','edge');b=int(v('--chunk',1));time=10 if mode=='edge' and '--chunk' not in args else 8
row=dict(vertices=3,undirected_edges=3,triangles=1,verified=True,device='FAKE TEST DEVICE',compute_capability='0.0',cuda_runtime_version=0,cuda_driver_version=0,blocks=int(v('--blocks',0)),threads_per_block=256,first_total_ms=time*10,first_count_reduce_ms=time,count_reduce_ms_median=time,warmup=int(v('--warmup',0)),repeat=int(v('--repeat',1)),device_buffer_bytes=1024)
row['count_reduce_ms_samples']=[time]*row['repeat']
row.update(query_mode=mode,query_chunk=b,query_budget=int(v('--budget',8)),remaining_limit=int(v('--remaining-limit',16)),bypass_threshold_edges=int(v('--bypass-edges',0)))
if '--profile-gpu' in args:row['first_stages_ms']=dict(light_count_queue_ms=1,warp_count_ms=time-2,global_reduce_ms=1)
print(json.dumps(row))
'''
class ResearchWorkflow(unittest.TestCase):
 def test_stages_and_plots(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);build=root/'build';build.mkdir();path=root/'g.txt';path.write_text('0 1\n1 2\n0 2\n')
   for name in ['tc_gpu_edge_baseline','tc_gpu_query_v2','tc_gpu_query_global']:
    exe=build/name;exe.write_text(FAKE);exe.chmod(0o755)
   base=[sys.executable,str(ROOT/'scripts/run_research.py'),'--build',str(build),'--inputs',str(path),'--runs','2','--repeat','3','--warmup','1','--blocks','3']
   specs=[('k',['--ks','0','8'],3),('r',['--rs','0','32'],4),('b',['--bs','1','8'],3),('bypass',['--cutoffs','0','100'],5),('profile',[],3)]
   for stage,extra,n in specs:
    out=root/stage;cmd=base+['--stage',stage,'--out',str(out),*extra]
    dry=subprocess.run(cmd+['--dry-run'],capture_output=True,text=True,check=True)
    self.assertEqual(json.loads(dry.stdout)['processes'],n*2);self.assertFalse(out.exists())
    result=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
    summary=json.loads((out/'summary.json').read_text())['g.txt'];self.assertEqual(len(summary),n)
    for s in summary.values():self.assertEqual(s['n'],2)
    meta=json.loads((out/'metadata.json').read_text());self.assertEqual(len(meta['input_sha256']),1)
    self.assertEqual(len(list(out.glob('*.resource.txt'))),n*2 if Path('/usr/bin/time').exists() else 0)
    if importlib.util.find_spec('matplotlib'):
     result=subprocess.run([sys.executable,str(ROOT/'scripts/plot_research.py'),str(out)],capture_output=True,text=True)
     self.assertEqual(result.returncode,0,result.stderr)
     with (out/'figures/plot_data.csv').open() as stream:rows=list(csv.DictReader(stream))
     self.assertEqual(len(rows),n*3)
     self.assertEqual(len(list((out/'figures').glob('*.pdf'))),4 if stage=='profile' else 3)
    self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
   frozen=root/'freeze.json';f=json.loads((ROOT/'configs/research_freeze.example.json').read_text());f['development_inputs']=['g.txt'];frozen.write_text(json.dumps(f))
   cmd=base+['--stage','final','--frozen',str(frozen),'--out',str(root/'final')]
   self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
   f['frozen']=True;frozen.write_text(json.dumps(f))
   self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
   result=subprocess.run(cmd+['--allow-development-overlap'],capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
   # An actual failed process is persisted, never silently skipped.
   (build/'tc_gpu_query_global').write_text('#!/usr/bin/env python3\nraise SystemExit(9)\n')
   bad=root/'failed';result=subprocess.run(base+['--stage','b','--bs','1','--out',str(bad)],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertEqual(json.loads((bad/'status.json').read_text())[-1]['status'],'failed')
   if importlib.util.find_spec('matplotlib'):
    self.assertNotEqual(subprocess.run([sys.executable,str(ROOT/'scripts/plot_research.py'),str(bad)],capture_output=True).returncode,0)
 def test_nested_scales(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);path=root/'g.txt';path.write_text('0 1\n1 2\n2 0\n2 3\n3 4\n')
   out=root/'scales';cmd=[sys.executable,str(ROOT/'scripts/make_scale_graphs.py'),str(path),'--sizes','2','4','--out',str(out)]
   subprocess.run(cmd,check=True,capture_output=True)
   manifest=json.loads((out/'manifest.json').read_text());self.assertEqual([g['vertices'] for g in manifest['graphs']],[2,4])
   selected=[]
   for g in manifest['graphs']:
    es=[tuple(map(int,line.split())) for line in Path(g['path']).read_text().splitlines() if not line.startswith('#')]
    selected.append({u for u,v in es if u==v})
    self.assertTrue(all(u in selected[-1] and v in selected[-1] for u,v in es))
   self.assertTrue(selected[0]<selected[1])
if __name__=='__main__':unittest.main()
