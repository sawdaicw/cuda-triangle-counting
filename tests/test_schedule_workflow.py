"""End-to-end fake-device training/model/evaluation protocol, never performance evidence."""
import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FAKE=r'''#!/usr/bin/env python3
import sys,json,math
from pathlib import Path
sys.path.insert(0,__ROOT__+'/python')
from reference import normalize,count_bruteforce
args=sys.argv[2:]
def v(flag,default):return args[args.index(flag)+1] if flag in args else default
records=[tuple(map(int,line.split())) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip() and not line.startswith(('#','%'))]
nodes,edges=normalize(records);m=len(edges);cal='calibrate' in Path(sys.argv[0]).name
repeat=int(v('--repeat',2));base=.05+m*.001
row=dict(schema='schedule-v1',kind='calibration' if cal else 'execution',device='FAKE DEVICE',major=0,minor=0,sms=1,cuda_runtime_version=0,cuda_driver_version=0,blocks=int(v('--blocks',3)),threads_per_block=256,vertices=len(nodes),undirected_edges=m,triangles=count_bruteforce(records),verified=True,segment_length=128,queue_mib=64,gate_policy=0,warmup=int(v('--warmup',1)),repeat=repeat,feature_gpu_ms=.003,feature_wall_ms=.005,cheap_wall_ms=.0005,cheap_work=m*100,cheap_max_cost=100,cheap_queries=m*10,features=dict(m=m,queries=m*10,work=m*100,warp_work=m*15,tail_work=100,max_cost=100,max_queries=10,lane_eff=.2,warp_eff=.8,segment_tasks=[10,5,0],split_edges=[2,1,0]),preprocess_options=dict(threads_requested=int(v('--threads',2))))
if cal:
 row['measurements']=[]
 for round in range(int(v('--rounds',2))):
  for strategy,policy in [('baseline',0),('bucket',0),('segment',0),('segment',1),('segment',2)]:
   time=base*({'baseline':1,'bucket':.9,'segment':.8}[strategy])+.002*policy
   row['measurements'].append(dict(strategy=strategy,policy=policy,round=round,gpu_ms=time,prepare_ms_mean=.001,count_ms_mean=time-.002,reduce_ms_mean=.001,gpu_ms_samples=[time]*repeat))
else:
 strategy=v('--strategy','auto');selected='baseline' if strategy=='auto' else strategy
 row.update(requested_strategy=strategy,selected_strategy=selected,coarse_bypass=False,first_total_ms=100+base,first_execution_with_selection_wall_ms=base+.005,first_candidate_gpu_ms=base,candidate_gpu_ms_samples=[base]*repeat)
print(json.dumps(row))
'''
class ScheduleWorkflow(unittest.TestCase):
 def test_training_fit_and_frozen_evaluation(self):
  if not importlib.util.find_spec('numpy'):self.skipTest('numpy required for model fitting')
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);data=root/'data';build=root/'build';build.mkdir()
   gen=[sys.executable,str(ROOT/'scripts/generate_schedule_training.py'),'--out',str(data),'--tiny','--relabel-ids','g00']
   subprocess.run(gen,check=True,capture_output=True);manifest=json.loads((data/'manifest.json').read_text());self.assertEqual(len(manifest['graphs']),37)
   self.assertEqual(manifest['graphs'][0]['group'],manifest['graphs'][1]['group']);self.assertEqual(manifest['graphs'][0]['split'],manifest['graphs'][1]['split'])
   for name in ['tc_schedule_calibrate','tc_schedule_auto']:
    p=build/name;p.write_text(FAKE.replace('__ROOT__',repr(str(ROOT))));p.chmod(0o755)
   run=[sys.executable,str(ROOT/'scripts/schedule_experiments.py')]
   cal=root/'calibration';cmd=run+['calibrate','--manifest',str(data/'manifest.json'),'--build',str(build),'--out',str(cal),'--threads','2','--blocks','3','--repeat','2','--rounds','2','--warmup','1']
   r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   self.assertEqual(len(json.loads((cal/'status.json').read_text())),37)
   fit=[sys.executable,str(ROOT/'scripts/fit_schedule_model.py'),str(cal),'--out',str(root/'model.txt')]
   self.assertNotEqual(subprocess.run(fit,capture_output=True).returncode,0)
   r=subprocess.run(fit+['--allow-debug'],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   model=json.loads((root/'model.json').read_text());self.assertEqual(len(model['selected']['coefficients']),3);self.assertEqual(len(model['selected']['coefficients'][0]),5);self.assertTrue((root/'model.calibration.csv').is_file());self.assertTrue((root/'model.validation.csv').is_file())
   # Compile the actual C++ reader to catch Python/C++ format drift.
   reader=root/'reader.cpp';reader.write_text('#include "'+str(ROOT/'src/common/schedule_policy.hpp')+'"\n#include <iostream>\nint main(int argc,char**argv){auto m=tc::read_schedule_model(argv[1]);std::cout<<m.device<<" "<<m.policy;}\n')
   compiler=subprocess.run(['g++','-std=c++17',str(reader),'-o',str(root/'reader')],capture_output=True,text=True);self.assertEqual(compiler.returncode,0,compiler.stderr)
   r=subprocess.run([str(root/'reader'),str(root/'model.txt')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('FAKE DEVICE',r.stdout)
   held=root/'heldout.txt';held.write_text('0 1\n1 2\n0 2\n');out=root/'eval'
   ev=run+['evaluate','--build',str(build),'--inputs',str(held),'--model',str(root/'model.txt'),'--out',str(out),'--threads','2','--blocks','3','--repeat','2','--warmup','1','--runs','2']
   self.assertNotEqual(subprocess.run(ev,capture_output=True).returncode,0)
   r=subprocess.run(ev+['--allow-debug'],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   r=subprocess.run([sys.executable,str(ROOT/'scripts/summarize_schedule.py'),str(out)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   summary=json.loads((out/'summary.json').read_text());self.assertEqual(summary['heldout']['auto']['n'],2);self.assertTrue((out/'summary.csv').is_file())
   # Modified models cannot silently reuse the original provenance.
   (root/'model.txt').write_text((root/'model.txt').read_text()+'\n')
   self.assertNotEqual(subprocess.run(ev+['--allow-debug'],capture_output=True).returncode,0)
if __name__=='__main__':unittest.main()
