"""Server GPU checks for all three schedules and forced automatic branches."""
import itertools,json,math,subprocess,sys,tempfile
from pathlib import Path
import check_serial

def main():
 if len(sys.argv)!=2:raise SystemExit('Usage: python3 tests/check_schedule.py BUILD_DIR')
 build=Path(sys.argv[1]).resolve();exe=str(build/'tc_schedule_auto');cal=str(build/'tc_schedule_calibrate')
 for strategy in ['baseline','bucket','segment']:
  sys.argv=[sys.argv[0],exe,'--strategy',strategy,'--segment-length','8','--queue-mib','1','--verify','--threads','1','--blocks','1','--warmup','0','--repeat','1'];check_serial.main()
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);path=root/'g.txt';path.write_text(''.join(f'{u} {v}\n' for u,v in itertools.combinations(range(67),2)))
  command=[cal,str(path),'--blocks','3','--segment-length','8','--queue-mib','1','--warmup','1','--repeat','2','--rounds','2','--verify','--threads','2']
  row=json.loads(subprocess.run(command,capture_output=True,text=True,check=True,timeout=120).stdout)
  assert row['triangles']==math.comb(67,3) and row['verified'] and len(row['measurements'])==10
  assert row['features']['queries']==math.comb(67,3)
  for strategy in ['baseline','bucket','segment']:
   m=root/'model.txt'
   coefficients=[[10000.,0,0,0,0] for _ in range(3)];coefficients[['baseline','bucket','segment'].index(strategy)]=[0.]*5
   lines=['TC_SCHEDULE_MODEL_V1',json.dumps(row['device']),f"{row['major']} {row['minor']} {row['sms']} {row['cuda_runtime_version']} {row['cuda_driver_version']} 3",'8 1 0 0 0 0','10000 0 0 0']+[' '.join(map(str,c)) for c in coefficients]
   m.write_text('\n'.join(lines)+'\n')
   out=json.loads(subprocess.run([exe,str(path),'--model',str(m),'--blocks','3','--verify','--warmup','0','--repeat','2','--threads','2'],capture_output=True,text=True,check=True,timeout=120).stdout)
   assert out['selected_strategy']==strategy and out['triangles']==math.comb(67,3)
   assert out['first_total_ms']>=out['first_execution_with_selection_wall_ms'] and out['feature_wall_ms']>=out['feature_gpu_ms']*.8
  # Queue overflow must finish unqueued work locally, without losing or repeating queries.
  path.write_text(''.join(f'{u} {v}\n' for u,v in itertools.combinations(range(192),2)))
  out=json.loads(subprocess.run([exe,str(path),'--strategy','segment','--segment-length','8','--queue-mib','1','--blocks','3','--verify','--warmup','0','--repeat','2','--threads','2'],capture_output=True,text=True,check=True,timeout=120).stdout)
  assert out['triangles']==math.comb(192,3) and out['last_reserved_tasks']>out['queue_capacity_tasks']
  for flags in [['--strategy','unknown'],['--segment-length','0'],['--gate-policy','3'],['--strategy','auto']]:
   assert subprocess.run([exe,str(path),*flags],capture_output=True,timeout=30).returncode!=0
 print('PASS: schedule counts, feature queries, calibration, automatic branches, queue overflow and invalid options')
if __name__=='__main__':main()
