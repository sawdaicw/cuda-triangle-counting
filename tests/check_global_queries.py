"""Run on CUDA hardware; host-model success does not replace this check."""
import itertools,json,subprocess,sys,tempfile
from pathlib import Path
import check_serial

def main():
 if len(sys.argv)!=2:raise SystemExit('Usage: python3 tests/check_global_queries.py BUILD/tc_gpu_query_global')
 exe=str(Path(sys.argv[1]).resolve())
 for b in [1,8]:
  sys.argv=[sys.argv[0],exe,'--chunk',str(b),'--verify','--blocks','1','--warmup','0','--repeat','1'];check_serial.main()
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'g.txt'
  for records in [[],[(0,1)],list(itertools.combinations(range(40),2)),[(2*i,2*i+1) for i in range(257)]]:
   path.write_text(''.join(f'{u} {v}\n' for u,v in records));nodes,edges=check_serial.normalize(records)
   adj={u:set() for u in nodes}
   for u,v in edges:adj[u].add(v);adj[v].add(u)
   f={u:{v for v in adj[u] if (len(adj[u]),u)<(len(adj[v]),v)} for u in nodes}
   q=sum(min(len(f[u]),len(f[v])) for u in f for v in f[u])
   for b in [1,2,4,8,16,32,65535]:
    for cutoff in [0,1000000]:
     row=json.loads(subprocess.run([exe,str(path),'--chunk',str(b),'--bypass-edges',str(cutoff),'--blocks','3','--verify','--warmup','1','--repeat','3','--profile-gpu'],capture_output=True,text=True,check=True,timeout=120).stdout)
     assert row['triangles']==check_serial.count_bruteforce(records) and row['verified']
     bypass=len(edges)<cutoff;assert row['bypass_used']==bypass
     assert row['query_index_bytes']==(0 if bypass else 8*(len(edges)+1))
     assert row['total_queries']==(None if bypass else q)
     assert len(row['count_reduce_ms_samples'])==3 and row['index_rebuilt_each_run']
     assert abs(sum(row['first_stages_ms'].values())-row['first_count_reduce_ms'])<max(.003,row['first_count_reduce_ms']*.01)
  for flags in [['--chunk','0'],['--chunk','65536'],['--query-mode','budget'],['--query-diagnostics']]:
   assert subprocess.run([exe,str(path),*flags],capture_output=True).returncode!=0
 print('PASS: global queries, scan rebuild, chunks, empty/tail, bypass, repeat and timing')
if __name__=='__main__':main()
