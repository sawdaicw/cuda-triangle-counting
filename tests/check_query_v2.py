"""CUDA server validation for specialized budget/score kernels; not a host test."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import check_serial


def expected_stats(records,mode,budget=8,small=4,setup=16,locate=8,margin_percent=20):
    nodes,edges=check_serial.normalize(records)
    adj={u:set() for u in nodes}
    for u,v in edges: adj[u].add(v);adj[v].add(u)
    forward={u:sorted(v for v in adj[u] if (len(adj[u]),u)<(len(adj[v]),v)) for u in sorted(nodes)}
    tasks=[(min(len(forward[u]),len(forward[v])),max(len(forward[u]),len(forward[v])))
           for u in forward for v in forward[u]]
    stats=dict(batches=0,fast_batches=0,scalar_batches=0,pool_batches=0,nonempty_edges=0,
               pooled_queries=0,all_queries=0,all_scores=0,sum_batch_max_scores=0,
               prefix_queries=0,remaining_queries=0)
    for start in range(0,len(tasks),32):
        batch=tasks[start:start+32];batch += [(0,0)]*(32-len(batch))
        sizes=[a for a,b in batch]
        scores=[a*b.bit_length() for a,b in batch]
        prefixes=[min(a,budget) if mode=='budget' else 0 for a in sizes]
        remaining=sum(sizes)-sum(prefixes)
        fast=(mode=='budget' and remaining==0) or (mode=='score' and max(sizes)<=small)
        depth=max((b.bit_length() if a else 0 for a,b in batch))
        cost=setup+((sum(sizes)+31)//32)*(locate+depth)
        pool=mode=='pool' or (mode=='budget' and remaining!=0) or (mode=='score' and not fast and sum(sizes)!=0 and 100*max(scores)>(100+margin_percent)*cost)
        stats['batches']+=1;stats['fast_batches']+=fast;stats['scalar_batches']+=not pool and not fast
        stats['pool_batches']+=pool;stats['nonempty_edges']+=sum(a!=0 for a in sizes)
        stats['pooled_queries']+=remaining if pool else 0
        stats['all_queries']+=sum(sizes);stats['all_scores']+=sum(scores)
        stats['sum_batch_max_scores']+=max(scores)
        stats['prefix_queries']+=sum(prefixes);stats['remaining_queries']+=remaining
    return stats


def run(exe,path,mode,records,blocks=1,budget=8,small=4,setup=16,locate=8,margin=20):
    command=[exe,str(path),'--query-mode',mode,'--budget',str(budget),'--small',str(small),
             '--setup-cost',str(setup),'--locate-cost',str(locate),'--margin-percent',str(margin),
             '--blocks',str(blocks),'--verify','--warmup','1','--repeat','3','--query-diagnostics','--profile-gpu']
    row=json.loads(subprocess.run(command,check=True,text=True,capture_output=True,timeout=120).stdout)
    assert row['backend']=='gpu_query_v2_binary' and row['query_mode']==mode and row['kernel_mode_specialized']
    assert row['verified'] and row['triangles']==check_serial.count_bruteforce(records)
    assert row['query_budget']==budget and len(row['count_reduce_ms_samples'])==3
    assert row['query_policy']==dict(small=small,setup=setup,locate=locate,margin_percent=margin)
    stats=row['query_diagnostics'];expected=expected_stats(records,mode,budget,small,setup,locate,margin)
    assert all(stats[k]==v for k,v in expected.items()),(mode,stats,expected)
    assert stats['prefix_queries']+stats['remaining_queries']==stats['all_queries']
    assert stats['fast_batches']+stats['scalar_batches']+stats['pool_batches']==stats['batches']
    assert stats['extra_device_buffer_bytes']==88
    if mode=='budget': assert stats['pooled_queries']==stats['remaining_queries']
    for key in ['first_total_ms','first_count_reduce_ms','first_query_count_ms']:
        assert math.isfinite(row[key]) and row[key]>=0
    assert abs(sum(row['first_stages_ms'].values())-row['first_count_reduce_ms'])<=max(.002,row['first_count_reduce_ms']*.01)
    assert row['query_count_ms_samples']==row['warp_count_ms_samples']
    return row


def main():
    if len(sys.argv)!=2: raise SystemExit('Usage: python3 tests/check_query_v2.py BUILD/tc_gpu_query_v2')
    exe=str(Path(sys.argv[1]).resolve())
    for mode in ['budget','score']:
        sys.argv=[sys.argv[0],exe,'--query-mode',mode,'--verify','--blocks','1','--warmup','0','--repeat','1']
        check_serial.main()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'graph.txt'
        fixtures=[[(2*i,2*i+1) for i in range(m)] for m in [0,1,31,32,33,255,256,257]]
        fixtures += [list(itertools.combinations(range(n),2)) for n in [18,40,67]]
        fixtures += [list(itertools.combinations(range(40),2))+[(1000,i) for i in range(1001,1042)]+[(0,1),(1,0),(2000,2000)]]
        for index,records in enumerate(fixtures):
            path.write_text(''.join(f'{u} {v}\n' for u,v in records))
            for mode in ['scalar','pool','budget','score']:
                for blocks in [1,3]: run(exe,path,mode,records,blocks,budget=[4,8,16][index%3])
        for budget in [0,1,10,65535]: run(exe,path,'budget',records,budget=budget)
        for policy in [(0,0,0,0),(65535,65535,65535,1000)]:
            row=run(exe,path,'score',records,small=policy[0],setup=policy[1],locate=policy[2],margin=policy[3])
            if policy==(0,0,0,0): assert row['query_diagnostics']['pool_batches']>0
            else: assert row['query_diagnostics']['pool_batches']==0
        for mode in ['budget','score']:
            subprocess.run([exe,str(path),'--query-mode',mode,'--verify','--warmup','0','--repeat','1',
                '--parser','omp','--mapping','omp','--sort','omp','--threads','3','--check-csr'],check=True,capture_output=True,timeout=120)
        for flags in [['--query-mode','adaptive'],['--query-mode'],['--budget','-1'],['--budget','65536'],
                      ['--setup-cost','x'],['--locate-cost'],['--margin-percent','1001'],['--pool-overhead','64']]:
            assert subprocess.run([exe,str(path),*flags],capture_output=True,timeout=30).returncode!=0,flags
    print('PASS: v2 budget/score, exact counts, diagnostics, tails/grid stride, parameters, CSR and repeat consistency')


if __name__=='__main__': main()
