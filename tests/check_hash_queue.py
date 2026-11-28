"""Run on a CUDA server: exact queue routing, counts, tail batches and bypass."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import check_serial


def expected_routing(records,small):
    nodes,edges=check_serial.normalize(records)
    adj={u:set() for u in nodes}
    for u,v in edges: adj[u].add(v);adj[v].add(u)
    forward={u:sorted(v for v in adj[u] if (len(adj[u]),u)<(len(adj[v]),v)) for u in sorted(nodes)}
    tasks=[]
    for u in forward:
        for v in forward[u]:
            short,long=(v,u) if len(forward[u])>len(forward[v]) else (u,v)
            tasks.append((len(forward[short]),len(forward[long]),long))
    result=dict(batches=0,fast_batches=0,pool_batches=0,common_target_batches=0,
                hash_batches=0,hash_queries=0,hash_inserted_keys=0)
    for i in range(0,len(tasks),32):
        batch=tasks[i:i+32];result['batches']+=1
        if max(a for a,b,long in batch)<=small:
            result['fast_batches']+=1;continue
        result['pool_batches']+=1
        live=[t for t in batch if t[0]]
        if not live or len({long for a,b,long in live})!=1:continue
        result['common_target_batches']+=1
        b=live[0][1];q=sum(a for a,b,long in batch)
        if 64<=b<=512 and q>=8*b:
            result['hash_batches']+=1;result['hash_queries']+=q;result['hash_inserted_keys']+=b
    return result


def run(exe,path,records,blocks=1,small=8,bypass=0):
    row=json.loads(subprocess.run([exe,str(path),'--query-mode','hash-queue','--small',str(small),
        '--bypass-edges',str(bypass),'--blocks',str(blocks),'--verify','--warmup','1','--repeat','3',
        '--query-diagnostics','--profile-gpu'],check=True,capture_output=True,text=True,timeout=300).stdout)
    assert row['verified'] and row['query_mode']=='hash-queue'
    samples=row['count_reduce_ms_samples']
    assert len(samples)==3 and all(math.isfinite(t) and t>0 for t in samples)
    if row['bypass_used']:
        assert not row['hash_queue_used'] and row['hash_queue_buffer_bytes']==0
        assert row['hash_queue_batches']==0 and row['hash_queue_capacity_batches']==0
        assert row['query_diagnostics_skipped_for_bypass']
        assert not row['queue_resource_attributes_measured']
        assert row['first_queued_hash_ms']==0 and all(t==0 for t in row['queued_hash_ms_samples'])
    else:
        expected=expected_routing(records,small)
        for k,value in expected.items(): assert row['query_diagnostics'][k]==value,(k,row['query_diagnostics'],expected)
        assert row['hash_queue_batches']==expected['hash_batches']
        cap=(row['undirected_edges']+31)//32
        assert row['hash_queue_capacity_batches']==cap
        assert row['hash_queue_buffer_bytes']==max(cap,1)*8+8
        assert row['hash_queue_batches']<=cap
        assert row['queue_resource_attributes_measured']
        assert row['queue_primary_static_shared_bytes']<32768
        assert row['queue_consumer_static_shared_bytes']>=32768
        assert row['hash_queue_rebuilt_each_run'] and not row['hash_queue_host_readback_in_timing']
    assert not row['diagnostics_inspect_candidate']
    stages=row['first_reuse_route_ms']+row['first_queued_hash_ms']+row['first_stages_ms']['global_reduce_ms']
    assert abs(stages-row['first_count_reduce_ms'])<=max(.005,row['first_count_reduce_ms']*.03)
    return row


def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 tests/check_hash_queue.py BUILD/tc_gpu_query_v2')
    exe=str(Path(sys.argv[1]).resolve())
    sys.argv=[sys.argv[0],exe,'--query-mode','hash-queue','--small','8','--verify','--blocks','1','--warmup','0','--repeat','1']
    check_serial.main()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'g.txt'
        fixtures=[[],[(i*2,i*2+1) for i in range(257)]]
        fixtures += [list(itertools.combinations(range(n),2)) for n in [18,80,128,513,520]]
        fixtures += [list(itertools.combinations(range(128),2))+list(itertools.combinations(range(200,280),2))+
                     [(0,200)]+[(1000,i) for i in range(1001,1042)]+[(0,1),(1,0),(2000,2000)]]
        for records in fixtures:
            path.write_text(''.join(f'{u} {v}\n' for u,v in records))
            for blocks in [1,3]:
                row=run(exe,path,records,blocks)
                if len(records)<10000:assert row['triangles']==check_serial.count_bruteforce(records)
            run(exe,path,records,bypass=262144)
        # Force all batches to scalar completion: no queued work and no stale
        # consumer sums across repeats. Opposite extreme exercises all candidates.
        run(exe,path,records,small=65535)
        run(exe,path,records,small=0)
        # Intentional hash collisions, preserve numeric-id gaps with isolated loops.
        vertices=[1024*i for i in range(80)]
        records=list(itertools.combinations(vertices,2))+[(v,v) for v in range(vertices[-1]+1)]
        path.write_text(''.join(f'{u} {v}\n' for u,v in records))
        row=run(exe,path,records)
        assert row['triangles']==math.comb(80,3) and row['hash_queue_batches']>0
    print('PASS: tiny oracle, exact queue routing, empty queue, grid stride/tails, capacity boundary, collisions, bypass and resource separation')


if __name__=='__main__':main()
