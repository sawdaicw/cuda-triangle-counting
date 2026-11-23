"""Run on CUDA hardware: oracle, tail/grid-stride, policy and diagnostic checks."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import check_serial


def expected_stats(records, mode, small=4, overhead=64, per_round=2, margin=32):
    nodes, edges = check_serial.normalize(records)
    adj = {u: set() for u in nodes}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    forward = {u: sorted(v for v in adj[u] if (len(adj[u]), u) < (len(adj[v]), v)) for u in sorted(nodes)}
    tasks = [(min(len(forward[u]), len(forward[v])), max(len(forward[u]), len(forward[v])))
             for u in forward for v in forward[u]]
    stats = dict(batches=0, fast_batches=0, scalar_batches=0, pool_batches=0,
                 nonempty_edges=sum(a != 0 for a, b in tasks), pooled_queries=0,
                 all_queries=0, all_scores=0, sum_batch_max_scores=0)
    for start in range(0, len(tasks), 32):
        batch = tasks[start:start+32] + [(0, 0)] * (32-len(tasks[start:start+32]))
        fast = mode == 'adaptive' and max(a for a, b in batch) <= small
        scores = [a*b.bit_length() for a, b in batch]
        queries = sum(a for a, b in batch)
        pool = mode == 'pool' or (mode == 'adaptive' and not fast and
                max(scores)*32 > sum(scores) + 32*(overhead+margin+per_round*((queries+31)//32)))
        stats['all_queries'] += queries
        stats['all_scores'] += sum(scores)
        stats['sum_batch_max_scores'] += max(scores)
        stats['batches'] += 1
        stats['fast_batches'] += fast
        stats['scalar_batches'] += not pool and not fast
        stats['pool_batches'] += pool
        stats['pooled_queries'] += queries if pool else 0
    return stats


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 tests/check_query.py BUILD/tc_gpu_query')
    exe = str(Path(sys.argv[1]).resolve())
    for mode in ['scalar', 'pool', 'adaptive']:
        sys.argv = [sys.argv[0], exe, '--verify', '--warmup', '0', '--repeat', '1',
                    '--blocks', '1', '--query-mode', mode]
        check_serial.main()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'graph.txt'
        fixtures = [[(2*i, 2*i+1) for i in range(m)] for m in [0,1,31,32,33,255,256,257]]
        fixtures += [list(itertools.combinations(range(n),2)) for n in [18,40,67]]
        fixtures += [list(itertools.combinations(range(40),2)) + [(1000,i) for i in range(1001,1042)] + [(0,1),(1,0),(2000,2000)]]
        for records in fixtures:
            path.write_text(''.join(f'{u} {v}\n' for u,v in records))
            oracle = check_serial.count_bruteforce(records)
            for mode in ['scalar','pool','adaptive']:
                for blocks in [1,3]:
                    command = [exe,str(path),'--verify','--warmup','1','--repeat','3','--blocks',str(blocks),
                               '--query-mode',mode,'--query-diagnostics','--profile-gpu']
                    row = json.loads(subprocess.run(command,check=True,text=True,capture_output=True,timeout=120).stdout)
                    assert row['triangles'] == oracle and row['verified'], row
                    assert row['backend'] == 'gpu_query_binary' and row['query_mode'] == mode
                    assert len(row['count_reduce_ms_samples']) == 3
                    assert all(math.isfinite(row[k]) and row[k]>=0 for k in ['first_total_ms','first_count_reduce_ms'])
                    stats = row['query_diagnostics']
                    assert all(stats[k] == v for k,v in expected_stats(records,mode).items()), (stats,expected_stats(records,mode))
                    stages = row['first_stages_ms']
                    assert abs(sum(stages.values())-row['first_count_reduce_ms']) <= max(.002,row['first_count_reduce_ms']*.01)
        # Force the adaptive pool branch and the scalar branch on the same CSR.
        for small,overhead,per_round,margin in [(0,0,0,0),(65535,0,0,0),(0,65535,65535,65535)]:
            flags = ['--small',str(small),'--pool-overhead',str(overhead),'--pool-per-round',str(per_round),'--pool-margin',str(margin)]
            row = json.loads(subprocess.run([exe,str(path),'--verify','--warmup','0','--repeat','1','--query-diagnostics',*flags],check=True,text=True,capture_output=True,timeout=120).stdout)
            assert all(row['query_diagnostics'][k] == v for k,v in expected_stats(records,'adaptive',small,overhead,per_round,margin).items())
        subprocess.run([exe,str(path),'--verify','--warmup','0','--repeat','1','--parser','omp','--mapping','omp','--sort','omp','--threads','3','--check-csr'],check=True,capture_output=True,timeout=120)
        for flags in [['--query-mode','x'],['--query-mode'],['--small','-1'],['--small','65536'],['--pool-overhead','x'],['--pool-margin'],['--threshold','32']]:
            assert subprocess.run([exe,str(path),*flags],capture_output=True,timeout=30).returncode != 0, flags
    print('PASS: query modes, oracle, tails, grid stride, diagnostics, policy overrides, CSR and invalid options')


if __name__ == '__main__':
    main()
