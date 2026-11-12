"""GPU execution checks for fixed-warp and light/heavy split backends."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import check_serial


def run(exe, path, *options):
    result = subprocess.run([exe, str(path), '--verify', '--warmup', '1',
                             '--repeat', '3', *options], text=True,
                            capture_output=True, check=True, timeout=120)
    row = json.loads(result.stdout)
    assert row['verified'] and len(row['count_reduce_ms_samples']) == 3
    assert row['threads_per_block'] == 256 and row['warps_per_block'] == 8
    for key in ['first_total_ms', 'first_count_reduce_ms', *['count_reduce_ms_min', 'count_reduce_ms_median', 'count_reduce_ms_mean']]:
        assert math.isfinite(row[key]) and row[key] >= 0, (key, row)
    if row['gpu_stage_profiled']:
        first = row['first_stages_ms']
        assert abs(sum(first.values()) - row['first_count_reduce_ms']) <= max(.002, row['first_count_reduce_ms'] * .01)
        arrays = [row[k] for k in ['light_count_queue_ms_samples', 'warp_count_ms_samples', 'global_reduce_ms_samples']]
        assert all(len(a) == 3 for a in arrays)
        for total, parts in zip(row['count_reduce_ms_samples'], zip(*arrays)):
            assert all(math.isfinite(v) and v >= 0 for v in parts)
            assert abs(total-sum(parts)) <= max(.002, total*.01)
    else:
        assert 'first_stages_ms' not in row
    return row


def expected_heavy(records, threshold):
    # Independent Python construction using original vertex IDs.
    _, edges = check_serial.normalize(records)
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v); adj.setdefault(v, set()).add(u)
    forward = {u: {v for v in vs if (len(vs), u) < (len(adj[v]), v)} for u, vs in adj.items()}
    return sum(min(len(vs), len(forward[v])) >= threshold for u, vs in forward.items() for v in vs)


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python3 tests/check_warp.py BUILD/tc_gpu_warp BUILD/tc_gpu_split')
    warp, split = [str(Path(p).resolve()) for p in sys.argv[1:]]
    for exe in (warp, split):
        sys.argv = [sys.argv[0], exe, '--verify', '--warmup', '0', '--repeat', '1', '--blocks', '1']
        check_serial.main()  # 76 oracle cases + 3 malformed inputs per backend
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'graph.txt'
        fixtures = [(f'matching-{m}', [(2*i,2*i+1) for i in range(m)], 0)
                    for m in [0,1,7,8,9,31,32,33,255,256,257]]
        fixtures += [(f'K{n}', list(itertools.combinations(range(n),2)), math.comb(n,3))
                     for n in [18,34,35,66,67,96]]
        # Mixture of heavy dense tasks, zero-work star edges, duplicates, loops.
        dense = list(itertools.combinations(range(67),2))
        mixed = dense + [(1000, v) for v in range(1001,1101)] + [(0,1),(1,0),(2000,2000)]
        fixtures.append(('mixed', mixed, math.comb(67,3)))
        for name, records, triangles in fixtures:
            path.write_text(''.join(f'{u} {v}\n' for u,v in records))
            nodes, edges = check_serial.normalize(records)
            for blocks in [1,3]:
                for profile in [False,True]:
                    flags = ['--blocks',str(blocks)] + (['--profile-gpu'] if profile else [])
                    row = run(warp,path,*flags)
                    assert row['backend'] == 'gpu_warp_binary'
                    assert (row['vertices'],row['undirected_edges'],row['triangles']) == (len(nodes),len(edges),triangles), (name,row)
                    for threshold in [1,16,32,64,1000000]:
                        row = run(split,path,*flags,'--threshold',str(threshold))
                        heavy = expected_heavy(records,threshold)
                        assert row['backend'] == 'gpu_split_binary'
                        assert (row['vertices'],row['undirected_edges'],row['triangles']) == (len(nodes),len(edges),triangles), (name,row)
                        assert row['heavy_edges'] == heavy, (name,threshold,row)
                        assert row['light_edges_including_empty'] + heavy == len(edges)
                        assert row['queue_capacity_edges'] == len(edges)
                        assert row['queue_buffer_bytes'] == max(1,len(edges))*8 + 8
                        assert row['device_buffer_bytes'] >= row['queue_buffer_bytes']
                        expected_fraction = heavy/len(edges) if edges else 0
                        assert abs(row['heavy_edge_fraction']-expected_fraction) < 1e-9
        # Exercise common optimized preprocessing on both backends.
        for exe in (warp,split):
            run(exe,path,'--parser','omp','--mapping','omp','--sort','omp','--threads','3','--check-csr')
        for exe in (warp,split):
            bad = [('--blocks','0'),('--repeat','0'),('--unknown',)]
            bad += [('--threshold',v) for v in ['0','-1','x','2147483648']] if exe == split else [('--threshold','32')]
            if exe == split: bad.append(('--threshold',))
            for options in bad:
                result = subprocess.run([exe,str(path),*options],text=True,capture_output=True,timeout=30)
                assert result.returncode != 0, options
    print('PASS: GPU warp/split boundaries, mixed/empty queues, thresholds, stage timings, repeated resets and CSR equivalence')


if __name__ == '__main__':
    main()
