"""Independent selection statistics and fake-device orchestration (not GPU data)."""
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('query_v2_checks',ROOT/'tests/check_query_v2.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)


class QueryV2Workflow(unittest.TestCase):
    def test_prefix_and_remaining_identities(self):
        graphs=[[],[(0,1),(1,2),(0,2)],list(itertools.combinations(range(40),2))]
        for graph in graphs:
            for k in [0,1,4,8,10,16,65535]:
                stats=checks.expected_stats(graph,'budget',k)
                self.assertEqual(stats['prefix_queries']+stats['remaining_queries'],stats['all_queries'])
                self.assertEqual(stats['remaining_queries'],stats['pooled_queries'])
                self.assertEqual(stats['batches'],stats['fast_batches']+stats['pool_batches'])
                if k==0: self.assertEqual(stats['prefix_queries'],0)
                if k==65535: self.assertEqual(stats['remaining_queries'],0)
            scalar=checks.expected_stats(graph,'scalar')
            pool=checks.expected_stats(graph,'pool')
            self.assertEqual(scalar['all_queries'],pool['pooled_queries'])
            if len(graph)>3:
                forced=checks.expected_stats(graph,'score',small=0,setup=0,locate=0,margin_percent=0)
                self.assertGreater(forced['pool_batches'],0)
            score=checks.expected_stats(graph,'score',small=65535)
            self.assertEqual(score['pool_batches'],0)
            self.assertEqual(score['fast_batches'],score['batches'])

    def test_v2_protocol_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build=root/'build';build.mkdir()
            path=root/'triangle.txt';path.write_text('0 1\n1 2\n0 2\n')
            source='''#!/usr/bin/env python3
import json,sys
args=sys.argv[2:]
def value(flag,default): return args[args.index(flag)+1] if flag in args else default
mode=value('--query-mode','edge')
times={'edge':10,'scalar':10.1,'pool':15,'budget':8,'score':9}
row=dict(vertices=3,undirected_edges=3,triangles=1,verified=True,device='FAKE TEST DEVICE',compute_capability='0.0',cuda_runtime_version=0,cuda_driver_version=0,blocks=int(value('--blocks',0)),threads_per_block=256,first_total_ms=times[mode]*10,first_count_reduce_ms=times[mode])
if mode!='edge': row.update(backend='gpu_query_v2_binary',query_mode=mode,query_budget=int(value('--budget',8)),kernel_mode_specialized=True,query_diagnostics_enabled=False,query_policy={key:int(value(flag,0)) for key,flag in [('small','--small'),('setup','--setup-cost'),('locate','--locate-cost'),('margin_percent','--margin-percent')]})
print(json.dumps(row))
'''
            for name in ['tc_gpu_edge_baseline','tc_gpu_query_v2']:
                exe=build/name;exe.write_text(source);exe.chmod(0o755)
            out=root/'output'
            command=[sys.executable,str(ROOT/'scripts/benchmark_query_v2.py'),'--build',str(build),'--inputs',str(path),
                     '--blocks','3','--threads','2','--runs','2','--budgets','0','4','8','16','--include-controls','--out',str(out)]
            result=subprocess.run(command,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((out/'summary.json').read_text())['triangle.txt']
            self.assertEqual(set(summary),{'edge','budget0','budget4','budget8','budget16','score','scalar','pool'})
            self.assertEqual(summary['budget8']['first_count_reduce_ms']['speedup_vs_edge'],1.25)
            self.assertEqual(len(json.loads((out/'commands.json').read_text())),16)
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            for extra in [['--budgets','4','4'],['--budgets','-1'],['--margin-percent','1001']]:
                self.assertNotEqual(subprocess.run(command+extra,capture_output=True).returncode,0)


if __name__=='__main__': unittest.main()
