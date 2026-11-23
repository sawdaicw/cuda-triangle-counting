"""Exercise benchmark orchestration with explicit fake devices, not GPU results."""
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('query_checks',ROOT/'tests/check_query.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class QueryBenchmarkChecks(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'g++ preprocessor unavailable')
    def test_query_launch_is_excluded_from_other_backends(self):
        # Preprocessing only: empty CUDA/CUB stubs do NOT compile or emulate CUDA.
        # Catch accidental name lookup in discarded non-template constexpr paths.
        with tempfile.TemporaryDirectory() as directory:
            include=Path(directory)
            for name in ['cuda_runtime.h','cub/block/block_reduce.cuh','cub/device/device_reduce.cuh']:
                stub=include/name;stub.parent.mkdir(parents=True,exist_ok=True);stub.write_text('')
            for mode in range(4):
                result=subprocess.run(['g++','-std=c++17','-E','-x','c++',
                    '-I',str(include),f'-DTC_EDGE_MODE={mode}',str(ROOT/'src/cuda/edge_baseline.cu')],
                    text=True,capture_output=True,check=True)
                if mode==3:
                    self.assertIn('count_query_batches<true>',result.stdout)
                    self.assertIn('count_query_batches<false>',result.stdout)
                else:
                    self.assertNotIn('count_query_batches',result.stdout)

    def test_diagnostic_oracle(self):
        triangle = [(0,1),(1,2),(0,2)]
        for mode in ['scalar','pool','adaptive']:
            stats=checks.expected_stats(triangle,mode)
            self.assertEqual(stats['batches'],1)
            self.assertEqual(stats['nonempty_edges'],1)
            self.assertEqual(stats['pool_batches'],int(mode=='pool'))
            self.assertEqual(stats['pooled_queries'],int(mode=='pool'))
        records=list(__import__('itertools').combinations(range(40),2))
        forced=checks.expected_stats(records,'adaptive',0,0,0,0)
        self.assertGreater(forced['pool_batches'],0)

    def test_fresh_process_protocol_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build=root/'build';build.mkdir()
            path=root/'triangle.txt';path.write_text('0 1\n1 2\n0 2\n')
            source='''#!/usr/bin/env python3
import json,sys
from pathlib import Path
args=sys.argv[2:]
def value(flag,default): return args[args.index(flag)+1] if flag in args else default
mode=value('--query-mode','edge')
times={'edge':10,'scalar':10.1,'pool':15,'adaptive':8}
row=dict(vertices=3,undirected_edges=3,triangles=1,verified=True,device='FAKE TEST DEVICE',compute_capability='0.0',cuda_runtime_version=0,cuda_driver_version=0,blocks=int(value('--blocks',0)),threads_per_block=256,first_total_ms=times[mode]*10,first_count_reduce_ms=times[mode])
if mode!='edge': row.update(query_mode=mode,query_diagnostics_enabled=False,query_policy={key:int(value(flag,0)) for key,flag in [('small','--small'),('overhead','--pool-overhead'),('per_round','--pool-per-round'),('margin','--pool-margin')]})
print(json.dumps(row))
'''
            for name in ['tc_gpu_edge_baseline','tc_gpu_query']:
                exe=build/name;exe.write_text(source);exe.chmod(0o755)
            out=root/'output'
            command=[sys.executable,str(ROOT/'scripts/benchmark_query.py'),'--build',str(build),'--inputs',str(path),'--blocks','3','--threads','2','--runs','2','--out',str(out)]
            result=subprocess.run(command,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((out/'summary.json').read_text())['triangle.txt']
            self.assertEqual(summary['adaptive']['n'],2)
            self.assertEqual(summary['adaptive']['first_count_reduce_ms']['speedup_vs_edge'],1.25)
            self.assertEqual(len(json.loads((out/'commands.json').read_text())),8)
            meta=json.loads((out/'metadata.json').read_text())
            self.assertFalse(meta['query_diagnostics'])
            self.assertEqual(meta['environment']['OMP_WAIT_POLICY'],'PASSIVE')
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__': unittest.main()
