"""Separate, explicitly profiled passes: never combine these timings with the main samples."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import os
import statistics
import hashlib


def collect(out):
    out=Path(out).resolve();meta=json.loads((out/'metadata.json').read_text())
    summary=json.loads((out/'summary.json').read_text());args=meta['parameters']
    exe=next(Path(p) for p in meta['executable_sha256'] if Path(p).name=='tc_gpu_query_v2')
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==meta['executable_sha256'][str(exe)], 'Executable changed since main benchmark'
    diagnostic=out/'diagnostics';diagnostic.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE')
    rows=[]
    for input_name in meta['input_sha256']:
        path=Path(input_name)
        cmd=[str(exe),str(path),'--query-mode','hash-queue','--small','8','--bypass-edges','262144',
             '--parser','omp','--mapping','omp','--sort','omp','--threads',str(args['threads']),
             '--blocks',str(args['blocks']),'--verify','--profile-gpu','--query-diagnostics',
             '--warmup','2','--repeat','5']
        print('DIAGNOSTIC: '+path.stem,flush=True)
        (diagnostic/(path.stem+'.command.json')).write_text(json.dumps(cmd))
        with (diagnostic/(path.stem+'.json')).open('w') as stdout,(diagnostic/(path.stem+'.stderr')).open('w') as stderr:
            result=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=env,timeout=args['timeout'])
        if result.returncode:raise RuntimeError('Diagnostic failed: '+path.stem)
        r=json.loads((diagnostic/(path.stem+'.json')).read_text())
        assert r['verified'] and r['triangles']==summary[path.stem]['hash_queue']['triangles']
        stats=r.get('query_diagnostics',{})
        rows.append(dict(graph=path.stem,bypass=r['bypass_used'],queued_batches=r['hash_queue_batches'],
            all_batches=stats.get('batches',0),short_batches=stats.get('fast_batches',0),
            common_target_batches=stats.get('common_target_batches',0),hash_queries=stats.get('hash_queries',0),
            all_queries=stats.get('all_queries',0),hash_inserted_keys=stats.get('hash_inserted_keys',0),
            queue_bytes=r['hash_queue_buffer_bytes'],primary_shared_bytes=r['queue_primary_static_shared_bytes'],
            consumer_shared_bytes=r['queue_consumer_static_shared_bytes'],primary_registers=r['queue_primary_registers_per_thread'],
            consumer_registers=r['queue_consumer_registers_per_thread'],
            profiled_route_median_ms=statistics.median(r['reuse_route_ms_samples']),
            profiled_hash_median_ms=statistics.median(r['queued_hash_ms_samples']),
            profiled_reduce_median_ms=statistics.median(r['global_reduce_ms_samples'])))
    with (diagnostic/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('DIAGNOSTICS DONE: '+str(diagnostic),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',required=True)
    collect(parser.parse_args().results)
