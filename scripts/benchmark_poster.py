"""Frozen six-schedule evaluation: independent processes, randomized rounds."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['com-amazon.ungraph','roadNet-CA','web-Google','amazon0302',
         'Wiki-Vote','soc-Epinions1','cit-Patents']
VARIANTS = {
    'edge': ('tc_gpu_edge_baseline', []),
    'pool': ('tc_gpu_query_v2', ['--query-mode','pool']),
    'reuse': ('tc_gpu_query_v2', ['--query-mode','reuse']),
    'gate_reuse': ('tc_gpu_query_v2', ['--query-mode','reuse-gated','--small','8','--bypass-edges','262144']),
    'gate_hash': ('tc_gpu_query_v2', ['--query-mode','hash-gated','--small','8','--bypass-edges','262144']),
    # Existing queue uses shorter >= threshold: 33 implements strictly >32.
    'split_gt32': ('tc_gpu_split', ['--threshold','33']),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def capture(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'error':str(e)}


def aggregate(values):
    q=statistics.quantiles(values,n=4,method='inclusive') if len(values)>1 else [values[0]]*3
    return dict(median_ms=statistics.median(values),mean_ms=statistics.mean(values),
        q25_ms=q[0],q75_ms=q[2],min_ms=min(values),max_ms=max(values),
        stddev_ms=statistics.stdev(values) if len(values)>1 else 0.,process_values_ms=values)


def validate(row,variant,blocks,repeat,warmup):
    assert row['verified'] is True, 'CPU verification missing'
    assert row['blocks']==blocks and row['threads_per_block']==256, 'grid changed'
    assert row['repeat']==repeat and row['warmup']==warmup, 'sample protocol changed'
    pp=row['preprocess_options']
    assert all(pp[k]=='omp' for k in ['parser','mapping','sort']), 'preprocess mode changed'
    samples=row['count_reduce_ms_samples']
    assert len(samples)==repeat and all(math.isfinite(t) and t>0 for t in samples)
    assert math.isclose(statistics.median(samples),row['count_reduce_ms_median'],rel_tol=1e-7,abs_tol=1e-8)
    assert row['first_total_ms']>0 and row['first_count_reduce_ms']>0
    if variant=='edge': assert row['backend']=='gpu_edge_binary'
    elif variant=='split_gt32':
        assert row['backend']=='gpu_split_binary' and row['threshold']==33
    else:
        expected={'pool':'pool','reuse':'reuse','gate_reuse':'reuse-gated','gate_hash':'hash-gated','hash_queue':'hash-queue'}[variant]
        assert row['query_mode']==expected and row['backend']=='gpu_query_v2_binary'
        assert not row['query_diagnostics_enabled']
        if variant.startswith('gate_') or variant=='hash_queue':
            assert row['query_policy']['small']==8 and row['bypass_threshold_edges']==262144
            assert row['bypass_used']==(row['undirected_edges']<262144)
        if variant in ['gate_hash','hash_queue']:
            assert (row['hash_gate_min_length'],row['hash_gate_max_length'],row['hash_gate_queries_per_key'])==(64,512,8)
        if variant=='hash_queue':
            assert row['hash_queue_version']=='warp-batch-queue-v1'
            assert row['hash_queue_rebuilt_each_run'] and not row['hash_queue_host_readback_in_timing']
            assert row['hash_queue_batches']<=row['hash_queue_capacity_batches']
            assert row['hash_queue_used']==(not row['bypass_used'])
            if row['hash_queue_used']:
                assert row['queue_resource_attributes_measured']
                assert row['queue_primary_static_shared_bytes']<32768
                assert row['queue_consumer_static_shared_bytes']>=32768
            else: assert row['hash_queue_buffer_bytes']==0


def summarize(out,records,variants,runs):
    summary={};flat=[]
    for graph,by_variant in records.items():
        summary[graph]={}
        baseline=statistics.median(statistics.median(r['count_reduce_ms_samples']) for r in by_variant['edge'])
        base_full=statistics.median(r['first_total_ms'] for r in by_variant['edge'])
        for variant in variants:
            rows=by_variant[variant]
            assert len(rows)==runs
            medians=[statistics.median(r['count_reduce_ms_samples']) for r in rows]
            item={'n_processes':runs,'gpu_count_reduce':aggregate(medians),
                  'first_count_reduce':aggregate([r['first_count_reduce_ms'] for r in rows]),
                  'full_task':aggregate([r['first_total_ms'] for r in rows]),
                  'triangles':rows[0]['triangles'],'vertices':rows[0]['vertices'],
                  'undirected_edges':rows[0]['undirected_edges'],
                  'bypass_runs':sum(r.get('bypass_used',False) for r in rows)}
            t=item['gpu_count_reduce']['median_ms'];total=item['full_task']['median_ms']
            item.update(gpu_speedup=baseline/t,gpu_delta_ms=t-baseline,
                        full_task_speedup=base_full/total)
            if variant=='hash_queue':
                item['hash_queue_batches']=[r['hash_queue_batches'] for r in rows]
                item['hash_queue_buffer_bytes']=[r['hash_queue_buffer_bytes'] for r in rows]
            summary[graph][variant]=item
            flat.append(dict(graph=graph,variant=variant,n=runs,gpu_median_ms=t,
                gpu_q25_ms=item['gpu_count_reduce']['q25_ms'],gpu_q75_ms=item['gpu_count_reduce']['q75_ms'],
                gpu_speedup=baseline/t,gpu_delta_ms=t-baseline,
                first_gpu_median_ms=item['first_count_reduce']['median_ms'],
                full_task_median_ms=total,full_task_speedup=base_full/total,
                vertices=item['vertices'],edges=item['undirected_edges'],triangles=item['triangles']))
        print(graph+' | '+' | '.join(f'{v}: {summary[graph][v]["gpu_count_reduce"]["median_ms"]:.6f} ms ({summary[graph][v]["gpu_speedup"]:.3f}x)' for v in variants),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',default='build')
    ap.add_argument('--out',required=True)
    ap.add_argument('--runs',type=int,default=10)
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--repeat',type=int,default=30)
    ap.add_argument('--threads',type=int,default=12)
    ap.add_argument('--blocks',type=int,default=864)
    ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--seed',type=int,default=20261126)
    ap.add_argument('--without-split',action='store_true')
    ap.add_argument('--inputs',nargs='+',help='optional input subset; default is all 14 core graphs')
    args=ap.parse_args()
    if not(1<=args.runs<=1000 and 0<=args.warmup<=10000 and 1<=args.repeat<=10000
           and 1<=args.threads<=256 and 1<=args.blocks<=2147483647 and args.timeout>0):
        ap.error('invalid experiment options')
    variants=dict(VARIANTS)
    if args.without_split: variants.pop('split_gt32')
    inputs=[ROOT/'data'/f'{n}.txt' for n in NAMES]+[ROOT/'data/generated'/f'S{i}.txt' for i in range(1,8)]
    if args.inputs: inputs=[Path(p).resolve() for p in args.inputs]
    if len({p.stem for p in inputs})!=len(inputs): ap.error('input stems must be unique')
    missing=[str(p) for p in inputs if not p.is_file()]
    build=Path(args.build).resolve()
    binaries={exe:build/exe for exe,_ in variants.values()}
    missing += [str(p) for p in binaries.values() if not p.is_file()]
    if missing: ap.error('missing files: '+', '.join(missing))
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE')
    common=['--parser','omp','--mapping','omp','--sort','omp','--threads',str(args.threads),
            '--blocks',str(args.blocks),'--verify','--warmup',str(args.warmup),'--repeat',str(args.repeat)]
    sources=[p for base in ['src/common','src/cuda'] for p in (ROOT/base).glob('*') if p.is_file()]
    sources += [ROOT/'CMakeLists.txt',Path(__file__).resolve()]
    manifest={'schema':'poster-benchmark-v1','parameters':vars(args),'variants':variants,
        'launch_common':common,'gpu_metric':'median over independent processes of median count_reduce_ms_samples',
        'full_metric':'median first_total_ms across independent processes; includes preprocessing/setup/transfers',
        'includes':'count kernels, pool/hash preparation and CUB reduction; split includes queue reset/build and both count passes',
        'file_cache':'not flushed; input hashing reads files before measurement; filesystem cache typically warm',
        'ordering':'randomized graph order and randomized variant order within each graph, every round',
        'input_sha256':{str(p):digest(p) for p in inputs},
        'executable_sha256':{str(p):digest(p) for p in binaries.values()},
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'archive_source_commit':(ROOT/'SOURCE_COMMIT.txt').read_text().strip() if (ROOT/'SOURCE_COMMIT.txt').exists() else None,
        'local_patch':json.loads((ROOT/'LOCAL_POSTER_PATCH.json').read_text()) if (ROOT/'LOCAL_POSTER_PATCH.json').exists() else None,
        'hardware':capture(['nvidia-smi']), 'cpu':capture(['lscpu']),
        'platform':platform.platform(),'environment':{k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_DYNAMIC','OMP_WAIT_POLICY','OMP_PROC_BIND','OMP_PLACES','OMP_THREAD_LIMIT']}}
    (out/'metadata.json').write_text(json.dumps(manifest,indent=2))
    rng=random.Random(args.seed);records={p.stem:{v:[] for v in variants} for p in inputs}
    commands=[];identities={};device=None;completed=0;total=args.runs*len(inputs)*len(variants)
    for round_id in range(args.runs):
        graphs=list(inputs);rng.shuffle(graphs)
        for path in graphs:
            order=list(variants);rng.shuffle(order)
            for variant in order:
                exe,flags=variants[variant];stem=f'{path.stem}.{variant}.{round_id+1:02d}'
                cmd=[str(binaries[exe]),str(path),*common,*flags]
                entry={'sample':stem,'round':round_id+1,'command':cmd,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'gpu_snapshot':capture(['nvidia-smi','--query-gpu=index,name,pstate,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'])}
                commands.append(entry);(out/'commands.json').write_text(json.dumps(commands,indent=2))
                print(f'{completed+1}/{total}: round {round_id+1}/{args.runs} {path.stem} {variant}',flush=True)
                with (out/(stem+'.json')).open('w') as stdout,(out/(stem+'.stderr')).open('w') as stderr:
                    result=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=env,timeout=args.timeout)
                if result.returncode: raise RuntimeError(f'{stem}: exit {result.returncode}; see stderr')
                row=json.loads((out/(stem+'.json')).read_text());validate(row,variant,args.blocks,args.repeat,args.warmup)
                assert row['preprocess_options']['threads_requested']==args.threads
                identity=tuple(row[k] for k in ['vertices','undirected_edges','triangles'])
                if path.stem in identities: assert identities[path.stem]==identity, 'graph/count changed'
                identities[path.stem]=identity
                current=tuple(row[k] for k in ['device','compute_capability','cuda_runtime_version','cuda_driver_version'])
                if device is not None: assert device==current, 'GPU identity changed'
                device=current;records[path.stem][variant].append(row);completed+=1
                (out/'status.json').write_text(json.dumps({'completed':completed,'total':total,'last':stem}))
    summarize(out,records,list(variants),args.runs)
    (out/'hardware_end.json').write_text(json.dumps(capture(['nvidia-smi']),indent=2))
    print('ALL DONE: '+str(out),flush=True)
    return out


if __name__=='__main__':
    main()
