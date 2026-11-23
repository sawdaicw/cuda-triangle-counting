"""Query v2 budget-first and cost-score experiments; equal grids, fresh processes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default='build-lab')
    parser.add_argument('--inputs', nargs='+', default=['data/com-amazon.ungraph.txt', 'data/roadNet-CA.txt', 'data/web-Google.txt'])
    parser.add_argument('--small', type=int, default=4)
    parser.add_argument('--budgets', nargs='+', type=int, default=[4,8,16])
    parser.add_argument('--setup-cost', type=int, default=16)
    parser.add_argument('--locate-cost', type=int, default=8)
    parser.add_argument('--margin-percent', type=int, default=20)
    parser.add_argument('--include-controls', action='store_true', help='also measure specialized scalar and pool controls')
    parser.add_argument('--threads', type=int, default=12, help='same OpenMP preprocessing for ALL variants')
    parser.add_argument('--runs', type=int, default=3, help='fresh full-pipeline processes per graph/variant')
    parser.add_argument('--repeat', type=int, default=1, help='additional GPU-only samples per process; not full-pipeline samples')
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--blocks', type=int, required=True, help='mandatory equal grid size for ALL variants')
    parser.add_argument('--timeout', type=int, default=300, help='seconds per process')
    parser.add_argument('--out', required=True, help='new directory; never overwrite a previous experiment')
    args = parser.parse_args()
    if args.runs<1 or not 1<=args.repeat<=10000 or not 0<=args.warmup<=10000 or not 1<=args.threads<=256 or args.timeout<1:
        parser.error('invalid runs/repeat/warmup/threads/timeout')
    if any(not 0<=value<=65535 for value in [args.small,args.setup_cost,args.locate_cost]) or not 0<=args.margin_percent<=1000:
        parser.error('small/setup/locate in [0,65535], margin-percent in [0,1000]')
    if len(set(args.budgets))!=len(args.budgets) or any(not 0<=value<=65535 for value in args.budgets):
        parser.error('budgets must be distinct integers in [0,65535]')
    if args.blocks is not None and not 1<=args.blocks<=2147483647:
        parser.error('blocks must be a positive signed-int value')
    build = Path(args.build).resolve()
    policy = ['--small',str(args.small),'--setup-cost',str(args.setup_cost),
              '--locate-cost',str(args.locate_cost),'--margin-percent',str(args.margin_percent)]
    variants = {'edge':('tc_gpu_edge_baseline',[])}
    variants.update({f'budget{k}':('tc_gpu_query_v2',['--query-mode','budget','--budget',str(k),*policy])
                     for k in args.budgets})
    variants['score']=('tc_gpu_query_v2',['--query-mode','score',*policy])
    if args.include_controls:
        variants.update({mode:('tc_gpu_query_v2',['--query-mode',mode,*policy]) for mode in ['scalar','pool']})
    inputs = [Path(p).resolve() for p in args.inputs]
    if any(not p.is_file() for p in inputs) or len({p.name for p in inputs}) != len(inputs):
        parser.error('inputs must exist and have distinct basenames')
    executables = {name:build/name for name,_ in variants.values()}
    if any(not p.is_file() for p in executables.values()):
        parser.error('build tc_gpu_edge_baseline and tc_gpu_query_v2 first')
    output = Path(args.out); output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['OMP_DYNAMIC']='FALSE';env['OMP_WAIT_POLICY']='PASSIVE'
    common = ['--parser','omp','--mapping','omp','--sort','omp','--threads',str(args.threads),
              '--verify','--warmup',str(args.warmup),'--repeat',str(args.repeat)]
    if args.blocks is not None: common += ['--blocks',str(args.blocks)]
    repo=Path(__file__).resolve().parents[1]
    def git(*command):
        return subprocess.run(['git',*command],cwd=repo,text=True,capture_output=True).stdout.strip()
    source_file=repo/'SOURCE_COMMIT.txt'
    manifest={'purpose':'query-v2 budget/cost-score exploration; uncalibrated policy, no per-input winner selection',
              'commit':git('rev-parse','HEAD'), 'worktree_status':git('status','--porcelain'),
              'archive_source_commit':source_file.read_text().strip() if source_file.is_file() else None,
              'input_sha256':{str(p):sha256(p) for p in inputs},
              'executable_sha256':{str(p):sha256(p) for p in executables.values()},
              'runs_per_variant':args.runs,'common_options':common,'variants':variants,
              'primary_metric':'first_total_ms','secondary_metric':'first_count_reduce_ms',
              'additional_gpu_samples_per_process':args.repeat,'order_seed':42,
              'gpu_stage_profiling':False,'query_diagnostics':False,'selection':'fixed policy; no per-input timing-based tuning','file_cache':'not flushed; hashes read inputs before measurement',
              'environment':{k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_DYNAMIC','OMP_PROC_BIND','OMP_PLACES','OMP_THREAD_LIMIT','OMP_WAIT_POLICY']}}
    (output/'metadata.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    rng=random.Random(42); reference={};records={p.name:{v:[] for v in variants} for p in inputs};sequence=[]
    devices=set()
    for index in range(1,args.runs+1):
        work=[(p,v) for p in inputs for v in variants];rng.shuffle(work)
        for path,variant in work:
            name,flags=variants[variant]
            command=[str(executables[name]),str(path),*common,*flags]
            stem=f'{path.name}.{variant}.{index:02d}'
            sequence.append({'sample':stem,'command':command})
            (output/'commands.json').write_text(json.dumps(sequence,indent=2,ensure_ascii=False))
            print(f'{index}/{args.runs} {path.name} {variant}',flush=True)
            with (output/f'{stem}.json').open('w') as stdout,(output/f'{stem}.stderr').open('w') as stderr:
                result=subprocess.run(command,stdout=stdout,stderr=stderr,env=env,timeout=args.timeout)
            if result.returncode: raise RuntimeError(f'{stem}: exit {result.returncode}; see stderr')
            row=json.loads((output/f'{stem}.json').read_text())
            identity=tuple(row[k] for k in ['vertices','undirected_edges','triangles'])
            if path.name in reference and reference[path.name]!=identity: raise RuntimeError(f'{stem}: graph/count mismatch')
            reference[path.name]=identity
            if row['blocks'] != args.blocks or row['threads_per_block'] != 256:
                raise RuntimeError(f'{stem}: unequal launch geometry')
            if variant != 'edge':
                mode='budget' if variant.startswith('budget') else variant
                budget=int(variant[6:]) if mode=='budget' else 8
                if (row.get('backend')!='gpu_query_v2_binary' or row.get('query_mode')!=mode or
                        row.get('query_budget')!=budget or row.get('kernel_mode_specialized') is not True or
                        row.get('query_diagnostics_enabled') is not False or
                        row.get('query_policy')!=dict(small=args.small,setup=args.setup_cost,
                                                     locate=args.locate_cost,margin_percent=args.margin_percent)):
                    raise RuntimeError(f'{stem}: query v2 policy/mode mismatch')
            if not row['verified']: raise RuntimeError(f'{stem}: missing CPU verification')
            devices.add((row['device'],row['compute_capability'],row['cuda_runtime_version'],row['cuda_driver_version']))
            if len(devices)>1: raise RuntimeError('device/runtime identity changed during experiment')
            for key in ['first_total_ms','first_count_reduce_ms']:
                if not math.isfinite(row[key]) or row[key]<=0: raise RuntimeError(f'{stem}: invalid {key}')
            records[path.name][variant].append(row)
    summary={}
    for name,variants_rows in records.items():
        summary[name]={}
        for variant,rows in variants_rows.items():
            entry={'n':len(rows)}
            for metric in ['first_total_ms','first_count_reduce_ms']:
                times=[row[metric] for row in rows];base=[row[metric] for row in variants_rows['edge']]
                entry[metric]={'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),
                               'speedup_vs_edge':statistics.median(base)/statistics.median(times),
                               'change_vs_edge_percent':100*(statistics.median(times)/statistics.median(base)-1)}
            summary[name][variant]=entry
            print(name,variant,json.dumps(entry),flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
