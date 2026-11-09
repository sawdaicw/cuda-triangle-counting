"""Interleave preprocessing variants; each sample is a fresh process/full run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

VARIANTS = {
    'legacy': ['--parser', 'legacy', '--mapping', 'fused', '--sort', 'serial'],
    'fast': ['--parser', 'fast', '--mapping', 'fused', '--sort', 'serial'],
    'map1': ['--parser', 'legacy', '--mapping', 'serial', '--sort', 'serial'],
}
for threads in (2, 4, 8):
    VARIANTS[f'parse{threads}'] = ['--parser', 'omp', '--mapping', 'fused', '--sort', 'serial', '--threads', str(threads)]
    VARIANTS[f'map{threads}'] = ['--parser', 'legacy', '--mapping', 'omp', '--sort', 'serial', '--threads', str(threads)]
    VARIANTS[f'sort{threads}'] = ['--parser', 'legacy', '--mapping', 'fused', '--sort', 'omp', '--threads', str(threads)]
    VARIANTS[f'all{threads}'] = ['--parser', 'omp', '--mapping', 'omp', '--sort', 'omp', '--threads', str(threads)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--gpu', action='store_true', help='measure first_total_ms; otherwise preprocess_ms')
    parser.add_argument('--inputs', nargs='+', default=['data/com-amazon.ungraph.txt', 'data/roadNet-CA.txt', 'data/web-Google.txt'])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['legacy', 'fast', 'parse2', 'parse4', 'parse8', 'map1', 'map2', 'map4', 'map8', 'sort2', 'sort4', 'sort8', 'all2', 'all4', 'all8'])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--out', required=True, help='new output directory; existing directories are rejected')
    args = parser.parse_args()
    if args.runs < 1 or len(args.variants) != len(set(args.variants)):
        parser.error('runs must be positive and variants must be distinct')
    exe = Path(args.exe).resolve()
    inputs = [Path(p).resolve() for p in args.inputs]
    if not exe.is_file() or any(not p.is_file() for p in inputs): parser.error('executable/input missing')
    if len({p.name for p in inputs}) != len(inputs): parser.error('input basenames must be distinct')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env['OMP_DYNAMIC'] = 'FALSE'
    metric = 'first_total_ms' if args.gpu else 'preprocess_ms'
    hashes = {}
    for path in inputs:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    repo = Path(__file__).resolve().parents[1]
    def git(*command):
        return subprocess.run(['git', *command], cwd=repo, text=True, capture_output=True).stdout.strip()
    metadata = {'metric': metric, 'runs_per_variant': args.runs, 'gpu': args.gpu,
                'profile': args.profile, 'executable': str(exe), 'input_sha256': hashes,
                'commit': git('rev-parse', 'HEAD'), 'worktree_status': git('status', '--porcelain'),
                'variants': {v: VARIANTS[v] for v in args.variants},
                'environment': {k: env.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES', 'OMP_THREAD_LIMIT', 'OMP_WAIT_POLICY']},
                'order_seed': 42, 'file_cache': 'not flushed; SHA256 reads inputs before measurement'}
    (root/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    rng = random.Random(42)
    records = {path.name: {v: [] for v in args.variants} for path in inputs}
    reference = {}
    sequence = []
    for run in range(1, args.runs+1):
        work = [(path, variant) for path in inputs for variant in args.variants]
        rng.shuffle(work)
        for path, variant in work:
            command = [str(exe), str(path), *VARIANTS[variant]]
            if args.gpu: command += ['--verify', '--warmup', '0', '--repeat', '1']
            if args.profile: command += ['--profile-preprocess']
            stem = f'{path.name}.{variant}.{run:02d}'
            print(f'{run}/{args.runs} {path.name} {variant}', flush=True)
            sequence.append({'sample':stem,'command':command})
            (root/'commands.json').write_text(json.dumps(sequence, ensure_ascii=False, indent=2))
            with (root/f'{stem}.json').open('w') as output, (root/f'{stem}.stderr').open('w') as error:
                result = subprocess.run(command, stdout=output, stderr=error, env=env, timeout=300)
            if result.returncode: raise RuntimeError(f'{stem}: exit {result.returncode}; see stderr')
            row = json.loads((root/f'{stem}.json').read_text())
            identity = tuple(row[k] for k in ['vertices','undirected_edges','triangles'])
            if path.name in reference and reference[path.name] != identity:
                raise RuntimeError(f'Count/graph mismatch for {stem}')
            reference[path.name] = identity
            if args.gpu and not row['verified']: raise RuntimeError(f'CPU verification missing: {stem}')
            records[path.name][variant].append(row[metric])
    summary = {}
    for name, variants in records.items():
        summary[name] = {}
        for variant, times in variants.items():
            summary[name][variant] = {'n':len(times),'median_ms':statistics.median(times),
                                     'min_ms':min(times),'max_ms':max(times)}
            if 'legacy' in variants:
                summary[name][variant]['speedup_vs_legacy'] = statistics.median(variants['legacy'])/statistics.median(times)
            print(name, variant, json.dumps(summary[name][variant]), flush=True)
    (root/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
