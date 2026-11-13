"""CPU correctness/model exploration only; not a GPU timing benchmark."""
from bisect import bisect_right
from itertools import combinations, accumulate
import json
from pathlib import Path
import random


def forward(n, records):
    edges = {tuple(sorted((u, v))) for u, v in records if u != v}
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    rows = [[] for _ in range(n)]
    for u, v in edges:
        if (degree[u], u) > (degree[v], v):
            u, v = v, u
        rows[u].append(v)
    return [sorted(row) for row in rows], edges


def query_count(rows, batch_size=32):
    tasks = []
    for u, row in enumerate(rows):
        for v in row:
            a, b = rows[u], rows[v]
            if len(a) > len(b):
                a, b = b, a
            tasks.append((a, b))
    result = 0
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        prefix = [0, *accumulate(len(a) for a, _ in batch)]
        seen = set()
        # Models lane r processing virtual queries r, r+32, ... .
        for lane in range(32):
            for q in range(lane, prefix[-1], 32):
                owner = bisect_right(prefix, q) - 1
                local = q - prefix[owner]
                assert (owner, local) not in seen
                seen.add((owner, local))
                a, b = batch[owner]
                key = a[local]
                lo, hi = 0, len(b)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if b[mid] == key:
                        result += 1
                        break
                    if b[mid] < key:
                        lo = mid + 1
                    else:
                        hi = mid
        assert len(seen) == prefix[-1]
    return result


def verify():
    rng = random.Random(20261113)
    cases = [(0, []), (3, [(0, 1), (1, 2), (2, 0)]),
             (5, list(combinations(range(5), 2))),
             (9, [(u, v) for u in range(4) for v in range(4, 9)])]
    for _ in range(200):
        n = rng.randrange(21)
        probability = rng.choice([0, .05, .2, .5, .95, 1])
        records = [(u, v) for u, v in combinations(range(n), 2)
                   if rng.random() < probability]
        cases.append((n, records + [(v, u) for u, v in records[:3]]
                      + [(u, u) for u in range(n)]))
    checks = 0
    for n, records in cases:
        rows, edges = forward(n, records)
        oracle = sum((a, b) in edges and (a, c) in edges and (b, c) in edges
                     for a, b, c in combinations(range(n), 3))
        for batch_size in [1, 7, 32, 64]:
            assert query_count(rows, batch_size) == oracle
            checks += 1
    return dict(graphs=len(cases), batch_configurations=checks,
                status='PASS; CPU task-mapping model only, no CUDA validation')


def synthetic_model():
    import numpy as np

    def clique_lengths(k):
        # For a disjoint K_k, row u has degree k-1-u, and u<v.
        return np.concatenate([np.arange(k-2-u, -1, -1, dtype=np.int64)
                               for u in range(k-1)])

    def measures(lengths):
        pad = (-len(lengths)) % 32
        matrix = np.pad(lengths, (0, pad)).reshape(-1, 32)
        sums = matrix.sum(axis=1)
        maxima = matrix.max(axis=1)
        independent_rounds = int(maxima.sum())
        local_rounds = int(((sums+31)//32).sum())
        global_rounds = (int(sums.sum())+31)//32
        return dict(edges=len(lengths), queries=int(sums.sum()),
                    independent_query_rounds=independent_rounds,
                    local_flat_query_rounds=local_rounds,
                    global_flat_query_rounds=global_rounds,
                    nonempty_batches=int((sums > 0).sum()),
                    batches_with_short_length_at_least_16=int((maxima >= 16).sum()),
                    ideal_local_round_ratio=independent_rounds/local_rounds,
                    warning='Equal-cost queries, free task mapping, no memory/cache/synchronization or GPU grid model. Ratio is NOT a GPU speedup prediction.')

    result = {}
    for name, small, large in [('S6', 70235, 4), ('S7', 37595, 32)]:
        lengths = np.concatenate([np.tile(clique_lengths(8), small),
                                  np.tile(clique_lengths(256), large)])
        original = measures(lengths)
        shuffled = lengths.copy()
        np.random.default_rng(20261113).shuffle(shuffled)
        result[name] = dict(original_order=original,
                            shuffled_order=measures(shuffled),
                            construction='Exact disjoint mixed-clique lengths, source-ID then destination-ID order; no real-graph input or GPU run.')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['verify', 'model', 'all'], default='verify')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results/query_exploration_v1/model_results.json')
    args = parser.parse_args()
    result = {}
    if args.mode in ['verify', 'all']:
        result['correctness'] = verify()
    if args.mode in ['model', 'all']:
        result['idealized_models'] = synthetic_model()
    # A correctness-only run must not overwrite an archived full model result.
    if args.mode != 'verify':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
