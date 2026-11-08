"""Compare a compiled CPU or GPU executable against independent tiny-graph enumeration."""
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from reference import count_bruteforce, normalize


def course_cases():
    """Exact D1-D4 cases plus the suggested G(20, 0.2) graphs.

    Each case stores independently specified (n, m, triangles).
    Self-loop records declare isolated vertices; preprocessing removes those
    loops but retains the vertices. No extra edges survive normalization.
    """
    clique = lambda vertices: list(itertools.combinations(vertices, 2))
    cases = [
        ('D1-empty', [], (0, 0, 0)),
        ('D1-single-edge', [(0, 1)], (2, 1, 0)),
        ('D1-path3', [(0, 1), (1, 2)], (3, 2, 0)),
        ('D1-K3', clique(range(3)), (3, 3, 1)),
        ('D2-K5', clique(range(5)), (5, 10, 10)),
        ('D2-K4,5', [(u, v) for u in range(4) for v in range(4, 9)], (9, 20, 0)),
        ('D2-star8', [(0, v) for v in range(1, 9)], (9, 8, 0)),
        ('D3-K3+K4+2isolates',
         clique(range(3)) + clique(range(3, 7)) + [(7, 7), (8, 8)],
         (9, 9, 5)),
        ('D4-preprocessing',
         [(0, 1), (1, 0), (0, 1), (1, 2), (2, 0), (2, 2)],
         (3, 3, 1)),
    ]
    for seed in range(30):
        rng = random.Random(seed)
        edges = [edge for edge in itertools.combinations(range(20), 2)
                 if rng.random() < 0.2]
        # Declare all 20 vertices, including any isolated ones.
        records = edges + [(v, v) for v in range(20)]
        expected = count_bruteforce(edges, vertices=range(20))
        cases.append((f'course-random-n20-p0.2-seed{seed}', records,
                      (20, len(edges), expected)))
    return cases


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 tests/check_serial.py ./build/tc_gpu_serial')
    executable = str(Path(sys.argv[1]).resolve())
    cases = [[], [(9, 9)], [(10, 20), (20, 12), (12, 10)],
             list(itertools.combinations(range(8), 2)),
             [(0, i) for i in range(1, 20)],
             [(0, 1), (1, 2), (2, 3), (3, 0)],
             [(-9, 1000000000000), (1000000000000, 42), (42, -9),
              (42, -9), (-9, 42), (77, 77)]]
    for seed in range(30):
        rng = random.Random(seed)
        labels = rng.sample(range(-5000, 5000), 16)
        edges = [edge for edge in itertools.combinations(labels, 2) if rng.random() < .3]
        cases.append(edges + [(v, u) for u, v in edges[:5]] + [(labels[0], labels[0])])
    named_cases = course_cases() + [
        (f'extra-{i}', edges, None) for i, edges in enumerate(cases)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'graph.txt'
        for name, edges, specified in named_cases:
            path.write_text('# test\n% comment\n\n' + ''.join(f'{u} {v}\n' for u, v in edges))
            result = subprocess.run([executable, str(path)], text=True, capture_output=True, check=True)
            output = json.loads(result.stdout)
            nodes, clean = normalize(edges)
            expected = count_bruteforce(edges)
            oracle = (len(nodes), len(clean), expected)
            if specified is not None and oracle != specified:
                raise AssertionError((name, 'fixture disagrees with specified answer', oracle, specified))
            actual = (output['vertices'], output['undirected_edges'], output['triangles'])
            if actual != oracle:
                raise AssertionError((name, actual, oracle))
            print(f'PASS {name}: vertices={actual[0]} edges={actual[1]} triangles={actual[2]}')
        for text in ['1 2 3\n', 'not an edge\n', '9223372036854775808 1\n']:
            path.write_text(text)
            result = subprocess.run([executable, str(path)], text=True, capture_output=True)
            if result.returncode == 0 or 'Line 1:' not in result.stderr:
                raise AssertionError(result)
    print(f'PASS: {len(named_cases)} graphs match brute-force oracle; 3 malformed inputs rejected')


if __name__ == '__main__':
    main()
