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
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'graph.txt'
        for i, edges in enumerate(cases):
            path.write_text('# test\n% comment\n\n' + ''.join(f'{u} {v}\n' for u, v in edges))
            result = subprocess.run([executable, str(path)], text=True, capture_output=True, check=True)
            output = json.loads(result.stdout)
            nodes, clean = normalize(edges)
            expected = count_bruteforce(edges)
            assert (output['vertices'], output['undirected_edges'], output['triangles']) == (
                len(nodes), len(clean), expected), (i, output, expected)
        for text in ['1 2 3\n', 'not an edge\n', '9223372036854775808 1\n']:
            path.write_text(text)
            result = subprocess.run([executable, str(path)], text=True, capture_output=True)
            assert result.returncode != 0 and 'Line 1:' in result.stderr, result
    print(f'PASS: {len(cases)} graphs match brute-force oracle; 3 malformed inputs rejected')


if __name__ == '__main__':
    main()
