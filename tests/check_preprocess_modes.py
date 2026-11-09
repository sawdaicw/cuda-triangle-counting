"""Boundary/error tests and exact CSR comparison across the full option matrix."""
import itertools
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python3 tests/check_preprocess_modes.py CPU_EXECUTABLE EQUIVALENCE_EXECUTABLE')
    cpu, equivalence = (str(Path(p).resolve()) for p in sys.argv[1:])
    valid = [b'', b'\n\r\n\t \n# comment\n% comment', b'1 2', b'1 2\r\n2 3\r\n3 1\r\n',
             b'+1 +2\n+2 -3\n-3 1\n', b'-9223372036854775808 9223372036854775807\n',
             b'1-2\n', b'1+2\n', b'1\v2\f\r\n', b'#' + b'x'*50003 + b'\n1 2\n',
             b' ' * 30001 + b'1 2\n2 3\n3 1', b'7 7\n'*4097,
             (b'1 2\n2 1\n1 2\n2 3\n3 1\n77 77\n')*1003]
    for seed in range(10):
        rng = random.Random(seed)
        labels = rng.sample(range(-10**9, 10**9), 40)
        edges = [e for e in itertools.combinations(labels, 2) if rng.random() < 0.2]
        edges += [(v, u) for u, v in edges[:20]] + edges[:30] + [(99, 99)]
        rng.shuffle(edges)
        valid.append(('\n'.join(f'{u}\t{v}' for u, v in edges) + ('\n' if seed % 2 else '')).encode())
    # Nontrivial merge rounds, uneven runs, and duplicate patterns.
    for n in [31, 32, 33, 63, 64, 65, 257]:
        edges = list(itertools.combinations(range(n), 2))
        if n == 257: edges = [(u, v) for u, v in edges if (u + v) % 7 == 0]
        valid.append(''.join(f'{u} {v}\n' for u, v in reversed(edges)).encode())
    invalid = [b'1 2 3\n', b'no edge\n', b'1\n', b'9223372036854775808 0\n',
               b'-9223372036854775809 0\n', b'+-1 2\n', b'\v\n', b'\v# comment\n',
               b'1 2 # trailing\n', b'1 2\x00\n', b'1 2\n# x\n3 a\n4 b\n',
               b'#' + b'x'*30000 + b'\n1 2\n3 4\n5\n']
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'input.txt'
        for data in valid:
            path.write_bytes(data)
            result = subprocess.run([equivalence, str(path)], text=True, capture_output=True, timeout=120)
            if result.returncode: raise AssertionError((data[:100], result.stderr))
        for data in invalid:
            path.write_bytes(data)
            reference = subprocess.run([cpu, str(path)], text=True, capture_output=True, timeout=30)
            line = re.search(r'Line (\d+):', reference.stderr)
            if reference.returncode == 0 or not line: raise AssertionError(reference)
            for parser, threads in [('fast', 1), ('omp', 1), ('omp', 3), ('omp', 8)]:
                result = subprocess.run([cpu, str(path), '--parser', parser, '--threads', str(threads)],
                                        text=True, capture_output=True, timeout=30)
                actual = re.search(r'Line (\d+):', result.stderr)
                if result.returncode == 0 or not actual or actual[1] != line[1]:
                    raise AssertionError((data[:100], parser, threads, reference.stderr, result.stderr))
        path.write_text('1 2\n')
        for flags in [('--threads','0'),('--threads','-1'),('--threads','257'),('--parser','bad'),('--sort','bad'),('--mapping','bad')]:
            result = subprocess.run([cpu,str(path),*flags],capture_output=True,timeout=30)
            if result.returncode == 0: raise AssertionError(flags)
    print(f'PASS: {len(valid)} inputs x54 exact-CSR configurations; {len(invalid)} malformed inputs preserve error lines; invalid modes rejected')


if __name__ == '__main__':
    main()
