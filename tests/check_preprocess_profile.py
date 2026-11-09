"""Validate profiled/unprofiled graph results, JSON fields, and accounting."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from check_serial import course_cases

STAGES = (
    'read_parse_normalize_ms', 'vertex_sort_unique_ms', 'edge_sort_unique_ms',
    'remap_degree_ms', 'orient_row_counts_ms', 'csr_scan_ms', 'csr_fill_ms',
    'neighbor_sort_ms',
)


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python3 tests/check_preprocess_profile.py EXECUTABLE [EXTRA OPTIONS]')
    executable = str(Path(sys.argv[1]).resolve())
    extra = sys.argv[2:]
    cases = course_cases()
    # Signed sparse IDs, only-loop vertices, duplicate and reverse edges.
    cases.append(('sparse-id', [(-5, 10**12), (10**12, 42), (42, -5),
                              (-5, 42), (77, 77)], (4, 3, 1)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'graph.txt'
        for name, edges, expected in cases:
            path.write_text('# comment\n% comment\n\n' + ''.join(f'{u} {v}\n' for u, v in edges))
            outputs = []
            for flags in ([], ['--profile-preprocess']):
                result = subprocess.run([executable, str(path), *extra, *flags],
                                        capture_output=True, text=True, check=True, timeout=120)
                outputs.append(json.loads(result.stdout))
            plain, profiled = outputs
            for output in outputs:
                actual = tuple(output[k] for k in ('vertices', 'undirected_edges', 'triangles'))
                if actual != expected:
                    raise AssertionError((name, actual, expected))
            if 'preprocess_profile' in plain:
                raise AssertionError('profiling unexpectedly enabled by default')
            profile = profiled['preprocess_profile']
            if set(profile) != set(STAGES) | {'accounted_ms', 'unattributed_ms'}:
                raise AssertionError(('unexpected profile schema', profile))
            if not all(math.isfinite(x) and x >= 0 for x in profile.values()):
                raise AssertionError(('invalid time', profile))
            if not math.isclose(sum(profile[k] for k in STAGES), profile['accounted_ms'], rel_tol=1e-7, abs_tol=1e-7):
                raise AssertionError('stage sum mismatch')
            if not math.isclose(profile['accounted_ms'] + profile['unattributed_ms'], profiled['preprocess_ms'], rel_tol=1e-7, abs_tol=1e-7):
                raise AssertionError('outer time accounting mismatch')
    print(f'PASS: {len(cases)} profiled/unprofiled graphs agree; timing schema and accounting valid')


if __name__ == '__main__':
    main()
