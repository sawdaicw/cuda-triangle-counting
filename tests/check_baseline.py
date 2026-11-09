"""GPU-only regression: course cases, partial blocks, and repeated execution."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import check_serial


def run(executable, path, *options):
    result = subprocess.run([executable, str(path), *options], text=True,
                            capture_output=True, check=True, timeout=120)
    output = json.loads(result.stdout)
    if not output['verified']:
        raise AssertionError(('verification not performed', output))
    if len(output['count_reduce_ms_samples']) != output['repeat']:
        raise AssertionError('wrong number of timing samples')
    return output


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 tests/check_baseline.py ./build/tc_gpu_edge_baseline')
    executable = str(Path(sys.argv[1]).resolve())
    # Reuse all 76 graph fixtures and malformed inputs. One forced block also
    # exercises inactive threads at collective reduction points.
    sys.argv = [sys.argv[0], executable, '--verify', '--repeat', '1', '--warmup', '0', '--blocks', '1']
    check_serial.main()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'boundary.txt'
        for m in [0, 1, 31, 32, 33, 255, 256, 257, 513]:
            path.write_text(''.join(f'{2*i} {2*i+1}\n' for i in range(m)))
            for blocks in [1, 2]:
                output = run(executable, path, '--verify', '--blocks', str(blocks), '--repeat', '3', '--warmup', '1')
                if (output['vertices'], output['undirected_edges'], output['triangles']) != (2*m, m, 0):
                    raise AssertionError((m, blocks, output))
        # More edges than a single block has threads; many nonzero tasks must
        # contribute on later grid-stride iterations. K64 has C(64,3)=41664.
        path.write_text(''.join(f'{u} {v}\n' for u, v in itertools.combinations(range(64), 2)))
        for blocks in [1, 2]:
            output = run(executable, path, '--verify', '--blocks', str(blocks), '--repeat', '5', '--warmup', '2')
            if (output['vertices'], output['undirected_edges'], output['triangles']) != (64, 2016, 41664):
                raise AssertionError(output)
        for options in [('--repeat', '0'), ('--warmup', '-1'), ('--blocks', '0'), ('--unknown',)]:
            result = subprocess.run([executable, str(path), *options], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                raise AssertionError(('invalid option accepted', options))
    print('PASS: baseline partial blocks, grid-stride iterations, repeat consistency and invalid options')


if __name__ == '__main__':
    main()
