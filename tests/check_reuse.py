"""Run on the CUDA server: independent tiny oracle plus actual hash-path checks."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import check_serial


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 tests/check_reuse.py BUILD/tc_gpu_query_v2')
    exe = str(Path(sys.argv[1]).resolve())
    for mode in ['reuse', 'hash']:
        sys.argv = [sys.argv[0], exe, '--query-mode', mode, '--verify', '--blocks', '1', '--warmup', '0', '--repeat', '1']
        check_serial.main()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'graph.txt'
        # Hash gate, partial batches, capacity fallback (>512), table reuse across
        # grid-stride iterations; repeated runs rebuild rather than retain tables.
        for n in [16,18,40,512,520]:
            path.write_text(''.join(f'{u} {v}\n' for u,v in itertools.combinations(range(n),2)))
            for mode in ['reuse','hash']:
                for blocks in [1,3]:
                    command = [exe,str(path),'--query-mode',mode,'--blocks',str(blocks),'--verify',
                               '--warmup','1','--repeat','3','--query-diagnostics']
                    row = json.loads(subprocess.run(command,check=True,capture_output=True,text=True,timeout=300).stdout)
                    assert row['verified'] and row['triangles']==math.comb(n,3)
                    stats=row['query_diagnostics']
                    assert stats['hash_batches']<=stats['common_target_batches']<=stats['batches']
                    if n>=40:
                        assert stats['common_target_batches']>0
                        if mode=='hash': assert stats['hash_batches']>0
                    if mode=='reuse': assert stats['hash_batches']==0
        # Preserve dense id gaps using isolated self-loop records. All clique
        # keys collide at the same initial slot modulo 1024; full-key checks
        # and linear probing must still count exactly, including misses.
        vertices=[1024*i for i in range(40)]
        path.write_text(''.join(f'{u} {v}\n' for u,v in itertools.combinations(vertices,2))
                        + ''.join(f'{v} {v}\n' for v in range(vertices[-1]+1)))
        row=json.loads(subprocess.run([exe,str(path),'--query-mode','hash','--blocks','1','--verify',
            '--warmup','1','--repeat','3','--query-diagnostics'],check=True,capture_output=True,text=True,timeout=300).stdout)
        assert row['triangles']==math.comb(40,3) and row['query_diagnostics']['hash_batches']>0
    print('PASS: reuse/hash tiny oracle, capacity boundary, repeated table reuse and collisions')


if __name__ == '__main__':
    main()
