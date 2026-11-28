"""Server checks for the two gates, shared hash and host baseline bypass."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import check_serial


def main():
    exe=str(Path(sys.argv[1]).resolve())
    for mode in ['reuse-gated','hash-gated']:
        sys.argv=[sys.argv[0],exe,'--query-mode',mode,'--small','8','--verify',
                  '--blocks','1','--warmup','0','--repeat','1']
        check_serial.main()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'g.txt'
        for n in [18,80,128,520]:
            path.write_text(''.join(f'{u} {v}\n' for u,v in itertools.combinations(range(n),2)))
            for mode in ['reuse-gated','hash-gated']:
                for threshold in [0,262144]:
                    cmd=[exe,str(path),'--query-mode',mode,'--small','8','--bypass-edges',str(threshold),
                         '--verify','--blocks','3','--warmup','1','--repeat','3']
                    row=json.loads(subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=300).stdout)
                    assert row['triangles']==math.comb(n,3) and row['verified']
                    assert row['bypass_used']==(math.comb(n,2)<threshold)
                row=json.loads(subprocess.run([exe,str(path),'--query-mode',mode,'--small','8',
                    '--verify','--blocks','1','--warmup','0','--repeat','1','--query-diagnostics'],
                    check=True,capture_output=True,text=True,timeout=300).stdout)
                stats=row['query_diagnostics']
                if n>=80 and mode=='hash-gated': assert stats['hash_batches']>0
                if mode=='reuse-gated': assert stats['hash_batches']==0
                assert stats['fast_batches']+stats['scalar_batches']+stats['pool_batches']==stats['batches']
                assert stats['hash_batches']<=stats['common_target_batches']<=stats['pool_batches']
    print('PASS: gates, strict shared hash, capacity fallback and baseline bypass')


if __name__=='__main__':
    main()
