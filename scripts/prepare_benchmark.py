"""List/download/generate the benchmark catalog, using only Python's stdlib (+ wget for downloads)."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / 'data' / 'benchmark.json'
GENERATOR_VERSION = 'tc-synthetic-v1'


def load_catalog(path=CATALOG):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    ids = [row['id'] for row in data['datasets']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate benchmark IDs')
    return data


def synthetic_spec(family, p):
    """Analytic graph counts, independently checked by tiny-graph enumeration in tests."""
    def positive(key, minimum=1):
        n = p[key]
        if type(n) is not int or n < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
        return n
    if family == 'clique_chain':
        c, k = positive('cliques'), positive('size', 2)
        return dict(vertices=c*k, undirected_edges=c*math.comb(k,2)+c-1, triangles=c*math.comb(k,3))
    if family == 'grid':
        r, c = positive('rows'), positive('cols')
        return dict(vertices=r*c, undirected_edges=r*(c-1)+(r-1)*c, triangles=0)
    if family == 'shared_hub':
        g, k = positive('groups'), positive('size')
        return dict(vertices=1+g*k, undirected_edges=g*(math.comb(k,2)+k), triangles=g*math.comb(k+1,3))
    if family == 'mixed_cliques':
        a, b = positive('small_count',0), positive('large_count',0)
        psize, qsize = positive('small_size',2), positive('large_size',2)
        if a+b == 0:
            raise ValueError('at least one clique required')
        return dict(vertices=a*psize+b*qsize,
                    undirected_edges=a*math.comb(psize,2)+b*math.comb(qsize,2),
                    triangles=a*math.comb(psize,3)+b*math.comb(qsize,3))
    raise ValueError(f'unknown synthetic family: {family}')


def synthetic_edges(family, p):
    synthetic_spec(family,p)  # validate before yielding
    if family == 'clique_chain':
        k=p['size']
        for group in range(p['cliques']):
            base=group*k
            yield from itertools.combinations(range(base,base+k),2)
            if group:
                yield (base-k,base)  # only consecutive first vertices; every connector is a bridge
    elif family == 'grid':
        r,c=p['rows'],p['cols']
        if r*c == 1:
            yield (0,0)  # declare the isolated vertex; removed by normal graph cleaning
        for y in range(r):
            for x in range(c):
                u=y*c+x
                if x+1<c: yield (u,u+1)
                if y+1<r: yield (u,u+c)
    elif family == 'shared_hub':
        k=p['size']
        for group in range(p['groups']):
            base=1+group*k
            yield from itertools.combinations(range(base,base+k),2)
            for v in range(base,base+k): yield (0,v)
    else:
        base=0
        for count,k in [(p['small_count'],p['small_size']),(p['large_count'],p['large_size'])]:
            for _ in range(count):
                yield from itertools.combinations(range(base,base+k),2)
                base+=k


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def generate(row, target):
    """Stream deterministic edge records; do not materialize a large graph in RAM."""
    target=Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_suffix(target.suffix+'.part')
    if part.exists():
        raise FileExistsError(f'stale generation temporary file: {part}')
    expected=synthetic_spec(row['family'],row['parameters'])
    digest=hashlib.sha256(); count=0
    try:
        with part.open('xb') as out:
            def write(text):
                block=text.encode('utf-8');out.write(block);digest.update(block)
            write(f"# {row['id']} {GENERATOR_VERSION}\n# parameters: {json.dumps(row['parameters'],sort_keys=True)}\n")
            lines=[]
            for u,v in synthetic_edges(row['family'],row['parameters']):
                lines.append(f'{u} {v}\n');count+=1
                if len(lines)>=8192:
                    write(''.join(lines));lines=[]
            if lines: write(''.join(lines))
        extra_isolate = row['family']=='grid' and expected['vertices']==1
        if count != expected['undirected_edges']+int(extra_isolate):
            raise RuntimeError('generated record count disagrees with formula')
        part.rename(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {'id':row['id'],'generator_version':GENERATOR_VERSION,'family':row['family'],
            'parameters':row['parameters'],'deterministic':True,'edge_order':'deterministic construction order; no shuffle or relabeling',
            'expected_clean_graph':expected,'input_records':count,'sha256':digest.hexdigest(),
            'bytes':target.stat().st_size,'status':'generated; analytic counts are not a GPU benchmark result'}


def unpack_gzip(archive,target):
    target=Path(target)
    part=target.with_suffix(target.suffix+'.unpack.part')
    if target.exists() or part.exists():
        raise FileExistsError(target if target.exists() else part)
    try:
        with gzip.open(archive,'rb') as source, part.open('xb') as out:
            shutil.copyfileobj(source,out,1024*1024)  # reads to EOF and validates gzip CRC
        part.rename(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def selected_rows(catalog,suite,ids):
    rows=catalog['datasets']; known={row['id']:row for row in rows}
    if ids:
        if len(ids)!=len(set(ids)) or any(id not in known for id in ids):
            raise ValueError('IDs must be known and distinct')
        return [known[id] for id in ids]
    if suite=='course':
        return [row for row in rows if row['id'] in ['C1','C2','C3','S1','S2','S3']]
    return [row for row in rows if suite=='extended' or row['tier']=='core']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['list','paths','download','generate'])
    parser.add_argument('--catalog',type=Path,default=CATALOG)
    parser.add_argument('--suite',choices=['course','core','extended'],default='core')
    parser.add_argument('--ids',nargs='+',help='explicit IDs override suite selection')
    parser.add_argument('--data-dir',type=Path,default=REPO/'data')
    args=parser.parse_args()
    try:
        catalog=load_catalog(args.catalog); rows=selected_rows(catalog,args.suite,args.ids)
        if args.action=='paths':
            paths=[args.data_dir/Path(row['path']).relative_to('data') for row in rows]
            missing=[str(p) for p in paths if not p.is_file()]
            if missing: raise ValueError('prepare these inputs first: '+', '.join(missing))
            for path in paths: print(path.resolve())
            return
        for row in rows:
            if args.action=='list':
                details=row.get('download_url') or json.dumps(row['parameters'],sort_keys=True)
                print(row['id'],row['kind'],row['path'],details,sep='\t');continue
            if row['kind'] != ('download' if args.action=='download' else 'generate'):
                continue
            relative=Path(row['path']).relative_to('data')
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('invalid catalog path')
            target=args.data_dir/relative
            if target.exists():
                # Existing files are deliberately not overwritten or relabeled as verified downloads.
                print(f'SKIP {row["id"]}: {target} exists; SHA256={sha256(target)}',flush=True)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            print(f'{args.action.upper()} {row["id"]}: {target}',flush=True)
            if args.action=='generate':
                metadata=generate(row,target)
            else:
                if not shutil.which('wget'): raise RuntimeError('download requires wget; URLs are also listed in data/benchmark.txt')
                archive=target.with_suffix(target.suffix+'.gz')
                # -c permits resuming an interrupted download; gzip CRC verifies the final stream.
                subprocess.run(['wget','-c','--timeout=30','--tries=3','-O',str(archive),row['download_url']],check=True)
                unpack_gzip(archive,target)
                metadata={'id':row['id'],'download_url':row['download_url'],'source_page':row['source_page'],
                          'archive_sha256':sha256(archive),'sha256':sha256(target),'bytes':target.stat().st_size,
                          'status':'gzip CRC checked; local hashes recorded (not a publisher-provided checksum); graph counts must be measured'}
            target.with_suffix(target.suffix+'.meta.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(metadata,ensure_ascii=False),flush=True)
    except (ValueError,OSError,EOFError,RuntimeError,subprocess.CalledProcessError) as error:
        parser.exit(1,f'ERROR: {error}\n')


if __name__=='__main__':
    main()
