"""Nested vertex-induced development graphs for small-input bypass experiments."""
import argparse,hashlib,json,random
from pathlib import Path

def edges(path):
 with path.open() as f:
  for line in f:
   s=line.strip()
   if not s or s.startswith(('#','%')):continue
   fields=s.split()
   if len(fields)!=2:raise ValueError('expected exactly two vertex IDs')
   yield tuple(map(int,fields))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--sizes',type=int,nargs='+',default=[128,512,2048,4096]);a=p.parse_args()
 path=Path(a.input);vertices=sorted({v for e in edges(path) for v in e});random.Random(a.seed).shuffle(vertices)
 if not a.sizes or any(n<1 or n>len(vertices) for n in a.sizes) or len(set(a.sizes))!=len(a.sizes):p.error('sizes must be distinct and <= source vertices')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=[]
 for size in sorted(a.sizes):
  chosen=set(vertices[:size]);target=out/f'{path.stem}_n{size}.txt';clean=set()
  with target.open('w') as f:
   f.write('# Nested induced development graph; self-loops declare isolated vertices, then are removed.\n')
   for v in sorted(chosen):f.write(f'{v} {v}\n')
   for u,v in edges(path):
    if u in chosen and v in chosen:
     f.write(f'{u} {v}\n')
     if u!=v:clean.add((min(u,v),max(u,v)))
  rows.append(dict(path=str(target),vertices=size,undirected_edges=len(clean),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
 (out/'manifest.json').write_text(json.dumps(dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=a.seed,method='fixed shuffled vertex order, nested induced subsets; preserve original IDs; not an unbiased sample of all graphs',graphs=rows),indent=2))
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
