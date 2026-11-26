"""Deterministic structural coverage, not preassigned optimal-schedule labels."""
import argparse,hashlib,json,math,random
from pathlib import Path
import prepare_benchmark as gen

def specs(tiny=False):
 rows=[]
 def add(regime,family,p):rows.append(dict(id=f'g{len(rows):02d}',regime=regime,family=family,parameters=p))
 for k in [4,32,128]:
  for c in [1,2]:add('low_work','clique_chain',dict(cliques=c,size=min(k,8) if tiny else k))
 for n in [8,16,32]:add('low_work','grid',dict(rows=min(n,4) if tiny else n,cols=min(n,4) if tiny else n))
 for n in [32,256,2048]:add('low_work','star',dict(leaves=min(n,12) if tiny else n))
 for k in [8,32,128]:
  for target in [65536,262144,2097152]:add('normal','clique_chain',dict(cliques=2 if tiny else max(1,target//math.comb(k,2)),size=min(k,8) if tiny else k))
 for n in [128,512,1024]:add('normal','grid',dict(rows=4 if tiny else n,cols=4 if tiny else n))
 for k in [128,512,1024]:
  for fraction in [.125,.5,.875]:
   target=2097152;large=max(1,int(target*fraction/math.comb(k,2)));small=max(0,(target-large*math.comb(k,2))//28)
   add('skew','mixed_cliques',dict(small_count=3 if tiny else small,small_size=4 if tiny else 8,large_count=1 if tiny else large,large_size=12 if tiny else k))
 for n in [1024,4096,16384]:add('skew','skew_random',dict(vertices=24 if tiny else n,seed=42000+n))
 assert len(rows)==36
 for i,row in enumerate(rows):row['group']=row['id'];row['split']='validation' if i%3==2 else 'train'
 return rows

def records(row):
 family,p=row['family'],row['parameters']
 if family in ['clique_chain','mixed_cliques','grid']:yield from gen.synthetic_edges(family,p)
 elif family=='star':yield from ((0,v) for v in range(1,p['leaves']+1))
 else:
  rng=random.Random(p['seed']);n=p['vertices'];edges=set()
  for u in range(n):
   degree=min(n-1,int(rng.paretovariate(1.8)*2))
   for v in rng.sample(range(n-1),degree):
    v+=v>=u;edges.add((min(u,v),max(u,v)))
  # Self-loops declare any isolated vertices, matching the project's parser.
  yield from ((v,v) for v in range(n));yield from sorted(edges)

def write_graph(row,target,permutation=None):
 h=hashlib.sha256();count=0;part=target.with_suffix('.part')
 with part.open('xb') as f:
  def put(s):b=s.encode();f.write(b);h.update(b)
  put('# schedule-training-v1 '+json.dumps(row,sort_keys=True)+'\n')
  for u,v in records(row):
   if permutation is not None:u,v=permutation[u],permutation[v]
   put(f'{u} {v}\n');count+=1
 part.rename(target)
 return dict(path=target.name,sha256=h.hexdigest(),input_records=count,bytes=target.stat().st_size)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--tiny',action='store_true');p.add_argument('--relabel-ids',nargs='*',default=[]);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 rows=specs(a.tiny)
 if any(x not in {r['id'] for r in rows} for x in a.relabel_ids):p.error('unknown relabel ID')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);result=[]
 for row in rows:
  r=dict(row);r.update(write_graph(row,out/(row['id']+'.txt')))
  if row['family'] in ['clique_chain','mixed_cliques','grid']:r['expected_clean_graph']=gen.synthetic_spec(row['family'],row['parameters'])
  elif row['family']=='star':r['expected_clean_graph']=dict(vertices=row['parameters']['leaves']+1,undirected_edges=row['parameters']['leaves'],triangles=0)
  result.append(r);print(r['id'],r['regime'],r['split'],r['bytes'],flush=True)
  if row['id'] in a.relabel_ids:
   # A second pass determines the declared vertex range, without storing all edges.
   n=1+max((max(e) for e in records(row)),default=-1);permutation=list(range(n));random.Random(a.seed).shuffle(permutation)
   variant=dict(row,id=row['id']+'_relabel',relabel_seed=a.seed)
   variant.update(write_graph(row,out/(variant['id']+'.txt'),permutation));result.append(variant)
 manifest=dict(schema='schedule-training-v1',debug_tiny=a.tiny,seed=a.seed,graphs=result,
               note='Regime is generation intent, not a measured performance label. Splits are by base graph group; inspect oriented-query features. Relabel variants stay in their base split. Synthetic parameter holdout, not family holdout or unseen real validation.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out/'manifest.json')
if __name__=='__main__':main()
