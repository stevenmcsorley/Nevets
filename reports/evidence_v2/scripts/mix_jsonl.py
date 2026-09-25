"""Create a deterministic weighted JSONL curriculum. Example: --source spatial.jsonl:5 --source decisions.jsonl:1"""
import argparse, json, random
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--source',action='append',required=True); ap.add_argument('--n',type=int,required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',required=True); a=ap.parse_args(); rng=random.Random(a.seed)
sources=[]
for spec in a.source:
    path,weight=spec.rsplit(':',1); rows=[x for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]; sources.append((path,float(weight),rows))
weights=[s[1] for s in sources]; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); counts={s[0]:0 for s in sources}
with out.open('w',encoding='utf-8') as f:
    for _ in range(a.n):
        idx=rng.choices(range(len(sources)),weights=weights,k=1)[0]; path,_,rows=sources[idx]; f.write(rng.choice(rows)+'\n'); counts[path]+=1
print(json.dumps(counts,indent=2))
