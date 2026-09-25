"""Generate exact soft targets from partially observed latent spatial worlds."""
import argparse, json, random
from pathlib import Path
from systemone_lab.data.spatial_worlds import LABELS, relation
ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--n',type=int,default=10000); ap.add_argument('--seed',type=int,default=811); a=ap.parse_args(); rng=random.Random(a.seed); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
grid=[-2,-1,0,1,2]
with out.open('w',encoding='utf-8') as f:
    for i in range(a.n):
        # B fixed at origin; A is known to lie in a randomly selected candidate set. The exact conditional distribution is enumerable.
        candidates=rng.sample([(x,y) for x in grid for y in grid],k=rng.randint(2,8)); counts={k:0 for k in LABELS}
        for p in candidates: counts[relation(p,(0,0))]+=1
        dist={k:v/len(candidates) for k,v in counts.items()}
        shown='; '.join(f'({x},{y})' for x,y in candidates)
        state=f"B is at coordinate (0,0). A is equally likely to occupy exactly one of these possible coordinates: {shown}. No other position is possible."
        q={"type":"choice","instructions":"What is the spatial relation of A to B?","criteria":{x:x for x in LABELS}}
        label=max(dist,key=dist.get)
        rec={"id":f"probspace-{i}","state":state,"questions":{"spatial":q},"labels":{"spatial":label},"distributions":{"spatial":dist},"meta":{"source":"latent-probabilistic-spatial","candidates":candidates}}
        f.write(json.dumps(rec)+"\n")
print(out)
