"""Balanced 1-2-hop name curriculum, with disjoint held-out and paired name tests."""
import json, random
from pathlib import Path
from systemone_lab.data.spatial_worlds import LABELS,make_latent,rename_latent,render_latent

def record(ex,i):
    return {'id':i,'state':ex.state,'questions':{'spatial':ex.question},'labels':{'spatial':ex.label},'meta':ex.meta}

def main():
    rng=random.Random(99817); seen=set(); root=Path('reports/name_curriculum'); root.mkdir(exist_ok=True)
    for split,n in [('train',800),('heldout',200)]:
        records=[]
        for i in range(n):
            label=LABELS[i%9]; rename=(i//9)%2==0
            while True:
                latent=make_latent(rng,rng.choice([1,2]),0)
                ex=render_latent(rename_latent(latent,rng) if rename else latent)
                key=(ex.state,ex.question['instructions'])
                if ex.label==label and key not in seen: break
            seen.add(key); ex.meta['renamed']=rename; records.append(record(ex,f'{split}-{i}'))
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    pairs=[]
    for i in range(200):
        label=LABELS[i%9]
        while True:
            latent=make_latent(rng,rng.choice([1,2]),0); a=render_latent(latent); b=render_latent(rename_latent(latent,rng))
            keya=(a.state,a.question['instructions']); keyb=(b.state,b.question['instructions'])
            if a.label==label and keya not in seen and keyb not in seen: break
        seen.update([keya,keyb]); pairs.append({'id':i,'label':label,'numbered':record(a,f'name-{i}-numbered'),'renamed':record(b,f'name-{i}-renamed')})
    (root/'name_pairs.json').write_text(json.dumps(pairs,indent=2))
    assert len(seen)==1400
    print(root)
if __name__=='__main__': main()
