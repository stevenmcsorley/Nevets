from __future__ import annotations
import json, random
from dataclasses import dataclass
from pathlib import Path

REL = {
    (-1, 1): "upper-left", (0, 1): "above", (1, 1): "upper-right",
    (-1, 0): "left", (0, 0): "overlap", (1, 0): "right",
    (-1,-1): "lower-left", (0,-1): "below", (1,-1): "lower-right",
}
LABELS = list(REL.values())
INVERSE_REL = {label: REL[(-dx, -dy)] for (dx, dy), label in REL.items()}
TEMPLATES = {
    "left": ["{a} is left of {b}.", "{a} sits west of {b}.", "{b} is to the right of {a}."],
    "right": ["{a} is right of {b}.", "{a} sits east of {b}.", "{b} is to the left of {a}."],
    "above": ["{a} is above {b}.", "{a} sits north of {b}.", "{b} is below {a}."],
    "below": ["{a} is below {b}.", "{a} sits south of {b}.", "{b} is above {a}."],
    "upper-left": ["{a} is upper-left of {b}.", "{a} is northwest of {b}."],
    "upper-right": ["{a} is upper-right of {b}.", "{a} is northeast of {b}."],
    "lower-left": ["{a} is lower-left of {b}.", "{a} is southwest of {b}."],
    "lower-right": ["{a} is lower-right of {b}.", "{a} is southeast of {b}."],
    "overlap": ["{a} occupies the same position as {b}."]
}
STEPS=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

@dataclass
class SpatialExample:
    state: str
    question: dict
    label: str
    meta: dict

def sign(x): return 0 if x == 0 else (1 if x > 0 else -1)
def relation(a, b):
    dx,dy = a[0]-b[0], a[1]-b[1]
    return REL[(sign(dx), sign(dy))]

def transform_point(p, kind):
    x,y=p
    return {"identity":(x,y),"rot90":(-y,x),"rot180":(-x,-y),"rot270":(y,-x),"mirror_x":(-x,y),"mirror_y":(x,-y)}[kind]

def make_latent(rng: random.Random,hops=4,distractors=2):
    names=[f"obj_{i}" for i in range(hops+1+distractors*2)]
    coords={names[0]:(0,0)}; edges=[]
    for i in range(1,hops+1):
        dx,dy=rng.choice(STEPS); px,py=coords[names[i-1]]; coords[names[i]]=(px+dx,py+dy); edges.append((names[i],names[i-1]))
    base=hops+1
    for j in range(distractors):
        a,b=names[base+2*j],names[base+2*j+1]; coords[a]=(rng.randint(-6,6),rng.randint(-6,6)); dx,dy=rng.choice(STEPS); coords[b]=(coords[a][0]+dx,coords[a][1]+dy); edges.append((b,a))
    edge_order=list(range(len(edges))); rng.shuffle(edge_order)
    template_slots=[rng.randrange(3) for _ in edges]
    return {"names":names,"coords":coords,"edges":edges,"edge_order":edge_order,"template_slots":template_slots,"query":(names[hops],names[0]),"hops":hops}

def rename_latent(latent,rng):
    alphabet=list("ABCDEFGHJKLMNPQRSTUVWXYZ"); rng.shuffle(alphabet)
    mapping={n:f"{alphabet[i%len(alphabet)]}{i//len(alphabet) or ''}" for i,n in enumerate(latent["names"])}
    return {**latent,"names":[mapping[n] for n in latent["names"]],"coords":{mapping[k]:v for k,v in latent["coords"].items()},"edges":[(mapping[a],mapping[b]) for a,b in latent["edges"]],"query":tuple(mapping[x] for x in latent["query"])}

def transformed_latent(latent,kind="identity",translate=(0,0)):
    tx,ty=translate
    return {**latent,"coords":{k:(transform_point(v,kind)[0]+tx,transform_point(v,kind)[1]+ty) for k,v in latent["coords"].items()}}

def render_latent(latent,rng=None):
    parts=[]
    for idx in latent["edge_order"]:
        a,b=latent["edges"][idx]; r=relation(latent["coords"][a],latent["coords"][b]); variants=TEMPLATES[r]; slot=latent["template_slots"][idx] % len(variants); parts.append(variants[slot].format(a=a,b=b))
    a,b=latent["query"]; label=relation(latent["coords"][a],latent["coords"][b])
    return SpatialExample(" ".join(parts),{"type":"choice","instructions":f"What is the spatial relation of {a} to {b}?","criteria":{x:x for x in LABELS}},label,{"hops":latent["hops"],"query":[a,b],"coords":latent["coords"]})

def make_example(rng: random.Random,hops=4,distractors=2,transform="identity",translate=(0,0),renamed=False):
    lat=make_latent(rng,hops,distractors)
    if renamed: lat=rename_latent(lat,rng)
    lat=transformed_latent(lat,transform,translate)
    ex=render_latent(lat); ex.meta.update({"transform":transform,"translate":translate,"renamed":renamed}); return ex

def make_counterfactual_pair(rng: random.Random,hops=4,distractors=2):
    """Minimal pair: move only the queried endpoint until its relation to origin changes."""
    lat=make_latent(rng,hops,distractors); base=render_latent(lat); moving,anchor=lat["query"]; before=base.label
    cf={**lat,"coords":dict(lat["coords"])}
    for _ in range(100):
        dx,dy=rng.choice([(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2)])
        x,y=lat["coords"][moving]; cf["coords"][moving]=(x+dx,y+dy)
        if relation(cf["coords"][moving],cf["coords"][anchor]) != before: break
    changed=render_latent(cf)
    pair_id=f"pair-{rng.getrandbits(64):016x}"
    base.meta.update({"pair_id":pair_id,"variant":"base"}); changed.meta.update({"pair_id":pair_id,"variant":"counterfactual","intervention":f"move {moving}"})
    return base,changed

def generate_jsonl(out: str|Path,n:int,seed=1234,split="train",min_hops=1,max_hops=6,rename_probability:float=0.5):
    if not 0 <= rename_probability <= 1:
        raise ValueError("rename_probability must be between 0 and 1")
    rng=random.Random(seed); p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for i in range(n):
            hops=rng.randint(min_hops,max_hops); transforms=["identity"] if split=="train" else ["identity","rot90","rot180","rot270","mirror_x","mirror_y"]
            ex=make_example(rng,hops,rng.randint(0,4),rng.choice(transforms),(rng.randint(-20,20),rng.randint(-20,20)) if split!="train" else (0,0),renamed=(rng.random()<rename_probability))
            rec={"id":f"spatial-{split}-{seed}-{i}","state":ex.state,"questions":{"spatial":ex.question},"labels":{"spatial":ex.label},"meta":ex.meta}; f.write(json.dumps(rec,ensure_ascii=False)+"\n")

def generate_counterfactual_jsonl(out: str|Path,pairs:int,seed=4401,min_hops=2,max_hops=8):
    rng=random.Random(seed); p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for i in range(pairs):
            a,b=make_counterfactual_pair(rng,rng.randint(min_hops,max_hops),rng.randint(0,3))
            for ex in (a,b):
                rec={"id":f"cf-{i}-{ex.meta['variant']}","state":ex.state,"questions":{"spatial":ex.question},"labels":{"spatial":ex.label},"meta":ex.meta}; f.write(json.dumps(rec,ensure_ascii=False)+"\n")
