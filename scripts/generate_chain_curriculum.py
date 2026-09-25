"""Multi-hop chain worlds with branch and disconnected distractors.

Each world is a chain of unit steps, optional branch objects attached to chain objects (they mention
chain names but do not change any chain relation) and optional disconnected pairs. Facts are shuffled.
The query is two chain objects `hops` links apart. Every world is rendered with numbered and renamed
objects and asked in both query orders (meta.pair_id / meta.query_role), so the same file serves the
role-swap and naming gates. Numbered names are drawn at random so the numbering never reveals chain order.
"""
import argparse
import json
import random
import re
from pathlib import Path

from systemone_lab.data.spatial_worlds import STEPS, rename_latent, render_latent


def chain_latent(rng, hops, branches, disconnected):
    n_chain = hops + 1 + rng.randint(0, 1)  # sometimes a longer chain than the queried span
    total = n_chain + branches + 2 * disconnected
    names = [f"obj_{i}" for i in rng.sample(range(max(40, total * 2)), total)]
    coords = {names[0]: (0, 0)}; edges = []
    for i in range(1, n_chain):
        dx, dy = rng.choice(STEPS); px, py = coords[names[i - 1]]; coords[names[i]] = (px + dx, py + dy)
        edges.append((names[i], names[i - 1]) if rng.random() < 0.5 else (names[i - 1], names[i]))
    for j in range(branches):
        new = names[n_chain + j]; anchor = rng.choice(names[:n_chain])
        dx, dy = rng.choice(STEPS); ax, ay = coords[anchor]; coords[new] = (ax + dx, ay + dy)
        edges.append((new, anchor) if rng.random() < 0.5 else (anchor, new))
    base = n_chain + branches
    for j in range(disconnected):
        a, b = names[base + 2 * j], names[base + 2 * j + 1]
        coords[a] = (rng.randint(-8, 8), rng.randint(-8, 8)); dx, dy = rng.choice(STEPS)
        coords[b] = (coords[a][0] + dx, coords[a][1] + dy); edges.append((b, a))
    start = rng.randint(0, n_chain - 1 - hops)
    query = (names[start + hops], names[start]) if rng.random() < 0.5 else (names[start], names[start + hops])
    order = list(range(len(edges))); rng.shuffle(order)
    return {"names": names, "coords": coords, "edges": edges, "edge_order": order,
            "template_slots": [rng.randrange(6) for _ in edges], "query": query, "hops": hops}


def aux_coords(world, state):
    """Training-only targets: position of every object in the queried component, relative to the
    component object mentioned first in the state (a reference computable from the state alone)."""
    component = {world["query"][1]}; changed = True
    while changed:
        changed = False
        for a, b in world["edges"]:
            if (a in component) != (b in component):
                component |= {a, b}; changed = True
    first = {n: re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", state).start() for n in component}
    ref = min(component, key=first.get); rx, ry = world["coords"][ref]
    return {n: [world["coords"][n][0] - rx, world["coords"][n][1] - ry] for n in sorted(component)}


def aux_dist(world, coords):
    """Hop distance from the reference (the object at [0, 0]) over the undirected fact graph.

    Training-only hints for looped models: iteration t should know every object within t hops.
    """
    ref = next(n for n, xy in coords.items() if xy == [0, 0] and n in coords)
    adj = {}
    for a, b in world["edges"]:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    dist, frontier = {ref: 0}, [ref]
    while frontier:
        nxt = []
        for x in frontier:
            for y in sorted(adj.get(x, [])):
                if y not in dist and y in coords: dist[y] = dist[x] + 1; nxt.append(y)
        frontier = nxt
    return {n: dist[n] for n in sorted(coords) if n in dist}


def generate(n_worlds, seed, min_hops, max_hops, max_branches, max_disconnected, prefix):
    rng = random.Random(seed); rows = []
    for i in range(n_worlds):
        hops = rng.randint(min_hops, max_hops)
        lat = chain_latent(rng, hops, rng.randint(0, max_branches), rng.randint(0, max_disconnected))
        for variant, world in (("numbered", lat), ("renamed", rename_latent(lat, rng))):
            pair_id = f"{prefix}-{i}-{variant}"
            for role in ("forward", "reversed"):
                q = world["query"] if role == "forward" else world["query"][::-1]
                ex = render_latent({**world, "query": q})
                rows.append({"id": f"{pair_id}-{role}", "state": ex.state, "questions": {"spatial": ex.question},
                             "labels": {"spatial": ex.label},
                             "meta": {"suite": f"hops{hops}", "pair_id": pair_id, "names": variant, "query_role": role,
                                      "hops": hops, "facts": len(lat["edges"]),
                                      "aux_coords": (ac := aux_coords(world, ex.state)),
                                      "aux_dist": aux_dist(world, ac)}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--min-hops", type=int, default=1)
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--max-branches", type=int, default=2)
    ap.add_argument("--max-disconnected", type=int, default=2)
    ap.add_argument("--exclude", action="append", default=[], help="jsonl whose states must not appear")
    a = ap.parse_args()
    rows = generate(a.worlds, a.seed, a.min_hops, a.max_hops, a.max_branches, a.max_disconnected, f"chain{a.seed}")
    banned = set()
    for path in a.exclude:
        banned |= {json.loads(line)["state"] for line in open(path, encoding="utf-8")}
    bad = {r["meta"]["pair_id"] for r in rows if r["state"] in banned}
    rows = [r for r in rows if r["meta"]["pair_id"] not in bad]
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"records": len(rows), "dropped_pairs": len(bad), "seed": a.seed}))


if __name__ == "__main__":
    main()
