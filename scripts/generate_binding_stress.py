"""Read-only stress suites for the entity-binding head: distractors and two-hop chains.

Every world is rendered with numbered and renamed objects and asked in both query orders,
so each record belongs to a role-swap pair (meta.pair_id, meta.query_role).

Suites:
  onehop_disconnected  one direct fact plus 1-3 disconnected facts about other objects
  onehop_mention       one direct fact plus 1-2 facts linking a new object to a queried object;
                       the queried relation is unchanged but the queried names occur more often
  twohop               a chain A-X-B with no direct fact between the queried objects
  twohop_disconnected  the same chain plus 1-2 disconnected facts
"""
import argparse
import json
import random
from pathlib import Path

from systemone_lab.data.spatial_worlds import (STEPS, make_latent, rename_latent, render_latent)

SUITES = ("onehop_disconnected", "onehop_mention", "twohop", "twohop_disconnected")


def mention_latent(rng):
    lat = make_latent(rng, hops=1, distractors=0)
    names = list(lat["names"]); coords = dict(lat["coords"]); edges = list(lat["edges"])
    for j in range(rng.randint(1, 2)):
        new = f"obj_{len(names)}"; anchor = rng.choice(lat["query"])
        dx, dy = rng.choice(STEPS); ax, ay = coords[anchor]
        coords[new] = (ax + dx * rng.randint(1, 3), ay + dy * rng.randint(1, 3))
        names.append(new)
        edges.append((new, anchor) if rng.random() < 0.5 else (anchor, new))
    order = list(range(len(edges))); rng.shuffle(order)
    return {**lat, "names": names, "coords": coords, "edges": edges, "edge_order": order,
            "template_slots": [rng.randrange(3) for _ in edges]}


def suite_latent(rng, suite):
    if suite == "onehop_disconnected": return make_latent(rng, hops=1, distractors=rng.randint(1, 3))
    if suite == "onehop_mention": return mention_latent(rng)
    if suite == "twohop": return make_latent(rng, hops=2, distractors=0)
    if suite == "twohop_disconnected": return make_latent(rng, hops=2, distractors=rng.randint(1, 2))
    raise ValueError(suite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds-per-suite", type=int, default=150)
    ap.add_argument("--seed", type=int, default=30917)
    a = ap.parse_args()
    rng = random.Random(a.seed); rows = []
    for suite in SUITES:
        for i in range(a.worlds_per_suite):
            lat = suite_latent(rng, suite)
            for variant, world in (("numbered", lat), ("renamed", rename_latent(lat, rng))):
                pair_id = f"{suite}-{i}-{variant}"
                for role in ("forward", "reversed"):
                    q = world["query"] if role == "forward" else world["query"][::-1]
                    ex = render_latent({**world, "query": q})
                    rows.append({"id": f"{pair_id}-{role}", "state": ex.state,
                                 "questions": {"spatial": ex.question}, "labels": {"spatial": ex.label},
                                 "meta": {"suite": suite, "pair_id": pair_id, "names": variant,
                                          "query_role": role, "hops": lat["hops"],
                                          "facts": len(lat["edges"])}})
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"records": len(rows), "suites": SUITES, "seed": a.seed}))


if __name__ == "__main__":
    main()
