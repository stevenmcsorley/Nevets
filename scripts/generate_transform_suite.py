"""Rotation/reflection consistency suite.

Each chain world (with branch and disconnected distractors) is rendered under identity, three rotations
and two mirrors. meta.world_id groups the renderings and meta.transform names the transform, so an
evaluator can check that the prediction on a transformed world equals the transformed prediction.
Translation is not rendered: every fact is relative, so translating a world leaves its text unchanged.
"""
import argparse
import json
import random
from pathlib import Path

from generate_chain_curriculum import chain_latent
from systemone_lab.data.spatial_worlds import rename_latent, render_latent, transformed_latent

TRANSFORMS = ("identity", "rot90", "rot180", "rot270", "mirror_x", "mirror_y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=53001)
    ap.add_argument("--min-hops", type=int, default=1)
    ap.add_argument("--max-hops", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed); rows = []
    for i in range(a.worlds):
        hops = rng.randint(a.min_hops, a.max_hops)
        lat = chain_latent(rng, hops, rng.randint(0, 2), rng.randint(0, 2))
        variant = "renamed" if i % 2 else "numbered"
        if variant == "renamed": lat = rename_latent(lat, rng)
        for t in TRANSFORMS:
            ex = render_latent(transformed_latent(lat, t))
            rows.append({"id": f"tf{a.seed}-{i}-{t}", "state": ex.state, "questions": {"spatial": ex.question},
                         "labels": {"spatial": ex.label},
                         "meta": {"suite": f"hops{hops}", "world_id": f"tf{a.seed}-{i}", "transform": t,
                                  "names": variant, "hops": hops}})
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"records": len(rows), "worlds": a.worlds, "seed": a.seed}))


if __name__ == "__main__":
    main()
