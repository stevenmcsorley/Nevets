"""GENERAL-3 training data (owner design, 27 Sep).

Worlds for the 7 general domains are generated from a FRESH seed and rendered TWICE, in two different formats
drawn from V3_TRAIN_FORMATS (prose, json, kv, shuffled-field json, csv, key=value, bullets); both renderings
share meta.render_pair so the trainer can apply a same-world consistency loss. Never rendered: P3's held-out
formats (pipe table, compact symbolic). A pair is kept only if both renderings share labels and questions, and
dropped if either state appears in any registered eval file. Spatial data = spatial_only_v1 (S30's own).
"""
import argparse
import json
import random
from pathlib import Path

from systemone_lab.data.io import read_jsonl
from systemone_lab.eval_registry import eval_states
from systemone_lab.worlds import DOMAINS, V3_TRAIN_FORMATS

DOMS = ["kinship", "temporal", "dependency", "probability", "rules", "infogather", "causal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100000); ap.add_argument("--seed", type=int, default=93001)
    ap.add_argument("--out", default="data/processed/general_v3")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev = eval_states(); rng = random.Random(a.seed); recs = []; dropped = {"eval_collision": 0, "mismatch": 0}
    fmt_count = {f: 0 for f in V3_TRAIN_FORMATS}; n = 0
    while n < a.pairs:
        d = DOMS[n % len(DOMS)]; f1, f2 = rng.sample(V3_TRAIN_FORMATS, 2); st = rng.getstate(); pair = []
        for f in (f1, f2):
            r = random.Random(); r.setstate(st); rec, _ = DOMAINS[d](r, f, f"g3-{n}-{f}"); pair.append(rec)
        rng.setstate(r.getstate()); rng.random()  # advance past this world
        if pair[0]["labels"] != pair[1]["labels"] or pair[0]["questions"] != pair[1]["questions"]: dropped["mismatch"] += 1; continue
        if any(p["state"] in ev for p in pair): dropped["eval_collision"] += 1; continue
        for p in pair: p["meta"]["render_pair"] = f"g3-{n}"; fmt_count[p["meta"]["format"]] += 1
        recs += pair; n += 1
    n_worlds = len(recs)
    for r in read_jsonl("data/processed/spatial_only_v1/train.jsonl"):
        if r["state"] in ev: dropped["eval_collision"] += 1; continue
        r.setdefault("meta", {})["domain"] = "spatial"; recs.append(r)
    random.Random(a.seed + 1).shuffle(recs)
    with open(out / "train.jsonl", "w", encoding="utf-8") as fh:
        for r in recs: fh.write(json.dumps(r) + "\n")
    man = {"records": len(recs), "world_records": n_worlds, "pairs": a.pairs, "spatial_records": len(recs) - n_worlds, "seed": a.seed,
           "formats": fmt_count, "dropped": dropped, "never_rendered": ["table (pipe)", "symbolic (P3 held-out)"]}
    (out / "manifest.json").write_text(json.dumps(man, indent=2)); print(json.dumps(man))


if __name__ == "__main__":
    main()
