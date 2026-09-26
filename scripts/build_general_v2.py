"""GENERAL-2 training data: worlds_v2 train (7 domains, table held out) + spatial_only_v1 (S30's own data,
tagged meta.domain = "spatial"). Sampling shares are set at train time (--domain-shares), not by copying."""
import json
import random
from pathlib import Path

from systemone_lab.data.io import read_jsonl
from systemone_lab.eval_registry import eval_states

OUT = Path("data/processed/general_v2"); OUT.mkdir(parents=True, exist_ok=True)
recs = list(read_jsonl("data/processed/worlds_v2/train.jsonl")); n_worlds = len(recs)
for r in read_jsonl("data/processed/spatial_only_v1/train.jsonl"):
    r.setdefault("meta", {})["domain"] = "spatial"; recs.append(r)
ev = eval_states(); n_all = len(recs)
recs = [r for r in recs if r["state"] not in ev]  # Rule 2: filter against every registered eval state
random.Random(20260927).shuffle(recs)
with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
(OUT / "manifest.json").write_text(json.dumps({"records": len(recs), "sources": {"worlds_v2/train": n_worlds,
    "spatial_only_v1/train (meta.domain=spatial)": len(recs) - n_worlds}, "dropped_eval_collisions": n_all - len(recs), "shuffle_seed": 20260927,
    "sampling": "--domain-shares spatial=0.5 (domain first, then label within domain)"}, indent=2))
print(len(recs))
