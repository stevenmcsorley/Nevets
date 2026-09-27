"""VESSEL-1c data: freeze the live Claude-labelled set at launch and build the training mix.

  data/processed/vessel_c/train.jsonl     vessel_v1b mix (synthetic vessel + general_v2 worlds/spatial) + the live
                                          train split, relabelled meta.domain = "vessel_real" so it gets its own share
  data/processed/vessel_c/eval_real.jsonl frozen copy of vessel_v1/eval_real.jsonl at launch (locked; eval only)
  data/processed/vessel_c/baselines_real.json  frozen LightGBM-real figures on that same file (from channel-watch)
  data/processed/vessel_c/manifest.json   sizes, hashes and the step budget

Prints the step budget: min(12000, max(3000, 6 x live train records)), so the small live set is repeated at most
~30 times at a 0.3 share and batch 16.
"""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

P = Path("data/processed"); OUT = P / "vessel_c"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    live = [json.loads(l) for l in open(P / "vessel_real_v1/train.jsonl", encoding="utf-8") if l.strip()]
    for r in live: r["meta"]["domain"] = "vessel_real"
    shutil.copy(P / "vessel_v1/eval_real.jsonl", OUT / "eval_real.jsonl")
    shutil.copy(P / "vessel_v1/baselines_real.json", OUT / "baselines_real.json")
    by = Counter()
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
        for l in open(P / "vessel_v1b/train.jsonl", encoding="utf-8"):
            if l.strip(): f.write(l if l.endswith("\n") else l + "\n"); by[json.loads(l)["meta"]["domain"]] += 1
        for r in live: f.write(json.dumps(r) + "\n"); by["vessel_real"] += 1
    steps = min(12000, max(3000, 6 * len(live)))
    ev = sum(1 for l in open(OUT / "eval_real.jsonl", encoding="utf-8") if l.strip())
    man = {"by_domain": dict(by), "live_train": len(live), "eval_real": ev, "steps": steps,
           "hashes": {"eval_real": sha(OUT / "eval_real.jsonl"), "baselines_real": sha(OUT / "baselines_real.json"), "live_train": sha(P / "vessel_real_v1/train.jsonl")},
           "sampling": "--domain-shares vessel_real=0.3,vessel=0.2,spatial=0.2 (7 world domains split 0.3)"}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(steps)


if __name__ == "__main__":
    main()
