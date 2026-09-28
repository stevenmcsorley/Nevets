"""P2 dataset (force vs observe): data/processed/p2_scm_v1/{train, eval_iid, eval_heldout_wording, eval_heldout_structure}.jsonl.

Seeds are fixed per split; the three eval files are registered as locked in eval_registry (never trained on). Formats:
prose / json / kv (P3's held-out formats are never used). See src/systemone_lab/scm.py for the splits.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from systemone_lab import scm

OUT = Path("data/processed/p2_scm_v1")
SPLITS = {"train": (20000, 5001), "eval_iid": (1000, 5002), "eval_heldout_wording": (1000, 5003), "eval_heldout_structure": (1000, 5004)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scale", type=float, default=1.0); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); man = {}
    for split, (n, seed) in SPLITS.items():
        recs = scm.generate(max(1, int(n * a.scale)), seed, split)
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        man[split] = {"records": len(recs), "pairs": len(recs) // 2, "seed": seed,
                      "structures": dict(Counter(r["meta"]["structure"] for r in recs)),
                      "discriminating_pairs": sum(r["meta"]["discriminating"] for r in recs) // 2,
                      "yes_share": round(sum(r["labels"]["answer"] for r in recs) / len(recs), 3)}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1)); print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
