"""P3 held-out-format evaluation set: every world domain rendered in the "symbolic" postfix operator notation
(data/processed/p3_v1/eval_symbolic.jsonl), plus the same worlds in prose (eval_prose_twin.jsonl) so format robustness is
measured on identical content. Both files are registered as locked; the symbolic format is rejected for training
(check_contamination.py)."""
import json
import zlib
import random
from collections import Counter
from pathlib import Path

from systemone_lab.worlds import DOMAINS

OUT = Path("data/processed/p3_v1"); N_PER_DOMAIN = 250; SEED = 6001


def main():
    OUT.mkdir(parents=True, exist_ok=True); sym, prose = [], []
    for d, fn in DOMAINS.items():
        for i in range(N_PER_DOMAIN):
            s = SEED * 1000 + zlib.crc32(d.encode()) % 997 * 10000 + i
            r1, _ = fn(random.Random(s), "symbolic", f"p3-{d}-{i}-sym"); r2, _ = fn(random.Random(s), "prose", f"p3-{d}-{i}-prose")
            if r1["labels"] != r2["labels"] or r1["questions"] != r2["questions"]: continue  # same world, same questions
            r1["meta"]["twin"] = r2["id"]; r2["meta"]["twin"] = r1["id"]; sym.append(r1); prose.append(r2)
    for name, recs in (("eval_symbolic", sym), ("eval_prose_twin", prose)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    man = {"records_per_file": len(sym), "by_domain": dict(Counter(r["meta"]["domain"] for r in sym)), "seed": SEED}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1)); print(json.dumps(man))


if __name__ == "__main__":
    main()
