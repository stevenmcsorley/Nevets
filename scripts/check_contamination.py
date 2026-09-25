"""Exit non-zero if any training state occurs in a registered evaluation file (eval_registry.py)."""
import argparse, json, sys
from collections import Counter
from systemone_lab.eval_registry import eval_states

ap = argparse.ArgumentParser(); ap.add_argument("train", nargs="+"); a = ap.parse_args()
ev = eval_states(); bad = Counter(); n = 0
for path in a.train:
    for line in open(path, encoding="utf-8"):
        n += 1; s = json.loads(line)["state"]
        for f in ev.get(s, ()): bad[f] += 1
print(json.dumps({"train_records": n, "eval_states": len(ev), "collisions_by_eval_file": dict(bad)}))
sys.exit(1 if bad else 0)
