"""Exit non-zero if any training state occurs in a registered evaluation file (eval_registry.py), or if any training
record uses a P3 held-out representation format (meta.format in P3_HELDOUT_FORMATS, or a state in symbolic notation)."""
import argparse, json, re, sys
from collections import Counter
from systemone_lab.eval_registry import eval_states
from systemone_lab.worlds import P3_HELDOUT_FORMATS, SYMBOLIC_RE

ap = argparse.ArgumentParser(); ap.add_argument("train", nargs="+"); a = ap.parse_args()
ev = eval_states(); bad = Counter(); n = 0; p3 = 0; sym = re.compile(SYMBOLIC_RE)
for path in a.train:
    for line in open(path, encoding="utf-8"):
        n += 1; r = json.loads(line); s = r["state"]
        for f in ev.get(s, ()): bad[f] += 1
        if r.get("meta", {}).get("format") in P3_HELDOUT_FORMATS or sym.search(s): p3 += 1
print(json.dumps({"train_records": n, "eval_states": len(ev), "collisions_by_eval_file": dict(bad), "p3_heldout_format_records": p3}))
sys.exit(1 if bad or p3 else 0)
