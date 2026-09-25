"""Fail loudly if any evaluation chess position (placement, side, castling, en passant) appears in training."""
import argparse, json, sys
key = lambda fen: " ".join(fen.split()[:4])
ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--eval", action="append", required=True)
a = ap.parse_args()
ev = {key(json.loads(l)["meta"]["fen"]) for path in a.eval for l in open(path, encoding="utf-8")}
hits = sum(key(json.loads(l)["meta"]["fen"]) in ev for l in open(a.train, encoding="utf-8"))
print(json.dumps({"train": a.train, "eval_positions": len(ev), "overlap": hits}))
sys.exit(1 if hits else 0)
