"""Read-only per-suite evaluation: accuracy, ECE15, Brier and role-swap both-correct."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from systemone_lab.model import SystemOneModel
from systemone_lab.training import load_checkpoint, pick_device
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.data.io import read_jsonl
from systemone_lab.eval import predict_record, ece


def summarize(rows):
    pairs = defaultdict(list)
    for r in rows: pairs[r["pair_id"]].append(r["correct"])
    return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "ece15": ece([r["confidence"] for r in rows], [int(r["correct"]) for r in rows], 15),
            "brier": sum(r["brier"] for r in rows) / len(rows),
            "role_swap_both_correct": sum(all(v) for v in pairs.values()) / len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    rows = []
    for rec in read_jsonl(a.data):
        p = predict_record(model, tok, rec, device)["spatial"]; y = rec["labels"]["spatial"]; pred = max(p, key=p.get)
        rows.append({"id": rec["id"], **{k: rec["meta"][k] for k in ("suite", "pair_id", "names", "query_role")},
                     "label": y, "prediction": pred, "confidence": p[pred], "correct": pred == y,
                     "brier": sum((v - (k == y)) ** 2 for k, v in p.items())})
    groups = defaultdict(list)
    for r in rows:
        groups[r["suite"]].append(r); groups[f"{r['suite']}/{r['names']}"].append(r)
    out = {"ckpt": a.ckpt, "overall": summarize(rows), "suites": {k: summarize(v) for k, v in sorted(groups.items())},
           "results": rows}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"overall": out["overall"], "suites": {k: {m: round(x, 3) for m, x in v.items()}
                                                             for k, v in out["suites"].items() if "/" not in k}}, indent=1))


if __name__ == "__main__":
    main()
