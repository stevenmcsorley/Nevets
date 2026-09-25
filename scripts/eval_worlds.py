"""Evaluate a checkpoint on a data-factory split.

Reports accuracy per domain and format (in-format vs held-out format), calibration (ECE15, Brier,
selective accuracy at 0.9), KL to the exact Bayesian posterior on the probability domain, and
counterfactual both-correct / correct probability direction on twin pairs.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from systemone_lab.data.io import read_jsonl
from systemone_lab.eval import ece, predict_record
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint, pick_device


def score(model, tok, device, rec):
    p = predict_record(model, tok, rec, device)["answer"]
    y = rec["labels"]["answer"]; y = str(y).lower() if isinstance(y, bool) else y
    pred = max(p, key=p.get)
    return {"p": p, "y": y, "pred": pred, "ok": pred == y, "conf": p[pred], "meta": rec["meta"],
            "exact": rec.get("distributions", {}).get("answer")}


def summary(rows):
    n = len(rows); sure = [r for r in rows if r["conf"] >= 0.9]
    out = {"n": n, "accuracy": sum(r["ok"] for r in rows) / n, "ece15": ece([r["conf"] for r in rows], [int(r["ok"]) for r in rows], 15),
           "brier": sum(sum((v - (k == r["y"])) ** 2 for k, v in r["p"].items()) for r in rows) / n,
           "coverage_at_0.9": len(sure) / n, "accuracy_at_0.9": sum(r["ok"] for r in sure) / len(sure) if sure else None}
    exact = [r for r in rows if r["exact"]]
    if exact:
        kl = [sum(q * math.log(q / max(r["p"][k], 1e-9)) for k, q in r["exact"].items() if q > 0) for r in exact]
        out["kl_to_exact_posterior"] = sum(kl) / len(kl)
        out["mean_abs_prob_error"] = sum(abs(r["p"][k] - q) for r in exact for k, q in r["exact"].items()) / sum(len(r["exact"]) for r in exact)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max records per domain per file (0 = all)")
    a = ap.parse_args(); device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    out = {"ckpt": a.ckpt, "split": a.dir}
    for f in sorted(Path(a.dir).glob("eval_*.jsonl")):
        recs, seen = [], defaultdict(int)
        for r in read_jsonl(f):  # --limit applies per domain so every domain is represented
            d = r["meta"]["domain"]
            if not a.limit or seen[d] < a.limit: recs.append(r); seen[d] += 1
        rows = [score(model, tok, device, r) for r in recs]; by = defaultdict(list)
        for r in rows: by[r["meta"]["domain"]].append(r)
        out[f.stem] = {"overall": summary(rows), **{d: summary(v) for d, v in sorted(by.items())}}
    pairs = defaultdict(dict)
    for r in read_jsonl(Path(a.dir) / "counterfactual.jsonl"):
        s = score(model, tok, device, r); pairs[(r["meta"]["domain"], r["meta"]["pair_id"])][r["meta"]["variant"]] = s
    cf = defaultdict(lambda: {"n": 0, "both_correct": 0, "direction": 0})
    for (dom, _), z in pairs.items():
        b, c = z["base"], z["counterfactual"]; t = cf[dom]; t["n"] += 1
        t["both_correct"] += b["ok"] and c["ok"]
        t["direction"] += c["p"][c["y"]] > b["p"][c["y"]] and c["p"][b["y"]] < b["p"][b["y"]]
    out["counterfactual"] = {d: {"pairs": t["n"], "both_correct": t["both_correct"] / t["n"], "correct_direction": t["direction"] / t["n"]}
                             for d, t in cf.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2))
    for split in [k for k in out if k.startswith("eval_")]:
        print(split, {d: round(v["accuracy"], 3) for d, v in out[split].items()})
    print("counterfactual", out["counterfactual"])


if __name__ == "__main__":
    main()
