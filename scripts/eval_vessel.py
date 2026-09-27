"""VESSEL-1 evaluation: a checkpoint on the Channel Watch vessel-behaviour eval sets, against the baselines.

Data contract (produced by the channel-watch repo, Phase 2; copied to data/processed/vessel_v1/):
  eval_synthetic.jsonl  held-out synthetic tracks (fresh simulator seeds), Nevets records with one or more typed
                        questions (e.g. activity, went_dark_suspicious, collision_risk) and meta.domain = "vessel"
  counterfactual.jsonl  pairs sharing meta.pair_id with meta.variant base/counterfactual and meta.cf_question
                        (the question whose answer must flip or stay, as the generator decided)
  baselines.json        {"rules": {...}, "lgbm": {...}}, each with accuracy, ece15, brier, cf_both, computed by
                        channel-watch on these exact files (rules give one-hot probabilities)
Pre-registered WIN (ledger VESSEL-1): (accuracy > both baselines OR ECE15 < both baselines) AND counterfactual
both-correct >= both baselines. Pooled over every question in eval_synthetic. One seed first.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from systemone_lab.data.io import read_jsonl
from systemone_lab.eval import ece
from systemone_lab.gates import predict_batch
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint, pick_device


def norm(y):
    return str(y).lower() if isinstance(y, bool) else y


def score(rows):
    n = len(rows)
    return {"n": n, "accuracy": sum(r["ok"] for r in rows) / n, "ece15": ece([r["conf"] for r in rows], [int(r["ok"]) for r in rows], 15),
            "brier": sum(sum((v - (k == r["y"])) ** 2 for k, v in r["p"].items()) for r in rows) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dir", default="data/processed/vessel_v1"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = pick_device(); d = Path(a.dir)
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    ev = list(read_jsonl(d / "eval_synthetic.jsonl")); rows, byq = [], defaultdict(list)
    for r, p in zip(ev, predict_batch(model, tok, ev, device)):
        for q, dist in p.items():
            y = norm(r["labels"][q]); pred = max(dist, key=dist.get)
            row = {"p": dist, "y": y, "ok": pred == y, "conf": dist[pred]}; rows.append(row); byq[q].append(row)
    out = {"ckpt": a.ckpt, "overall": score(rows), "by_question": {q: score(v) for q, v in sorted(byq.items())}}
    cf = list(read_jsonl(d / "counterfactual.jsonl")); pairs = defaultdict(dict)
    for r, p in zip(cf, predict_batch(model, tok, cf, device)):
        q = r["meta"]["cf_question"]; dist = p[q]; pred = max(dist, key=dist.get)
        pairs[r["meta"]["pair_id"]][r["meta"]["variant"]] = pred == norm(r["labels"][q])
    both = [z["base"] and z["counterfactual"] for z in pairs.values() if len(z) == 2]
    out["cf_both"] = sum(both) / max(1, len(both)); out["cf_pairs"] = len(both)
    base = json.loads((d / "baselines.json").read_text())
    m = {"accuracy": out["overall"]["accuracy"], "ece15": out["overall"]["ece15"], "cf_both": out["cf_both"]}
    acc_win = all(m["accuracy"] > b["accuracy"] for b in base.values())
    ece_win = all(m["ece15"] < b["ece15"] for b in base.values())
    cf_ok = all(m["cf_both"] >= b["cf_both"] for b in base.values())
    out["baselines"] = base; out["verdict"] = {"accuracy_beats_both": acc_win, "ece_beats_both": ece_win, "cf_no_worse": cf_ok,
                                               "WIN": (acc_win or ece_win) and cf_ok}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"nevets": m, **{k: {x: v.get(x) for x in ("accuracy", "ece15", "cf_both")} for k, v in base.items()}, "verdict": out["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
