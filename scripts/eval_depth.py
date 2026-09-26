"""Depth and test-time-compute evaluation (P1 key test), batched.

For each number of loop passes K: per-hop accuracy on the dev chain suite (1-10 hops), accuracy on
cancellation labels (a zero on some axis: left/right/above/below/overlap) versus diagonal labels,
calibration (NLL, Brier, ECE15, risk/coverage at 0.9/0.8), and rotation/reflection consistency on the
dev transform suite. Non-looped checkpoints are evaluated once.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from systemone_lab.data.io import read_jsonl
from systemone_lab.data.spatial_worlds import REL, sign, transform_point
from systemone_lab.eval import ece
from systemone_lab.gates import predict_batch
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint, pick_device

CANCEL = {"left", "right", "above", "below", "overlap"}


def calib(rows):
    n = len(rows); out = {"n": n, "accuracy": sum(r["ok"] for r in rows) / n,
                          "nll": -sum(math.log(max(r["p"][r["y"]], 1e-9)) for r in rows) / n,
                          "brier": sum(sum((v - (k == r["y"])) ** 2 for k, v in r["p"].items()) for r in rows) / n,
                          "ece15": ece([r["conf"] for r in rows], [int(r["ok"]) for r in rows], 15)}
    for c in (0.9, 0.8):
        sure = [r for r in rows if r["conf"] >= c]
        out[f"coverage@{c}"] = len(sure) / n; out[f"accuracy@{c}"] = sum(r["ok"] for r in sure) / len(sure) if sure else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", default="1,2,4,6,8,12")
    ap.add_argument("--chains", default="reports/chain/eval_hops.jsonl"); ap.add_argument("--transforms", default="reports/chain/transforms.jsonl")
    a = ap.parse_args(); device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    looped = (model.cfg.arch or {}).get("type") == "looped"
    chains = list(read_jsonl(a.chains)); tfs = list(read_jsonl(a.transforms))
    vec = {label: xy for xy, label in REL.items()}
    moved = lambda label, t: REL[tuple(sign(v) for v in transform_point(vec[label], t))]
    out = {"ckpt": a.ckpt, "looped": looped, "by_iters": {}}
    for K in ([int(k) for k in a.iters.split(",")] if looped else [None]):
        if K: model.iters = K
        rows = []
        for r, p in zip(chains, predict_batch(model, tok, chains, device)):
            p = p["spatial"]; pred = max(p, key=p.get); y = r["labels"]["spatial"]
            rows.append({"hops": r["meta"]["hops"], "p": p, "y": y, "ok": pred == y, "conf": p[pred], "cancel": y in CANCEL})
        by = defaultdict(list)
        for x in rows: by[x["hops"]].append(x)
        preds = predict_batch(model, tok, tfs, device); worlds = defaultdict(dict)
        for r, p in zip(tfs, preds):
            p = p["spatial"]; worlds[r["meta"]["world_id"]][r["meta"]["transform"]] = max(p, key=p.get)
        pairs = [(w["identity"], t, pr) for w in worlds.values() for t, pr in w.items() if t != "identity"]
        out["by_iters"][str(K)] = {
            "overall": calib(rows), "by_hops": {h: calib(v)["accuracy"] for h, v in sorted(by.items())},
            "cancellation_labels": calib([x for x in rows if x["cancel"]]), "diagonal_labels": calib([x for x in rows if not x["cancel"]]),
            "transform_consistency": sum(pr == moved(base, t) for base, t, pr in pairs) / len(pairs)}
        s = out["by_iters"][str(K)]
        print(f"K={K}: overall {s['overall']['accuracy']:.3f} | hops " + " ".join(f"{h}:{v:.2f}" for h, v in s["by_hops"].items())
              + f" | cancel {s['cancellation_labels']['accuracy']:.3f} diag {s['diagonal_labels']['accuracy']:.3f}"
              + f" | ECE {s['overall']['ece15']:.3f} NLL {s['overall']['nll']:.3f} | rot {s['transform_consistency']:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
