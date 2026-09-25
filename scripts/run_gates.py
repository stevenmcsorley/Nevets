"""Run every locked development gate for one checkpoint in a single process.

Metrics match eval_role_swap.py, eval_name_pairs.py, eval_counterfactual.py, eval_spatial.py and
eval_stress.py. Pass extra suites with --stress name=path (records need meta.suite/pair_id).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from systemone_lab.model import SystemOneModel
from systemone_lab.training import load_checkpoint, pick_device
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.data.io import read_jsonl
from systemone_lab.data.spatial_worlds import INVERSE_REL, REL, sign, transform_point
from systemone_lab.eval import predict_record, ece

QUESTION = r"What is the spatial relation of (.+) to (.+)\?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stress", action="append", default=["stress=reports/binding_stress/stress.jsonl"])
    ap.add_argument("--transforms", help="suite from generate_transform_suite.py")
    ap.add_argument("--iters", type=int, help="looped arch: override core iterations (test-time depth)")
    ap.add_argument("--skip-onehop", action="store_true", help="only run the --stress/--transforms suites")
    a = ap.parse_args()
    device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval()
    if a.iters: model.iters = a.iters
    tok = LabTokenizer(ck["tokenizer"])

    def predict(rec):
        p = predict_record(model, tok, rec, device)["spatial"]; pred = max(p, key=p.get)
        return p, pred

    def role_swap(path):
        both = 0; n = 0
        for pair in json.loads(Path(path).read_text()):
            for variant in ("numbered", "renamed"):
                r = pair[variant]; y = r["labels"]["spatial"]
                x1, x2 = re.fullmatch(QUESTION, r["questions"]["spatial"]["instructions"]).groups()
                s = json.loads(json.dumps(r)); s["questions"]["spatial"]["instructions"] = f"What is the spatial relation of {x2} to {x1}?"
                both += predict(r)[1] == y and predict(s)[1] == INVERSE_REL[y]; n += 1
        return both / n

    def name_pairs(path):
        pairs = json.loads(Path(path).read_text())
        return sum(predict(p["numbered"])[1] == predict(p["renamed"])[1] == p["label"] for p in pairs) / len(pairs)

    def shuffle_changed(path, limit=400):
        """Permanent regression gate: does a fixed-seed shuffle of the option order change the answer?"""
        import random
        rng = random.Random(0); changed = n = 0
        for i, rec in enumerate(read_jsonl(path)):
            if i % 5 or n >= limit: continue
            s = json.loads(json.dumps(rec)); q = s["questions"]["spatial"]; items = list(q["criteria"].items())
            rng.shuffle(items); q["criteria"] = dict(items)
            changed += predict(rec)[1] != predict(s)[1]; n += 1
        return changed / n if n else None

    def scored(path):
        rows = []
        for rec in read_jsonl(path):
            p, pred = predict(rec); y = rec["labels"]["spatial"]
            rows.append({"meta": rec.get("meta", {}), "p": p, "y": y, "pred": pred, "conf": p[pred], "ok": pred == y})
        return rows

    def calib(rows):
        sure = [r for r in rows if r["conf"] >= 0.9]
        return {"n": len(rows), "accuracy": sum(r["ok"] for r in rows) / len(rows),
                "coverage_at_0.9": len(sure) / len(rows),
                "accuracy_at_0.9": sum(r["ok"] for r in sure) / len(sure) if sure else None,
                "ece15": ece([r["conf"] for r in rows], [int(r["ok"]) for r in rows], 15),
                "brier": sum(sum((v - (k == r["y"])) ** 2 for k, v in r["p"].items()) for r in rows) / len(rows)}

    out = {"ckpt": a.ckpt, "iters": getattr(model, "iters", None)}
    if not a.skip_onehop:
        out.update({"vertical_role_swap": role_swap("reports/paired_onehop/vertical_probe_pairs.json"),
                    "heldout_role_swap": role_swap("reports/role_swap_onehop/name_pairs.json"),
                    "name_pairs": name_pairs("reports/paired_onehop/name_pairs.json"),
                    "vertical_name_pairs": name_pairs("reports/paired_onehop/vertical_probe_pairs.json")})
        cf = defaultdict(dict)
        for r in scored("reports/paired_onehop/counterfactual.jsonl"): cf[r["meta"]["pair_id"]][r["meta"]["variant"]] = r
        cf = [z for z in cf.values() if "base" in z and "counterfactual" in z]
        out["counterfactual_both"] = sum(z["base"]["ok"] and z["counterfactual"]["ok"] for z in cf) / len(cf)
        out["counterfactual_shift"] = sum(z["counterfactual"]["p"][z["counterfactual"]["y"]] > z["base"]["p"][z["counterfactual"]["y"]]
                                          and z["counterfactual"]["p"][z["base"]["y"]] < z["base"]["p"][z["base"]["y"]] for z in cf) / len(cf)
        out["heldout"] = calib(scored("reports/role_swap_onehop/heldout.jsonl"))
    for spec in a.stress:
        name, path = spec.split("=", 1); rows = scored(path); suites = defaultdict(list)
        for r in rows: suites[r["meta"].get("suite", "all")].append(r)
        res = {"overall": calib(rows)}
        for s, rs in sorted(suites.items()):
            pairs = defaultdict(list)
            for r in rs: pairs[r["meta"].get("pair_id", id(r))].append(r["ok"])
            res[s] = {**calib(rs), "pair_both_correct": sum(all(v) for v in pairs.values()) / len(pairs)}
        res["option_shuffle_changed"] = shuffle_changed(path)
        out[name] = res
    if a.transforms:
        vec = {label: xy for xy, label in REL.items()}
        def moved(label, t):
            x, y = transform_point(vec[label], t); return REL[(sign(x), sign(y))]
        worlds = defaultdict(dict)
        for r in scored(a.transforms): worlds[r["meta"]["world_id"]][r["meta"]["transform"]] = r
        consistent = both = n = 0
        for w in worlds.values():
            base = w["identity"]
            for t, r in w.items():
                if t == "identity": continue
                n += 1; consistent += r["pred"] == moved(base["pred"], t); both += r["ok"] and base["ok"]
        out["transform_consistency"] = consistent / n; out["transform_both_correct"] = both / n
        out["transform_accuracy"] = calib([r for w in worlds.values() for r in w.values()])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    flat = {k: v for k, v in out.items() if isinstance(v, float)}
    if "heldout" in out: flat.update({"heldout_acc": out["heldout"]["accuracy"], "heldout_ece": out["heldout"]["ece15"]})
    for spec in a.stress:
        name = spec.split("=", 1)[0]
        flat.update({f"{name}/{s}": v["accuracy"] for s, v in out[name].items() if isinstance(v, dict)})
        flat[f"{name}/ece"] = out[name]["overall"]["ece15"]
        flat[f"{name}/option_shuffle_changed"] = out[name]["option_shuffle_changed"]
    print("\n".join(f"{k:40s} {v:.3f}" for k, v in flat.items()))


if __name__ == "__main__":
    main()
