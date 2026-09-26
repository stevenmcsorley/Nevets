"""PT-2 report: downstream-probe metrics per LM checkpoint and their slope per 250M tokens (owner: judge 35M vs
150M on the slope across checkpoints, not the 1B endpoint). Slope = least squares over the 250M..1B points;
the 0-token (random-init) point is shown for reference. Also val loss per checkpoint and the legacy LM probe."""
import glob
import json
import re
from pathlib import Path

REP = Path("reports/pt/pt2")
METRICS = ["chains", "stress", "in_format", "table", "cf_both", "ece_in"]


def probe(name):
    g, w = REP / f"probe_{name}_gates.json", REP / f"probe_{name}_worlds.json"
    if not (g.exists() and w.exists()): return None
    g, w = json.loads(g.read_text()), json.loads(w.read_text()); cf = w["counterfactual"]
    return {"chains": g["hops"]["overall"]["accuracy"], "stress": g["stress"]["overall"]["accuracy"],
            "in_format": w["eval_in_format"]["overall"]["accuracy"], "table": w["eval_heldout_table"]["overall"]["accuracy"],
            "cf_both": sum(v["both_correct"] for v in cf.values()) / len(cf), "ece_in": w["eval_in_format"]["overall"]["ece15"]}


def slope(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n; den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def main():
    out = {}
    for size in ("35m", "150m"):
        val = {}
        for line in open(f"checkpoints/pt2/pt_{size}/train_log.jsonl") if Path(f"checkpoints/pt2/pt_{size}/train_log.jsonl").exists() else []:
            r = json.loads(line)
            if "val_loss" in r: val[int(round(r["tokens"] / 1e6))] = r["val_loss"]
        pts = {}
        for f in glob.glob(str(REP / f"probe_pt_{size}_tok_*M_gates.json")):
            m = int(re.search(r"tok_(\d+)M", f).group(1)); p = probe(f"pt_{size}_tok_{m}M")
            if p: pts[m] = {**p, "val_loss": val.get(m)}
        xs = sorted(k for k in pts if k > 0)
        sl = {k: slope([x / 250 for x in xs], [pts[x][k] for x in xs]) for k in METRICS} if len(xs) >= 2 else {}
        out[size] = {"points": {k: pts[k] for k in sorted(pts)}, "slope_per_250M": sl}
        print(f"== pt_{size}")
        for k in sorted(pts):
            print(f"  {k:5d}M  val {pts[k]['val_loss'] if pts[k]['val_loss'] is None else round(pts[k]['val_loss'], 3)}  "
                  + "  ".join(f"{m} {pts[k][m]:.3f}" for m in METRICS))
        print("  slope/250M " + "  ".join(f"{m} {v:+.4f}" for m, v in sl.items() if v is not None))
    leg = probe("legacy_s1_35m"); out["legacy_s1_35m"] = leg
    if leg: print("legacy s1-35m-pretrain (same probe): " + "  ".join(f"{m} {leg[m]:.3f}" for m in METRICS))
    (REP / "report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
