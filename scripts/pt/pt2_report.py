"""PT-2 report: downstream-probe metrics per LM checkpoint, their slope per 250M tokens, and the noise band.

Owner rules: judge 35M vs 150M on the slope across checkpoints (not the 1B endpoint); the noise band comes from
2-3 fine-tune seeds on the first and last slope checkpoint of each size, and is recorded BEFORE comparing
(`--band-only` prints just the band; the comparison is run after the band is in the ledger).

Slope = least squares over the >0-token checkpoints (fine-tune seed 7 at every point), x in units of 250M tokens.
Noise: for each metric, sigma = pooled standard deviation across fine-tune seeds (seeds 7/8/9; probes named
`<ckpt>` and `<ckpt>_s8`, `<ckpt>_s9`) over the (size, checkpoint) cells that have >= 2 seeds.
One slope's standard error = sigma / sqrt(Sxx), Sxx = sum (x - mean x)^2 over that size's slope points; the difference
of two sizes' slopes has SE = sigma * sqrt(1/Sxx_35M + 1/Sxx_150M) (= sigma*sqrt(2/5) on the 250M..1B grid). Band = 2 x SE: |slope_150M - slope_35M| inside the band = "within noise".
"""
import argparse
import glob
import json
import math
import re
from pathlib import Path

METRICS = ["chains", "stress", "in_format", "table", "cf_both", "ece_in"]
GENERAL = ["in_format", "table", "cf_both"]  # the decision metrics (general domains); ECE is a guard


def probe(rep, name):
    g, w = rep / f"probe_{name}_gates.json", rep / f"probe_{name}_worlds.json"
    if not (g.exists() and w.exists()): return None
    g, w = json.loads(g.read_text()), json.loads(w.read_text()); cf = w["counterfactual"]
    return {"chains": g["hops"]["overall"]["accuracy"], "stress": g["stress"]["overall"]["accuracy"],
            "in_format": w["eval_in_format"]["overall"]["accuracy"], "table": w["eval_heldout_table"]["overall"]["accuracy"],
            "cf_both": sum(v["both_correct"] for v in cf.values()) / len(cf), "ece_in": w["eval_in_format"]["overall"]["ece15"]}


def slope(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n; den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def collect(rep, root, size):
    val = {}; log = Path(root) / f"pt_{size}" / "train_log.jsonl"
    for line in (open(log) if log.exists() else []):
        r = json.loads(line)
        if "val_loss" in r: val[int(round(r["tokens"] / 1e6))] = r["val_loss"]
    pts, seeds = {}, {}
    for f in glob.glob(str(rep / f"probe_pt_{size}_tok_*M_gates.json")):
        m = int(re.search(r"tok_(\d+)M_gates", f).group(1)); p = probe(rep, f"pt_{size}_tok_{m}M")
        if not p: continue
        pts[m] = {**p, "val_loss": val.get(m)}
        extra = [probe(rep, f"pt_{size}_tok_{m}M_s{s}") for s in (8, 9)]
        if any(extra): seeds[m] = [p] + [e for e in extra if e]
    return pts, seeds


def band(all_seeds, grids):
    sxx = {k: sum((x - sum(v) / len(v)) ** 2 for x in v) for k, v in grids.items() if len(v) >= 2}
    inv = sum(1 / v for v in sxx.values()) if len(sxx) == 2 and all(sxx.values()) else None
    out = {"cells": [f"{s}@{m}M(n={len(v)})" for s, cells in all_seeds.items() for m, v in sorted(cells.items())], "sxx": sxx}
    for k in METRICS:
        var, dof = 0.0, 0
        for cells in all_seeds.values():
            for runs in cells.values():
                ys = [r[k] for r in runs]; mu = sum(ys) / len(ys); var += sum((y - mu) ** 2 for y in ys); dof += len(ys) - 1
        sigma = math.sqrt(var / dof) if dof else None
        out[k] = {"sigma_finetune": sigma, "slope_diff_band": 2 * sigma * math.sqrt(inv) if sigma is not None and inv else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="reports/pt/pt2"); ap.add_argument("--ckpt-root", default="checkpoints/pt2")
    ap.add_argument("--band-only", action="store_true", help="print and save only the noise band (record it before comparing)")
    a = ap.parse_args(); rep = Path(a.rep)
    data = {size: collect(rep, a.ckpt_root, size) for size in ("35m", "150m")}
    grids = {size: sorted(m / 250 for m in data[size][0] if m > 0) for size in data}
    b = band({size: data[size][1] for size in data}, grids)
    (rep / "band.json").write_text(json.dumps(b, indent=2))
    print("NOISE BAND (fine-tune seeds; cells " + ", ".join(b["cells"]) + ")")
    for k in METRICS:
        s, d = b[k]["sigma_finetune"], b[k]["slope_diff_band"]
        print(f"  {k:9s} sigma {s if s is None else round(s, 4)}   slope-difference band (per 250M) {d if d is None else round(d, 4)}")
    if a.band_only: return
    out = {"band": b}
    for size, (pts, _) in data.items():
        px = sorted(k for k in pts if k > 0)
        sl = {k: slope([x / 250 for x in px], [pts[x][k] for x in px]) for k in METRICS} if len(px) >= 2 else {}
        out[size] = {"points": {k: pts[k] for k in sorted(pts)}, "slope_per_250M": sl}
        print(f"== pt_{size}")
        for k in sorted(pts):
            vl = pts[k]["val_loss"]
            print(f"  {k:5d}M  val {vl if vl is None else round(vl, 3)}  " + "  ".join(f"{m} {pts[k][m]:.3f}" for m in METRICS))
        print("  slope/250M " + "  ".join(f"{m} {v:+.4f}" for m, v in sl.items() if v is not None))
    s35, s150 = out["35m"]["slope_per_250M"], out["150m"]["slope_per_250M"]
    if s35 and s150:
        verdict = {}
        for k in GENERAL + ["ece_in"]:
            d, bd = s150[k] - s35[k], b[k]["slope_diff_band"]
            better = (d < 0) if k == "ece_in" else (d > 0)
            verdict[k] = "within noise" if bd is None or abs(d) <= bd else ("150M" if better else "35M")
            print(f"  decision {k:9s} slope150-slope35 {d:+.4f} vs band {bd if bd is None else round(bd, 4)} -> {verdict[k]}")
        out["verdict"] = verdict
    leg = probe(rep, "legacy_s1_35m"); out["legacy_s1_35m"] = leg
    if leg: print("legacy s1-35m-pretrain (same probe): " + "  ".join(f"{m} {leg[m]:.3f}" for m in METRICS))
    (rep / "report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
