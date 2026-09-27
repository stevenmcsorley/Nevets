"""GENERAL-3 LOSS diagnosis (owner, 27 Sep): are infogather and probability exposure-limited?

No intermediate checkpoints were kept (each save overwrote the last), so learning curves come from the training
diagnostics: every 10th update logs the batch loss and the 16 record ids. A batch's mean loss is the mix of its
domains' losses, loss_i = sum_d f_id * L_d, so least squares over windows of logged batches estimates each domain's
training loss L_d(t). Exposure = samples of that domain drawn so far (logged counts x 10).
Caveat: GENERAL-3's logged loss includes label smoothing (eps 0.05) and the pair-consistency term, which raise its
floor, so loss LEVELS compare only within a run; the cross-run comparison uses endpoint eval accuracy vs exposure.

Reading rule (fixed before looking at the plots): a domain is EXPOSURE-LIMITED if (a) its training loss is still falling
over the last quarter of its exposure in every run (slope of loss vs log exposure < 0 by more than twice its standard
error), and (b) its endpoint eval accuracy rises with total exposure across the five runs (Spearman rho >= 0.8).
Outputs: reports/general/exposure/{curves.json, curves.png, endpoint.png, verdict.txt}.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = {  # name: (diagnostics, training data, eval json)
    "G1": ("checkpoints/general/general-v1.diagnostics.jsonl", "data/processed/general_v1/train.jsonl", "reports/general/v1/worlds_v2.json"),
    "G2 s7": ("checkpoints/general/general-v2_s7.diagnostics.jsonl", "data/processed/general_v2/train.jsonl", "reports/general/v2/worlds_v2_s7.json"),
    "G2 s8": ("checkpoints/general/general-v2_s8.diagnostics.jsonl", "data/processed/general_v2/train.jsonl", "reports/general/v2/worlds_v2_s8.json"),
    "G3 s7": ("checkpoints/general/general-v3_s7.diagnostics.jsonl", "data/processed/general_v3/train.jsonl", "reports/general/v3/worlds_v2_s7.json"),
    "G3 s8": ("checkpoints/general/general-v3_s8.diagnostics.jsonl", "data/processed/general_v3/train.jsonl", "reports/general/v3/worlds_v2_s8.json"),
}
DOMAINS = ["infogather", "probability", "causal", "dependency", "kinship", "rules", "temporal", "spatial"]
OUT = Path("reports/general/exposure"); WINDOW = 150  # logged batches per window (= 1,500 updates)


def dom_of(r):
    d = (r.get("meta") or {}).get("domain", "none")
    return "spatial" if d.startswith("spatial") else d


def curves(diag, data):
    idmap = {}
    for line in open(data, encoding="utf-8"):
        r = json.loads(line); idmap[r["id"]] = dom_of(r)
    rows = [json.loads(l) for l in open(diag, encoding="utf-8")]
    F = np.zeros((len(rows), len(DOMAINS))); y = np.zeros(len(rows)); steps = np.zeros(len(rows)); missing = 0
    for i, r in enumerate(rows):
        for bid in r["batch_ids"]:
            d = idmap.get(bid)
            if d in DOMAINS: F[i, DOMAINS.index(d)] += 1
            else: missing += 1
        n = F[i].sum(); F[i] /= max(n, 1); y[i] = r["loss"]; steps[i] = r["step"]
    counts = F * 16  # per logged batch
    exposure = np.cumsum(counts, axis=0) * 10  # every 10th update is logged
    out = {d: {"step": [], "exposure": [], "loss": []} for d in DOMAINS}
    for s in range(0, len(rows) - WINDOW + 1, WINDOW // 2):
        sl = slice(s, s + WINDOW); A = F[sl]; present = A.sum(0) > 0
        coef, *_ = np.linalg.lstsq(A[:, present], y[sl], rcond=None)
        full = np.full(len(DOMAINS), np.nan); full[present] = coef
        for j, d in enumerate(DOMAINS):
            if present[j]:
                out[d]["step"].append(float(steps[sl].mean())); out[d]["exposure"].append(float(exposure[s + WINDOW - 1, j])); out[d]["loss"].append(float(full[j]))
    totals = {d: float(exposure[-1, j]) for j, d in enumerate(DOMAINS)}
    return out, totals, missing


def late_slope(ex, lo):
    """Slope of loss vs log(exposure) over the last quarter of exposure, with its standard error."""
    ex, lo = np.asarray(ex), np.asarray(lo); m = ex >= ex.max() * 0.75
    if m.sum() < 4: return None, None
    x = np.log(ex[m]); X = np.vstack([x, np.ones_like(x)]).T
    beta, res, *_ = np.linalg.lstsq(X, lo[m], rcond=None)
    sigma2 = (res[0] / (m.sum() - 2)) if len(res) else np.var(lo[m] - X @ beta)
    se = np.sqrt(sigma2 / ((x - x.mean()) ** 2).sum())
    return float(beta[0]), float(se)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True); res = {}
    for name, (diag, data, ev) in RUNS.items():
        c, tot, missing = curves(diag, data)
        acc = json.loads(Path(ev).read_text())["eval_in_format"]
        res[name] = {"curves": c, "exposure_total": tot, "unmapped_ids": missing,
                     "endpoint_acc": {d: acc[d]["accuracy"] for d in DOMAINS if d in acc}}
    (OUT / "curves.json").write_text(json.dumps(res))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8)); colors = {"G1": "#444", "G2 s7": "#d9542b", "G2 s8": "#f08a5d", "G3 s7": "#2f6fd6", "G3 s8": "#7fa8ea"}
    for ax, d in zip(axes.flat, DOMAINS):
        for name, r in res.items():
            c = r["curves"][d]
            if c["exposure"]: ax.plot(c["exposure"], c["loss"], color=colors[name], label=name, lw=1.4)
        ax.set_xscale("log"); ax.set_title(d); ax.set_xlabel("samples of this domain seen"); ax.set_ylabel("est. training loss"); ax.grid(alpha=.3)
    axes.flat[0].legend(); fig.suptitle("Per-domain training loss vs sampled exposure (decomposed from batch losses)"); fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=110)
    fig, axes = plt.subplots(1, 7, figsize=(22, 3.6))
    for ax, d in zip(axes, DOMAINS[:7]):
        xs = [res[n]["exposure_total"][d] for n in res]; ys = [res[n]["endpoint_acc"][d] for n in res]
        for n, x, yv in zip(res, xs, ys): ax.scatter(x, yv, color=colors[n]); ax.annotate(n, (x, yv), fontsize=7)
        ax.set_xscale("log"); ax.set_title(d); ax.set_xlabel("total samples"); ax.set_ylabel("eval accuracy (in-format)"); ax.grid(alpha=.3)
    fig.suptitle("Endpoint accuracy vs total sampled exposure"); fig.tight_layout(); fig.savefig(OUT / "endpoint.png", dpi=110)
    lines = []
    for d in DOMAINS[:7]:
        slopes = {n: late_slope(res[n]["curves"][d]["exposure"], res[n]["curves"][d]["loss"]) for n in res}
        still = all(s is not None and s < 0 and abs(s) > 2 * se for s, se in slopes.values())
        rho = spearman([res[n]["exposure_total"][d] for n in res], [res[n]["endpoint_acc"][d] for n in res])
        verdict = "EXPOSURE-LIMITED" if still and rho >= 0.8 else "not exposure-limited" if not still and rho < 0.8 else "mixed"
        lines.append(f"{d:12s} late-slope falling in every run: {still}  | Spearman(exposure, accuracy) = {rho:+.2f}  -> {verdict}")
        lines.append("             " + "  ".join(f"{n}: {res[n]['exposure_total'][d] / 1000:.0f}k samples, acc {res[n]['endpoint_acc'][d]:.3f}, slope "
                                                 f"{'n/a' if slopes[n][0] is None else f'{slopes[n][0]:+.3f}±{slopes[n][1]:.3f}'}" for n in res))
    (OUT / "verdict.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    print("unmapped ids per run:", {n: r["unmapped_ids"] for n, r in res.items()})


if __name__ == "__main__":
    main()
