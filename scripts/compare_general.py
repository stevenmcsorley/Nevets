"""Pre-registered verdicts for S30 promotion and GENERAL-2 (all dev sets; locked sets are never touched here).

  s30: S30 seeds 7 and 8 must both beat the previous BEST_SPATIAL (dev: GENERAL-1) and S6 on dev chains at K=4,
       with neither run killed by the Rule-4 gates. Prints the seed spread; the last line is the chosen init
       (higher dev overall).
  g2:  GENERAL-2 (every seed that finished) against
       - S30 (the init seed) on spatial: 1-6-hop mean, 7-10-hop mean and rotation consistency each >= S30 minus
         max(S30 seed spread, 0.02);
       - GENERAL-1 on worlds_v2 (same --limit 400 eval): in-format overall >= G1, no domain more than 3 points
         below G1, held-out table overall >= G1, counterfactual both-correct mean >= G1, ECE15 (in-format and
         table) <= G1 + 0.01.
       Criteria are applied to the WORST seed. Exit 0 = WIN (promotable), 1 = not.
"""
import json
import sys
from pathlib import Path

R = Path("reports")
DOMAINS = ["causal", "dependency", "infogather", "kinship", "probability", "rules", "temporal"]


def depth(path, K="4"):
    d = json.loads(Path(path).read_text())["by_iters"][K]; h = {int(k): v for k, v in d["by_hops"].items()}
    return {"overall": d["overall"]["accuracy"], "h1_6": sum(h[i] for i in range(1, 7)) / 6,
            "h7_10": sum(h[i] for i in range(7, 11)) / 4, "rot": d["transform_consistency"], "ece": d["overall"]["ece15"]}


def worlds(path):
    d = json.loads(Path(path).read_text()); cf = d["counterfactual"]
    return {"in_format": d["eval_in_format"]["overall"]["accuracy"], "table": d["eval_heldout_table"]["overall"]["accuracy"],
            "ece_in": d["eval_in_format"]["overall"]["ece15"], "ece_table": d["eval_heldout_table"]["overall"]["ece15"],
            "cf_both": sum(v["both_correct"] for v in cf.values()) / len(cf),
            **{f"dom_{k}": d["eval_in_format"][k]["accuracy"] for k in DOMAINS}}


def fmt(d): return {k: round(v, 4) for k, v in d.items()}


def s30():
    runs = {s: depth(R / f"p1/depth_S30_s{s}.json") for s in (7, 8)}
    ref = {"general-v1 (dev BEST_SPATIAL)": depth(R / "p1/depth_general-v1.json")}
    for s, v in runs.items(): print(f"S30_s{s}", fmt(v))
    for k, v in ref.items(): print(k, fmt(v))
    spread = {k: abs(runs[7][k] - runs[8][k]) for k in runs[7]}; print("seed spread", fmt(spread))
    killed = [s for s in (7, 8) if Path(f"checkpoints/p0/S30_s{s}.killed.json").exists()]
    floor = max(v["overall"] for v in ref.values())
    ok = not killed and all(v["overall"] > floor + spread["overall"] for v in runs.values())
    print("S6 reference (P0 ledger): 0.587 +/- 0.002 overall, below the floor above")
    print("PROMOTE S30" if ok else f"S30 NOT promotable (killed={killed}, floor={floor:.3f})")
    best = max(runs, key=lambda s: runs[s]["overall"]); print(f"checkpoints/p0/S30_s{best}.pt")
    return 0 if ok else 1


def general(v, label):
    """GENERAL-v verdict (same criteria for v2 and v3; v3 pre-registered 27 Sep before any result)."""
    init = (R / "general/v2/s30_promotion.txt").read_text().strip().splitlines()[-1]
    seed = init.split("_s")[-1].split(".")[0]
    s30_runs = {s: depth(R / f"p1/depth_S30_s{s}.json") for s in (7, 8)}
    s30_ref = s30_runs[int(seed)]; s30_spread = {k: abs(s30_runs[7][k] - s30_runs[8][k]) for k in s30_ref}
    g1 = worlds(R / "general/v1/worlds_v2.json"); g1_raw = json.loads((R / "general/v1/worlds_v2.json").read_text())["counterfactual"]
    seeds = [s for s in (7, 8) if (R / f"general/{v}/worlds_v2_s{s}.json").exists()]
    if len(seeds) < 2: print(f"only seeds {seeds} finished: two seeds are required before promotion")
    checks = {}
    for s in seeds:
        sp, w = depth(R / f"general/{v}/depth_s{s}.json"), worlds(R / f"general/{v}/worlds_v2_s{s}.json")
        print(f"{label}_s{s} spatial", fmt(sp)); print(f"{label}_s{s} worlds", fmt(w))
        cf = json.loads((R / f"general/{v}/worlds_v2_s{s}.json").read_text())["counterfactual"]
        print(f"{label}_s{s} counterfactual per domain (both-correct, vs G1): "
              + "  ".join(f"{d} {cf[d]['both_correct']:.3f} ({g1_raw[d]['both_correct']:.3f})" for d in sorted(cf)))
        for k in ("h1_6", "h7_10", "rot"):
            checks.setdefault(f"spatial {k} >= S30 - tol", []).append(sp[k] >= s30_ref[k] - max(s30_spread[k], 0.02))
        checks.setdefault("in-format >= G1", []).append(w["in_format"] >= g1["in_format"])
        checks.setdefault("no domain < G1 - 0.03", []).append(all(w[f"dom_{k}"] >= g1[f"dom_{k}"] - 0.03 for k in DOMAINS))
        checks.setdefault("table >= G1", []).append(w["table"] >= g1["table"])
        checks.setdefault("counterfactual mean >= G1", []).append(w["cf_both"] >= g1["cf_both"])
        checks.setdefault("ECE (in-format and table) <= G1 + 0.01", []).append(w["ece_in"] <= g1["ece_in"] + 0.01 and w["ece_table"] <= g1["ece_table"] + 0.01)
    print("S30 ref", init, fmt(s30_ref), "spread", fmt(s30_spread)); print("GENERAL-1", fmt(g1))
    for k, val in checks.items(): print(f"{'PASS' if all(val) else 'FAIL'}  {k}  {val}")
    win = len(seeds) == 2 and all(all(val) for val in checks.values())
    print(f"{label} WIN (promote)" if win else f"{label} not promotable"); return 0 if win else 1


def g2(): return general("v2", "GENERAL-2")


def g3(): return general("v3", "GENERAL-3")


if __name__ == "__main__":
    sys.exit({"s30": s30, "g2": g2, "g3": g3}[sys.argv[1]]())
