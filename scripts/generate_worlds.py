"""Build a data-factory split: train formats, in-format eval, held-out-format eval, counterfactual twins.

Counterfactual twins change exactly one causally relevant fact and keep only twins whose answer
changes: rules flip one attribute; dependency removes one edge on the failure path.
"""
import argparse
import json
import random
from pathlib import Path

from fractions import Fraction

from systemone_lab.worlds import DOMAINS, causal_example, dependency_example, infogather_example, rules_example, _closure


def rules_twins(rng, n, fmt, prefix):
    out = []
    while len(out) < 2 * n:
        state = rng.getstate(); base, attrs = rules_example(rng, fmt, f"{prefix}-{len(out)}-base")
        key = rng.choice(["age", "income", "prior_default", "veteran"])
        value = {"age": rng.randint(15, 40), "income": rng.randint(10, 60)}.get(key, not attrs.get(key))
        r2 = random.Random(); r2.setstate(state)
        twin, _ = rules_example(r2, fmt, f"{prefix}-{len(out)}-cf", flip=(key, value))
        if twin["labels"] != base["labels"]:
            pid = f"{prefix}-{len(out)}"
            base["meta"].update(pair_id=pid, variant="base"); twin["meta"].update(pair_id=pid, variant="counterfactual", intervention=f"{key}={value}")
            out += [base, twin]
    return out


def dependency_twins(rng, n, fmt, prefix):
    out = []
    while len(out) < 2 * n:
        state = rng.getstate(); base, world = dependency_example(rng, fmt, f"{prefix}-{len(out)}-base")
        if not base["labels"]["answer"]: continue
        down, target = base["meta"]["query"]
        # An edge on some path target -> ... -> down; dropping it may disconnect them.
        reach = _closure(world["svc"], world["edges"])
        on_path = [(a, b) for a, b in sorted(world["edges"]) if (a == target or a in reach[target]) and (b == down or down in reach[b])]
        edge = rng.choice(on_path)
        r2 = random.Random(); r2.setstate(state)
        twin, _ = dependency_example(r2, fmt, f"{prefix}-{len(out)}-cf", drop_edge=edge, query=(down, target))
        if twin["labels"] != base["labels"]:
            pid = f"{prefix}-{len(out)}"
            base["meta"].update(pair_id=pid, variant="base"); twin["meta"].update(pair_id=pid, variant="counterfactual", intervention=f"drop {edge}")
            out += [base, twin]
    return out


def infogather_twins(rng, n, fmt, prefix):
    """Same world; the diagnostic's cost moves across the act/test break-even point."""
    out = []
    while len(out) < 2 * n:
        state = rng.getstate(); base, _ = infogather_example(rng, fmt, f"{prefix}-{len(out)}-base")
        voi = base["meta"]["value_of_information"]
        if voi <= 0.02: continue  # a test is never worth it here; no break-even to cross
        cheap, dear = Fraction(1, 100), Fraction(round(min(0.99, voi + 0.05) * 100), 100)
        twins = []
        for tag, cost in (("base", cheap), ("counterfactual", dear)):
            r2 = random.Random(); r2.setstate(state)
            rec, _ = infogather_example(r2, fmt, f"{prefix}-{len(out)}-{tag}", cost=cost); twins.append(rec)
        if twins[0]["labels"] != twins[1]["labels"]:
            pid = f"{prefix}-{len(out)}"
            twins[0]["meta"].update(pair_id=pid, variant="base")
            twins[1]["meta"].update(pair_id=pid, variant="counterfactual", intervention=f"cost -> {dear}")
            out += twins
    return out


def causal_twins(rng, n, fmt, prefix):
    """Same world and state; only the question changes between observing X and intervening on X."""
    out = []
    while len(out) < 2 * n:
        state = rng.getstate(); recs = []
        for kind in ("see", "do"):
            r2 = random.Random(); r2.setstate(state)
            rec, _ = causal_example(r2, fmt, f"{prefix}-{len(out)}-{kind}", kind=kind); recs.append(rec)
        rng.setstate(r2.getstate())
        if recs[0]["meta"]["x"] == recs[1]["meta"]["x"] and recs[0]["labels"] != recs[1]["labels"]:
            pid = f"{prefix}-{len(out)}"
            recs[0]["meta"].update(pair_id=pid, variant="base"); recs[1]["meta"].update(pair_id=pid, variant="counterfactual", intervention="see -> do")
            out += recs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True); ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--eval", type=int, default=1000, help="per domain"); ap.add_argument("--twins", type=int, default=300)
    ap.add_argument("--train-formats", default="prose,json,kv"); ap.add_argument("--heldout-format", default="table")
    ap.add_argument("--seed", type=int, default=90001)
    ap.add_argument("--domains", default="kinship,temporal,dependency,probability,rules",
                    help="comma list; worlds_v1 used the default five")
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_fmts = a.train_formats.split(","); doms = a.domains.split(",")
    rng = random.Random(a.seed + 1)
    ev_in = [DOMAINS[d](rng, rng.choice(train_fmts), f"evin-{d}-{i}")[0] for d in doms for i in range(a.eval)]
    ev_out = [DOMAINS[d](rng, a.heldout_format, f"evout-{d}-{i}")[0] for d in doms for i in range(a.eval)]
    twins = rules_twins(rng, a.twins, "prose", "cf-rules") + dependency_twins(rng, a.twins, "prose", "cf-dep")
    if "infogather" in doms: twins += infogather_twins(rng, a.twins, "prose", "cf-info")
    if "causal" in doms: twins += causal_twins(rng, a.twins, "prose", "cf-causal")
    banned = {r["state"] for r in ev_in + ev_out + twins}
    rng = random.Random(a.seed); train = []; dropped = 0
    while len(train) < a.train:
        d = doms[len(train) % len(doms)]
        rec = DOMAINS[d](rng, rng.choice(train_fmts), f"train-{len(train)}")[0]
        if rec["state"] in banned: dropped += 1; continue
        train.append(rec)
    for name, rows in (("train", train), ("eval_in_format", ev_in), (f"eval_heldout_{a.heldout_format}", ev_out), ("counterfactual", twins)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {"seed": a.seed, "train": len(train), "dropped_train_collisions": dropped, "eval_per_domain": a.eval,
                "twins_per_family": a.twins, "train_formats": train_fmts, "heldout_format": a.heldout_format, "domains": doms}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest))


if __name__ == "__main__":
    main()
