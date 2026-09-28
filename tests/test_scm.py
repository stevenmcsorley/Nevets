import random
from fractions import Fraction

import pytest

from systemone_lab import scm


def test_confounder_matches_the_closed_form_used_by_the_causal_domain():
    rng = random.Random(3)
    w = scm.sample_world(rng, "confounder")
    pz = w.cpt["Z"][()]
    for xv in (0, 1):
        do = sum((pz if b else 1 - pz) * w.cpt["Y"][(b, xv)] for b in (0, 1))  # parents of Y are (Z, X)
        assert w.prob("Y", do={"X": xv}) == do
        joint = {b: (pz if b else 1 - pz) * (w.cpt["X"][(b,)] if xv else 1 - w.cpt["X"][(b,)]) for b in (0, 1)}
        see = sum(joint[b] / sum(joint.values()) * w.cpt["Y"][(b, xv)] for b in (0, 1))
        assert w.prob("Y", given={"X": xv}) == see


def test_seeing_equals_doing_without_confounding_and_differs_with_it():
    rng = random.Random(0)
    for _ in range(20):
        w = scm.sample_world(rng, "chain")          # X -> M -> Y: no back-door path
        for xv in (0, 1): assert w.prob("Y", given={"X": xv}) == w.prob("Y", do={"X": xv})
    diffs = 0
    for _ in range(50):
        w = scm.sample_world(rng, "confounder")
        diffs += any(w.prob("Y", given={"X": v}) != w.prob("Y", do={"X": v}) for v in (0, 1))
    assert diffs >= 45


def test_front_door_do_matches_the_front_door_formula():
    rng = random.Random(5)
    w = scm.sample_world(rng, "front_door")      # U confounds X, Y; X -> M -> Y
    for xv in (0, 1):
        pm = {m: w.prob("M", given={"X": xv}) if m else 1 - w.prob("M", given={"X": xv}) for m in (0, 1)}
        px = {x: w.prob("X") if x else 1 - w.prob("X") for x in (0, 1)}
        fd = sum(pm[m] * sum(w.prob("Y", given={"X": x, "M": m}) * px[x] for x in (0, 1)) for m in (0, 1))
        assert w.prob("Y", do={"X": xv}) == fd


def test_pairs_share_the_state_and_carry_exact_answers():
    recs = scm.pair(random.Random(1), "confounder", "prose", "p0")
    assert len(recs) == 2 and recs[0]["state"] == recs[1]["state"]
    assert {r["meta"]["variant"] for r in recs} == {"see", "do"} and recs[0]["meta"]["pair_id"] == recs[1]["meta"]["pair_id"]
    for r in recs:
        p = r["meta"]["p_exact"]; assert r["labels"]["answer"] == (p >= 0.5)
        assert abs(r["distributions"]["answer"]["true"] - p) < 1e-12 and Fraction(r["meta"]["p_exact_frac"]) == Fraction(p).limit_denominator(10**6)


def test_discriminating_pairs_really_discriminate():
    rng = random.Random(2)
    for s in scm.TRAIN_STRUCTURES:
        recs = scm.pair(rng, s, "kv", f"d-{s}", discriminating=True)
        if recs[0]["meta"]["discriminating"]:
            assert recs[0]["labels"]["answer"] != recs[1]["labels"]["answer"]


def test_splits_keep_heldout_structures_and_wordings_out_of_training():
    train = scm.generate(300, 11, "train")
    assert {r["meta"]["structure"] for r in train} <= set(scm.TRAIN_STRUCTURES)
    held_q = {scm.OBSERVE_Q[i].split("{")[0] for i in scm.HELDOUT_OBSERVE} | {scm.DO_Q[i].split("{")[0] for i in scm.HELDOUT_DO}
    for r in train:
        assert r["meta"]["family"] not in (4, 5)
        assert not any(r["questions"]["answer"]["instructions"].startswith(h) for h in held_q if h)
    hs = scm.generate(40, 12, "eval_heldout_structure")
    assert {r["meta"]["structure"] for r in hs} == set(scm.HELDOUT_STRUCTURES)
    hw = scm.generate(40, 13, "eval_heldout_wording")
    assert all(r["meta"]["family"] in (4, 5) for r in hw)


def test_generation_is_deterministic_and_answers_are_balanced_enough():
    a, b = scm.generate(50, 7, "train"), scm.generate(50, 7, "train")
    assert a == b
    share = sum(r["labels"]["answer"] for r in scm.generate(400, 8, "train")) / 800
    assert 0.25 < share < 0.75


def test_collider_observe_question_conditions_on_the_collider():
    recs = scm.pair(random.Random(4), "collider", "prose", "c0")
    see = next(r for r in recs if r["meta"]["kind"] == "see")
    assert "among cases with" in see["questions"]["answer"]["instructions"]


def test_p3_symbolic_format_renders_postfix_and_is_rejected_for_training(tmp_path):
    import json, re, subprocess, sys
    from systemone_lab.worlds import DOMAINS, P3_HELDOUT_FORMATS, SYMBOLIC_RE
    assert "symbolic" in P3_HELDOUT_FORMATS
    rng = random.Random(9)
    for dom, fn in DOMAINS.items():
        rec, _ = fn(rng, "symbolic", f"p3-{dom}")
        st = rec["state"]
        assert re.search(SYMBOLIC_RE, st), (dom, st[:120])
        facts = [ln for ln in st.splitlines() if re.search(SYMBOLIC_RE, ln)][0]
        assert "(" not in facts and "{" not in facts and "|" not in facts and "," not in facts
    for fmt in ("prose", "json", "kv", "csv", "bullets"):
        rec, _ = DOMAINS["kinship"](rng, fmt, "ok"); assert not re.search(SYMBOLIC_RE, rec["state"])
    bad = tmp_path / "bad.jsonl"; rec, _ = DOMAINS["rules"](rng, "symbolic", "x"); bad.write_text(json.dumps(rec) + "\n")
    r = subprocess.run([sys.executable, "scripts/check_contamination.py", str(bad)], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert r.returncode == 1 and '"p3_heldout_format_records": 1' in r.stdout
