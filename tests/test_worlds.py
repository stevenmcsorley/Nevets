import json
import random

from systemone_lab.worlds import (FORMATS, Fact, generate, kin_relation, probability_example, render,
                                  rules_example, temporal_example, _closure)


def test_kinship_relations_from_tree():
    w = {"people": ["g", "p1", "p2", "c1", "c2", "x"], "parent": {"p1": "g", "p2": "g", "c1": "p1", "c2": "p2"}}
    assert kin_relation(w, "g", "p1") == "parent" and kin_relation(w, "p1", "g") == "child"
    assert kin_relation(w, "g", "c1") == "grandparent" and kin_relation(w, "c2", "g") == "grandchild"
    assert kin_relation(w, "p1", "p2") == "sibling" and kin_relation(w, "p1", "c2") == "aunt or uncle"
    assert kin_relation(w, "c1", "p2") == "niece or nephew" and kin_relation(w, "c1", "c2") == "cousin"
    assert kin_relation(w, "x", "g") == "unrelated"


def test_temporal_labels_follow_transitive_closure():
    rng = random.Random(0)
    for i in range(300):
        rec, world = temporal_example(rng, "kv", f"t{i}")
        a, b = rec["meta"]["query"]; reach = _closure(world["events"], world["edges"])
        want = "before" if b in reach[a] else "after" if a in reach[b] else "cannot be determined"
        assert rec["labels"]["answer"] == want


def test_probability_posterior_is_exact_bayes():
    rng = random.Random(1)
    for i in range(200):
        rec, _ = probability_example(rng, "json", f"p{i}")
        post = rec["distributions"]["answer"]
        assert abs(sum(post.values()) - 1) < 1e-9 and rec["labels"]["answer"] == max(post, key=post.get)
        assert set(post) == set(rec["questions"]["answer"]["criteria"])


def test_rules_counterfactual_flip_changes_only_that_attribute():
    rng = random.Random(2); seen_flip = 0
    for i in range(200):
        state = rng.getstate()
        rec, attrs = rules_example(rng, "prose", f"r{i}")
        rng2 = random.Random(); rng2.setstate(state)
        twin, attrs2 = rules_example(rng2, "prose", f"r{i}", flip=("prior_default", not attrs["prior_default"]))
        assert {k for k in attrs if attrs[k] != attrs2[k]} == {"prior_default"}
        if attrs["prior_default"]: assert rec["labels"]["answer"] is False
        seen_flip += rec["labels"]["answer"] != twin["labels"]["answer"]
    assert seen_flip > 0


def test_every_format_carries_every_fact():
    facts = [Fact("a", "r", "b", ["{s} r {o}."]), Fact("c", "r", "d", ["{s} r {o}."])]
    for fmt in FORMATS:
        text = render(facts, fmt, random.Random(0))
        assert all(x in text for x in "abcd")
    assert len(json.loads(render(facts, "json", random.Random(0)))["facts"]) == 2


def test_generate_is_deterministic_and_labels_are_options():
    a, b = generate(100, 7), generate(100, 7)
    assert a == b
    for r in a:
        q = r["questions"]["answer"]
        if q["type"] == "choice": assert r["labels"]["answer"] in q["criteria"]
        else: assert isinstance(r["labels"]["answer"], bool)


def test_dependency_twin_keeps_question_and_differs_by_one_edge():
    import sys; sys.path.insert(0, "scripts")
    from generate_worlds import dependency_twins, rules_twins
    twins = dependency_twins(random.Random(3), 20, "prose", "t") + rules_twins(random.Random(4), 20, "prose", "r")
    for base, cf in zip(twins[::2], twins[1::2]):
        assert base["meta"]["pair_id"] == cf["meta"]["pair_id"] and base["labels"] != cf["labels"]
        assert base["questions"] == cf["questions"]
        if base["meta"]["domain"] == "dependency":
            assert len(set(base["state"].split(". ")) ^ set(cf["state"].split(". "))) <= 2 or len(cf["state"]) < len(base["state"])


def test_infogather_label_is_expected_utility_optimum_and_cost_monotone():
    from fractions import Fraction
    from systemone_lab.worlds import infogather_example
    rng = random.Random(9)
    for i in range(300):
        state = rng.getstate(); rec, _ = infogather_example(rng, "kv", f"i{i}")
        eu = rec["meta"]["eu"]; best = max(eu, key=eu.get)
        acts = {k: v for k, v in eu.items() if k != "run the diagnostic"}
        want = "run the diagnostic" if eu["run the diagnostic"] > max(acts.values()) + 1e-12 else max(acts, key=acts.get)
        assert rec["labels"]["answer"] == want and set(rec["questions"]["answer"]["criteria"]) == set(eu)
        # Raising the test cost can only move the decision from testing to acting, never the reverse.
        r2 = random.Random(); r2.setstate(state)
        dear, _ = infogather_example(r2, "kv", f"i{i}", cost=Fraction(99, 100))
        assert dear["labels"]["answer"] != "run the diagnostic"


def test_generation_is_independent_of_python_hash_seed(tmp_path):
    import os, subprocess, sys
    outs = []
    for seed in ("1", "12345"):
        d = tmp_path / seed
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH="src")
        subprocess.run([sys.executable, "scripts/generate_worlds.py", "--out-dir", str(d), "--train", "300",
                        "--eval", "40", "--twins", "10"], check=True, env=env, capture_output=True)
        outs.append({f.name: f.read_bytes() for f in d.glob("*.jsonl")})
    assert outs[0] == outs[1]


def test_causal_targets_match_brute_force_enumeration():
    from itertools import product
    from systemone_lab.worlds import causal_example
    rng = random.Random(11); differ = 0
    for i in range(300):
        rec, w = causal_example(rng, "prose", f"c{i}")
        pz, px, py = w["pz"], w["px"], w["py"]; xv = rec["meta"]["x"]
        joint = {}
        for zv, xx, yv in product((0, 1), repeat=3):
            p = (pz if zv else 1 - pz) * (px[zv] if xx else 1 - px[zv]) * (py[(xx, zv)] if yv else 1 - py[(xx, zv)])
            joint[(zv, xx, yv)] = p
        see = sum(v for (zv, xx, yv), v in joint.items() if xx == xv and yv) / sum(v for (zv, xx, yv), v in joint.items() if xx == xv)
        do = sum((pz if zv else 1 - pz) * py[(xv, zv)] for zv in (0, 1))
        want = float(do if rec["meta"]["kind"] == "do" else see)
        assert abs(rec["meta"]["p_exact"] - want) < 1e-12
        assert abs(rec["distributions"]["answer"]["true"] - want) < 1e-12
        differ += abs(float(do) - float(see)) > 0.05
    assert differ > 150  # strong confounding: seeing and doing differ in most worlds
