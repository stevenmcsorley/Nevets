import random
from fractions import Fraction

from systemone_lab.causal_worlds import STRUCTURES, TRAIN_WORDING, HELDOUT_WORDING, generate, query, world


def test_do_queries_follow_the_mutilated_graph():
    rng = random.Random(0)
    for _ in range(50):
        w = world(rng, "fork")  # Z -> X, Z -> Y: forcing X cannot change Y
        pz = w["cpt"][("Z", ())]
        py = lambda z: w["cpt"][("Y", (z,))]
        marginal = pz * py(1) + (1 - pz) * py(0)
        for v in (0, 1):
            assert query(w["roles"], w["parents"], w["cpt"], "Y", "X", v, "do") == marginal
        w = world(rng, "chain")  # no back door: seeing X and doing X agree
        for v in (0, 1):
            assert query(w["roles"], w["parents"], w["cpt"], "Y", "X", v, "do") == query(w["roles"], w["parents"], w["cpt"], "Y", "X", v, "see")
        w = world(rng, "collider")  # intervening on a common effect leaves its causes alone
        px = w["cpt"][("X", ())]
        assert query(w["roles"], w["parents"], w["cpt"], "X", "C", 1, "do") == px


def test_pairs_differ_only_in_the_condition_and_labels_match_probabilities():
    rows = generate(70, 3, tuple(STRUCTURES), TRAIN_WORDING, "t")
    for see, do in zip(rows[::2], rows[1::2]):
        assert see["state"] == do["state"] and see["meta"]["pair_id"] == do["meta"]["pair_id"]
        for r in (see, do):
            p = r["distributions"]["answer"]["true"]
            assert r["labels"]["answer"] == (p >= 0.5) and abs(p - 0.5) >= 0.06 - 1e-9
            assert r["meta"]["wording"] in TRAIN_WORDING
        q_see, q_do = see["questions"]["answer"]["instructions"], do["questions"]["answer"]["instructions"]
        qt = see["meta"]["question"]; assert qt == do["meta"]["question"] and q_see != q_do
        assert q_see.lower().endswith(qt.lower()) and q_do.lower().endswith(qt.lower())
    kinds = {r["meta"]["pair_kind"] for r in rows}
    assert kinds == {"differ", "same"}
    same = [r for r in rows if r["meta"]["structure"] in ("chain", "mediator", "independent")]
    assert all(r["meta"]["pair_kind"] == "same" for r in same)


def test_wording_families_are_disjoint():
    assert not set(TRAIN_WORDING) & set(HELDOUT_WORDING)


def test_backdoor_structures_yield_many_differ_pairs():
    from collections import Counter
    rows = generate(400, 5, ("fork", "confounder", "collider", "reverse"), TRAIN_WORDING, "b")
    c = Counter(r["meta"]["pair_kind"] for r in rows[::2])
    assert 0.45 < c["differ"] / sum(c.values()) < 0.75
