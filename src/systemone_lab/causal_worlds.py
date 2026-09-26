"""P2: force vs observe. Small binary structural causal models with exactly computable answers.

Each world is a DAG over 3-4 named binary variables with conditional probability tables. A question
asks P(target = 1 | observe(V = v)) or P(target = 1 | do(V = v)); both are computed exactly by
enumerating the joint (the do-query uses the mutilated graph: V's parents are cut and V is fixed).
Pairs share the state and differ only in the observe/do phrase. "differ" pairs must change answer,
"same" pairs (no back-door path, e.g. chains and mediators) must not.

Wording families are split so whole families can be held out; graph structures are split likewise.
"""
from __future__ import annotations

import itertools
import random
from fractions import Fraction

NAMES = ["alarm", "backlog", "cooling", "delay", "error spike", "failover", "heat", "inspection", "leak",
         "maintenance", "outage", "pressure", "rain", "recall", "retry storm", "sale", "shortage", "surge",
         "throttling", "traffic", "upgrade", "vibration", "wear", "workload"]

# (edges as (parent, child) over roles, observed/intervened role V, target role T)
STRUCTURES = {
    "chain":            ([("X", "M"), ("M", "Y")], "X", "Y"),             # no back door: see == do
    "mediator":         ([("X", "M"), ("M", "Y"), ("X", "Y")], "X", "Y"),  # no back door: see == do
    "fork":             ([("Z", "X"), ("Z", "Y")], "X", "Y"),              # do(X) has no effect on Y
    "confounder":       ([("Z", "X"), ("Z", "Y"), ("X", "Y")], "X", "Y"),  # back door through Z
    "independent":      ([("Z", "Y")], "X", "Y"),                          # X unrelated: see == do
    "collider":         ([("X", "C"), ("Y", "C")], "C", "X"),              # observing a common effect informs X
    "reverse":          ([("Y", "X")], "X", "Y"),                          # observing an effect informs its cause
}
TRAIN_STRUCTURES = ("chain", "mediator", "fork", "confounder", "independent")
BACKDOOR = ("fork", "confounder", "collider", "reverse")  # structures where seeing and doing can disagree
HELDOUT_STRUCTURES = ("collider", "reverse")

OBSERVE = {  # family -> (positive, negative) templates for "V = v"
    "observe_plain":   ("We observe that {v} happened.", "We observe that {v} did not happen."),
    "observe_records": ("Records show {v} occurred.", "Records show no {v}."),
    "observe_given":   ("Given that {v} is present,", "Given that {v} is absent,"),
    "observe_among":   ("Among cases where {v} occurs,", "Among cases where {v} does not occur,"),
    "observe_learn":   ("Suppose we learn that {v} is on.", "Suppose we learn that {v} is off."),
}
DO = {
    "do_force":        ("We force {v} to happen.", "We prevent {v} from happening."),
    "do_set":          ("An intervention sets {v} on.", "An intervention sets {v} off."),
    "do_make":         ("If we make {v} happen,", "If we stop {v} from happening,"),
    "do_trial":        ("In a randomized trial we assign {v}.", "In a randomized trial we withhold {v}."),
    "do_switch":       ("Suppose we switch {v} on regardless of anything else.", "Suppose we switch {v} off regardless of anything else."),
}
TRAIN_WORDING = ("observe_plain", "observe_records", "observe_given", "do_force", "do_set", "do_make")
HELDOUT_WORDING = ("observe_among", "observe_learn", "do_trial", "do_switch")
QUESTIONS = ["How likely is it that {t} happens? Is it more likely than not?", "Will {t} probably occur?",
             "Is {t} more likely than not?"]
PROBS = [Fraction(k, 20) for k in range(2, 19)]  # 0.10 .. 0.90


def _cpts(rng, edges, roles):
    parents = {r: [p for p, c in edges if c == r] for r in roles}
    cpt = {}
    for r in roles:
        for vals in itertools.product((0, 1), repeat=len(parents[r])):
            cpt[(r, vals)] = rng.choice(PROBS)
    if parents.get("Y"):  # make the treatment matter so do-queries are non-trivial
        for vals in itertools.product((0, 1), repeat=len(parents["Y"])):
            cpt[("Y", vals)] = rng.choice(PROBS[:5] if vals and vals[0] == 0 else PROBS[-5:])
    return parents, cpt


def _joint(roles, parents, cpt, do=None):
    out = {}
    for vals in itertools.product((0, 1), repeat=len(roles)):
        a = dict(zip(roles, vals)); p = Fraction(1)
        for r in roles:
            if do and r == do[0]:
                p *= 1 if a[r] == do[1] else 0
                continue
            q = cpt[(r, tuple(a[x] for x in parents[r]))]; p *= q if a[r] else 1 - q
        out[vals] = p
    return out


def query(roles, parents, cpt, target, var, val, kind):
    """Exact P(target = 1 | observe(var = val)) or P(target = 1 | do(var = val))."""
    j = _joint(roles, parents, cpt, do=(var, val) if kind == "do" else None)
    ti, vi = roles.index(target), roles.index(var)
    num = sum(p for v, p in j.items() if v[ti] == 1 and v[vi] == val)
    den = sum(p for v, p in j.items() if v[vi] == val)
    return num / den


def world(rng, structure):
    edges, V, T = STRUCTURES[structure]
    roles = sorted({r for e in edges for r in e} | {V, T})
    names = dict(zip(roles, rng.sample(NAMES, len(roles))))
    parents, cpt = _cpts(rng, edges, roles)
    return {"structure": structure, "roles": roles, "names": names, "edges": edges, "parents": parents,
            "cpt": cpt, "V": V, "T": T}


def state_text(w, rng):
    n = w["names"]; f = lambda q: f"{q.numerator}/{q.denominator}"; lines = []
    for p, c in w["edges"]:
        lines.append(rng.choice(["{p} directly influences {c}.", "{c} depends on {p}.", "{p} is a cause of {c}."]).format(p=n[p], c=n[c]))
    for r in w["roles"]:
        ps = w["parents"][r]
        for vals in itertools.product((0, 1), repeat=len(ps)):
            cond = " and ".join(f"{n[p]}" if v else f"no {n[p]}" for p, v in zip(ps, vals))
            q = w["cpt"][(r, vals)]
            lines.append(f"With {cond}, {n[r]} happens with probability {f(q)}." if ps else f"{n[r]} happens with probability {f(q)}.")
    if w["V"] not in {c for _, c in w["edges"]} | {p for p, _ in w["edges"]}:
        lines.append(f"{n[w['V']]} happens with probability 1/2 and is unrelated to the others.")
    rng.shuffle(lines)
    return " ".join(lines)


def pair(rng, structure, families, qid, differ_rate=0.6):
    """A (see, do) pair over one world, identical except for the condition phrase.

    For back-door structures, tables are rejection-sampled so that about `differ_rate` of pairs have
    observe and do on opposite sides of 0.5; other structures are inherently "same" pairs.
    """
    want_differ = structure in BACKDOOR and rng.random() < differ_rate
    for _ in range(2000):
        w = world(rng, structure)
        val = rng.choice([0, 1])
        ps = {k: query(w["roles"], w["parents"], w["cpt"], w["T"], w["V"], val, k) for k in ("see", "do")}
        if any(abs(float(p) - 0.5) < 0.06 for p in ps.values()): continue  # keep labels well away from the threshold
        if ((float(ps["see"]) >= .5) != (float(ps["do"]) >= .5)) == want_differ: break
    state = state_text(w, rng); qt = rng.choice(QUESTIONS).format(t=w["names"][w["T"]])
    obs_f = rng.choice([f for f in families if f.startswith("observe")]); do_f = rng.choice([f for f in families if f.startswith("do")])
    out = []
    for kind, fam, table in (("see", obs_f, OBSERVE), ("do", do_f, DO)):
        cond = table[fam][0 if val else 1].format(v=w["names"][w["V"]])
        q_text = qt[0].lower() + qt[1:] if cond.endswith(",") else qt  # "Given that X is present, will ..."
        p = float(ps[kind])
        out.append({"id": f"{qid}-{kind}", "state": state,
                    "questions": {"answer": {"type": "noul", "instructions": f"{cond} {q_text}"}},
                    "labels": {"answer": p >= 0.5}, "distributions": {"answer": {"true": p, "false": 1 - p}},
                    "meta": {"domain": "causal_sd", "structure": structure, "kind": kind, "wording": fam, "pair_id": qid,
                             "variant": "base" if kind == "see" else "counterfactual",
                             "pair_kind": "differ" if (float(ps["see"]) >= .5) != (float(ps["do"]) >= .5) else "same",
                             "p_see": float(ps["see"]), "p_do": float(ps["do"]), "question": qt}})
    return out


def generate(n_pairs, seed, structures, families, prefix):
    rng = random.Random(seed); rows = []
    for i in range(n_pairs):
        rows += pair(rng, structures[i % len(structures)], families, f"{prefix}-{seed}-{i}")
    return rows
