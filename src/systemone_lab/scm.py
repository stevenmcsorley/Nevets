"""P2 (force vs observe): small structural causal models with exactly computable answers.

A world is a binary SCM over 3-5 named variables: a DAG from a structure library plus conditional probability tables
with rational entries. Answers are computed exactly by enumerating all assignments (Fractions), with graph surgery for
interventions, so observing (P(Y | X=x)) and forcing (P(Y | do(X=x))) can be compared on the same world.

Generalisation splits are enforced by construction (and tested):
  * structures: TRAIN_STRUCTURES are used for training; HELDOUT_STRUCTURES (M-bias, front-door) only for evaluation;
  * wordings: every question and fact sentence has paraphrase families; families listed in HELDOUT_* appear only in the
    held-out-wording evaluation set;
  * each world yields an observe/do PAIR on the same query (meta.pair_id, meta.variant), the P2 analogue of the
    counterfactual both-correct metric. A pair is "discriminating" when the two exact answers fall on different sides
    of 1/2, so only telling seeing from doing gets both right.
Questions are yes/no ("more likely than not?") with the exact probability as a soft target, like the causal domain.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction

from systemone_lab.worlds import Fact, record, render

PROBS = [Fraction(k, 10) for k in range(1, 10)]
VAR_NAMES = ["rain", "sprinkler", "wet grass", "traffic", "delay", "stress", "fatigue", "errors", "alarm", "outage",
             "demand", "price", "sales", "promotion", "reviews", "fever", "infection", "vaccine", "cough", "rest",
             "wind", "waves", "ferry cancellation", "fog", "tourism", "budget", "staffing", "backlog", "overtime", "morale"]

# Structures: variables by role; X = treatment, Y = outcome. Edges are (parent, child).
STRUCTURES = {
    "confounder":  (["Z", "X", "Y"], [("Z", "X"), ("Z", "Y"), ("X", "Y")]),
    "chain":       (["X", "M", "Y"], [("X", "M"), ("M", "Y")]),
    "collider":    (["X", "Y", "C"], [("X", "C"), ("Y", "C"), ("X", "Y")]),
    "mediated_confounded": (["Z", "X", "M", "Y"], [("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y")]),
    "instrument":  (["I", "Z", "X", "Y"], [("I", "X"), ("Z", "X"), ("Z", "Y"), ("X", "Y")]),
    "independent_cause": (["X", "W", "Y"], [("X", "Y"), ("W", "Y")]),
    # held out (evaluation only):
    "m_bias":      (["A", "B", "M", "X", "Y"], [("A", "M"), ("B", "M"), ("A", "X"), ("B", "Y"), ("X", "Y")]),
    "front_door":  (["U", "X", "M", "Y"], [("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")]),
}
TRAIN_STRUCTURES = ("confounder", "chain", "collider", "mediated_confounded", "instrument", "independent_cause")
HELDOUT_STRUCTURES = ("m_bias", "front_door")

# Paraphrase families. {x} treatment, {y} outcome, {nx} "no {x}" handled by the caller via on().
OBSERVE_Q = [
    "We observe {xo}. How likely is {y}?",
    "Given that we see {xo}, is {y} more likely than not?",
    "Suppose records show {xo}. Would you expect {y}?",
    "In cases where {xo} is found, does {y} usually happen?",
    "Looking only at situations with {xo}, is {y} probable?",
    "A report notes {xo}. Is {y} likely?",
]
DO_Q = [
    "If we {force} {x}, how likely is {y}?",
    "Suppose we intervene to {force} {x}. Is {y} more likely than not?",
    "If {x} is {forced} by an outside decision, would you expect {y}?",
    "We set {x} ourselves ({forced}). Does {y} usually happen?",
    "Imagine a policy that {forces} {x} for everyone. Is {y} probable?",
    "An experimenter {forces} {x}. Is {y} likely?",
]
HELDOUT_OBSERVE, HELDOUT_DO = (4, 5), (4, 5)          # families used only in the held-out-wording set
ROOT_T = ["{s} occurs with probability {o}.", "The chance of {s} is {o}.", "{s} happens {o} of the time."]
COND_T = ["With {c}, {s} happens with probability {o}.", "When {c}, the chance of {s} is {o}.", "Given {c}, {s} occurs {o} of the time."]
EDGE_T = ["{s} directly affects {o}.", "{s} has a direct effect on {o}.", "{o} depends directly on {s}."]


def _f(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


def _on(v: int, name: str) -> str:
    return name if v else f"no {name}"


@dataclass
class World:
    structure: str
    names: dict            # role -> variable name
    parents: dict          # role -> tuple of parent roles (topological order kept in `order`)
    order: list
    cpt: dict              # role -> {parent assignment tuple: P(role=1)}

    def joint(self, do: dict | None = None):
        """Exact joint over all assignments; `do` = {role: value} applies graph surgery."""
        do = do or {}
        for vals in itertools.product((0, 1), repeat=len(self.order)):
            a = dict(zip(self.order, vals)); p = Fraction(1)
            for r in self.order:
                if r in do:
                    if a[r] != do[r]: p = Fraction(0); break
                    continue
                q = self.cpt[r][tuple(a[pa] for pa in self.parents[r])]
                p *= q if a[r] else 1 - q
            if p: yield a, p

    def prob(self, target: str, given: dict | None = None, do: dict | None = None) -> Fraction:
        given = given or {}; num = den = Fraction(0)
        for a, p in self.joint(do):
            if all(a[k] == v for k, v in given.items()):
                den += p
                if a[target]: num += p
        if den == 0: raise ZeroDivisionError("conditioning event has probability 0")
        return num / den


def sample_world(rng: random.Random, structure: str) -> World:
    roles, edges = STRUCTURES[structure]
    names = dict(zip(roles, rng.sample(VAR_NAMES, len(roles))))
    parents = {r: tuple(p for p, c in edges if c == r) for r in roles}
    order, left = [], set(roles)
    while left:  # topological order
        r = next(r for r in roles if r in left and all(p in order for p in parents[r])); order.append(r); left.remove(r)
    cpt = {}
    for r in order:
        rows = list(itertools.product((0, 1), repeat=len(parents[r])))
        vals = [rng.choice(PROBS) for _ in rows]
        if len(rows) > 1:  # make every parent matter (at least one 0.3 gap per parent), so edges are real
            for i, pa in enumerate(parents[r]):
                a = next(k for k, row in enumerate(rows) if row[i] == 0); b = rows.index(tuple(1 if j == i else v for j, v in enumerate(rows[a])))
                while abs(vals[a] - vals[b]) < Fraction(3, 10): vals[b] = rng.choice(PROBS)
        cpt[r] = dict(zip(rows, vals))
    return World(structure, names, parents, order, cpt)


def world_facts(w: World) -> list[Fact]:
    facts = []
    for r in w.order:
        s = w.names[r]
        if not w.parents[r]:
            facts.append(Fact(s, "base_rate", _f(w.cpt[r][()]), ROOT_T)); continue
        for row, q in w.cpt[r].items():
            cond = " and ".join(_on(v, w.names[p]) for p, v in zip(w.parents[r], row))
            facts.append(Fact(s, "given_" + "_and_".join(_on(v, w.names[p]).replace(" ", "_") for p, v in zip(w.parents[r], row)), _f(q),
                              [t.replace("{c}", cond) for t in COND_T]))
    for r in w.order:
        for p in w.parents[r]:
            facts.append(Fact(w.names[p], "causes", w.names[r], EDGE_T))
    return facts


def question_text(kind: str, family: int, x: str, y: str, xv: int) -> str:
    if kind == "see":
        return OBSERVE_Q[family].format(xo=_on(xv, x), y=y)
    verbs = ("force", "forced", "forces") if xv else ("prevent", "prevented", "prevents")
    return DO_Q[family].format(force=verbs[0], forced=verbs[1], forces=verbs[2], x=x, y=y)


def pair(rng: random.Random, structure: str, fmt: str, rid: str, families=None, discriminating: bool | None = None):
    """One world -> an observe record and a do record on the same query (X=x, outcome Y). For colliders the observe
    question also conditions on the collider C (selection), the classic see/do divergence."""
    for _ in range(200):
        w = sample_world(rng, structure); xv = rng.choice([0, 1])
        given = {"X": xv}
        if structure == "collider": given["C"] = 1
        try:
            p_see = w.prob("Y", given=given); p_do = w.prob("Y", do={"X": xv})
        except ZeroDivisionError:
            continue
        disc = (p_see >= Fraction(1, 2)) != (p_do >= Fraction(1, 2))
        if discriminating is None or disc == discriminating: break
    fam_see, fam_do = families if families else (rng.randrange(4), rng.randrange(4))
    facts = world_facts(w); x, y = w.names["X"], w.names["Y"]
    state = render(facts, fmt, rng)  # one shared state for both questions of the pair
    see_text = question_text("see", fam_see, x, y, xv)
    if structure == "collider": see_text = see_text.rstrip("?") + f", among cases with {w.names['C']}?"
    out = []
    for kind, text, p, fam in (("see", see_text, p_see, fam_see), ("do", question_text("do", fam_do, x, y, xv), p_do, fam_do)):
        pf = float(p)
        out.append(record(f"{rid}-{kind}", state, {"answer": {"type": "noul", "instructions": text}}, {"answer": pf >= 0.5},
                          {"domain": "causal_scm", "format": fmt, "structure": structure, "kind": kind, "x": xv, "p_exact": pf,
                           "p_exact_frac": _f(p), "family": fam, "pair_id": rid, "variant": kind, "discriminating": disc},
                          {"answer": {"true": pf, "false": 1 - pf}}))
    return out


def generate(n_pairs: int, seed: int, split: str, formats=("prose", "json", "kv"), prefix="scm", disc_share=0.5):
    """split: train | eval_iid | eval_heldout_wording | eval_heldout_structure."""
    rng = random.Random(seed); recs = []
    for i in range(n_pairs):
        structs = HELDOUT_STRUCTURES if split == "eval_heldout_structure" else TRAIN_STRUCTURES
        s = rng.choice(structs); fmt = rng.choice(formats)
        fam = (rng.choice(HELDOUT_OBSERVE), rng.choice(HELDOUT_DO)) if split == "eval_heldout_wording" else (rng.randrange(4), rng.randrange(4))
        recs += pair(rng, s, fmt, f"{prefix}-{split}-{i}", families=fam, discriminating=rng.random() < disc_share)
    return recs
