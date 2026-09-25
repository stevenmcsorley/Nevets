"""Data factory: latent worlds with known truth, several renderings, counterfactual twins.

Every generator returns a latent world plus questions whose answers are computed from the latent
world, never from the text. `render(facts, fmt)` turns the same fact list into prose, JSON, a
table or key-value records, so a format can be held out entirely. `counterfactual` changes exactly
one causally relevant fact and recomputes the answer.

Domains
  kinship      family tree; relation of X to Y composes parent/child links (multi-hop)
  temporal     partial order of events; before / after / cannot be determined
  dependency   service graph; does a failure in X take down Y (reachability)
  probability  boxes, sensors and base rates; exact Bayesian posterior as a soft target
  rules        eligibility policy with an exception; nested conditions over attributes
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from fractions import Fraction

FORMATS = ("prose", "json", "table", "kv")
NAMES = ["Ada", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena", "Milo",
         "Nia", "Omar", "Pia", "Quin", "Rosa", "Sam", "Tara", "Uma", "Vic", "Wren", "Xavi", "Yara", "Zed"]
SERVICES = ["auth", "billing", "cache", "db", "dns", "email", "gateway", "index", "ledger", "metrics",
            "notify", "orders", "payments", "queue", "search", "storage", "users", "web"]
EVENTS = ["audit", "backup", "briefing", "call", "deploy", "demo", "hiring", "launch", "lunch", "meeting",
          "migration", "review", "standup", "survey", "training", "upgrade", "vote", "workshop"]


@dataclass
class Fact:
    subject: str
    relation: str
    obj: str
    prose: list[str] = field(default_factory=list)  # alternative wordings; {s} {o}


def render(facts: list[Fact], fmt: str, rng: random.Random) -> str:
    facts = list(facts); rng.shuffle(facts)
    if fmt == "prose":
        return " ".join(rng.choice(f.prose).format(s=f.subject, o=f.obj) for f in facts)
    rows = [{"subject": f.subject, "relation": f.relation, "object": f.obj} for f in facts]
    if fmt == "json":
        return json.dumps({"facts": rows}, separators=(",", ":"))
    if fmt == "table":
        return "subject | relation | object\n" + "\n".join(f"{r['subject']} | {r['relation']} | {r['object']}" for r in rows)
    if fmt == "kv":
        return "\n".join(f"{r['relation']}({r['subject']}, {r['object']})" for r in rows)
    raise ValueError(fmt)


def choice_question(text, options):
    return {"type": "choice", "instructions": text, "criteria": {o: o for o in options}}


def record(rid, state, questions, labels, meta, distributions=None):
    rec = {"id": rid, "state": state, "questions": questions, "labels": labels, "meta": meta}
    if distributions: rec["distributions"] = distributions
    return rec


# ---------------------------------------------------------------- kinship
KIN = ["parent", "child", "grandparent", "grandchild", "sibling", "aunt or uncle", "niece or nephew", "cousin", "unrelated"]


def kinship_world(rng, generations=3, width=2):
    names = rng.sample(NAMES, 20); people, parent = [], {}
    roots = [names.pop()]; people += roots; frontier = roots
    for _ in range(generations - 1):
        nxt = []
        for p in frontier:
            for _ in range(rng.randint(1, width)):
                if not names: break
                c = names.pop(); parent[c] = p; people.append(c); nxt.append(c)
        frontier = nxt
    if rng.random() < 0.5 and names:  # an unrelated person keeps "unrelated" honest
        people.append(names.pop())
    return {"people": people, "parent": parent}


def kin_relation(world, a, b):
    par = world["parent"]
    anc = lambda x: [x] + (anc(par[x]) if x in par else [])
    A, B = anc(a), anc(b)
    common = next((x for x in A if x in B), None)
    if common is None or a == b: return "unrelated"
    da, db = A.index(common), B.index(common)
    table = {(0, 1): "parent", (1, 0): "child", (0, 2): "grandparent", (2, 0): "grandchild", (1, 1): "sibling",
             (1, 2): "aunt or uncle", (2, 1): "niece or nephew", (2, 2): "cousin"}
    return table.get((da, db), "unrelated")


def kinship_facts(world):
    return [Fact(p, "parent_of", c, ["{s} is the parent of {o}.", "{o} is a child of {s}.", "{s} raised their child {o}."])
            for c, p in world["parent"].items()]


def kinship_example(rng, fmt, rid):
    # Pick the answer first, then a pair realising it, so deep relations are not starved.
    want = rng.choice(KIN)
    for _ in range(50):
        world = kinship_world(rng, generations=rng.choice([3, 3, 4]))
        pairs = [(x, y) for x in world["people"] for y in world["people"] if x != y and kin_relation(world, x, y) == want]
        if pairs: break
    a, b = rng.choice(pairs) if pairs else rng.sample(world["people"], 2)
    q = choice_question(f"What is {a} to {b}?", KIN)
    label = kin_relation(world, a, b)
    return record(rid, render(kinship_facts(world), fmt, rng), {"answer": q}, {"answer": label},
                  {"domain": "kinship", "format": fmt, "query": [a, b]}), world


# ---------------------------------------------------------------- temporal
TEMP = ["before", "after", "cannot be determined"]


def temporal_example(rng, fmt, rid, n=None):
    n = n or rng.randint(3, 7); ev = rng.sample(EVENTS, n)
    order = list(ev); rng.shuffle(order)
    # Reveal a random subset of adjacent-or-skip precedences; the rest stays unknown.
    edges = set()
    for i in range(n - 1):
        if rng.random() < 0.8: edges.add((order[i], order[i + 1]))
        if i + 2 < n and rng.random() < 0.2: edges.add((order[i], order[i + 2]))
    facts = [Fact(a, "before", b, ["The {s} happens before the {o}.", "The {o} happens after the {s}.",
                                    "The {s} precedes the {o}."]) for a, b in sorted(edges)]  # sorted: set order is hash-seeded
    reach = _closure(ev, edges)
    a, b = rng.sample(ev, 2)
    label = "before" if b in reach[a] else "after" if a in reach[b] else "cannot be determined"
    q = choice_question(f"Does the {a} happen before or after the {b}?", TEMP)
    return record(rid, render(facts, fmt, rng), {"answer": q}, {"answer": label},
                  {"domain": "temporal", "format": fmt, "query": [a, b], "events": n}), {"edges": edges, "events": ev}


def _closure(nodes, edges):
    succ = {x: {b for a, b in edges if a == x} for x in nodes}; reach = {}
    for x in nodes:
        seen, stack = set(), list(succ[x])
        while stack:
            y = stack.pop()
            if y not in seen: seen.add(y); stack += list(succ[y])
        reach[x] = seen
    return reach


# ---------------------------------------------------------------- dependency
def dependency_example(rng, fmt, rid, drop_edge=None, query=None):
    n = rng.randint(4, 9); svc = rng.sample(SERVICES, n); edges = set()
    for i in range(1, n):  # a DAG: each service may depend on earlier ones
        for j in rng.sample(range(i), min(i, rng.randint(1, 2))):
            edges.add((svc[i], svc[j]))  # (dependent, dependency)
    if drop_edge: edges.discard(drop_edge)
    facts = [Fact(a, "depends_on", b, ["{s} depends on {o}.", "{s} calls {o}.", "{o} is required by {s}."]) for a, b in sorted(edges)]
    reach = _closure(svc, edges)
    pairs = [(d, t) for d in svc for t in svc if d != t]
    hit = [(d, t) for d, t in pairs if d in reach[t]]; miss = [pt for pt in pairs if pt not in hit]  # pairs is list-ordered
    down, target = rng.choice(hit if hit and (rng.random() < 0.5 or not miss) else miss)
    if query: down, target = query  # counterfactual twins keep the base question
    label = target in {x for x in svc if down in reach[x]}
    q = {"type": "noul", "instructions": f"If {down} fails, does {target} fail?"}
    return record(rid, render(facts, fmt, rng), {"answer": q}, {"answer": label},
                  {"domain": "dependency", "format": fmt, "query": [down, target]}), {"edges": edges, "svc": svc}


# ---------------------------------------------------------------- probability
def probability_example(rng, fmt, rid):
    """Two or three boxes with known contents and priors; one ball drawn. Exact posterior over boxes."""
    k = rng.choice([2, 2, 3]); boxes = ["box " + x for x in rng.sample("ABCDEFG", k)]
    reds = {b: rng.randint(0, 5) for b in boxes}; blues = {b: rng.randint(1 if reds[b] == 0 else 0, 5) for b in boxes}
    weights = {b: rng.randint(1, 4) for b in boxes}; total_w = sum(weights.values())
    colour = rng.choice(["red", "blue"])
    like = {b: Fraction(reds[b] if colour == "red" else blues[b], reds[b] + blues[b]) for b in boxes}
    joint = {b: Fraction(weights[b], total_w) * like[b] for b in boxes}
    if sum(joint.values()) == 0: return probability_example(rng, fmt, rid)
    post = {b: float(joint[b] / sum(joint.values())) for b in boxes}
    facts = []
    for b in boxes:
        facts.append(Fact(b, "contains_red", str(reds[b]), ["The {s} holds {o} red balls."]))
        facts.append(Fact(b, "contains_blue", str(blues[b]), ["The {s} holds {o} blue balls."]))
        facts.append(Fact(b, "chosen_weight", f"{weights[b]}/{total_w}", ["The {s} is picked with probability {o}."]))
    facts.append(Fact("draw", "colour", colour, ["One box was picked and a {o} ball was drawn from it."]))
    q = choice_question("Which box was the ball drawn from?", boxes)
    return record(rid, render(facts, fmt, rng), {"answer": q}, {"answer": max(post, key=post.get)},
                  {"domain": "probability", "format": fmt, "posterior": post}, {"answer": post}), None


# ---------------------------------------------------------------- rules
def rules_example(rng, fmt, rid, flip=None):
    age_min = rng.choice([18, 21, 25]); income_min = rng.choice([20, 30, 40])
    person = rng.choice(NAMES)
    attrs = {"age": rng.randint(15, 40), "income": rng.randint(10, 60),
             "prior_default": rng.random() < 0.3, "veteran": rng.random() < 0.2}
    if flip: attrs[flip[0]] = flip[1]
    eligible = attrs["age"] >= age_min and not attrs["prior_default"] and (attrs["income"] >= income_min or attrs["veteran"])
    policy = (f"Policy: applicants aged {age_min} or older are eligible if their income is at least {income_min}k, "
              f"unless they have a prior default. Veterans are exempt from the income requirement.")
    facts = [Fact(person, "age", str(attrs["age"]), ["{s} is {o} years old."]),
             Fact(person, "income_k", str(attrs["income"]), ["{s} earns {o}k a year."]),
             Fact(person, "prior_default", "yes" if attrs["prior_default"] else "no", ["{s} has a prior default: {o}."]),
             Fact(person, "veteran", "yes" if attrs["veteran"] else "no", ["Is {s} a veteran? {o}."])]
    state = policy + ("\n" if fmt != "prose" else " ") + render(facts, fmt, rng)
    q = {"type": "noul", "instructions": f"Is {person} eligible?"}
    return record(rid, state, {"answer": q}, {"answer": eligible},
                  {"domain": "rules", "format": fmt, "attrs": attrs, "age_min": age_min, "income_min": income_min}), attrs


DOMAINS = {"kinship": kinship_example, "temporal": temporal_example, "dependency": dependency_example,
           "probability": probability_example, "rules": rules_example}


def generate(n, seed, domains=tuple(DOMAINS), formats=FORMATS, prefix="w"):
    rng = random.Random(seed); out = []
    for i in range(n):
        dom = domains[i % len(domains)]; fmt = rng.choice(formats)
        rec, _ = DOMAINS[dom](rng, fmt, f"{prefix}-{seed}-{i}")
        out.append(rec)
    return out


# ---------------------------------------------------------------- information gathering
PARTS = ["pump", "valve", "sensor", "fan", "relay", "motor", "filter", "board", "cable", "battery"]


def infogather_example(rng, fmt, rid, cost=None):
    """Repair now or pay for a diagnostic first? The label maximises exact expected utility.

    Utility: correct repair 1, wrong repair 0, the test costs `cost`. EU(repair h) = P(h | evidence);
    EU(test) = -cost + sum_o P(o) max_h P(h | evidence, o). Ties go to acting (no wasted cost).
    """
    k = rng.choice([2, 2, 3]); parts = rng.sample(PARTS, k)
    prior = [rng.randint(1, 6) for _ in parts]; z = sum(prior); prior = [Fraction(w, z) for w in prior]
    # One diagnostic, positive with a part-specific probability.
    sens = [Fraction(rng.choice([1, 2, 3, 5, 7, 8, 9]), 10) for _ in parts]
    cost = Fraction(rng.choice([1, 2, 3, 5, 8, 12, 20]), 100) if cost is None else cost
    eu_act = {f"repair the {p}": prior[i] for i, p in enumerate(parts)}
    p_pos = sum(prior[i] * sens[i] for i in range(k)); ev = Fraction(0)
    for pos, po in ((True, p_pos), (False, 1 - p_pos)):
        if po == 0: continue
        post = [prior[i] * (sens[i] if pos else 1 - sens[i]) / po for i in range(k)]
        ev += po * max(post)
    eu = dict(eu_act); eu["run the diagnostic"] = ev - cost
    best_act = max(eu_act.values())
    label = "run the diagnostic" if eu["run the diagnostic"] > best_act else max(eu_act, key=eu_act.get)
    facts = []
    for i, p in enumerate(parts):
        facts.append(Fact(p, "fault_prior", f"{prior[i].numerator}/{prior[i].denominator}",
                          ["The fault is in the {s} with probability {o}."]))
        facts.append(Fact(p, "test_positive_rate", f"{sens[i].numerator}/{sens[i].denominator}",
                          ["If the {s} is faulty, the diagnostic reads positive with probability {o}."]))
    facts.append(Fact("diagnostic", "cost", f"{cost.numerator}/{cost.denominator}",
                      ["Running the diagnostic costs {o} (a correct repair is worth 1, a wrong one 0)."]))
    options = list(eu_act) + ["run the diagnostic"]
    q = choice_question("What should you do next?", options)
    return record(rid, render(facts, fmt, rng), {"answer": q}, {"answer": label},
                  {"domain": "infogather", "format": fmt, "eu": {o: float(v) for o, v in eu.items()},
                   "cost": float(cost), "value_of_information": float(ev - best_act)}), {"cost": cost}


DOMAINS["infogather"] = infogather_example


# ---------------------------------------------------------------- causal (confounded SCM)
CAUSAL_NAMES = [("heatwave", "ice cream sales", "sunburn"), ("promotion", "ad spend", "revenue"),
                ("old hardware", "restarts", "outages"), ("winter", "heating", "illness"),
                ("rush hour", "detours", "delays"), ("exam season", "coffee", "stress"),
                ("high traffic", "caching", "latency"), ("drought", "irrigation", "yield")]
PROBS = [Fraction(k, 10) for k in range(1, 10)]


def causal_example(rng, fmt, rid, kind=None, override=None):
    """Z confounds X and Y (Z -> X, Z -> Y, X -> Y). Ask P(Y | do(X=x)) or P(Y | X=x) exactly.

    The two differ whenever Z shifts both X and Y; the model must tell intervening from observing.
    `override` = (table, key, value) changes one CPT entry for counterfactual twins.
    """
    z, x, y = rng.choice(CAUSAL_NAMES)
    pz = rng.choice(PROBS)

    def apart(gap):  # two probabilities at least `gap` apart, in random order: a real confounder
        while True:
            u, v = rng.choice(PROBS), rng.choice(PROBS)
            if abs(u - v) >= gap: return u, v
    hi, lo = apart(Fraction(4, 10)); px = {1: hi, 0: lo}                # P(X=1 | Z)
    py = {}
    for a in (0, 1):
        u, v = apart(Fraction(3, 10)); py[(a, 1)], py[(a, 0)] = u, v    # P(Y=1 | X=a, Z=b)
    if override:
        table, key, value = override
        {"pz": None, "px": px, "py": py}[table][key] = value if table != "pz" else None
        if table == "pz": pz = value
    kind = kind or rng.choice(["do", "see"]); xv = rng.choice([0, 1])
    pzv = {1: pz, 0: 1 - pz}
    if kind == "do":
        p = sum(pzv[b] * py[(xv, b)] for b in (0, 1))
        text = f"If we {'force' if xv else 'prevent'} {x}, how likely is {y}?"
    else:
        pxz = {b: (px[b] if xv else 1 - px[b]) for b in (0, 1)}
        joint = {b: pzv[b] * pxz[b] for b in (0, 1)}; zsum = sum(joint.values())
        p = sum(joint[b] / zsum * py[(xv, b)] for b in (0, 1))
        text = f"We observe {'' if xv else 'no '}{x}. How likely is {y}?"
    f = lambda q: f"{q.numerator}/{q.denominator}"
    on = lambda v, s: s if v else f"no {s}"
    facts = [Fact(z, "base_rate", f(pz), ["{s} occurs with probability {o}."])]
    for b in (1, 0):
        facts.append(Fact(x, f"given_{on(b, z)}", f(px[b]), ["With " + on(b, z) + ", {s} happens with probability {o}."]))
        for a in (1, 0):
            facts.append(Fact(y, f"given_{on(a, x)}_and_{on(b, z)}", f(py[(a, b)]),
                              ["With " + on(a, x) + " and " + on(b, z) + ", {s} happens with probability {o}."]))
    facts.append(Fact(x, "causes", y, ["{s} directly affects {o}; " + z + " affects both."]))
    pf = float(p)
    q = {"type": "noul", "instructions": text}
    return record(rid, render(facts, fmt, rng), {"answer": q}, {"answer": pf >= 0.5},
                  {"domain": "causal", "format": fmt, "kind": kind, "x": xv, "p_exact": pf},
                  {"answer": {"true": pf, "false": 1 - pf}}), {"pz": pz, "px": px, "py": py}


DOMAINS["causal"] = causal_example
