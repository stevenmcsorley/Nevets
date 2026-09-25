"""Interactive benchmark: the Treasure Hunt game played headlessly by a checkpoint.

Reproduces ui/game.html / site/treasure.html: a 9x9 grid, a robot, a treasure, 4 landmarks and 6 rocks.
Each turn the world is described only in words, the model answers "What is the spatial relation of
<treasure> to <robot>?", and the robot steps along the model's most likely legal direction. In
landmark mode the treasure is given only relative to a landmark whose two facts determine the answer
(sign relations cannot compose when they point opposite ways on an axis).

Reports per mode: decision accuracy, accuracy on two-fact turns, win rate within 30 steps, path
efficiency (shortest / taken on wins), mean confidence and correct-when-confident (>= 0.9).
"""
import argparse
import json
import random
from pathlib import Path

from systemone_lab.eval import predict_record
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint, pick_device

N, STEP_LIMIT = 9, 30
LABELS = ["upper-left", "above", "upper-right", "left", "overlap", "right", "lower-left", "below", "lower-right"]
VEC = {"upper-left": (-1, 1), "above": (0, 1), "upper-right": (1, 1), "left": (-1, 0), "overlap": (0, 0),
       "right": (1, 0), "lower-left": (-1, -1), "below": (0, -1), "lower-right": (1, -1)}
TEMPLATES = {"left": ["{a} is left of {b}.", "{a} sits west of {b}.", "{b} is to the right of {a}."],
             "right": ["{a} is right of {b}.", "{a} sits east of {b}.", "{b} is to the left of {a}."],
             "above": ["{a} is above {b}.", "{a} sits north of {b}.", "{b} is below {a}."],
             "below": ["{a} is below {b}.", "{a} sits south of {b}.", "{b} is above {a}."],
             "upper-left": ["{a} is upper-left of {b}.", "{a} is northwest of {b}."],
             "upper-right": ["{a} is upper-right of {b}.", "{a} is northeast of {b}."],
             "lower-left": ["{a} is lower-left of {b}.", "{a} is southwest of {b}."],
             "lower-right": ["{a} is lower-right of {b}.", "{a} is southeast of {b}."],
             "overlap": ["{a} occupies the same position as {b}."]}
sgn = lambda v: (v > 0) - (v < 0)
rel = lambda p, q: next(l for l in LABELS if VEC[l] == (sgn(p[0] - q[0]), sgn(p[1] - q[1])))


def play(model, tok, device, mode, noise, rng):
    letters = rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", 12)
    cells = [(x, y) for x in range(N) for y in range(N)]; rng.shuffle(cells)
    robot, treasure = cells.pop(), cells.pop()
    while max(abs(robot[0] - treasure[0]), abs(robot[1] - treasure[1])) < 4:
        robot, treasure = cells.pop(), cells.pop()
    P = {letters[0]: robot, letters[1]: treasure}; R, T = letters[0], letters[1]
    lands = letters[2:6]; rocks = letters[6:12]
    for n in lands + rocks: P[n] = cells.pop()
    say = lambda a, b: rng.choice(TEMPLATES[rel(P[a], P[b])]).format(a=a, b=b)
    optimal = max(abs(robot[0] - treasure[0]), abs(robot[1] - treasure[1])); turns = []
    for step in range(1, STEP_LIMIT + 1):
        via = None
        if mode == "landmark":
            good = [l for l in lands if P[l] != P[R] and P[l] != P[T]
                    and all(a * b >= 0 for a, b in zip(VEC[rel(P[T], P[l])], VEC[rel(P[l], P[R])]))]
            via = rng.choice(good) if good else None
        facts = ([say(T, via) if rng.random() < .5 else say(via, T), say(via, R) if rng.random() < .5 else say(R, via)]
                 if via else [say(T, R) if rng.random() < .5 else say(R, T)])
        others = rocks + [l for l in lands if l != via]
        for _ in range(noise):
            a = rng.choice(others); b = rng.choice([o for o in others + [R] if o != a])
            if P[a] != P[b]: facts.append(say(a, b))
        rng.shuffle(facts)
        q = {"move": {"type": "choice", "instructions": f"What is the spatial relation of {T} to {R}?", "criteria": {l: l for l in LABELS}}}
        p = predict_record(model, tok, {"state": " ".join(facts), "questions": q}, device)["move"]
        ranked = sorted(p, key=p.get, reverse=True); truth = rel(P[T], P[R])
        turns.append({"two_fact": via is not None, "correct": ranked[0] == truth, "conf": p[ranked[0]]})
        for label in ranked:
            if label == "overlap": continue
            nxt = (P[R][0] + VEC[label][0], P[R][1] + VEC[label][1])
            if 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in [P[r] for r in rocks]:
                P[R] = nxt; break
        if P[R] == P[T]: return {"won": True, "steps": step, "optimal": optimal, "turns": turns}
    return {"won": False, "steps": STEP_LIMIT, "optimal": optimal, "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=60); ap.add_argument("--noise", type=int, default=2)
    ap.add_argument("--iters", type=int); ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args(); device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    if a.iters: model.iters = a.iters
    out = {"ckpt": a.ckpt, "iters": getattr(model, "iters", None), "episodes": a.episodes, "noise": a.noise, "seed": a.seed}
    for mode in ("direct", "landmark"):
        rng = random.Random(a.seed); games = [play(model, tok, device, mode, a.noise, rng) for _ in range(a.episodes)]
        turns = [t for g in games for t in g["turns"]]; two = [t for t in turns if t["two_fact"]]; sure = [t for t in turns if t["conf"] >= 0.9]
        wins = [g for g in games if g["won"]]
        out[mode] = {"win_rate": len(wins) / len(games), "decision_accuracy": sum(t["correct"] for t in turns) / len(turns),
                     "two_fact_accuracy": sum(t["correct"] for t in two) / len(two) if two else None,
                     "path_efficiency": sum(g["optimal"] for g in wins) / sum(g["steps"] for g in wins) if wins else None,
                     "mean_confidence": sum(t["conf"] for t in turns) / len(turns),
                     "accuracy_when_conf_ge_0.9": sum(t["correct"] for t in sure) / len(sure) if sure else None,
                     "turns": len(turns)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[m].items()} for m in ("direct", "landmark")}))


if __name__ == "__main__":
    main()
