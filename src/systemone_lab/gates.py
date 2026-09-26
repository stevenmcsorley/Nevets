"""Fast in-training dev gates (Rule 4: evaluate at least every 500 optimiser steps).

Batched inference over the full one-hop regression sets that a run must not break, plus a fixed
chain subsample for learning curves. Metric definitions match scripts/run_gates.py:
  vertical_role_swap / heldout_role_swap: both the forward and the swapped question correct
  name_pairs / vertical_name_pairs: numbered and renamed renderings both correct
  option_order_stable: fraction of chain questions whose answer is unchanged by an option shuffle
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data.spatial_worlds import INVERSE_REL
from .training import _pad_batch, _packed, option_signs

QUESTION = r"What is the spatial relation of (.+) to (.+)\?"
KILL_GATES = ("vertical_role_swap", "heldout_role_swap", "name_pairs", "vertical_name_pairs", "option_order_stable")


@torch.inference_mode()
def predict_batch(model, tokenizer, records, device, batch_size=64):
    """Probabilities for every question of every record, batched; equals predict_record per record."""
    was_training = model.training; model.eval(); device = torch.device(device); out = []
    if not model.supports_batched_decisions():
        from .eval import predict_record
        res = [predict_record(model, tokenizer, r, device) for r in records]
        model.train(was_training); return res
    for s in range(0, len(records), batch_size):
        chunk = records[s:s + batch_size]
        packed = [_packed(tokenizer, r, model.isolated_options) for r in chunk]
        ids, pos, masks = _pad_batch(model, tokenizer, packed, device)
        h = model.hidden(ids, pos, masks)
        items = [(i, l) for i, x in enumerate(packed) for l in x.layouts]
        K = max(len(l.option_keys) for _, l in items); N = len(items)
        opt_pos = torch.zeros((N, K), dtype=torch.long); opt_mask = torch.zeros((N, K), dtype=torch.bool)
        rows, bb, bp, bw = [], [], [], []
        for n, (i, l) in enumerate(items):
            k = len(l.option_keys); opt_pos[n, :k] = torch.tensor(l.option_end_positions); opt_mask[n, :k] = True
            first, second = l.query_entity_state_positions or ((), ())
            if first and second:
                for group, sign in ((first, 1.0), (second, -1.0)):
                    rows += [n] * len(group); bb += [i] * len(group); bp += list(group); bw += [sign / len(group)] * len(group)
        mv = lambda x: x.to(device)
        bind = (mv(torch.tensor(rows, dtype=torch.long)), mv(torch.tensor(bb, dtype=torch.long)),
                mv(torch.tensor(bp, dtype=torch.long)), mv(torch.tensor(bw, dtype=torch.float32)))
        signs, spatial = option_signs([l for _, l in items], K)
        logits, _, _ = model.decision_logits_batch(h, mv(torch.tensor([i for i, _ in items])),
                                                   mv(torch.tensor([l.decide_position for _, l in items])),
                                                   mv(opt_pos), mv(opt_mask), bind, mv(signs), mv(spatial))
        probs = F.softmax(logits.float(), -1).cpu()
        res = [dict() for _ in chunk]
        for n, (i, l) in enumerate(items):
            res[i][l.key] = {k: float(probs[n, j]) for j, k in enumerate(l.option_keys)}
        out += res
    model.train(was_training)
    return out


def _swap(rec):
    s = json.loads(json.dumps(rec)); q = s["questions"]["spatial"]
    a, b = re.fullmatch(QUESTION, q["instructions"]).groups()
    q["instructions"] = f"What is the spatial relation of {b} to {a}?"; s["labels"]["spatial"] = INVERSE_REL[rec["labels"]["spatial"]]
    return s


class FastGates:
    """Build the gate records once; evaluate a model in a few seconds of batched inference."""

    def __init__(self, root=".", chain_sample=600, seed=0):
        R = lambda p: json.loads((Path(root) / p).read_text(encoding="utf-8"))
        self.jobs = {}
        for name, path in (("vertical_role_swap", "reports/paired_onehop/vertical_probe_pairs.json"),
                           ("heldout_role_swap", "reports/role_swap_onehop/name_pairs.json")):
            fwd = [p[v] for p in R(path) for v in ("numbered", "renamed")]
            self.jobs[name] = ("pairs", fwd, [_swap(r) for r in fwd])
        for name, path in (("name_pairs", "reports/paired_onehop/name_pairs.json"),
                           ("vertical_name_pairs", "reports/paired_onehop/vertical_probe_pairs.json")):
            ps = R(path); self.jobs[name] = ("pairs", [p["numbered"] for p in ps], [p["renamed"] for p in ps])
        chains = [json.loads(l) for l in open(Path(root) / "reports/chain/eval_hops.jsonl", encoding="utf-8")]
        rng = random.Random(seed); sample = rng.sample(chains, min(chain_sample, len(chains)))
        shuffled = []
        for r in sample:
            s = json.loads(json.dumps(r)); items = list(s["questions"]["spatial"]["criteria"].items())
            rng.shuffle(items); s["questions"]["spatial"]["criteria"] = dict(items); shuffled.append(s)
        self.chains, self.shuffled = sample, shuffled

    def evaluate(self, model, tokenizer, device):
        top = lambda p: max(p["spatial"], key=p["spatial"].get)
        out = {}
        for name, (_, a, b) in self.jobs.items():
            pa, pb = predict_batch(model, tokenizer, a, device), predict_batch(model, tokenizer, b, device)
            out[name] = sum(top(x) == r["labels"]["spatial"] and top(y) == s["labels"]["spatial"]
                            for x, y, r, s in zip(pa, pb, a, b)) / len(a)
        pc, ps = predict_batch(model, tokenizer, self.chains, device), predict_batch(model, tokenizer, self.shuffled, device)
        out["option_order_stable"] = sum(top(x) == top(y) for x, y in zip(pc, ps)) / len(pc)
        by = defaultdict(list)
        for p, r in zip(pc, self.chains): by[r["meta"]["hops"]].append(top(p) == r["labels"]["spatial"])
        out["chain_acc"] = sum(sum(v) for v in by.values()) / len(pc)
        out["chain_by_hops"] = {h: sum(v) / len(v) for h, v in sorted(by.items())}
        return out


def regressions(baseline, current, tolerance=0.02):
    """Kill-gates that fell more than `tolerance` below the parent baseline."""
    return {g: (baseline[g], current[g]) for g in KILL_GATES if current[g] < baseline[g] - tolerance}
