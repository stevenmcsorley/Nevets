"""Per-iteration probe of a looped checkpoint: does the recurrent state converge or drift past training depth?

Runs prelude -> core iterations t = 1..K_max manually (identical to SystemOneModel.hidden for a given K;
tested) and, at every t, reads the full model output for that depth (coda + norm + decision head):
  - drift_state: mean ||h_t - h_{t-1}|| / ||h_{t-1}|| over state tokens (0 = fixed point)
  - early-exit accuracy on dev chains, trained hops (1-6) and unseen hops (7-10) separately
  - entity_cosine: mean pairwise cosine between different objects' mention states (rising = oversmoothing)
  - coord_err (if the model has a coordinate head): mean |predicted - true| displacement between the
    two queried objects, in grid units, using aux-coord ground truth from the eval generator
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from systemone_lab.data.io import read_jsonl
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import _pad_batch, _packed, load_checkpoint, option_signs, pick_device


def iterate_states(model, ids, pos, masks, k_max):
    """Yield (t, full-depth output for K=t, raw core state h_t) for t = 1..k_max."""
    arch = model.cfg.arch; p, c = arch["prelude"], arch["core"]
    x = model.embed(ids)
    for block in model.blocks[:p]: x = block(x, pos, masks)
    inject, h = x, torch.zeros_like(x)
    for t in range(1, k_max + 1):
        h = h + inject
        for block in model.blocks[p:p + c]: h = block(h, pos, masks)
        y = h
        for block in model.blocks[p + c:]: y = block(y, pos, masks)
        yield t, model.norm(y), h


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k-max", type=int, default=12); ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--chains", default="reports/chain/eval_hops_aux.jsonl"); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args(); device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    assert (model.cfg.arch or {}).get("type") == "looped", "probe needs a looped checkpoint"
    recs = list(read_jsonl(a.chains)); random.Random(0).shuffle(recs); recs = recs[:a.n]
    acc = defaultdict(lambda: defaultdict(list)); drift = defaultdict(list); ecos = defaultdict(list); cerr = defaultdict(list)
    for s in range(0, len(recs), a.batch):
        chunk = recs[s:s + a.batch]
        packed = [_packed(tok, r, model.isolated_options) for r in chunk]
        ids, pos, masks = _pad_batch(model, tok, packed, device)
        items = [(i, l) for i, x in enumerate(packed) for l in x.layouts]
        K = max(len(l.option_keys) for _, l in items); N = len(items)
        opt_pos = torch.zeros((N, K), dtype=torch.long); opt_mask = torch.zeros((N, K), dtype=torch.bool)
        rows, bb, bp, bw = [], [], [], []
        for n, (i, l) in enumerate(items):
            k = len(l.option_keys); opt_pos[n, :k] = torch.tensor(l.option_end_positions); opt_mask[n, :k] = True
            first, second = l.query_entity_state_positions or ((), ())
            if first and second:
                for g, sgn in ((first, 1.0), (second, -1.0)):
                    rows += [n] * len(g); bb += [i] * len(g); bp += list(g); bw += [sgn / len(g)] * len(g)
        mv = lambda x: x.to(device)
        bind = (mv(torch.tensor(rows, dtype=torch.long)), mv(torch.tensor(bb, dtype=torch.long)),
                mv(torch.tensor(bp, dtype=torch.long)), mv(torch.tensor(bw, dtype=torch.float32)))
        signs, spatial = option_signs([l for _, l in items], K)
        state_mask = torch.zeros(ids.shape, dtype=torch.bool, device=device)
        for i, x in enumerate(packed): state_mask[i, :int((x.branch_ids == 0).sum())] = True
        prev = None
        for t, hidden, h in iterate_states(model, ids, pos, masks, a.k_max):
            logits, _, _ = model.decision_logits_batch(hidden, mv(torch.tensor([i for i, _ in items])),
                                                       mv(torch.tensor([l.decide_position for _, l in items])),
                                                       mv(opt_pos), mv(opt_mask), bind, mv(signs), mv(spatial))
            pred = logits.argmax(-1).cpu()
            for n, (i, l) in enumerate(items):
                r = chunk[i]; hops = r["meta"]["hops"]
                acc[t]["trained" if hops <= 6 else "unseen"].append(l.option_keys[int(pred[n])] == r["labels"]["spatial"])
            hf = h.float()
            if prev is not None:
                num = (hf - prev).norm(dim=-1)[state_mask]; den = prev.norm(dim=-1)[state_mask].clamp_min(1e-6)
                drift[t].append((num / den).mean().item())
            prev = hf
            # entity identity: pairwise cosine between different objects' mention states (final-depth output)
            for i, r in enumerate(chunk):
                pts = [p for grp in (packed[i].layouts[0].query_entity_state_positions or ()) for p in grp[:1]]
                if len(pts) == 2:
                    ecos[t].append(F.cosine_similarity(hidden[i, pts[0]].float(), hidden[i, pts[1]].float(), dim=0).item())
            if model.coord_head is not None and bind[0].numel():
                bound = torch.zeros(N, hidden.size(-1), device=device); bound.index_add_(0, bind[0], hidden[bind[1], bind[2]].float() * bind[3][:, None])
                d = (bound @ model.coord_head.weight.float().t()) * 4  # back to grid units
                for n, (i, l) in enumerate(items):
                    r = chunk[i]; a_, b_ = r["meta"].get("query", [None, None])
                    coords = r["meta"].get("aux_coords") or {}
                    q = r["questions"]["spatial"]["instructions"].split(" of ", 1)[1].rstrip("?").split(" to ")
                    if len(q) == 2 and q[0] in coords and q[1] in coords:
                        true = torch.tensor([coords[q[0]][0] - coords[q[1]][0], coords[q[0]][1] - coords[q[1]][1]], dtype=torch.float32, device=device)
                        cerr[t].append((d[n] - true).abs().mean().item())
    out = {"ckpt": a.ckpt, "k_max": a.k_max, "n": len(recs), "trained_iters_max": 6, "by_iter": {}}
    for t in range(1, a.k_max + 1):
        out["by_iter"][t] = {"acc_trained_hops": sum(acc[t]["trained"]) / max(1, len(acc[t]["trained"])),
                             "acc_unseen_hops": sum(acc[t]["unseen"]) / max(1, len(acc[t]["unseen"])),
                             "drift_state": sum(drift[t]) / len(drift[t]) if drift[t] else None,
                             "entity_cosine": sum(ecos[t]) / len(ecos[t]) if ecos[t] else None,
                             "coord_err_grid": sum(cerr[t]) / len(cerr[t]) if cerr[t] else None}
        b = out["by_iter"][t]
        print(f"t={t:2d} acc 1-6: {b['acc_trained_hops']:.3f}  7-10: {b['acc_unseen_hops']:.3f}  drift {b['drift_state'] if b['drift_state'] is None else round(b['drift_state'],4)}"
              f"  ent-cos {b['entity_cosine'] and round(b['entity_cosine'],3)}  coord-err {b['coord_err_grid'] and round(b['coord_err_grid'],3)}", flush=True)
    d = [out["by_iter"][t]["drift_state"] for t in range(2, a.k_max + 1)]
    out["verdict"] = ("converging" if d[-1] < 0.5 * d[4] else "drifting" if d[-1] > 1.2 * d[4] else "steady") if len(d) > 5 else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2)); print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
