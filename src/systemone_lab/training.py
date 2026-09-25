from __future__ import annotations
import json, math, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from .formatting import pack_request, branch_attention_mask, entity_state_positions, _text


def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def choose_dtype(device):
    if device.type == "cuda":
        major,_ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def tokenizer_metadata(tokenizer_path):
    import hashlib
    from .tokenizer import LabTokenizer
    tok = LabTokenizer(tokenizer_path)
    return {"tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
            "vocab_size": tok.vocab_size, "special_token_ids": tok.ids}


def save_checkpoint(path, model, cfg, tokenizer_path, step, metrics=None):
    import hashlib
    path = Path(path)
    if path.resolve() == Path("checkpoints/s1-35m-pretrain.pt").resolve():
        raise ValueError("protected pretrained LM checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = tokenizer_metadata(tokenizer_path)
    metadata["model_config_hash"] = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "tokenizer": str(tokenizer_path),
                "step": step, "training_step": step, "metrics": metrics or {}, **metadata}, path)


def load_checkpoint(path, model_cls, device="cpu", tokenizer_path=None, allow_tokenizer_mismatch=False):
    import hashlib
    from .config import ModelConfig
    ck = torch.load(path, map_location=device)
    actual = tokenizer_metadata(tokenizer_path or ck["tokenizer"])
    if not allow_tokenizer_mismatch:
        if "tokenizer_sha256" not in ck:
            raise ValueError("legacy checkpoint has no tokenizer hash; explicitly allow unverified tokenizer")
        for key, value in actual.items():
            if ck.get(key) != value:
                raise ValueError(f"checkpoint tokenizer mismatch: {key}")
    if "model_config_hash" in ck:
        expected = hashlib.sha256(json.dumps(ck["config"], sort_keys=True).encode()).hexdigest()
        if ck["model_config_hash"] != expected:
            raise ValueError("checkpoint model config hash mismatch")
    cfg = ModelConfig(**ck["config"])
    if cfg.vocab_size != actual["vocab_size"]:
        raise ValueError("checkpoint vocabulary size mismatch")
    m = model_cls(cfg)
    # A child can inherit modules its own decision head no longer uses (e.g. ptr_bind from a spatial
    # parent fine-tuned in plain decide mode). Recreate them so the checkpoint loads strictly; unused
    # modules never affect outputs.
    if "ptr_bind.weight" in ck["model"] and getattr(m, "ptr_bind", None) is None:
        m.ptr_bind = torch.nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
    m.load_state_dict(ck["model"])
    return m, ck


def _target(layout, rec):
    value = rec["labels"][layout.key]
    target_key = str(value)
    if layout.qtype == "noul" and isinstance(value, bool):
        target_key = str(value).lower()
    if target_key not in layout.option_keys:
        raise ValueError(f"label {target_key!r} missing from candidates {layout.option_keys}")
    target = layout.option_keys.index(target_key)
    assert 0 <= target < len(layout.option_keys)
    assert layout.option_keys[target] == target_key
    return target


def decision_loss(model, tokenizer, rec, device):
    return decision_batch_loss(model, tokenizer, [rec], device)

def _pad_batch(model, tokenizer, packed, device):
    """Pad packed requests on the CPU and move them to the device in one transfer."""
    tmax = max(x.input_ids.numel() for x in packed); bsz = len(packed)
    ids = torch.full((bsz, tmax), tokenizer.id("<pad>"), dtype=torch.long)
    pos = torch.zeros((bsz, tmax), dtype=torch.long)
    branch = torch.full((bsz, tmax), -1, dtype=torch.long)  # -1 marks padding
    opts = torch.zeros((bsz, tmax), dtype=torch.long)
    for i, x in enumerate(packed):
        t = x.input_ids.numel()
        ids[i, :t] = x.input_ids; pos[i, :t] = x.position_ids; branch[i, :t] = x.branch_ids
        if x.option_ids is not None: opts[i, :t] = x.option_ids
    nb = device.type == "cuda"
    branch = branch.to(device, non_blocking=nb)  # build the [B,T,T] mask on the device from [B,T] ids
    opts = opts.to(device, non_blocking=nb) if model.isolated_options else None
    valid = branch >= 0
    masks = branch_attention_mask(branch.clamp(min=0), model.bidirectional_state, opts) & valid[:, None, :] & valid[:, :, None]
    # Keep padding query rows numerically well-defined. They never contribute to loss.
    masks |= torch.diag_embed(~valid)
    return ids.to(device, non_blocking=nb), pos.to(device, non_blocking=nb), masks


def _packed(tokenizer, rec, isolate_options=False):
    """Pack a record once and reuse it; records are sampled many times per run."""
    key = "_packed_iso" if isolate_options else "_packed"
    cached = rec.get(key)
    if cached is None:
        cached = pack_request(tokenizer, rec["state"], rec["questions"], isolate_options=isolate_options)
        if isinstance(rec, dict): rec[key] = cached
    return cached


def _aux_targets(tokenizer, rec):
    """Cached (state position, [dx, dy], hop distance) for every mention of a supervised object."""
    cached = rec.get("_aux3")
    if cached is None:
        meta = rec.get("meta", {}); coords = meta.get("aux_coords") or {}; dist = meta.get("aux_dist") or {}
        cached = []
        if coords and dist:
            text = _text(rec["state"]); _, offsets = tokenizer.encode_with_offsets(text)
            for name, xy in coords.items():
                if name in dist:
                    cached += [(p, xy, dist[name]) for p in entity_state_positions(text, offsets, name, 2)]
        if isinstance(rec, dict): rec["_aux3"] = cached
    return cached


def _target_distribution(layout, rec):
    dist = rec.get("distributions", {}).get(layout.key)
    if dist:
        p = torch.tensor([float(dist.get(k, 0.0)) for k in layout.option_keys])
        if not torch.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
            raise ValueError("invalid target distribution")
        return p / p.sum()
    p = torch.zeros(len(layout.option_keys)); p[_target(layout, rec)] = 1.0
    return p


def decision_batch_loss(model, tokenizer, records, device, diagnostics=None, force_legacy=False):
    """Mean decision loss over every question in a batch of packed requests."""
    device = torch.device(device)
    packed = [_packed(tokenizer, r, model.isolated_options) for r in records]
    ids, pos, masks = _pad_batch(model, tokenizer, packed, device)
    h = model.hidden(ids, pos, masks)
    items = [(i, layout, rec) for i, (x, rec) in enumerate(zip(packed, records)) for layout in x.layouts]
    move = lambda x: x.to(device, non_blocking=device.type == "cuda")
    if model.supports_batched_decisions() and not force_legacy:
        K = max(len(l.option_keys) for _, l, _ in items); N = len(items)
        b_idx = torch.tensor([i for i, _, _ in items]); decide = torch.tensor([l.decide_position for _, l, _ in items])
        opt_pos = torch.zeros((N, K), dtype=torch.long); opt_mask = torch.zeros((N, K), dtype=torch.bool)
        target = torch.zeros((N, K)); rows, bb, bp, bw = [], [], [], []
        for n, (i, l, rec) in enumerate(items):
            k = len(l.option_keys); opt_pos[n, :k] = torch.tensor(l.option_end_positions); opt_mask[n, :k] = True
            target[n, :k] = _target_distribution(l, rec)
            first, second = l.query_entity_state_positions or ((), ())
            if first and second:
                for group, sign in ((first, 1.0), (second, -1.0)):
                    rows += [n] * len(group); bb += [i] * len(group); bp += list(group); bw += [sign / len(group)] * len(group)
        bind = (move(torch.tensor(rows, dtype=torch.long)), move(torch.tensor(bb, dtype=torch.long)),
                move(torch.tensor(bp, dtype=torch.long)), move(torch.tensor(bw, dtype=torch.float32)))
        logits, q, k = model.decision_logits_batch(h, move(b_idx), move(decide), move(opt_pos), move(opt_mask), bind)
        target, opt_mask = move(target), move(opt_mask)
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        scores = {"q": q.detach(), "k": k.detach(), "logits": logits.detach(), "mask": opt_mask}
    else:
        scores, losses = [], []
        for i, layout, rec in items:
            logits = model.decision_logits(h[i], layout.decide_position, layout.option_end_positions,
                                           scores, layout.query_entity_token_ids,
                                           layout.query_entity_state_positions)
            p = move(_target_distribution(layout, rec))
            losses.append(-(p * F.log_softmax(logits.float(), dim=-1)).sum())
        loss = torch.stack(losses).float().mean()
    cw = float((model.cfg.arch or {}).get("converge_weight", 0))
    if cw > 0 and getattr(model, "fixed_point_delta", None) is not None:
        loss = loss + cw * model.fixed_point_delta
        if diagnostics is not None:
            diagnostics["fixed_point_delta"] = model.fixed_point_delta.detach().float().item()
    hint_w = float((model.cfg.arch or {}).get("hint_weight", 0))
    if hint_w > 0 and getattr(model, "iter_states", None):
        hops_per_iter = float((model.cfg.arch or {}).get("hint_hops_per_iter", 1))
        gb, gp, goal, gd = [], [], [], []
        for i, rec in enumerate(records):
            for p, xy, d in _aux_targets(tokenizer, rec):
                gb.append(i); gp.append(p); goal.append(xy); gd.append(d)
        if gb:
            gb_t, gp_t = move(torch.tensor(gb)), move(torch.tensor(gp))
            goal_t = move(torch.tensor(goal, dtype=torch.float32)) / 4; dist_t = move(torch.tensor(gd, dtype=torch.float32))
            terms = []
            for t_idx, hs in enumerate(model.iter_states, start=1):
                sel = dist_t <= t_idx * hops_per_iter  # iteration t is responsible for objects within t hops
                if sel.any():
                    with torch.autocast(device_type=h.device.type, enabled=False):
                        pred = model.coord_head(model.norm(hs[gb_t[sel], gp_t[sel]]).float())
                    terms.append(F.smooth_l1_loss(pred, goal_t[sel]))
            if terms:
                hint = torch.stack(terms).mean(); loss = loss + hint_w * hint
                if diagnostics is not None: diagnostics["hint_loss"] = hint.detach().float().item()
    weight = float(model.cfg.decision_head.get("aux_coord_weight", 0))
    if weight > 0 and model.coord_head is not None:
        gb, gp, goal = [], [], []
        for i, rec in enumerate(records):
            coords = rec.get("meta", {}).get("aux_coords")
            if not coords:
                continue
            cached = rec.get("_aux")
            if cached is None:
                text = _text(rec["state"]); _, offsets = tokenizer.encode_with_offsets(text); cached = []
                for name, xy in coords.items():
                    cached += [(p, xy) for p in entity_state_positions(text, offsets, name, 2)]
                rec["_aux"] = cached
            gb += [i] * len(cached); gp += [p for p, _ in cached]; goal += [xy for _, xy in cached]
        if gb:
            sel = h[move(torch.tensor(gb)), move(torch.tensor(gp))]
            with torch.autocast(device_type=h.device.type, enabled=False):
                out = model.coord_head(sel.float())
            # Coordinates are integer grid offsets; scale keeps the loss comparable to cross-entropy.
            aux = F.smooth_l1_loss(out, move(torch.tensor(goal, dtype=torch.float32)) / 4)
            loss = loss + weight * aux
            if diagnostics is not None:
                diagnostics["aux_coord_loss"] = aux.detach().float().item()
    if diagnostics is not None:
        valid_h = torch.cat([h[i, :x.input_ids.numel()].detach().float() for i, x in enumerate(packed)])
        diagnostics.update(numerical_statistics(valid_h, scores))
        diagnostics["batch_ids"] = [r.get("id", str(i)) for i, r in enumerate(records)]
    return loss


def numerical_statistics(hidden, scores):
    if isinstance(scores, dict):  # batched decisions: padded [N,K] tensors with a validity mask
        m = scores["mask"]; q = scores["q"].norm(dim=-1); k = scores["k"].norm(dim=-1)[m]
        logits = scores["logits"][m]; probs = scores["logits"].softmax(-1)[m]
    else:
        q = torch.stack([s["q"].norm() for s in scores])
        k = torch.cat([s["k"].norm(dim=-1) for s in scores])
        logits = torch.cat([s["logits"] for s in scores])
        probs = torch.cat([s["logits"].softmax(-1) for s in scores])
    return {key: value.detach().float().item() for key, value in {
        "hidden_mean": hidden.mean(), "hidden_std": hidden.std(unbiased=False),
        "hidden_max_abs": hidden.abs().max(), "q_norm_mean": q.mean(), "q_norm_max": q.max(),
        "k_norm_mean": k.mean(), "k_norm_max": k.max(), "logit_min": logits.min(),
        "logit_max": logits.max(), "logit_max_abs": logits.abs().max(),
        "logit_std": logits.std(unbiased=False), "probability_min": probs.min(),
        "probability_max": probs.max()}.items()}


def parameter_norm(parameters, gradients=False):
    values = [p.grad if gradients else p for p in parameters]
    values = [v.detach().float().norm().square() for v in values if v is not None]
    return torch.stack(values).sum().sqrt().item() if values else 0.0


def check_safety(stats, safety, failure_path, records):
    numeric = [v for v in stats.values() if isinstance(v, (int, float))]
    reason = None
    if safety.get("abort_on_nonfinite", True) and not all(math.isfinite(v) for v in numeric):
        reason = "nonfinite diagnostic"
    if stats.get("loss", 0) > safety.get("max_loss", 25.0):
        reason = "loss limit exceeded"
    if stats.get("logit_max_abs", 0) > safety.get("max_abs_logit", 100.0):
        reason = "logit limit exceeded"
    if reason:
        path = Path(failure_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"reason": reason, "diagnostics": stats, "records": records}, indent=2))
        raise FloatingPointError(f"{reason}; evidence saved to {path}")

