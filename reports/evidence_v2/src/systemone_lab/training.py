from __future__ import annotations
import json, math, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from .formatting import pack_request, branch_attention_mask


def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def choose_dtype(device):
    if device.type == "cuda":
        major,_ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def save_checkpoint(path, model, cfg, tokenizer_path, step, metrics=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"config":cfg.to_dict(),"tokenizer":str(tokenizer_path),"step":step,"metrics":metrics or {}},path)

def load_checkpoint(path, model_cls, device="cpu"):
    from .config import ModelConfig
    ck=torch.load(path,map_location=device)
    cfg=ModelConfig(**ck["config"])
    m=model_cls(cfg); m.load_state_dict(ck["model"])
    return m,ck

def _target(layout, rec):
    target_key = str(rec["labels"][layout.key])
    try:
        return layout.option_keys.index(target_key)
    except ValueError:
        if layout.qtype == "noul":
            return 1 if rec["labels"][layout.key] else 0
        if layout.qtype == "score":
            return int(rec["labels"][layout.key])
        raise

def decision_loss(model, tokenizer, rec, device):
    return decision_batch_loss(model, tokenizer, [rec], device)

def decision_batch_loss(model, tokenizer, records, device):
    """Pad a batch of packed requests while preserving per-example branch masks."""
    packed = [pack_request(tokenizer, r["state"], r["questions"], device=device) for r in records]
    tmax = max(x.input_ids.numel() for x in packed)
    bsz = len(packed)
    pad_id = tokenizer.id("<pad>")
    ids = torch.full((bsz, tmax), pad_id, dtype=torch.long, device=device)
    pos = torch.zeros((bsz, tmax), dtype=torch.long, device=device)
    masks = torch.zeros((bsz, tmax, tmax), dtype=torch.bool, device=device)
    for i, x in enumerate(packed):
        t = x.input_ids.numel()
        ids[i, :t] = x.input_ids
        pos[i, :t] = x.position_ids
        masks[i, :t, :t] = branch_attention_mask(x.branch_ids)
        if t < tmax:
            # Keep padding query rows numerically well-defined. They never contribute to loss.
            z = torch.arange(t, tmax, device=device)
            masks[i, z, z] = True
    h = model.hidden(ids, pos, masks)
    losses = []
    for i, (x, rec) in enumerate(zip(packed, records)):
        for layout in x.layouts:
            logits = model.decision_logits(h[i], layout.decide_position, layout.option_end_positions)
            dist = rec.get("distributions", {}).get(layout.key)
            if dist:
                target_p = torch.tensor([float(dist.get(k, 0.0)) for k in layout.option_keys], device=device)
                target_p = target_p / target_p.sum().clamp_min(1e-12)
                losses.append(-(target_p * F.log_softmax(logits.float(), dim=-1)).sum())
            else:
                target = _target(layout, rec)
                losses.append(F.cross_entropy(logits[None, :], torch.tensor([target], device=device)))
    return torch.stack(losses).mean()
