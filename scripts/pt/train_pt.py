"""PT-2/PT-3 language-model pretraining from random weights on uint16 token shards.

Data: the sorted train shards, cut into non-overlapping (seq+1)-token windows (windows never cross a shard),
restricted to the first shards that cover --tokens, then visited in ONE seeded permutation: every model given
the same --tokens/--seq/--global-batch/--seed sees exactly the same windows in the same order (controlled
size ladder), and no window repeats within the budget. Training: bf16 autocast, torch.compile (optional),
AdamW (0.9, 0.95, wd 0.1 on matrices), linear warmup then cosine to 10% of peak, grad clip 1.0.
Every --ckpt-every tokens: validation loss on a fixed val slice and a registered-style checkpoint
`<out>/tok_<N>M.pt` (loadable by load_checkpoint). A resume file `<out>/resume.pt` is refreshed on each
checkpoint; restarting the same command continues from it.
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from systemone_lab.config import ModelConfig
from systemone_lab.model import SystemOneModel
from systemone_lab.training import cap_gpu_memory, save_checkpoint


def windows(shards, seq, budget):
    """(shard index, start) of non-overlapping windows over the first shards covering `budget` tokens."""
    maps, index, total = [], [], 0
    for f in shards:
        if total >= budget: break
        m = np.memmap(f, dtype=np.uint16, mode="r"); n = len(m) // (seq + 1)
        index.append(np.stack([np.full(n, len(maps)), np.arange(n) * (seq + 1)], 1)); maps.append(m); total += n * (seq + 1)
    return maps, np.concatenate(index)


def lr_at(step, total, peak, warmup):
    if step < warmup: return peak * (step + 1) / warmup
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup)))))


def batch_of(maps, idx, seq, device):
    rows = np.stack([np.asarray(maps[s][o:o + seq + 1], dtype=np.int64) for s, o in idx])
    return torch.from_numpy(rows).pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def val_loss(model, val, seq, micro, device):
    model.eval(); tot = 0.0; n = 0
    for i in range(0, len(val), micro):
        x = torch.from_numpy(np.stack(val[i:i + micro])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"): tot += model.lm_loss(x).float().item() * len(x)
        n += len(x)
    model.train(); return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--tokenizer", default="tokenizers/pt_32k.json")
    ap.add_argument("--shards", default="data/pt/shards/train_*.bin"); ap.add_argument("--val", default="data/pt/shards/val_0000.bin")
    ap.add_argument("--tokens", type=float, default=1e9); ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--global-batch", type=int, default=120, help="sequences per optimiser update")
    ap.add_argument("--micro", type=int, default=40); ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.02); ap.add_argument("--ckpt-every", type=float, default=250e6)
    ap.add_argument("--val-tokens", type=float, default=2e6); ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--no-compile", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many updates (smoke tests)")
    ap.add_argument("--mem-margin-gb", type=float, default=1.5, help="cap own VRAM at free-at-start minus this")
    a = ap.parse_args(); assert a.global_batch % a.micro == 0, "global batch must be a multiple of the micro batch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    print(json.dumps({"vram_cap_gb": cap_gpu_memory(a.mem_margin_gb)}), flush=True)
    cfg = ModelConfig.load(a.config); model = SystemOneModel(cfg).to(device).train()
    maps, index = windows(sorted(glob.glob(a.shards)), a.seq, a.tokens)
    order = np.random.default_rng(a.seed).permutation(len(index))
    per_step = a.global_batch * a.seq; total = int(a.tokens // per_step); assert total * a.global_batch <= len(order), "not enough data"
    vm = np.memmap(a.val, dtype=np.uint16, mode="r"); nv = int(a.val_tokens // (a.seq + 1))
    val = [np.asarray(vm[i * (a.seq + 1):(i + 1) * (a.seq + 1)], dtype=np.int64) for i in range(nv)]
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]; no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")
    step = 0; resume = out / "resume.pt"
    if resume.exists():
        st = torch.load(resume, map_location=device); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
        print(json.dumps({"resumed_at_step": step}), flush=True)
    loss_fn = model.lm_loss if a.no_compile or device.type != "cuda" else torch.compile(model.lm_loss)
    warmup = max(1, int(a.warmup_frac * total)); accum = a.global_batch // a.micro
    ckpt_steps = {int(round(k * a.ckpt_every / per_step)) for k in range(1, int(a.tokens // a.ckpt_every) + 1)} | {total}
    log = open(out / "train_log.jsonl", "a"); meta = {"config": a.config, "params": model.num_parameters(), "total_steps": total,
        "tokens_per_step": per_step, "windows_available": int(len(index)), "shards_used": len(maps), "lr": a.lr, "seed": a.seed}
    print(json.dumps(meta), flush=True); log.write(json.dumps({"meta": meta}) + "\n")
    def checkpoint(s):
        vl = val_loss(model, val, a.seq, a.micro, device); tok = s * per_step
        name = out / f"tok_{int(round(tok / 1e6))}M.pt"
        save_checkpoint(name, model, cfg, a.tokenizer, s, {"val_loss": vl, "tokens": tok})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s}, resume)
        rec = {"step": s, "tokens": tok, "val_loss": vl, "ckpt": str(name)}; print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if step == 0 and not (out / "tok_0M.pt").exists(): checkpoint(0)  # random-init probe point
    t0 = time.time(); seen = 0
    while step < total and not (a.max_steps and step >= a.max_steps):
        for g in opt.param_groups: g["lr"] = lr_at(step, total, a.lr, warmup)
        ids = order[step * a.global_batch:(step + 1) * a.global_batch]; tot = 0.0
        for m in range(accum):
            x = batch_of(maps, index[ids[m * a.micro:(m + 1) * a.micro]], a.seq, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"): loss = loss_fn(x) / accum
            loss.backward(); tot += loss.detach()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1; seen += per_step
        if not torch.isfinite(tot): raise RuntimeError(f"non-finite loss at step {step}")
        if step % 20 == 0:
            rec = {"step": step, "tokens": step * per_step, "loss": tot.item(), "grad_norm": gn.item(), "lr": opt.param_groups[0]["lr"],
                   "tok_s": seen / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 200 == 0: print(json.dumps(rec), flush=True)
        if step in ckpt_steps: checkpoint(step)
    print(json.dumps({"done": step, "hours": (time.time() - t0) / 3600}), flush=True)


if __name__ == "__main__":
    main()
