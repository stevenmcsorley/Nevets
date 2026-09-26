"""PT-0: LM training throughput by model size, batch and torch.compile; projections for token budgets."""
import argparse, json, time
import torch
from systemone_lab.config import ModelConfig
from systemone_lab.model import SystemOneModel

def run(cfg_path, batch, seq, compile_, steps=12, warmup=4):
    cfg = ModelConfig.load(cfg_path); model = SystemOneModel(cfg).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    step_fn = model.lm_loss
    if compile_: step_fn = torch.compile(model.lm_loss)
    x = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device="cuda"); times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = step_fn(x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize()
        if i >= warmup: times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    return {"params_M": round(model.num_parameters() / 1e6, 1), "tokens_per_s": batch * seq / mean,
            "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9, "step_s": mean}

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/pt/throughput.json"); ap.add_argument("--seq", type=int, default=1024)
a = ap.parse_args(); results = []
for cfg in ("configs/pt/pt_35m.yaml", "configs/pt/pt_70m.yaml", "configs/pt/pt_150m.yaml"):
    for compile_ in (False, True):
        for batch in (4, 8, 16, 24, 32, 48):
            rec = {"config": cfg, "batch": batch, "seq": a.seq, "compile": compile_}
            try: rec.update(run(cfg, batch, a.seq, compile_)); rec["ok"] = True
            except torch.cuda.OutOfMemoryError: rec["ok"] = False; rec["error"] = "OOM"
            except Exception as e: rec["ok"] = False; rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            torch.cuda.empty_cache(); results.append(rec); print(json.dumps(rec), flush=True)
            if not rec["ok"]: break  # larger batches will also fail
json.dump(results, open(a.out, "w"), indent=2)
