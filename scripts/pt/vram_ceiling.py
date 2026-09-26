"""Re-test the VRAM ceiling after setting 'CUDA Sysmem Fallback Policy = Prefer No Sysmem Fallback'.

Steps batch size up (compiled, bf16, seq 1024) until a genuine OOM is raised; records the largest safe
batch and its throughput per model size, and flags any "crawl" (throughput collapse without OOM).
Run on an idle GPU only.
"""
import json, sys
import torch
src = open("scripts/pt/bench_pt.py").read().split("ap = argparse")[0]; ns = {}; exec(src, ns)
out = {}
for cfg, batches in (("configs/pt/pt_35m.yaml", (16, 24, 32, 40, 48, 64)), ("configs/pt/pt_70m.yaml", (16, 24, 32, 40)),
                     ("configs/pt/pt_150m.yaml", (12, 16, 20, 24, 32))):
    rows = []
    for b in batches:
        try: r = ns["run"](cfg, b, 1024, True); r.update(batch=b, ok=True)
        except torch.cuda.OutOfMemoryError: r = {"batch": b, "ok": False, "error": "OOM (raised, as intended)"}
        except Exception as e: r = {"batch": b, "ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        torch.cuda.empty_cache(); rows.append(r); print(cfg, json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        if not r["ok"]: break
    ok = [r for r in rows if r["ok"]]
    best = max(ok, key=lambda r: r["tokens_per_s"]) if ok else None
    crawl = any(ok[i + 1]["tokens_per_s"] < 0.5 * ok[i]["tokens_per_s"] for i in range(len(ok) - 1))
    out[cfg] = {"rows": rows, "best": best, "oom_raised": any("OOM" in r.get("error", "") for r in rows), "crawl_detected": crawl}
json.dump(out, open("reports/pt/vram_ceiling.json", "w"), indent=2)
