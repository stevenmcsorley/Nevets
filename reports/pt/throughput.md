# PT-0 — throughput, wall-clock and disk budget

Measured on the lab machine (RTX 3060 12 GB, Windows 11, PyTorch 2.14 + CUDA 13.2, bf16 autocast, fused AdamW), LM training at seq 1024 with the new 32k tokenizer. Raw data: `reports/pt/throughput.json`; script: `scripts/pt/bench_pt.py`.

## Findings

1. **On this Windows (WDDM) driver, exceeding 12 GB of VRAM does not raise out-of-memory; the driver spills to system RAM and throughput collapses** (35M eager: 41.9k tok/s at batch 8 → 2.2k tok/s at batch 48 with a "28 GB" peak). Operating points stay below about 9 GB to leave room for fragmentation. Larger effective batches come from gradient accumulation.
2. **`torch.compile` requires Triton, which has no official Windows build.** The community package `triton-windows` 3.5.1 (installed in the lab venv) makes compile work, giving **+52% throughput and about half the memory at 35M, +32% at 150M.**

| model | params (non-emb.) | eager best | compiled best (setting) | peak VRAM |
|---|---|---|---|---|
| pt_35m | 40.2M (23.9M) | 41.9k tok/s (b8) | **68.3k tok/s** (b24) | 6.1 GB |
| pt_70m | 75.5M (55.0M) | 23.6k tok/s (b8) | **36.2k tok/s** (b16) | 7.0 GB |
| pt_150m | 157.6M (124.8M) | 14.4k tok/s (b8) | **19.8k tok/s** (b12) | 8.8 GB |

## Projected wall-clock (compiled, excluding downstream probes)

| model | 1B tokens | 5B | 10B | 20B |
|---|---|---|---|---|
| pt_35m | 4.1 h | 20 h | 1.7 d | 3.4 d |
| pt_70m | 7.7 h | 1.6 d | 3.2 d | 6.4 d |
| pt_150m | 14.0 h | 2.9 d | **5.8 d** | **11.7 d** |

**PT-2** (35M and 150M on the same 1B tokens) takes ≈ 18 h of pretraining plus 8 downstream probes (≈ 3 h) ≈ **21 h locally**. **PT-3** at 150M: 10B tokens fits locally (5.8 d, under the 7-day threshold); **20B does not (11.7 d)**.

## Disk

Currently checkpoints + data ≈ 22 GB (PT data 13.6 GB for 1.55B tokens: raw 4.1, clean 6.6, shards 2.9). Tokenised uint16 shards cost 2 bytes/token.
- **10B tokens** (all of `sample/10BT`): raw 28.5 GB + clean text ≈ 47 GB + shards 20 GB → peak ≈ 118 GB total, within the 150 GB budget (≈ 70 GB after deleting the clean intermediate).
- **20B tokens** (needs part of `sample/100BT`): raw ≈ 57 GB + clean ≈ 94 GB + shards 40 GB → ≈ 213 GB peak. **This exceeds the budget** unless the pipeline streams raw → shards without persisting the clean text and deletes raw shards after processing (≈ 120 GB peak). That is a storage decision for the owner.

## Rented-GPU estimate for PT-3 at 20B tokens (for the owner's decision; nothing has been spent)

Compute ≈ 6 × 157.6M × 20B ≈ **1.9 × 10¹⁹ FLOPs**.
- **H100** (median on-demand ≈ $3.44/GPU-h across 40 providers as of Sep 2026; marketplace offers from ≈ $2.10/h): at an assumed 20–30% utilization for a 150M model → **≈ 17–26 GPU-h ≈ $40–90**.
- **A100 80GB** (≈ $1.09–2.21/h): at ≈ 35% utilization → **≈ 48 GPU-h ≈ $55–105**.
- Add ≈ 30% for setup, uploading ≈ 40 GB of shards, and the downstream probes: **≈ $50–150 total.**
Prices from market trackers (see sources in the session log); verify on the provider's page at booking time.

**Recommendation:** run PT-2 locally first. Only if PT-2 shows downstream gains, choose between 150M × 10B locally (≈ 6 days, free, within budget) and 150M × 20B rented (≈ $50–150, owner approval required).
