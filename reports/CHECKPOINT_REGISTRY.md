# Checkpoint registry

An entry changes only with measured evidence (link the report). Read-only (protected) files are marked 🔒. Latency is on an RTX 3060 at BF16, for one spatial question with an 81-token state (`scripts/bench_latency.py`).

| Slot | Checkpoint | Params | Evidence | Latency | Notes |
|---|---|---|---|---|---|
| BASE | `s1-35m-pretrain.pt` 🔒 | 32.2M | 50k LM steps (FineWeb-Edu slice) | — | Root of every LM-initialized lineage. |
| REFERENCE | `paired-paraphrase-onehop-gate.pt` 🔒 | 32.2M | P-04 | — | Frozen comparison point for the historical gates. |
| BEST_SPATIAL | `exp9-aux-loop.pt` 🔒 | 32.2M | L-1: stress 100%, chains 96.6 / 81.4 / 57.1% (1–3 hops) | 75 ms | Loop ×2 costs 2.7× latency. |
| BEST_FAST (spatial) | `exp6a-noloop.pt` | 32.2M | dev chains 92.7 / 86.9 / 54.7%; one-hop gates 100% | 28 ms | About 5 points below BEST_SPATIAL on 1–3 hops, at 2.7× lower latency. |
| BEST_CALIBRATED | `exp6a-noloop.pt` | 32.2M | dev chain ECE 0.049; one-hop held-out ECE 0.004 | 28 ms | exp9 chain ECE is 0.071. |
| BEST_ONEHOP_MINIMAL | `entity-binding-onehop-gate.pt` 🔒 | 32.2M | EB-1 | — | Smallest change that fixed role binding. |
| BEST_CHESS | `chess/chess-base-r1.pt` | 32.2M | CH-1: top-1 25.2%, top-3 46.7%, cp loss 186; 9W/11D/0L vs random, 0W/6D/14L vs SF1320 | ~27 ms | Beginner level; options are order-dependent. |
| BEST_GENERAL | — | | | | No multi-domain model yet |
| BEST_SMALL | — | | | | T-R1 d=256 from-scratch runs failed to learn (see ledger) |
| REASONER | `tournament/r2/looped.pt` (run at K=4) | 32.2M | T-R2: dev chains 3-hop 72.5%, 4-hop 46.9% (control 59.3 / 38.2); one-hop gates 100% | ~2.5× flat (effective depth 20) | Best 3–4-hop model; degrades for K > 6. |
| Rejected, kept for the record | `exp9-calibrated.pt` (C-1), `exp7-loop8.pt` (X-7), `role-*` (P-05 to P-07), `s1-35m-spatial.pt` (P-01) | | | | Do not use as initialization. |
