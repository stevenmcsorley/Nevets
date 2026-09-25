# Research ledger

Append-only. Every significant experiment gets an entry, whether it succeeded or failed. Never delete or rewrite a past conclusion; add a later entry that supersedes it. Checkpoints are under `checkpoints/`, configs under `configs/`, and results under `reports/` at the paths given.

Format per entry: **ID** · parent → artifact · architecture · data · hypothesis · compute · result · conclusion · next.

---

## Pre-September lineage (summarized from README and `reports/decision_failure_v2.md`; not re-run)

| ID | Parent → checkpoint | Result | Conclusion |
|---|---|---|---|
| P-01 | LM → `s1-35m-spatial.pt` | pointer weights diverged (BF16, unbounded scores) | Superseded; do not use. |
| P-02 | LM → `s1-35m-spatial-v2.pt` (10k steps, cosine) | stable, but 38.6% dev, 26% balanced held-out, 2/100 counterfactual pairs; 72% → 25.5% under renaming | Failed the reasoning gates. |
| P-03 | LM → `paired-onehop-gate.pt` (600 steps, paired names) | 88% held-out; names both-correct 82%; vertical probe both-correct 34% | The aggregate hid an inverse-wording failure. |
| P-04 | LM → `paired-paraphrase-onehop-gate.pt` (800 steps, all wordings) | 88.9% held-out; names 88.5%; counterfactual 86%; vertical role swap 7.8% | **Reference checkpoint** (protected). |
| P-05 | P-04 → `role-swap-onehop-gate.pt` | role swap 7.8% → 1.2%; decision-state cosine across swaps 0.9995 | Failed: the model ignored query order. |
| P-06 | P-04 → `role-aware-onehop-gate.pt` (name-embedding difference in the query) | role swap 46.2%, but names 33% and counterfactual 27% | Failed: traded away old skills. |
| P-07 | P-04 → `role-adapter-only-gate.pt` (frozen, adapter only) | role swap 18.7%, names 87.5% | Failed: static embeddings carry no state information. |

## September 2026 session

**EB-1** · P-04 → `entity-binding-onehop-gate.pt` · `entity_binding` (zero-initialized `W_bind` on the difference of contextual name occurrences), trained alone with everything else frozen · `role_swap_onehop/train.jsonl`, 1,200 steps · *H: role information must come from the state occurrences, not the names.* · ~4 min · vertical role swap 7.8 → **90.0%**, held-out role swap 89.8%, names 98.5%, counterfactual 100%, held-out accuracy 94.6% (`reports/candidate_gate/summary.json`) · **Confirmed.** Linking query names to their state occurrences fixed role binding.

**EB-2** · EB-1 → stress suite (read-only) · 4 suites × 600 records · one-hop with one irrelevant fact: 41%, two-hop 29% · **Failure found:** every curriculum since P-03 used one-sentence states.

**EB-3A / EB-3B** · EB-1 → `entity-binding-multifact-{bindonly,full}.pt` · multi-fact mix, 1,200 steps · A (frozen): 46.7% disconnected; B (full): 93.2% disconnected, 80.3% mention, 41.2% two-hop, one-hop gates about 100% (`reports/binding_stress/summary.json`) · **B confirmed** for one-hop; two-hop is an *underfit* (train 35.5% ≈ held-out).

**X-1** · EB-3B → `exp1-bb3e5.pt` · backbone LR 3e-6 → 3e-5 · 1,200 steps · two-hop 41 → 64%, disconnected 99.5% · **Optimization was a bottleneck.**

**X-2** · X-1 → `exp2-chain.pt` · 1–6-hop chain curriculum (random names, branches, disconnected facts) · 3,000 steps · chain overall 33%; accuracy falls with fact count (1 hop: 100% at one fact → 61% at four) · Retrieval among many facts is weak.

**X-3** · X-2 → `exp3-stageA.pt` · 1–2-hop curriculum stage · chain two-hop 43% · Marginal.

**X-4** · X-1 → `exp4-bidir.pt` · **bidirectional state attention**, controlled against X-2 · stress 83.9 → 93.2%, two-hop disconnected 63.5 → 85.8%, chain 33 → 36% · **Confirmed.**

**X-5** · X-4 → `exp5-bidir-long.pt` · 15,000 steps, 120k fresh chain records · stress 99.1%, chain 2-hop 73.6%, 4+ ≈ 27–40% · Longer training helps up to about three hops.

**X-6A / X-6B** · X-5 → `exp6a-noloop.pt` / `exp6b-loop.pt` (top-4 loop, zero gate) · 1–4-hop curriculum, 6,000 steps · both: 2-hop 87%, 3-hop 55%; loop gates 0.01 · Curriculum helps; **the loop was never engaged**.

**X-7** · X-5 → `exp7-loop8.pt` (8-block loop, gate LR 3e-3) · gates 0.11; no change against X-6A · **Rejected:** grafted loop.

**X-8** · X-5 → `exp8-aux.pt` (coordinate auxiliary loss, weight 1) · no change; probe: exact position 28% at one link, 0% at three · **Rejected** at this weight.

**X-9** · X-8 → `exp9-aux-loop.pt` (auxiliary weight 10 + 8-block loop) · 8,000 steps · 1–4 hops: 97.7 / 91.3 / 64.2 / 38.8%; 5+ unchanged; probe 37 / 27 / 6% · **Best spatial model**; composition is still unsolved.

**C-1** · X-9 → `exp9-calibrated.pt` · global temperature fitted on a separate dev set (T = 6.87) · NLL 1.108 → 1.054, but ECE 0.062 → 0.076 · **Rejected:** mid-band overconfidence cannot be fixed with one temperature. The checkpoint is kept for the record.

**L-1** · X-9 locked evaluation (one time) · `reports/locked_final/summary.json` · stress 100%; chains 96.6 / 81.4 / 57.1% at 1–3 hops; 7–10 hops 26–38%; transforms 85.4%; `spatial_locked` 38.4%; locked counterfactual 14.8% · **Not promotable.**

**A-1 (audit, 25 September)** · see `FRONTIER_STATE.md` · found: spatial checkpoints crashed on non-spatial questions (fixed with a fallback when no entities are present); training was launch-bound (about 375 ms CPU against 30 ms GPU per step) · batched decision head, device-side masks, and packing caches: flat d256 step 280 → 50 ms (5.6×); equivalence test against the legacy path passes.

**T-R1 — FAILED (optimization)** · from scratch, d=256, 1–6-hop chains (`data/processed/tournament/chain_h1_6_train.jsonl`, seed 81001, 239,932 records, eval and locked states excluded) · arms flat4 and loop (P1-C2×K-D1, input injection, K ~ U[2,10]) completed; flat8 and loopaux stopped early · 12,000 steps × 32 · *H: a weight-tied core with input injection and variable depth learns composition that extrapolates with test-time iterations (Fan et al. 2024; Yang et al. 2023; Dehghani et al. 2018).* · **No arm learned**: loss 2.197 → 2.14 (ln 9 = 2.197); dev chains at chance (flat4 one-hop 24%; loop 20% at every K) (`reports/tournament/r1/`) · **Conclusion:** from random weights with this data mix and LR, neither architecture leaves the plateau within 12k steps. This says nothing about architecture. **Next:** T-R2 retrofits the loop onto a competent checkpoint so that K=1 equals the flat model.

**T-R2 (running)** · `exp6a-noloop` restructured as prelude (blocks 0–1) / weight-tied core (2–5) / coda (6–7); K=1 is exactly the flat model (test `test_looped_k1_equals_flat_checkpoint`) · control K=1 against looped K ~ U[1,6], same data (T-R1 chains), 6,000 steps × 16, seed 42 · K sweep at evaluation · `scripts/tournament_r2.sh`, `reports/tournament/r2/`.

**CH-1 — BASE→CHESS (done)** · LM → `checkpoints/chess/chess-base-r1.pt` · 18,000 × 16 (≈1.9 passes over 150k positions), soft MultiPV targets, causal (order-dependent) options sorted by UCI · loss 3.18 → 2.98 (plateau) · 3,000 held-out positions (Lichess 2013-02): **top-1 25.2%** (random ≈ 3%), top-3 46.7%, mean cp loss 186, blunder rate (≥ 200 cp) 26.6%, ECE 0.091; games: random mover 9W/11D/0L, Stockfish UCI_Elo 1320 0W/6D/14L (`reports/chess/r1/eval_base.json`) · **Conclusion:** beginner-level move choice; it cannot convert won positions against a random mover (11 draws). This matches the literature expectation (Ruoss et al. 2024: strength needs far more data plus action-value targets). **Next (CH-3):** isolated options, per-move action-value targets, and a much larger position set.

**CH-2 (queued, SPATIAL→CHESS)** · identical recipe from `exp6a-noloop`, for the transfer comparison.

**CH-1 / CH-2 (setup)** · BASE→CHESS (LM init) against SPATIAL→CHESS (`exp6a-noloop` init) · `train_2013-01.jsonl` (150,001 positions, zero overlap with eval by `scripts/check_chess_leakage.py`), 18,000 steps × 16 · `scripts/chess_r1.sh`, `reports/chess/r1/` · *H: spatial decision training transfers to chess (faster or better move choice).*

**CH-0 (data, running)** · Lichess 2013-01 (train, 150k positions) / 2013-02 (eval, 3,000), Stockfish 19 MultiPV over all legal moves, depth 10, τ = 80 cp · `data/processed/chess/`.

**S-1 (shortcut audit, 25 September)** · read-only on exp9 / exp6a · 400 dev chain questions, option order shuffled with a fixed seed · prediction changed on 52% / 50%; accuracy 48.5 → 30.3% / 44.8 → 29.0% (`reports/frontier/option_order_sensitivity.json`) · **Shortcut confirmed:** positional option identity. **Response:** `option_attention: isolated` (exact invariance, tested), `--shuffle-options` augmentation, and a permanent `option_shuffle_changed` gate.

**ISO-A / ISO-B (queued, `scripts/queue_q1.sh`)** · exp6a → chain data (the same as T-R2), 6,000 × 16, seed 42 · A: causal options with shuffle augmentation; B: isolated options · compared with the T-R2 control (no augmentation) · *H: architectural invariance matches or beats augmentation on accuracy with zero order sensitivity.*

**TM-1 (queued)** · `worlds_v1` (5 domains; train prose/JSON/kv; table held out; 600 counterfactual twins) · inits BASE / SPATIAL (exp6a) / SCRATCH · 6,000 × 16, seed 11 · `scripts/transfer_tm1.sh`, `reports/worlds/v1/` · *H: prior training (LM, spatial) transfers to new domains and formats.*

**R-1 (reproducibility defect, 25 September)** · regenerating `worlds_v1` with the same seed gave different files: temporal and dependency facts came from a Python `set`, whose order depends on `PYTHONHASHSEED`. Contents and labels are identical; only fact order differed (3,229/5,000 eval records byte-identical). **The stored `worlds_v1` remains canonical** (read-only; TM-1 uses it). The generators now sort set-derived facts; `test_generation_is_independent_of_python_hash_seed` regenerates in two processes with different hash seeds and requires identical bytes.

**INFO-1 (data factory)** · new `infogather` domain: repair a part now, or pay for a diagnostic first. The label is the exact expected-utility optimum (a correct repair is worth 1; EU(test) = −cost + Σ_o P(o) max_h P(h | o)). Act/test ≈ 57/43. Tested against the utility definition and for cost monotonicity. The domain goes into worlds_v2, not v1.

**CAUSAL-1 (data factory)** · new `causal` domain: a confounded SCM (Z → X, Z → Y, X → Y) with exact CPTs; questions ask P(Y | do(X)) or P(Y | X) as exact soft targets. Confounding is forced strong (Z shifts X by ≥ 0.4 and Y by ≥ 0.3) because the first version separated seeing from doing in only 25% of worlds. Tested against brute-force enumeration. Twins ask *see* against *do* on the same state, keeping pairs whose answers differ.

**WV-2 (data)** · `data/processed/worlds_v2`, seed 91001: 7 domains (plus infogather and causal), 120,000 train (prose/JSON/kv), 7,000 in-format eval, 7,000 held-out-table eval, and 1,200 counterfactual twins (rules, dependency, infogather cost flips, causal see/do).

**DEMO-1 (deployment)** · In-browser demos on GitHub Pages (https://stevenmcsorley.github.io/Nevets/): Treasure Hunt with `exp6a-noloop` and chess with `chess-base-r1`. ONNX export of backbone + head (`scripts/export_onnx.py`) with weight-only int8 (`scripts/quantize_weights_int8.py --min-size 200000`, 40 MB each). Dynamic int8 (activations quantized) was rejected: 96.25% argmax agreement. Parity against PyTorch: spatial 99.4% argmax / max |Δp| 0.040 (160 requests); chess 100% / 0.004 (100 positions), measured end to end through the JS pipeline in onnxruntime. JS tokenizer, packing, masks and chess rendering match Python exactly (`tests/test_js_parity.py`, about 2,800 chess positions). Models live in release `demo-models-v1`, verified against the committed `site/model/SHA256SUMS` at deploy time. Live check in headless Chrome: Treasure Hunt 6/6 treasures at ~115 ms/decision; chess ~500–650 ms/move on CPU WASM. **Found and fixed along the way:** RoPE/GQA `repeat_interleave` baked the sequence length into the ONNX graph (replaced with numerically identical ops; gates reproduce), and uneven board rows in both game UIs.

**T-R2 (preliminary; K=8/12 still running)** · looped retrofit of exp6a, K ~ U[1,6] against a K=1 control with the same data and steps: at K=4, 3-hop **72.5% against 59.3%** and 4-hop 46.9% against 38.2%, with one-hop held-out still 100%. Accuracy saturates past K=4 and 6–10 hops barely move. **Early conclusion:** a trained loop adds effective composition depth up to about four hops but does not extrapolate with more test-time iterations on this task. Final analysis follows.

**T-R2 — FINAL** · looped retrofit of exp6a (prelude 0–1, tied core 2–5, coda 6–7), trained with K ~ U[1,6], 6,000 × 16, seed 42 · K sweep on the dev chain suite (`reports/tournament/r2/gates_looped_k*.txt`):

| K | overall | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 hops | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| control (K=1, same training) | 0.500 | .987 | .920 | .593 | .382 | — | .354 | .331 | .320 | .033 |
| 1 | 0.508 | .987 | .904 | .652 | .404 | .410 | .366 | .325 | .326 | .046 |
| 2 | 0.526 | .990 | .927 | .682 | .427 | .453 | .396 | .325 | .349 | .047 |
| **4** | **0.537** | .997 | .929 | **.725** | **.469** | .472 | .376 | .338 | .315 | .057 |
| 6 | 0.526 | 1.00 | .906 | .715 | .452 | .463 | .376 | .331 | .302 | .054 |
| 8 (beyond training) | 0.513 | .997 | .881 | .677 | .433 | .445 | .386 | .322 | .292 | .049 |
| 12 (beyond training) | 0.490 | .992 | .853 | .642 | .362 | .407 | .356 | .318 | .297 | .045 |

All one-hop gates stay at 100% for every K. **Conclusion:** a *trained* weight-tied loop adds real composition depth (+13 points at 3 hops, +9 at 4 over a controlled baseline). This is the first mechanism in the programme that lifts 3–4 hops. But **it does not extrapolate**: iterations beyond the trained range degrade every hop count, so the recurrence has no stable fixed point. **Next (T-R3):** tie K to the problem during training (K ≥ hops, following Fan et al. 2024) and add a convergence objective (penalize ‖h_{K+1} − h_K‖, or train with extra no-gradient iterations) so that more test-time iterations refine rather than drift.

**ISO-A / ISO-B — FINAL** · exp6a → chain data (the same as T-R2), 6,000 × 16, seed 42; the T-R2 control is the no-fix baseline (`reports/tournament/iso/`):

| Arm | dev chains overall | 1 / 2 / 3 hops | option shuffle changes prediction (chains / stress) | one-hop gates |
|---|---|---|---|---|
| control (no fix) | 0.500 | .987 / .920 / .593 | 0.490 / 0.458 | 100% |
| ISO-A shuffle augmentation | 0.505 | .971 / .924 / .627 | 0.037 / 0.000 | 100% |
| **ISO-B isolated options** | **0.524** | .977 / .927 / **.655** | **0.000 / 0.000** | 100% |

**Conclusion:** architectural isolation is strictly better than augmentation. It gives exact order invariance *and* higher accuracy (+2.4 overall, +6.2 at 3 hops against the control), because each option is encoded without sibling interference. Augmentation leaves 3.7% residual order dependence. **Decision:** `option_attention: isolated` is the default for every new lineage (GENERAL, CHESS r2, REASONER r3+). Existing demo checkpoints keep causal options and a fixed option order.

**CONTAM-1 (disclosure, 25 September)** · While building GENERAL-1's mix, the collision guard found 31 training states that also occur in evaluation files: 19 from the tournament chain curriculum and 12 from `binding_stress/train_mixed.jsonl` (inherited since EB-3), against `paired_onehop/name_pairs.json` (18), `vertical_probe_pairs.json` (11) and `chain/eval_hops.jsonl` (2). All are one-fact states such as "W sits north of U."; the space of such sentences is tiny, so independent generators collide. **Affected claims:** one-hop name-pair and vertical-probe gates for EB-3 onward (≤ 2% of their items), which were already about 100%. **Unaffected:** every multi-hop, stress, transform, locked, worlds and chess result. **Fix:** GENERAL-1's mix drops all 31 (`data/processed/general_v1/manifest.json`), and new mixes are checked against *every* evaluation file. Earlier generators excluded only the sets listed in their scripts, and those lists omitted the name-pair gates.
