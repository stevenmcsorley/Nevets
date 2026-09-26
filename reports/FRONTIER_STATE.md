# SystemOne Lab — frontier state audit

Audited 25 September 2026 against the code, checkpoints, and saved evaluation JSON, not README claims alone. Every number cites the file it came from. Update this document at each synthesis point; do not rewrite history. Experiments are logged in `RESEARCH_LEDGER.md`, and models are ranked in `CHECKPOINT_REGISTRY.md`.

## 1. What exists

| Area | Implementation | Verified |
|---|---|---|
| Backbone | `SystemOneModel`: 8-layer pre-norm decoder, d=512, GQA 8/4 heads, SwiGLU, RoPE, tied LM head. 32.19M parameters (`reports/frontier/latency_*.json`) | code read |
| Attention topology | One shared state plus isolated question branches, with branch positions restarting after the state. The state is causal by default; `state_attention: bidirectional` lets state tokens see the whole state. Branches stay causal and never see siblings. | `tests/test_mask.py` |
| Decision head | Pointer readout: `<decide>` query against option-end keys, cosine scorer at temperature 10, logits in FP32 | `tests/test_decision_safety.py` |
| Query binding | `entity_binding`: `W_bind(mean h[first name occurrences] − mean h[second])` added to the query; reversing the query order exactly negates it. Before this audit it **raised on any question without two named entities**; it now falls back to the plain query. | fixed in this audit |
| Iteration | Gated weight-tied loop over the top-k blocks (`loop_iters`, `loop_blocks`), with zero-initialized gates | tests |
| Auxiliary losses | Coordinate regression at entity mentions (`aux_coord_weight`); training only | tests |
| Tokenizer | 16k byte-level BPE (`data/tokenizer.json`); checkpoints verify its hash | code read |
| Pretraining | `s1-35m-pretrain.pt`: 50k LM steps on a FineWeb-Edu slice | protected, read-only |
| Training | AdamW with separate backbone, head, and loop-gate LRs; cosine schedule; BF16 autocast with FP32 decision math; safety aborts; class-balanced sampling; soft targets via `distributions` | code read |
| Generators | spatial worlds, counterfactual pairs, paired one-hop, paraphrase, vertical probe, role-swap, binding stress, chain curriculum (branches, disconnected facts, `aux_coords`), transforms, probabilistic spatial (unused), chess | scripts |
| Gates | `scripts/run_gates.py`: role swap, name pairs, counterfactual, held-out, stress suites, rotation/reflection consistency, selective accuracy | reproduced (§8) |
| Serving | FastAPI `/v1/systemone`; playground `/`; Treasure Hunt `/game`; chess `/chess` and `/v1/chess` | live-tested |
| Chess | `chess_format.py` (piece-list state, every legal move as an option); Stockfish 19 MultiPV soft targets (τ = 80 cp); Lichess 2013-01 for training and 2013-02 for evaluation; FEN-level exclusion (§6) | 3,000-position eval set built; 150k training set building |
| Not present | temporal, causal, relational (non-spatial), rule, planning, or probabilistic-world generators beyond `generate_probabilistic_spatial.py`; any non-text state format (JSON or table); any transfer measurement | searched |

## 2. What is genuinely solved (survives attacks)

**One-hop spatial relations in text**, including inverse wording, role swaps, both name styles, irrelevant and name-mentioning distractors, and single-fact counterfactual interventions. `exp9-aux-loop` scores 100% on every one-hop development gate (ECE15 0.004) and 100% on the locked stress set (1,432 records, overlaps removed) (`reports/locked_final/summary.json`). Every attack in the development gates (rename, reverse, distractor, intervention) is passed. This is a narrow capability: a single domain, one wording family, nine labels.

## 3. What merely appears solved

- **Simple two-hop (stress `twohop` 100%).** Numbered worlds in that suite name objects `obj_0 → obj_1 → obj_2` in chain order. On the harder chain suite (random names, shuffled facts, branches), two-hop accuracy is 81.4% locked.
- **The confidence ≥ 0.9 rule.** On our own generators it holds at 95–100%, but on the original `spatial_locked` distribution it gives only 82.8% at 4% coverage. The threshold does not transfer across generators.
- **Low latency.** The advertised "~25 ms" is mostly overhead, not compute: 28 ms for an 81-token request on a 32M model, with each extra question costing only about 1.3 ms (`reports/frontier/latency_exp6a-noloop.json`).

## 4. Where reasoning breaks

| Evidence | Source |
|---|---|
| Chain accuracy 96.6 / 81.4 / 57.1 / 42.8% at 1–4 hops, and 26–38% at 7–10 hops (majority label 17–22%) | locked summary |
| Per-axis accuracy falls with hop count; labels that need cancellation (net zero on one axis) fall to 9% at six hops | `reports/loop/` analysis in the README |
| The coordinate probe recovers exact positions 37 / 27 / 6% at 1 / 2 / 3 links from the reference | README, exp9 probe |
| Locked counterfactual both-correct, 3–10 hops: 14.8% | `reports/locked_final/exp9-aux-loop.counterfactual.json` |
| Rotation/reflection consistency: 85.4% (gate 95%) | locked summary |

**Diagnosis:** the model reads individual facts but does not integrate displacements along a path. This is a depth and algorithm failure, not a parsing failure. It did not respond to a zero-gated loop over pretrained blocks, to auxiliary coordinates, or to their combination beyond four hops.

## 5. Shortcuts found (each is now a regression test or documented)

1. Every decision set before September used single-sentence states, so any multi-fact state collapsed (candidate: 41% with one irrelevant fact). Fixed by the multi-fact curriculum; covered by the stress suites.
2. Sequential numbered names reveal chain order (`obj_0→obj_1→obj_2`). Fixed in `generate_chain_curriculum.py` with random numbering; the stress `twohop` suite still carries the shortcut and is labelled as easy.
3. Letter names underperform numbered names on harder suites; every suite now reports both.
4. Accuracy falls with the number of facts at fixed hop count (1 hop: 100% at one fact, 61% at four, exp2). Mostly fixed by bidirectional state attention and curriculum.
6. **Option-order dependence (found 25 September, severe).** Every training record lists candidates in the same order, and option tokens attend causally to earlier options. Shuffling the option order changes the prediction on 52% (exp9) and 50% (exp6a) of dev chain questions; accuracy falls from 48.5% to 30.3% (`reports/frontier/option_order_sensitivity.json`). This violates the dynamic-candidate contract for every existing checkpoint. **Fix implemented:** `option_attention: isolated` (options share position ids, never see sibling options, and the decide token sees no option), which makes the model exactly order-invariant (test `test_isolated_options_make_probabilities_order_invariant`), plus `--shuffle-options` augmentation. Permanent gate: `option_shuffle_changed` in `run_gates.py`. Controlled comparison queued (ISO-A augmentation against ISO-B isolation).
5. Sign relations do not compose deterministically when distances vary. The Treasure Hunt landmark mode originally asked undetermined questions; it now uses only determinate compositions. The training generators use unit steps, so they always compose; this assumption is not in the data format and a general model must handle it.

## 6. Chess pipeline (verified)

- Positions: Lichess 2013-01 (training) and 2013-02 (evaluation); the SHA-256 of both dumps matches the published sums.
- Leakage: the training generator excludes every evaluation position by the first four FEN fields (placement, side, castling, en passant). The months are disjoint by game. Re-check after generation: `scripts/check_chess_leakage.py` (to be written with the training set).
- Targets: every legal move is scored (MultiPV = number of legal moves) at depth 10; target `softmax(cp/80)`, mate = ±10,000 cp. The hard label is the argmax.
- No chess model has been trained yet.

## 7. Transfer

**No cross-domain transfer has been measured.** Every trained decision model is spatial. The chess branch will give the first measurement: BASE→CHESS (language checkpoint) against SPATIAL→CHESS (`exp6a-noloop`, which now runs chess questions after the binding fix).

## 8. Reproducibility and infrastructure

- `exp9-aux-loop` gates re-run after this audit's code changes: see `reports/frontier/exp9_repro.txt`, compared with `reports/loop/exp9.txt`.
- The tests (51) pass.
- Protected read-only: LM pretrain, reference, entity-binding one-hop, exp9, the Lichess dumps, the chess eval set, `spatial_locked`, and the locked counterfactual set.
- Hardware: RTX 3060 12 GB, 20 CPU threads, 185 GB free disk. The GPU sat idle during chess labelling; queue GPU work behind CPU jobs.

## 9. Architectural bottlenecks (ranked)

1. **No mechanism for iterative composition that the model actually uses.** A single forward pass of 8 pretrained layers limits effective path length; the grafted loop was never engaged (gate 0.05–0.11).
2. **Overhead-bound inference.** Per-call kernel launches dominate. Candidates: precomputed RoPE, fused or static masks, CUDA graphs or `torch.compile`, batched question scoring.
3. **One text format and one domain.** No evidence the model reasons from the state rather than from templates. Build JSON, table, and symbolic renderings with held-out formats.
4. **Calibration is miscalibrated in the middle band** (overconfident at 0.3–0.7). One temperature cannot fix it; train on worlds with known uncertainty.

## 9b. Strategic decision for the lab owner
Every competitive open System-One reproduction uses a large frozen pretrained backbone (see `LITERATURE_NOTES.md`). The lab's founding rule is random weights. Clean-room is right for reasoning research; a general-purpose product would likely need a licensed open backbone. This needs a human decision; the programme continues clean-room until then.

## 10. Highest-value research question

*Can a weight-tied recurrent reasoning core, trained from scratch with input injection and variable iteration counts, learn path composition that extrapolates to longer chains through more test-time iterations?* If yes, the same core becomes the REASONER lineage and is tested for transfer: chess, then new domains. If no, the next candidates are explicit entity-slot message passing and scale.


## 11. Synthesis — 25 September (end of session)

**What moved the frontier (controlled evidence):**
1. *Isolated options* (ISO-B): exact option-order invariance and higher accuracy. Adopted everywhere.
2. *Trained weight-tied loop* (T-R2): +13 points at 3 hops against a matched control. The first mechanism to lift 3–4-hop composition.
3. *A decision-trained init transfers* to unseen symbolic domains (TM-1), above all on counterfactual interventions; LM pretraining beats scratch. **Caveat (TM-1-FLAG, 26 Sep):** the spatial init had 28,400 prior decision updates and BASE had 0, so the effect is not attributable to *spatial* content until a matched non-spatial decision-trained init is tested.
4. *Multi-domain training* (GENERAL-1) gave the best model on every spatial dev gate and the first broad multi-domain model (confounded with 5× more steps).

**What failed (and why it matters):** from-scratch small models never left chance (T-R1); convergence training stabilized the loop but cost 18 points at 3 hops (T-R3); per-iteration BFS hints did not reach the answer head (T-R4); spatial training did not transfer to chess (CH-2).

**Current bottlenecks, ranked:**
1. *Readout, not depth*: length extrapolation fails because the decision head never reads the propagated coordinates. Next: a coordinate-readout head.
2. *Causal see-versus-do* is unlearned (8.7%). Next: train on paired see/do twins.
3. *Held-out format generalization* is partial and miscalibrated (table 54%, ECE 0.157).
4. *Chess* is limited by data and targets, not by transfer. Next: action-value targets and much larger position sets.
5. *Inference latency* is overhead-bound (`reports/frontier/latency_*.json`).

**Public artefacts:** https://stevenmcsorley.github.io/Nevets/ (General playground, Treasure Hunt, Chess), release `demo-models-v3`.
