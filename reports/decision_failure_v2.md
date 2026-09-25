# Decision failure v2 — preserved audit

The supplied command used scripts/train_decision.py with the default configs/s1_35m.yaml, pretrained checkpoint, data/processed/spatial_train.jsonl, batch 8, 10,000 steps, output checkpoints/s1-35m-spatial.pt. Effective defaults: AdamW LR 3e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1, accumulation 1, gradient clipping 1.0, seed 42, no scheduler. CUDA autocast chooses BF16 on this RTX 3060. Model parameters remain FP32. Pointer projection and dot product execute under BF16 autocast. Cross entropy under CUDA autocast promotes to FP32, but cannot undo rounded pointer scores. Soft-target log-softmax already explicitly casts to FP32.

User-reported losses rise from roughly 2.2 to 327680, interspersed with low-loss batches. Original console log is unavailable; the supplied excerpts are preserved in evidence_v2/request.txt. Failed checkpoint and original configs/source/scripts are copied into evidence_v2. Checkpoint SHA256 values are preserved there. Git status failed: this directory has no .git repository. The LM checkpoint will not be written.

Initial audit: one backward per microbatch, one optimizer step per accumulation window, zero_grad before training and after steps; no accidental loss multiplication in backprop. Existing clipping was silent. No scheduler or separate head/backbone LR existed. Option endpoints point to closing </opt> tokens; decide points to <decide>. Masks use True=allowed, causal state and causal same-question branches; padding rows attend themselves. Candidate targets are looked up by name. These require automated validation, not assumptions.

Root cause remains under investigation. An unbounded bilinear head and aggressive full-model optimization are hypotheses; BF16 quantization explains quantized numbers but is not by itself evidence of why scores diverged. Numerical checkpoint evidence will be recorded in evidence_v2/audit.json.

## Completed investigation

The immediate failure mechanism is an unbounded bilinear pointer head whose query and candidate vectors grew very large and nearly collinear. In the failed checkpoint, the first two candidate keys have cosine similarity 0.9999983. Their dot products have an enormous common offset (~122 million). BF16 score quantization then replaces useful margins with ties or jumps of 524,288. Cross entropy correctly returns small losses for tied/correct candidates and enormous losses when the target falls one BF16 bin below the maximum. In the fixed eight-record forensic batch, individual losses include 0, log(3), log(4), log(8), log(9), and 524,288. Two losses of 524,288 averaged over eight examples give 131,072 plus small terms. This directly explains the reported alternating small/power-of-two-like losses.

Evidence: `checkpoint_forensics.json`, reproduced by `scripts/audit_decision_failure.py`; original scorer in `evidence_v2/src/systemone_lab/model.py`. Switching just the failed checkpoint's dot product to FP32 still gives batch loss 91,648: FP32 alone cannot repair already-diverged weights. Backbone hidden activations did not explode in the original audit (first-record max abs fell from 60.46 to 8.13), while pointer norms increased by thousands of times. The LM checkpoint is not the source of the catastrophic scores.

The controlled frozen-backbone ablation further isolates the head: even FP32 scaled-dot exceeds the 100-logit safety limit at head LRs 1e-5, 3e-5, and 1e-4. Cosine passes at every tested LR. Thus reducing LR or changing cross-entropy dtype alone is insufficient. The original shared 3e-4 LR and absent warmup were aggressive, but no preserved optimizer trajectory exists to prove which original update first initiated norm growth. Do not interpret this report as proof that BF16 alone caused parameter growth. We can prove head norm explosion, quantify its scoring consequences, and reproduce unsafe norm/score growth with a frozen backbone.

### Comparable checkpoint forensics

Same first eight original training records; BF16 backbone and projections in both cases. After = cosine checkpoint trained only on the balanced 800-example gate, not the full corpus. These are numerical comparisons, not claims of equal task accuracy.

| Quantity | Failed checkpoint | Repaired gate checkpoint |
|---|---:|---:|
| Max absolute logit | 122,683,392 | 4.6133 |
| Max q norm | 52,160.76 | 22.177 |
| Max k norm | 37,616.16 | 24.142 |
| Pre-clip gradient norm | 520,861.13 | 235.98 |
| Batch loss | 131,072.84375 | 3.84063 |
| Per-example loss range | 0–524,288 | 1.45793–7.42106 |

The repaired model is not yet good at the original corpus's harder requests; its finite loss on these records must not be confused with solved spatial reasoning.

### Gates and experimental protocol

All runs start from the same original 50,000-step LM weights, seed 42. Frozen runs cache BF16 backbone hidden states under no_grad, then train only ptr_q/ptr_k. Unfrozen runs recompute the backbone each update. Final predictions use the normal FP32 inference path, which is a stricter check than training-time BF16 accuracy. Fixed 32 examples contain four each of the first five classes and three each of the remaining four; no labels or record IDs are inserted into the prompt.

- Unit tests: 26 passed, including all nine labels, permuted candidate order, option endpoint content, sibling isolation, padding, soft targets, actual SDPA boolean/additive semantics, FP32 loss/logits under autocast, optimizer groups/scheduler, checkpoint mismatch, and nonfinite abort.
- Fixed eight examples, frozen backbone, cosine LR 1e-5: 100% at step 200; stopped at 300; loss 2.24799 to 0.09746, zero increases; final FP32 evaluation loss 0.09545; max logit 5.99; max pre-clip gradient 5.81.
- 32 examples, frozen backbone, cosine LR 1e-5: 100% at step 350; stopped at 450; loss 2.21855 to 0.04442, zero increases; final FP32 loss 0.04869; max logit 7.00; max gradient 9.51.
- 32 examples, unfrozen, head LR 1e-5/backbone LR 3e-6: 100% at step 100; stopped at 200; zero increases; final loss 0.0000336; max logit 9.19; max gradient 20.89.
- Scorer/LR sweep: same 32 records and seed; frozen backbone; constant head LR; maximum 1,500 updates per run, with the same early-stop rule (100 steps after first observed 99% and loss below 0.15). Accuracy checked every 50 steps. Full predictions/confusion matrices are in each run JSON. Safety aborts are recorded as failures, not hidden. This sweep establishes a head LR, not an optimal backbone LR.
- Winner: cosine, head LR 1e-4 (largest tested stable), 100% final accuracy, 0.000940 final loss, 99% observed at step 100, max logit 8.50. The lowest scaled-dot LR avoided abort but finished at 96.875% FP32 accuracy; the other three exceeded the configured logit limit. Details: `pointer_scorer_ablation.json` and `.md`.
- Balanced generalisation run: 800 train / 200 held-out, random renamed objects, 1–2 hops, no duplicate rendered state+question across splits. Generator seed 7721. 600 optimizer steps, batch 16, cosine head LR 1e-4, backbone 3e-6, AdamW betas (0.9,0.95), eps 1e-8, decay 0.01, 100-step warmup and cosine decay, gradient clipping 1.0. No full-corpus training occurred.

| Split | Accuracy | ECE | Brier | Random | Train-majority baseline |
|---|---:|---:|---:|---:|---:|
| Train (800) | 59.625% | 0.11004 | 0.52621 | 11.111% | 11.125% |
| Held-out (200) | 43.5% | 0.07436 | 0.63476 | 11.111% | 11.5% |

Held-out one-hop accuracy is 64.89%, two-hop 24.53%. These are above baseline but leave substantial reasoning work. The test uses the same rendering family as training; it is not an out-of-distribution benchmark.

The 600-step trainer's every-10-step diagnostics show max |logit| 8.50374, max q norm 26.6668, max k norm 28.4284, max pre-clip gradient 142.4882, post-clip norm <=1.00000012. Gradients are not negligible; clipping remains material. No thousands/millions of gradient norm, nonfinite values, or safety-limit violations occurred. Sampled loss maximum was 2.32999. Safety checks run every microbatch, even between logged updates.

100 held-out counterfactual pairs, same model, no adaptation: original correct 57%, counterfactual correct 57%, both correct 36%, correct label flip 36%, probability moved in both required directions 75%. Independent uniform guessing gives 11.11% per answer and 1.23% both; constant predictions cannot produce a correct changed-label pair. Exact predictions and train/held-out confusion matrices: `generalisation_and_counterfactual.json`. These establish meaningful sensitivity to interventions, not complete reasoning ability.

### Data, layout, and checkpoint checks

All 50,000 training and 5,000 dev records passed target-name/index and option/decide token-position validation. Counts and percentages: `data_validation.json`. Training overlap is 4.762%, versus diagonal classes 14.13–14.474%; dev overlap 5.42%, diagonals 14.92–16.84%. Optional inverse-class-frequency sampling addresses this imbalance without silently rewriting the original datasets. `decision_token_trace.md` gives a hand-authored raw record and every token's ID, position, and role. Masks implement state causal attention and per-question causal branches with access to state; sibling questions are isolated. Padding query rows attend themselves.

New checkpoints store tokenizer SHA256, vocabulary size, special-token IDs, model config hash, and training step. Loading mismatched or legacy-unverified checkpoints fails unless explicitly overridden. The original LM predates this metadata, so controlled diagnostics explicitly opt into loading it without historical tokenizer verification. Its original bytes remain untouched; the present tokenizer hash cannot prove the historical tokenizer identity.

LM SHA256 before and after: `031c952b91ceb18314873969f60af595ce8b8e8df45cc7ec6c8882e077a804fc`.
Current tokenizer SHA256: `a78287861bc84f8d291fad598ae7d7c87f1da13bc3bc6e3019f1fa7924aeaa84`.

### Changes

- `src/systemone_lab/model.py`: explicit non-autocast FP32 score math, cosine/scaled-dot selection, positive finite fixed temperature, position assertions, q/k/logit diagnostics.
- `src/systemone_lab/config.py`: decision-head config and model/training config separation.
- `src/systemone_lab/training.py`: FP32 loss, validated targets/distributions, padding-mask assertion, diagnostics, safety evidence, checkpoint metadata/verification and LM overwrite protection.
- `src/systemone_lab/formatting.py`: candidate validity checks.
- `src/systemone_lab/eval.py`: model.eval and entire prediction path under inference_mode.
- `scripts/train_decision.py`: safe config, differential LR, freeze mode, warmup/cosine schedule, accumulation defined in optimizer updates, pre/post-clip logs, per-microbatch safeguards, balanced sampling, detached logging, checkpoint overwrite guard.
- `scripts/sanity_decision_overfit.py`: deterministic 8/32 gates, predictions/confusion matrices and full statistics.
- `scripts/sweep_decision_lr.py`: controlled four-LR/two-scorer ablation gated on micro-overfit success.
- `scripts/sanity_decision_generalisation.py`: balanced disjoint data, held-out metrics, counterfactual pairs and evaluation.
- `scripts/audit_decision_failure.py`: reproducible read-only failed/pretrained/repaired checkpoint comparison.
- `scripts/eval_spatial.py`, `scripts/eval_counterfactual.py`: explicit legacy-tokenizer override flags.
- `scripts/train_lm.py`: detached loss logging only; not run and no LM weights modified.
- `configs/s1_35m_decision_safe.yaml`: conservative diagnostic baseline.
- `configs/s1_35m_decision_validated.yaml`: sweep-selected 1e-4 head LR, 3e-6 backbone LR.
- `tests/test_decision_safety.py`: numerical, mapping, layout, mask, inference, optimizer and checkpoint regression tests.
- Reports/evidence and diagnostic checkpoints described above. Original `configs/s1_35m.yaml` and both original checkpoints remain unchanged.

### Next command — deliberately not executed

The legacy override is required because the untouched LM checkpoint has no stored tokenizer hash; use only with the tokenizer hash documented above. The head LR is validated on these small tests; larger-corpus generalisation remains an experiment, protected by fail-fast limits.

```powershell
.\.venv\Scripts\python.exe scripts/train_decision.py --config configs/s1_35m_decision_validated.yaml --init checkpoints/s1-35m-pretrain.pt --allow-tokenizer-mismatch --tokenizer data/tokenizer.json --data data/processed/spatial_train.jsonl --batch 8 --steps 10000 --balanced-sampling --diagnostics --out checkpoints/s1-35m-spatial-v2.pt
```


### Follow-up: user-run 10,000-step spatial training

The user completed the exact recommended command on 24 September 2026. Checkpoint: `checkpoints/s1-35m-spatial-v2.pt`, verified step 10,000, cosine temperature 10, tokenizer SHA256 matched. The protected LM checkpoint hash remained unchanged. This follow-up changes the recommendation: pause further long training until the generalisation and counterfactual failure is addressed.

Across 1,001 diagnostic records (step 1 and every 10 steps), no nonfinite values appeared. Max loss 3.008, max |logit| 9.203, max q/k norms 66.73/70.12, max pre-clip gradient norm 287.17, and post-clip norm about 1.0. Sampled average loss was 2.232 over the first 100 steps and 1.145 over the last 100. The final sampled batch loss was 0.671. These statistics confirm numerical stability, not strong reasoning.

On the original 5,000-record spatial dev set: accuracy 38.64%, ECE15 0.19035, Brier 0.79966. The largest dev class is upper-left at 16.84%, so accuracy exceeds the majority-class baseline.

On the previously frozen balanced diagnostic benchmark: train accuracy 26.5%, held-out accuracy 26.0%, held-out ECE 0.44368, Brier 1.09926. The earlier 800-example diagnostic checkpoint achieved 43.5% held-out on these exact records. The checkpoints differ in training data and schedule, so this comparison indicates a failure of transfer, not a controlled estimate of data-size effects. Held-out accuracy is 26.60% one-hop and 25.47% two-hop.

The balanced held-out prediction distribution is strongly skewed: right 115/200, below 40/200, left 37/200, upper-right 3/200, above 3/200, overlap 1/200, lower-left 1/200, upper-left 0/200, lower-right 0/200. Right is correct on all 22 held-out right examples, while upper-left, overlap, and lower-right have zero class accuracy. This exposes class collapse hidden by aggregate dev accuracy.

On 100 paired interventions, original accuracy 31%, changed-state accuracy 12%, both correct and correct label flip only 2%, even though the correct-direction probability shift metric is 70%. A probability shift alone does not establish correct counterfactual reasoning. The paired test fails the intended acceptance criterion, so the checkpoint should not be presented as a solved spatial decision model.

Artifacts: `spatial_v2_run_summary.json`, `spatial_v2_dev_eval.json`, `spatial_v2_assessment.json`, and `spatial_v2_generalisation_and_counterfactual.json`. The earlier benchmark result is restored in `generalisation_and_counterfactual.json`.


### Next diagnosis: object-name shortcut

A read-only stratified dev evaluation found 52.85% accuracy on 2,494 numbered-name records and 24.50% on 2,506 renamed records. The original training generator always emitted `obj_#` names, while dev randomly renamed about half. On 200 newly generated matched one/two-hop worlds, where only object names changed, the 10,000-step checkpoint scored 72.0% with numbered names versus 25.5% with letter names; predictions changed on 71.0% of pairs. This is direct evidence of name dependence, beyond aggregate class imbalance. The reproducible script and full per-example outputs are `scripts/diagnose_spatial_v2.py` and `spatial_v2_name_shift.json`.

The generator now accepts a rename probability for training and dev; CLI default 0.5. Existing 50,000/5,000-record files and checkpoints were not modified. `scripts/generate_name_curriculum.py` prepared a disjoint balanced 800/200 one/two-hop curriculum, half numbered and half letter names, plus 200 held-out matched name pairs. New dataset artifacts live under `reports/name_curriculum/`.

A controlled 600-step run from the untouched LM checkpoint using the same cosine scorer, LRs, warmup, clipping, and safety guards produced `checkpoints/name-curriculum-gate.pt`. The result is mixed and not sufficient to promote: 34% balanced held-out accuracy and 26% both-correct on the earlier 100 counterfactual pairs, versus 26% and 2% for the 10,000-step checkpoint. On the matched name pairs, however, it scored only 20.0% numbered, 19.5% renamed, and 8% both correct; predictions changed on 64% of pairs. The earlier all-letter diagnostic checkpoint scored 9.5% numbered, 29% letter, and 7% both. Thus independent name randomization changes which names the model favors but does not establish name invariance. These checkpoint comparisons use different curricula and are diagnostic rather than a controlled estimate of a single treatment effect.

The next bounded experiment should train paired renderings of the *same latent world* with identical labels, starting with balanced one-hop relations. Gate it on high accuracy for both names and low prediction-change rate on held-out matched pairs before adding two-hop chains. Continue to test counterfactual correctness, not just probability movement. Do not launch another 10,000-step run on the current corpus while these basic gates fail.


### Paired one-hop follow-up

The user completed the 600-step paired-name one-hop diagnostic (`checkpoints/paired-onehop-gate.pt`) from the original LM checkpoint. The 800 training worlds each had numbered and letter-name renderings with the same label. Held-out sets were disjoint and class-balanced. Checkpoint diagnostics contain 61 samples from steps 1 through 600: no nonfinite values, max loss 2.2473, max |logit| 8.6725, max q/k norms 24.12/31.79, max pre-clip gradient 197.37, post-clip about 1.0. Mean sampled loss fell from 2.0944 in the first 100 steps to 0.2134 in the last 100.

On 200 held-out matched-name pairs, numbered accuracy is 89%, letter-name accuracy 87%, both correct 82%, and predictions changed on 12%. The same 400 renderings yield 88% accuracy, ECE15 0.06777, Brier 0.11866. On 100 held-out one-hop counterfactual pairs, both correct is 82% and directional probability shift 100%. These results are materially better than the previous long run but below the project's 97%/98%/90% promotion gates. The 100% probability movement must not be mistaken for perfect counterfactual answers.

`reports/paired_onehop/run_assessment.json` and `template_errors.json` expose the remaining problem: above and below account for almost all errors. On letter names, the template `{b} is below {a}.` (correct label above) was wrong on all 11 held-out instances; the model predicted below. For the correct label below, `{b} is above {a}.` was correct on only 2/8 letter-name instances and 1/8 numbered instances. The sample counts are modest, but the inverse-wording pattern is concrete. The next small experiment should balance direct, synonym, and reversed vertical phrasings and test them on unseen names before adding two-hop composition. No further long corpus run is justified yet.


The initial 200-pair one-hop held-out set had modest vertical-template counts. A separate, disjoint, template-balanced vertical probe therefore tested 50 new name pairs for each of six above/below phrasings (300 pairs total) without further training. Results: numbered 58.67%, letter-name 55.67%, both correct 34%, prediction changed 46.33%. The reversed `below` template `{b} is above {a}.` was correct on only 3/50 numbered and 13/50 letter-name examples; the reversed `above` template `{b} is below {a}.` was correct on 41/50 numbered but only 11/50 letter-name examples. Detailed predictions: `reports/paired_onehop/vertical_probe_results.json` and `vertical_template_errors.json`. This validates the vertical/inverse-wording weakness independently of the small original template buckets. The next supervised micro experiment should pair direct, synonym, and reversed verbalizations of each *same world* under the same label and retain both name renderings; then evaluate on new names before two-hop training.


### Prepared next experiment: paired one-hop paraphrases

`scripts/generate_paired_paraphrase_onehop.py` generated 800 training worlds rendered with every relation wording template and both numbered/letter names (3,734 records), plus 200 disjoint held-out worlds (934 records / 467 matched name pairs). All rendered requests passed target mapping checks and were verified disjoint from the earlier one-hop held-out, counterfactual, and vertical probe requests. Class frequencies differ because template counts differ, so the next 800-step diagnostic command uses balanced sampling. The exact PowerShell sequence and evaluation gates are in `README.md`. No new paraphrase training run has been started by this preparation.


### Paired-paraphrase follow-up and role-swap gate

The user completed an 800-step run from the original LM checkpoint on 800 worlds, all available one-hop wording templates, and both numbered/letter names (`checkpoints/paired-paraphrase-onehop-gate.pt`). Its 81 logged diagnostics have no nonfinite values; max |logit| 8.8507, max pre-clip gradient 212.92, and mean sampled loss fell from 2.0802 in the first 100 steps to 0.1368 in the last 100.

On the new held-out set, accuracy is 88.87%, ECE15 0.02894, Brier 0.12133. Same-set comparisons with the prior 600-step one-hop checkpoint: original 200 name-pair both-correct 82% → 88.5% and prediction-change 12% → 8%; 300-pair vertical-probe both-correct 34% → 45.67%; original 100 counterfactual-pair both-correct 82% → 86%. These checkpoints also differ in step count; do not attribute the gains solely to paraphrase pairing. Artifacts: `reports/paraphrase_onehop/same_set_comparison.json`, `run_summary.json`, and `vertical_template_errors.json`.

A new read-only role-swap gate uses the same 300 vertical facts under both name styles (600 cases), retains each state, and reverses the queried objects; expected labels flip above↔below. The paraphrase checkpoint got the original orientation correct 63.67%, swapped orientation correct 36.67%, but both correct on only 7.83%. The earlier one-hop checkpoint got both correct on 6.5%. This is a stronger failure than the ordinary name-pair and counterfactual metrics reveal. See `scripts/eval_role_swap.py` and `reports/paraphrase_onehop/vertical_role_swap.json`. The next training gate should include both question orientations with inverse labels for each one-hop fact before any two-hop or long-corpus training.


### Prepared next experiment: paired question-role supervision

`scripts/prepare_role_swap_curriculum.py` doubles each numbered/letter paraphrase example by reversing the query objects and using the mathematically inverse relation (including overlap→overlap). It drops complete four-rendering groups when prompts duplicate training examples or overlap protected evaluation prompts. The prepared set has 6,996 training and 1,844 held-out records. All target mappings, distinct option positions, role inversions, and train/evaluation prompt disjointness were checked. The exact 1,200-step continuation from `checkpoints/paired-paraphrase-onehop-gate.pt` and nine evaluation commands are in `README.md`. No role-swap training run has been started by this preparation.


### User-run role-swap continuation: failed diagnostic

The user authorized the nine-command role-swap sequence. The 1,200-step continuation (`checkpoints/role-swap-onehop-gate.pt`) completed with verified tokenizer metadata and no numerical abort. Max logged loss 3.4124, max |logit| 9.4640, max pre-clip gradient norm 323.67, no nonfinite values. Mean sampled loss fell from 1.616 in the first 100 steps to 0.651 in the last 100. The LM checkpoint hash remains `031c952b91ceb18314873969f60af595ce8b8e8df45cc7ec6c8882e077a804fc`.

Same-set outcomes versus the prior paraphrase checkpoint: vertical role-swap both-correct 7.83% → 1.17%; old name-pair both-correct 88.5% → 49.5%; vertical name-pair both-correct 45.67% → 38.33%; 100 counterfactual-pair both-correct 86% → 30% (probability-shift 100% → 96%). New held-out accuracy is 52.44%, ECE15 0.02312, Brier 0.47802; held-out role-swap both-correct only 6.40%. These losses of accuracy rule out promotion despite stable numerical metrics.

On the same vertical probe, forward and swapped questions receive identical top predictions in 97.83% of cases (before: 84.67%). Above-labeled facts have 85% forward accuracy but 15.33% swapped accuracy; below-labeled facts have 20% forward and 80% swapped accuracy. In 20 inspected letter-name examples, swapping just the two query tokens produced mean cosine similarity 0.9995 between decision hidden states, and mean absolute logit difference fell from 0.1525 before training to 0.0745 after. This supports the interpretation that the model is largely ignoring query object order. It does not by itself establish whether attention, representation, head design, or optimization is the first cause.

The original `checkpoints/paired-paraphrase-onehop-gate.pt` remains the better one-hop checkpoint. Do not continue from the failed role-swap checkpoint or launch longer corpus training. Next investigate how the query names are represented at `<decide>` and test a bounded architectural or input-format intervention with the fixed role-swap probe. Artifacts: `reports/role_swap_onehop/comparison.json`, `sensitivity.json`, `vertical_role_swap.json`, `heldout_role_swap.json`, `old_name_pairs.json`, `vertical_name_pairs.json`, `vertical_template_errors.json`, `heldout_metrics.json`, and `counterfactual_metrics.json`.


### Ordered-query interventions: partial role signal, no promotion

The request formatter now extracts the ordered entities from spatial questions (or accepts explicit `query_entities`). `decision_logits` can add a normalized ordered name-embedding difference to its pointer query. The API, trainer, and evaluator all pass the same parsed entity order. A read-only sensitivity check confirmed that this signal changes the scorer rather than disappearing inside the nearly invariant `<decide>` state.

The 1,200-step full-model `role_aware` continuation from the paraphrase checkpoint remained numerically stable but traded away old competence: vertical role-swap both-correct 46.17%, old 200 name-pair both-correct 33%, old 100 counterfactual-pair both-correct 27%, and new held-out accuracy 50.2%. `checkpoints/role-aware-onehop-gate.pt` is diagnostic only. Its fixed-set artifacts have the `role_aware_` prefix in `reports/role_swap_onehop/`.

A separate 1,200-step `role_adapter` continuation initialized a new role projection from the old pointer query and froze the backbone plus original pointer head. On the same fixed suites it scored 18.67% both-correct on 600 vertical role-swap cases, 87.5% both-correct on the old 200 name pairs, 85% both-correct on 100 counterfactual pairs, 45% both-correct on the 300 vertical name pairs, and 10.20% both-correct on 922 new held-out role-swap pairs. New held-out accuracy was 51.74%; final sampled training loss was 2.64. Its logged scores stayed finite (final max |logit| 7.49). The checkpoint `checkpoints/role-adapter-only-gate.pt` is also diagnostic only; its artifacts have the `role_adapter_` prefix in `reports/role_swap_onehop/`.

The read-only scale sweep (`query_scale_sweep.json`) used 54 old name pairs, 120 vertical role cases, and 30 counterfactual pairs. At adapter scale 1.0, role-swap both-correct reached 35% while old name-pair both-correct fell to 85.2%; scale 2.0 reduced both. This indicates an expressivity/binding problem, not merely an underweighted role vector. A likely hypothesis is that static name embeddings provide order without reliably binding those names to the fact's contextual entity occurrences; this is an inference from the interventions, not an established mechanism. Next run a bounded entity-binding architecture diagnostic against the same locked suites before spending more compute on training. The paraphrase checkpoint remains the reference. All 37 tests pass after these changes, including an API regression check for passing query-entity order.
