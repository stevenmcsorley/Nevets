# PT-3 approved and queued — 28 September 2026

The owner approved the positive PROBE-2 recommendation: **35M, local compute, up to
10B tokens**. The validated positive-control gain was 6.97 percentage points versus
a 0.77-point noise band. Per 250M tokens, 35M probe slopes were +8.15 points in-format,
+5.07 held-out table, +3.48 counterfactual, and improved calibration. The size comparison
favoured 35M for in-format/table; counterfactual and calibration differences were within
noise. No 20B extension or rented compute is authorized.

Queue order: existing GENERAL-4 seed 7, its conditional seed 8, S60, then PT-3.
`scripts/pt/pt3.py --run` waits for S60 completion and clear GPU marker files.

The run starts from random weights in a new `checkpoints/pt3/pt_35m` directory. It uses
the proven PT-2 35M optimizer/batching settings and a new long cosine schedule; it does
not modify or resume the short-schedule PT-2 experiment. Usable one-pass corpus budget:
**9,878,323,200 predicted tokens**, 80,390 updates, 106 existing train shards. The tiny
shortfall from 10B is deliberate: no repeated windows. No additional download is needed.

Settings: config `pt_35m.yaml`, 32k tokenizer, sequence 1024, effective batch 120,
microbatch 30, peak LR 0.0015, seed 2027, compilation enabled, existing VRAM margin.
The frozen `protocol.json` records configuration/tokenizer/probe hashes and shard sizes.
Preparation checks the corpus/tokenizer match, input availability and PROBE-2 result;
the queued process repeats these checks before starting. A mismatched input stops it.

Every approximately 1B tokens (and at the final budget), checkpoint the LM, release its
process, and run the validated 6,000-update PROBE-2 fine-tune with seed 11 in its own
output directory. Evaluate worlds_v1 in-format, table, counterfactual and calibration.
The LM then resumes its optimizer and position in the same deterministic window order.

Plateau rule, specified before this run: stop after **two consecutive readouts without
a >=0.5 percentage-point new best on any of in-format/table/counterfactual**. Earliest
stop is the third (~3B) readout. This is an operational compute limit, not a statistical
claim that further training cannot help. Retain all checkpoints and readouts for review.
Calibration is reported at every readout; completing PT-3 does not promote a model.

Logs/status: `reports/pt/pt3/status.txt`, `train.log`, `readouts.json`, and per-probe logs.
`COMPLETE` marks the approved budget or plateau stop. Failures are recorded as ERROR
and do not mark completion. A singleton directory prevents duplicate queue processes;
a crash may require checking that the old process is gone before removing that lock.

Expected maximum duration after GPU launch: roughly 41 hours of LM training at the
observed 35M throughput, plus probes/checks. Four control/plateau unit tests pass;
preflight validates the exact training targets without touching the busy GPU.
