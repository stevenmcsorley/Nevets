# Approved GENERAL repair comparison and S60 replication

Owner approved on 29 September: "ok do it" after the proposed label review,
controlled GENERAL comparison, S60 second seed and continued PT-3.

Status: background runner started on 29 September; waiting for PT-3 COMPLETE.
It also waits for all three shared GPU markers to clear, then holds GPU_BUSY.
No PT-3 interruption. An error or gate kill is recorded; there is no deployment.

## Fixed local budget

- GENERAL: two starting checkpoints (S30_s8 and GENERAL-1) x original/semantic
  sampler x seeds 7/8 = eight runs, 10,000 updates each, batch 16.
- Same general_v3 training data, domain shares, pair-consistency .5, label
  smoothing .05, optimizer/config and K~U[1,6] in every comparison.
- Exact optimizer/sampler resume at 5,000 updates; preserve both 5k and 10k
  weights. Each segment uses the same full 10k learning-rate schedule.
- S60 seed 8: same S30_s8 initialization and 30,000 spatial-only updates as
  S60 seed 7, changing only the training seed. Existing checkpoints untouched.
- Total ceiling: 110,000 training updates. Roughly another working day of local
  GPU time after PT-3, subject to throughput and gate stops. No rented compute.

Fresh development set: 200 records in each of seven domains, plus 100 matched
diagnostic-cost intervention pairs. Exact-state duplicates against known training
and evaluation are removed; this is not a claim of latent-world separation.
The extreme cost=0 versus cost=1 pairs are an easy diagnostic of responsiveness,
not a replacement for realistic counterfactual evaluation. These files are now
registered as evaluation-only, with hashes in the protocol.

Readouts include every domain, diagnostic/repair accuracy, expected-utility
regret, cost-flip both-correct, spatial depth and calibration. Compare fixed 10k
endpoints within each initialization, report both seeds and all arms. A useful
sampler signal requires >=3 points information-gathering gain in BOTH seeds,
non-increasing utility regret, no other domain down >3 points and spatial
long-chain loss <=2 points. These are development criteria, not promotion.

Input/config/code hashes are checked when queued and again before GPU acquisition.
`QUEUE_LOCK` prevents duplicate runners. Runtime progress: `status.txt`, `queue.err`
and per-arm outputs. A crash requires deliberate recovery; no silent restart loop.

Checks completed before launch: queue marker gating, matched sampler arguments,
semantic balance, exact interruption/resume, pair consistency; contamination scan
of 718,304 training records had zero registered-eval collisions or held-out formats.

Reproduce: `python scripts/general_repair.py --prepare`, then `--run`, with
`PYTHONPATH=src`. The existing runner is already queued; do not start a duplicate.
