# Resumed Claude session — 28 September

## Current work

- VESSEL-1b continues its original GPU training process; VESSEL-1c retains its original
  waiting process. Neither was restarted by Codex.
- PROBE-2 is preregistered and queued behind VESSEL-1c. Existing PT checkpoints only;
  six positive-control fine-tunes first. The runner is `scripts/pt/probe2_queue.sh`;
  status/logs are in `reports/pt/probe2/`.
- The old waiting round-2b and GENERAL-4/S60 shell processes were replaced with queues
  that explicitly wait for `reports/pt/probe2/COMPLETE`. Completed r2b1 is skipped.
  A runtime failure leaves COMPLETE unset; a genuine failed positive control records
  INCONCLUSIVE, skips all slope work and releases lower-priority jobs.
- The probe noise band is recorded and committed before slope comparison. PT-3 is
  never launched automatically. A positive recommendation requires owner approval.
- The original worlds_v1 training overlapped 59 later worlds_v2 eval records. The
  frozen PROBE-2 copy excludes them, with manifest/hash in the ledger. Original data
  and PT checkpoints are unchanged.

## Extrapolation attack

The first extrapolation win is recorded in FRONTIER_STATE and the ledger with both
seeds' numbers. `scripts/p1_attack.sh` is running on CPU with two threads; no GPU slot
is consumed. New 12–16-hop suite: 250 worlds / 1,000 records, seed 92816, registered,
zero collisions with spatial training. Both r2b1 seeds and both controls evaluate
K=6,8,12,16,20,24. Logs/results: `reports/p1/attack_12_16/`.

Partial seed-7 r2b1 result: overall 12–16-hop accuracy K6=0.351, K8=0.342. Remaining
depths and controls are pending; this is not a final verdict. Do not promote it or
claim the earlier 7–10-hop win extends to 16 hops.

GENERAL-4 integration is planned as a preregistered mixed-domain variant: depth-tied
random passes and deep supervision on spatial batches; preserve existing scheduling
for other domains. The already registered G4 control recipe has not been silently
changed.

## Pi

See `reports/vessel/pi_profile/REPORT.md`. Two threads / one vessel is best measured:
0.489 seconds full-state int8, 0.439 seconds with an essential-facts ablation. Both
remain experimental because stand-in parity fails. Batched fp32 itself passes parity.
The original live relay is healthy; no model or rendering change was deployed.

Channel Watch has local batch-inference support, tests and benchmark tools; its Dockerfile
includes the new batch module. Activation requires explicit model metadata
`batch_axis: true`; inference batch defaults to one, independently of HTTP pull size.

## Verification

Positive-control gate tests: 2 passed. JS batch isolation/binding rejection tests:
2 passed. Shell/JS/Python syntax and git whitespace checks passed. Real-model fp32
batch-vs-PyTorch checks passed; int8 failures are retained in reports. Pi completed
24 full-state/compact configurations plus 6 essential-facts configurations.

Long jobs continue in background processes; this file does not imply an unattended
assistant is watching or that pending research results are already complete.
