# GENERAL-4 / S60 diagnosis — 29 September 2026

GENERAL-4 improved aggregate skill but failed skill retention. In-format accuracy
rose from GENERAL-1's 66.32% to 78.96%, held-out table from 54.14% to 69.36%,
and counterfactual both-correct from 28.42% to 39%. Information gathering fell
70% -> 57%, below the pre-registered 67% floor. Probability reached 69%, inside
its 68.75% floor. Other domain floors passed. Seed 8 was skipped as registered.
GENERAL-1 remains the promoted general model. No production regression was deployed.

S60 is successful as an exposure control: long-chain accuracy 77.31%, compared
with GENERAL-2's 65.52% / 67.25%. Its overall spatial accuracy is 90.62% and
short-chain accuracy 99.53%. More spatial-only training is sufficient to match
GENERAL-2's gain; this does not prove multi-domain training can never help.
S60 uses a larger spatial exposure than GENERAL-2, and is one seed. ECE 0.0535
is worse than the S30 reference 0.0363, so this is not an automatic promotion.

## Evidence and limits

1. G4 was a longer run of G3's recipe, not a new reasoning mechanism. G3 seed 7
   information gathering was 55.5%; 50k updates only brought it to 57%. Another
   duration increase is weakly motivated.
2. S30 spatial initialization differs from G1's `tournament/r2/looped.pt` init.
   Domain mix, renderings and regularization also changed. Initialization bias,
   competing tasks and numeric-reasoning difficulty are hypotheses, not isolated causes.
3. A concrete sampling distortion is now measured in `sampling.json`: information
   gathering contains 42.18% diagnostic labels, but balancing ten named repairs
   and one diagnostic label gives diagnostics only 9.09% of sampling weight.
   Logged G4 batches contain 1,464 diagnostic versus 14,512 repair examples
   (9.16% diagnostics). The model is rarely trained on paying for information.
   This also affects earlier label-balanced recipes and cannot alone explain
   the G4-minus-G1 difference.
4. G3's cross-domain pair sampler defect was previously fixed before G4. It is
   not a new explanation for G4. Aggregate improvements must not hide per-domain
   or per-counterfactual regression.

## Implemented correction and next controlled experiment

`--semantic-infogather` with `--domain-shares` balances test versus act 50/50
inside the same domain share. Existing runs retain the original default. Tests
cover domain mass, semantic balance, legacy behavior and rejection of unknown labels.
This changes training distribution; no accuracy improvement is claimed yet.

Next experiment should compare a 2x2 of initialization (S30 versus GENERAL-1)
and sampler (legacy versus semantic), with identical data, format loss, updates
and seeds. Use fresh development worlds with exact expected-utility targets;
measure diagnostic/repair recall, utility regret and cost-flip consistency as
well as all-domain retention. Store intermediate checkpoints instead of only
overwriting the endpoint. Select on development data, then use a fresh locked
evaluation and two seeds before promotion. If numeric reasoning remains weak,
test an auxiliary expected-utility objective separately rather than changing
initialization, sampling and objective together.

This diagnostic training matrix is **not launched or queued**; PT-3 retains its
current GPU slot. The sampling correction is ready for the next controlled run.
