# Pi inference profile — 28 September 2026

The best measured full-state candidate is **0.489 s/vessel**, using two threads and
one vessel per inference. The int8 candidate and shortened states **fail the stand-in
parity gate** and have not been deployed. No quality threshold was relaxed.

## Method

Pi 5, existing `general-v1` stand-in, ONNX Runtime Node 1.22, isolated throwaway
containers limited to two CPUs / 900 MB, network disabled. Twelve frozen vessels,
three questions each; one warm-up batch; elapsed time includes packing, inference
and probability conversion. Each configuration was measured once. The original
1.2 s figure used a different representation/sample, so use the matched rows below
for speed comparisons. Memory is process RSS, not the whole container.

Records: first 12 of `pi-remote:~/nbench/recs.jsonl`, SHA256 of the full source file
`533796d57bcca902450abaaf3f68cd5d3759b86741437ea7d2bf5b0540a76bdb`.
Prose is reconstructed from the frozen JSON facts; this is a controlled benchmark,
not an exact replay of the live renderer's Python-style float/null serialization.

## Matched full-state prose results

Mean packed input: 286.5 tokens, including all questions/options.

| Graph | Threads | True inference batch | Mean seconds/vessel | RSS MB |
|---|---:|---:|---:|---:|
| fp32 | 1 | 1 | 1.443 | 322 |
| fp32 | 2 | 1 | 0.871 | 331 |
| fp32 | 2 | 4 | 1.280 | 437 |
| Original dynamic int8 | 2 | 1 | 0.691 | 289 |
| Original dynamic int8 | 2 | 4 | 1.067 | 419 |
| Full weight-MatMul int8 | 1 | 1 | 0.704 | 191 |
| Full weight-MatMul int8 | 2 | 1 | **0.489** | 205 |
| Full weight-MatMul int8 | 2 | 4 | 0.972 | 321 |

**Root causes:** the original weight-only int8 exporter explicitly dequantizes weights
and keeps computation in fp32. The original dynamic quantizer misses repeated loop
weights behind ONNX Identity aliases: 58 MatMulInteger nodes and 124 floating MatMul
nodes. Folding those aliases gives 142 MatMulInteger nodes; the remaining 40 floating
MatMuls are attention products. This 35M-parameter looped model executes 20 transformer
blocks at its default four recurrent passes, so parameter count understates compute.

The service's original batch setting only batches HTTP work retrieval. The new batch
graph performs a real padded batch. On this workload, padding/compute outweighs any
benefit; **retain two threads and inference batch one**. Batches two and four are
supported explicitly for further measurements, separately from HTTP pull size.

## Shortened states and quality

| Representation | Mean packed tokens | fp32 seconds/vessel | Full int8 seconds/vessel |
|---|---:|---:|---:|
| Full prose | 286.5 | 0.871 | 0.489 |
| Shorter wording, all facts retained | 278.5 | 0.857 | See raw timings |
| Essential-facts ablation | 261.7 | 0.798 | 0.439 |

The essential ablation removes length, report/window counts, path/net distance,
coordinates, speed maximum/deviation, course variance and neighbouring-vessel speed.
It is a benchmark candidate only. Both shorter states change 6/36 fp32 argmaxes
versus prose (83.3% agreement); maximum probability differences are 0.127/0.123.
They do not satisfy the existing 99% / 0.02 parity guard.

Full int8 on prose agrees with Pi fp32 on 29/36 questions (80.6%), maximum probability
change 0.0454: also a failure. The stand-in is not trained for vessels, so this does
not determine the trained VESSEL model's parity. It does rule out claiming the current
speed candidate is ready to deploy.

The batched fp32 implementation passes: on Pi B4 versus B1, 100% agreement and maximum
probability difference 8.21e-8. Desktop JSON-input checks against per-request PyTorch
also pass B1/B2/B4 (36 questions; maximum probability difference 5.37e-7).

## Before VESSEL deployment

1. Let VESSEL-1b/1c training and retention gates finish.
2. Export the selected checkpoint; rerun batch/quantization parity on all held-out
   vessels and the exact serving renderer. Compare held-out Claude-label agreement
   before changing rendering; never silently promote the shortened representation.
3. Benchmark the passing graph on the Pi. Keep the existing parity thresholds.
4. Package/deploy only the passing model. PT-3 remains a separate owner decision.

Reproduction: `scripts/export_onnx_batch.py`, `scripts/check_onnx_batch.py`,
`reports/vessel/onnx/profile-pi.sh`, `profile-essential.sh`; Channel Watch
`nevets/bench-profile.mjs` and `summarize-profile.mjs`. Raw measurements and Pi parity
are beside this report; desktop parity/operator counts are in `../onnx/batch_standin/`.
