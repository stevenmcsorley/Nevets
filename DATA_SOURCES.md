# Data-source plan

Use datasets as **separate named sources with frozen revisions and independent held-out splits**. Never concatenate train/test partitions merely to increase volume.

## Tier 0 — our latent spatial world generator

Primary training source. Unlimited deterministic generation from known coordinates and transformations. This is where counterfactual pairs, rotations, reflections, translations, renamed entities, distractors, uncertainty and later 3D/topological worlds belong.

Why it matters: the latent world gives exact ground truth and lets us construct interventions that defeat shortcut learning.

## Tier 1 — spatial transfer/evaluation

### StepGame
- Purpose: textual multi-hop spatial reasoning, 1–10 hop evaluation.
- Public HF dataset: `ZhengyanShi/StepGame`.
- Original repository includes generation code and is MIT licensed.
- Policy: train on official train; development on validation; preserve test for external transfer.

### SpaRTUN / SpartQA
- Purpose: richer spatial relations, scene graphs, containment and multiple question types.
- Repository: `HLR/SpaRTUN`; predecessor `HLR/SpartQA_generation`.
- Policy: add an adapter and preserve real/human sets for transfer evaluation.

### bAbI
- Purpose: world-simulation controls; especially task 17 positional reasoning and task 19 path finding.
- Repository: `facebookarchive/bAbI-tasks`.
- Policy: regenerate with different seeds and hold out path lengths/decoy regimes.

### CLUTRR
- Purpose: non-spatial relational systematic-generalisation control.
- Repository: `facebookresearch/clutrr` / HF `CLUTRR/v1`.
- Policy: hold out relation-chain lengths and rule compositions.

### GQA scene graphs (later)
- Purpose: real-world scene-graph reasoning and eventual visual grounding.
- 22M compositional questions with structured scene representations and functional programs.
- Do not pull images into the first text-only model. First use scene graphs/programs as structured reasoning data after license review.

## Tier 2 — semantic pretraining from random weights

### FineWeb-Edu
- HF: `HuggingFaceFW/fineweb-edu`
- ODC-By; sourced from Common Crawl and subject to its terms.
- Use streaming and freeze exact document IDs/revision in the run manifest.
- Start with a bounded sample rather than downloading the full 1.53B-row corpus.

### FineWeb2 / Dolma 3 alternatives
- FineWeb2: multilingual, ODC-By.
- Dolma 3: large open pretraining corpus; current AI2 docs describe it as ODC-BY.
- Pick one primary corpus initially. Mixing giant corpora before the architecture is validated makes experiments harder to interpret.

## Tier 3 — broad typed-decision post-training

Included preparation script:

```bash
python scripts/prepare_decisions.py --per-source 10000 --out data/processed/decision_public.jsonl
```

It currently converts:
- BoolQ -> Noul (dataset card: CC BY-SA 3.0)
- CommonsenseQA -> Choice (MIT)
- Banking77 -> Choice intent classification (MTEB mirror card marks MIT; trace upstream obligations before redistribution)

Add later only after license review: NLI, sentiment, science QA, policy/rule simulations, tool routing, document classification, anomaly/risk worlds and calibrated latent simulations.

## Leakage discipline

For every source, record:
- dataset repository + exact revision;
- license;
- source split;
- row IDs/hashes;
- transformation code version;
- generator seed;
- whether the source is train, dev, transfer, or locked-test.

A benchmark that appears anywhere in LM pretraining cannot be treated as a pristine contamination-free capability test. Our own freshly generated latent-world suites are the primary causal/generalisation tests.
