# Literature and open-implementation notes

These notes are ideas only. Nothing from these projects is copied into the lab without a license check. Jev outputs are never used as labels (hard rule 10).

## Iterative depth and length generalization
- **Universal Transformers** (Dehghani et al., ICLR 2019): a weight-tied recurrent transformer with ACT halting. Motivates the looped core.
- **Adaptive Computation Time** (Graves 2016): learned halting. Candidate once K sweeps show depth helps.
- **Looped Transformers for Length Generalization** (Fan, Du, Ramchandran, Lee; arXiv 2409.15647, ICLR 2025). Looped models with **input injection** and an **adaptive number of steps** extrapolate on n-RASP-L tasks (addition, parity, copy) by adding iterations. *Implemented:* `arch: looped` (prelude / tied core / coda, input injection, `--iters-range`, test-time `--iters`). https://arxiv.org/abs/2409.15647
- **Looped Transformers are Better at Learning Learning Algorithms** (Yang et al., ICLR 2024), and **Looped Transformers as Programmable Computers** (Giannou et al., 2023): expressivity arguments for looping.
- **Recurrent-depth latent reasoning** (Geiping et al., 2025): scaling test-time compute through recurrence in latent space rather than tokens.

## Relational and algorithmic reasoning
- **Relation Networks** (Santoro et al., 2017), **Neural Algorithmic Reasoning** (Veličković & Blundell, 2021), **A Generalist Neural Algorithmic Learner** (Ibarz et al., 2022): processors aligned with algorithm steps; per-step hint supervision. Relevant to per-iteration auxiliary targets.
- **CLUTRR** (Sinha et al., 2019): kinship composition benchmark. `worlds.py` kinship is an in-house analogue (generated, no data reused).
- **StepGame** (Shi et al., 2022): multi-hop spatial benchmark; the adapter exists in `scripts/prepare_stepgame.py`.

## Chess
- **Amortized Planning with Large-Scale Transformers: A Case Study on Chess** (Ruoss et al., NeurIPS 2024; `google-deepmind/searchless_chess`). ChessBench: 10M games with Stockfish-16 action values. Models up to 270M parameters reach Lichess blitz Elo 2895 without search, and **action-value prediction beat behavioural cloning**. Implications: (1) our soft MultiPV targets are a behavioural-cloning variant, so consider binned action values per move; (2) strength rises steeply with scale and data, so a 32M model trained on 150k positions should be expected to be weak. https://arxiv.org/pdf/2402.04494

## Calibration
- **On Calibration of Modern Neural Networks** (Guo et al., 2017): temperature scaling. Our C-1 shows its limit when miscalibration is non-uniform.
- Non-global recalibration (e.g. Dirichlet calibration, Kull et al., 2019) and conformal abstention are candidates for the CALIBRATION lineage.

## Open System-One-style projects (surveyed from `cobanov/awesome-jev`)
- **Common pattern:** a large *frozen pretrained* backbone (Qwen3-1.7B in NanoJev and minojev, Qwen2.5-0.5B in kev, Qwen3.5-0.8B in JevForge, DiffusionGemma 26B-A4B in OpenJev) plus a small decision head (~0.8M parameters in NanoJev). General knowledge comes from the backbone. https://github.com/TianyuCodings/NanoJev · https://github.com/razorback16/openjev (Apache-2.0 base model)
- **AnyJev:** cyclic-shift marginalization over option orders reduces **option-order sensitivity**. We measured the same flaw in our models (below) and fixed it architecturally instead. https://github.com/MorrisZJ/AnyJev
- **Jevlike:** trainable encoder with an option-attention head over variable candidate sets (listwise scoring). https://github.com/vinnylarouge/jevlike
- **openJev Verdict 2.0:** ModernBERT with separate distribution and confidence heads. https://github.com/Heman10x-NGU/openJev-verdict-2.0
- **poorjev:** temperature scaling plus conformal abstention on NLI models. https://github.com/rupeshpoojary9/poorjev
- **PlayJev** and the reported Jev Doom demo: interactive game control with typed options per frame, which supports the interactive-agent direction.

## Strategic question raised (needs the lab owner's decision)
Every competitive open reproduction relies on a large pretrained backbone. Our founding principle is *random weights, no inherited model weights* (README design priority 3). Our 32M LM, pretrained on a FineWeb-Edu slice, carries far less world knowledge. For *reasoning* research the clean-room model is right. For a *general-purpose* decision model, a licensed open backbone (e.g. Apache-2.0) would be a different product direction. This is recorded as an open decision, not acted on.
