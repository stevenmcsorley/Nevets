# Research basis (2026-09-24)

Public TypeSafe documentation describes Jev 1.13 as a System One model accepting one state and isolated typed Choice/Score/Noul questions; it ingests the state once and evaluates questions in parallel. The current context limits are 64k aggregate request and ~32k state plus longest question. TypeSafe describes RLCD as post-training a pretrained language model to produce calibrated decisions rather than generated text.

Independent work by Archer Hume argues, from pre-2026-09-19 black-box probes and public evidence, for direct readout, shared state, isolated question branches, joint/listwise option processing and a likely causal transformer. The sparse-MoE hypothesis is explicitly less certain.

Kev is an independent open implementation demonstrating a prefill-only causal model with shared state, isolated branches and a pointer readout. Kev uses pretrained Qwen weights + LoRA, so it is architectural evidence, not our from-scratch training recipe.

Legal boundary: TypeSafe's Master Customer Agreement updated 2026-09-19 prohibits using the service/output for model distillation, imitation, competing-product development, or reverse engineering. This lab therefore does not query Jev as an oracle or use its outputs as labels.
