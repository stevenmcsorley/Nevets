#!/bin/sh
# Tournament round 3: make the loop converge so more test-time iterations refine instead of drift.
# Same init/data/steps/seed/K-range as T-R2 looped; R3a adds no-grad warm-up iterations,
# R3b adds warm-up + a fixed-point loss. Waits for GPU queue 1 to finish (sequential GPU use).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
until [ -s reports/chess/r1/eval_spatial.json ]; do sleep 60; done
DATA=data/processed/tournament/chain_h1_6_train.jsonl; OUT=checkpoints/tournament/r3; REP=reports/tournament/r3; mkdir -p $OUT $REP
arm() {  # name config
  $PY scripts/train_decision.py --config configs/tournament/$2.yaml --init checkpoints/exp6a-noloop.pt --tokenizer data/tokenizer.json \
    --data $DATA --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 2000 --seed 42 \
    --iters-range 1,6 --nograd-range 0,6 --out $OUT/$1.pt > $REP/train_$1.log 2>&1
  for k in 1 2 4 6 8 12 16; do
    $PY scripts/run_gates.py --ckpt $OUT/$1.pt --out $REP/gates_$1_k$k.json --stress hops=reports/chain/eval_hops.jsonl --iters $k > $REP/gates_$1_k$k.txt 2>&1
  done
}
arm nograd r3_nograd
arm converge r3_converge
echo r3 done
