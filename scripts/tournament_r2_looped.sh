#!/bin/sh
# Resume tournament r2 looped arm once the GPU is free of chess BASE (it spilled VRAM when run concurrently).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
until [ -s reports/chess/r1/eval_base.json ]; do sleep 60; done
DATA=data/processed/tournament/chain_h1_6_train.jsonl; OUT=checkpoints/tournament/r2; REP=reports/tournament/r2
$PY scripts/train_decision.py --config configs/tournament/r2_looped.yaml --init checkpoints/exp6a-noloop.pt \
  --tokenizer data/tokenizer.json --data $DATA --batch 16 --steps 6000 --balanced-sampling --diagnostics \
  --save-every 2000 --seed 42 --out $OUT/looped.pt --iters-range 1,6 > $REP/train_looped.log 2>&1
for k in 1 2 4 6 8 12; do
  $PY scripts/run_gates.py --ckpt $OUT/looped.pt --out $REP/gates_looped_k$k.json --stress hops=reports/chain/eval_hops.jsonl --iters $k > $REP/gates_looped_k$k.txt 2>&1
done
echo r2 looped done
