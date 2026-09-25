#!/bin/sh
# Tournament round 1 (REASONER question): from-scratch d=256 models on 1-6-hop chains.
# Identical data, seed, steps, batch and LR for every arm. Run from the repo root:
#   sh scripts/tournament_r1.sh lane1   (flat4, flat8)
#   sh scripts/tournament_r1.sh lane2   (loop, loopaux)
set -e
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe
DATA=data/processed/tournament/chain_h1_6_train.jsonl
OUT=checkpoints/tournament/r1; REP=reports/tournament/r1
mkdir -p $OUT $REP
train() {  # arm, extra args
  $PY scripts/train_decision.py --config configs/tournament/$1.yaml --tokenizer data/tokenizer.json \
    --data $DATA --batch 32 --steps 12000 --balanced-sampling --diagnostics --save-every 4000 --seed 42 \
    --out $OUT/$1.pt $2 > $REP/train_$1.log 2>&1
}
gate() {  # arm, iters (optional)
  suffix=${2:+_k$2}
  $PY scripts/run_gates.py --ckpt $OUT/$1.pt --out $REP/gates_$1$suffix.json \
    --stress hops=reports/chain/eval_hops.jsonl ${2:+--iters $2} > $REP/gates_$1$suffix.txt 2>&1
}
case "$1" in
  lane1) for a in flat4 flat8; do train $a ""; gate $a; done ;;
  lane2) for a in loop loopaux; do train $a "--iters-range 2,10"; for k in 2 4 6 8 12 16; do gate $a $k; done; done ;;
esac
echo "$1 done"
