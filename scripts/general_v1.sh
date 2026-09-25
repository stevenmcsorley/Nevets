#!/bin/sh
# GENERAL-1: first multi-domain decision model. Every mechanism that survived a controlled test:
# bidirectional state, entity binding, isolated options, trained looped core (K~U[1,6]).
# Data: worlds_v2 (7 domains, table held out) + spatial chains + one-hop retention, checked clean.
#   sh scripts/general_v1.sh <init-checkpoint> [extra train args]
set -e
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe; INIT=$1; shift
OUT=checkpoints/general; REP=reports/general/v1; mkdir -p $OUT $REP
$PY scripts/check_contamination.py data/processed/general_v1/train.jsonl
$PY scripts/train_decision.py --config configs/general_v1.yaml --init $INIT --tokenizer data/tokenizer.json \
  --data data/processed/general_v1/train.jsonl --batch 16 --steps 30000 --balanced-sampling --diagnostics \
  --save-every 5000 --seed 7 --iters-range 1,6 --out $OUT/general-v1.pt "$@" > $REP/train.log 2>&1
$PY scripts/eval_worlds.py --ckpt $OUT/general-v1.pt --dir data/processed/worlds_v2 --out $REP/worlds_v2.json --limit 400 > $REP/worlds_v2.txt 2>&1
$PY scripts/run_gates.py --ckpt $OUT/general-v1.pt --out $REP/gates.json --stress stress=reports/binding_stress/stress.jsonl \
  --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates.txt 2>&1
echo general-v1 done
