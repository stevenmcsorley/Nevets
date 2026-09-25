#!/bin/sh
# Chess round 1: BASE->CHESS (LM init) vs SPATIAL->CHESS (exp6a-noloop init), identical data/steps.
#   sh scripts/chess_r1.sh base | spatial
set -e
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
DATA=data/processed/chess/train_2013-01.jsonl; OUT=checkpoints/chess; REP=reports/chess/r1; mkdir -p $OUT $REP
case "$1" in
  base) INIT="--init checkpoints/s1-35m-pretrain.pt --allow-tokenizer-mismatch";;
  spatial) INIT="--init checkpoints/exp6a-noloop.pt";;
esac
$PY scripts/train_decision.py --config configs/s1_35m_chess.yaml $INIT --tokenizer data/tokenizer.json --data $DATA \
  --batch 16 --steps 18000 --diagnostics --save-every 3000 --seed 7 --out $OUT/chess-$1-r1.pt > $REP/train_$1.log 2>&1
$PY scripts/eval_chess.py --ckpt $OUT/chess-$1-r1.pt --data data/processed/chess/eval_2013-02.jsonl \
  --out $REP/eval_$1.json --games 20 > $REP/eval_$1.txt 2>&1
echo "$1 done"
