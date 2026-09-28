#!/bin/sh
# Existing LM checkpoints only. Own tokenizers; TM-1 decision fine-tune.
set -eu
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe
CK=$1; TOK=$2; NAME=$3; SEED=$4
LEGACY=""
[ "$CK" != checkpoints/s1-35m-pretrain.pt ] || LEGACY=--allow-tokenizer-mismatch
REP=reports/pt/probe2; OUT=checkpoints/pt2/probe2
mkdir -p "$REP" "$OUT"
if grep -qx "probe $NAME done" "$REP/status.txt" 2>/dev/null; then exit 0; fi
$PY scripts/train_decision.py --config configs/pt/probe_v2.yaml --init "$CK" --tokenizer "$TOK" $LEGACY \
  --data data/processed/probe2/train.jsonl --batch 16 --steps 6000 --balanced-sampling \
  --diagnostics --save-every 2000 --seed "$SEED" --eval-every 0 --resume --out "$OUT/$NAME.pt" > "$REP/probe_${NAME}_train.log" 2>&1
$PY scripts/eval_worlds.py --ckpt "$OUT/$NAME.pt" --dir data/processed/worlds_v1 \
  --out "$REP/probe_${NAME}_worlds.json" --limit 400 > "$REP/probe_${NAME}_worlds.txt" 2>&1
echo "probe $NAME done" >> "$REP/status.txt"
