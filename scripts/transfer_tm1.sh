#!/bin/sh
# Transfer matrix 1: worlds_v1 (5 domains; train prose/json/kv, table held out) from three inits.
# Same data/steps/seed. Arms: base (LM), spatial (exp6a-noloop), scratch (random d512 flat).
set -e
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
DATA=data/processed/worlds_v1/train.jsonl; OUT=checkpoints/transfer/tm1; REP=reports/worlds/v1; mkdir -p $OUT $REP
arm() {
  case "$1" in
    base) INIT="--init checkpoints/s1-35m-pretrain.pt --allow-tokenizer-mismatch";;
    spatial) INIT="--init checkpoints/exp6a-noloop.pt";;
    scratch) INIT="";;
  esac
  $PY scripts/train_decision.py --config configs/s1_35m_entity_binding_bidir.yaml $INIT --tokenizer data/tokenizer.json \
    --data $DATA --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 2000 --seed 11 \
    --out $OUT/$1.pt > $REP/train_$1.log 2>&1
  $PY scripts/eval_worlds.py --ckpt $OUT/$1.pt --dir data/processed/worlds_v1 --out $REP/eval_$1.json --limit 400 > $REP/eval_$1.txt 2>&1
}
for a in ${@:-base spatial scratch}; do arm $a; done
echo done
