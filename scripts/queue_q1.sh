#!/bin/sh
# GPU queue 1: waits for tournament r2 to finish, then ISO-A, ISO-B, TM-1, CH-2 (sequential).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
until [ -s reports/tournament/r2/gates_looped_k12.txt ]; do sleep 60; done
DATA=data/processed/tournament/chain_h1_6_train.jsonl; OUT=checkpoints/tournament/iso; REP=reports/tournament/iso; mkdir -p $OUT $REP
iso() {  # arm config extra
  $PY scripts/train_decision.py --config configs/tournament/$2.yaml --init checkpoints/exp6a-noloop.pt --tokenizer data/tokenizer.json \
    --data $DATA --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 2000 --seed 42 --out $OUT/$1.pt $3 > $REP/train_$1.log 2>&1
  $PY scripts/run_gates.py --ckpt $OUT/$1.pt --out $REP/gates_$1.json --stress hops=reports/chain/eval_hops.jsonl \
    --stress stress=reports/binding_stress/stress.jsonl > $REP/gates_$1.txt 2>&1
}
iso aug iso_aug "--shuffle-options"
iso isolated iso_isolated ""
sh scripts/transfer_tm1.sh base spatial scratch
sh scripts/chess_r1.sh spatial
echo queue_q1 done
