#!/bin/sh
# Tournament round 2: restructure exp6a-noloop into prelude(0-1) / tied core(2-5) / coda(6-7).
# control: K=1 (identical to the flat model); looped: K ~ U[1,6]. Same data/steps/seed.
set -e
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
DATA=data/processed/tournament/chain_h1_6_train.jsonl; OUT=checkpoints/tournament/r2; REP=reports/tournament/r2
mkdir -p $OUT $REP
run() {  # arm, extra
  $PY scripts/train_decision.py --config configs/tournament/r2_$1.yaml --init checkpoints/exp6a-noloop.pt \
    --tokenizer data/tokenizer.json --data $DATA --batch 16 --steps 6000 --balanced-sampling --diagnostics \
    --save-every 2000 --seed 42 --out $OUT/$1.pt $2 > $REP/train_$1.log 2>&1
}
gate() { $PY scripts/run_gates.py --ckpt $OUT/$1.pt --out $REP/gates_$1_k$2.json --stress hops=reports/chain/eval_hops.jsonl --iters $2 > $REP/gates_$1_k$2.txt 2>&1; }
run control ""; for k in 1 2 4; do gate control $k; done
run looped "--iters-range 1,6"; for k in 1 2 4 6 8 12; do gate looped $k; done
echo done
