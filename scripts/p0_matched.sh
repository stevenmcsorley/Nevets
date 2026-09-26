#!/bin/sh
# P0: settle the GENERAL-1 caveat (training length vs curriculum). Same init/config/optimiser/batch.
set -e
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
INIT=checkpoints/tournament/r2/looped.pt; OUT=checkpoints/p0; REP=reports/p0; mkdir -p $OUT $REP
SPATIAL=data/processed/spatial_only_v1/train.jsonl; MULTI=data/processed/general_v1/train.jsonl
arm() {  # name data steps seed
  $PY scripts/check_contamination.py $2 > $REP/contamination_$1.json
  $PY scripts/train_decision.py --config configs/general_v1.yaml --init $INIT --tokenizer data/tokenizer.json --data $2 \
    --batch 16 --steps $3 --balanced-sampling --diagnostics --save-every 3000 --seed $4 --iters-range 1,6 \
    --eval-every 500 --out $OUT/$1.pt > $REP/train_$1.log 2>&1 || echo "$1 exited $?" >> $REP/status.txt
  [ -f $OUT/$1.killed.json ] && return 0
  $PY scripts/run_gates.py --ckpt $OUT/$1.pt --out $REP/gates_$1.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_$1.txt 2>&1
  case "$1" in M*) $PY scripts/eval_worlds.py --ckpt $OUT/$1.pt --dir data/processed/worlds_v2 --out $REP/worlds_$1.json --limit 400 > $REP/worlds_$1.txt 2>&1;; esac
  echo "$1 done" >> $REP/status.txt
}
arm S6_s7 $SPATIAL 6000 7
arm M6_s7 $MULTI 6000 7
arm S6_s8 $SPATIAL 6000 8
arm M6_s8 $MULTI 6000 8
arm S30_s7 $SPATIAL 30000 7
echo p0 done
