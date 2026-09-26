#!/bin/sh
# P1: make decisions come from the latent world model. Every arm continues from the P0 winner S30 for
# 6000 updates on the same spatial-only data (1-6 hops), same seed/optimiser/gates; only the mechanism differs.
# Key test: eval_depth K sweep on 1-10 hops (7-10 never trained).
set -e
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
INIT=checkpoints/p0/S30_s7.pt; DATA=data/processed/spatial_only_v1/train.jsonl; OUT=checkpoints/p1; REP=reports/p1; mkdir -p $OUT $REP
arm() {  # name seed
  $PY scripts/check_contamination.py $DATA > $REP/contamination.json
  $PY scripts/train_decision.py --config configs/p1/$1.yaml --init $INIT --tokenizer data/tokenizer.json --data $DATA \
    --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 3000 --seed $2 --iters-range 1,6 \
    --eval-every 500 --out $OUT/$1_s$2.pt > $REP/train_$1_s$2.log 2>&1 || echo "$1_s$2 exited $?" >> $REP/status.txt
  [ -f $OUT/$1_s$2.killed.json ] && { echo "$1_s$2 KILLED" >> $REP/status.txt; return 0; }
  $PY scripts/eval_depth.py --ckpt $OUT/$1_s$2.pt --out $REP/depth_$1_s$2.json --iters 1,2,4,6,8,12 > $REP/depth_$1_s$2.txt 2>&1
  $PY scripts/run_gates.py --ckpt $OUT/$1_s$2.pt --out $REP/gates_$1_s$2.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_$1_s$2.txt 2>&1
  echo "$1_s$2 done" >> $REP/status.txt
}
for a in ctl A_only B_hybrid C_consistency D_hybrid_hints; do arm $a 7; done
echo p1 round1 done
