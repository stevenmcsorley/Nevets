#!/bin/sh
# P1 round 2 (waits for round 1): A-warm = readout-only initialised from C's trained coordinate head
# (NOT matched: 12k total updates); S30 seed 8 = Rule-5 replication of the P0 winner before promotion.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
until grep -q "D_hybrid_hints_s7" reports/p1/status.txt 2>/dev/null; do sleep 60; done
DATA=data/processed/spatial_only_v1/train.jsonl; OUT=checkpoints/p1; REP=reports/p1
if [ -f $OUT/C_consistency_s7.pt ] && [ ! -f $OUT/C_consistency_s7.killed.json ]; then
  $PY scripts/train_decision.py --config configs/p1/A_only.yaml --init $OUT/C_consistency_s7.pt --tokenizer data/tokenizer.json --data $DATA \
    --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 3000 --seed 7 --iters-range 1,6 --eval-every 500 \
    --out $OUT/A_warm_s7.pt > $REP/train_A_warm_s7.log 2>&1 || echo "A_warm_s7 exited $?" >> $REP/status.txt
  if [ ! -f $OUT/A_warm_s7.killed.json ]; then
    $PY scripts/eval_depth.py --ckpt $OUT/A_warm_s7.pt --out $REP/depth_A_warm_s7.json --iters 1,2,4,6,8,12 > $REP/depth_A_warm_s7.txt 2>&1
    $PY scripts/run_gates.py --ckpt $OUT/A_warm_s7.pt --out $REP/gates_A_warm_s7.json --stress stress=reports/binding_stress/stress.jsonl \
      --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_A_warm_s7.txt 2>&1
    echo "A_warm_s7 done" >> $REP/status.txt
  fi
fi
$PY scripts/train_decision.py --config configs/general_v1.yaml --init checkpoints/tournament/r2/looped.pt --tokenizer data/tokenizer.json \
  --data $DATA --batch 16 --steps 30000 --balanced-sampling --diagnostics --save-every 3000 --seed 8 --iters-range 1,6 --eval-every 500 \
  --out checkpoints/p0/S30_s8.pt > reports/p0/train_S30_s8.log 2>&1 || echo "S30_s8 exited $?" >> reports/p0/status.txt
$PY scripts/eval_depth.py --ckpt checkpoints/p0/S30_s8.pt --out reports/p1/depth_S30_s8.json --iters 1,2,4,6,8,12 > reports/p1/depth_S30_s8.txt 2>&1
$PY scripts/run_gates.py --ckpt checkpoints/p0/S30_s8.pt --out reports/p0/gates_S30_s8.json --stress stress=reports/binding_stress/stress.jsonl \
  --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > reports/p0/gates_S30_s8.txt 2>&1
echo "S30_s8 done" >> reports/p0/status.txt
echo p1 round2 done
