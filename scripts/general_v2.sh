#!/bin/sh
# GENERAL-2 (owner request, 26 Sep): from promoted S30, on GENERAL-1's multi-domain mix with spatial kept at a
# 50% sampling share (domain first, then label within domain: SAMPLING-1 fix). Two seeds, same init.
# Holds reports/general/GPU_BUSY while training so P1 round 2b only fills GPU gaps; PT-2 follows GENERAL-2.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
OUT=checkpoints/general; REP=reports/general/v2; mkdir -p $OUT $REP
touch reports/general/GPU_BUSY
until grep -q "S30_s8 done" reports/p0/status.txt 2>/dev/null; do sleep 60; done
# S30 promotion evidence (two seeds) + choice of init on dev chains (K=4 overall; dev only, never locked sets).
$PY scripts/compare_general.py s30 > $REP/s30_promotion.txt 2>&1 || { echo "S30 not promotable" >> $REP/status.txt; rm -f reports/general/GPU_BUSY; exit 1; }
INIT=$(tail -1 $REP/s30_promotion.txt)
DATA=data/processed/general_v2/train.jsonl
$PY scripts/check_contamination.py $DATA > $REP/contamination.json || { echo contaminated >> $REP/status.txt; rm -f reports/general/GPU_BUSY; exit 1; }
for s in 7 8; do
  $PY scripts/train_decision.py --config configs/general_v1.yaml --init $INIT --tokenizer data/tokenizer.json --data $DATA \
    --batch 16 --steps 30000 --domain-shares spatial=0.5 --diagnostics --save-every 5000 --seed $s --iters-range 1,6 --eval-every 500 \
    --out $OUT/general-v2_s$s.pt > $REP/train_s$s.log 2>&1 || echo "s$s exited $?" >> $REP/status.txt
  [ -f $OUT/general-v2_s$s.killed.json ] && { echo "s$s KILLED" >> $REP/status.txt; continue; }
  $PY scripts/eval_worlds.py --ckpt $OUT/general-v2_s$s.pt --dir data/processed/worlds_v2 --out $REP/worlds_v2_s$s.json --limit 400 > $REP/worlds_v2_s$s.txt 2>&1
  $PY scripts/eval_depth.py --ckpt $OUT/general-v2_s$s.pt --out $REP/depth_s$s.json --iters 1,2,4,6,8,12 > $REP/depth_s$s.txt 2>&1
  $PY scripts/run_gates.py --ckpt $OUT/general-v2_s$s.pt --out $REP/gates_s$s.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_s$s.txt 2>&1
  $PY scripts/probe_loop_states.py --ckpt $OUT/general-v2_s$s.pt --out $REP/probe_s$s.json > $REP/probe_s$s.txt 2>&1
  echo "s$s done" >> $REP/status.txt
done
# GENERAL-1 depth profile on the same eval, if missing (baseline for the spatial comparison).
[ -f reports/p1/depth_general-v1.json ] || $PY scripts/eval_depth.py --ckpt $OUT/general-v1.pt --out reports/p1/depth_general-v1.json --iters 1,2,4,6,8,12
$PY scripts/compare_general.py g2 > $REP/verdict.txt 2>&1
rm -f reports/general/GPU_BUSY; echo "general_v2 done" >> $REP/status.txt
