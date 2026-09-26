#!/bin/sh
# After P1 round 1: VRAM ceiling re-test (idle GPU), loop-state probes for every arm, then round 2
# (A-warm, S30 seed 8, ctl seed 8) and their probes. Sequential GPU use.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
until grep -q "D_hybrid_hints_s7" reports/p1/status.txt 2>/dev/null; do sleep 60; done
$PY scripts/pt/vram_ceiling.py > reports/pt/vram_ceiling.log 2>&1; echo "vram_ceiling done" >> reports/p1/status.txt
probe() { [ -f $1 ] && $PY scripts/probe_loop_states.py --ckpt $1 --out reports/p1/probe_$2.json > reports/p1/probe_$2.txt 2>&1; }
probe checkpoints/p0/S30_s7.pt S30_s7
for a in ctl A_only B_hybrid C_consistency D_hybrid_hints; do [ -f checkpoints/p1/${a}_s7.killed.json ] || probe checkpoints/p1/${a}_s7.pt ${a}_s7; done
echo "probes round1 done" >> reports/p1/status.txt
DATA=data/processed/spatial_only_v1/train.jsonl; OUT=checkpoints/p1; REP=reports/p1
arm() {  # name config init seed
  $PY scripts/train_decision.py --config $2 --init $3 --tokenizer data/tokenizer.json --data $DATA --batch 16 --steps 6000 \
    --balanced-sampling --diagnostics --save-every 3000 --seed $4 --iters-range 1,6 --eval-every 500 --out $OUT/$1.pt > $REP/train_$1.log 2>&1 || echo "$1 exited $?" >> $REP/status.txt
  [ -f $OUT/$1.killed.json ] && { echo "$1 KILLED" >> $REP/status.txt; return 0; }
  $PY scripts/eval_depth.py --ckpt $OUT/$1.pt --out $REP/depth_$1.json --iters 1,2,4,6,8,12 > $REP/depth_$1.txt 2>&1
  $PY scripts/run_gates.py --ckpt $OUT/$1.pt --out $REP/gates_$1.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_$1.txt 2>&1
  probe $OUT/$1.pt $1; echo "$1 done" >> $REP/status.txt
}
arm ctl_s8 configs/p1/ctl.yaml checkpoints/p0/S30_s7.pt 8
[ -f $OUT/C_consistency_s7.pt ] && [ ! -f $OUT/C_consistency_s7.killed.json ] && arm A_warm_s7 configs/p1/A_only.yaml $OUT/C_consistency_s7.pt 7
$PY scripts/train_decision.py --config configs/general_v1.yaml --init checkpoints/tournament/r2/looped.pt --tokenizer data/tokenizer.json \
  --data $DATA --batch 16 --steps 30000 --balanced-sampling --diagnostics --save-every 3000 --seed 8 --iters-range 1,6 --eval-every 500 \
  --out checkpoints/p0/S30_s8.pt > reports/p0/train_S30_s8.log 2>&1
$PY scripts/eval_depth.py --ckpt checkpoints/p0/S30_s8.pt --out reports/p1/depth_S30_s8.json --iters 1,2,4,6,8,12 > reports/p1/depth_S30_s8.txt 2>&1
$PY scripts/run_gates.py --ckpt checkpoints/p0/S30_s8.pt --out reports/p0/gates_S30_s8.json --stress stress=reports/binding_stress/stress.jsonl \
  --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > reports/p0/gates_S30_s8.txt 2>&1
probe checkpoints/p0/S30_s8.pt S30_s8; echo "S30_s8 done" >> reports/p0/status.txt
echo p1 queue2 done
