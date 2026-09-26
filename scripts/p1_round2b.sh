#!/bin/sh
# P1 round 2b (waits for S30 seed 8): loops that keep computing past the trained depth.
# From S30, +6000 updates, same data/seed/gates as ctl. Training: batches from one hop bucket h,
# K ~ U(h, h+4), decision loss at every pass t >= h (deep supervision). Key metric: does unseen-hop
# accuracy rise with passes past 6? eval_depth K in 1..16 (+ oracle K = hops+2) and probes to K=16.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
until grep -q "S30_s8 done" reports/p0/status.txt 2>/dev/null; do sleep 60; done
# Same data as ctl (spatial_only_v1).
INIT=checkpoints/p0/S30_s7.pt; DATA=data/processed/spatial_only_v1/train.jsonl; OUT=checkpoints/p1; REP=reports/p1
$PY scripts/check_contamination.py $DATA > $REP/contamination_r2b.json || { echo contaminated; exit 1; }
for a in r2b1_depthK r2b2_edge r2b3_incoff r2b4_combo; do
  [ -f reports/pt/PT2_RUNNING ] && while [ -f reports/pt/PT2_RUNNING ]; do sleep 120; done  # PT-2 has priority on the GPU
  $PY scripts/train_decision.py --config configs/p1/$a.yaml --init $INIT --tokenizer data/tokenizer.json --data $DATA \
    --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 3000 --seed 7 --hop-bucketed 0,4 --eval-every 500 \
    --out $OUT/${a}_s7.pt > $REP/train_${a}_s7.log 2>&1 || echo "${a}_s7 exited $?" >> $REP/status.txt
  [ -f $OUT/${a}_s7.killed.json ] && { echo "${a}_s7 KILLED" >> $REP/status.txt; continue; }
  $PY scripts/eval_depth.py --ckpt $OUT/${a}_s7.pt --out $REP/depth_${a}_s7.json --iters 1,2,4,6,8,12,16 --hop-offset 2 > $REP/depth_${a}_s7.txt 2>&1
  $PY scripts/probe_loop_states.py --ckpt $OUT/${a}_s7.pt --out $REP/probe_${a}_s7.json --k-max 16 > $REP/probe_${a}_s7.txt 2>&1
  $PY scripts/run_gates.py --ckpt $OUT/${a}_s7.pt --out $REP/gates_${a}_s7.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_${a}_s7.txt 2>&1
  echo "${a}_s7 done" >> $REP/status.txt
done
echo p1 round2b done
