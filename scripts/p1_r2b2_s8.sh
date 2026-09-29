#!/bin/sh
# r2b2_edge seed 8 (owner, 29 Sep): Rule-5 replication of the r2b2 seed-7 result. Identical recipe to seed 7 (init S30_s7,
# +6000 updates, spatial_only_v1, hop-bucketed deep supervision), only --seed changes. Slot: after the GENERAL repair queue,
# before PT-3 resumes. It takes reports/p1/ARM_RUNNING once the repair holds GPU_BUSY (the repair never re-checks markers
# after acquiring; PT-3's runner waits for every marker), then trains only after the repair releases the GPU.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
a=r2b2_edge; INIT=checkpoints/p0/S30_s7.pt; DATA=data/processed/spatial_only_v1/train.jsonl; OUT=checkpoints/p1; REP=reports/p1
until [ -f reports/general/GPU_BUSY ] || [ -f reports/general/repair_20260929/COMPLETE ]; do sleep 20; done
touch reports/p1/ARM_RUNNING; echo "$(date -u +%FT%TZ) ${a}_s8 queued behind GENERAL repair (holds ARM_RUNNING)" >> $REP/status.txt
while [ -f reports/general/GPU_BUSY ] || [ -f reports/pt/PT2_RUNNING ]; do sleep 60; done
$PY scripts/check_contamination.py $DATA > $REP/contamination_r2b2_s8.json || { echo "${a}_s8 contaminated" >> $REP/status.txt; rm -f reports/p1/ARM_RUNNING; exit 1; }
$PY scripts/train_decision.py --config configs/p1/$a.yaml --init $INIT --tokenizer data/tokenizer.json --data $DATA \
  --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 3000 --seed 8 --hop-bucketed 0,4 --eval-every 500 --resume \
  --out $OUT/${a}_s8.pt > $REP/train_${a}_s8.log 2>&1 || echo "${a}_s8 exited $?" >> $REP/status.txt
if [ -f $OUT/${a}_s8.killed.json ]; then echo "${a}_s8 KILLED" >> $REP/status.txt; else
  $PY scripts/eval_depth.py --ckpt $OUT/${a}_s8.pt --out $REP/depth_${a}_s8.json --iters 1,2,4,6,8,12,16 --hop-offset 2 > $REP/depth_${a}_s8.txt 2>&1
  $PY scripts/probe_loop_states.py --ckpt $OUT/${a}_s8.pt --out $REP/probe_${a}_s8.json --k-max 16 > $REP/probe_${a}_s8.txt 2>&1
  $PY scripts/run_gates.py --ckpt $OUT/${a}_s8.pt --out $REP/gates_${a}_s8.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_${a}_s8.txt 2>&1
  $PY scripts/eval_depth.py --ckpt $OUT/${a}_s8.pt --chains $REP/attack_12_16/chains.jsonl --iters 6,8,12,16,20,24 --batch-size 4 \
    --out $REP/attack_12_16/${a}_s8.json > $REP/attack_12_16/${a}_s8.log 2>&1
  echo "${a}_s8 done" >> $REP/status.txt
fi
rm -f reports/p1/ARM_RUNNING
