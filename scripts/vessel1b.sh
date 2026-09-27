#!/bin/sh
# VESSEL-1b (owner, 27 Sep): after PT-2 150M, BEFORE the remaining round-2b arms. Same init (general-v1), same
# pre-registered criteria, robustness benchmark and Rule-4 gates as VESSEL-1, but with a retention mix
# (vessel 50%, spatial 20%, the 7 world domains 30%) so general skills are not forgotten.
# Holds reports/general/GPU_BUSY from launch (round 2b is already paused by PT2_RUNNING; nothing else waits on
# GPU_BUSY now), so round 2b cannot start between PT-2 and VESSEL-1b.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
D=data/processed/vessel_v1b; E=data/processed/vessel_v1; OUT=checkpoints/vessel; REP=reports/vessel/v1b; mkdir -p $OUT $REP
touch reports/general/GPU_BUSY
until grep -q "pt2 (all) done" reports/pt/pt2/status.txt 2>/dev/null; do sleep 120; done
while [ -f reports/pt/PT2_RUNNING ] || [ -f reports/p1/ARM_RUNNING ]; do sleep 30; done
echo "start $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
if $PY scripts/check_contamination.py $D/train.jsonl > $REP/contamination.json; then
  $PY scripts/train_decision.py --config configs/general_v1.yaml --init checkpoints/general/general-v1.pt --tokenizer data/tokenizer.json \
    --data $D/train.jsonl --batch 16 --steps 12000 --domain-shares vessel=0.5,spatial=0.2 --diagnostics --save-every 3000 --seed 7 \
    --iters-range 1,6 --eval-every 500 --resume --out $OUT/vessel-v1b_s7.pt > $REP/train_s7.log 2>&1 || echo "s7 exited $?" >> $REP/status.txt
  if [ -f $OUT/vessel-v1b_s7.killed.json ]; then echo "s7 KILLED (Rule 4)" >> $REP/status.txt
  else
    $PY scripts/eval_vessel.py --ckpt $OUT/vessel-v1b_s7.pt --dir $E --out $REP/eval_s7.json > $REP/verdict_s7.txt 2>&1
    $PY scripts/eval_vessel.py --robust --ckpt $OUT/vessel-v1b_s7.pt --dir $E --out $REP/robust_s7.json > $REP/robust_s7.txt 2>&1
    $PY scripts/run_gates.py --ckpt $OUT/vessel-v1b_s7.pt --out $REP/gates_s7.json --stress stress=reports/binding_stress/stress.jsonl \
      --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_s7.txt 2>&1
    echo "s7 done" >> $REP/status.txt
  fi
else echo "contaminated" >> $REP/status.txt; fi
rm -f reports/general/GPU_BUSY
echo "vessel_v1b done $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
