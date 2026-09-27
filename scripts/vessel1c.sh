#!/bin/sh
# VESSEL-1c (owner, 27 Sep evening: "train on Claude labels ... get off simulation asap"): Nevets on live tracks labelled
# by Claude, with the VESSEL-1b retention mix. Queued straight after VESSEL-1b, BEFORE the remaining round-2b arms
# (takes GPU_BUSY the moment VESSEL-1b releases it). Pre-registered in the ledger before launch. Data frozen at launch
# (scripts/build_vessel_c.py); eval on the frozen held-out vessels (data/processed/vessel_c/eval_real.jsonl).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
D=data/processed/vessel_c; OUT=checkpoints/vessel; REP=reports/vessel/v1c; mkdir -p $OUT $REP
echo "queued $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
until grep -qE "s7 done|s7 KILLED|s7 exited|contaminated" reports/vessel/v1b/status.txt 2>/dev/null; do sleep 60; done
until grep -q "vessel_v1b done" reports/vessel/v1b/status.txt 2>/dev/null; do sleep 1; done
touch reports/general/GPU_BUSY
while [ -f reports/pt/PT2_RUNNING ] || [ -f reports/p1/ARM_RUNNING ]; do sleep 30; done
echo "start $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
STEPS=$($PY scripts/build_vessel_c.py) || { echo "build failed" >> $REP/status.txt; rm -f reports/general/GPU_BUSY; exit 1; }
echo "steps $STEPS; manifest $(tr -d '\n ' < $D/manifest.json | cut -c1-300)" >> $REP/status.txt
if $PY scripts/check_contamination.py $D/train.jsonl > $REP/contamination.json; then
  $PY scripts/train_decision.py --config configs/general_v1.yaml --init checkpoints/general/general-v1.pt --tokenizer data/tokenizer.json \
    --data $D/train.jsonl --batch 16 --steps $STEPS --domain-shares vessel_real=0.3,vessel=0.2,spatial=0.2 --diagnostics --save-every 1000 \
    --seed 7 --iters-range 1,6 --eval-every 500 --resume --out $OUT/vessel-v1c_s7.pt > $REP/train_s7.log 2>&1 || echo "s7 exited $?" >> $REP/status.txt
  if [ -f $OUT/vessel-v1c_s7.killed.json ]; then echo "s7 KILLED (Rule 4)" >> $REP/status.txt
  else
    $PY scripts/eval_vessel.py --real --ckpt $OUT/vessel-v1c_s7.pt --dir $D --out $REP/real_s7.json > $REP/real_s7.txt 2>&1
    $PY scripts/eval_vessel.py --ckpt $OUT/vessel-v1c_s7.pt --dir data/processed/vessel_v1 --out $REP/eval_s7.json > $REP/verdict_synthetic_s7.txt 2>&1
    $PY scripts/run_gates.py --ckpt $OUT/vessel-v1c_s7.pt --out $REP/gates_s7.json --stress stress=reports/binding_stress/stress.jsonl \
      --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_s7.txt 2>&1
    echo "s7 done" >> $REP/status.txt
  fi
else echo "contaminated" >> $REP/status.txt; fi
rm -f reports/general/GPU_BUSY
echo "vessel_v1c done $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
