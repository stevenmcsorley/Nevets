#!/bin/sh
# VESSEL-1 (owner, 27 Sep): fine-tune the best promoted checkpoint on the Channel Watch vessel dataset, one seed,
# ~1 h, in the slot after round-2b seed 8 and BEFORE PT-2 150M. This script owns reports/pt/HOLD from that point
# and releases it when done (or deferred), so PT-2 150M starts afterwards. Kept out of GENERAL decisions.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
D=data/processed/vessel_v1; OUT=checkpoints/vessel; REP=reports/vessel/v1; mkdir -p $OUT $REP
until grep -q "VESSEL-1 slot open" reports/pt/pt2/status.txt 2>/dev/null && [ ! -f reports/general/GPU_BUSY ]; do sleep 120; done
touch reports/general/GPU_BUSY
echo "slot open $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
# The dataset comes from channel-watch Phase 2. Wait up to 4 h for it; otherwise defer and free the GPU for PT-2.
i=0; while [ ! -f $D/READY ] && [ $i -lt 240 ]; do sleep 60; i=$((i+1)); done
if [ ! -f $D/READY ]; then
  echo "VESSEL-1 deferred: dataset not ready after 4 h (runs in a later gap)" >> $REP/status.txt
else
  INIT=$(cat $REP/init.txt 2>/dev/null || echo checkpoints/general/general-v1.pt)  # best promoted (BEST_GENERAL) at launch
  echo "init $INIT" >> $REP/status.txt
  if $PY scripts/check_contamination.py $D/train.jsonl > $REP/contamination.json; then
    $PY scripts/train_decision.py --config configs/general_v1.yaml --init $INIT --tokenizer data/tokenizer.json --data $D/train.jsonl \
      --batch 16 --steps 12000 --diagnostics --save-every 3000 --seed 7 --iters-range 1,6 --eval-every 500 --resume \
      --out $OUT/vessel-v1_s7.pt > $REP/train_s7.log 2>&1 || echo "s7 exited $?" >> $REP/status.txt
    if [ -f $OUT/vessel-v1_s7.killed.json ]; then echo "s7 KILLED (Rule 4)" >> $REP/status.txt
    else $PY scripts/eval_vessel.py --ckpt $OUT/vessel-v1_s7.pt --dir $D --out $REP/eval_s7.json > $REP/verdict_s7.txt 2>&1; echo "s7 done" >> $REP/status.txt; fi
  else echo "contaminated: training states overlap a registered eval file" >> $REP/status.txt; fi
fi
rm -f reports/general/GPU_BUSY
rm -f reports/pt/HOLD   # releases PT-2 150M
echo "released PT-2 150M hold after VESSEL-1 $(date '+%Y-%m-%d %H:%M')" >> reports/pt/pt2/status.txt
