#!/bin/sh
# Owner brief (28 Sep), item (e) after the remaining round-2b arms: GENERAL-4, then S60. PT-3 was skipped (PT-2 LOSS).
# Pre-registered in the ledger before launch. Holds GPU_BUSY from start to end; never overlaps another GPU job.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
OUT=checkpoints/general; mkdir -p reports/general/v4 reports/general/s60
until grep -qE "r2b4_combo_s7 (done|KILLED)|r2b4_combo_s7 exited" reports/p1/status.txt 2>/dev/null; do sleep 60; done
while [ -f reports/general/GPU_BUSY ] || [ -f reports/p1/ARM_RUNNING ] || [ -f reports/pt/PT2_RUNNING ]; do sleep 30; done
touch reports/general/GPU_BUSY
# ---- GENERAL-4: GENERAL-3's recipe at 50k updates (seed 7; seed 8 only if seed 7 clears the domain floor)
REP=reports/general/v4; INIT=$(tail -1 reports/general/v2/s30_promotion.txt); DATA=data/processed/general_v3/train.jsonl
SHARES=spatial=0.35,infogather=0.20,probability=0.135,dependency=0.0715,kinship=0.0715,temporal=0.0715,rules=0.0715,causal=0.029
echo "start $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
if $PY scripts/check_contamination.py $DATA > $REP/contamination.json; then
  for s in 7 8; do
    if [ $s = 8 ] && ! grep -q "s7 floor PASS" $REP/status.txt; then echo "s8 skipped (s7 failed the domain floor)" >> $REP/status.txt; break; fi
    $PY scripts/train_decision.py --config configs/general_v1.yaml --init $INIT --tokenizer data/tokenizer.json --data $DATA \
      --batch 16 --steps 50000 --domain-shares $SHARES --pair-consistency 0.5 --label-smoothing 0.05 --diagnostics --save-every 5000 \
      --seed $s --iters-range 1,6 --eval-every 500 --resume --out $OUT/general-v4_s$s.pt > $REP/train_s$s.log 2>&1 || echo "s$s exited $?" >> $REP/status.txt
    [ -f $OUT/general-v4_s$s.killed.json ] && { echo "s$s KILLED" >> $REP/status.txt; break; }
    $PY scripts/eval_worlds.py --ckpt $OUT/general-v4_s$s.pt --dir data/processed/worlds_v2 --out $REP/worlds_v2_s$s.json --limit 400 > $REP/worlds_v2_s$s.txt 2>&1
    $PY scripts/eval_depth.py --ckpt $OUT/general-v4_s$s.pt --out $REP/depth_s$s.json --iters 1,2,4,6,8,12 > $REP/depth_s$s.txt 2>&1
    $PY scripts/run_gates.py --ckpt $OUT/general-v4_s$s.pt --out $REP/gates_s$s.json --stress stress=reports/binding_stress/stress.jsonl \
      --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_s$s.txt 2>&1
    echo "s$s done" >> $REP/status.txt
    if [ $s = 7 ]; then $PY scripts/compare_general.py g4 > $REP/verdict_s7.txt 2>&1; grep -q "PASS  no domain < G1 - 0.03" $REP/verdict_s7.txt && echo "s7 floor PASS" >> $REP/status.txt; fi
  done
  $PY scripts/compare_general.py g4 > $REP/verdict.txt 2>&1
else echo contaminated >> $REP/status.txt; fi
echo "general_v4 done $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
# ---- S60: S30_s8 + 30k spatial-only updates (exposure control for GENERAL-2's spatial gain)
REP=reports/general/s60; DATA=data/processed/spatial_only_v1/train.jsonl
echo "start $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
if $PY scripts/check_contamination.py $DATA > $REP/contamination.json; then
  $PY scripts/train_decision.py --config configs/general_v1.yaml --init checkpoints/p0/S30_s8.pt --tokenizer data/tokenizer.json --data $DATA \
    --batch 16 --steps 30000 --balanced-sampling --diagnostics --save-every 5000 --seed 7 --iters-range 1,6 --eval-every 500 --resume \
    --out $OUT/s60_s7.pt > $REP/train_s7.log 2>&1 || echo "s7 exited $?" >> $REP/status.txt
  if [ ! -f $OUT/s60_s7.killed.json ]; then
    $PY scripts/eval_depth.py --ckpt $OUT/s60_s7.pt --out $REP/depth_s7.json --iters 1,2,4,6,8,12 > $REP/depth_s7.txt 2>&1
    $PY scripts/run_gates.py --ckpt $OUT/s60_s7.pt --out $REP/gates_s7.json --stress stress=reports/binding_stress/stress.jsonl \
      --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_s7.txt 2>&1
    $PY scripts/compare_general.py s60 > $REP/verdict.txt 2>&1; echo "s7 done" >> $REP/status.txt
  else echo "s7 KILLED" >> $REP/status.txt; fi
else echo contaminated >> $REP/status.txt; fi
rm -f reports/general/GPU_BUSY
echo "s60 done $(date '+%Y-%m-%d %H:%M')" >> $REP/status.txt
