#!/bin/sh
set -eu
export PYTHONPATH=src S1_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=./.venv/Scripts/python.exe
REP=reports/p1/attack_12_16
for arm in r2b1_depthK ctl; do for s in 7 8; do
  $PY scripts/eval_depth.py --ckpt checkpoints/p1/${arm}_s$s.pt --chains "$REP/chains.jsonl" \
    --iters 6,8,12,16,20,24 --batch-size 4 --out "$REP/${arm}_s$s.json" > "$REP/${arm}_s$s.log" 2>&1
done; done
echo 'attack done' > "$REP/status.txt"
