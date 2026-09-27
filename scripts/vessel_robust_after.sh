#!/bin/sh
# VESSEL-ROBUST: score vessel-v1_s7 on the robustness benchmark on CPU once VESSEL-1 has finished (no GPU use).
export PYTHONPATH=src S1_DEVICE=cpu; PY=./.venv/Scripts/python.exe; REP=reports/vessel/v1
until grep -qE "s7 done|s7 KILLED|deferred|contaminated|exited" $REP/status.txt 2>/dev/null; do sleep 120; done
if grep -q "s7 done" $REP/status.txt; then
  $PY scripts/eval_vessel.py --robust --ckpt checkpoints/vessel/vessel-v1_s7.pt --out $REP/robust_s7.json > $REP/robust_s7.txt 2>&1 && echo "robust s7 done" >> $REP/status.txt
fi
