#!/bin/sh
# PT-2 size ladder (after GENERAL-2): pt_35m and pt_150m from random weights on the SAME first 1B tokens (same
# seeded window order, global batch 120 x 1024), checkpoints + val loss every 250M tokens, then the fixed downstream
# probe (scripts/pt/probe_pt.sh) on every checkpoint incl. 0 tokens, plus the legacy s1-35m-pretrain for reference.
# Judged on the SLOPE of the probe across checkpoints (owner), not the 1B endpoint: scripts/pt/pt2_report.py.
# Holds reports/pt/PT2_RUNNING (P1 round 2b pauses); waits for any round-2b arm in flight (reports/p1/ARM_RUNNING).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe; REP=reports/pt/pt2; mkdir -p $REP checkpoints/pt2
until grep -q "general_v2 done" reports/general/v2/status.txt 2>/dev/null; do sleep 120; done
touch reports/pt/PT2_RUNNING
while [ -f reports/p1/ARM_RUNNING ]; do sleep 60; done
lm() {  # size lr micro
  $PY scripts/pt/train_pt.py --config configs/pt/pt_$1.yaml --tokens 1e9 --global-batch 120 --micro $3 --lr $2 \
    --out checkpoints/pt2/pt_$1 >> $REP/train_pt_$1.log 2>&1 || { echo "pt_$1 exited $?" >> $REP/status.txt; return 1; }
  echo "pt_$1 lm done" >> $REP/status.txt
  for c in checkpoints/pt2/pt_$1/tok_*M.pt; do n=$(basename $c .pt); sh scripts/pt/probe_pt.sh $c tokenizers/pt_32k.json pt_$1_$n; done
}
lm 35m 1.5e-3 40
grep -q "probe legacy_s1_35m done" $REP/status.txt 2>/dev/null || \
  sh scripts/pt/probe_pt.sh checkpoints/s1-35m-pretrain.pt data/tokenizer.json legacy_s1_35m --allow-tokenizer-mismatch
lm 150m 6e-4 12
$PY scripts/pt/pt2_report.py > $REP/report.txt 2>&1
rm -f reports/pt/PT2_RUNNING; echo "pt2 done" >> $REP/status.txt
