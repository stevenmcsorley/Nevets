#!/bin/sh
# PT-2 size ladder (after GENERAL-2): pt_35m and pt_150m from random weights on the SAME first 1B tokens (same
# seeded window order, global batch 120 x 1024), checkpoints + val loss every 250M tokens, then the fixed downstream
# probe (scripts/pt/probe_pt.sh) on every checkpoint incl. 0 tokens, plus the legacy s1-35m-pretrain for reference.
# Judged on the SLOPE of the probe across checkpoints (owner), not the 1B endpoint, against a noise band from fine-tune
# seeds 7/8/9 on the first and last slope checkpoints: scripts/pt/pt2_report.py (band recorded first).
# Holds reports/pt/PT2_RUNNING (P1 round 2b pauses); waits for any round-2b arm in flight (reports/p1/ARM_RUNNING).
# Smoke test = this same script with small budgets (scripts/pt/pt2_smoke.sh sets PT2_SMOKE=1 and the variables below).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
export PT2_REP=${PT2_REP:-reports/pt/pt2}
REP=$PT2_REP; ROOT=${PT2_ROOT:-checkpoints/pt2}; TOKENS=${PT2_TOKENS:-1e9}; EVERY=${PT2_CKPT_EVERY:-250e6}
M35=${PT2_MICRO35:-40}; M150=${PT2_MICRO150:-12}; ONLY=${PT2_ONLY:-all}; mkdir -p $REP $ROOT
if [ -z "$PT2_SMOKE" ]; then
  until grep -q "general_v2 done" reports/general/v2/status.txt 2>/dev/null; do sleep 120; done
  touch reports/pt/PT2_RUNNING
  while [ -f reports/p1/ARM_RUNNING ]; do sleep 60; done
fi
lm() {  # size lr micro
  $PY scripts/pt/train_pt.py --config configs/pt/pt_$1.yaml --tokens $TOKENS --ckpt-every $EVERY --global-batch 120 \
    --micro $3 --lr $2 --out $ROOT/pt_$1 >> $REP/train_pt_$1.log 2>&1 || { echo "pt_$1 exited $?" >> $REP/status.txt; return 1; }
  echo "pt_$1 lm done" >> $REP/status.txt
  for c in $ROOT/pt_$1/tok_*M.pt; do n=$(basename $c .pt); sh scripts/pt/probe_pt.sh $c tokenizers/pt_32k.json pt_$1_$n; done
  # Noise band (owner): fine-tune seeds 8 and 9 on the first and last slope checkpoints (first >0 = 250M, and the final one).
  toks=$(ls $ROOT/pt_$1/tok_*M.pt | sed 's/.*tok_\([0-9]*\)M.pt/\1/' | sort -n | grep -v '^0$')
  for m in $(echo "$toks" | head -1) $(echo "$toks" | tail -1); do for s in 8 9; do
    sh scripts/pt/probe_pt.sh $ROOT/pt_$1/tok_${m}M.pt tokenizers/pt_32k.json pt_$1_tok_${m}M_s$s --seed $s
  done; done
}
if [ "$ONLY" != 150m ]; then
  lm 35m 1.5e-3 $M35
  grep -q "probe legacy_s1_35m done" $REP/status.txt 2>/dev/null || \
    sh scripts/pt/probe_pt.sh checkpoints/s1-35m-pretrain.pt data/tokenizer.json legacy_s1_35m --allow-tokenizer-mismatch
fi
[ "$ONLY" != 35m ] && lm 150m 6e-4 $M150
# Band only: the band goes in the ledger BEFORE the slope comparison (pt2_report.py without --band-only) is run.
$PY scripts/pt/pt2_report.py --rep $REP --ckpt-root $ROOT --band-only > $REP/band.txt 2>&1
[ -z "$PT2_SMOKE" ] && rm -f reports/pt/PT2_RUNNING
echo "pt2 ($ONLY) done (band computed; comparison pending ledger entry)" >> $REP/status.txt
