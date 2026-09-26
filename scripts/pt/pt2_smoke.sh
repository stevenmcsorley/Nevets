#!/bin/sh
# PT-2 smoke test (owner, before the full run): the real pt2.sh on the real shards and configs with tiny budgets:
# 100 updates of pt_35m (12.3M tokens) and 50 of pt_150m (6.1M tokens), checkpoints at 0 / mid / end, 300-update
# probes on every checkpoint incl. the extra fine-tune seeds and the legacy LM, the band-only report, then the full
# comparison report. Small micro-batches so it can share the GPU with a running decision job. Scratch outputs.
export PT2_SMOKE=1 PT2_REP=reports/pt/pt2_smoke PT2_ROOT=checkpoints/pt2_smoke PT2_OUT=checkpoints/pt2_smoke/probes PROBE_STEPS=300
export PT2_MICRO35=8 PT2_MICRO150=4
rm -rf $PT2_REP $PT2_ROOT
PT2_TOKENS=1.2288e7 PT2_CKPT_EVERY=6.144e6 PT2_ONLY=35m sh scripts/pt/pt2.sh
PT2_TOKENS=6.144e6 PT2_CKPT_EVERY=3.072e6 PT2_ONLY=150m sh scripts/pt/pt2.sh
./.venv/Scripts/python.exe scripts/pt/pt2_report.py --rep $PT2_REP --ckpt-root $PT2_ROOT > $PT2_REP/report.txt 2>&1
echo "report exit=$?"; cat $PT2_REP/status.txt; cat $PT2_REP/report.txt
