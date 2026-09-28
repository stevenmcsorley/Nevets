#!/bin/sh
# Priority after VESSEL-1c. Lower-priority queues wait for our completion marker.
set -eu
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe
REP=reports/pt/probe2
mkdir -p "$REP"
echo "queued $(date -u +%FT%TZ)" >> "$REP/status.txt"
until grep -q 'vessel_v1c done' reports/vessel/v1c/status.txt 2>/dev/null; do sleep 30; done
touch reports/pt/PT2_RUNNING
trap 'rm -f reports/pt/PT2_RUNNING' EXIT
while [ -f reports/general/GPU_BUSY ] || [ -f reports/p1/ARM_RUNNING ]; do sleep 15; done
$PY scripts/check_contamination.py data/processed/probe2/train.jsonl > "$REP/contamination.json"
echo "start $(date -u +%FT%TZ)" >> "$REP/status.txt"
for s in 11 12 13; do
  sh scripts/pt/probe_v2.sh checkpoints/s1-35m-pretrain.pt data/tokenizer.json legacy_s1_35m_s$s $s
  sh scripts/pt/probe_v2.sh checkpoints/pt2/pt_35m/tok_0M.pt tokenizers/pt_32k.json pt_35m_tok_0M_s$s $s
done
CONTROL_RC=0
$PY scripts/pt/probe2_report.py control || CONTROL_RC=$?
if [ "$CONTROL_RC" -eq 2 ]; then
  echo 'INCONCLUSIVE: positive control failed; owner review required' >> "$REP/status.txt"
  touch "$REP/COMPLETE"
  exit 0
fi
[ "$CONTROL_RC" -eq 0 ] || exit "$CONTROL_RC"
for size in 35m 150m; do
  for m in 250 500 750 1000; do
    sh scripts/pt/probe_v2.sh checkpoints/pt2/pt_$size/tok_${m}M.pt tokenizers/pt_32k.json pt_${size}_tok_${m}M_s11 11
  done
  for m in 250 1000; do for s in 12 13; do
    sh scripts/pt/probe_v2.sh checkpoints/pt2/pt_$size/tok_${m}M.pt tokenizers/pt_32k.json pt_${size}_tok_${m}M_s$s $s
  done; done
done
$PY scripts/pt/probe2_report.py band
# Durably record the band before calculating slopes.
git add "$REP/band.json" reports/RESEARCH_LEDGER.md
git commit --only -m 'PROBE-2: record fine-tune noise band before slope comparison' "$REP/band.json" reports/RESEARCH_LEDGER.md
$PY scripts/pt/probe2_report.py compare
echo "probe2 done $(date -u +%FT%TZ); PT-3 requires owner approval if recommended" >> "$REP/status.txt"
touch "$REP/COMPLETE"
