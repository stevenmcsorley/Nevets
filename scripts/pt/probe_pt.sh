#!/bin/sh
# PT-2 downstream probe for one LM checkpoint:  sh scripts/pt/probe_pt.sh <ckpt> <tokenizer> <name> [extra train args]
# Fixed budget (PROBE_STEPS, default 3000; smoke tests only change it): 3000 updates x 16 on general_v2 (spatial 50%, 7 domains share the rest; domain-then-label sampling),
# seed 7, configs/pt/probe.yaml. Readout: stress + dev chains (run_gates) and worlds_v2 (7 domains, held-out table,
# counterfactual pairs; --limit 150 per domain). Extra fine-tune seeds: pass "--seed N" (last one wins) and a suffixed name. Probe fine-tunes are scratch checkpoints (not registered).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe; CK=$1; TOK=$2; N=$3; shift 3
OUT=${PT2_OUT:-checkpoints/pt2/probes}; REP=${PT2_REP:-reports/pt/pt2}; STEPS=${PROBE_STEPS:-3000}; mkdir -p $OUT $REP
$PY scripts/train_decision.py --config configs/pt/probe.yaml --init $CK --tokenizer $TOK --data data/processed/general_v2/train.jsonl \
  --batch 16 --steps $STEPS --domain-shares spatial=0.5 --save-every $STEPS --seed 7 --eval-every 0 --out $OUT/$N.pt "$@" > $REP/probe_${N}_train.log 2>&1 || { echo "probe $N failed" >> $REP/status.txt; exit 1; }
$PY scripts/run_gates.py --ckpt $OUT/$N.pt --out $REP/probe_${N}_gates.json --stress stress=reports/binding_stress/stress.jsonl \
  --stress hops=reports/chain/eval_hops.jsonl > $REP/probe_${N}_gates.txt 2>&1
$PY scripts/eval_worlds.py --ckpt $OUT/$N.pt --dir data/processed/worlds_v2 --out $REP/probe_${N}_worlds.json --limit 150 > $REP/probe_${N}_worlds.txt 2>&1
echo "probe $N done" >> $REP/status.txt
