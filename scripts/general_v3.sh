#!/bin/sh
# Owner GPU order (27 Sep): PT-2 35M + all its probes -> GENERAL-3 (2 seeds) -> round-2b seed 8 of the
# randomised-loop arm (r2b1) -> PT-2 150M (held by reports/pt/HOLD until this script releases it) -> other 2b arms.
# GENERAL-3 design (pre-registered in the ledger): init = better S30 seed; GENERAL-2's 30k updates and LR schedule;
# need-based domain shares; same-world pairs rendered in two training formats (never P3's pipe table / symbolic)
# with a consistency loss; mild label smoothing.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
OUT=checkpoints/general; REP=reports/general/v3; mkdir -p $OUT $REP
until grep -q "probe legacy_s1_35m done" reports/pt/pt2/status.txt 2>/dev/null; do sleep 120; done  # PT-2 35M phase complete
touch reports/general/GPU_BUSY
INIT=$(tail -1 reports/general/v2/s30_promotion.txt)
DATA=data/processed/general_v3/train.jsonl
SHARES=spatial=0.35,infogather=0.20,probability=0.135,dependency=0.0715,kinship=0.0715,temporal=0.0715,rules=0.0715,causal=0.029
$PY scripts/check_contamination.py $DATA > $REP/contamination.json || { echo contaminated >> $REP/status.txt; rm -f reports/general/GPU_BUSY; exit 1; }
for s in 7 8; do
  $PY scripts/train_decision.py --config configs/general_v1.yaml --init $INIT --tokenizer data/tokenizer.json --data $DATA \
    --batch 16 --steps 30000 --domain-shares $SHARES --pair-consistency 0.5 --label-smoothing 0.05 --diagnostics --save-every 5000 \
    --seed $s --iters-range 1,6 --eval-every 500 --resume --out $OUT/general-v3_s$s.pt > $REP/train_s$s.log 2>&1 || echo "s$s exited $?" >> $REP/status.txt
  [ -f $OUT/general-v3_s$s.killed.json ] && { echo "s$s KILLED" >> $REP/status.txt; continue; }
  $PY scripts/eval_worlds.py --ckpt $OUT/general-v3_s$s.pt --dir data/processed/worlds_v2 --out $REP/worlds_v2_s$s.json --limit 400 > $REP/worlds_v2_s$s.txt 2>&1
  $PY scripts/eval_depth.py --ckpt $OUT/general-v3_s$s.pt --out $REP/depth_s$s.json --iters 1,2,4,6,8,12 > $REP/depth_s$s.txt 2>&1
  $PY scripts/run_gates.py --ckpt $OUT/general-v3_s$s.pt --out $REP/gates_s$s.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $REP/gates_s$s.txt 2>&1
  $PY scripts/probe_loop_states.py --ckpt $OUT/general-v3_s$s.pt --out $REP/probe_s$s.json > $REP/probe_s$s.txt 2>&1
  echo "s$s done" >> $REP/status.txt
done
$PY scripts/compare_general.py g3 > $REP/verdict.txt 2>&1
echo "general_v3 done" >> $REP/status.txt
# Round-2b seed 8 for the randomised-loop arm (same init/data/recipe as its seed 7 and the controls).
P1=checkpoints/p1; R1=reports/p1; A=r2b1_depthK
$PY scripts/train_decision.py --config configs/p1/$A.yaml --init checkpoints/p0/S30_s7.pt --tokenizer data/tokenizer.json \
  --data data/processed/spatial_only_v1/train.jsonl --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 3000 \
  --seed 8 --hop-bucketed 0,4 --eval-every 500 --resume --out $P1/${A}_s8.pt > $R1/train_${A}_s8.log 2>&1 || echo "${A}_s8 exited $?" >> $R1/status.txt
if [ -f $P1/${A}_s8.killed.json ]; then echo "${A}_s8 KILLED" >> $R1/status.txt; else
  $PY scripts/eval_depth.py --ckpt $P1/${A}_s8.pt --out $R1/depth_${A}_s8.json --iters 1,2,4,6,8,12,16 --hop-offset 2 > $R1/depth_${A}_s8.txt 2>&1
  $PY scripts/probe_loop_states.py --ckpt $P1/${A}_s8.pt --out $R1/probe_${A}_s8.json --k-max 16 > $R1/probe_${A}_s8.txt 2>&1
  $PY scripts/run_gates.py --ckpt $P1/${A}_s8.pt --out $R1/gates_${A}_s8.json --stress stress=reports/binding_stress/stress.jsonl \
    --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > $R1/gates_${A}_s8.txt 2>&1
  echo "${A}_s8 done" >> $R1/status.txt
fi
rm -f reports/general/GPU_BUSY
rm -f reports/pt/HOLD  # releases PT-2 150M (train_pt.py is waiting on it)
echo "released PT-2 150M hold $(date '+%Y-%m-%d %H:%M')" >> reports/pt/pt2/status.txt
