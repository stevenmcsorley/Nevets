#!/bin/sh
# GPU queue 2 (spatial reasoning priority). Waits for TM-1 to finish, then:
#   T-R3 (no-grad warm-up / fixed-point loss) -> T-R4 (per-iteration BFS hint supervision) -> CH-2.
# All looped arms: exp6a init, K~U[1,6], 6000 x 16, seed 42; K sweep 1..16 on the dev chain suite.
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
until [ -s reports/worlds/v1/eval_scratch.json ]; do sleep 60; done
sweep() {  # ckpt repdir name
  for k in 1 2 4 6 8 12 16; do
    $PY scripts/run_gates.py --ckpt $1 --out $2/gates_$3_k$k.json --stress hops=reports/chain/eval_hops.jsonl --iters $k > $2/gates_$3_k$k.txt 2>&1
  done
}
loop_arm() {  # round name config data extra
  OUT=checkpoints/tournament/$1; REP=reports/tournament/$1; mkdir -p $OUT $REP
  $PY scripts/check_contamination.py $4 > $REP/contamination_$2.json || { echo "contaminated $4"; return 1; }
  $PY scripts/train_decision.py --config configs/tournament/$3.yaml --init checkpoints/exp6a-noloop.pt --tokenizer data/tokenizer.json \
    --data $4 --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 2000 --seed 42 --iters-range 1,6 \
    --out $OUT/$2.pt $5 > $REP/train_$2.log 2>&1
  sweep $OUT/$2.pt $REP $2
}
OLD=data/processed/tournament/chain_h1_6_train.jsonl        # T-R2 data (T-R3 keeps it for comparability)
NEW=data/processed/tournament/chain_h1_6_train_dist.jsonl   # same records + aux_dist, registry-clean
# T-R3 uses the T-R2 file for exact comparability; its 72 one-fact collisions are disclosed in CONTAM-1.
mkdir -p reports/tournament/r3 checkpoints/tournament/r3
$PY scripts/train_decision.py --config configs/tournament/r3_nograd.yaml --init checkpoints/exp6a-noloop.pt --tokenizer data/tokenizer.json \
  --data $OLD --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 2000 --seed 42 --iters-range 1,6 --nograd-range 0,6 \
  --out checkpoints/tournament/r3/nograd.pt > reports/tournament/r3/train_nograd.log 2>&1
sweep checkpoints/tournament/r3/nograd.pt reports/tournament/r3 nograd
$PY scripts/train_decision.py --config configs/tournament/r3_converge.yaml --init checkpoints/exp6a-noloop.pt --tokenizer data/tokenizer.json \
  --data $OLD --batch 16 --steps 6000 --balanced-sampling --diagnostics --save-every 2000 --seed 42 --iters-range 1,6 --nograd-range 0,6 \
  --out checkpoints/tournament/r3/converge.pt > reports/tournament/r3/train_converge.log 2>&1
sweep checkpoints/tournament/r3/converge.pt reports/tournament/r3 converge
loop_arm r4 hint1 r4_hint1 $NEW ""
loop_arm r4 hint3 r4_hint3 $NEW ""
sh scripts/chess_r1.sh spatial
echo queue_q2 done
