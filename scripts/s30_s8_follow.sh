#!/bin/sh
# Follow-up for S30 seed 8 (replaces the tail of p1_queue2.sh, whose shell was stopped while the training process
# kept running). Waits for the training process; if it ended before step 30000 (interrupted), resumes it with
# --resume (exact from <out>.resume.pt when present, else approximate from the last 3000-step weights), up to 3
# attempts; then runs the same evals as queue2 and marks "S30_s8 done" (which GENERAL-2 waits for).
export PYTHONPATH=src; PY=./.venv/Scripts/python.exe
CK=checkpoints/p0/S30_s8.pt; DATA=data/processed/spatial_only_v1/train.jsonl
alive() { [ "$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*p0/S30_s8.pt*' }).Count" | tr -d '\r')" != "0" ]; }
step() { $PY -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['training_step'])" 2>/dev/null || echo 0; }
tries=0
while :; do
  while alive; do sleep 120; done
  [ -f checkpoints/p0/S30_s8.killed.json ] && { echo "S30_s8 KILLED" >> reports/p0/status.txt; exit 1; }
  s=$(step); [ "$s" = 30000 ] && break
  tries=$((tries+1)); [ $tries -gt 3 ] && { echo "S30_s8 failed to finish after 3 resumes (step $s)" >> reports/p0/status.txt; exit 1; }
  echo "S30_s8 interrupted at step $s; resume attempt $tries" >> reports/p0/status.txt
  $PY scripts/train_decision.py --config configs/general_v1.yaml --init checkpoints/tournament/r2/looped.pt --tokenizer data/tokenizer.json \
    --data $DATA --batch 16 --steps 30000 --balanced-sampling --diagnostics --save-every 3000 --seed 8 --iters-range 1,6 --eval-every 500 \
    --resume --out $CK >> reports/p0/train_S30_s8.log 2>&1
done
$PY scripts/eval_depth.py --ckpt $CK --out reports/p1/depth_S30_s8.json --iters 1,2,4,6,8,12 > reports/p1/depth_S30_s8.txt 2>&1
$PY scripts/run_gates.py --ckpt $CK --out reports/p0/gates_S30_s8.json --stress stress=reports/binding_stress/stress.jsonl \
  --stress hops=reports/chain/eval_hops.jsonl --transforms reports/chain/transforms.jsonl > reports/p0/gates_S30_s8.txt 2>&1
$PY scripts/probe_loop_states.py --ckpt $CK --out reports/p1/probe_S30_s8.json > reports/p1/probe_S30_s8.txt 2>&1
echo "S30_s8 done" >> reports/p0/status.txt
