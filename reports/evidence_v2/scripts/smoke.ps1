$ErrorActionPreference="Stop"
python scripts/generate_spatial.py --out data/processed/spatial_smoke.jsonl --n 500 --split train --max-hops 4
python scripts/generate_spatial.py --out data/processed/spatial_smoke_eval.jsonl --n 100 --split test --min-hops 2 --max-hops 6 --seed 9001
python scripts/build_tokenizer.py --input data/processed/spatial_smoke.jsonl --out data/tokenizer.json --vocab 4000
python scripts/train_decision.py --config configs/s1_35m.yaml --tokenizer data/tokenizer.json --data data/processed/spatial_smoke.jsonl --steps 100 --accum 4 --save-every 100 --out checkpoints/smoke.pt
python scripts/eval_spatial.py --ckpt checkpoints/smoke.pt --data data/processed/spatial_smoke_eval.jsonl
$env:S1_CHECKPOINT="checkpoints/smoke.pt"
Write-Host "Start playground with: uvicorn systemone_lab.api:app --app-dir src --host 127.0.0.1 --port 8008"
