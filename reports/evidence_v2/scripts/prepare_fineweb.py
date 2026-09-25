"""Stream a bounded FineWeb-Edu sample to JSONL without downloading the corpus."""
import argparse, json
from pathlib import Path
from datasets import load_dataset
ap=argparse.ArgumentParser(); ap.add_argument("--docs",type=int,default=250000); ap.add_argument("--out",default="data/processed/fineweb_edu.jsonl"); ap.add_argument("--min-score",type=float,default=3.0); a=ap.parse_args()
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); ds=load_dataset("HuggingFaceFW/fineweb-edu",split="train",streaming=True)
n=0
with out.open("w",encoding="utf-8") as f:
 for r in ds:
  if float(r.get("score",0))<a.min_score: continue
  f.write(json.dumps({"text":r["text"],"source":"fineweb-edu","id":r.get("id")},ensure_ascii=False)+"\n"); n+=1
  if n>=a.docs: break
print(f"wrote {n} docs to {out}")
