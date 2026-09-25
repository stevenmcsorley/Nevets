import argparse, json
from pathlib import Path
from datasets import load_dataset
LABELS=["left","right","above","below","upper-left","upper-right","lower-left","lower-right","overlap"]
ap=argparse.ArgumentParser(); ap.add_argument("--split",default="train"); ap.add_argument("--out",default="data/processed/stepgame.jsonl"); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
ds=load_dataset("ZhengyanShi/StepGame",split=a.split)
with out.open("w",encoding="utf-8") as f:
 for i,r in enumerate(ds):
  if a.limit and i>=a.limit: break
  label=r["label"]; state=" ".join(r["story"]) if isinstance(r["story"],list) else r["story"]
  rec={"id":f"stepgame-{a.split}-{i}","state":state,"questions":{"spatial":{"type":"choice","instructions":r["question"],"criteria":{x:x for x in LABELS}}},"labels":{"spatial":label},"meta":{"source":"StepGame","k_hop":r.get("k_hop")}}
  f.write(json.dumps(rec,ensure_ascii=False)+"\n")
print(out)
