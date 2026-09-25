"""Prepare a small clearly-licensed broad decision curriculum from public datasets."""
import argparse, json
from pathlib import Path
from datasets import load_dataset
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="data/processed/decision_public.jsonl"); ap.add_argument("--per-source",type=int,default=10000); a=ap.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
rows=[]
# BoolQ (CC BY-SA 3.0)
for i,r in enumerate(load_dataset("google/boolq",split="train")):
    if i>=a.per_source: break
    rows.append({"id":f"boolq-{i}","state":r["passage"],"questions":{"answer":{"type":"noul","instructions":r["question"]}},"labels":{"answer":bool(r["answer"])},"meta":{"source":"google/boolq","license":"CC-BY-SA-3.0"}})
# CommonsenseQA (MIT)
for i,r in enumerate(load_dataset("tau/commonsense_qa",split="train")):
    if i>=a.per_source: break
    crit={lab:txt for lab,txt in zip(r["choices"]["label"],r["choices"]["text"])}
    rows.append({"id":f"csqa-{i}","state":"","questions":{"answer":{"type":"choice","instructions":r["question"],"criteria":crit}},"labels":{"answer":r["answerKey"]},"meta":{"source":"tau/commonsense_qa","license":"MIT"}})
# Banking77 mirror in parquet form (dataset card marks MIT; verify upstream obligations for redistribution).
bank=load_dataset("mteb/banking77",split="train")
labels=sorted(set(str(x) for x in bank["label_text"]))
crit={x:x.replace("_"," ") for x in labels}
for i,r in enumerate(bank):
    if i>=a.per_source: break
    rows.append({"id":f"banking77-{i}","state":r["text"],"questions":{"intent":{"type":"choice","instructions":"Which banking intent best matches this message?","criteria":crit}},"labels":{"intent":str(r["label_text"])},"meta":{"source":"mteb/banking77","license":"MIT-card; trace upstream before redistribution"}})
with out.open("w",encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"wrote {len(rows)} records to {out}")
