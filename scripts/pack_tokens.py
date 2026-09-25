"""Convert JSONL text into a contiguous uint32 token stream for fast random-access LM training."""
import argparse, json
from pathlib import Path
import numpy as np
from systemone_lab.tokenizer import LabTokenizer
ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--tokenizer",default="data/tokenizer.json"); ap.add_argument("--out",default="data/processed/pretrain_tokens.bin"); ap.add_argument("--max-docs",type=int,default=0); a=ap.parse_args()
tok=LabTokenizer(a.tokenizer); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); count=docs=0
with out.open("wb") as w, Path(a.input).open(encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        try: text=json.loads(line).get("text","")
        except json.JSONDecodeError: text=line
        ids=[tok.id("<bos>")]+tok.encode(text)+[tok.id("<eos>")]
        np.asarray(ids,dtype=np.uint32).tofile(w); count+=len(ids); docs+=1
        if a.max_docs and docs>=a.max_docs: break
meta={"tokens":count,"docs":docs,"dtype":"uint32","tokenizer":a.tokenizer}
out.with_suffix(out.suffix+".json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
