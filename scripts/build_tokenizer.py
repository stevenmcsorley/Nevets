import argparse, json
from pathlib import Path
from systemone_lab.tokenizer import train_tokenizer

def texts(paths):
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r=json.loads(line)
                    if "text" in r: yield r["text"]
                    if "state" in r: yield str(r["state"])
                    for q in r.get("questions",{}).values():
                        yield str(q.get("instructions",""))
                        c=q.get("criteria",{})
                        if isinstance(c,dict):
                            for k,v in c.items(): yield f"{k} {v}"
                except json.JSONDecodeError: yield line.strip()

ap=argparse.ArgumentParser(); ap.add_argument("--input",nargs="+",required=True); ap.add_argument("--out",default="data/tokenizer.json"); ap.add_argument("--vocab",type=int,default=16000)
a=ap.parse_args(); train_tokenizer(texts(a.input),a.out,a.vocab); print(a.out)
