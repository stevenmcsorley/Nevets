import argparse, json
import torch
from systemone_lab.model import SystemOneModel
from systemone_lab.training import load_checkpoint, pick_device
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.data.io import read_jsonl
from systemone_lab.eval import predict_record, ece
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--data",required=True); a=ap.parse_args(); device=pick_device(); model,ck=load_checkpoint(a.ckpt,SystemOneModel,device); model.to(device).eval(); tok=LabTokenizer(ck["tokenizer"])
n=ok=0; conf=[]; corr=[]; brier=0.0
for rec in read_jsonl(a.data):
 p=predict_record(model,tok,rec,device)["spatial"]; y=str(rec["labels"]["spatial"]); pred=max(p,key=p.get); c=p[pred]; good=pred==y; n+=1; ok+=good; conf.append(c); corr.append(int(good)); brier+=sum((v-(1.0 if k==y else 0.0))**2 for k,v in p.items())
print(json.dumps({"n":n,"accuracy":ok/n,"ece15":ece(conf,corr,15),"brier":brier/n},indent=2))
