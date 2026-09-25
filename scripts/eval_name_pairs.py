"""Evaluate matched spatial examples that differ only in object names."""
import argparse,json
from pathlib import Path
from collections import Counter
from systemone_lab.training import load_checkpoint
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.eval import predict_record
ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--pairs',default='reports/name_curriculum/name_pairs.json'); ap.add_argument('--out',required=True); a=ap.parse_args()
model,ck=load_checkpoint(a.ckpt,SystemOneModel); model.cuda().eval(); tok=LabTokenizer(ck['tokenizer'])
pairs=json.loads(Path(a.pairs).read_text()); count=Counter(); results=[]
for pair in pairs:
    y=pair['label']; p={}
    for key in ('numbered','renamed'):
        probs=predict_record(model,tok,pair[key],'cuda')['spatial']; p[key]=max(probs,key=probs.get)
    flags={'numbered_correct':p['numbered']==y,'renamed_correct':p['renamed']==y,'both_correct':p['numbered']==p['renamed']==y,'prediction_changed':p['numbered']!=p['renamed']}
    count.update({k:int(v) for k,v in flags.items()}); results.append({'id':pair['id'],'label':y,**p,**flags})
out={'n':len(pairs),'rates':{k:v/len(pairs) for k,v in count.items()},'results':results}
Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2))
