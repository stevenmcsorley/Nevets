import argparse, json
from collections import defaultdict
from systemone_lab.model import SystemOneModel
from systemone_lab.training import load_checkpoint, pick_device
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.data.io import read_jsonl
from systemone_lab.eval import predict_record
ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--data',required=True); a=ap.parse_args(); device=pick_device(); model,ck=load_checkpoint(a.ckpt,SystemOneModel,device); model.to(device).eval(); tok=LabTokenizer(ck['tokenizer'])
pairs=defaultdict(dict)
for rec in read_jsonl(a.data):
    p=predict_record(model,tok,rec,device)['spatial']; y=str(rec['labels']['spatial']); pred=max(p,key=p.get); pairs[rec['meta']['pair_id']][rec['meta']['variant']]={'p':p,'y':y,'pred':pred}
both=shift=0; n=0
for z in pairs.values():
    if 'base' not in z or 'counterfactual' not in z: continue
    b,c=z['base'],z['counterfactual']; n+=1; both += int(b['pred']==b['y'] and c['pred']==c['y'])
    # Counterfactual target should gain probability and old target should lose it.
    shift += int(c['p'].get(c['y'],0)>b['p'].get(c['y'],0) and c['p'].get(b['y'],0)<b['p'].get(b['y'],0))
print(json.dumps({'pairs':n,'both_correct':both/n if n else 0,'correct_probability_shift':shift/n if n else 0},indent=2))
