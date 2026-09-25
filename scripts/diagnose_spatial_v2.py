"""Read-only tests of object-name dependence and dev-split performance."""
import argparse, json, random
from collections import Counter,defaultdict
from pathlib import Path
import torch
from systemone_lab.model import SystemOneModel
from systemone_lab.training import load_checkpoint
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.eval import predict_record
from systemone_lab.data.spatial_worlds import LABELS,make_latent,rename_latent,render_latent

ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',default='checkpoints/s1-35m-spatial-v2.pt'); ap.add_argument('--out',default='reports/spatial_v2_name_shift.json'); a=ap.parse_args()
model,ck=load_checkpoint(a.ckpt,SystemOneModel)
model.cuda().eval(); tok=LabTokenizer(ck['tokenizer']); rng=random.Random(92831)
summary={}; groups=defaultdict(lambda:Counter())
with Path('data/processed/spatial_dev.jsonl').open() as f:
 for line in f:
  r=json.loads(line); y=r['labels']['spatial']; pred=max((p:=predict_record(model,tok,r,'cuda')['spatial']),key=p.get)
  for key in ('overall',f"renamed={r['meta']['renamed']}",f"hops={r['meta']['hops']}",f"renamed={r['meta']['renamed']},hops={r['meta']['hops']}"):
   groups[key]['n']+=1; groups[key]['correct']+=int(pred==y); groups[key][f'pred:{pred}']+=1
summary['dev_groups']={k:{'n':v['n'],'accuracy':v['correct']/v['n'],'prediction_counts':{x[5:]:n for x,n in v.items() if x.startswith('pred:')}} for k,v in groups.items()}
pairs=[]; seen=set()
for i in range(200):
 target=LABELS[i%9]
 while True:
  lat=make_latent(rng,rng.choice([1,2]),0); a=render_latent(lat)
  if a.label==target and a.state not in seen: break
 seen.add(a.state); b=render_latent(rename_latent(lat,rng)); assert a.label==b.label
 pairs.append((a,b))
count=Counter(); per_hop=defaultdict(Counter); details=[]
for i,(a,b) in enumerate(pairs):
 def pred(ex):
  r={'state':ex.state,'questions':{'spatial':ex.question}}
  return max((p:=predict_record(model,tok,r,'cuda')['spatial']),key=p.get)
 pa,pb=pred(a),pred(b); y=a.label; hops=a.meta['hops']
 z={'id':i,'hops':hops,'label':y,'original_prediction':pa,'renamed_prediction':pb,'original_correct':pa==y,'renamed_correct':pb==y,'both_correct':pa==pb==y,'prediction_changed':pa!=pb}
 details.append(z)
 for c in [count,per_hop[hops]]:
  c['n']+=1
  for k in ['original_correct','renamed_correct','both_correct','prediction_changed']: c[k]+=int(z[k])
summary['name_only_pairs']={'overall':{k:v/ count['n'] if k!='n' else v for k,v in count.items()},'by_hops':{str(h):{k:v/c['n'] if k!='n' else v for k,v in c.items()} for h,c in per_hop.items()},'details':details}
Path(a.out).write_text(json.dumps(summary,indent=2))
print(json.dumps({'dev_groups':{k:v for k,v in summary['dev_groups'].items() if k in ['overall','renamed=False','renamed=True']},'name_only_pairs':{k:v for k,v in summary['name_only_pairs'].items() if k!='details'}},indent=2))
