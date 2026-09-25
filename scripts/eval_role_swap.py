"""Read-only role-swap test: keep the fact, reverse the queried object order."""
import argparse,json,re
from pathlib import Path
from collections import Counter
from systemone_lab.training import load_checkpoint
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.eval import predict_record
from systemone_lab.data.spatial_worlds import INVERSE_REL

ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--pairs',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
model,ck=load_checkpoint(a.ckpt,SystemOneModel); model.cuda().eval(); tok=LabTokenizer(ck['tokenizer'])
rows=[]; c=Counter()
for pair in json.loads(Path(a.pairs).read_text()):
 for variant in ('numbered','renamed'):
  r=pair[variant]; label=r['labels']['spatial']
  match=re.fullmatch(r'What is the spatial relation of (.+) to (.+)\?',r['questions']['spatial']['instructions'])
  if not match: raise ValueError('unrecognized question')
  swapped=json.loads(json.dumps(r)); swapped['questions']['spatial']['instructions']=f'What is the spatial relation of {match[2]} to {match[1]}?'
  swapped['labels']['spatial']=INVERSE_REL[label]
  p=predict_record(model,tok,r,'cuda')['spatial']; q=predict_record(model,tok,swapped,'cuda')['spatial']
  before=max(p,key=p.get); after=max(q,key=q.get)
  z={'id':pair['id'],'names':variant,'label':label,'swapped_label':INVERSE_REL[label],
     'prediction':before,'swapped_prediction':after,'original_correct':before==label,
     'swapped_correct':after==INVERSE_REL[label],
     'both_correct':before==label and after==INVERSE_REL[label],
     'correct_flip':before==label and after==INVERSE_REL[label] and before!=after}
  rows.append(z)
  for k in ('original_correct','swapped_correct','both_correct','correct_flip'): c[k]+=int(z[k])
out={'n':len(rows),'rates':{k:v/len(rows) for k,v in c.items()},'results':rows}
Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps({'n':out['n'],'rates':out['rates']},indent=2))
