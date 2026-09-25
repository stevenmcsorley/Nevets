"""Read-only query-scale diagnostic on fixed small paired suites."""
import json,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'scripts')
from prepare_role_swap_curriculum import swap_record
from systemone_lab.training import load_checkpoint
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.eval import predict_record
name_pairs=json.loads(Path('reports/paired_onehop/name_pairs.json').read_text())[:54]
all_vertical=json.loads(Path('reports/paired_onehop/vertical_probe_pairs.json').read_text())
vertical=[p for i,p in enumerate(all_vertical) if i%50<10]
cf=[json.loads(s) for s in Path('reports/paired_onehop/counterfactual.jsonl').read_text().splitlines()][:60]
tok=LabTokenizer('data/tokenizer.json'); output={}
for model_name,checkpoint,mode in [('paraphrase','checkpoints/paired-paraphrase-onehop-gate.pt','role_aware'),('role_aware_trained','checkpoints/role-aware-onehop-gate.pt','role_aware'),('role_adapter_only','checkpoints/role-adapter-only-gate.pt','role_adapter')]:
 model,_=load_checkpoint(checkpoint,SystemOneModel); model.cuda().eval(); output[model_name]={}
 for scale in [0,0.1,0.25,0.5,1.0,2.0]:
  model.cfg.decision_head={'scorer':'cosine','temperature':10.0,'query_mode':'decide' if scale==0 else mode,'query_scale':scale if scale else 1.0}
  count=defaultdict(int)
  def answer(r):
   p=predict_record(model,tok,r,'cuda')['spatial']; return max(p,key=p.get)
  for pair in name_pairs:
   y=pair['label']; a=answer(pair['numbered']); b=answer(pair['renamed']); count['name_both']+=int(a==b==y)
  for pair in vertical:
   for variant in ('numbered','renamed'):
    r=pair[variant]; s=swap_record(r)
    a=answer(r); b=answer(s); count['role_both']+=int(a==r['labels']['spatial'] and b==s['labels']['spatial'])
  for a,b in zip(cf[::2],cf[1::2]):
   count['cf_both']+=int(answer(a)==a['labels']['spatial'] and answer(b)==b['labels']['spatial'])
  output[model_name][str(scale)]={'name_both':count['name_both']/len(name_pairs),
                                  'role_both':count['role_both']/(2*len(vertical)),
                                  'counterfactual_both':count['cf_both']/(len(cf)//2)}
  print(model_name,scale,output[model_name][str(scale)],flush=True)
Path('reports/role_swap_onehop/query_scale_sweep.json').write_text(json.dumps(output,indent=2))
