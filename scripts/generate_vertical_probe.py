"""Balanced held-out vertical wording probe; never trains or modifies a checkpoint."""
import argparse,json,random
from pathlib import Path
from systemone_lab.data.spatial_worlds import TEMPLATES
from generate_paired_onehop import ALPHABET,_record

ap=argparse.ArgumentParser(); ap.add_argument('--out',default='reports/paired_onehop/vertical_probe_pairs.json'); ap.add_argument('--per-template',type=int,default=50); ap.add_argument('--seed',type=int,default=19123); a=ap.parse_args()
if a.per_template<1: ap.error('--per-template must be positive')
root=Path('reports/paired_onehop'); seen=set()
for path in (root/'train.jsonl',root/'heldout.jsonl',root/'counterfactual.jsonl'):
 for line in path.read_text().splitlines():
  r=json.loads(line); seen.add((r['state'],r['questions']['spatial']['instructions']))
rng=random.Random(a.seed); pairs=[]
for label in ('above','below'):
 for template_index,template in enumerate(TEMPLATES[label]):
  for j in range(a.per_template):
   for _ in range(10000):
    numbers=rng.sample(range(24),2); letters=rng.sample(ALPHABET,2)
    numbered=_record(f'vertical-{label}-{template_index}-{j}','numbered',label,template,(f'obj_{numbers[0]}',f'obj_{numbers[1]}'))
    renamed=_record(f'vertical-{label}-{template_index}-{j}','renamed',label,template,letters)
    keys=[(r['state'],r['questions']['spatial']['instructions']) for r in (numbered,renamed)]
    if len(set(keys))==2 and all(k not in seen for k in keys): break
   else: raise RuntimeError('insufficient unique name pairs')
   seen.update(keys)
   pairs.append({'id':f'vertical-{label}-{template_index}-{j}','label':label,'numbered':numbered,'renamed':renamed})
Path(a.out).write_text(json.dumps(pairs,indent=2)); print(len(pairs))
