"""Generate and evaluate the held-out and paired reasoning gates."""
import argparse, json, random, math
from pathlib import Path
from collections import Counter
from systemone_lab.data.spatial_worlds import (LABELS, make_example, make_latent, rename_latent, render_latent, relation)
from systemone_lab.eval import predict_record, ece
from systemone_lab.training import load_checkpoint
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer


def record(ex,i):
    return {'id':i,'state':ex.state,'questions':{'spatial':ex.question},'labels':{'spatial':ex.label},'meta':ex.meta}


def generate():
    rng=random.Random(7721); seen=set()
    for split,n in [('train',800),('heldout',200)]:
        records=[]
        for i in range(n):
            target=LABELS[i%9]
            while True:
                ex=make_example(rng,rng.choice([1,2]),0,renamed=True)
                key=(ex.state,ex.question['instructions'])
                if ex.label==target and key not in seen: break
            seen.add(key); records.append(record(ex,f'{split}-{i}'))
        Path(f'reports/gate_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    pairs=[]
    for i in range(100):
        while True:
            lat=rename_latent(make_latent(rng,rng.choice([1,2]),0),rng)
            before=render_latent(lat); moving,anchor=lat['query']
            changed={**lat,'coords':dict(lat['coords'])}
            ax,ay=lat['coords'][anchor]; changed['coords'][moving]=(ax+rng.randint(-3,3),ay+rng.randint(-3,3))
            after=render_latent(changed)
            if before.label!=after.label and all((ex.state,ex.question['instructions']) not in seen for ex in [before,after]): break
        for variant,ex in [('base',before),('counterfactual',after)]:
            ex.meta.update(pair_id=f'pair-{i}',variant=variant)
            pairs.append(record(ex,f'pair-{i}-{variant}'))
    Path('reports/gate_counterfactual.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))


def evaluate(ckpt, train_path="reports/gate_train.jsonl", heldout_path="reports/gate_heldout.jsonl", pairs_path="reports/gate_counterfactual.jsonl", out="reports/generalisation_and_counterfactual.json"):
    model,ck=load_checkpoint(ckpt,SystemOneModel); model.cuda().eval(); tok=LabTokenizer(ck['tokenizer'])
    out_path=out; out={}; train_counts=Counter()
    for split in ['train','heldout']:
        records=[json.loads(x) for x in Path(train_path if split=='train' else heldout_path).read_text().splitlines()]
        counts=Counter(r['labels']['spatial'] for r in records)
        if split=='train': train_counts=counts
        majority=train_counts.most_common(1)[0][0]; cm=[[0]*9 for _ in range(9)]; conf=[]; corr=[]; brier=[]; by_hops={1:[],2:[]}
        for r in records:
            p=predict_record(model,tok,r,'cuda')['spatial']; pred=max(p,key=p.get); y=r['labels']['spatial']; good=int(pred==y)
            cm[LABELS.index(y)][LABELS.index(pred)]+=1; conf.append(p[pred]); corr.append(good); by_hops[r['meta']['hops']].append(good)
            brier.append(sum((v-int(k==y))**2 for k,v in p.items()))
        out[split]={'n':len(records),'accuracy':sum(corr)/len(corr),'ece':ece(conf,corr),'brier':sum(brier)/len(brier),
            'confusion_matrix':cm,'class_counts':counts,'random_baseline':1/9,'majority_baseline':counts[majority]/len(records),
            'accuracy_by_hops':{k:sum(v)/len(v) for k,v in by_hops.items()}}
    # All paired predictions use exactly the same model, with no adaptation.
    records=[json.loads(x) for x in Path(pairs_path).read_text().splitlines()]
    totals=Counter(); pair_results=[]
    for a,b in zip(records[::2],records[1::2]):
        pa=predict_record(model,tok,a,'cuda')['spatial']; pb=predict_record(model,tok,b,'cuda')['spatial']
        ya,yb=a['labels']['spatial'],b['labels']['spatial']; aa,bb=max(pa,key=pa.get),max(pb,key=pb.get)
        z={'original_correct':aa==ya,'counterfactual_correct':bb==yb,'both_correct':aa==ya and bb==yb,
           'correct_label_flip':aa==ya and bb==yb and aa!=bb,
           'probability_moved_correctly':pb[yb]>pa[yb] and pb[ya]<pa[ya]}
        totals.update({k:int(v) for k,v in z.items()}); pair_results.append({'id':a['id'],'original_target':ya,'counterfactual_target':yb,'original_prediction':aa,'counterfactual_prediction':bb,**z})
    out['counterfactual']={'pairs':100,**{k:v/100 for k,v in totals.items()},'pair_results':pair_results}
    Path(out_path).write_text(json.dumps(out,indent=2))
    print(json.dumps({**out,'counterfactual':{k:v for k,v in out['counterfactual'].items() if k!='pair_results'}},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--generate',action='store_true'); ap.add_argument('--ckpt'); ap.add_argument('--train',default='reports/gate_train.jsonl'); ap.add_argument('--heldout',default='reports/gate_heldout.jsonl'); ap.add_argument('--pairs',default='reports/gate_counterfactual.jsonl'); ap.add_argument('--out',default='reports/generalisation_and_counterfactual.json'); a=ap.parse_args()
    if a.generate: generate()
    if a.ckpt: evaluate(a.ckpt,a.train,a.heldout,a.pairs,a.out)
