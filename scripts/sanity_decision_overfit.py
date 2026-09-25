"""Deterministic decision gates. Never trains on the full spatial corpus."""
import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from systemone_lab.data.spatial_worlds import LABELS, TEMPLATES
from systemone_lab.formatting import pack_request, branch_attention_mask
from systemone_lab.training import (load_checkpoint, parameter_norm, numerical_statistics,
    check_safety, decision_batch_loss, save_checkpoint)
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.eval import predict_record
from train_decision import make_optimizer


def tiny_records(n=32, seed=42):
    rng=random.Random(seed); records=[]
    for i in range(n):
        label=LABELS[i%9]; a,b=rng.sample(list('ABCDEFGHJKLMNPQRSTUVWXYZ'),2)
        text=TEMPLATES[label][i//9%len(TEMPLATES[label])].format(a=a,b=b)
        records.append({'id':f'tiny-{seed}-{i}', 'state':text,
            'questions':{'spatial':{'type':'choice','instructions':f'What is the spatial relation of {a} to {b}?',
            'criteria':{k:k for k in LABELS}}}, 'labels':{'spatial':label}})
    return records


def prepare(model,tok,records):
    cache=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for r in records:
            p=pack_request(tok,r['state'],r['questions'],device='cuda'); l=p.layouts[0]
            h=model.hidden(p.input_ids[None],p.position_ids[None],branch_attention_mask(p.branch_ids))[0]
            cache.append((h.detach(),l,LABELS.index(r['labels']['spatial'])))
    return cache


def cached_loss(model,cache,stats):
    scores=[]; logits=[]; targets=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for h,l,target in cache:
            logits.append(model.decision_logits(h,l.decide_position,l.option_end_positions,scores)); targets.append(target)
    logits=torch.stack(logits)
    stats.update(numerical_statistics(torch.cat([x[0] for x in cache]),scores))
    return F.cross_entropy(logits.float(),torch.tensor(targets,device='cuda')),logits


def run(records, freeze=True, scorer='cosine', lr=1e-5, steps=1500, seed=42, name='micro', save=False):
    torch.manual_seed(seed); random.seed(seed)
    tok=LabTokenizer('data/tokenizer.json')
    model,_=load_checkpoint('checkpoints/s1-35m-pretrain.pt',SystemOneModel,
        tokenizer_path='data/tokenizer.json',allow_tokenizer_mismatch=True)
    model.cfg.decision_head={'scorer':scorer,'temperature':10.0}; model.cuda().train()
    config={'optimizer':{'backbone_lr':min(3e-6,lr),'decision_head_lr':lr,'weight_decay':0.01}}
    opt=make_optimizer(model,config,freeze); params=[p for p in model.parameters() if p.requires_grad]
    cache=prepare(model,tok,records) if freeze else None
    history=[]; first99=None; failures=0
    path=Path('reports')/name; path.parent.mkdir(exist_ok=True)
    with Path(str(path)+'.jsonl').open('w') as log:
        for step in range(1,steps+1):
            opt.zero_grad(set_to_none=True); stats={'step':step,'learning_rate':lr}
            if freeze: loss,logits=cached_loss(model,cache,stats)
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=decision_batch_loss(model,tok,records,torch.device('cuda'),stats)
            stats['loss']=loss.detach().float().item()
            try: check_safety(stats,{},str(path)+'.failure.json',records)
            except FloatingPointError: failures+=1; history.append(stats); break
            loss.backward(); stats['grad_norm_pre_clip']=parameter_norm(params,True)
            torch.nn.utils.clip_grad_norm_(params,1.0); stats['grad_norm_post_clip']=parameter_norm(params,True)
            stats['parameter_norm']=parameter_norm(params)
            check_safety(stats,{},str(path)+'.failure.json',records)
            opt.step(); history.append(stats)
            if step==1 or step%10==0:
                log.write(json.dumps(stats)+'\n'); log.flush()
            if step==1 or step%50==0:
                if freeze:
                    with torch.no_grad(): _,lp=cached_loss(model,cache,{})
                    pred=lp.argmax(-1).tolist()
                else:
                    pred=[LABELS.index(max((p:=predict_record(model,tok,r,'cuda')['spatial']),key=p.get)) for r in records]; model.train()
                acc=sum(p==LABELS.index(r['labels']['spatial']) for p,r in zip(pred,records))/len(records)
                print(f'{name} step={step} loss={stats["loss"]:.6f} accuracy={acc:.4f} logit={stats["logit_max_abs"]:.3f} grad={stats["grad_norm_pre_clip"]:.3f}',flush=True)
                if acc>=.99 and first99 is None: first99=step
                # Require a sustained correct solution and a substantially lower loss.
                if first99 and step>=first99+100 and stats['loss']<0.15: break
    predictions=[]; cm=[[0]*9 for _ in range(9)]
    for r in records:
        p=predict_record(model,tok,r,'cuda')['spatial']; pred=max(p,key=p.get); y=r['labels']['spatial']
        cm[LABELS.index(y)][LABELS.index(pred)]+=1; predictions.append({'id':r['id'],'target':y,'prediction':pred,'probabilities':p})
    final_loss=sum(-__import__('math').log(max(p['probabilities'][p['target']],1e-30)) for p in predictions)/len(records)
    result={'scorer':scorer,'lr':lr,'freeze':freeze,'steps':step,'steps_to_99':first99,
        'accuracy':sum(p['target']==p['prediction'] for p in predictions)/len(records),
        'final_loss':final_loss,'loss_range':[min(s['loss'] for s in history),max(s['loss'] for s in history)],
        'max_q_norm':max(s['q_norm_max'] for s in history),'max_k_norm':max(s['k_norm_max'] for s in history),
        'max_logit':max(s['logit_max_abs'] for s in history),'max_gradient_norm':max(s.get('grad_norm_pre_clip',0) for s in history),
        'numerical_failures':failures,'loss_increases':sum(b['loss']>a['loss']+1e-4 for a,b in zip(history,history[1:])),
        'confusion_matrix':cm,'predictions':predictions}
    Path(str(path)+'.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)
    print("Predictions: " + json.dumps([{k:v for k,v in p.items() if k!='probabilities'} for p in predictions]),flush=True)
    if save: save_checkpoint(str(path)+'.pt',model,model.cfg,'data/tokenizer.json',step,result)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--examples',type=int,choices=[8,32],default=32)
    ap.add_argument('--freeze-backbone',action='store_true'); ap.add_argument('--scorer',choices=['cosine','scaled_dot'],default='cosine')
    ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--steps',type=int,default=1500); ap.add_argument('--name',default='micro'); a=ap.parse_args()
    records=tiny_records(a.examples); Path(f'reports/tiny_{a.examples}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    r=run(records,a.freeze_backbone,a.scorer,a.lr,a.steps,name=a.name)
    if r['accuracy']<.99 or r['numerical_failures']: raise SystemExit('OVERFIT GATE FAILED')
if __name__=='__main__': main()
