"""Decision training. Steps are optimizer updates; each contains --accum microbatches."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from systemone_lab.config import ModelConfig
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.model import SystemOneModel
from systemone_lab.data.io import read_jsonl
from systemone_lab.training import (pick_device, choose_dtype, decision_batch_loss,
    save_checkpoint, load_checkpoint, parameter_norm, check_safety)


def shuffled_options(rec):
    """Copy a record with every question's candidate order shuffled (labels are keyed, so unchanged)."""
    out = {k: v for k, v in rec.items() if not k.startswith('_')}
    qs = {}
    for key, q in rec['questions'].items():
        q = dict(q); crit = q.get('criteria')
        if isinstance(crit, dict):
            items = list(crit.items()); random.shuffle(items); q['criteria'] = dict(items)
        qs[key] = q
    out['questions'] = qs
    return out


def make_optimizer(model, config, freeze=False, lr=None, role_adapter_only=False, binding_only=False):
    options = config.get('optimizer', {})
    base = float(lr if lr is not None else options.get('lr', 1e-5))
    groups = {'backbone': [], 'decision_head': [], 'loop_gate': []}
    for name, p in model.named_parameters():
        head = name.startswith(('ptr_q.', 'ptr_k.', 'ptr_role.', 'ptr_bind.', 'loop_gate', 'coord_head.'))
        if role_adapter_only: trainable = name.startswith('ptr_role.')
        elif binding_only: trainable = name.startswith('ptr_bind.')
        else: trainable = head or not freeze
        p.requires_grad_(trainable)
        if p.requires_grad:
            groups['loop_gate' if name == 'loop_gate' else 'decision_head' if head else 'backbone'].append(p)
    return torch.optim.AdamW([
        {'params': ps, 'lr': base if lr is not None else float(options.get(name+'_lr',
            options.get('decision_head_lr', base) if name == 'loop_gate' else base)), 'name': name}
        for name, ps in groups.items() if ps], betas=tuple(options.get('betas', [0.9,0.95])),
        eps=float(options.get('eps',1e-8)), weight_decay=float(options.get('weight_decay',0.01)))


def make_scheduler(optimizer, config, steps):
    spec = config.get('scheduler', {})
    warmup = int(spec.get('warmup_steps', 100))
    kind = spec.get('type', 'cosine')
    if kind not in ('cosine', 'constant'):
        raise ValueError('scheduler type must be cosine or constant')
    def factor(step):
        if step < warmup:
            return (step+1)/max(1,warmup)
        if kind == 'constant': return 1.0
        return 0.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,steps-warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/s1_35m_decision_safe.yaml')
    ap.add_argument('--tokenizer',default='data/tokenizer.json')
    ap.add_argument('--data',required=True)
    ap.add_argument('--init')
    ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--lr',type=float)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--accum',type=int,default=1)
    ap.add_argument('--save-every',type=int,default=500)
    ap.add_argument('--out',default='checkpoints/spatial-safe.pt')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--diagnostics',action='store_true')
    ap.add_argument('--freeze-backbone',action='store_true')
    ap.add_argument('--role-adapter-only',action='store_true',
                    help='Freeze all pretrained weights and train only the role adapter')
    ap.add_argument('--binding-only',action='store_true',
                    help='Freeze all pretrained weights and train only the entity-binding projection')
    ap.add_argument('--allow-tokenizer-mismatch',action='store_true')
    ap.add_argument('--balanced-sampling',action='store_true')
    ap.add_argument('--shuffle-options',action='store_true',help='augment: shuffle the option order of every sampled question')
    ap.add_argument('--iters-range',help='looped arch: sample core iterations uniformly from "lo,hi" per update')
    a=ap.parse_args()
    if min(a.steps,a.batch,a.accum,a.save_every)<1: ap.error('counts must be positive')
    if Path(a.out).resolve() == Path('checkpoints/s1-35m-pretrain.pt').resolve() or (a.init and Path(a.out).resolve()==Path(a.init).resolve()):
        ap.error('output must not overwrite initialization or protected LM')
    random.seed(a.seed); torch.manual_seed(a.seed)
    config=yaml.safe_load(Path(a.config).read_text())
    device=pick_device(); dtype=choose_dtype(device); tok=LabTokenizer(a.tokenizer)
    if a.init:
        model,ck=load_checkpoint(a.init,SystemOneModel,tokenizer_path=a.tokenizer,allow_tokenizer_mismatch=a.allow_tokenizer_mismatch)
        decision_head=config.get('decision_head',model.cfg.decision_head)
        if decision_head.get('query_mode')=='role_adapter':
            model.enable_role_adapter(decision_head)
        elif decision_head.get('query_mode')=='entity_binding':
            model.enable_entity_binding(decision_head)
        else:
            model.cfg.decision_head=decision_head
        model.enable_state_loop(decision_head)
        model.enable_coord_head(decision_head)
        if 'arch' in config: model.set_arch(config['arch'])
    else:
        cfg=ModelConfig.load(a.config); cfg.vocab_size=tok.vocab_size; model=SystemOneModel(cfg)
    model.to(device).train()
    if a.role_adapter_only and model.ptr_role is None:
        ap.error('--role-adapter-only requires a role_adapter decision head')
    if a.binding_only and model.ptr_bind is None:
        ap.error('--binding-only requires an entity_binding decision head')
    opt=make_optimizer(model,config,a.freeze_backbone,a.lr,a.role_adapter_only,a.binding_only)
    scheduler=make_scheduler(opt,config,a.steps)
    records=list(read_jsonl(a.data))
    if not records: raise ValueError('empty dataset')
    weights=None
    if a.balanced_sampling:
        from collections import Counter
        labels=[tuple(sorted(r['labels'].items())) for r in records]; counts=Counter(labels)
        weights=[1/counts[label] for label in labels]
    params=[p for p in model.parameters() if p.requires_grad]
    iters_range=tuple(int(x) for x in a.iters_range.split(',')) if a.iters_range else None
    if iters_range and (model.cfg.arch or {}).get('type')!='looped': ap.error('--iters-range needs a looped arch')
    log=Path(a.out).with_suffix('.diagnostics.jsonl'); log.parent.mkdir(parents=True,exist_ok=True)
    safety=config.get('safety',{})
    print(json.dumps({'device':str(device),'dtype':str(dtype),'config':config,'args':vars(a)},default=str),flush=True)
    with log.open('w') as f:
        for step in range(1,a.steps+1):
            opt.zero_grad(set_to_none=True); batch_stats=[]; all_records=[]
            if iters_range: model.iters=random.randint(*iters_range)
            for micro in range(a.accum):
                recs=random.choices(records,weights=weights,k=a.batch); stats={'step':step,'microbatch':micro}
                if a.shuffle_options: recs=[shuffled_options(r) for r in recs]
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=='cuda' and dtype==torch.bfloat16):
                    loss=decision_batch_loss(model,tok,recs,device,stats)
                stats['loss']=loss.detach().float().item()
                check_safety(stats,safety,str(log)+'.failure.json',recs)
                (loss/a.accum).backward(); batch_stats.append(stats); all_records.extend(recs)
            stats=dict(batch_stats[-1]); stats['loss']=sum(s['loss'] for s in batch_stats)/a.accum
            stats['microbatches']=batch_stats
            stats['learning_rate']={g['name']:g['lr'] for g in opt.param_groups}
            stats['grad_norm_pre_clip']=parameter_norm(params,True)
            torch.nn.utils.clip_grad_norm_(params,float(config.get('training',{}).get('max_grad_norm',1.0)))
            stats['grad_norm_post_clip']=parameter_norm(params,True)
            stats['parameter_norm']=parameter_norm(params)
            check_safety(stats,safety,str(log)+'.failure.json',all_records)
            opt.step(); scheduler.step()
            stats['parameter_norm']=parameter_norm(params)
            check_safety(stats,safety,str(log)+'.failure.json',all_records)
            if step% (10 if a.diagnostics else 50)==0 or step==1:
                f.write(json.dumps(stats)+'\n'); f.flush(); print(json.dumps(stats),flush=True)
            if step%a.save_every==0:
                save_checkpoint(a.out,model,model.cfg,a.tokenizer,step,stats)
    if iters_range: model.iters=int((model.cfg.arch or {}).get('iters',1))
    save_checkpoint(a.out,model,model.cfg,a.tokenizer,a.steps,stats)

if __name__=='__main__': main()
