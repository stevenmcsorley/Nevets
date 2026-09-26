"""Decision training. Steps are optimizer updates; each contains --accum microbatches."""
import argparse
import json
import sys
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
    save_checkpoint, load_checkpoint, parameter_norm, check_safety, cap_gpu_memory)


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
        head = name.startswith(('ptr_q.', 'ptr_k.', 'ptr_role.', 'ptr_bind.', 'loop_gate', 'coord_head.', 'coord_readout_params'))
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
    ap.add_argument('--resume',action='store_true',help='continue from <out>.resume.pt if present (model, optimiser, '
                    'scheduler, step, RNG states, parent gate baseline); logs are appended')
    ap.add_argument('--resume-every',type=int,default=500,help='write <out>.resume.pt every N updates (atomic)')
    ap.add_argument('--mem-margin-gb',type=float,default=1.5,help='cap own VRAM at free-at-start minus this (protects co-running jobs)')
    ap.add_argument('--stop-after',type=int,default=0,help=argparse.SUPPRESS)  # tests: simulate an interruption
    ap.add_argument('--domain-shares',default=None,help='domain-first then label-balanced sampling, e.g. "spatial=0.5" '
                    '(unnamed domains split the rest equally; overrides --balanced-sampling)')
    ap.add_argument('--eval-every',type=int,default=500,help='dev gates every N updates (Rule 4); 0 disables')
    ap.add_argument('--no-kill',action='store_true',help='log gate regressions without stopping the run')
    ap.add_argument('--shuffle-options',action='store_true',help='augment: shuffle the option order of every sampled question')
    ap.add_argument('--hop-bucketed',help='looped arch: sample each batch from one hop bucket h and set K ~ U(h+lo, h+hi), '
                    'with deep supervision from pass h (arch.deep_supervision); value "lo,hi", e.g. "0,4"')
    ap.add_argument('--nograd-range',help='looped arch: warm-up iterations without gradient, sampled from "lo,hi" per update')
    ap.add_argument('--iters-range',help='looped arch: sample core iterations uniformly from "lo,hi" per update')
    a=ap.parse_args()
    if min(a.steps,a.batch,a.accum,a.save_every)<1: ap.error('counts must be positive')
    if Path(a.out).resolve() == Path('checkpoints/s1-35m-pretrain.pt').resolve() or (a.init and Path(a.out).resolve()==Path(a.init).resolve()):
        ap.error('output must not overwrite initialization or protected LM')
    random.seed(a.seed); torch.manual_seed(a.seed)
    config=yaml.safe_load(Path(a.config).read_text())
    device=pick_device(); dtype=choose_dtype(device); tok=LabTokenizer(a.tokenizer)
    if device.type=='cuda': print(json.dumps({'vram_cap_gb':cap_gpu_memory(a.mem_margin_gb)}),flush=True)
    gate_baseline=None; gates=None
    if a.init:
        model,ck=load_checkpoint(a.init,SystemOneModel,tokenizer_path=a.tokenizer,allow_tokenizer_mismatch=a.allow_tokenizer_mismatch)
        gate_baseline=None
        if a.eval_every:
            # Rule 4 parent baseline: the init checkpoint in its own native configuration.
            from systemone_lab.gates import FastGates
            gates=FastGates(); model.to(device); gate_baseline=gates.evaluate(model,tok,device)
            print(json.dumps({'parent_gates':gate_baseline}),flush=True)
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
    if a.domain_shares is not None:
        from systemone_lab.training import domain_balanced_weights
        weights=domain_balanced_weights(records,a.domain_shares)
        from collections import defaultdict as _dd
        mass=_dd(float)
        for r,w in zip(records,weights): mass[(r.get('meta') or {}).get('domain','none')]+=w
        print(json.dumps({'domain_shares':{k:round(v,4) for k,v in sorted(mass.items())}}),flush=True)
    params=[p for p in model.parameters() if p.requires_grad]
    iters_range=tuple(int(x) for x in a.iters_range.split(',')) if a.iters_range else None
    hop_buckets=None
    if a.hop_bucketed:
        if (model.cfg.arch or {}).get('type')!='looped': ap.error('--hop-bucketed needs a looped arch')
        hop_range=tuple(int(x) for x in a.hop_bucketed.split(','))
        from collections import defaultdict
        hop_buckets=defaultdict(list)
        for i,r in enumerate(records): hop_buckets[int(r.get('meta',{}).get('hops',1))].append(i)
        hop_keys=sorted(hop_buckets); hop_sizes=[len(hop_buckets[k]) for k in hop_keys]
        print(json.dumps({'hop_buckets':{k:len(v) for k,v in hop_buckets.items()}}),flush=True)
    nograd_range=tuple(int(x) for x in a.nograd_range.split(',')) if a.nograd_range else None
    if iters_range and (model.cfg.arch or {}).get('type')!='looped': ap.error('--iters-range needs a looped arch')
    log=Path(a.out).with_suffix('.diagnostics.jsonl'); log.parent.mkdir(parents=True,exist_ok=True)
    safety=config.get('safety',{})
    print(json.dumps({'device':str(device),'dtype':str(dtype),'config':config,'args':vars(a)},default=str),flush=True)
    resume_path=Path(a.out).with_suffix('.resume.pt'); start=1
    if a.resume and resume_path.exists():
        st=torch.load(resume_path,map_location=device,weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); scheduler.load_state_dict(st['scheduler'])
        random.setstate(st['py_rng']); torch.set_rng_state(st['torch_rng'].cpu())
        if st.get('cuda_rng') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda_rng']])
        gate_baseline=st['gate_baseline']; start=st['step']+1
        print(json.dumps({'resumed_from_step':st['step']}),flush=True)
    elif a.resume and Path(a.out).exists():
        # Fallback for runs started before exact resume existed: continue from the last periodic weights checkpoint.
        # APPROXIMATE: fresh AdamW moments and a new sampling stream (reseeded by step); the LR schedule is exact.
        ck=torch.load(a.out,map_location=device,weights_only=False); k=int(ck.get('training_step',0))
        if k>=a.steps: print(json.dumps({'already_complete':k}),flush=True); return
        if k>0:
            model.load_state_dict(ck['model'])
            for _ in range(k): scheduler.step()
            random.seed(a.seed*1000003+k); torch.manual_seed(a.seed*1000003+k); start=k+1
            print(json.dumps({'resumed_from_weights_step':k,'approximate':'fresh optimiser moments, new sampling stream'}),flush=True)
    def save_resume(step):
        tmp=resume_path.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'scheduler':scheduler.state_dict(),'step':step,
                    'py_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'gate_baseline':gate_baseline},tmp)
        tmp.replace(resume_path)
    mode='a' if start>1 else 'w'
    glog=Path(a.out).with_suffix('.gates.jsonl').open(mode)
    if gate_baseline is not None and start==1: glog.write(json.dumps({'step':0,'parent':True,**gate_baseline})+chr(10)); glog.flush()
    with log.open(mode) as f:
        for step in range(start,a.steps+1):
            opt.zero_grad(set_to_none=True); batch_stats=[]; all_records=[]
            if iters_range: model.iters=random.randint(*iters_range)
            if nograd_range: model.iters_nograd=random.randint(*nograd_range)
            for micro in range(a.accum):
                if hop_buckets:
                    h=random.choices(hop_keys,weights=hop_sizes,k=1)[0]; idx=hop_buckets[h]
                    recs=random.choices([records[i] for i in idx],weights=[weights[i] for i in idx] if weights else None,k=a.batch)
                    model.iters=random.randint(h+hop_range[0],h+hop_range[1]); model.deep_from=h
                else:
                    recs=random.choices(records,weights=weights,k=a.batch)
                stats={'step':step,'microbatch':micro}
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
            if a.eval_every and step%a.eval_every==0:
                if gates is None:
                    from systemone_lab.gates import FastGates
                    gates=FastGates()
                saved_iters=model.iters; model.iters=int((model.cfg.arch or {}).get('iters',1))
                result=gates.evaluate(model,tok,device); model.iters=saved_iters
                glog.write(json.dumps({'step':step,**result})+chr(10)); glog.flush()
                print(json.dumps({'gates_step':step,**{k:v for k,v in result.items() if k!='chain_by_hops'}}),flush=True)
                if gate_baseline is not None:
                    from systemone_lab.gates import regressions
                    bad=regressions(gate_baseline,result)
                    if bad and not a.no_kill:
                        save_checkpoint(a.out,model,model.cfg,a.tokenizer,step,stats)
                        Path(a.out).with_suffix('.killed.json').write_text(json.dumps(
                            {'step':step,'reason':'Rule 4: gate regression > 2 points vs parent','regressions':bad,
                             'parent':gate_baseline,'current':result},indent=2))
                        print(json.dumps({'KILLED':bad}),flush=True); sys.exit(3)
            if a.resume_every and step%a.resume_every==0 and step<a.steps: save_resume(step)
            if a.stop_after and step>=a.stop_after: print(json.dumps({'stopped_after':step}),flush=True); return
    if iters_range or hop_buckets: model.iters=int((model.cfg.arch or {}).get('iters',1))
    model.iters_nograd=0; model.deep_from=1
    save_checkpoint(a.out,model,model.cfg,a.tokenizer,a.steps,stats)
    resume_path.unlink(missing_ok=True)  # finished: the resume state is scratch

if __name__=='__main__': main()
