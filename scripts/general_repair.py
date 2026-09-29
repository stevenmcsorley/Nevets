"""Bounded sampler/init diagnosis and S60 seed replication, queued after PT-3.

--prepare freezes inputs and fresh development records without using CUDA.
--run waits for PT-3 COMPLETE, then runs serially under the shared GPU marker.
--evaluate is an internal checkpoint readout with per-domain and utility metrics.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'reports/general/repair_20260929'
DATA=ROOT/'data/processed/general_repair_dev'
SHARES='spatial=0.35,infogather=0.20,probability=0.135,dependency=0.0715,kinship=0.0715,temporal=0.0715,rules=0.0715,causal=0.029'
INITS={'s30':'checkpoints/p0/S30_s8.pt','g1':'checkpoints/general/general-v1.pt'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def digest(state): return hashlib.sha256(state.encode()).digest()


def build_dev():
    from systemone_lab.eval_registry import eval_states
    from systemone_lab.worlds import DOMAINS, infogather_example
    if (DATA/'manifest.json').exists(): return
    if DATA.exists() and any(DATA.iterdir()): raise ValueError('Partial development build; inspect before recovery')
    DATA.mkdir(parents=True,exist_ok=True)
    forbidden={digest(s) for s in eval_states()}
    training=['general_v1','general_v2','general_v3','spatial_only_v1']
    sources={}
    for name in training:
        path=ROOT/f'data/processed/{name}/train.jsonl'; sources[str(path.relative_to(ROOT))]=sha(path)
        with path.open(encoding='utf-8') as f:
            for line in f: forbidden.add(digest(json.loads(line)['state']))
    rng=random.Random(9292026); records=[]; pairs=[]
    for domain in ['causal','dependency','infogather','kinship','probability','rules','temporal']:
        kept=0
        while kept<200:
            rec,_=DOMAINS[domain](rng,rng.choice(['prose','json','kv']),f'repairdev-{domain}-{kept}')
            key=digest(rec['state'])
            if key in forbidden: continue
            forbidden.add(key); records.append(rec); kept+=1
    while len(pairs)<200:
        seed=rng.randrange(2**50); fmt=rng.choice(['prose','json','kv'])
        lo,_=infogather_example(random.Random(seed),fmt,'probe',cost=Fraction(0))
        if lo['meta']['value_of_information']<=.01: continue
        # cost=1 exceeds the value of information (repair utility is bounded by 1).
        hi,_=infogather_example(random.Random(seed),fmt,'probe',cost=Fraction(1))
        if lo['labels']==hi['labels']: raise AssertionError('Cost intervention failed')
        if any(digest(r['state']) in forbidden for r in (lo,hi)): continue
        pair_id=len(pairs)//2
        for variant,rec in [('base',lo),('counterfactual',hi)]:
            rec['id']=f'repairdev-cf-{pair_id}-{variant}'
            rec['meta'].update(pair_id=pair_id,variant=variant)
            forbidden.add(digest(rec['state'])); pairs.append(rec)
    for name,rows in [('eval_in_format.jsonl',records),('counterfactual.jsonl',pairs)]:
        (DATA/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (DATA/'manifest.json').write_text(json.dumps(dict(seed=9292026,domains=7,per_domain=200,cost_flip_pairs=100,
        cost_flip='0 versus 1 on the identical generated world; deliberately easy structural diagnostic',
        exclusion='exact state SHA-256 against prior general/spatial training and registered eval, plus internal deduplication; not latent-world disjointness',
        training_sha256=sources,files={p.name:sha(p) for p in DATA.glob('*.jsonl')}),indent=2))


def plan():
    paths=['configs/general_v1.yaml','data/tokenizer.json','data/processed/general_v3/train.jsonl',
           'data/processed/spatial_only_v1/train.jsonl',*INITS.values(),
           'scripts/general_repair.py','scripts/train_decision.py','src/systemone_lab/training.py',
           'src/systemone_lab/worlds.py','scripts/eval_depth.py','scripts/eval_worlds.py',
           'data/processed/general_repair_dev/eval_in_format.jsonl',
           'data/processed/general_repair_dev/counterfactual.jsonl']
    return dict(approved='Owner "ok do it", 29 September 2026: review labels, controlled GENERAL comparison, S60 second seed',
        hashes={p:sha(ROOT/p) for p in paths},domain_shares=SHARES,
        arms=[dict(init=init,semantic=semantic,seed=seed,steps=10000)
              for init in INITS for seed in [7,8] for semantic in [False,True]],
        readout_steps=[5000,10000],pair_consistency=.5,label_smoothing=.05,iters_range='1,6',batch=16,
        s60=dict(init=INITS['s30'],seed=8,steps=30000,balanced_sampling=True),
        total_max_training_updates=110000,
        queue='after PT-3 COMPLETE and shared GPU markers clear; one job at a time',
        criteria='Diagnostic only. Sampler effect >=3pp infogather in BOTH seeds within init, non-increasing expected-utility regret, no other domain down >3pp, spatial long-chain down <=2pp. Compare 10k endpoints, report all arms and 5k trajectory; no automatic selection/promotion.',
        interpretation='Matched additional compute, not matched lifetime exposure across initializations. Fresh dev is not a locked promotion test.',
        gpu_extensions=False,automatic_deployment=False)


def ready(root=ROOT):
    # Owner, 29 Sep: GENERAL repair may run while PT-3 is paused at a 1B checkpoint (PAUSED_FOR_GENERAL marker).
    pt3=root/'reports/pt/pt3'
    return ((pt3/'COMPLETE').exists() or (pt3/'PAUSED_FOR_GENERAL').exists()) and not any((root/p).exists() for p in
        ['reports/pt/PT2_RUNNING','reports/general/GPU_BUSY','reports/p1/ARM_RUNNING'])


def status(message):
    with (REP/'status.txt').open('a') as f: f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')


def command(args, log):
    with log.open('a',encoding='utf-8') as f:
        return subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,
            env={**os.environ,'PYTHONPATH':'src','S1_DEVICE':'cuda'},stdout=f,stderr=subprocess.STDOUT).returncode


def required(args, log):
    code=command(args,log)
    if code: raise RuntimeError(f'Command failed ({code}); see {log}')


def evaluate(checkpoint, output):
    from systemone_lab.training import load_checkpoint,pick_device
    from systemone_lab.model import SystemOneModel
    from systemone_lab.tokenizer import LabTokenizer
    from eval_worlds import score,summary
    device=pick_device(); model,ck=load_checkpoint(checkpoint,SystemOneModel,device)
    model.to(device).eval(); tok=LabTokenizer(ck['tokenizer']); by=defaultdict(list)
    for line in (DATA/'eval_in_format.jsonl').open():
        row=score(model,tok,device,json.loads(line)); by[row['meta']['domain']].append(row)
    info=by['infogather']; recalls={}
    for group,is_test in [('test',True),('act',False)]:
        subset=[r for r in info if (r['y']=='run the diagnostic')==is_test]
        recalls[group]=dict(n=len(subset),accuracy=sum(r['ok'] for r in subset)/len(subset) if subset else None)
    regret=sum(max(r['meta']['eu'].values())-r['meta']['eu'][r['pred']] for r in info)/len(info)
    pairs=defaultdict(list)
    for line in (DATA/'counterfactual.jsonl').open():
        r=score(model,tok,device,json.loads(line)); pairs[r['meta']['pair_id']].append(r)
    result=dict(checkpoint=str(checkpoint),domains={d:summary(v) for d,v in by.items()},
        infogather_actions=recalls,utility_regret=regret,
        cost_flip_both_correct=sum(all(r['ok'] for r in v) for v in pairs.values())/len(pairs))
    Path(output).write_text(json.dumps(result,indent=2))


def train_args(arm,out):
    args=['scripts/train_decision.py','--config','configs/general_v1.yaml','--init',INITS[arm['init']],
        '--tokenizer','data/tokenizer.json','--data','data/processed/general_v3/train.jsonl',
        '--batch','16','--steps','10000','--domain-shares',SHARES,'--pair-consistency','.5',
        '--label-smoothing','.05','--diagnostics','--save-every','5000','--seed',str(arm['seed']),
        '--iters-range','1,6','--eval-every','500','--resume','--out',str(out)]
    if arm['semantic']: args.append('--semantic-infogather')
    return args


def run(protocol):
    if (REP/'COMPLETE').exists(): return
    singleton=REP/'QUEUE_LOCK'; singleton.mkdir(); (singleton/'pid.txt').write_text(str(os.getpid()))
    gpu=ROOT/'reports/general/GPU_BUSY'; acquired=False
    try:
        status('queued after PT-3; eight 10k GENERAL arms then 30k S60 seed 8')
        while not ready(): time.sleep(30)
        if plan()!=protocol: raise ValueError('Frozen inputs changed while queued')
        with gpu.open('x') as f: f.write(f'GENERAL_REPAIR {os.getpid()}\n')
        acquired=True
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; refusing fallback')
        required(['scripts/check_contamination.py','data/processed/general_v3/train.jsonl'],REP/'contamination.log')
        for init,path in INITS.items():
            target=REP/f'{init}_baseline.json'
            if not target.exists(): required(['scripts/general_repair.py','--evaluate',path,'--out',target],REP/'eval.log')
        for arm in protocol['arms']:
            name=f"{arm['init']}_{'semantic' if arm['semantic'] else 'legacy'}_s{arm['seed']}"
            out=ROOT/f'checkpoints/general_repair/{name}.pt'; out.parent.mkdir(parents=True,exist_ok=True)
            if (REP/f'{name}.done').exists() or out.with_suffix('.killed.json').exists(): continue
            status('started '+name)
            for step in [5000,10000]:
                readout=REP/f'{name}_{step}.json'
                if readout.exists(): continue
                args=train_args(arm,out)
                if step==5000: args+=['--stop-after','5000']
                existing_step=int(torch.load(out,map_location='cpu',weights_only=False).get('training_step',0)) if out.exists() else 0
                if existing_step>step: raise RuntimeError(f'{name}: missing earlier readout; inspect before recovery')
                code=0 if existing_step==step else command(args,REP/f'{name}.log')
                if code==3 and out.with_suffix('.killed.json').exists(): status('gate killed '+name); break
                if code: raise RuntimeError(f'{name} failed with code {code}')
                actual=int(torch.load(out,map_location='cpu',weights_only=False)['training_step'])
                if actual!=step: raise RuntimeError(f'Expected checkpoint at {step}, found {actual}')
                saved=out.with_name(f'{out.stem}_{step}.pt'); shutil.copy2(out,saved)
                required(['scripts/general_repair.py','--evaluate',saved,'--out',readout],REP/'eval.log')
                status(f'readout {name} step {step}')
            if out.with_suffix('.killed.json').exists(): continue
            required(['scripts/eval_depth.py','--ckpt',out,'--out',REP/f'{name}_depth.json',
                      '--iters','1,2,4,6,8,12'],REP/'depth.log')
            (REP/f'{name}.done').write_text('completed\n')
        # Independent spatial-only replication; original S60 seed-7 checkpoint is untouched.
        out=ROOT/'checkpoints/general_repair/s60_s8.pt'
        if not (REP/'s60_s8.done').exists():
            status('started S60 seed 8')
            code=command(['scripts/train_decision.py','--config','configs/general_v1.yaml',
                '--init',INITS['s30'],'--tokenizer','data/tokenizer.json',
                '--data','data/processed/spatial_only_v1/train.jsonl','--batch','16','--steps','30000',
                '--balanced-sampling','--diagnostics','--save-every','5000','--seed','8',
                '--iters-range','1,6','--eval-every','500','--resume','--out',out],REP/'s60_s8.log')
            if code==3 and out.with_suffix('.killed.json').exists(): status('gate killed S60 seed 8')
            elif code: raise RuntimeError(f'S60 failed ({code})')
            else:
                required(['scripts/eval_depth.py','--ckpt',out,'--out',REP/'s60_s8_depth.json',
                          '--iters','1,2,4,6,8,12'],REP/'depth.log')
                (REP/'s60_s8.done').write_text('completed\n')
        (REP/'COMPLETE').write_text('Scheduled experiments finished, including any gate-killed arms; inspect results. No promotion.\n')
        status('done; no model promoted')
    except Exception as error:
        status(f'ERROR {type(error).__name__}: {error}'); raise
    finally:
        if acquired: gpu.unlink()
        (singleton/'pid.txt').unlink(); singleton.rmdir()


def main():
    ap=argparse.ArgumentParser(); group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true'); group.add_argument('--run',action='store_true')
    group.add_argument('--evaluate'); ap.add_argument('--out'); a=ap.parse_args(); os.chdir(ROOT)
    if a.evaluate: evaluate(a.evaluate,a.out); return
    REP.mkdir(parents=True,exist_ok=True)
    if a.prepare: build_dev()
    current=plan(); path=REP/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=current: raise ValueError('Frozen protocol differs')
    if a.prepare:
        path.write_text(json.dumps(current,indent=2)); print('Prepared eight 10k GENERAL arms + S60 30k; no GPU used'); return
    if not path.exists(): raise ValueError('Run --prepare first')
    run(current)


if __name__=='__main__': main()
