"""Approved 35M PT-3: frozen one-pass budget, periodic PROBE-2 readouts, plateau stop.

Prepare without touching the GPU: python scripts/pt/pt3.py --prepare
Queue after GENERAL-4/S60: python scripts/pt/pt3.py --run
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / 'reports/pt/pt3'
OUT = ROOT / 'checkpoints/pt3/pt_35m'
PER_STEP = 120 * 1024
METRICS = ('in_format', 'table', 'cf_both')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plateau(history, delta=.005, patience=2):
    """Stop after two readouts without a >=0.5pp new best on any general metric.

    An operational compute stop, not a significance test. Earliest stop is third readout.
    """
    best = None; stale = 0
    for row in history:
        values = row['metrics']
        if best is None:
            best = {k: values[k] for k in METRICS}
            continue
        improved = any(values[k] >= best[k] + delta for k in METRICS)
        stale = 0 if improved else stale + 1
        best = {k: max(best[k], values[k]) for k in METRICS}
    return len(history) >= patience + 1 and stale >= patience


def plan():
    shards = sorted((ROOT/'data/pt/shards').glob('train_*.bin'))
    if not shards or any(p.stat().st_size % 2 for p in shards):
        raise ValueError('Missing or invalid uint16 shards')
    available = sum((p.stat().st_size//2)//1025 for p in shards)
    steps = min(available//120, int(10e9//PER_STEP))
    targets = sorted({round(n*1e9/PER_STEP) for n in range(1,10) if round(n*1e9/PER_STEP)<steps} | {steps})
    tokenizer = ROOT/'tokenizers/pt_32k.json'
    stream = json.loads((ROOT/'data/pt/stream_manifest.json').read_text())
    if sha(tokenizer) != stream['tokenizer_sha256']:
        raise ValueError('Corpus tokenizer hash mismatch')
    control=json.loads((ROOT/'reports/pt/probe2/control.json').read_text())
    result=json.loads((ROOT/'reports/pt/probe2/report.json').read_text())
    if not control['passed'] or '35m' not in result['pt3_recommendation']:
        raise ValueError('Expected positive control and 35M recommendation')
    if (ROOT/'data/pt/shards/val_0000.bin').stat().st_size < 4_000_000:
        raise ValueError('Insufficient validation tokens')
    return dict(approved='Owner: "ok do it", 28 September 2026; local 35M, <=10B tokens',
        init='random weights; fresh long cosine schedule; PT-2 checkpoints remain untouched',
        config='configs/pt/pt_35m.yaml',config_sha256=sha(ROOT/'configs/pt/pt_35m.yaml'),
        tokenizer='tokenizers/pt_32k.json',tokenizer_sha256=sha(tokenizer),
        corpus_manifest_sha256=sha(ROOT/'data/pt/stream_manifest.json'),
        shards=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in shards],
        unique_windows=available,tokens=steps*PER_STEP,total_steps=steps,target_steps=targets,
        seq=1024,global_batch=120,micro=30,lr=.0015,seed=2027,checkpoint_every=1_000_000_000,
        probe=dict(config='configs/pt/probe_v2.yaml',config_sha256=sha(ROOT/'configs/pt/probe_v2.yaml'),
                   train_sha256=sha(ROOT/'data/processed/probe2/train.jsonl'),steps=6000,batch=16,seed=11),
        plateau=dict(min_improvement=.005,patience=2,metrics=list(METRICS),earliest_readout=3),
        queue='after s60 done and all existing GPU marker files clear',rented_compute=False)


def status(message):
    with (REP/'status.txt').open('a',encoding='utf-8') as f:
        f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')


def command(args, log):
    env={**os.environ,'PYTHONPATH':'src','S1_DEVICE':'cuda'}
    with log.open('a',encoding='utf-8') as f:
        subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    ap=argparse.ArgumentParser(); action=ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true'); action.add_argument('--run',action='store_true')
    a=ap.parse_args(); os.chdir(ROOT); REP.mkdir(parents=True,exist_ok=True)
    current=plan(); protocol=REP/'protocol.json'
    if a.prepare:
        if protocol.exists() and json.loads(protocol.read_text())!=current:
            raise ValueError('Existing frozen protocol differs')
        protocol.write_text(json.dumps(current,indent=2))
        print(json.dumps({k:current[k] for k in ('tokens','total_steps','target_steps','micro','queue')})); return
    if json.loads(protocol.read_text())!=current:
        raise ValueError('Frozen protocol/config/data changed; refusing launch')
    if (REP/'COMPLETE').exists():
        print('PT-3 already complete'); return
    # Atomic singleton lock: a crash leaves it for deliberate recovery, not duplicate training.
    singleton=REP/'QUEUE_LOCK'; singleton.mkdir()
    (singleton/'pid.txt').write_text(str(os.getpid()))
    gpu=ROOT/'reports/pt/PT2_RUNNING'; acquired=False
    try:
        status('queued after GENERAL-4/S60')
        while True:
            done=ROOT/'reports/general/s60/status.txt'
            finished=done.exists() and 's60 done ' in done.read_text()
            busy=any((ROOT/p).exists() for p in ('reports/general/GPU_BUSY','reports/p1/ARM_RUNNING','reports/pt/PT2_RUNNING'))
            if finished and not busy: break
            time.sleep(30)
        if plan()!=current: raise ValueError('Protocol inputs changed while queued')
        with gpu.open('x') as f: f.write(f'PT3 {os.getpid()}\n')
        acquired=True
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; refusing CPU fallback')
        command(['scripts/check_contamination.py','data/processed/probe2/train.jsonl'],REP/'contamination.log')
        status('started approved 35M PT-3')
        history_path=REP/'readouts.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else []
        from probe2_report import metrics
        for step in ([] if plateau(history) else current['target_steps']):
            if any(r['step']==step for r in history): continue
            command(['scripts/pt/train_pt.py','--config',current['config'],'--tokens',str(current['tokens']),
                     '--global-batch','120','--micro','30','--lr','.0015','--seed','2027',
                     '--ckpt-every','1e9','--max-steps',str(step),'--out',str(OUT)],REP/'train.log')
            name=f'pt3_35m_tok_{round(step*PER_STEP/1e6)}M_s11'
            ck=OUT/f'tok_{round(step*PER_STEP/1e6)}M.pt'
            probeout=ROOT/f'checkpoints/pt3/probes/{name}.pt'
            command(['scripts/train_decision.py','--config','configs/pt/probe_v2.yaml','--init',str(ck),
                '--tokenizer','tokenizers/pt_32k.json','--data','data/processed/probe2/train.jsonl',
                '--batch','16','--steps','6000','--balanced-sampling','--save-every','2000','--seed','11',
                '--eval-every','0','--resume','--out',str(probeout)],REP/f'probe_{name}_train.log')
            command(['scripts/eval_worlds.py','--ckpt',str(probeout),'--dir','data/processed/worlds_v1',
                '--out',str(REP/f'probe_{name}_worlds.json'),'--limit','400'],REP/f'probe_{name}_worlds.log')
            history.append(dict(step=step,tokens=step*PER_STEP,checkpoint=str(ck),metrics=metrics(REP,name)))
            history_path.write_text(json.dumps(history,indent=2))
            status(f'readout {step*PER_STEP} tokens: '+json.dumps(history[-1]['metrics']))
            if plateau(history):
                status('stopped early: preregistered downstream plateau rule'); break
        (REP/'COMPLETE').write_text('Completed approved budget or plateau stop; no extension or deployment authorized.\n')
        status('done; results ready for review')
    except Exception as e:
        status(f'ERROR: {type(e).__name__}: {e}'); raise
    finally:
        if acquired: gpu.unlink()
        (singleton/'pid.txt').unlink(); singleton.rmdir()


if __name__=='__main__': main()
