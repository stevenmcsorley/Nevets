"""Measure local training throughput and turn token budgets into machine-specific estimates."""
import argparse, time
import torch
from systemone_lab.config import ModelConfig
from systemone_lab.model import SystemOneModel
from systemone_lab.training import pick_device, choose_dtype
ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/s1_35m.yaml'); ap.add_argument('--seq',type=int,default=1024); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--warmup',type=int,default=5); ap.add_argument('--steps',type=int,default=20); a=ap.parse_args()
device=pick_device(); dtype=choose_dtype(device); cfg=ModelConfig.load(a.config); model=SystemOneModel(cfg).to(device).train(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
x=torch.randint(0,cfg.vocab_size,(a.batch,a.seq+1),device=device)
for i in range(a.warmup+a.steps):
    if device.type=='cuda': torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=='cuda'): loss=model.lm_loss(x)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type=='cuda': torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    if i>=a.warmup:
        globals().setdefault('times',[]).append(dt)
mean=sum(times)/len(times); tps=(a.batch*a.seq)/mean
print(f'device={device} model={cfg.name} params={model.num_parameters()/1e6:.1f}M batch={a.batch} seq={a.seq}')
print(f'mean_step={mean:.3f}s tokens/sec={tps:,.0f}')
for budget in [100_000_000,500_000_000,1_000_000_000,3_000_000_000]:
    print(f'{budget/1e9:.1f}B tokens -> {budget/tps/3600:.1f} hours at measured raw throughput')
