import argparse, random, time
import torch
from systemone_lab.config import ModelConfig
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.model import SystemOneModel
from systemone_lab.data.io import read_jsonl
from systemone_lab.training import pick_device, choose_dtype, decision_batch_loss, save_checkpoint

ap=argparse.ArgumentParser()
ap.add_argument("--config",default="configs/s1_35m.yaml")
ap.add_argument("--tokenizer",default="data/tokenizer.json")
ap.add_argument("--data",required=True)
ap.add_argument("--init",default=None,help="Optional from-scratch LM checkpoint to continue from")
ap.add_argument("--steps",type=int,default=5000)
ap.add_argument("--lr",type=float,default=3e-4)
ap.add_argument("--batch",type=int,default=8)
ap.add_argument("--accum",type=int,default=1)
ap.add_argument("--save-every",type=int,default=500)
ap.add_argument("--out",default="checkpoints/spatial.pt")
ap.add_argument("--seed",type=int,default=42)
a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)

device=pick_device(); dtype=choose_dtype(device); tok=LabTokenizer(a.tokenizer)
if a.init:
    ck=torch.load(a.init,map_location="cpu")
    cfg=ModelConfig(**ck["config"]); cfg.vocab_size=tok.vocab_size
    model=SystemOneModel(cfg); model.load_state_dict(ck["model"])
    print(f"continued from {a.init}")
else:
    cfg=ModelConfig.load(a.config); cfg.vocab_size=tok.vocab_size; model=SystemOneModel(cfg)
model=model.to(device)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.1)
records=list(read_jsonl(a.data)); print(f"device={device} dtype={dtype} params={model.num_parameters()/1e6:.1f}M records={len(records)} batch={a.batch} accum={a.accum}")
model.train(); opt.zero_grad(set_to_none=True); t=time.time(); last=0.0
for step in range(1,a.steps+1):
    recs=random.choices(records,k=a.batch)
    with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"):
        loss=decision_batch_loss(model,tok,recs,device)/a.accum
    loss.backward(); last=float(loss)*a.accum
    if step%a.accum==0:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step%50==0: print(f"step={step} loss={last:.4f} examples/s={(step*a.batch)/(time.time()-t):.1f}")
    if step%a.save_every==0: save_checkpoint(a.out,model,cfg,a.tokenizer,step,{"train_loss":last})
save_checkpoint(a.out,model,cfg,a.tokenizer,a.steps,{"train_loss":last}); print(a.out)
