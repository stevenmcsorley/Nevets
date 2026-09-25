"""Random-init causal semantic pretraining from a memory-mapped uint32 token stream."""
import argparse, random
import numpy as np
import torch
from systemone_lab.config import ModelConfig
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.model import SystemOneModel
from systemone_lab.training import pick_device, choose_dtype, save_checkpoint

ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/s1_35m.yaml"); ap.add_argument("--tokenizer",default="data/tokenizer.json"); ap.add_argument("--tokens",required=True); ap.add_argument("--steps",type=int,default=10000); ap.add_argument("--seq",type=int,default=1024); ap.add_argument("--batch",type=int,default=8); ap.add_argument("--accum",type=int,default=1); ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--out",default="checkpoints/pretrain.pt"); ap.add_argument("--seed",type=int,default=42)
a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
device=pick_device(); dtype=choose_dtype(device); cfg=ModelConfig.load(a.config); tok=LabTokenizer(a.tokenizer); cfg.vocab_size=tok.vocab_size; model=SystemOneModel(cfg).to(device); data=np.memmap(a.tokens,dtype=np.uint32,mode="r")
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.1); opt.zero_grad(set_to_none=True)
print(f"device={device} dtype={dtype} params={model.num_parameters()/1e6:.1f}M tokens={len(data):,}")
last=0.0
for step in range(1,a.steps+1):
    starts=np.random.randint(0,len(data)-(a.seq+1),size=a.batch)
    rows=np.stack([np.asarray(data[s:s+a.seq+1],dtype=np.int64) for s in starts])
    x=torch.from_numpy(rows).to(device)
    with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"): loss=model.lm_loss(x)/a.accum
    loss.backward(); last=loss.detach().float().item()*a.accum
    if step%a.accum==0:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step%50==0: print(f"step={step} lm_loss={last:.4f}")
    if step%1000==0: save_checkpoint(a.out,model,cfg,a.tokenizer,step,{"lm_loss":last})
save_checkpoint(a.out,model,cfg,a.tokenizer,a.steps,{"lm_loss":last}); print(a.out)
