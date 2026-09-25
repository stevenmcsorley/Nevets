from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .formatting import pack_request, branch_attention_mask

def predict_record(model, tokenizer, rec, device):
    packed=pack_request(tokenizer,rec["state"],rec["questions"],device=device)
    mask=branch_attention_mask(packed.branch_ids)
    with torch.no_grad():
        h=model.hidden(packed.input_ids[None,:],packed.position_ids[None,:],mask)
        out={}
        for layout in packed.layouts:
            logits=model.decision_logits(h,layout.decide_position,layout.option_end_positions)
            probs=F.softmax(logits.float(),-1)
            out[layout.key]={k:float(v) for k,v in zip(layout.option_keys,probs.cpu())}
    return out

def ece(conf, correct, bins=15):
    total=len(conf); s=0.0
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins
        ids=[i for i,c in enumerate(conf) if lo <= c <= hi if (b==bins-1 or c<hi)]
        if not ids: continue
        acc=sum(correct[i] for i in ids)/len(ids)
        avg=sum(conf[i] for i in ids)/len(ids)
        s += len(ids)/total*abs(acc-avg)
    return s
