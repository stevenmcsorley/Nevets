from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .formatting import pack_request, branch_attention_mask

@torch.inference_mode()
def predict_record(model, tokenizer, rec, device):
    if model.cfg.decision_head.get("coord_readout", "none") != "none":
        from .gates import predict_batch  # the coordinate readout lives in the batched decision path
        return predict_batch(model, tokenizer, [rec], device)[0]
    model.eval()
    packed=pack_request(tokenizer,rec["state"],rec["questions"],device=device,isolate_options=model.isolated_options)
    mask=model.attention_mask(packed)
    with torch.no_grad():
        h=model.hidden(packed.input_ids[None,:],packed.position_ids[None,:],mask)
        out={}
        for layout in packed.layouts:
            logits=model.decision_logits(h,layout.decide_position,layout.option_end_positions,
                                         query_entity_token_ids=layout.query_entity_token_ids,
                                         query_entity_state_positions=layout.query_entity_state_positions)
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
