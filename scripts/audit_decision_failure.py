"""Read-only numerical forensics; never changes or trains either checkpoint."""
import json, math, types
from pathlib import Path
import torch
from systemone_lab.config import ModelConfig
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import decision_batch_loss, parameter_norm
from systemone_lab.formatting import pack_request,branch_attention_mask

records=[]
with Path('data/processed/spatial_train.jsonl').open() as f:
    for _ in range(8): records.append(json.loads(next(f)))
tok=LabTokenizer('data/tokenizer.json'); results={}
for name,path in [('pretrained','checkpoints/s1-35m-pretrain.pt'),('failed','reports/evidence_v2/s1-35m-spatial.failed.pt'),('repaired','checkpoints/decision-gate-800.pt')]:
    ck=torch.load(path,map_location='cpu'); model=SystemOneModel(ModelConfig(**ck['config'])).cuda(); model.load_state_dict(ck['model']); del ck
    variants=['bf16_dot','fp32_dot'] if name!='repaired' else ['cosine']
    results[name]={}
    for variant in variants:
        scores=[]
        def scorer(self,h,decide_pos,option_pos,diagnostics=None):
            if h.ndim==3: h=h[0]
            q=self.ptr_q(h[decide_pos]); k=self.ptr_k(h[option_pos])
            if variant=='bf16_dot': logits=(k@q)/math.sqrt(q.numel())
            else:
                with torch.autocast('cuda',enabled=False):
                    if variant=='cosine': logits=10*(torch.nn.functional.normalize(k.float(),dim=-1)@torch.nn.functional.normalize(q.float(),dim=-1))
                    else: logits=(k.float()@q.float())/math.sqrt(q.numel())
            z={'q':q.detach().float(),'k':k.detach().float(),'logits':logits.detach().float()}
            if diagnostics is not None: diagnostics.append(z)
            scores.append(z)
            return logits
        model.decision_logits=types.MethodType(scorer,model); model.zero_grad(set_to_none=True); stats={}
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=decision_batch_loss(model,tok,records,torch.device('cuda'),stats)
        loss.backward(); stats['loss']=loss.detach().float().item(); stats['grad_norm_pre_clip']=parameter_norm(model.parameters(),True)
        stats['per_example_loss']=[torch.nn.functional.cross_entropy(s['logits'][None],torch.tensor([list(records[i]['questions']['spatial']['criteria']).index(records[i]['labels']['spatial'])],device='cuda')).item() for i,s in enumerate(scores)]
        stats['first_logits']=scores[0]['logits'].tolist(); stats['first_k_pair_cosine']=torch.nn.functional.cosine_similarity(scores[0]['k'][0],scores[0]['k'][1],dim=0).item()
        results[name][variant]=stats
    del model; torch.cuda.empty_cache()
Path('reports/checkpoint_forensics.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
