import torch
from systemone_lab.formatting import branch_attention_mask

def test_questions_cannot_see_siblings():
    b=torch.tensor([0,0,0,1,1,2,2])
    m=branch_attention_mask(b)
    assert not bool(m[4,5])
    assert not bool(m[6,3])
    assert bool(m[4,0]) and bool(m[6,1])
    assert bool(m[4,3])


def test_bidirectional_state_only_changes_state_rows():
    b=torch.tensor([0,0,0,1,1,2,2])
    causal=branch_attention_mask(b); bi=branch_attention_mask(b,bidirectional_state=True)
    assert bool(bi[0,2]) and not bool(causal[0,2])
    assert not bi[:3,3:].any()
    assert torch.equal(bi[3:],causal[3:])


def test_bidirectional_state_lets_early_state_see_later_facts(tmp_path):
    from systemone_lab.config import ModelConfig
    from systemone_lab.model import SystemOneModel
    from systemone_lab.tokenizer import LabTokenizer
    from systemone_lab.formatting import pack_request
    tok=LabTokenizer('data/tokenizer.json')
    model=SystemOneModel(ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16))
    q={'s':{'type':'choice','instructions':'What is the spatial relation of A to B?','criteria':{'left':'left','right':'right'}}}
    p1=pack_request(tok,'A is left of B. C is above D.',q); p2=pack_request(tok,'A is left of B. C is below D.',q)
    def h(p,flag):
        model.cfg.decision_head={'scorer':'cosine','state_attention':'bidirectional' if flag else 'causal'}
        assert model.bidirectional_state==flag
        return model.hidden(p.input_ids[None],p.position_ids[None],branch_attention_mask(p.branch_ids,model.bidirectional_state))[0]
    with torch.no_grad():
        assert torch.allclose(h(p1,False)[:4],h(p2,False)[:4])
        assert not torch.allclose(h(p1,True)[:4],h(p2,True)[:4])
    model.cfg.decision_head={'state_attention':'sideways'}
    import pytest
    with pytest.raises(ValueError): model.bidirectional_state



def test_batched_mask_matches_per_example():
    rows=[torch.tensor([0,0,0,1,1,2,2]),torch.tensor([0,0,1,1,1,2,2])]
    b=torch.stack(rows)
    for flag in (False,True):
        batched=branch_attention_mask(b,flag)
        for i,r in enumerate(rows): assert torch.equal(batched[i],branch_attention_mask(r,flag))
