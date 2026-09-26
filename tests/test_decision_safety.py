import random
import copy
import torch
import torch.nn.functional as F
import pytest
from systemone_lab.config import ModelConfig
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.formatting import pack_request, branch_attention_mask
from systemone_lab.training import decision_batch_loss, _target, save_checkpoint, load_checkpoint, check_safety
from systemone_lab.data.spatial_worlds import LABELS
from systemone_lab.eval import predict_record

@pytest.fixture
def tok(): return LabTokenizer('data/tokenizer.json')
@pytest.fixture
def model(tok): return SystemOneModel(ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16))
def rec(label='right'):
    return {'id':'test','state':'B is right of A.','questions':{'spatial':{'type':'choice','instructions':'B relative to A?', 'criteria':{k:k for k in reversed(LABELS)}}},'labels':{'spatial':label}}

@pytest.mark.parametrize('label',LABELS)
def test_mapping_positions(tok,label):
    r=rec(label); p=pack_request(tok,r['state'],r['questions']); l=p.layouts[0]
    assert l.option_keys[_target(l,r)]==label
    assert len(set(l.option_end_positions))==9
    assert all(p.input_ids[x]==tok.id('</opt>') for x in l.option_end_positions)
    assert p.input_ids[l.decide_position]==tok.id('<decide>')
    for key,pos in zip(l.option_keys,l.option_end_positions):
        encoded=tok.encode(key); assert p.input_ids[pos-len(encoded):pos].tolist()==encoded

@pytest.mark.parametrize('scorer',['cosine','scaled_dot'])
def test_loss_dtype_backward(model,tok,scorer):
    model.cfg.decision_head={'scorer':scorer,'temperature':10}
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss=decision_batch_loss(model,tok,[rec()],torch.device('cpu'))
    assert loss.dtype==torch.float32
    loss.backward(); assert torch.isfinite(model.ptr_q.weight.grad).all()
    h=torch.randn(20,32)
    with torch.autocast('cpu',dtype=torch.bfloat16): out=model.decision_logits(h,19,[3,6])
    assert out.dtype==torch.float32
    if scorer=='cosine': assert out.abs().max()<=10.00001

def test_mask_sdpa(tok):
    r=rec(); r['questions']['sibling']=copy.deepcopy(r['questions']['spatial'])
    p=pack_request(tok,r['state'],r['questions']); mask=branch_attention_mask(p.branch_ids)
    assert mask.any(-1).all()
    for i,bi in enumerate(p.branch_ids.tolist()):
        for j,bj in enumerate(p.branch_ids.tolist()):
            assert bool(mask[i,j]) == (j<=i and (bj==0 or (bi!=0 and bi==bj)))
    t=len(mask); q=torch.zeros(1,1,t,2); v=torch.eye(t)[None,None]
    actual=F.scaled_dot_product_attention(q,q,v,attn_mask=mask)[0,0]
    expected=mask.float()/mask.sum(-1,keepdim=True)
    assert torch.allclose(actual,expected,atol=1e-6)
    additive=torch.zeros_like(mask,dtype=torch.float).masked_fill(~mask,float('-inf'))
    assert torch.allclose(actual,F.scaled_dot_product_attention(q,q,v,attn_mask=additive)[0,0],atol=1e-6)

def test_eval_entire_path(model,tok):
    with torch.inference_mode(): p=predict_record(model,tok,rec(),'cpu')
    assert not model.training and abs(sum(p['spatial'].values())-1)<1e-5
    with torch.enable_grad(): predict_record(model,tok,rec(),'cpu')
    assert all(p.grad is None for p in model.parameters())

def test_checkpoint_metadata(model,tok,tmp_path):
    path=tmp_path/'test.pt'; save_checkpoint(path,model,model.cfg,'data/tokenizer.json',1)
    _,ck=load_checkpoint(path,SystemOneModel); assert ck['training_step']==1
    ck['tokenizer_sha256']='bad'; torch.save(ck,path)
    with pytest.raises(ValueError,match='tokenizer mismatch'): load_checkpoint(path,SystemOneModel)

def test_safety(tmp_path):
    path=tmp_path/'failure.json'
    with pytest.raises(FloatingPointError): check_safety({'loss':26},{},path,[rec()])
    assert path.exists()

def test_invalid_target(tok):
    r=rec('invalid'); l=pack_request(tok,r['state'],r['questions']).layouts[0]
    with pytest.raises(ValueError): _target(l,r)

def test_sibling_isolation(model,tok):
    r=rec(); p1=pack_request(tok,r['state'],r['questions'])
    r['questions']['other']={'type':'choice','instructions':'A completely different question','criteria':{'yes':'yes','no':'no'}}
    p2=pack_request(tok,r['state'],r['questions'])
    with torch.no_grad():
        a=model.hidden(p1.input_ids[None],p1.position_ids[None],branch_attention_mask(p1.branch_ids))
        b=model.hidden(p2.input_ids[None],p2.position_ids[None],branch_attention_mask(p2.branch_ids))
    assert torch.allclose(a,b[:,:len(p1.input_ids)],atol=1e-6)


def test_padding_and_soft_loss(model,tok):
    a=rec(); b=rec(); b['state']+=' C is above D.'
    a['distributions']={'spatial':{k:1/9 for k in LABELS}}
    stats={}
    loss=decision_batch_loss(model,tok,[a,b],torch.device('cpu'),stats)
    assert loss.dtype==torch.float32 and torch.isfinite(loss)
    assert stats['logit_max_abs']>=0
    loss.backward(); assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_abort(tmp_path,value):
    with pytest.raises(FloatingPointError): check_safety({'grad_norm_pre_clip':value},{},tmp_path/'failure.json',[rec()])

def test_freeze_optimizer_scheduler(model):
    import importlib.util
    spec=importlib.util.spec_from_file_location('trainer','scripts/train_decision.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg={'optimizer':{'backbone_lr':3e-6,'decision_head_lr':1e-5},'scheduler':{'type':'cosine','warmup_steps':2}}
    opt=module.make_optimizer(model,cfg,True)
    assert len(opt.param_groups)==1 and opt.param_groups[0]['name']=='decision_head'
    assert all(p.requires_grad == name.startswith(('ptr_q.','ptr_k.')) for name,p in model.named_parameters())
    opt=module.make_optimizer(model,cfg,False)
    sched=module.make_scheduler(opt,cfg,10)
    assert opt.param_groups[0]['lr']==pytest.approx(1.5e-6)
    opt.step(); sched.step()
    assert opt.param_groups[0]['lr']==pytest.approx(3e-6)
    for _ in range(9): opt.step(); sched.step()
    assert opt.param_groups[0]['lr']==0

def test_invalid_temperature(model):
    model.cfg.decision_head={'scorer':'cosine','temperature':-1}
    with pytest.raises(ValueError): model.decision_logits(torch.randn(10,32),9,[2,4])
import json
from systemone_lab.data.spatial_worlds import generate_jsonl

def test_spatial_generator_mixes_names_and_preserves_labels(tmp_path):
    path=tmp_path/'mixed.jsonl'
    generate_jsonl(path,100,seed=81,split='train',min_hops=1,max_hops=2,rename_probability=.5)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(records)==100
    renamed=sum(r['meta']['renamed'] for r in records)
    assert 30<=renamed<=70
    assert all(r['labels']['spatial'] in r['questions']['spatial']['criteria'] for r in records)
    assert all(not r['meta']['renamed'] or not r['meta']['query'][0].startswith('obj_') for r in records)

def test_spatial_generator_rename_probability_valid(tmp_path):
    with pytest.raises(ValueError,match='rename_probability'):
        generate_jsonl(tmp_path/'bad.jsonl',1,rename_probability=1.1)

def test_role_aware_query_encodes_entity_order(model,tok):
    model.cfg.decision_head={'scorer':'cosine','temperature':10.0,
                             'query_mode':'role_aware','query_scale':1.0}
    r=rec(); r['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    original=pack_request(tok,r['state'],r['questions']).layouts[0]
    r['questions']['spatial']['instructions']='What is the spatial relation of A to B?'
    reversed_layout=pack_request(tok,r['state'],r['questions']).layouts[0]
    assert original.query_entity_token_ids==(tok.encode('B'),tok.encode('A'))
    assert reversed_layout.query_entity_token_ids==(tok.encode('A'),tok.encode('B'))
    assert original.decide_position==reversed_layout.decide_position
    h=torch.randn(original.decide_position+1,model.cfg.d_model)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        before=model.decision_logits(h,original.decide_position,original.option_end_positions,
                                     query_entity_token_ids=original.query_entity_token_ids)
        after=model.decision_logits(h,reversed_layout.decide_position,reversed_layout.option_end_positions,
                                    query_entity_token_ids=reversed_layout.query_entity_token_ids)
    assert before.dtype==after.dtype==torch.float32
    assert (before-after).abs().max()>1e-4


def test_role_aware_requires_query_and_positive_scale(model):
    model.cfg.decision_head={'scorer':'cosine','temperature':10.0,'query_mode':'role_aware'}
    h=torch.randn(10,model.cfg.d_model)
    with pytest.raises(ValueError,match='requires two query entities'):
        model.decision_logits(h,9,[2,4])
    model.cfg.decision_head['query_scale']=-1
    with pytest.raises(ValueError,match='query_scale'):
        model.decision_logits(h,9,[2,4],query_entity_token_ids=([1],[2]))

def test_role_adapter_preserves_pointer_head_and_freezes_base(model,tok):
    from scripts.train_decision import make_optimizer
    old_q=model.ptr_q.weight.detach().clone()
    assert model.ptr_role is None
    model.enable_role_adapter({'scorer':'cosine','temperature':10.0,
                               'query_mode':'role_adapter','query_scale':0.5})
    assert torch.equal(model.ptr_role.weight,old_q)
    opt=make_optimizer(model,{'optimizer':{'decision_head_lr':1e-4}},role_adapter_only=True)
    assert len(opt.param_groups)==1
    assert opt.param_groups[0]['name']=='decision_head'
    assert all(p.requires_grad==name.startswith('ptr_role.') for name,p in model.named_parameters())
    r=rec(); r['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    layout=pack_request(tok,r['state'],r['questions']).layouts[0]
    h=torch.randn(layout.decide_position+1,model.cfg.d_model)
    out=model.decision_logits(h,layout.decide_position,layout.option_end_positions,
                              query_entity_token_ids=layout.query_entity_token_ids)
    assert out.dtype==torch.float32 and torch.isfinite(out).all()
    out.sum().backward()
    assert model.ptr_role.weight.grad is not None
    assert model.ptr_q.weight.grad is None


def test_api_passes_spatial_query_order(monkeypatch,tok):
    from systemone_lab import api

    class StubModel:
        seen = None
        bidirectional_state = False
        isolated_options = False

        def attention_mask(self, packed):
            return branch_attention_mask(packed.branch_ids)

        def hidden(self, input_ids, position_ids, attention_mask):
            return torch.zeros(1,input_ids.shape[1],32)

        def decision_logits(self, hidden, decide_pos, option_pos, **kwargs):
            self.seen = kwargs['query_entity_token_ids']
            return torch.zeros(len(option_pos))

    stub=StubModel()
    monkeypatch.setattr(api,'model',stub)
    monkeypatch.setattr(api,'tok',tok)
    record=rec()
    record['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    api.systemone(api.Request(state=record['state'],questions=record['questions']))
    assert stub.seen==(tok.encode('B'),tok.encode('A'))


def test_entity_state_positions_match_whole_names(tok):
    q={'spatial':{'type':'choice','instructions':'What is the spatial relation of obj_16 to obj_1?',
                  'criteria':{'above':'above','below':'below'}}}
    p=pack_request(tok,'obj_1 is above obj_16. obj_16 is left of obj_1.',q)
    first,second=p.layouts[0].query_entity_state_positions
    assert len(first)==2 and len(second)==2 and not set(first)&set(second)
    assert all(tok.decode([int(p.input_ids[i])]).strip()=='16' for i in first)
    assert all(tok.decode([int(p.input_ids[i])]).strip()=='1' for i in second)
    assert max(first+second)<p.layouts[0].option_end_positions[0]
    p=pack_request(tok,'S is below K.',{'spatial':{**q['spatial'],'instructions':'What is the spatial relation of K to Z?'}})
    assert len(p.layouts[0].query_entity_state_positions[0])==1
    assert p.layouts[0].query_entity_state_positions[1]==[]


def test_entity_binding_starts_identical_and_role_swap_negates(model,tok):
    base={'scorer':'cosine','temperature':10.0}
    r=rec(); r['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    forward=pack_request(tok,r['state'],r['questions']).layouts[0]
    r['questions']['spatial']['instructions']='What is the spatial relation of A to B?'
    swapped=pack_request(tok,r['state'],r['questions']).layouts[0]
    assert forward.query_entity_state_positions==swapped.query_entity_state_positions[::-1]
    h=torch.randn(forward.decide_position+1,model.cfg.d_model)
    model.cfg.decision_head=dict(base)
    plain=model.decision_logits(h,forward.decide_position,forward.option_end_positions)
    model.enable_entity_binding({**base,'query_mode':'entity_binding'})
    assert not model.ptr_bind.weight.any()
    def logits(layout):
        return model.decision_logits(h,layout.decide_position,layout.option_end_positions,
                                     query_entity_state_positions=layout.query_entity_state_positions)
    assert torch.allclose(logits(forward),plain,atol=1e-5)
    with torch.no_grad(): model.ptr_bind.weight.normal_(0,0.1)
    first,second=forward.query_entity_state_positions
    bound=model.ptr_bind(h[first].mean(0)-h[second].mean(0))
    q=F.normalize(model.ptr_q(h[forward.decide_position]),dim=-1)
    k=F.normalize(model.ptr_k(h[forward.option_end_positions]),dim=-1)
    assert torch.allclose(logits(forward),10*k@F.normalize(q+bound,dim=-1),atol=1e-4)
    assert torch.allclose(logits(swapped),10*k@F.normalize(q-bound,dim=-1),atol=1e-4)
    with torch.no_grad(): model.ptr_bind.weight.zero_()
    assert torch.allclose(model.decision_logits(h,forward.decide_position,forward.option_end_positions),plain,atol=1e-5)
    with torch.no_grad(): model.ptr_bind.weight.normal_(0,0.1)
    unbound=model.decision_logits(h,forward.decide_position,forward.option_end_positions)
    assert torch.allclose(unbound,plain,atol=1e-5)
    model.cfg.decision_head['scorer']='scaled_dot'
    with pytest.raises(ValueError,match='cosine'):
        logits(forward)


def test_binding_only_optimizer_trains_only_binding(model,tok):
    from scripts.train_decision import make_optimizer
    model.enable_entity_binding({'scorer':'cosine','temperature':10.0,'query_mode':'entity_binding'})
    opt=make_optimizer(model,{'optimizer':{'decision_head_lr':1e-4}},binding_only=True)
    assert [g['name'] for g in opt.param_groups]==['decision_head']
    assert all(p.requires_grad==name.startswith('ptr_bind.') for name,p in model.named_parameters())
    r=rec(); r['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    decision_batch_loss(model,tok,[r],torch.device('cpu')).backward()
    assert model.ptr_bind.weight.grad is not None and model.ptr_bind.weight.grad.abs().sum()>0
    assert model.ptr_q.weight.grad is None


def test_entity_binding_checkpoint_round_trip(model,tok,tmp_path):
    model.enable_entity_binding({'scorer':'cosine','temperature':10.0,'query_mode':'entity_binding'})
    with torch.no_grad(): model.ptr_bind.weight.normal_()
    path=tmp_path/'bind.pt'
    save_checkpoint(path,model,model.cfg,'data/tokenizer.json',1)
    loaded,_=load_checkpoint(path,SystemOneModel)
    assert torch.equal(loaded.ptr_bind.weight,model.ptr_bind.weight)


def test_state_loop_starts_identical_and_round_trips(model,tok,tmp_path):
    r=rec(); p=pack_request(tok,r['state'],r['questions']); mask=branch_attention_mask(p.branch_ids)
    with torch.no_grad(): before=model.hidden(p.input_ids[None],p.position_ids[None],mask)
    model.enable_state_loop({'scorer':'cosine','loop_iters':2,'loop_blocks':1})
    assert model.loop_gate.shape==(2,) and not model.loop_gate.any()
    with torch.no_grad():
        assert torch.allclose(model.hidden(p.input_ids[None],p.position_ids[None],mask),before,atol=1e-6)
        model.loop_gate.fill_(0.5)
        assert not torch.allclose(model.hidden(p.input_ids[None],p.position_ids[None],mask),before,atol=1e-4)
    loss=decision_batch_loss(model,tok,[r],torch.device('cpu')); loss.backward()
    assert model.loop_gate.grad is not None and torch.isfinite(model.loop_gate.grad).all()
    save_checkpoint(tmp_path/'loop.pt',model,model.cfg,'data/tokenizer.json',1)
    loaded,_=load_checkpoint(tmp_path/'loop.pt',SystemOneModel)
    assert torch.equal(loaded.loop_gate,model.loop_gate)
    with pytest.raises(ValueError,match='loop_iters'):
        model.enable_state_loop({'loop_iters':2})


def test_aux_coord_loss_trains_only_with_targets(model,tok):
    r=rec(); r['meta']={'aux_coords':{'A':[0,0],'B':[1,0]}}
    base=decision_batch_loss(model,tok,[r],torch.device('cpu'))
    model.enable_coord_head({'scorer':'scaled_dot','temperature':10.0,'aux_coord_weight':1.0})
    stats={}
    loss=decision_batch_loss(model,tok,[r],torch.device('cpu'),stats)
    assert 'aux_coord_loss' in stats and loss>base
    loss.backward(); assert model.coord_head.weight.grad.abs().sum()>0
    plain=decision_batch_loss(model,tok,[rec()],torch.device('cpu'))
    assert torch.allclose(plain,base)


def test_chain_aux_coords_are_relative_to_first_mention():
    import random, re
    from scripts.generate_chain_curriculum import generate
    from systemone_lab.data.spatial_worlds import REL
    for r in generate(20, 5, 1, 4, 2, 1, 't'):
        c=r['meta']['aux_coords']
        mention=lambda n: re.search(rf'(?<![\w-]){re.escape(n)}(?![\w-])',r['state']).start()
        assert c[min(c,key=mention)]==[0,0]
        for sentence in r['state'].split('. '):
            names=[n for n in c if re.search(rf'(?<![\w-]){re.escape(n)}(?![\w-])',sentence)]
            if len(names)==2:
                a,b=sorted(names,key=sentence.index)
                assert c[a]!=c[b]


def test_looped_arch_iterations_and_round_trip(tok,tmp_path):
    cfg=ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=3,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16,
                    decision_head={'scorer':'cosine','temperature':10.0,'state_attention':'bidirectional'},
                    arch={'type':'looped','prelude':1,'core':1,'coda':1,'iters':3})
    m=SystemOneModel(cfg); r=rec(); p=pack_request(tok,r['state'],r['questions'])
    mask=branch_attention_mask(p.branch_ids,True)
    with torch.no_grad():
        a=m.hidden(p.input_ids[None],p.position_ids[None],mask); m.iters=6
        b=m.hidden(p.input_ids[None],p.position_ids[None],mask)
    assert not torch.allclose(a,b) and torch.isfinite(b).all()
    m.iters=3; decision_batch_loss(m,tok,[r],torch.device('cpu')).backward()
    assert all(bl.attn.q.weight.grad is not None for bl in m.blocks)
    save_checkpoint(tmp_path/'looped.pt',m,m.cfg,'data/tokenizer.json',1)
    loaded,_=load_checkpoint(tmp_path/'looped.pt',SystemOneModel)
    assert loaded.iters==3 and loaded.cfg.arch['core']==1
    with pytest.raises(ValueError,match='looped arch'):
        SystemOneModel(ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=3,n_heads=4,n_kv_heads=2,d_ff=64,
                                   arch={'type':'looped','prelude':1,'core':1,'coda':2}))



@pytest.mark.parametrize('mode,scorer',[('decide','cosine'),('decide','scaled_dot'),('entity_binding','cosine')])
def test_batched_decisions_match_per_question_path(tok,mode,scorer):
    torch.manual_seed(0)
    m=SystemOneModel(ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16,
                                 decision_head={'scorer':scorer,'temperature':10.0,'query_mode':mode}))
    if mode=='entity_binding':
        m.enable_entity_binding(m.cfg.decision_head)
        with torch.no_grad(): m.ptr_bind.weight.normal_(0,0.2)
    a=rec('right'); a['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    b=rec('left'); b['state']='C is left of D. A is above B.'; b['questions']['spatial']['instructions']='What is the spatial relation of C to D?'
    c={'id':'soft','state':'X is near Y.','questions':{'ok':{'type':'noul','instructions':'Is X near Y?'},
       'pick':{'type':'choice','instructions':'Pick one','criteria':{'a':'a','b':'b','c':'c'}}},
       'labels':{'ok':True,'pick':'b'},'distributions':{'pick':{'a':0.2,'b':0.5,'c':0.3}}}
    batch=[a,b,c]; s1,s2={},{}
    new=decision_batch_loss(m,tok,batch,torch.device('cpu'),s1)
    old=decision_batch_loss(m,tok,batch,torch.device('cpu'),s2,force_legacy=True)
    assert torch.allclose(new,old,atol=1e-5)
    for key in ('logit_max_abs','q_norm_mean','k_norm_mean','probability_min'):
        assert abs(s1[key]-s2[key])<1e-4,key
    new.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_looped_k1_equals_flat_checkpoint(tok):
    cfg=ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=4,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16,
                    decision_head={'scorer':'cosine','temperature':10.0,'state_attention':'bidirectional'})
    m=SystemOneModel(cfg); r=rec(); p=pack_request(tok,r['state'],r['questions']); mask=branch_attention_mask(p.branch_ids,True)
    with torch.no_grad():
        flat=m.hidden(p.input_ids[None],p.position_ids[None],mask)
        m.set_arch({'type':'looped','prelude':1,'core':2,'coda':1,'iters':1})
        assert torch.allclose(m.hidden(p.input_ids[None],p.position_ids[None],mask),flat,atol=1e-5)
        m.iters=3; assert not torch.allclose(m.hidden(p.input_ids[None],p.position_ids[None],mask),flat,atol=1e-4)



def test_isolated_options_make_probabilities_order_invariant(tok):
    torch.manual_seed(1)
    m=SystemOneModel(ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16,
                                 decision_head={'scorer':'cosine','temperature':10.0,'query_mode':'entity_binding',
                                                'state_attention':'bidirectional','option_attention':'isolated'}))
    m.enable_entity_binding(m.cfg.decision_head)
    with torch.no_grad(): m.ptr_bind.weight.normal_(0,0.2)
    r=rec('right'); r['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    base=predict_record(m,tok,r,'cpu')['spatial']
    for seed in range(3):
        s=copy.deepcopy(r); items=list(s['questions']['spatial']['criteria'].items()); random.Random(seed).shuffle(items)
        s['questions']['spatial']['criteria']=dict(items)
        out=predict_record(m,tok,s,'cpu')['spatial']
        assert all(abs(out[k]-base[k])<1e-5 for k in base)
    m.cfg.decision_head['option_attention']='causal'
    s=copy.deepcopy(r); s['questions']['spatial']['criteria']=dict(reversed(list(s['questions']['spatial']['criteria'].items())))
    assert any(abs(predict_record(m,tok,s,'cpu')['spatial'][k]-predict_record(m,tok,r,'cpu')['spatial'][k])>1e-4 for k in base)
    stats={}; loss=decision_batch_loss(m.__class__(m.cfg) if False else m,tok,[r],torch.device('cpu'),stats)
    assert torch.isfinite(loss)


def test_nograd_warmup_and_convergence_loss(tok):
    cfg=ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=3,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16,
                    decision_head={'scorer':'cosine','temperature':10.0,'state_attention':'bidirectional'},
                    arch={'type':'looped','prelude':1,'core':1,'coda':1,'iters':2,'converge_weight':0.5})
    m=SystemOneModel(cfg); m.train(); r=rec()
    base=decision_batch_loss(m,tok,[r],torch.device('cpu'))
    assert m.fixed_point_delta is not None and m.fixed_point_delta>0
    stats={}; m.iters_nograd=3
    loss=decision_batch_loss(m,tok,[r],torch.device('cpu'),stats); loss.backward()
    assert 'fixed_point_delta' in stats and torch.isfinite(loss)
    assert all(p.grad is not None for p in m.blocks[1].parameters())
    m.eval(); m.iters_nograd=0
    with torch.no_grad(): decision_batch_loss(m,tok,[r],torch.device('cpu'))
    assert m.fixed_point_delta is None  # eval never pays for the extra iteration



def test_hinted_loop_supervises_each_iteration_within_its_hop_radius(tok):
    cfg=ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=3,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16,
                    decision_head={'scorer':'cosine','temperature':10.0,'state_attention':'bidirectional'},
                    arch={'type':'looped','prelude':1,'core':1,'coda':1,'iters':3,'hint_weight':1.0})
    m=SystemOneModel(cfg); m.train(); assert m.coord_head is not None
    r=rec(); r['state']='A is left of B. B is left of C.'
    r['meta']={'aux_coords':{'A':[0,0],'B':[1,0],'C':[2,0]},'aux_dist':{'A':0,'B':1,'C':2}}
    stats={}; loss=decision_batch_loss(m,tok,[r],torch.device('cpu'),stats); loss.backward()
    assert 'hint_loss' in stats and len(m.iter_states)==3
    assert m.coord_head.weight.grad is not None and m.coord_head.weight.grad.abs().sum()>0
    m.eval()
    with torch.no_grad(): decision_batch_loss(m,tok,[r],torch.device('cpu'))
    assert m.iter_states is None


def test_checkpoint_with_inherited_unused_binding_loads(model,tok,tmp_path):
    model.enable_entity_binding({'scorer':'cosine','temperature':10.0,'query_mode':'entity_binding'})
    model.cfg.decision_head={'scorer':'cosine','temperature':10.0,'query_mode':'decide'}  # child switches mode
    save_checkpoint(tmp_path/'child.pt',model,model.cfg,'data/tokenizer.json',1)
    loaded,_=load_checkpoint(tmp_path/'child.pt',SystemOneModel)
    assert loaded.ptr_bind is not None and loaded.cfg.decision_head['query_mode']=='decide'


@pytest.mark.parametrize('mode,iso,looped',[('entity_binding',True,True),('decide',False,False)])
def test_predict_batch_matches_predict_record(tok,mode,iso,looped):
    from systemone_lab.gates import predict_batch
    torch.manual_seed(3)
    dh={'scorer':'cosine','temperature':10.0,'query_mode':mode,'state_attention':'bidirectional'}
    if iso: dh['option_attention']='isolated'
    cfg=ModelConfig(vocab_size=tok.vocab_size,d_model=32,n_layers=3 if looped else 2,n_heads=4,n_kv_heads=2,d_ff=64,pointer_dim=16,
                    decision_head=dh,arch={'type':'looped','prelude':1,'core':1,'coda':1,'iters':2} if looped else {})
    m=SystemOneModel(cfg)
    if mode=='entity_binding':
        m.enable_entity_binding(dh)
        with torch.no_grad(): m.ptr_bind.weight.normal_(0,0.2)
    a=rec('right'); a['questions']['spatial']['instructions']='What is the spatial relation of B to A?'
    b=rec('left'); b['state']='C is left of D. A is above B.'; b['questions']['spatial']['instructions']='What is the spatial relation of C to D?'
    c={'state':'X is near Y.','questions':{'ok':{'type':'noul','instructions':'Is X near Y?'}},'labels':{'ok':True}}
    batch=predict_batch(m,tok,[a,b,c],'cpu',batch_size=2)
    for r,got in zip([a,b,c],batch):
        want=predict_record(m,tok,r,'cpu')
        for key in want:
            assert all(abs(got[key][k]-want[key][k])<1e-5 for k in want[key])
