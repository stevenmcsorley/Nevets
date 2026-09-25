from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k, position_ids, theta=10000.0):
    d = q.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d))
    ang = position_ids.float().unsqueeze(-1) * inv.unsqueeze(0)
    cos = torch.repeat_interleave(ang.cos(), 2, -1).to(q.dtype).unsqueeze(2)
    sin = torch.repeat_interleave(ang.sin(), 2, -1).to(q.dtype).unsqueeze(2)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.nh = cfg.n_heads
        self.nkv = cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.theta = cfg.rope_theta

    def forward(self, x, position_ids, allow_mask=None):
        B,T,_ = x.shape
        q = self.q(x).view(B,T,self.nh,self.hd)
        k = self.k(x).view(B,T,self.nkv,self.hd)
        v = self.v(x).view(B,T,self.nkv,self.hd)
        q, k = apply_rope(q, k, position_ids, self.theta)
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        if allow_mask is None:
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            if allow_mask.dim() == 2:
                allow_mask = allow_mask[None,None,:,:]
            elif allow_mask.dim() == 3:
                allow_mask = allow_mask[:,None,:,:]
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=allow_mask,is_causal=False)
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        return self.o(y)

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg)
    def forward(self, x, pos, mask=None):
        x = x + self.attn(self.n1(x), pos, mask)
        x = x + self.ff(self.n2(x))
        return x

class SystemOneModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.ptr_q = nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
        self.ptr_k = nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def hidden(self, input_ids, position_ids=None, attention_mask=None):
        B,T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).expand(B,T)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, position_ids, attention_mask)
        return self.norm(x)

    def lm_loss(self, input_ids):
        h = self.hidden(input_ids[:, :-1])
        logits = self.lm_head(h)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), input_ids[:,1:].reshape(-1))

    def decision_logits(self, hidden, decide_pos: int, option_pos: list[int]):
        # hidden: [T,D] or [1,T,D]
        if hidden.dim() == 3:
            hidden = hidden[0]
        q = self.ptr_q(hidden[decide_pos])
        k = self.ptr_k(hidden[torch.tensor(option_pos, device=hidden.device)])
        return (k @ q) / math.sqrt(q.numel())

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
