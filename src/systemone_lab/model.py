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
    # stack+flatten == repeat_interleave(2, -1), but keeps the sequence length dynamic in ONNX export
    cos = torch.stack((ang.cos(), ang.cos()), -1).flatten(-2).to(q.dtype).unsqueeze(2)
    sin = torch.stack((ang.sin(), ang.sin()), -1).flatten(-2).to(q.dtype).unsqueeze(2)
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
            # expand+flatten == repeat_interleave(rep, dim=2), export-friendly
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).flatten(2, 3)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).flatten(2, 3)
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
        self.ptr_role = (nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
                         if cfg.decision_head.get("query_mode") == "role_adapter" else None)
        self.ptr_bind = (nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
                         if cfg.decision_head.get("query_mode") == "entity_binding" else None)
        self.loop_gate = None
        self.set_arch(cfg.arch or {})
        self.coord_head = (nn.Linear(cfg.d_model, 2)
                           if float(cfg.decision_head.get("aux_coord_weight", 0)) > 0 else None)
        self.apply(self._init)
        if self.ptr_bind is not None:
            nn.init.zeros_(self.ptr_bind.weight)
        self.enable_state_loop(cfg.decision_head)

    def enable_role_adapter(self, decision_head):
        """Add a separate role projection when continuing an older checkpoint."""
        if decision_head.get("query_mode") != "role_adapter":
            raise ValueError("decision head must select role_adapter")
        if self.ptr_role is None:
            self.ptr_role = nn.Linear(self.cfg.d_model, self.cfg.pointer_dim,
                                      bias=False, device=self.ptr_q.weight.device)
            with torch.no_grad():
                self.ptr_role.weight.copy_(self.ptr_q.weight)
        self.cfg.decision_head = dict(decision_head)

    def enable_entity_binding(self, decision_head):
        """Add a zero-initialized state-binding projection when continuing an older checkpoint.

        At initialization the scorer is identical to the checkpoint's plain decide-position scorer.
        """
        if decision_head.get("query_mode") != "entity_binding":
            raise ValueError("decision head must select entity_binding")
        if self.ptr_bind is None:
            self.ptr_bind = nn.Linear(self.cfg.d_model, self.cfg.pointer_dim,
                                      bias=False, device=self.ptr_q.weight.device)
            nn.init.zeros_(self.ptr_bind.weight)
        self.cfg.decision_head = dict(decision_head)

    def set_arch(self, arch):
        """Choose flat or looped layout over the existing blocks (no new weights).

        A looped layout with iters=1 computes exactly the flat model, so a trained flat checkpoint
        can be restructured into prelude / weight-tied core / coda and then trained with K > 1.
        """
        arch = dict(arch or {})
        if arch.get("type", "flat") not in ("flat", "looped"):
            raise ValueError("arch type must be flat or looped")
        if arch.get("type") == "looped":
            if arch["prelude"] + arch["core"] + arch["coda"] != self.cfg.n_layers or arch["core"] < 1:
                raise ValueError("looped arch needs prelude + core + coda == n_layers and core >= 1")
        self.cfg.arch = arch
        self.iters = int(arch.get("iters", 1))  # overridable per call for test-time depth sweeps
        self.iters_nograd = 0   # training only: warm-up core iterations run without gradient
        self.fixed_point_delta = None  # set when arch["converge_weight"] > 0 during training

    def enable_state_loop(self, decision_head):
        """Optionally re-apply the top `loop_blocks` blocks `loop_iters` extra times (weight-tied).

        Each extra pass is mixed in through a zero-initialized gate, so enabling the loop on an
        existing checkpoint leaves its outputs unchanged until the gate is trained.
        """
        iters = int(decision_head.get("loop_iters", 0)); blocks = int(decision_head.get("loop_blocks", 0))
        if iters < 0 or not 0 <= blocks <= self.cfg.n_layers or (iters > 0) != (blocks > 0):
            raise ValueError("loop_iters and loop_blocks must both be zero or both positive")
        if iters and (self.loop_gate is None or self.loop_gate.numel() != iters):
            self.loop_gate = nn.Parameter(torch.zeros(iters, device=self.ptr_q.weight.device))
        self.cfg.decision_head = dict(decision_head)

    def enable_coord_head(self, decision_head):
        """Training-only auxiliary head: regress each state object's coordinates from its mentions."""
        if float(decision_head.get("aux_coord_weight", 0)) > 0 and self.coord_head is None:
            self.coord_head = nn.Linear(self.cfg.d_model, 2, device=self.ptr_q.weight.device)
        self.cfg.decision_head = dict(decision_head)

    @property
    def isolated_options(self) -> bool:
        mode = self.cfg.decision_head.get("option_attention", "causal")
        if mode not in ("causal", "isolated"):
            raise ValueError("option_attention must be causal or isolated")
        return mode == "isolated"

    def attention_mask(self, packed):
        """The attention mask this model expects for a packed request."""
        from .formatting import branch_attention_mask
        return branch_attention_mask(packed.branch_ids, self.bidirectional_state,
                                     packed.option_ids if self.isolated_options else None)

    @property
    def bidirectional_state(self) -> bool:
        mode = self.cfg.decision_head.get("state_attention", "causal")
        if mode not in ("causal", "bidirectional"):
            raise ValueError("state_attention must be causal or bidirectional")
        return mode == "bidirectional"

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
        arch = self.cfg.arch or {}
        if arch.get("type") == "looped":
            p, c = arch["prelude"], arch["core"]
            for block in self.blocks[:p]:
                x = block(x, position_ids, attention_mask)
            inject, h = x, torch.zeros_like(x)
            core = self.blocks[p:p + c]

            def step(h):
                h = h + inject  # input injection keeps the problem visible at every iteration
                for block in core:
                    h = block(h, position_ids, attention_mask)
                return h
            if self.training and self.iters_nograd:
                # Truncated backprop through the recurrence: warm up without gradient, so the core
                # learns to improve states that have already been iterated many times.
                with torch.no_grad():
                    for _ in range(self.iters_nograd):
                        h = step(h)
                h = h.detach()
            for _ in range(self.iters):
                h = step(h)
            self.fixed_point_delta = None
            if self.training and float(arch.get("converge_weight", 0)) > 0:
                nxt = step(h)  # one more iteration should change little at a fixed point
                self.fixed_point_delta = (nxt - h).float().pow(2).mean() / h.float().pow(2).mean().clamp_min(1e-6)
            x = h
            for block in self.blocks[p + c:]:
                x = block(x, position_ids, attention_mask)
            return self.norm(x)
        for block in self.blocks:
            x = block(x, position_ids, attention_mask)
        if self.loop_gate is not None:
            top = self.blocks[-int(self.cfg.decision_head["loop_blocks"]):]
            for gate in self.loop_gate:
                y = x
                for block in top:
                    y = block(y, position_ids, attention_mask)
                x = x + gate.to(x.dtype) * (y - x)
        return self.norm(x)

    def lm_loss(self, input_ids):
        h = self.hidden(input_ids[:, :-1])
        logits = self.lm_head(h)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), input_ids[:,1:].reshape(-1))

    def decision_logits(self, hidden, decide_pos: int, option_pos: list[int], diagnostics=None,
                        query_entity_token_ids=None, query_entity_state_positions=None):
        if hidden.dim() == 3:
            hidden = hidden[0]
        assert len(option_pos) > 0 and len(set(option_pos)) == len(option_pos)
        assert all(0 <= p < decide_pos < hidden.size(0) for p in option_pos)
        q = self.ptr_q(hidden[decide_pos])
        k = self.ptr_k(hidden[option_pos])
        # Explicitly disable autocast: float() alone does not prevent matmul recasting.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            q, k = q.float(), k.float()
            query_mode = self.cfg.decision_head.get("query_mode", "decide")
            if query_mode in ("role_aware", "role_adapter"):
                if query_entity_token_ids is None:
                    raise ValueError("role_aware scoring requires two query entities")
                first, second = query_entity_token_ids
                first_emb = self.embed(torch.tensor(first, device=hidden.device)).float().mean(0)
                second_emb = self.embed(torch.tensor(second, device=hidden.device)).float().mean(0)
                projection = self.ptr_role if query_mode == "role_adapter" else self.ptr_q
                role = projection(first_emb - second_emb).float()
                scale = float(self.cfg.decision_head.get("query_scale", 1.0))
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("query_scale must be finite and positive")
                q = F.normalize(q, dim=-1) + scale * F.normalize(role, dim=-1)
            elif query_mode == "entity_binding":
                if self.cfg.decision_head.get("scorer", "scaled_dot") != "cosine":
                    raise ValueError("entity_binding requires the cosine scorer")
                # Questions without two named query entities (e.g. chess, yes/no) use the plain
                # decide query, exactly as when a queried name does not occur in the state.
                first, second = query_entity_state_positions or ((), ())
                if first and second:
                    # Contextual state occurrences of each queried name. The state is unchanged by a
                    # role swap, so reversing the query order exactly negates this term.
                    bound = hidden[first].float().mean(0) - hidden[second].float().mean(0)
                    q = F.normalize(q, dim=-1) + self.ptr_bind.weight.float() @ bound
            elif query_mode != "decide":
                raise ValueError("unknown query mode")
            if self.cfg.decision_head.get("scorer", "scaled_dot") == "cosine":
                temperature = float(self.cfg.decision_head.get("temperature", 10.0))
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError("decision temperature must be finite and positive")
                logits = temperature * (F.normalize(k, dim=-1) @ F.normalize(q, dim=-1))
            elif self.cfg.decision_head.get("scorer", "scaled_dot") == "scaled_dot":
                logits = (k @ q) / math.sqrt(q.numel())
            else:
                raise ValueError("unknown decision scorer")
        if diagnostics is not None:
            diagnostics.append({"q": q.detach(), "k": k.detach(), "logits": logits.detach()})
        return logits

    def supports_batched_decisions(self) -> bool:
        return self.cfg.decision_head.get("query_mode", "decide") in ("decide", "entity_binding")

    def decision_logits_batch(self, hidden, b_idx, decide, opt_pos, opt_mask, bind=None):
        """Score every question of a padded batch at once; matches decision_logits per question.

        hidden [B,T,D]; b_idx, decide [N]; opt_pos, opt_mask [N,K]; bind is an optional
        (rows, b, pos, weight) tuple of flat tensors whose weighted sum per row gives the
        entity_binding difference. Returns logits [N,K] (padding = -1e9), q [N,P], k [N,K,P].
        """
        q = self.ptr_q(hidden[b_idx, decide])
        k = self.ptr_k(hidden[b_idx[:, None].expand_as(opt_pos), opt_pos])
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            q, k = q.float(), k.float()
            mode = self.cfg.decision_head.get("query_mode", "decide")
            if mode == "entity_binding":
                if self.cfg.decision_head.get("scorer", "scaled_dot") != "cosine":
                    raise ValueError("entity_binding requires the cosine scorer")
                if bind is not None and bind[0].numel():
                    rows, bb, pos, w = bind
                    bound = torch.zeros(q.size(0), hidden.size(-1), device=q.device)
                    bound.index_add_(0, rows, hidden[bb, pos].float() * w[:, None])
                    has = torch.zeros(q.size(0), dtype=torch.bool, device=q.device); has[rows] = True
                    # Same as decision_logits: only bound questions use normalize(q) + binding.
                    q = torch.where(has[:, None], F.normalize(q, dim=-1) + bound @ self.ptr_bind.weight.float().t(), q)
            elif mode != "decide":
                raise ValueError("batched decisions support decide and entity_binding query modes")
            scorer = self.cfg.decision_head.get("scorer", "scaled_dot")
            if scorer == "cosine":
                temperature = float(self.cfg.decision_head.get("temperature", 10.0))
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError("decision temperature must be finite and positive")
                logits = temperature * torch.einsum("nkp,np->nk", F.normalize(k, dim=-1), F.normalize(q, dim=-1))
            elif scorer == "scaled_dot":
                logits = torch.einsum("nkp,np->nk", k, q) / math.sqrt(q.size(-1))
            else:
                raise ValueError("unknown decision scorer")
            logits = logits.masked_fill(~opt_mask, -1e9)
        return logits, q, k

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
