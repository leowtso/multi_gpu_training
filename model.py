"""GPT-2 style decoder-only transformer.

Kept deliberately flat (no nn.ModuleDict nesting tricks) so that later wrapping
with DDP / FSDP2 `fully_shard` per `Block` is straightforward.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    # GPT-2 has 50257 tokens; padding to a multiple of 128 gives the LM-head
    # matmul tensor-core friendly shapes. Extra logits are never targets.
    vocab_size: int = 50304


PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, n_embd=128, block_size=256),
    "124m": dict(n_layer=12, n_head=12, n_embd=768),
    "350m": dict(n_layer=24, n_head=16, n_embd=1024),
    "774m": dict(n_layer=36, n_head=20, n_embd=1280),
    "1.5b": dict(n_layer=48, n_head=25, n_embd=1600),
    "2.7b": dict(n_layer=32, n_head=32, n_embd=2560),
    "6.7b": dict(n_layer=32, n_head=32, n_embd=4096),  # GPT-3 6.7B shape
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj.is_residual_proj = True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention kernel
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.proj.is_residual_proj = True

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "is_residual_proj", False):
                # GPT-2: scale residual projections so the residual stream's
                # variance doesn't grow with depth (2 residual adds per block).
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        # autocast runs cross_entropy in fp32; the logits tensor (B*T*vocab) is
        # often the single largest activation, so avoid an extra explicit copy.
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

    def num_params(self, exclude_pos_emb=True):
        # parameters() de-duplicates the tied wte/lm_head weight.
        n = sum(p.numel() for p in self.parameters())
        return n - self.wpe.weight.numel() if exclude_pos_emb else n

    def flops_per_token(self, seq_len):
        """Training FLOPs per token (forward + backward), PaLM appendix B convention.

        6N for the weight matmuls (2 fwd + 4 bwd per param) plus attention
        score/value matmuls, 12 * L * d * T. Causal masking is *not* discounted,
        matching the usual MFU convention.
        """
        c = self.cfg
        return 6 * self.num_params() + 12 * c.n_layer * c.n_embd * seq_len


def build_model(name: str, **overrides) -> GPT:
    cfg = GPTConfig(**{**PRESETS[name], **overrides})
    return GPT(cfg)
