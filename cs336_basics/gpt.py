from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        output = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).to(x.dtype)


class RoPEMHA(nn.Module):
    def __init__(self, d_model, num_heads, context_length, theta):
        super().__init__()
        if d_model % num_heads or (d_model // num_heads) % 2:
            raise ValueError("d_model/num_heads must be even")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        positions = torch.arange(context_length, dtype=torch.float32)
        frequencies = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inverse = theta ** (-frequencies / self.head_dim)
        angles = positions[:, None] * inverse[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def _rope(self, x):
        length = x.shape[-2]
        cos = self.cos[:length].to(x.dtype)[None, None, :, :]
        sin = self.sin[:length].to(x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        output = torch.empty_like(x)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output

    def forward(self, x):
        batch, length, d_model = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self._rope(self.q_proj(x).view(shape).transpose(1, 2))
        k = self._rope(self.k_proj(x).view(shape).transpose(1, 2))
        v = self.v_proj(x).view(shape).transpose(1, 2)
        values = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        values = values.transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.output_proj(values)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, context_length, theta,
                 residual=True, norm_position="pre"):
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError("norm_position must be 'pre' or 'post'")
        self.residual = residual
        self.norm_position = norm_position
        self.ln1 = RMSNorm(d_model)
        self.attn = RoPEMHA(d_model, num_heads, context_length, theta)
        self.ln2 = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        if self.norm_position == "pre":
            attention = self.attn(self.ln1(x))
            x = x + attention if self.residual else attention
            normalized = self.ln2(x)
            feed_forward = self.w2(torch.nn.functional.silu(self.w1(normalized)) * self.w3(normalized))
            return x + feed_forward if self.residual else feed_forward

        attention = self.attn(x)
        x = self.ln1(x + attention) if self.residual else self.ln1(attention)
        feed_forward = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return self.ln2(x + feed_forward) if self.residual else self.ln2(feed_forward)


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                 rope_theta=10000.0, residual=True, norm_position="pre"):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta,
                             residual=residual, norm_position=norm_position)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = math.sqrt(2 / (module.in_features + module.out_features))
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=1.0)

    def forward(self, token_ids):
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("input exceeds context length")
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
