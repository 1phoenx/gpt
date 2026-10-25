from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def linear(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return inputs @ weights.transpose(-1, -2)


def embedding(weights: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return weights[token_ids]


def rmsnorm(inputs: torch.Tensor, weights: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = inputs.dtype
    values = inputs.float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * weights.float()).to(dtype)


def rope(inputs: torch.Tensor, theta: float, max_seq_len: int, token_positions: torch.Tensor) -> torch.Tensor:
    del max_seq_len
    dimension = inputs.shape[-1]
    if dimension % 2:
        raise ValueError("RoPE dimension must be even")
    frequency_index = torch.arange(0, dimension, 2, device=inputs.device, dtype=torch.float32)
    inverse_frequency = theta ** (-frequency_index / dimension)
    angles = token_positions.float().unsqueeze(-1) * inverse_frequency
    cosine = angles.cos().to(inputs.dtype)
    sine = angles.sin().to(inputs.dtype)
    even = inputs[..., 0::2]
    odd = inputs[..., 1::2]
    output = torch.empty_like(inputs)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output


def scaled_dot_product_attention(query, key, value, mask=None):
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1) @ value


def multihead_self_attention(
    inputs, q_proj, k_proj, v_proj, output_proj, num_heads,
    theta=None, max_seq_len=None, token_positions=None,
):
    leading_shape = inputs.shape[:-2]
    sequence_length, d_model = inputs.shape[-2:]
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    head_dimension = d_model // num_heads
    query = linear(q_proj, inputs).reshape(*leading_shape, sequence_length, num_heads, head_dimension).transpose(-3, -2)
    key = linear(k_proj, inputs).reshape(*leading_shape, sequence_length, num_heads, head_dimension).transpose(-3, -2)
    value = linear(v_proj, inputs).reshape(*leading_shape, sequence_length, num_heads, head_dimension).transpose(-3, -2)
    if theta is not None:
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=inputs.device).expand(*leading_shape, -1)
        positions = token_positions.unsqueeze(-2)
        query = rope(query, theta, max_seq_len or sequence_length, positions)
        key = rope(key, theta, max_seq_len or sequence_length, positions)
    mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=inputs.device).tril()
    attended = scaled_dot_product_attention(query, key, value, mask)
    attended = attended.transpose(-3, -2).reshape(*leading_shape, sequence_length, d_model)
    return linear(output_proj, attended)


def swiglu(inputs, w1, w2, w3):
    return linear(w2, F.silu(linear(w1, inputs)) * linear(w3, inputs))


def transformer_block(inputs, weights, d_model, num_heads, d_ff, max_seq_len, theta):
    del d_model, d_ff
    attention_output = multihead_self_attention(
        rmsnorm(inputs, weights["ln1.weight"]),
        weights["attn.q_proj.weight"], weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"], weights["attn.output_proj.weight"],
        num_heads, theta, max_seq_len,
    )
    residual = inputs + attention_output
    return residual + swiglu(
        rmsnorm(residual, weights["ln2.weight"]),
        weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"],
    )


def transformer_lm(in_indices, weights, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
    del vocab_size
    sequence_length = in_indices.shape[-1]
    if sequence_length > context_length:
        raise ValueError("input sequence exceeds context length")
    hidden = embedding(weights["token_embeddings.weight"], in_indices)
    for layer_index in range(num_layers):
        prefix = f"layers.{layer_index}."
        layer_weights = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        hidden = transformer_block(hidden, layer_weights, d_model, num_heads, d_ff, context_length, rope_theta)
    return linear(weights["lm_head.weight"], rmsnorm(hidden, weights["ln_final.weight"]))
