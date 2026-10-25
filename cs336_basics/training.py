from __future__ import annotations

import math
from typing import BinaryIO

import numpy as np
import torch
from torch import Tensor


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    if dataset.ndim != 1 or len(dataset) <= context_length:
        raise ValueError("dataset must be one-dimensional and longer than context_length")
    starts = torch.randint(len(dataset) - context_length, (batch_size,))
    offsets = torch.arange(context_length)
    indices = starts[:, None] + offsets[None, :]
    inputs = torch.as_tensor(dataset[indices.numpy()], dtype=torch.long, device=device)
    targets = torch.as_tensor(dataset[(indices + 1).numpy()], dtype=torch.long, device=device)
    return inputs, targets


def softmax(inputs: Tensor, dim: int = -1) -> Tensor:
    shifted = inputs - inputs.amax(dim=dim, keepdim=True)
    return shifted.exp() / shifted.exp().sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    log_normalizer = torch.logsumexp(inputs, dim=-1)
    correct = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - correct).mean()


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for parameter in parameters:
            parameter.grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW hyperparameters")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step = state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = group["lr"] / bias_correction1
                denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denominator.add_(group["eps"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)
        return loss


def cosine_lr(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coefficient = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + coefficient * (max_learning_rate - min_learning_rate)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | BinaryIO) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src: str | BinaryIO, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
