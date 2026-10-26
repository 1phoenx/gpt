from __future__ import annotations

import torch

from .training import cross_entropy, get_batch, gradient_clipping


def train_step(model, optimizer, dataset, batch_size, context_length, device, max_grad_norm=None):
    inputs, targets = get_batch(dataset, batch_size, context_length, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    if max_grad_norm is not None:
        gradient_clipping(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    generated = input_ids
    for _ in range(max_new_tokens):
        context = generated[:, -getattr(model, "context_length", generated.shape[1]):]
        logits = model(context)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
            next_logits[next_logits < values[:, [-1]]] = float("-inf")
        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        generated = torch.cat((generated, next_token), dim=1)
    return generated
