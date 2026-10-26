from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .gpt import GPTLanguageModel
from .pipeline import generate
from .training import AdamW, cross_entropy, get_batch, gradient_clipping, cosine_lr, load_checkpoint, save_checkpoint
from .tokenizer import get_tokenizer


class FastAdamW:
    """Minimal AdamW that avoids torch.optim's optional dynamo import on CPU."""

    def __init__(self, parameters, lr, weight_decay):
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self):
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            state = self.state.setdefault(parameter, {
                "step": 0,
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            })
            state["step"] += 1
            beta1, beta2 = 0.9, 0.999
            state["exp_avg"].mul_(beta1).add_(parameter.grad, alpha=1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(parameter.grad, parameter.grad, value=1 - beta2)
            step = state["step"]
            bias1 = 1 - beta1**step
            bias2 = 1 - beta2**step
            denominator = state["exp_avg_sq"].sqrt() / math.sqrt(bias2) + 1e-8
            parameter.mul_(1 - self.lr * self.weight_decay)
            parameter.addcdiv_(state["exp_avg"], denominator, value=-self.lr / bias1)

    @property
    def param_groups(self):
        return [{"lr": self.lr}]

    def state_dict(self):
        return {"lr": self.lr, "weight_decay": self.weight_decay, "step": max(
            (value["step"] for value in self.state.values()), default=0
        )}


def evaluate(model, data, batch_size, context_length, device, batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = get_batch(data, batch_size, context_length, device)
            logits = model(inputs)
            losses.append(cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1)))
    model.train()
    return torch.stack(losses).mean().item()


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-position", choices=["pre", "post"], default="pre")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train_data = np.load(args.train, mmap_mode="r")
    valid_data = np.load(args.valid, mmap_mode="r")
    model = GPTLanguageModel(args.vocab_size, args.context_length, args.d_model, args.layers, args.heads, args.d_ff,
                             residual=args.residual, norm_position=args.norm_position).to(args.device)
    optimizer = FastAdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log_path = Path(args.log_path) if args.log_path else Path(args.checkpoint).with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": args.run_name,
        "parameters": parameter_count(model),
        "context_length": args.context_length,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "d_ff": args.d_ff,
        "residual": args.residual,
        "norm_position": args.norm_position,
        "seed": args.seed,
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"type": "config", **config}) + "\n")
    for iteration in range(args.steps):
        lr = cosine_lr(iteration, args.lr, args.min_lr, args.warmup, args.steps)
        optimizer.lr = lr
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1))
        loss.backward()
        gradient_clipping(model.parameters(), args.clip)
        optimizer.step()
        if iteration == 0 or (iteration + 1) % args.eval_every == 0 or iteration + 1 == args.steps:
            valid_loss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device, args.eval_batches)
            record = {"type": "metrics", "step": iteration + 1, "train_loss": loss.item(),
                      "valid_loss": valid_loss, "lr": lr,
                      "tokens_seen": (iteration + 1) * args.batch_size * args.context_length,
                      "elapsed_s": time.time() - start}
            print(json.dumps(record), flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            save_checkpoint(model, optimizer, iteration + 1, args.checkpoint)


if __name__ == "__main__":
    main()
