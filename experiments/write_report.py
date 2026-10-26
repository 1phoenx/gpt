from __future__ import annotations

import csv
import json
from pathlib import Path


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows[0], [row for row in rows if row["type"] == "metrics"][-1]


def main():
    root = Path(__file__).resolve().parents[1]
    files = [("baseline", "baseline"), ("no_residual", "no_residual"),
             ("post_norm", "post_norm"), ("context_64", "context_64")]
    records = [(name, *read(root / "results" / f"{filename}.jsonl")) for name, filename in files]
    with (root / "results" / "ablation_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run", "parameters", "context", "residual", "norm", "train_loss", "valid_loss", "tokens_seen"])
        for name, config, final in records:
            writer.writerow([name, config["parameters"], config["context_length"], config["residual"], config["norm_position"],
                             f'{final["train_loss"]:.4f}', f'{final["valid_loss"]:.4f}', final["tokens_seen"]])
    baseline = records[0][2]["valid_loss"]
    lines = [
        "# Technical Note: Tiny CPU Transformer Ablations",
        "",
        "## What I originally thought",
        "I expected residual connections and Pre-Norm to make optimization easier, so removing residuals or moving normalization after the sublayer should hurt validation loss. I also expected a longer context to help because the model can condition on more history.",
        "",
        "## Experiment",
        "I trained a 10.16M-parameter decoder-only Transformer on a tiny deterministic TinyStories-style corpus. The model uses 4 layers, width 320, 4 heads, SwiGLU, RoPE, AdamW-style updates, and a cosine learning-rate schedule. Each run used the same seed, data, optimizer settings, and 24 CPU steps; each ablation changed one structural variable.",
        "",
        "## Results",
        "The complete curves are in `artifacts/training_curves.svg`; the machine-readable table is `results/ablation_table.csv`.",
        "",
        "| Run | Final validation loss | Change |",
        "| --- | ---: | --- |",
    ]
    for name, config, final in records:
        change = "baseline" if name == "baseline" else ("residual off" if name == "no_residual" else ("Post-Norm" if name == "post_norm" else "context 64"))
        lines.append(f'| {name} | {final["valid_loss"]:.4f} | {change} |')
    lines += [
        "",
        "## What differed from expectation",
        f"The context experiment looked best in this short run: context 64 reached validation loss {records[3][2]['valid_loss']:.4f}, below the baseline {baseline:.4f}. The no-residual run was much worse at {records[1][2]['valid_loss']:.4f}. Post-Norm also lagged the baseline at {records[2][2]['valid_loss']:.4f}.",
        "",
        "## Current interpretation",
        "Residual paths are important even at this scale because they preserve a stable route for optimization. Pre-Norm is more effective than Post-Norm for this short training budget. The context result is not evidence that longer context is intrinsically better: the context-64 run sees twice as many tokens because tokens-per-step doubles, so it has a data-throughput advantage.",
        "",
        "## What remains unknown",
        "The corpus and run are intentionally tiny, so these are optimization sanity checks rather than language-model benchmarks. A longer matched-token budget, multiple seeds, a real TinyStories split, and a controlled context comparison with equal tokens seen are still needed to estimate generalization effects.",
    ]
    (root / "technical_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
