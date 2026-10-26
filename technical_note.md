# Technical Note: Tiny CPU Transformer Ablations

## What I originally thought
I expected residual connections and Pre-Norm to make optimization easier, so removing residuals or moving normalization after the sublayer should hurt validation loss. I also expected a longer context to help because the model can condition on more history.

## Experiment
I trained a 10.16M-parameter decoder-only Transformer on a tiny deterministic TinyStories-style corpus. The model uses 4 layers, width 320, 4 heads, SwiGLU, RoPE, AdamW-style updates, and a cosine learning-rate schedule. Each run used the same seed, data, optimizer settings, and 24 CPU steps; each ablation changed one structural variable.

## Results
The complete curves are in `artifacts/training_curves.svg`; the machine-readable table is `results/ablation_table.csv`.

| Run | Final validation loss | Change |
| --- | ---: | --- |
| baseline | 3.3888 | baseline |
| no_residual | 5.6388 | residual off |
| post_norm | 3.6200 | Post-Norm |
| context_64 | 2.6071 | context 64 |

## What differed from expectation
The context experiment looked best in this short run: context 64 reached validation loss 2.6071, below the baseline 3.3888. The no-residual run was much worse at 5.6388. Post-Norm also lagged the baseline at 3.6200.

## Current interpretation
Residual paths are important even at this scale because they preserve a stable route for optimization. Pre-Norm is more effective than Post-Norm for this short training budget. The context result is not evidence that longer context is intrinsically better: the context-64 run sees twice as many tokens because tokens-per-step doubles, so it has a data-throughput advantage.

## What remains unknown
The corpus and run are intentionally tiny, so these are optimization sanity checks rather than language-model benchmarks. A longer matched-token budget, multiple seeds, a real TinyStories split, and a controlled context comparison with equal tokens seen are still needed to estimate generalization effects.
