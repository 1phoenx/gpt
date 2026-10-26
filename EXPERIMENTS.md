# CPU Experiment Reproduction

All experiment work is on the `experiment` branch.

## Run

The experiment intentionally uses a small deterministic TinyStories-style corpus so it completes quickly on a CPU. The model is still 10,160,960 parameters.

```bash
git clone <repo>
cd <repo>
git switch experiment
python3 -m venv .venv
.venv/bin/pip install torch regex numpy
PYTHON=$PWD/.venv/bin/python bash experiments/run_experiments.sh
```

The script creates the corpus and 2,048-token BPE artifact, tokenizes the splits, trains four 24-step runs, writes checkpoints, and generates the report files. It uses only the Python standard library, PyTorch, NumPy, and `regex`; plotting does not require matplotlib.

## Controlled variables

| Run | Single changed variable |
| --- | --- |
| baseline | Pre-Norm, residuals enabled, context 32 |
| no_residual | residuals disabled |
| post_norm | Post-Norm instead of Pre-Norm |
| context_64 | context 64 instead of context 32 |

## Deliverables

- `artifacts/architecture.svg`: architecture diagram.
- `artifacts/training_curves.svg`: train loss, validation loss, learning rate, and tokens seen.
- `results/ablation_table.csv`: final measurements for all four runs.
- `technical_note.md`: one-page technical note covering hypothesis, experiment, results, surprise, interpretation, and open questions.
- `results/*.jsonl`: complete metrics and configuration logs.
- `checkpoints/*.pt`: model checkpoints.
