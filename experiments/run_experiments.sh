#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results checkpoints
PYTHON="${PYTHON:-python3}"

$PYTHON experiments/make_tiny_corpus.py
mkdir -p artifacts
$PYTHON - <<'PY'
import pickle
from cs336_basics.pretokenization import run_train_bpe

vocab, merges = run_train_bpe("data_small/train.txt", 2048, ["<|endoftext|>"])
with open("artifacts/tiny_bpe_2048.pkl", "wb") as handle:
    pickle.dump({"vocab": vocab, "merges": merges}, handle)
PY
$PYTHON -m cs336_basics.prepare_data data_small/train.txt data_small/train.npy --tokenizer artifacts/tiny_bpe_2048.pkl
$PYTHON -m cs336_basics.prepare_data data_small/valid.txt data_small/valid.npy --tokenizer artifacts/tiny_bpe_2048.pkl

common=(--train data_small/train.npy --valid data_small/valid.npy --steps 24 --eval-every 4
  --eval-batches 1 --batch-size 2 --d-model 320 --layers 4 --heads 4 --d-ff 1536
  --vocab-size 4096 --lr 0.0006 --min-lr 0.00006 --warmup 4 --clip 1.0 --threads 1)

$PYTHON -m cs336_basics.train "${common[@]}" --run-name baseline --context-length 32 \
  --checkpoint checkpoints/baseline.pt --log-path results/baseline.jsonl
$PYTHON -m cs336_basics.train "${common[@]}" --run-name no_residual --context-length 32 --no-residual \
  --checkpoint checkpoints/no_residual.pt --log-path results/no_residual.jsonl
$PYTHON -m cs336_basics.train "${common[@]}" --run-name post_norm --context-length 32 --norm-position post \
  --checkpoint checkpoints/post_norm.pt --log-path results/post_norm.jsonl
$PYTHON -m cs336_basics.train "${common[@]}" --run-name context_64 --context-length 64 \
  --checkpoint checkpoints/context_64.pt --log-path results/context_64.jsonl
$PYTHON experiments/plot_results.py
$PYTHON experiments/write_report.py
