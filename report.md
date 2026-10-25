# GPT Reproduction Report

## Scope and Status

This repository implements the main forward and optimization path of the CS336 Assignment 1 language model. It is a compact GPT-style decoder-only Transformer trained with next-token prediction. The core modules are implemented and the assignment's focused model/training tests pass. This does not mean that a full TinyStories training run, validation/evaluation workflow, leaderboard run, or polished command-line training application has been completed.

The implemented flow is:

```text
UTF-8 text
  -> byte-level BPE tokenizer
  -> token IDs
  -> token embeddings
  -> repeated pre-norm Transformer blocks
       -> RMSNorm
       -> causal multi-head self-attention (RoPE on Q/K)
       -> residual addition
       -> RMSNorm
       -> SwiGLU MLP
       -> residual addition
  -> final RMSNorm
  -> vocabulary logits
  -> next-token cross entropy
  -> autograd backward pass
  -> gradient clipping (optional)
  -> AdamW parameter update
  -> autoregressive token sampling
  -> tokenizer decode to text
```

## Files and Responsibilities

- `cs336_basics/pretokenization.py`: GPT-2-style regex pre-tokenization, special-token segmentation, BPE vocabulary/merge training, and chunk-boundary helper. The current training implementation reads and pre-tokenizes the full file in one process; the helper exists but is not used for parallel pre-tokenization.
- `cs336_basics/tokenizer.py`: BPE encoding and decoding. It applies learned merges in rank order per pre-token, preserves configured special tokens, and offers `encode_iterable` for streamed input strings.
- `artifacts/tinystories_bpe_10k.pkl`: serialized TinyStories tokenizer artifact. The recorded run used the TinyStories V2 GPT-4 training split, with vocabulary size 10,000, 9,743 merges, and 356.9 seconds reported in `logs/train_bpe_10k.log`.
- `data/TinyStoriesV2-GPT4-train.txt` and `data/TinyStoriesV2-GPT4-valid.txt`: downloaded training and validation text. The validation file is not evidence that a validation evaluation was run.
- `cs336_basics/model.py`: tensor operations and the complete Transformer forward path. `transformer_lm` consumes token IDs and a supplied state dictionary and returns unnormalized next-token logits.
- `cs336_basics/training.py`: random next-token batch sampling, stable softmax and cross entropy, global gradient clipping, AdamW, warmup/cosine learning-rate schedule, and checkpoint save/load.
- `cs336_basics/pipeline.py`: one optimization step (`train_step`) and autoregressive generation (`generate`) with temperature and optional top-k sampling.
- `tests/adapters.py`: connects assignment test hooks to the implementation. Section 2, Section 3, and selected Section 4/5 hooks are wired. Other later assignment hooks may still be unimplemented.
- `tests/test_model.py`, `tests/test_train_bpe.py`, `tests/test_tokenizer.py`, `tests/test_nn_utils.py`, `tests/test_optimizer.py`, `tests/test_data.py`, `tests/test_serialization.py`: correctness and regression tests for the corresponding pieces.
- `main/`: duplicate working copies of tokenizer/model source currently present on the remote worktree. The assignment package and test adapters import `cs336_basics`; treat that directory as canonical unless you deliberately reconcile the duplicate.

## Architecture Notes

### Tokenizer

Training starts from the 256 single-byte tokens, adds configured special tokens, and repeatedly selects the most frequent adjacent byte-token pair. Each selected pair is appended to the ordered merge list and its concatenated bytes are added to the vocabulary. Pair counts are updated for affected pre-token sequences. Special tokens split the corpus into independent regions so merges cannot cross document boundaries.

For inference, text is split into regex pre-tokens. Each pre-token is encoded as UTF-8 bytes and the available adjacent merges are applied from lowest merge rank to highest. Special-token matches are emitted as their dedicated IDs. Decoding concatenates token bytes and decodes UTF-8.

### Transformer

The model uses weights in the assignment's reference state-dict layout. For a hidden tensor `x`, each block performs:

```text
y = x + MHA(RMSNorm(x))
z = y + SwiGLU(RMSNorm(y))
```

MHA projects Q/K/V, reshapes them into heads, applies RoPE to Q and K, performs scaled dot-product attention under a lower-triangular causal mask, concatenates the heads, then applies the output projection. The LM embeds IDs, runs the configured number of blocks, applies final RMSNorm, and projects to vocabulary logits. The logits are not softmax-normalized; cross entropy consumes them directly.

### Optimization and Generation

`get_batch` samples contiguous token windows. Inputs contain `context_length` tokens; targets are the same windows shifted forward by one token. `train_step` clears old gradients, computes logits and mean cross entropy, calls `backward()`, optionally clips the global gradient norm, and performs `optimizer.step()`. AdamW maintains first and second moments, applies bias correction, and decoupled weight decay. Checkpoints preserve model state, optimizer state, and iteration count.

`generate` repeatedly feeds the current IDs to a model, takes the final position's logits, applies temperature and optional top-k filtering, samples one token, and appends it. Convert the final IDs to readable text with the tokenizer's `decode` method. The helper expects the caller to manage prompt tokenization and any context-window truncation required by the chosen model.

## Reproducing the Checks

From `/root/gpt`, using the project's configured environment:

```sh
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py -q
uv run pytest tests/test_model.py tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py -q
```

Recorded results on the remote environment:

```text
Model/training/data/checkpoint tests: 20 passed
Tokenizer/BPE tests: 27 passed, 1 xfailed
```

The xfailed tokenizer test is marked expected-failure by the test suite. A separate small smoke test ran one `train_step`, confirmed finite loss and changed parameters, and generated four additional IDs. This was not a full language-model training run and does not establish text quality.

The remote Python environment was missing several test dependencies initially; `pytest`, `einops`, `jaxtyping`, and `tiktoken` were installed with pip to execute these checks. For a clean reproduction, prefer `uv sync`/`uv run` from `pyproject.toml` and `uv.lock` rather than relying on those global pip installs.

## Not Yet Completed or Verified

- No full training loop/CLI currently orchestrates epochs or iterations, periodic validation, learning-rate updates, logging, and checkpoint cadence. `train_step` is the reusable per-step primitive.
- No full training run on the 2.23 GB TinyStories training file has been completed with this Transformer; the reported 356.9 seconds is tokenizer training time only.
- Validation loss/perplexity, sample-quality review, and a learning curve have not been produced.
- Tokenized dataset arrays have not been prepared and saved as reusable `uint16` files. The tokenizer output is saved, but model training needs a token-ID dataset.
- The artifact/log does not verify peak memory or longest vocabulary token, and the run command is not recorded there.
- Parallel BPE pre-tokenization is not wired into `run_train_bpe`.
- The complete assignment suite (`uv run pytest`) was not run. The tested subsets above passed; other stubs in `tests/adapters.py` correspond to later assignment items such as other optimizer/scheduler/checkpoint-related functions and need a fresh inventory before claiming full assignment completion.
- Generation is covered by a smoke test, not a dedicated test suite. Prompt encoding/decoding, context truncation, stopping tokens, and output-quality evaluation remain caller responsibilities.

## Suggested Independent Reproduction Order

1. Read Sections 2 and 3 of `cs336_assignment1_basics.pdf` alongside `pretokenization.py`, `tokenizer.py`, and `model.py`.
2. Run tokenizer tests, then inspect the serialized vocabulary and merge list. Record the exact training command, peak memory, and longest token for your own write-up.
3. Encode training and validation text to token-ID arrays, saving them in a compact integer format appropriate for the vocabulary. Keep the splits separate.
4. Instantiate the model using the assignment configuration, inspect parameter count and tensor shapes, and run a tiny batch through the forward pass.
5. Use `get_batch` and `train_step` on a small subset first. Check that loss is finite, gradients exist, and parameters change.
6. Add an outer loop that updates the optimizer learning rate from `cosine_lr`, periodically evaluates validation loss without gradients, and saves checkpoints.
7. Train on the intended dataset/configuration, record wall-clock time and memory, then generate samples by encoding a prompt, calling `generate`, and decoding IDs.
8. Compare train and validation loss and review generated samples before making claims about model quality.

## Important Worktree Note

The remote repository contains user/worktree changes including untracked datasets and artifacts, an untracked uppercase `AGENTS.MD`, and deletions of the original `AGENTS.md` and `CLAUDE.md`. These were preserved. This report does not attempt to reconcile or remove those existing files.
