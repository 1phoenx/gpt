from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .tokenizer import get_tokenizer


def _encoded_chunks(input_path, tokenizer, max_chars=None):
    remaining = max_chars
    with open(input_path, encoding="utf-8") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), ""):
            if not chunk:
                break
            if remaining is not None:
                chunk = chunk[:remaining]
                remaining -= len(chunk)
            if chunk:
                yield tokenizer.encode(chunk)
            if remaining is not None and remaining <= 0:
                break


def encode_file(input_path, output_path, tokenizer_path, max_chars=None):
    with open(tokenizer_path, "rb") as handle:
        artifact = pickle.load(handle)
    tokenizer = get_tokenizer(artifact["vocab"], artifact["merges"], ["<|endoftext|>"])
    token_count = sum(len(ids) for ids in _encoded_chunks(input_path, tokenizer, max_chars))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = Path(output_path)
    if output.suffix != ".npy":
        output = output.with_suffix(".npy")
    ids = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint16, shape=(token_count,))
    cursor = 0
    for chunk_ids in _encoded_chunks(input_path, tokenizer, max_chars):
        end = cursor + len(chunk_ids)
        ids[cursor:end] = chunk_ids
        cursor = end
    ids.flush()
    del ids
    return token_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--tokenizer", default="artifacts/tinystories_bpe_10k.pkl")
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()
    print(f"wrote {encode_file(args.input_path, args.output_path, args.tokenizer, args.max_chars)} token IDs")


if __name__ == "__main__":
    main()
