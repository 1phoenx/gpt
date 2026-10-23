import os
from typing import BinaryIO

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk a file at occurrences of a special token so chunks can be processed independently."""
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be bytes")
    if not split_special_token:
        raise ValueError("split_special_token must not be empty")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    overlap_size = len(split_special_token) - 1
    for boundary_index in range(1, len(chunk_boundaries) - 1):
        start_position = chunk_boundaries[boundary_index]
        position = start_position
        overlap = b""

        while position < file_size:
            file.seek(position)
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                break

            search_chunk = overlap + mini_chunk
            found_at = search_chunk.find(split_special_token)
            if found_at != -1:
                found_position = position - len(overlap) + found_at
                if found_position >= start_position:
                    chunk_boundaries[boundary_index] = found_position
                    break

            overlap = search_chunk[-overlap_size:] if overlap_size else b""
            position += len(mini_chunk)
        else:
            chunk_boundaries[boundary_index] = file_size

        if position >= file_size and chunk_boundaries[boundary_index] == start_position:
            chunk_boundaries[boundary_index] = file_size

    return sorted(set(chunk_boundaries))


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-pair encoding vocabulary from a UTF-8 corpus."""
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must include the 256 byte tokens and all special tokens")
    if any(not token for token in special_tokens):
        raise ValueError("special tokens must not be empty")

    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []

    special_tokens_set = set(special_tokens)
    if special_tokens:
        special_token_pattern = re.compile(
            "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        )
    else:
        special_token_pattern = None

    frequencies: dict[str, int] = {}
    with open(input_path, encoding="utf-8", newline="") as corpus_file:
        text = corpus_file.read()

    chunks = special_token_pattern.split(text) if special_token_pattern is not None else [text]
    for chunk in chunks:
        if not chunk or chunk in special_tokens_set:
            continue
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            frequencies[pre_token] = frequencies.get(pre_token, 0) + 1

    word_counts: dict[tuple[bytes, ...], int] = {
        tuple(bytes([byte]) for byte in pre_token.encode("utf-8")): count
        for pre_token, count in frequencies.items()
    }

    words = [list(word) for word in word_counts]
    word_frequencies = [word_counts[tuple(word)] for word in words]
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}
    for word_index, word in enumerate(words):
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + word_frequencies[word_index]
            pair_to_words.setdefault(pair, set()).add(word_index)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)

        merged_token = best_pair[0] + best_pair[1]
        vocab[next_id] = merged_token
        next_id += 1

        affected_words = pair_to_words.get(best_pair, set()).copy()
        for word_index in affected_words:
            word = words[word_index]
            count = word_frequencies[word_index]

            old_pairs = set()
            for index in range(len(word) - 1):
                pair = (word[index], word[index + 1])
                old_pairs.add(pair)
                pair_counts[pair] -= count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            for pair in old_pairs:
                pair_to_words[pair].discard(word_index)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            merged_word: list[bytes] = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and (word[index], word[index + 1]) == best_pair:
                    merged_word.append(merged_token)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1
            words[word_index] = merged_word

            for index in range(len(merged_word) - 1):
                pair = (merged_word[index], merged_word[index + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                pair_to_words.setdefault(pair, set()).add(word_index)

    return vocab, merges
