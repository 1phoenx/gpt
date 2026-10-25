from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex as re


_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self._token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        self.special_tokens = list(special_tokens or [])
        self._special_token_ids = {
            token: self._token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self._token_to_id
        }
        self._special_pattern = (
            re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens
            else None
        )

    def _encode_piece(self, piece: str) -> list[int]:
        byte_tokens = [bytes([value]) for value in piece.encode("utf-8")]
        if not byte_tokens:
            return []

        while len(byte_tokens) > 1:
            best_index = -1
            best_rank = len(self.merges)
            for index in range(len(byte_tokens) - 1):
                rank = self._merge_ranks.get((byte_tokens[index], byte_tokens[index + 1]))
                if rank is not None and rank < best_rank:
                    best_index = index
                    best_rank = rank
            if best_index < 0:
                break

            merged = byte_tokens[best_index] + byte_tokens[best_index + 1]
            byte_tokens[best_index : best_index + 2] = [merged]

        return [self._token_to_id[token] for token in byte_tokens]

    def _encode_text(self, text: str) -> Iterator[int]:
        if self._special_pattern is None:
            for match in _PRETOKEN_PATTERN.finditer(text):
                yield from self._encode_piece(match.group())
            return

        cursor = 0
        for match in self._special_pattern.finditer(text):
            before = text[cursor : match.start()]
            for pretoken in _PRETOKEN_PATTERN.finditer(before):
                yield from self._encode_piece(pretoken.group())
            token = match.group()
            token_id = self._special_token_ids.get(token)
            if token_id is None:
                yield from self._encode_piece(token)
            else:
                yield token_id
            cursor = match.end()

        remainder = text[cursor:]
        for pretoken in _PRETOKEN_PATTERN.finditer(remainder):
            yield from self._encode_piece(pretoken.group())

    def encode(self, text: str) -> list[int]:
        return list(self._encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode_text(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    return BPETokenizer(vocab, merges, special_tokens)
