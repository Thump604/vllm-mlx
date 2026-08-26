# SPDX-License-Identifier: Apache-2.0
"""Reference addressing for Qwen4-Exp PLE n-gram embeddings.

This module intentionally implements only the deterministic index calculation.
It does not load model weights or register a vllm-mlx model implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> tuple[int, ...]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return tuple(
        2
        * (
            _splitmix64((base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64)
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    )


@dataclass(frozen=True)
class Qwen4ExpNGramLayout:
    """Exact PLE table layout derived from a Qwen4-Exp text config."""

    unigram_vocab_size: int
    embedding_dim: int
    eos_token_id: int
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    divisible_by: int = 128
    split_parts: int = 128
    ple_layer_index: int = 0
    seed: int = 1234

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def head_dim(self) -> int:
        if self.embedding_dim % self.ngram_heads:
            raise ValueError("embedding_dim must be divisible by ngram_heads")
        return self.embedding_dim // self.ngram_heads

    @property
    def head_vocab_sizes(self) -> tuple[int, ...]:
        return tuple(
            _find_nth_prime_after(
                self.ngram_vocab_size_base - 1,
                self.ple_layer_index * self.ngram_heads + head + 1,
            )
            for head in range(self.ngram_heads)
        )

    @property
    def head_offsets(self) -> tuple[int, ...]:
        offsets = []
        current = 0
        for size in self.head_vocab_sizes:
            offsets.append(current)
            current += size
        return tuple(offsets)

    @property
    def padded_rows(self) -> int:
        rows = sum(self.head_vocab_sizes)
        return math.ceil(rows / self.divisible_by) * self.divisible_by

    @property
    def rows_per_split(self) -> int:
        if self.padded_rows % self.split_parts:
            raise ValueError("padded n-gram rows must divide evenly across splits")
        return self.padded_rows // self.split_parts

    @property
    def bf16_bytes(self) -> int:
        return self.padded_rows * self.head_dim * 2

    def split_address(self, row_id: int) -> tuple[int, int]:
        """Return ``(split_index, row_within_split)`` for a global row id."""
        if row_id < 0 or row_id >= self.padded_rows:
            raise ValueError(f"row id outside n-gram table: {row_id}")
        return divmod(row_id, self.rows_per_split)

    def indices_for_tokens(self, token_ids: Sequence[int]) -> list[tuple[int, ...]]:
        """Return the 16 global PLE row ids selected for every input token.

        History does not cross EOS boundaries, matching the released
        Transformers implementation's ``_shift_right_ignore_eos`` behavior.
        """
        sizes = self.head_vocab_sizes
        offsets = self.head_offsets
        multipliers = _build_layer_multipliers(
            self.unigram_vocab_size,
            self.ngram_size,
            self.ple_layer_index,
            self.seed,
        )
        output: list[tuple[int, ...]] = []
        segment: list[int] = []

        for token in token_ids:
            shifted = [token]
            shifted.extend(
                segment[-shift] if len(segment) >= shift else self.eos_token_id
                for shift in range(1, self.ngram_size)
            )
            rows = []
            for ngram in range(2, self.ngram_size + 1):
                start = (ngram - 2) * self.heads_per_ngram
                mixed = shifted[0] * multipliers[0]
                for position in range(1, ngram):
                    mixed ^= shifted[position] * multipliers[position]
                rows.extend(
                    mixed % sizes[head] + offsets[head]
                    for head in range(start, start + self.heads_per_ngram)
                )
            output.append(tuple(rows))
            if token == self.eos_token_id:
                segment.clear()
            else:
                segment.append(token)
                del segment[: -self.ngram_size + 1]

        return output
