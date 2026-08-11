# SPDX-License-Identifier: Apache-2.0
"""Authoritative request-local state for sparse SpecPrefill caches.

Sparse prompt caches cannot participate in ordinary prefix-cache matching.
Their physical KV length is shorter than the logical prompt position, and a
cache is valid only for the exact target, tokenizer, scorer, selector policy,
full prompt, and selected-token plan that created it.  This module owns that
metadata independently of model-layer cache objects.  It deliberately has no
MLX dependency so scheduler code can validate state before touching device
memory.
"""

from __future__ import annotations

import hashlib
import math
import struct
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

SPARSE_CACHE_STATE_VERSION = "specprefill-sparse-cache-v1"


class SparseCacheStateError(ValueError):
    """Sparse cache metadata is incomplete, inconsistent, or unsafe to reuse."""


class SparseCacheTransformUnsupported(SparseCacheStateError):
    """A cache tier cannot yet preserve sparse state as one atomic object."""


@dataclass(frozen=True)
class SparsePolicyTuning:
    """Versioned selector controls that influence a sparse cache's contents."""

    keep_pct: float
    backbone_pct: float
    halo_chunks: int
    anchor_chunks: int
    chunk_size: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.keep_pct) or not 0.0 < self.keep_pct <= 1.0:
            raise SparseCacheStateError("keep_pct must be finite and in (0, 1]")
        if not math.isfinite(self.backbone_pct) or not 0.0 <= self.backbone_pct <= 1.0:
            raise SparseCacheStateError("backbone_pct must be finite and in [0, 1]")
        if self.halo_chunks < 0:
            raise SparseCacheStateError("halo_chunks must be non-negative")
        if self.anchor_chunks < 0:
            raise SparseCacheStateError("anchor_chunks must be non-negative")
        if self.chunk_size <= 0:
            raise SparseCacheStateError("chunk_size must be positive")


@dataclass(frozen=True)
class SparseCacheExecutionConfig:
    """Configuration that must match before rows share a CB execution lane."""

    target_id: str
    tokenizer_id: str
    scorer_id: str
    selector_version: str
    tuning: SparsePolicyTuning
    state_version: str = SPARSE_CACHE_STATE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "tokenizer_id",
            "scorer_id",
            "selector_version",
            "state_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SparseCacheStateError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SparseCacheIdentity:
    """One row's complete identity required for exact sparse-cache reuse.

    Artifact IDs are caller-provided immutable identifiers, normally a model
    revision plus checksum.  Paths or object IDs are not sufficient: sharing a
    sparse target cache with a different tokenizer, scorer, selection plan, or
    full token sequence silently corrupts positional semantics.
    """

    target_id: str
    tokenizer_id: str
    scorer_id: str
    selector_version: str
    tuning: SparsePolicyTuning
    full_token_hash: str
    selection_fingerprint: str
    state_version: str = SPARSE_CACHE_STATE_VERSION

    def __post_init__(self) -> None:
        SparseCacheExecutionConfig(
            target_id=self.target_id,
            tokenizer_id=self.tokenizer_id,
            scorer_id=self.scorer_id,
            selector_version=self.selector_version,
            tuning=self.tuning,
            state_version=self.state_version,
        )
        self._validate_digest("full_token_hash", self.full_token_hash)
        self._validate_digest("selection_fingerprint", self.selection_fingerprint)

    @property
    def execution_config(self) -> SparseCacheExecutionConfig:
        return SparseCacheExecutionConfig(
            target_id=self.target_id,
            tokenizer_id=self.tokenizer_id,
            scorer_id=self.scorer_id,
            selector_version=self.selector_version,
            tuning=self.tuning,
            state_version=self.state_version,
        )

    @staticmethod
    def _validate_digest(name: str, value: str) -> None:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SparseCacheStateError(
                f"{name} must be a lowercase SHA-256 hex digest"
            )

    @classmethod
    def from_tokens(
        cls,
        *,
        target_id: str,
        tokenizer_id: str,
        scorer_id: str,
        selector_version: str,
        tuning: SparsePolicyTuning,
        tokens: Iterable[int],
        selection_fingerprint: str,
        state_version: str = SPARSE_CACHE_STATE_VERSION,
    ) -> "SparseCacheIdentity":
        return cls(
            target_id=target_id,
            tokenizer_id=tokenizer_id,
            scorer_id=scorer_id,
            selector_version=selector_version,
            tuning=tuning,
            full_token_hash=cls.hash_tokens(tokens),
            selection_fingerprint=selection_fingerprint,
            state_version=state_version,
        )

    @staticmethod
    def hash_tokens(tokens: Iterable[int]) -> str:
        """Hash an exact token sequence without delimiter ambiguity."""
        token_values = tuple(tokens)
        digest = hashlib.sha256()
        digest.update(b"vllm-mlx/specprefill/full-token-sequence/v1\0")
        digest.update(struct.pack(">Q", len(token_values)))
        for token in token_values:
            if isinstance(token, bool) or not isinstance(token, int):
                raise SparseCacheStateError("token IDs must be integers")
            if token < 0 or token > 0xFFFFFFFFFFFFFFFF:
                raise SparseCacheStateError("token ID is outside unsigned 64-bit range")
            digest.update(struct.pack(">Q", token))
        return digest.hexdigest()


@dataclass(frozen=True)
class SparseCacheRowState:
    """One batch row's physical KV occupancy and logical token coordinates."""

    identity: SparseCacheIdentity
    logical_positions: tuple[int, ...]
    physical_valid_length: int
    next_logical_position: int
    prefill_physical_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SparseCacheIdentity):
            raise SparseCacheStateError("row identity must be SparseCacheIdentity")
        if self.physical_valid_length != len(self.logical_positions):
            raise SparseCacheStateError(
                "physical_valid_length does not match logical_positions"
            )
        if self.prefill_physical_length < 0:
            raise SparseCacheStateError("prefill_physical_length must be non-negative")
        if self.prefill_physical_length > self.physical_valid_length:
            raise SparseCacheStateError(
                "prefill_physical_length cannot exceed physical_valid_length"
            )
        if self.next_logical_position < 0:
            raise SparseCacheStateError("next_logical_position must be non-negative")
        previous = -1
        for position in self.logical_positions:
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
            ):
                raise SparseCacheStateError(
                    "logical positions must be non-negative integers"
                )
            if position <= previous:
                raise SparseCacheStateError(
                    "logical positions must be strictly increasing"
                )
            previous = position
        if (
            self.logical_positions
            and self.next_logical_position <= self.logical_positions[-1]
        ):
            raise SparseCacheStateError(
                "next_logical_position must be after all logical positions"
            )
        generated_length = self.physical_valid_length - self.prefill_physical_length
        if generated_length:
            expected_generated_suffix = tuple(
                range(
                    self.next_logical_position - generated_length,
                    self.next_logical_position,
                )
            )
            if self.logical_positions[-generated_length:] != expected_generated_suffix:
                raise SparseCacheStateError(
                    "generated cache entries must form a contiguous decode suffix"
                )

    @property
    def generated_physical_length(self) -> int:
        return self.physical_valid_length - self.prefill_physical_length

    def clone(self) -> "SparseCacheRowState":
        return SparseCacheRowState(
            identity=self.identity,
            logical_positions=tuple(self.logical_positions),
            physical_valid_length=self.physical_valid_length,
            next_logical_position=self.next_logical_position,
            prefill_physical_length=self.prefill_physical_length,
        )

    def append_decode(self, count: int) -> "SparseCacheRowState":
        _require_nonnegative_count(count)
        if count == 0:
            return self.clone()
        new_positions = self.logical_positions + tuple(
            range(self.next_logical_position, self.next_logical_position + count)
        )
        return SparseCacheRowState(
            identity=self.identity,
            logical_positions=new_positions,
            physical_valid_length=self.physical_valid_length + count,
            next_logical_position=self.next_logical_position + count,
            prefill_physical_length=self.prefill_physical_length,
        )

    def rollback(self, count: int) -> "SparseCacheRowState":
        _require_nonnegative_count(count)
        if count > self.generated_physical_length:
            raise SparseCacheStateError(
                "rollback crosses the immutable sparse prefill boundary"
            )
        if count == 0:
            return self.clone()
        expected_suffix = tuple(
            range(self.next_logical_position - count, self.next_logical_position)
        )
        if self.logical_positions[-count:] != expected_suffix:
            raise SparseCacheStateError(
                "rollback requires a contiguous decode suffix at the logical cursor"
            )
        return SparseCacheRowState(
            identity=self.identity,
            logical_positions=self.logical_positions[:-count],
            physical_valid_length=self.physical_valid_length - count,
            next_logical_position=self.next_logical_position - count,
            prefill_physical_length=self.prefill_physical_length,
        )


@dataclass(frozen=True)
class SparseCacheState:
    """Immutable authoritative state for compatible CB sparse-cache rows.

    Exact prompt and selection identity are row-local: different prompts may
    share a batched execution lane only when their immutable execution config
    matches.  Prefix matching remains prohibited for every row.
    """

    rows: tuple[SparseCacheRowState, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(row, SparseCacheRowState) for row in self.rows):
            raise SparseCacheStateError(
                "sparse cache state rows must be SparseCacheRowState"
            )
        if not self.rows:
            return
        config = self.rows[0].identity.execution_config
        if any(row.identity.execution_config != config for row in self.rows[1:]):
            raise SparseCacheStateError(
                "sparse cache rows have incompatible shared execution config"
            )

    @classmethod
    def from_selection(
        cls,
        identities: SparseCacheIdentity | Sequence[SparseCacheIdentity],
        selected_logical_positions: Sequence[Sequence[int]],
        next_logical_positions: Sequence[int],
    ) -> "SparseCacheState":
        row_count = len(selected_logical_positions)
        if row_count == 0:
            raise SparseCacheStateError(
                "a sparse cache state must contain at least one row"
            )
        if len(next_logical_positions) != row_count:
            raise SparseCacheStateError(
                "selected_logical_positions and next_logical_positions must have one value per row"
            )
        if isinstance(identities, SparseCacheIdentity):
            row_identities = (identities,) * row_count
        else:
            row_identities = tuple(identities)
        if len(row_identities) != row_count:
            raise SparseCacheStateError(
                "identities and selected_logical_positions must have one value per row"
            )
        if any(not positions for positions in selected_logical_positions):
            raise SparseCacheStateError(
                "a sparse cache row must retain at least one selected logical position"
            )
        rows = tuple(
            SparseCacheRowState(
                identity=identity,
                logical_positions=tuple(positions),
                physical_valid_length=len(positions),
                next_logical_position=next_position,
                prefill_physical_length=len(positions),
            )
            for identity, positions, next_position in zip(
                row_identities,
                selected_logical_positions,
                next_logical_positions,
                strict=True,
            )
        )
        return cls(rows=rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def logical_positions(self) -> tuple[tuple[int, ...], ...]:
        return tuple(row.logical_positions for row in self.rows)

    @property
    def physical_valid_lengths(self) -> tuple[int, ...]:
        return tuple(row.physical_valid_length for row in self.rows)

    @property
    def next_logical_positions(self) -> tuple[int, ...]:
        return tuple(row.next_logical_position for row in self.rows)

    @property
    def identities(self) -> tuple[SparseCacheIdentity, ...]:
        return tuple(row.identity for row in self.rows)

    @property
    def execution_config(self) -> SparseCacheExecutionConfig | None:
        return self.rows[0].identity.execution_config if self.rows else None

    def clone(self) -> "SparseCacheState":
        return SparseCacheState(tuple(row.clone() for row in self.rows))

    def filter(self, keep_indices: Sequence[int]) -> "SparseCacheState":
        indices = tuple(keep_indices)
        _validate_indices(indices, self.row_count)
        return SparseCacheState(tuple(self.rows[index].clone() for index in indices))

    def extend(self, other: "SparseCacheState") -> "SparseCacheState":
        if not isinstance(other, SparseCacheState):
            raise SparseCacheStateError("can only extend SparseCacheState")
        if (
            self.execution_config is not None
            and other.execution_config is not None
            and self.execution_config != other.execution_config
        ):
            raise SparseCacheStateError(
                "cannot extend sparse cache state with incompatible shared execution config"
            )
        return SparseCacheState(
            tuple(row.clone() for row in self.rows)
            + tuple(row.clone() for row in other.rows)
        )

    def append_decode(self, counts: int | Sequence[int]) -> "SparseCacheState":
        row_counts = _normalize_row_counts(counts, self.row_count)
        return SparseCacheState(
            tuple(
                row.append_decode(count)
                for row, count in zip(self.rows, row_counts, strict=True)
            )
        )

    def trim(self, counts: int | Sequence[int]) -> "SparseCacheState":
        """Trim generated suffixes only; sparse prompt entries are immutable.

        Exact sparse reuse has no LCP/supersequence path.  A caller that wants
        to remove selected prompt entries must discard this state and prefill a
        fresh exact prompt rather than inventing a new logical cursor.
        """
        return self.rollback(counts)

    def rollback(self, counts: int | Sequence[int]) -> "SparseCacheState":
        row_counts = _normalize_row_counts(counts, self.row_count)
        # Build all successor rows before returning so a mixed batch never
        # publishes a partly rolled-back sparse-state object.
        rows = tuple(
            row.rollback(count)
            for row, count in zip(self.rows, row_counts, strict=True)
        )
        return SparseCacheState(rows)

    def for_quantized_storage(self) -> None:
        raise SparseCacheTransformUnsupported(
            "quantized sparse-cache storage is not supported until cache bytes and "
            "logical-position state can be transformed atomically"
        )

    def for_ssd_storage(self) -> None:
        raise SparseCacheTransformUnsupported(
            "SSD sparse-cache storage is not supported until cache bytes and "
            "logical-position state can be serialized atomically"
        )


T = TypeVar("T")


@dataclass(frozen=True)
class ExactSparseCacheEntry(Generic[T]):
    """One exact request row; payload lifecycle remains engine-owned."""

    state: SparseCacheState
    payload: T
    payload_cloner: Callable[[T], T]

    def clone_for_request(self) -> "ExactSparseCacheEntry[T]":
        return ExactSparseCacheEntry(
            state=self.state.clone(),
            payload=self.payload_cloner(self.payload),
            payload_cloner=self.payload_cloner,
        )


class ExactSparseCacheStore(Generic[T]):
    """A deliberately non-prefix store for exact sparse cache reuse only."""

    def __init__(self) -> None:
        self._entries: dict[SparseCacheIdentity, ExactSparseCacheEntry[T]] = {}
        self._lock = threading.RLock()

    def store(
        self,
        state: SparseCacheState,
        payload: T,
        *,
        clone_payload: Callable[[T], T],
    ) -> None:
        # The cache payload may have its own copy-on-write behaviour, but the
        # authoritative metadata is always detached from the active request.
        # Reuse entries are request-local; a batched state must be split first.
        if state.row_count != 1:
            raise SparseCacheStateError(
                "exact sparse cache store accepts exactly one request row"
            )
        if not callable(clone_payload):
            raise SparseCacheStateError(
                "exact sparse cache store requires a payload cloner"
            )
        entry = ExactSparseCacheEntry(
            state=state.clone(),
            payload=clone_payload(payload),
            payload_cloner=clone_payload,
        )
        with self._lock:
            self._entries[state.rows[0].identity] = entry

    def lookup(self, identity: SparseCacheIdentity) -> ExactSparseCacheEntry[T] | None:
        with self._lock:
            entry = self._entries.get(identity)
        if entry is None:
            return None
        # Payload cloning may touch MLX arrays or acquire engine/cache locks.
        # Keep the store critical section limited to the immutable entry lookup
        # so slow clones do not serialize unrelated exact-cache operations.
        return entry.clone_for_request()

    def discard(self, identity: SparseCacheIdentity) -> None:
        with self._lock:
            self._entries.pop(identity, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _require_nonnegative_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SparseCacheStateError("count must be a non-negative integer")


def _validate_indices(indices: tuple[int, ...], row_count: int) -> None:
    if len(set(indices)) != len(indices):
        raise SparseCacheStateError("filter indices must be unique")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise SparseCacheStateError("filter indices must be integers")
        if index < 0 or index >= row_count:
            raise SparseCacheStateError("filter index is outside sparse cache state")


def _normalize_row_counts(
    counts: int | Sequence[int], row_count: int
) -> tuple[int, ...]:
    if isinstance(counts, int) and not isinstance(counts, bool):
        _require_nonnegative_count(counts)
        return (counts,) * row_count
    if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
        raise SparseCacheStateError(
            "row counts must be an integer or an integer sequence"
        )
    values = tuple(counts)
    if len(values) != row_count:
        raise SparseCacheStateError("row counts must provide one value per row")
    for count in values:
        _require_nonnegative_count(count)
    return values
