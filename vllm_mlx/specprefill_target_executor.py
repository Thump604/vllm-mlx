# SPDX-License-Identifier: Apache-2.0
"""Bounded, request-local target execution for SpecPrefill.

This module is intentionally separate from the legacy :func:`sparse_prefill`
implementation.  The legacy function replaces RoPE children globally and
cannot safely coexist with concurrent requests.  Here each bounded target
forward uses :class:`TargetPositionHooks` and releases its request-local
session only after both logits and cache writes have been evaluated.

``SparseCacheState`` is immutable *post*-prefill metadata.  It records the
full selected logical sequence and the final physical KV occupancy.  The
executor derives a smaller ``TargetPositionPlan`` for every chunk; its physical
start is the committed KV length before that chunk, not its logical position.
"""

from __future__ import annotations

import time
import threading
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

import mlx.core as mx

from .specprefill_cache import SparseCacheState, SparseCacheStateError
from .specprefill_positions import (
    PositionPhase,
    PositionRow,
    TargetPositionAdapter,
    TargetPositionPlan,
)
from .specprefill_target_hooks import TargetPositionHooks


class SparseTargetPrefillError(SparseCacheStateError):
    """Sparse target execution cannot safely begin or finish."""


class SparseTargetPrefillLaneBusy(SparseTargetPrefillError):
    """Another request owns this target model's bounded execution quantum."""


@dataclass(frozen=True)
class SparseTargetPrefillTelemetry:
    """Executor-local facts suitable for request terminal metadata."""

    selected_tokens: int
    target_prefill_ms: float
    chunk_count: int
    physical_cache_starts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SparseTargetPrefillResult:
    """Successful target logits and the sole authoritative sparse cache state."""

    logits: mx.array
    cache_state: SparseCacheState
    telemetry: SparseTargetPrefillTelemetry


class SparseTargetPrefillPhase(str, Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    CLOSED = "closed"


@dataclass(frozen=True)
class SparseTargetPrefillProgress:
    """Non-publishable request-local progress after one target chunk quantum."""

    selected_tokens_processed: int
    chunk_count: int
    complete: bool


@dataclass(frozen=True)
class _CacheSnapshot:
    """A shallow MLX-safe cache checkpoint for all-or-nothing request execution."""

    states: tuple[Any, ...]
    meta_states: tuple[Any | None, ...]
    offsets: tuple[int | None, ...]


_TARGET_LANE_LOCK = threading.RLock()
_TARGET_LANES: dict[int, tuple[weakref.ReferenceType[Any], threading.Lock]] = {}


def _target_lane_for(model: Any) -> threading.Lock:
    """Return the non-reentrant target-forward lane for this exact model object."""
    model_id = id(model)
    with _TARGET_LANE_LOCK:
        entry = _TARGET_LANES.get(model_id)
        if entry is not None:
            model_ref, lane = entry
            if model_ref() is model:
                return lane
            if model_ref() is None:
                _TARGET_LANES.pop(model_id, None)
            else:
                raise SparseTargetPrefillError(
                    "target lane registry identity collision"
                )

        def _remove(model_ref: weakref.ReferenceType[Any]) -> None:
            with _TARGET_LANE_LOCK:
                current = _TARGET_LANES.get(model_id)
                if current is not None and current[0] is model_ref:
                    _TARGET_LANES.pop(model_id, None)

        try:
            model_ref = weakref.ref(model, _remove)
        except TypeError as exc:
            raise SparseTargetPrefillError(
                "target models must support weak references for execution lanes"
            ) from exc
        lane = threading.Lock()
        _TARGET_LANES[model_id] = (model_ref, lane)
        return lane


class SparseTargetPrefillSession:
    """One request's bounded sparse target prefill state machine.

    Static class dispatch remains installed on target RoPE objects.  This
    session never leaves a request position context or a target model lane
    active when ``step`` returns, so a scheduler can interleave compatible
    request caches one chunk at a time without globally serializing the full
    sparse prefill.
    """

    def __init__(
        self,
        model: Any,
        selected_tokens: mx.array | Sequence[int] | Sequence[Sequence[int]],
        cache: Sequence[Any],
        sparse_state: SparseCacheState,
        adapter: TargetPositionAdapter,
        *,
        step_size: int = 2048,
        cancel_check: Callable[[], None] | None = None,
    ):
        _validate_execution_inputs(cache, sparse_state, adapter, step_size)
        self._token_rows = _normalize_selected_tokens(selected_tokens, sparse_state)
        self.model = model
        self.cache = cache
        self.sparse_state = sparse_state
        self.adapter = adapter
        self._step_size = step_size
        self._cancel_check = cancel_check
        self._physical_entries = _physical_cache_entries(cache)
        _require_fresh_physical_cache(self._physical_entries, sparse_state.row_count)
        self._snapshot = _snapshot_cache(cache)
        self._hooks = TargetPositionHooks.for_model(model, adapter)
        self._lane = _target_lane_for(model)
        self._processed = 0
        self._starts: list[tuple[int, ...]] = []
        self._logits: mx.array | None = None
        self._result: SparseTargetPrefillResult | None = None
        self._target_prefill_seconds = 0.0
        self._phase = SparseTargetPrefillPhase.ACTIVE

    @property
    def phase(self) -> SparseTargetPrefillPhase:
        """Read-only lifecycle state; only executor transitions may change it."""
        return self._phase

    @property
    def complete(self) -> bool:
        return self.phase is SparseTargetPrefillPhase.COMPLETE

    @property
    def closed(self) -> bool:
        return self.phase is SparseTargetPrefillPhase.CLOSED

    @property
    def result(self) -> SparseTargetPrefillResult:
        """Return terminal metadata only after every chunk has committed."""
        if self._result is None:
            raise SparseTargetPrefillError(
                "sparse target prefill has no publishable result before completion"
            )
        return self._result

    def step(self) -> SparseTargetPrefillProgress:
        """Run one target chunk quantum, then release hook context and lane."""
        if self.closed:
            raise SparseTargetPrefillError("sparse target prefill session is closed")
        if self.complete:
            return self._progress()
        if not self._lane.acquire(blocking=False):
            raise SparseTargetPrefillLaneBusy(
                "sparse target prefill target lane is busy"
            )
        try:
            self._check_cancelled()
            chunk_size = min(self._step_size, self._total - self._processed)
            physical_start = _physical_cache_lengths(
                self._physical_entries, self.sparse_state.row_count
            )
            expected_start = (self._processed,) * self.sparse_state.row_count
            if physical_start != expected_start:
                raise SparseTargetPrefillError(
                    "target cache physical length disagrees with committed sparse "
                    "target chunk boundary"
                )
            self._starts.append(physical_start)
            plan = _chunk_plan(
                self.adapter,
                self.sparse_state,
                self._processed,
                chunk_size,
                physical_start,
            )
            forward_started_at = time.perf_counter()
            with self._hooks.session_for_plan(plan):
                self._logits = self.model(
                    self._token_rows[:, self._processed : self._processed + chunk_size],
                    cache=self.cache,
                )
                _eval_forward(self._logits, self.cache)
            self._target_prefill_seconds += time.perf_counter() - forward_started_at
            self._processed += chunk_size
            actual_end = _physical_cache_lengths(
                self._physical_entries, self.sparse_state.row_count
            )
            expected_end = (self._processed,) * self.sparse_state.row_count
            if actual_end != expected_end:
                raise SparseTargetPrefillError(
                    "target cache did not advance by the bounded sparse target chunk"
                )
            if self._processed == self._total:
                self._publish_result()
            return self._progress()
        except BaseException:
            self.cancel()
            raise
        finally:
            self._lane.release()

    def run_to_completion(self) -> SparseTargetPrefillResult:
        """Compatibility facade for the former monolithic executor."""
        while not self.complete:
            self.step()
        return self.result

    def cancel(self) -> None:
        """Restore the entry cache snapshot and discard this request's state."""
        if self.closed or self.complete:
            return
        _restore_cache(self.cache, self._snapshot)
        self._logits = None
        self._phase = SparseTargetPrefillPhase.CLOSED

    @property
    def _total(self) -> int:
        return self._token_rows.shape[1]

    def _check_cancelled(self) -> None:
        if self._cancel_check is not None:
            self._cancel_check()

    def _publish_result(self) -> None:
        if self._logits is None:  # pragma: no cover - step always sets logits.
            raise SparseTargetPrefillError("sparse target prefill produced no logits")
        self._result = SparseTargetPrefillResult(
            logits=self._logits,
            cache_state=self.sparse_state.clone(),
            telemetry=SparseTargetPrefillTelemetry(
                selected_tokens=self._total,
                target_prefill_ms=self._target_prefill_seconds * 1000.0,
                chunk_count=len(self._starts),
                physical_cache_starts=tuple(self._starts),
            ),
        )
        self._phase = SparseTargetPrefillPhase.COMPLETE

    def _progress(self) -> SparseTargetPrefillProgress:
        return SparseTargetPrefillProgress(
            selected_tokens_processed=self._processed,
            chunk_count=len(self._starts),
            complete=self.complete,
        )


def execute_sparse_target_prefill(
    model: Any,
    selected_tokens: mx.array | Sequence[int] | Sequence[Sequence[int]],
    cache: Sequence[Any],
    sparse_state: SparseCacheState,
    adapter: TargetPositionAdapter,
    *,
    step_size: int = 2048,
    cancel_check: Callable[[], None] | None = None,
) -> SparseTargetPrefillResult:
    """Run selected target tokens under request-local logical positions.

    The input contains already-selected token IDs in the same row/order as
    ``sparse_state.logical_positions``.  This narrow boundary prevents an
    executor from reinterpreting a selector plan or expanding a rotating-cache
    tail.  Initially only fresh target caches are admissible: sparse prefix
    reuse needs a separately proven atomic payload serializer.

    On cancellation, model failure, lazy-evaluation failure, or topology/cache
    disagreement the entry cache checkpoint is restored before the exception
    is raised.  The immutable ``sparse_state`` is returned only on success.
    """
    return SparseTargetPrefillSession(
        model,
        selected_tokens,
        cache,
        sparse_state,
        adapter,
        step_size=step_size,
        cancel_check=cancel_check,
    ).run_to_completion()


def _chunk_plan(
    adapter: TargetPositionAdapter,
    sparse_state: SparseCacheState,
    start: int,
    count: int,
    physical_starts: tuple[int, ...],
) -> TargetPositionPlan:
    rows = tuple(
        PositionRow(row.logical_positions[start : start + count], physical_start)
        for row, physical_start in zip(sparse_state.rows, physical_starts, strict=True)
    )
    return TargetPositionPlan(adapter, PositionPhase.SPARSE_PREFILL, rows, sparse_state)


def _validate_execution_inputs(
    cache: Sequence[Any],
    sparse_state: SparseCacheState,
    adapter: TargetPositionAdapter,
    step_size: int,
) -> None:
    if not isinstance(sparse_state, SparseCacheState) or not sparse_state.rows:
        raise SparseTargetPrefillError(
            "sparse target prefill needs non-empty cache state"
        )
    if not isinstance(adapter, TargetPositionAdapter):
        raise SparseTargetPrefillError(
            "sparse target prefill needs an explicit adapter"
        )
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        raise SparseTargetPrefillError(
            "sparse target prefill step_size must be positive"
        )
    if not isinstance(cache, Sequence) or not cache:
        raise SparseTargetPrefillError(
            "sparse target prefill needs a non-empty target cache"
        )


def _normalize_selected_tokens(
    selected_tokens: mx.array | Sequence[int] | Sequence[Sequence[int]],
    sparse_state: SparseCacheState,
) -> mx.array:
    tokens = (
        selected_tokens
        if isinstance(selected_tokens, mx.array)
        else mx.array(selected_tokens)
    )
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    if tokens.ndim != 2:
        raise SparseTargetPrefillError(
            "selected target tokens must have shape (batch, selected_tokens)"
        )
    if tokens.shape[1] == 0:
        raise SparseTargetPrefillError(
            "sparse target prefill requires at least one selected token"
        )
    expected = (sparse_state.row_count, sparse_state.rows[0].physical_valid_length)
    if tokens.shape != expected:
        raise SparseTargetPrefillError(
            "selected target token shape must match sparse cache rows and selected length"
        )
    if any(row.physical_valid_length != expected[1] for row in sparse_state.rows):
        raise SparseTargetPrefillError(
            "mixed selected lengths require scheduler lane splitting before sparse target prefill"
        )
    return tokens


def _physical_cache_entries(cache: Sequence[Any]) -> tuple[Any, ...]:
    entries = tuple(entry for entry in cache if hasattr(entry, "offset"))
    if not entries:
        raise SparseTargetPrefillError(
            "target cache exposes no physical offset entries; explicit adapter support is required"
        )
    return entries


def _physical_cache_lengths(entries: Iterable[Any], row_count: int) -> tuple[int, ...]:
    lengths = []
    for entry in entries:
        offset = getattr(entry, "offset", None)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise SparseTargetPrefillError(
                "target cache physical offsets must be non-negative host integers"
            )
        lengths.append(offset)
    if len(set(lengths)) != 1:
        raise SparseTargetPrefillError(
            "target cache offset-bearing entries must share one physical length"
        )
    return (lengths[0],) * row_count


def _require_fresh_physical_cache(entries: Iterable[Any], row_count: int) -> None:
    if _physical_cache_lengths(entries, row_count) != (0,) * row_count:
        raise SparseTargetPrefillError(
            "sparse target prefix reuse is disabled; target cache must start empty"
        )


def _clone_cache_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_cache_value(item) for key, item in value.items()}
    # MLX arrays are immutable values until a cache replaces them; retaining an
    # array avoids a device copy while container cloning prevents aliasing.
    return value


def _snapshot_cache(cache: Sequence[Any]) -> _CacheSnapshot:
    try:
        states = tuple(_clone_cache_value(entry.state) for entry in cache)
        meta_states = tuple(
            (
                _clone_cache_value(entry.meta_state)
                if hasattr(entry, "meta_state")
                else None
            )
            for entry in cache
        )
        offsets = tuple(getattr(entry, "offset", None) for entry in cache)
    except Exception as exc:
        raise SparseTargetPrefillError(
            "target cache cannot produce an atomic sparse target checkpoint"
        ) from exc
    return _CacheSnapshot(states, meta_states, offsets)


def _restore_cache(cache: Sequence[Any], snapshot: _CacheSnapshot) -> None:
    try:
        for entry, state, meta_state, offset in zip(
            cache,
            snapshot.states,
            snapshot.meta_states,
            snapshot.offsets,
            strict=True,
        ):
            entry.state = _clone_cache_value(state)
            if meta_state is not None:
                entry.meta_state = _clone_cache_value(meta_state)
            if offset is not None:
                entry.offset = offset
    except Exception as exc:
        raise SparseTargetPrefillError(
            "target cache rollback failed; request cache must be discarded"
        ) from exc


def _iter_cache_arrays(value: Any) -> Iterable[mx.array]:
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_cache_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_cache_arrays(item)
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        yield value


def _eval_forward(logits: mx.array, cache: Sequence[Any]) -> None:
    """Realize model/cache work while the request-local hook is still active."""
    values = [logits]
    for entry in cache:
        values.extend(_iter_cache_arrays(entry.state))
    mx.eval(values)
