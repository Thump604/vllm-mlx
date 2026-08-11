# SPDX-License-Identifier: Apache-2.0
"""Request-local sparse-cache positions for mlx-lm generation forwards.

Sparse target prefill writes selected prompt tokens into contiguous cache
slots while preserving their original logical positions.  mlx-lm owns the
subsequent decode and speculative-verification forwards, so this adapter turns
its ``model_forward_context`` metadata into one bounded
:class:`TargetPositionHooks` session per target call.

The state transition is optimistic until the next target call (or
``finish``): speculative generation may trim rejected target tokens after the
forward context exits.  Reconciliation uses the target cache's host offsets to
roll back only the contiguous generated suffix.  Sparse prompt entries can
never be trimmed or reinterpreted here.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .specprefill_cache import SparseCacheState, SparseCacheStateError
from .specprefill_positions import (
    TargetPositionAdapter,
    decode_plan,
    verify_plan,
)
from .specprefill_target_hooks import TargetPositionHooks


class SparseGenerationContextError(SparseCacheStateError):
    """A generation forward cannot preserve sparse logical positions."""


class SparseGenerationForwardContext:
    """Own one SimpleEngine request's sparse target decode state.

    Foreign draft-model forwards are delegated unchanged.  Target forwards
    must use the exact cache object produced by sparse target prefill and must
    be labelled ``decode`` or ``verify`` by the generation implementation.
    """

    def __init__(
        self,
        target_model: Any,
        target_cache: Sequence[Any],
        state: SparseCacheState,
        adapter: TargetPositionAdapter,
    ) -> None:
        if state.row_count != 1:
            raise SparseGenerationContextError(
                "SimpleEngine sparse generation requires exactly one cache row"
            )
        if not isinstance(adapter, TargetPositionAdapter):
            raise TypeError("adapter must be a TargetPositionAdapter")
        if not isinstance(target_cache, Sequence) or not target_cache:
            raise SparseGenerationContextError(
                "sparse generation requires a non-empty target cache"
            )
        self.target_model = target_model
        self.target_cache = target_cache
        self.adapter = adapter
        self._state = state.clone()
        self._hooks = TargetPositionHooks.for_model(target_model, adapter)
        self._reconcile_cache()

    @property
    def state(self) -> SparseCacheState:
        """Return an immutable snapshot of the currently retained target KV."""
        return self._state.clone()

    def __call__(self, forward: Any):
        """Return the request-local context for one mlx-lm forward."""
        return self._forward_context(forward)

    @contextmanager
    def _forward_context(self, forward: Any) -> Iterator[None]:
        model = getattr(forward, "model", None)
        cache = getattr(forward, "cache", None)
        if model is not self.target_model:
            if cache is self.target_cache:
                raise SparseGenerationContextError(
                    "a foreign draft model cannot share the sparse target cache"
                )
            yield
            return
        if cache is not self.target_cache:
            raise SparseGenerationContextError(
                "target generation forward replaced the admitted sparse cache"
            )

        self._reconcile_cache()
        tokens = getattr(forward, "input_tokens", None)
        shape = getattr(tokens, "shape", None)
        if shape is None or len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
            raise SparseGenerationContextError(
                "target generation tokens must have shape (1, positive_tokens)"
            )
        token_count = int(shape[1])
        phase = getattr(getattr(forward, "phase", None), "value", None)
        if phase is None:
            phase = getattr(forward, "phase", None)
        if phase == "decode":
            if token_count != 1:
                raise SparseGenerationContextError(
                    "decode forwards must contain exactly one target token"
                )
            plan = decode_plan(self.adapter, self._state)
        elif phase == "verify":
            plan = verify_plan(self.adapter, self._state, token_count)
        else:
            raise SparseGenerationContextError(
                "sparse target continuation accepts only decode or verify forwards"
            )

        expected_end = self._state.rows[0].physical_valid_length + token_count
        with self._hooks.session_for_plan(plan):
            yield
        actual_end = _cache_offset(self.target_cache)
        if actual_end != expected_end:
            raise SparseGenerationContextError(
                "target cache did not advance by the generation forward token count"
            )
        self._state = self._state.append_decode(token_count)

    def finish(self) -> SparseCacheState:
        """Reconcile final speculative rollback and return terminal state."""
        self._reconcile_cache()
        return self.state

    def _reconcile_cache(self) -> None:
        actual = _cache_offset(self.target_cache)
        row = self._state.rows[0]
        expected = row.physical_valid_length
        if actual > expected:
            raise SparseGenerationContextError(
                "target cache advanced outside the request-local forward context"
            )
        if actual < row.prefill_physical_length:
            raise SparseGenerationContextError(
                "target cache rollback crossed the immutable sparse prompt boundary"
            )
        if actual < expected:
            self._state = self._state.rollback(expected - actual)


def _cache_offset(cache: Sequence[Any]) -> int:
    offsets: list[int] = []
    for entry in cache:
        if not hasattr(entry, "offset"):
            continue
        offset = entry.offset
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise SparseGenerationContextError(
                "target cache offsets must be non-negative host integers"
            )
        offsets.append(offset)
    if not offsets:
        raise SparseGenerationContextError(
            "target cache exposes no host physical offsets"
        )
    if len(set(offsets)) != 1:
        raise SparseGenerationContextError(
            "target cache layers disagree on physical occupancy"
        )
    return offsets[0]
