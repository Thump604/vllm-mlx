# SPDX-License-Identifier: Apache-2.0
"""Synthetic request-local generation-forward position tests."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.specprefill_cache import (
    SparseCacheIdentity,
    SparseCacheState,
    SparsePolicyTuning,
)
from vllm_mlx.specprefill_generation_context import (
    SparseGenerationContextError,
    SparseGenerationForwardContext,
)
from vllm_mlx.specprefill_positions import (
    PositionPhase,
    QWEN_DENSE_TARGET,
)


class _Cache:
    def __init__(self, offset):
        self.offset = offset


class _Hooks:
    def __init__(self):
        self.plans = []
        self.active = False

    @contextmanager
    def session_for_plan(self, plan):
        assert not self.active
        self.active = True
        self.plans.append(plan)
        try:
            yield
        finally:
            self.active = False


def _state():
    tuning = SparsePolicyTuning(0.5, 0.0, 0, 1, 2)
    identity = SparseCacheIdentity.from_tokens(
        target_id="target",
        tokenizer_id="tokenizer",
        scorer_id="scorer",
        selector_version="selector-v1",
        tuning=tuning,
        tokens=range(8),
        selection_fingerprint="1" * 64,
    )
    return SparseCacheState.from_selection(identity, ((0, 3, 6, 7),), (8,))


def _context(monkeypatch):
    model = object()
    cache = [_Cache(4), _Cache(4)]
    hooks = _Hooks()
    monkeypatch.setattr(
        "vllm_mlx.specprefill_generation_context.TargetPositionHooks.for_model",
        lambda actual_model, adapter: hooks,
    )
    context = SparseGenerationForwardContext(model, cache, _state(), QWEN_DENSE_TARGET)
    return context, model, cache, hooks


def _forward(model, cache, count, phase):
    return SimpleNamespace(
        model=model,
        cache=cache,
        input_tokens=mx.zeros((1, count), dtype=mx.int32),
        phase=SimpleNamespace(value=phase),
    )


def _advance(cache, count):
    for entry in cache:
        entry.offset += count


def test_decode_uses_logical_cursor_without_overwriting_physical_offset(monkeypatch):
    context, model, cache, hooks = _context(monkeypatch)

    with context(_forward(model, cache, 1, "decode")):
        assert hooks.active
        _advance(cache, 1)

    plan = hooks.plans[-1]
    assert plan.phase is PositionPhase.DECODE
    assert plan.logical_positions == ((8,),)
    assert plan.physical_cache_lengths == (4,)
    assert context.state.next_logical_positions == (9,)
    assert context.state.physical_valid_lengths == (5,)


def test_verify_rollback_reconciles_before_the_next_target_forward(monkeypatch):
    context, model, cache, hooks = _context(monkeypatch)

    with context(_forward(model, cache, 3, "verify")):
        _advance(cache, 3)
    # The verifier keeps one of the three forwarded tokens.
    for entry in cache:
        entry.offset -= 2

    with context(_forward(model, cache, 1, "decode")):
        plan = hooks.plans[-1]
        assert plan.logical_positions == ((9,),)
        assert plan.physical_cache_lengths == (5,)
        _advance(cache, 1)

    assert context.finish().next_logical_positions == (10,)
    assert context.state.physical_valid_lengths == (6,)


def test_finish_reconciles_a_final_speculative_rejection(monkeypatch):
    context, model, cache, _ = _context(monkeypatch)

    with context(_forward(model, cache, 4, "verify")):
        _advance(cache, 4)
    for entry in cache:
        entry.offset -= 3

    final_state = context.finish()
    assert final_state.physical_valid_lengths == (5,)
    assert final_state.next_logical_positions == (9,)


def test_foreign_draft_forward_delegates_without_target_state_mutation(monkeypatch):
    context, _, _, hooks = _context(monkeypatch)
    draft_cache = [_Cache(8)]

    with context(_forward(object(), draft_cache, 2, "draft")):
        _advance(draft_cache, 2)

    assert hooks.plans == []
    assert context.state == _state()


def test_target_forward_rejects_cache_or_phase_drift(monkeypatch):
    context, model, cache, _ = _context(monkeypatch)
    with pytest.raises(SparseGenerationContextError, match="replaced"):
        with context(_forward(model, list(cache), 1, "decode")):
            pass
    with pytest.raises(SparseGenerationContextError, match="decode or verify"):
        with context(_forward(model, cache, 1, "prefill")):
            pass
    with pytest.raises(SparseGenerationContextError, match="exactly one"):
        with context(_forward(model, cache, 2, "decode")):
            pass


def test_forward_error_does_not_publish_a_logical_state_transition(monkeypatch):
    context, model, cache, hooks = _context(monkeypatch)

    with pytest.raises(RuntimeError, match="forward failed"):
        with context(_forward(model, cache, 1, "decode")):
            assert hooks.active
            raise RuntimeError("forward failed")

    assert not hooks.active
    assert context.state == _state()
    assert cache[0].offset == cache[1].offset == 4


def test_cache_reconciliation_fails_closed_at_sparse_prompt_boundary(monkeypatch):
    context, _, cache, _ = _context(monkeypatch)
    cache[0].offset = cache[1].offset = 3

    with pytest.raises(SparseGenerationContextError, match="prompt boundary"):
        context.finish()
