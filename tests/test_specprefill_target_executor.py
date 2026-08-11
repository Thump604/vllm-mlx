# SPDX-License-Identifier: Apache-2.0
"""Synthetic execution oracles for request-local sparse target prefill."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import vllm_mlx.specprefill_target_executor as target_executor
from vllm_mlx.specprefill_cache import (
    SparseCacheIdentity,
    SparseCacheState,
    SparsePolicyTuning,
)
from vllm_mlx.specprefill_positions import QWEN_DENSE_TARGET
from vllm_mlx.specprefill_target_executor import (
    SparseTargetPrefillError,
    execute_sparse_target_prefill,
)
from vllm_mlx.specprefill_target_hooks import TargetPositionHooks


class _Cache:
    def __init__(self):
        self.offset = 0
        self.state = mx.array([0], dtype=mx.int32)
        self.meta_state = ()


class _Attention:
    def __init__(self, rope):
        self.rope = rope


class _Model:
    def __init__(self, *, fail_on_call: int | None = None):
        self.layers = [SimpleNamespace(self_attn=_Attention(nn.RoPE(8, base=10000.0)))]
        self.calls = []
        self.rotated = []
        self.fail_on_call = fail_on_call

    @property
    def rope(self):
        return self.layers[0].self_attn.rope

    def __call__(self, token_rows, *, cache):
        self.calls.append((cache[0].offset, token_rows.shape[1]))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("target model failed")
        hidden = mx.broadcast_to(
            token_rows.astype(mx.float32)[:, None, :, None],
            (token_rows.shape[0], 1, token_rows.shape[1], 8),
        )
        rotated = self.rope(hidden, offset=cache[0].offset)
        self.rotated.append(rotated)
        cache[0].offset += token_rows.shape[1]
        cache[0].state = mx.array([cache[0].offset], dtype=mx.int32)
        return rotated[:, :, -1:, :]


def _identity() -> SparseCacheIdentity:
    return SparseCacheIdentity.from_tokens(
        target_id="target@sha256:target",
        tokenizer_id="tokenizer@sha256:tokenizer",
        scorer_id="scorer@sha256:scorer",
        selector_version="hybrid-v1",
        tuning=SparsePolicyTuning(0.5, 0.1, 1, 1, 2),
        tokens=(1, 2, 3, 4, 5),
        selection_fingerprint="a" * 64,
    )


def _state(positions: tuple[int, ...]) -> SparseCacheState:
    return SparseCacheState.from_selection(_identity(), (positions,), (5,))


def _assert_close(actual, expected):
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()


def _dense_prefill(model, token_rows, cache, *, step_size):
    logits = None
    for start in range(0, token_rows.shape[1], step_size):
        logits = model(token_rows[:, start : start + step_size], cache=cache)
        mx.eval(logits, cache[0].state)
    return logits


def test_keep_ratio_one_multichunk_matches_dense_logits_and_cache_state(monkeypatch):
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    state = _state((0, 1, 2, 3, 4))
    dense_model, dense_cache = _Model(), [_Cache()]
    dense_logits = _dense_prefill(dense_model, tokens, dense_cache, step_size=2)

    sparse_model, sparse_cache = _Model(), [_Cache()]
    hooks = TargetPositionHooks.for_model(sparse_model, QWEN_DENSE_TARGET)
    observed_sessions = []
    real_eval = target_executor._eval_forward

    def record_eval(logits, cache):
        observed_sessions.append(hooks.active_session)
        real_eval(logits, cache)

    monkeypatch.setattr(target_executor, "_eval_forward", record_eval)
    result = execute_sparse_target_prefill(
        sparse_model, tokens, sparse_cache, state, QWEN_DENSE_TARGET, step_size=2
    )

    _assert_close(result.logits, dense_logits)
    assert result.cache_state == state
    assert result.cache_state is not state
    assert sparse_cache[0].offset == dense_cache[0].offset == 5
    assert sparse_model.calls == dense_model.calls == [(0, 2), (2, 2), (4, 1)]
    assert result.telemetry.physical_cache_starts == ((0,), (2,), (4,))
    assert all(session is not None for session in observed_sessions)
    assert hooks.active_session is None


def test_noncontiguous_logical_positions_stay_separate_from_physical_cache():
    tokens = mx.array([[1, 4, 5]], dtype=mx.int32)
    state = _state((0, 3, 4))
    model, cache = _Model(), [_Cache()]
    result = execute_sparse_target_prefill(
        model, tokens, cache, state, QWEN_DENSE_TARGET, step_size=2
    )

    reference = _Model()
    hidden = mx.broadcast_to(tokens.astype(mx.float32)[:, None, :, None], (1, 1, 3, 8))
    expected_first = mx.concatenate(
        [
            reference.rope(hidden[:, :, index : index + 1], offset=position)
            for index, position in enumerate((0, 3))
        ],
        axis=2,
    )
    expected_last = reference.rope(hidden[:, :, 2:], offset=4)

    _assert_close(model.rotated[0], expected_first)
    _assert_close(model.rotated[1], expected_last)
    assert model.calls == [(0, 2), (2, 1)]
    assert result.telemetry.physical_cache_starts == ((0,), (2,))
    assert result.cache_state.logical_positions == ((0, 3, 4),)
    assert result.cache_state.physical_valid_lengths == (3,)
    assert cache[0].offset == 3


@pytest.mark.parametrize("mode", ("cancel", "failure"))
def test_cancel_or_model_failure_restores_entry_cache_atomically(mode):
    tokens = mx.array([[1, 4, 5]], dtype=mx.int32)
    state = _state((0, 3, 4))
    model = _Model(fail_on_call=2 if mode == "failure" else None)
    cache = [_Cache()]
    cancel_calls = 0

    def cancel_check():
        nonlocal cancel_calls
        cancel_calls += 1
        if mode == "cancel" and cancel_calls == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled|target model failed"):
        execute_sparse_target_prefill(
            model,
            tokens,
            cache,
            state,
            QWEN_DENSE_TARGET,
            step_size=2,
            cancel_check=cancel_check,
        )

    assert cache[0].offset == 0
    assert cache[0].state.tolist() == [0]
    assert state.physical_valid_lengths == (3,)
    assert state.logical_positions == ((0, 3, 4),)


def test_prefill_rejects_partial_sparse_prefix_reuse_before_target_forward():
    model, cache = _Model(), [_Cache()]
    cache[0].offset = 1
    cache[0].state = mx.array([1], dtype=mx.int32)
    with pytest.raises(SparseTargetPrefillError, match="prefix reuse is disabled"):
        execute_sparse_target_prefill(
            model,
            mx.array([[1, 4, 5]], dtype=mx.int32),
            cache,
            _state((0, 3, 4)),
            QWEN_DENSE_TARGET,
        )
    assert model.calls == []
