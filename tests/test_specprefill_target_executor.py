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
    SparseTargetPrefillLaneBusy,
    SparseTargetPrefillPhase,
    SparseTargetPrefillSession,
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


def test_session_publishes_no_result_until_all_target_chunks_commit():
    tokens = mx.array([[1, 4, 5]], dtype=mx.int32)
    state = _state((0, 3, 4))
    model, cache = _Model(), [_Cache()]
    session = SparseTargetPrefillSession(
        model, tokens, cache, state, QWEN_DENSE_TARGET, step_size=2
    )
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)

    progress = session.step()

    assert progress.selected_tokens_processed == 2
    assert progress.chunk_count == 1
    assert not progress.complete
    assert cache[0].offset == 2
    assert not hooks.active_session
    with pytest.raises(SparseTargetPrefillError, match="no publishable result"):
        _ = session.result

    progress = session.step()
    assert progress.complete
    assert session.result.cache_state == state
    assert session.result.telemetry.physical_cache_starts == ((0,), (2,))
    assert not hooks.active_session


def test_session_telemetry_excludes_scheduler_wait_between_target_quanta(monkeypatch):
    ticks = iter((100.0, 100.01, 107.0, 107.02))
    monkeypatch.setattr(target_executor.time, "perf_counter", lambda: next(ticks))
    session = SparseTargetPrefillSession(
        _Model(),
        mx.array([[1, 4, 5]], dtype=mx.int32),
        [_Cache()],
        _state((0, 3, 4)),
        QWEN_DENSE_TARGET,
        step_size=2,
    )

    session.step()
    # The mocked clock advances by nearly seven seconds while the scheduler has
    # control between calls.  It must not inflate target-prefill telemetry.
    session.step()

    assert session.result.telemetry.target_prefill_ms == pytest.approx(30.0)


def test_session_phase_is_read_only():
    session = SparseTargetPrefillSession(
        _Model(),
        mx.array([[1]], dtype=mx.int32),
        [_Cache()],
        _state((0,)),
        QWEN_DENSE_TARGET,
    )

    with pytest.raises(AttributeError):
        session.phase = SparseTargetPrefillPhase.COMPLETE
    assert not session.complete


def test_session_rejects_zero_selected_token_rows_before_forward():
    model = _Model()
    with pytest.raises(SparseTargetPrefillError, match="at least one selected token"):
        SparseTargetPrefillSession(
            model,
            mx.array([[]], dtype=mx.int32),
            [_Cache()],
            _state((0,)),
            QWEN_DENSE_TARGET,
        )
    assert model.calls == []


def test_same_target_sessions_alternate_one_chunk_at_a_time_with_distinct_caches():
    model = _Model()
    first = SparseTargetPrefillSession(
        model,
        mx.array([[1, 2, 3]], dtype=mx.int32),
        [_Cache()],
        _state((0, 1, 2)),
        QWEN_DENSE_TARGET,
        step_size=2,
    )
    second = SparseTargetPrefillSession(
        model,
        mx.array([[4, 5, 6]], dtype=mx.int32),
        [_Cache()],
        _state((0, 1, 2)),
        QWEN_DENSE_TARGET,
        step_size=2,
    )
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)

    while not (first.complete and second.complete):
        if not first.complete:
            first.step()
            assert not hooks.active_session
        if not second.complete:
            second.step()
            assert not hooks.active_session

    assert first.result.telemetry.physical_cache_starts == ((0,), (2,))
    assert second.result.telemetry.physical_cache_starts == ((0,), (2,))


def test_busy_target_lane_leaves_waiting_session_retryable():
    model = _Model()
    first = SparseTargetPrefillSession(
        model,
        mx.array([[1, 2, 3]], dtype=mx.int32),
        [_Cache()],
        _state((0, 1, 2)),
        QWEN_DENSE_TARGET,
        step_size=2,
    )
    second = SparseTargetPrefillSession(
        model,
        mx.array([[4, 5, 6]], dtype=mx.int32),
        [_Cache()],
        _state((0, 1, 2)),
        QWEN_DENSE_TARGET,
        step_size=2,
    )

    with first._lane:
        with pytest.raises(SparseTargetPrefillLaneBusy, match="target lane is busy"):
            second.step()
        assert not second.closed
        assert not second.complete

    second.run_to_completion()
    assert second.complete


def test_cancel_after_first_chunk_restores_entry_snapshot_and_publishes_nothing():
    tokens = mx.array([[1, 4, 5]], dtype=mx.int32)
    state = _state((0, 3, 4))
    model, cache = _Model(), [_Cache()]
    calls = 0

    def cancel_check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("cancelled")

    session = SparseTargetPrefillSession(
        model,
        tokens,
        cache,
        state,
        QWEN_DENSE_TARGET,
        step_size=2,
        cancel_check=cancel_check,
    )
    session.step()
    assert cache[0].offset == 2
    with pytest.raises(RuntimeError, match="cancelled"):
        session.step()

    assert session.closed
    assert cache[0].offset == 0
    assert cache[0].state.tolist() == [0]
    with pytest.raises(SparseTargetPrefillError, match="no publishable result"):
        _ = session.result


def test_facade_matches_stepwise_session_result():
    tokens = mx.array([[1, 4, 5]], dtype=mx.int32)
    state = _state((0, 3, 4))
    facade_model, facade_cache = _Model(), [_Cache()]
    facade = execute_sparse_target_prefill(
        facade_model, tokens, facade_cache, state, QWEN_DENSE_TARGET, step_size=2
    )
    session_model, session_cache = _Model(), [_Cache()]
    session = SparseTargetPrefillSession(
        session_model, tokens, session_cache, state, QWEN_DENSE_TARGET, step_size=2
    )
    stepwise = session.run_to_completion()

    _assert_close(facade.logits, stepwise.logits)
    assert facade.cache_state == stepwise.cache_state == state
    assert (
        facade.telemetry.physical_cache_starts
        == stepwise.telemetry.physical_cache_starts
    )
    assert facade_cache[0].offset == session_cache[0].offset == 3


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
