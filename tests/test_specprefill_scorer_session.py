# SPDX-License-Identifier: Apache-2.0
"""Bounded request-local scorer-session contracts without model weights."""

from __future__ import annotations

import threading
import math
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

import vllm_mlx.specprefill_scorer_session as scorer_sessions
import vllm_mlx.specprefill as specprefill
from vllm_mlx.specprefill import (
    SpecPrefillScorer,
    _compute_importance,
    _lookahead_decode,
    _prefill_draft,
)
from vllm_mlx.specprefill_scorer_session import (
    ScorerSessionPhase,
    SpecPrefillScorerSession,
)


class _Cache:
    def __init__(self):
        self.offset = 0
        self.keys = mx.zeros((1, 1, 0, 1), dtype=mx.float32)

    @property
    def state(self):
        return self.keys

    @state.setter
    def state(self, value):
        self.keys = value
        self.offset = value.shape[-2]


class _Attention:
    n_heads = 1
    n_kv_heads = 1

    def __init__(self):
        self.calls = 0

    def __call__(self, x, mask=None, cache=None, **_kwargs):
        del mask
        self.calls += 1
        return x


class _Model:
    def __init__(self, *, block_first_call=False, fail_on_call=None):
        self.config = SimpleNamespace(model_type="qwen3")
        self.layers = [SimpleNamespace(self_attn=_Attention())]
        self.block_first_call = block_first_call
        self.fail_on_call = fail_on_call
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.token_inputs = []

    @property
    def attention(self):
        return self.layers[0].self_attn

    def __call__(self, token_rows, *, cache):
        self.calls += 1
        self.token_inputs.append(
            (getattr(cache[0], "label", None), token_rows.tolist())
        )
        if self.block_first_call and self.calls == 1:
            self.entered.set()
            assert self.release.wait(timeout=3)
        if self.fail_on_call == self.calls:
            raise RuntimeError("draft failure")
        length = token_rows.shape[1]
        hidden = mx.ones((1, length, 1), dtype=mx.float32)
        self.attention(hidden, cache=cache[0])
        cache[0].keys = mx.concatenate(
            (cache[0].keys, mx.ones((1, 1, length, 1), dtype=mx.float32)), axis=2
        )
        cache[0].offset += length
        return mx.zeros((1, length, 2), dtype=mx.float32)


def _extract_queries(_attention, x, cache=None, **_kwargs):
    del cache
    return mx.ones((1, 1, x.shape[1], 1), dtype=mx.float32)


def _session(monkeypatch, model, tokens=(1, 2, 3, 4), **kwargs):
    monkeypatch.setattr(scorer_sessions, "make_prompt_cache", lambda _model: [_Cache()])
    options = {
        "n_lookahead": 2,
        "prefill_step_size": 2,
        "query_extractor": _extract_queries,
    }
    options.update(kwargs)
    return SpecPrefillScorerSession(
        SpecPrefillScorer.for_model(model),
        tokens,
        **options,
    )


def _legacy_reference(monkeypatch, model, tokens):
    monkeypatch.setattr(
        specprefill,
        "make_sampler",
        lambda **_kwargs: lambda logits: mx.argmax(logits, axis=-1),
    )
    scorer = SpecPrefillScorer.for_model(model)
    cache = [_Cache()]
    with scorer.capture_session(_extract_queries, capture_enabled=False) as capture:
        logits = _prefill_draft(model, tokens, cache, step_size=2)
        capture.capture_enabled = True
        _lookahead_decode(model, logits, cache, 2, temp=0.6, top_p=0.95)
        importance = _compute_importance(capture.query_buffer, cache, len(tokens))
        mx.eval(importance, capture.query_buffer)
    return importance


def _assert_close(actual, expected):
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()


def test_alternating_request_sessions_release_capture_between_quanta(monkeypatch):
    model = _Model()
    first = _session(monkeypatch, model, tokens=(1, 2, 3, 4))
    second = _session(monkeypatch, model, tokens=(5, 6, 7, 8))
    scorer = first.scorer

    while not (first.complete and second.complete):
        if not first.complete:
            first.step()
            assert not scorer.capture_active
        if not second.complete:
            second.step()
            assert not scorer.capture_active

    assert first.phase is ScorerSessionPhase.COMPLETE
    assert second.phase is ScorerSessionPhase.COMPLETE
    assert first.query_buffer is not second.query_buffer
    # The default centered pool has zero-padding at prompt edges, so an
    # otherwise uniform four-token distribution becomes 1 / kernel_size.
    _assert_close(first.importance, mx.full((4,), 1 / 13))
    _assert_close(second.importance, mx.full((4,), 1 / 13))


def test_bounded_session_matches_clean_monolithic_helper_reference(monkeypatch):
    tokens = (1, 2, 3, 4)
    session_model = _Model()
    session = _session(monkeypatch, session_model, tokens=tokens, temp=0)
    actual = session.run_to_completion()

    reference_model = _Model()
    expected = _legacy_reference(monkeypatch, reference_model, tokens)

    _assert_close(actual, expected)
    assert session_model.calls == reference_model.calls == 5
    assert not session.scorer.capture_active


def test_request_local_rng_is_invariant_when_another_session_interleaves(
    monkeypatch,
):
    baseline_model = _Model()
    baseline = _session(
        monkeypatch, baseline_model, tokens=(1, 2, 3, 4), sampling_seed=17
    )
    baseline.cache[0].label = "baseline"
    baseline.run_to_completion()
    expected_inputs = [
        tokens for label, tokens in baseline_model.token_inputs if label == "baseline"
    ]

    model = _Model()
    first = _session(monkeypatch, model, tokens=(1, 2, 3, 4), sampling_seed=17)
    second = _session(monkeypatch, model, tokens=(5, 6, 7, 8), sampling_seed=91)
    first.cache[0].label = "first"
    second.cache[0].label = "second"
    while not (first.complete and second.complete):
        if not first.complete:
            first.step()
        if not second.complete:
            second.step()

    first_inputs = [tokens for label, tokens in model.token_inputs if label == "first"]
    assert first_inputs == expected_inputs


def test_top_p_filters_normalized_logprobabilities_not_raw_logits():
    logits = mx.array([[2.0, 1.0, 0.0]])
    normalized = scorer_sessions._normalized_logprobs(logits)
    filtered = scorer_sessions.apply_top_p(normalized, 0.8)
    raw_filtered = scorer_sessions.apply_top_p(logits, 0.8)
    mx.eval(filtered, raw_filtered)

    # At p=0.8 the least likely token is excluded only after normalizing:
    # raw logits exponentiate to values greater than one and incorrectly keep
    # every token in mlx-lm's cumulative-probability implementation.
    assert math.isinf(float(filtered[0, 2])) and float(filtered[0, 2]) < 0
    assert not math.isinf(float(raw_filtered[0, 2]))


@pytest.mark.parametrize("temp", (float("nan"), float("inf"), -0.1, True))
def test_sampler_rejects_nonfinite_or_negative_temperature(temp):
    with pytest.raises(ValueError, match="finite non-negative"):
        scorer_sessions._RequestLocalSampler(temp=temp, top_p=0.9, seed=1)


@pytest.mark.parametrize("top_p", (float("nan"), float("inf"), -0.1, 1.1, True))
def test_sampler_rejects_invalid_top_p(top_p):
    with pytest.raises(ValueError, match="top_p"):
        scorer_sessions._RequestLocalSampler(temp=0.6, top_p=top_p, seed=1)


def test_session_rejects_zero_lookahead_before_allocating_a_draft_cache(monkeypatch):
    model = _Model()
    cache_calls = 0

    def make_cache(_model):
        nonlocal cache_calls
        cache_calls += 1
        return [_Cache()]

    monkeypatch.setattr(scorer_sessions, "make_prompt_cache", make_cache)
    with pytest.raises(ValueError, match="positive integer"):
        SpecPrefillScorerSession(
            SpecPrefillScorer.for_model(model),
            (1, 2),
            n_lookahead=0,
            query_extractor=_extract_queries,
        )
    assert cache_calls == 0


def test_session_materializes_prompt_token_array_once(monkeypatch):
    model = _Model()
    real_array = scorer_sessions.mx.array
    calls = 0

    def recording_array(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_array(*args, **kwargs)

    monkeypatch.setattr(scorer_sessions.mx, "array", recording_array)
    session = _session(monkeypatch, model)
    assert calls == 1
    session.run_to_completion()
    assert calls == 1


def test_scorer_lane_rejects_overlap_and_leaves_second_session_retryable(monkeypatch):
    model = _Model()
    first = _session(monkeypatch, model)
    second = _session(monkeypatch, model)
    with first.scorer.scoring_quantum():
        with pytest.raises(RuntimeError, match="scorer lane is busy"):
            second.step()
        assert second.phase is ScorerSessionPhase.PREFILL
        assert not second.closed

    second.run_to_completion()
    assert second.complete
    assert not first.scorer.capture_active


@pytest.mark.parametrize("mode", ("cancel", "failure"))
def test_cancel_or_failure_discards_only_own_session_state(monkeypatch, mode):
    model = _Model(fail_on_call=1 if mode == "failure" else None)
    cancel_calls = 0

    def cancel_check():
        nonlocal cancel_calls
        cancel_calls += 1
        if mode == "cancel":
            raise RuntimeError("cancelled")

    failed = _session(monkeypatch, model, cancel_check=cancel_check)
    with pytest.raises(RuntimeError, match="cancelled|draft failure"):
        failed.step()
    assert failed.closed
    assert failed.cache is None
    assert failed.query_buffer == []
    assert not failed.scorer.capture_active

    healthy = _session(monkeypatch, model, tokens=(6, 7, 8, 9))
    healthy.run_to_completion()
    assert healthy.complete
