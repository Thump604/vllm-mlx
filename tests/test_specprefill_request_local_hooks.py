# SPDX-License-Identifier: Apache-2.0
"""Request-local scorer capture contracts that do not load model weights."""

import threading
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

import vllm_mlx.specprefill as specprefill
from vllm_mlx.specprefill import (
    SpecPrefillScorer,
    _AttentionCapture,
    _gemma4_extract_queries,
)


class RecordingAttention:
    n_heads = 1
    n_kv_heads = 1

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0]


class ScorerModel:
    def __init__(self, model_type="qwen3"):
        self.config = SimpleNamespace(model_type=model_type)
        self.layers = [SimpleNamespace(self_attn=RecordingAttention())]


def passthrough_extractor(_attention, x, cache=None, **_kwargs):
    return x


def test_scorer_installs_one_stable_wrapper_and_reuses_it():
    model = ScorerModel()
    original = model.layers[0].self_attn

    first = SpecPrefillScorer.for_model(model)
    wrapper = model.layers[0].self_attn
    second = SpecPrefillScorer.for_model(model)

    assert first is second
    assert isinstance(wrapper, _AttentionCapture)
    assert model.layers[0].self_attn is wrapper
    assert wrapper._original is original
    with pytest.raises(RuntimeError, match="already installed"):
        SpecPrefillScorer(model)
    assert model.layers[0].self_attn is wrapper


def test_idle_wrapper_delegates_original_args_and_kwargs_unchanged():
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    wrapper = model.layers[0].self_attn
    original = wrapper._original
    x = mx.zeros((1, 1, 1))
    mask = object()
    cache = object()
    shared_kv = object()

    result = wrapper(x, mask, cache, shared_kv=shared_kv, offset=23)

    assert result is x
    assert len(original.calls) == 1
    delegated_args, delegated_kwargs = original.calls[0]
    assert delegated_args[0] is x
    assert delegated_args[1] is mask
    assert delegated_args[2] is cache
    assert delegated_kwargs == {"shared_kv": shared_kv, "offset": 23}
    assert not scorer.capture_active


def test_request_capture_preserves_call_shape_offset_and_shared_kv():
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    wrapper = model.layers[0].self_attn
    original = wrapper._original
    x = mx.zeros((1, 1, 1))
    mask = object()
    cache = SimpleNamespace(offset=4)
    offset = mx.array(17)
    shared_kv = (mx.array([1]), mx.array([2]))
    extracted = []

    def extractor(_attention, captured_x, cache=None, **kwargs):
        extracted.append((captured_x, cache, kwargs))
        return captured_x

    with scorer.capture_session(extractor) as session:
        wrapper(x, mask, cache, offset=offset, shared_kv=shared_kv)
        assert session.query_buffer[0] == [x]

    delegated_args, delegated_kwargs = original.calls[0]
    assert delegated_args[0] is x
    assert delegated_args[1] is mask
    assert delegated_args[2] is cache
    assert delegated_kwargs["offset"] is offset
    assert delegated_kwargs["shared_kv"] is shared_kv
    assert extracted[0][0] is x
    assert extracted[0][1] is cache
    assert extracted[0][2]["offset"] is offset
    assert extracted[0][2]["shared_kv"] is shared_kv

    wrapper(x, cache=cache)
    assert session.query_buffer[0] == [x]
    assert model.layers[0].self_attn is wrapper
    assert not scorer.capture_active


def test_gemma_capture_uses_explicit_offset_and_preserves_shared_kv():
    class Norm:
        def __call__(self, value):
            return value

    class Rope:
        def __init__(self):
            self.offsets = []

        def __call__(self, value, offset=0):
            self.offsets.append(offset)
            return value

    class GemmaAttention(RecordingAttention):
        n_heads = 1
        q_norm = Norm()

        def __init__(self):
            super().__init__()
            self.rope = Rope()

        def q_proj(self, value):
            return value

    model = ScorerModel(model_type="gemma4")
    attention = GemmaAttention()
    model.layers[0].self_attn = attention
    model.previous_kvs = [0]
    scorer = SpecPrefillScorer.for_model(model)
    x = mx.ones((1, 1, 1))
    cache = SimpleNamespace(offset=3)
    shared_kv = (mx.array([1]), mx.array([2]))
    offset = mx.array(19)

    with scorer.capture_session(_gemma4_extract_queries) as session:
        model.layers[0].self_attn(x, cache=cache, shared_kv=shared_kv, offset=offset)
        mx.eval(session.query_buffer)

    assert len(attention.rope.offsets) == 1
    assert attention.rope.offsets[0] is offset
    assert attention.calls[0][1]["shared_kv"] is shared_kv
    assert attention.calls[0][1]["offset"] is offset


def test_nested_and_foreign_thread_sessions_fail_closed():
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    foreign_errors = []
    x = mx.zeros((1, 1, 1))

    with scorer.capture_session(passthrough_extractor):
        with pytest.raises(RuntimeError, match="already has an active session"):
            with scorer.capture_session(passthrough_extractor):
                pass

        def invoke_from_foreign_thread():
            try:
                model.layers[0].self_attn(x)
            except Exception as exc:  # noqa: BLE001 - asserted below
                foreign_errors.append(exc)

            try:
                with scorer.capture_session(passthrough_extractor):
                    pass
            except Exception as exc:  # noqa: BLE001 - asserted below
                foreign_errors.append(exc)

        thread = threading.Thread(target=invoke_from_foreign_thread)
        thread.start()
        thread.join()

    assert len(foreign_errors) == 2
    assert "outside its active request session" in str(foreign_errors[0])
    assert "already has an active session" in str(foreign_errors[1])
    assert not scorer.capture_active


def test_session_exception_clears_state_without_uninstalling_wrapper():
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    wrapper = model.layers[0].self_attn

    with pytest.raises(ValueError, match="cancelled"):
        with scorer.capture_session(passthrough_extractor):
            raise ValueError("cancelled")

    assert not scorer.capture_active
    assert model.layers[0].self_attn is wrapper
    with scorer.capture_session(passthrough_extractor):
        assert scorer.capture_active


def test_public_score_tokens_realizes_lazy_work_before_session_release(monkeypatch):
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    wrapper = model.layers[0].self_attn
    cache = [SimpleNamespace(keys=mx.ones((1, 1, 2, 1)))]
    eval_session_states = []
    real_eval = specprefill.mx.eval

    monkeypatch.setattr(specprefill, "make_prompt_cache", lambda _model: cache)

    def prefill(*_args, **_kwargs):
        assert scorer.capture_active
        wrapper(mx.ones((1, 1, 1)), cache=cache[0])
        return mx.zeros((1, 1, 4))

    monkeypatch.setattr(specprefill, "_prefill_draft", prefill)

    def lookahead(_model, _logits, _cache, _steps, **_kwargs):
        wrapper(mx.ones((1, 1, 1)), cache=cache[0], offset=2)
        return [1]

    def compute_importance(query_buffer, *_args, **_kwargs):
        assert scorer.capture_active
        assert len(query_buffer[0]) == 1
        return mx.array([0.25, 0.75])

    def recording_eval(*values):
        eval_session_states.append(scorer.capture_active)
        return real_eval(*values)

    monkeypatch.setattr(specprefill, "_lookahead_decode", lookahead)
    monkeypatch.setattr(specprefill, "_compute_importance", compute_importance)
    monkeypatch.setattr(specprefill.mx, "eval", recording_eval)

    importance = specprefill.score_tokens(
        model,
        [1, 2],
        query_extractor=passthrough_extractor,
    )

    assert importance.tolist() == [0.25, 0.75]
    assert eval_session_states == [True]
    assert not scorer.capture_active
    assert model.layers[0].self_attn is wrapper


def test_score_tokens_rejects_overlap_before_second_prefill(monkeypatch):
    model = ScorerModel()
    SpecPrefillScorer.for_model(model)
    entered_prefill = threading.Event()
    release_prefill = threading.Event()
    first_errors = []
    prefill_calls = 0

    monkeypatch.setattr(specprefill, "make_prompt_cache", lambda _model: [])

    def blocking_prefill(*_args, **_kwargs):
        nonlocal prefill_calls
        prefill_calls += 1
        entered_prefill.set()
        release_prefill.wait(timeout=5)
        raise ValueError("stop first scorer")

    monkeypatch.setattr(specprefill, "_prefill_draft", blocking_prefill)

    def run_first_score():
        try:
            specprefill.score_tokens(model, [1, 2])
        except Exception as exc:  # noqa: BLE001 - asserted below
            first_errors.append(exc)

    thread = threading.Thread(target=run_first_score)
    thread.start()
    assert entered_prefill.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already has an active session"):
        specprefill.score_tokens(model, [3, 4])
    release_prefill.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert prefill_calls == 1
    assert len(first_errors) == 1
    assert str(first_errors[0]) == "stop first scorer"


def test_partial_wrapper_installation_rolls_back(monkeypatch):
    model = ScorerModel()
    model.layers.append(SimpleNamespace(self_attn=RecordingAttention()))
    originals = [layer.self_attn for layer in model.layers]
    real_set_attention = specprefill._set_attn_module
    set_calls = 0

    def fail_second_install(layer, module):
        nonlocal set_calls
        set_calls += 1
        if set_calls == 2:
            raise RuntimeError("install failed")
        real_set_attention(layer, module)

    monkeypatch.setattr(specprefill, "_set_attn_module", fail_second_install)
    with pytest.raises(RuntimeError, match="install failed"):
        SpecPrefillScorer(model)

    assert model.layers[0].self_attn is originals[0]
    assert model.layers[1].self_attn is originals[1]


def test_unknown_family_fails_before_installing_any_wrapper():
    model = ScorerModel(model_type="unknown-family")
    original = model.layers[0].self_attn
    with pytest.raises(ValueError, match="Unsupported SpecPrefill model_type"):
        SpecPrefillScorer.for_model(model)
    assert model.layers[0].self_attn is original


def test_wrapper_topology_tamper_fails_closed():
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    model.layers[0].self_attn = RecordingAttention()

    with pytest.raises(RuntimeError, match="wrapper topology was modified"):
        SpecPrefillScorer.for_model(model)
    with pytest.raises(RuntimeError, match="wrapper topology was modified"):
        with scorer.capture_session(passthrough_extractor):
            pass
