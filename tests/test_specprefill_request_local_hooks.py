# SPDX-License-Identifier: Apache-2.0
"""Request-local scorer capture contracts that do not load model weights."""

import threading
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten

import vllm_mlx.specprefill as specprefill
from vllm_mlx.specprefill import (
    SpecPrefillScorer,
    _compute_importance,
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


class ModuleAttention(nn.Module):
    n_heads = 1
    n_kv_heads = 1

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def __call__(self, x, mask=None, cache=None):
        del mask, cache
        return self.proj(x)


class ModuleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ModuleAttention()


class ModuleScorerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.layers = [ModuleLayer()]


def passthrough_extractor(_attention, x, cache=None, **_kwargs):
    return x


def test_scorer_installs_one_stable_dispatch_and_reuses_it():
    model = ScorerModel()
    original = model.layers[0].self_attn

    first = SpecPrefillScorer.for_model(model)
    second = SpecPrefillScorer.for_model(model)

    assert first is second
    assert model.layers[0].self_attn is original
    with pytest.raises(RuntimeError, match="already registered"):
        SpecPrefillScorer(model)
    assert model.layers[0].self_attn is original


def test_idle_dispatch_delegates_original_args_and_kwargs_unchanged():
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    original = model.layers[0].self_attn
    x = mx.zeros((1, 1, 1))
    mask = object()
    cache = object()
    shared_kv = object()

    result = original(x, mask, cache, shared_kv=shared_kv, offset=23)

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
    original = model.layers[0].self_attn
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
        original(x, mask, cache, offset=offset, shared_kv=shared_kv)
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

    original(x, cache=cache)
    assert session.query_buffer[0] == [x]
    assert model.layers[0].self_attn is original
    assert not scorer.capture_active


def test_real_mlx_module_identity_parameters_children_and_update_are_unchanged():
    model = ModuleScorerModel()
    attention = model.layers[0].self_attn
    with pytest.raises(TypeError):
        hash(model)
    before_parameters = tree_flatten(model.parameters())
    before_keys = tuple(key for key, _value in before_parameters)
    before_values = tuple(value for _key, value in before_parameters)
    before_children = model.children()
    before_output = attention(mx.ones((1, 1, 4)))
    mx.eval(before_output, model.parameters())

    scorer = SpecPrefillScorer.for_model(model)

    after_parameters = tree_flatten(model.parameters())
    assert model.layers[0].self_attn is attention
    assert tuple(key for key, _value in after_parameters) == before_keys
    assert all(
        after is before
        for (_key, after), before in zip(after_parameters, before_values, strict=True)
    )
    assert model.children().keys() == before_children.keys()
    assert model.children()["layers"][0] is before_children["layers"][0]
    assert model.children()["layers"][0].self_attn is attention
    model.load_weights(before_parameters, strict=True)
    model.update(model.parameters())
    after_output = attention(mx.ones((1, 1, 4)))
    mx.eval(after_output, model.parameters())
    assert mx.allclose(before_output, after_output).item()
    assert SpecPrefillScorer.for_model(model) is scorer


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
    attention = model.layers[0].self_attn

    with pytest.raises(ValueError, match="cancelled"):
        with scorer.capture_session(passthrough_extractor):
            raise ValueError("cancelled")

    assert not scorer.capture_active
    assert model.layers[0].self_attn is attention
    with scorer.capture_session(passthrough_extractor):
        assert scorer.capture_active


def test_public_score_tokens_delegates_to_bounded_request_local_session(monkeypatch):
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    attention = model.layers[0].self_attn
    import vllm_mlx.specprefill_scorer_session as scorer_sessions

    observed = {}

    class BoundedSession:
        def __init__(self, session_scorer, tokens, **kwargs):
            observed.update(scorer=session_scorer, tokens=tokens, kwargs=kwargs)

        def run_to_completion(self):
            assert not scorer.capture_active
            return mx.array([0.25, 0.75])

    monkeypatch.setattr(scorer_sessions, "SpecPrefillScorerSession", BoundedSession)

    importance = specprefill.score_tokens(
        model,
        [1, 2],
        n_lookahead=3,
        pool_kernel=7,
        query_extractor=passthrough_extractor,
    )

    assert importance.tolist() == [0.25, 0.75]
    assert observed["scorer"] is scorer
    assert observed["tokens"] == [1, 2]
    assert observed["kwargs"]["n_lookahead"] == 3
    assert observed["kwargs"]["pool_kernel"] == 7
    assert observed["kwargs"]["query_extractor"] is passthrough_extractor
    assert not scorer.capture_active
    assert model.layers[0].self_attn is attention


def test_partial_dispatch_installation_rolls_back(monkeypatch):
    model = ScorerModel()
    model.layers.append(SimpleNamespace(self_attn=RecordingAttention()))
    originals = [layer.self_attn for layer in model.layers]
    real_install = specprefill._install_attention_capture
    install_calls = 0

    def fail_second_install(attention, scorer, buffer_index):
        nonlocal install_calls
        install_calls += 1
        if install_calls == 2:
            raise RuntimeError("install failed")
        real_install(attention, scorer, buffer_index)

    monkeypatch.setattr(specprefill, "_install_attention_capture", fail_second_install)
    with pytest.raises(RuntimeError, match="install failed"):
        SpecPrefillScorer(model)

    assert model.layers[0].self_attn is originals[0]
    assert model.layers[1].self_attn is originals[1]
    with specprefill._ATTENTION_DISPATCH_LOCK:
        assert specprefill._attention_capture_for(originals[0]) is None


def test_unknown_family_fails_before_installing_any_wrapper():
    model = ScorerModel(model_type="unknown-family")
    original = model.layers[0].self_attn
    with pytest.raises(ValueError, match="Unsupported SpecPrefill model_type"):
        SpecPrefillScorer.for_model(model)
    assert model.layers[0].self_attn is original


def test_attention_topology_tamper_fails_closed():
    model = ScorerModel()
    scorer = SpecPrefillScorer.for_model(model)
    model.layers[0].self_attn = RecordingAttention()

    with pytest.raises(RuntimeError, match="attention topology was modified"):
        SpecPrefillScorer.for_model(model)
    with pytest.raises(RuntimeError, match="attention topology was modified"):
        with scorer.capture_session(passthrough_extractor):
            pass


def test_attention_class_dispatch_tamper_fails_closed_and_restores():
    model = ScorerModel()
    attention = model.layers[0].self_attn
    scorer = SpecPrefillScorer.for_model(model)
    attention_type = type(attention)
    installed_call = attention_type.__call__

    def tampered_call(instance, *args, **kwargs):
        return installed_call(instance, *args, **kwargs)

    attention_type.__call__ = tampered_call
    try:
        with pytest.raises(RuntimeError, match="class dispatcher was modified"):
            with scorer.capture_session(passthrough_extractor):
                pass
        assert model.layers[0].self_attn is attention
    finally:
        attention_type.__call__ = installed_call

    x = mx.ones((1, 1, 1))
    with scorer.capture_session(passthrough_extractor) as session:
        assert attention(x) is x
        assert session.query_buffer == [[x]]
    assert SpecPrefillScorer.for_model(model) is scorer


def test_mixed_gemma_head_groups_are_derived_per_layer():
    n_prompt = 3
    query_buffer = [
        [mx.ones((1, 32, 1, 256))],
        [mx.ones((1, 32, 1, 512))],
    ]
    caches = [
        SimpleNamespace(keys=mx.ones((1, 16, n_prompt, 256))),
        SimpleNamespace(keys=mx.ones((1, 4, n_prompt, 512))),
    ]

    importance = _compute_importance(
        query_buffer,
        caches,
        n_prompt,
        # Legacy global values describe only the first layer and must not leak
        # into the full-attention layer's 32-query/4-KV grouping.
        n_attn_heads=32,
        n_kv_heads=16,
        pool_kernel=None,
    )
    mx.eval(importance)

    assert importance.shape == (n_prompt,)
    assert mx.allclose(importance, mx.full((n_prompt,), 1 / n_prompt)).item()
