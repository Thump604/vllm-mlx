from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.simple import SimpleEngine, _consume_native_mtp_request
from vllm_mlx.models.llm import MLXLanguageModel
from vllm_mlx.native_mtp_request import (
    NativeMTPRequestConfig,
    NativeMTPRequestError,
    NativeMTPSampling,
    NativeMTPServerState,
    native_mtp_consumer_supported,
    resolve_native_mtp_request,
)
from vllm_mlx.server import (
    _attach_native_mtp_request_kwargs,
    _generation_metadata,
    _preflight_streaming_completion_native_mtp,
)


def _sampling(seed=None):
    return NativeMTPSampling(
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        presence_penalty=0.1,
        repetition_penalty=1.1,
        seed=seed,
    )


@dataclass(frozen=True)
class _FakeUpstreamSampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int | None


def _install_fake_upstream_mtp(monkeypatch, calls):
    import importlib

    import mlx_lm

    generate_module = importlib.import_module("mlx_lm.generate")

    unset = object()

    def fake_stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=256,
        draft_model=None,
        mtp=unset,
        mtp_sampling_config=unset,
        **kwargs,
    ):
        calls.append(
            {
                "mtp": mtp,
                "mtp_sampling_config": mtp_sampling_config,
                "kwargs": kwargs,
            }
        )
        yield SimpleNamespace(
            text="x",
            token=1,
            logprobs="target-logprobs",
            mtp_drafts=3,
            mtp_accepted=2,
            mtp_bypass_reason=None,
            finish_reason="stop",
        )

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(
        generate_module,
        "NativeMTPSamplingConfig",
        _FakeUpstreamSampling,
        raising=False,
    )
    return fake_stream_generate, unset


def _kwargs(**overrides):
    values = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "presence_penalty": 0.1,
        "repetition_penalty": 1.1,
    }
    values.update(overrides)
    return values


class _Engine:
    def __init__(self, state):
        self.state = state

    def native_mtp_server_state(self, *, has_media=False):
        if has_media:
            return NativeMTPServerState(
                server_default=self.state.server_default,
                capable=self.state.capable,
                incompatibility="native_mtp_media_unsupported",
            )
        return self.state


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, CompletionRequest])
@pytest.mark.parametrize("value", [1, 0, "true", "false"])
def test_public_mtp_is_strict_boolean(request_type, value):
    body = {"model": "m", "mtp": value}
    body["messages" if request_type is ChatCompletionRequest else "prompt"] = (
        [{"role": "user", "content": "hi"}]
        if request_type is ChatCompletionRequest
        else "hi"
    )
    with pytest.raises(ValidationError):
        request_type(**body)


@pytest.mark.parametrize("request_type", [ChatCompletionRequest, CompletionRequest])
@pytest.mark.parametrize("seed", [True, "7", -1, 0x100000000])
def test_public_seed_is_bounded_strict_int(request_type, seed):
    body = {"model": "m", "mtp": True, "seed": seed}
    body["messages" if request_type is ChatCompletionRequest else "prompt"] = (
        [{"role": "user", "content": "hi"}]
        if request_type is ChatCompletionRequest
        else "hi"
    )
    with pytest.raises(ValidationError):
        request_type(**body)


def test_seed_conflicts_with_explicit_mtp_opt_out():
    with pytest.raises(ValidationError, match="seed conflicts with mtp=false"):
        CompletionRequest(model="m", prompt="hi", mtp=False, seed=7)


def test_native_and_external_mtp_intents_conflict():
    with pytest.raises(ValidationError, match="conflicts with mllm_draft"):
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            mtp=True,
            mllm_draft=True,
        )


def test_resolver_explicit_true_never_silently_downgrades():
    with pytest.raises(NativeMTPRequestError) as exc:
        resolve_native_mtp_request(
            requested=True,
            sampling=_sampling(),
            server_default=False,
            capable=False,
            num_draft_tokens=1,
        )
    assert exc.value.reason == "native_mtp_unsupported"


def test_server_attaches_exact_immutable_sampling_only_when_selected():
    engine = _Engine(
        NativeMTPServerState(
            server_default=False,
            capable=True,
            num_draft_tokens=3,
        )
    )
    request = CompletionRequest(model="m", prompt="hi", mtp=True, seed=17)
    kwargs = _kwargs()
    _attach_native_mtp_request_kwargs(engine, request, kwargs)

    config = kwargs.pop("_native_mtp_request_config")
    assert config == NativeMTPRequestConfig(_sampling(seed=17), num_draft_tokens=3)
    assert kwargs == _kwargs()
    with pytest.raises(Exception):
        config.num_draft_tokens = 9


def test_seed_uses_request_effective_native_default():
    state = NativeMTPServerState(server_default=True, capable=True)
    request = CompletionRequest(model="m", prompt="hi", seed=23)
    kwargs = _kwargs()
    _attach_native_mtp_request_kwargs(_Engine(state), request, kwargs)
    assert kwargs["_native_mtp_request_config"].sampling.seed == 23


def test_seed_fails_when_native_default_is_not_effective():
    state = NativeMTPServerState(server_default=False, capable=True)
    request = CompletionRequest(model="m", prompt="hi", seed=23)
    with pytest.raises(HTTPException) as exc:
        _attach_native_mtp_request_kwargs(_Engine(state), request, _kwargs())
    assert exc.value.status_code == 422
    assert exc.value.detail == "native_mtp_seed_requires_effective_mtp"


def test_default_off_and_explicit_false_do_not_change_backend_kwargs():
    state = NativeMTPServerState(server_default=False, capable=True)
    for requested in (None, False):
        request = CompletionRequest(model="m", prompt="hi", mtp=requested)
        kwargs = _kwargs()
        _attach_native_mtp_request_kwargs(_Engine(state), request, kwargs)
        assert kwargs == _kwargs()


def test_external_or_incapable_default_does_not_publish_native_bypass():
    state = NativeMTPServerState(server_default=False, capable=False)
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        mllm_draft=True,
    )
    kwargs = _kwargs()
    _attach_native_mtp_request_kwargs(_Engine(state), request, kwargs)
    assert kwargs == _kwargs()


def test_explicit_false_disables_only_an_active_native_default():
    state = NativeMTPServerState(server_default=True, capable=True)
    request = CompletionRequest(model="m", prompt="hi", mtp=False)
    kwargs = _kwargs()
    _attach_native_mtp_request_kwargs(_Engine(state), request, kwargs)
    assert kwargs.pop("_native_mtp_disabled") is True
    assert kwargs == _kwargs()


@pytest.mark.parametrize(
    "incompatibility",
    [
        "native_mtp_max_kv_unsupported",
        "native_mtp_prefix_cache_unsupported",
        "native_mtp_media_unsupported",
    ],
)
def test_explicit_incompatibility_fails_before_engine_invocation(incompatibility):
    state = NativeMTPServerState(
        server_default=True,
        capable=True,
        incompatibility=incompatibility,
    )
    request = CompletionRequest(model="m", prompt="hi", mtp=True)
    with pytest.raises(HTTPException) as exc:
        _attach_native_mtp_request_kwargs(_Engine(state), request, _kwargs())
    assert exc.value.status_code == 422
    assert exc.value.detail == incompatibility


def test_streaming_completion_preflights_before_sse_generator_starts():
    state = NativeMTPServerState(
        server_default=True,
        capable=True,
        incompatibility="native_mtp_max_kv_unsupported",
    )
    request = CompletionRequest(model="m", prompt="hi", stream=True, mtp=True)
    with pytest.raises(HTTPException) as exc:
        _preflight_streaming_completion_native_mtp(
            _Engine(state),
            request,
            repetition_penalty=None,
        )
    assert exc.value.status_code == 422
    assert exc.value.detail == "native_mtp_max_kv_unsupported"


def test_any_constructed_processor_fails_explicit_native_mtp_closed():
    state = NativeMTPServerState(server_default=True, capable=True)
    request = CompletionRequest(model="m", prompt="hi", mtp=True)
    with pytest.raises(HTTPException) as exc:
        _attach_native_mtp_request_kwargs(
            _Engine(state),
            request,
            _kwargs(logits_processors=[object()]),
        )
    assert exc.value.detail == "native_mtp_logits_processors_unsupported"


def test_mllm_hidden_penalty_processors_fail_before_engine_invocation():
    state = NativeMTPServerState(
        server_default=True,
        capable=True,
        supports_penalty_processors=False,
    )
    request = CompletionRequest(model="m", prompt="hi", mtp=True)
    with pytest.raises(HTTPException) as exc:
        _attach_native_mtp_request_kwargs(
            _Engine(state),
            request,
            _kwargs(repetition_penalty=1.2),
        )
    assert exc.value.detail == "native_mtp_penalty_processors_unsupported"


def test_default_native_mtp_bypass_is_stable_and_terminally_visible():
    state = NativeMTPServerState(server_default=True, capable=True)
    request = CompletionRequest(model="m", prompt="hi")
    kwargs = _kwargs(logits_processors=[object()])
    _attach_native_mtp_request_kwargs(_Engine(state), request, kwargs)
    reason = kwargs.pop("_native_mtp_bypass_reason")
    assert reason == "native_mtp_logits_processors_unsupported"
    assert kwargs.pop("_native_mtp_disabled") is True

    metadata = _generation_metadata(
        None,
        GenerationOutput(text="ok", mtp_bypass_reason=reason),
    )
    assert metadata is not None
    assert metadata.mtp_bypass_reason == reason


def test_default_native_mtp_yields_to_external_draft_without_leaking_controls():
    state = NativeMTPServerState(server_default=True, capable=True)
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        mllm_draft=True,
    )
    kwargs = _kwargs()
    _attach_native_mtp_request_kwargs(_Engine(state), request, kwargs)
    assert kwargs["_native_mtp_disabled"] is True
    assert kwargs["_native_mtp_bypass_reason"] == "native_mtp_external_draft_selected"

    controls = {
        name: kwargs.pop(name)
        for name in tuple(kwargs)
        if name.startswith("_native_mtp_")
    }
    assert _consume_native_mtp_request(controls, server_default=True) == (
        None,
        True,
        False,
        "native_mtp_external_draft_selected",
    )
    assert controls == {}
    assert kwargs == _kwargs()


def test_ordinary_output_does_not_create_generation_metadata():
    assert _generation_metadata(None, GenerationOutput(text="ok")) is None


def test_simple_capability_rejects_max_kv_and_resident_prefix_state(monkeypatch):
    engine = object.__new__(SimpleEngine)
    engine._mtp = True
    engine._mtp_num_draft_tokens = 2
    engine._is_mllm = False
    engine._model = SimpleNamespace(
        model=SimpleNamespace(
            mtp_capability=SimpleNamespace(supported=True, reason="supported")
        )
    )
    monkeypatch.setattr(
        "vllm_mlx.engine.simple.resolve_native_mtp_consumer", lambda: object()
    )
    engine._max_kv_size = 128
    engine._system_kv_cache = {}
    assert (
        engine.native_mtp_server_state().incompatibility
        == "native_mtp_max_kv_unsupported"
    )
    engine._max_kv_size = 0
    engine._system_kv_cache = {"prefix": object()}
    assert (
        engine.native_mtp_server_state().incompatibility
        == "native_mtp_prefix_cache_unsupported"
    )


def test_configured_native_default_keeps_intent_when_head_is_missing():
    engine = object.__new__(SimpleEngine)
    engine._mtp = True
    engine._mtp_num_draft_tokens = 1
    engine._is_mllm = False
    engine._model = SimpleNamespace(model=SimpleNamespace())
    engine._max_kv_size = 0
    engine._system_kv_cache = {}
    state = engine.native_mtp_server_state()
    assert state.server_default is True
    assert state.capable is False

    request = CompletionRequest(model="m", prompt="hi")
    kwargs = _kwargs()
    _attach_native_mtp_request_kwargs(_Engine(state), request, kwargs)
    assert kwargs["_native_mtp_bypass_reason"] == "native_mtp_model_capability_missing"
    assert kwargs["_native_mtp_disabled"] is True


def test_private_controls_are_consumed_and_never_reach_backends():
    config = NativeMTPRequestConfig(_sampling(), num_draft_tokens=4)
    kwargs = {"_native_mtp_request_config": config, "ordinary": 1}
    assert _consume_native_mtp_request(kwargs, server_default=False) == (
        config,
        False,
        True,
        None,
    )
    assert kwargs == {"ordinary": 1}


def test_consumer_requires_explicit_current_mlx_lm_parameters():
    def legacy(model, tokenizer, prompt, **kwargs):
        del model, tokenizer, prompt, kwargs

    def current(model, tokenizer, prompt, mtp=False, mtp_sampling_config=None):
        del model, tokenizer, prompt, mtp, mtp_sampling_config

    assert native_mtp_consumer_supported(legacy) is False
    assert native_mtp_consumer_supported(current) is True


def _fake_language_model():
    model = object.__new__(MLXLanguageModel)
    model._loaded = True
    model.model = object()
    model.tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2])
    model._mtp = False
    model._mtp_num_draft_tokens = 4
    model._create_sampler = lambda *args: object()
    model._create_logits_processors = lambda *args: []
    return model


def test_pure_llm_selected_forwards_exact_native_mtp_kwargs(monkeypatch):
    calls = []
    _, unset = _install_fake_upstream_mtp(monkeypatch, calls)
    config = NativeMTPRequestConfig(_sampling(seed=11), num_draft_tokens=4)

    list(
        _fake_language_model().stream_generate(
            "hi", max_tokens=1, native_mtp_request=config
        )
    )

    assert calls[0]["mtp"] is True
    assert calls[0]["mtp"] is not unset
    assert calls[0]["mtp_sampling_config"] == _FakeUpstreamSampling(
        0.7, 0.9, 20, 0.05, 11
    )
    assert "num_draft_tokens" not in calls[0]["kwargs"]
    assert "sampler" not in calls[0]["kwargs"]


def test_pure_llm_default_off_and_false_do_not_forward_native_kwargs(monkeypatch):
    calls = []
    _, unset = _install_fake_upstream_mtp(monkeypatch, calls)
    model = _fake_language_model()

    list(model.stream_generate("hi", max_tokens=1))
    list(model.stream_generate("hi", max_tokens=1, native_mtp_disabled=True))

    assert [call["mtp"] for call in calls] == [unset, unset]
    assert [call["mtp_sampling_config"] for call in calls] == [unset, unset]
    assert all("sampler" in call["kwargs"] for call in calls)


@pytest.mark.anyio
async def test_mllm_text_selected_forwards_exact_native_mtp_kwargs(monkeypatch):
    calls = []
    _install_fake_upstream_mtp(monkeypatch, calls)
    import importlib

    sample_utils = importlib.import_module("mlx_lm.sample_utils")
    monkeypatch.setattr(sample_utils, "make_sampler", lambda **kwargs: object())
    monkeypatch.setattr(sample_utils, "make_logits_processors", lambda **kwargs: [])
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: "rendered",
        bos_token=None,
        eos_token_id=99,
        eos_token_ids={99},
    )
    engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
    engine._loaded = True
    engine._text_model = SimpleNamespace(
        mtp_capability=SimpleNamespace(supported=True, reason="supported")
    )
    engine._text_tokenizer = tokenizer
    config = NativeMTPRequestConfig(
        NativeMTPSampling(0.7, 0.9, 20, 0.05, 0.0, 1.0, None),
        num_draft_tokens=4,
    )

    outputs = [
        output
        async for output in engine._stream_generate_text(
            [{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.7,
            top_p=0.9,
            combined_mtp=True,
            _native_mtp_request_config=config,
        )
    ]

    assert outputs[-1].text == "x"
    assert outputs[-1].logprobs == "target-logprobs"
    assert (outputs[-1].mtp_drafts, outputs[-1].mtp_accepted) == (3, 2)
    assert calls[0]["mtp"] is True
    assert isinstance(calls[0]["mtp_sampling_config"], _FakeUpstreamSampling)
    assert "num_draft_tokens" not in calls[0]["kwargs"]
    assert "sampler" not in calls[0]["kwargs"]


def test_pure_llm_copies_native_target_logprobs_and_cumulative_telemetry(
    monkeypatch,
):
    import mlx_lm

    closed = []
    target_logprobs = object()

    def fake_stream_generate(
        model,
        tokenizer,
        prompt,
        *,
        mtp=False,
        mtp_sampling_config=None,
        **kwargs,
    ):
        del model, tokenizer, prompt, mtp, mtp_sampling_config, kwargs
        try:
            yield SimpleNamespace(
                text="x",
                token=3,
                logprobs=target_logprobs,
                mtp_drafts=2,
                mtp_accepted=1,
                mtp_bypass_reason=None,
                finish_reason="length",
            )
        finally:
            closed.append(True)

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    config = NativeMTPRequestConfig(_sampling(seed=11), num_draft_tokens=1)

    outputs = list(
        _fake_language_model().stream_generate(
            "hi", max_tokens=1, native_mtp_request=config
        )
    )

    assert len(outputs) == 1
    assert outputs[0].logprobs is target_logprobs
    assert (outputs[0].mtp_drafts, outputs[0].mtp_accepted) == (2, 1)
    assert outputs[0].finish_reason == "length"
    assert closed == [True]


def test_pure_llm_early_stop_deterministically_closes_upstream(monkeypatch):
    import mlx_lm

    closed = []

    def fake_stream_generate(
        model,
        tokenizer,
        prompt,
        *,
        mtp=False,
        mtp_sampling_config=None,
        **kwargs,
    ):
        del model, tokenizer, prompt, mtp, mtp_sampling_config, kwargs
        try:
            yield SimpleNamespace(text="STOP", token=3, finish_reason=None)
            yield SimpleNamespace(text="unreachable", token=4, finish_reason=None)
        finally:
            closed.append(True)

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    config = NativeMTPRequestConfig(_sampling(seed=11), num_draft_tokens=1)

    outputs = list(
        _fake_language_model().stream_generate(
            "hi",
            max_tokens=8,
            stop=["STOP"],
            native_mtp_request=config,
        )
    )

    assert len(outputs) == 1
    assert outputs[0].finish_reason == "stop"
    assert closed == [True]


def test_pure_llm_iteration_error_is_not_masked_by_close_error(monkeypatch):
    import mlx_lm

    class FailingStream:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("primary iteration failure")

        def close(self):
            raise ValueError("secondary close failure")

    def fake_stream_generate(
        model,
        tokenizer,
        prompt,
        *,
        mtp=False,
        mtp_sampling_config=None,
        **kwargs,
    ):
        del model, tokenizer, prompt, mtp, mtp_sampling_config, kwargs
        return FailingStream()

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    config = NativeMTPRequestConfig(_sampling(seed=11), num_draft_tokens=1)

    with pytest.raises(RuntimeError, match="primary iteration failure"):
        list(
            _fake_language_model().stream_generate(
                "hi", max_tokens=2, native_mtp_request=config
            )
        )


@pytest.mark.anyio
async def test_tracker_primary_error_survives_aclose_error_and_cleans_state():
    class FailingAsyncStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("primary async iteration failure")

        async def aclose(self):
            raise ValueError("secondary async close failure")

    engine = SimpleEngine("synthetic")
    with pytest.raises(RuntimeError, match="primary async iteration failure"):
        async for _ in engine._track_request_stream(FailingAsyncStream()):
            pass
    assert engine._active_requests == {}
    assert engine._num_running == 0


@pytest.mark.anyio
async def test_completion_preflight_rejection_finishes_metrics(monkeypatch):
    import vllm_mlx.server as server

    tracker = MagicMock()
    engine = _Engine(
        NativeMTPServerState(
            server_default=True,
            capable=True,
            incompatibility="native_mtp_consumer_contract_missing",
        )
    )

    async def acquire(*args, **kwargs):
        return engine

    async def release(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "_validate_model_name", lambda model: None)
    monkeypatch.setattr(server._metrics, "track_inference", lambda *a, **k: tracker)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_engine_for_request", release)
    monkeypatch.setattr(
        server, "_validate_cb_specprefill_request_tuning", lambda *a: None
    )

    request = CompletionRequest(model="m", prompt="hi", stream=True, mtp=True)
    with pytest.raises(HTTPException) as exc:
        await server.create_completion(request, raw_request=None)
    assert exc.value.status_code == 422
    tracker.finish.assert_called_once_with(result="client_error")


@pytest.mark.anyio
async def test_chat_preparation_rejection_finishes_metrics(monkeypatch):
    import vllm_mlx.server as server

    tracker = MagicMock()

    async def acquire(*args, **kwargs):
        return object()

    async def release(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "_validate_model_name", lambda model: None)
    monkeypatch.setattr(server._metrics, "track_inference", lambda *a, **k: tracker)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_engine_for_request", release)
    monkeypatch.setattr(
        server, "_validate_cb_specprefill_request_tuning", lambda *a: None
    )
    monkeypatch.setattr(
        server,
        "_prepare_chat_completion_invocation",
        MagicMock(side_effect=HTTPException(status_code=422, detail="native rejected")),
    )

    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], mtp=True
    )
    with pytest.raises(HTTPException) as exc:
        await server.create_chat_completion(request, raw_request=None)
    assert exc.value.status_code == 422
    tracker.finish.assert_called_once_with(result="client_error")
