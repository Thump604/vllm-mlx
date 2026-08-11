"""Behavioral certification for Simple B=1 native Qwen MTP integration.

These tests deliberately use the tiny real dense and MoE Qwen modules from
the clean mlx-lm native-MTP branch.  They exercise model-owned transactional
caches through vllm-mlx's two supported Simple text wrappers without loading a
checkpoint or starting a service.
"""

from __future__ import annotations

import importlib
import os
import asyncio
import threading

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from vllm_mlx.engine.simple import SimpleEngine
from vllm_mlx.mlx_streams import bind_generation_streams
from vllm_mlx.models.llm import MLXLanguageModel
from vllm_mlx.native_mtp_request import NativeMTPRequestConfig, NativeMTPSampling


_HIDDEN = 64
_HEAD_DIM = 16


def _config(*, moe: bool) -> dict:
    text = {
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "hidden_size": _HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        # The production Metal recurrent kernel consumes one 32-wide SIMD
        # lane per key head.  Smaller synthetic widths force n_per_t=0 and do
        # not represent the eval-mode route used by SimpleEngine.
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 3,
        "full_attention_interval": 2,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-5,
        "head_dim": _HEAD_DIM,
        "rope_theta": 1000.0,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 128,
        "mtp_num_hidden_layers": 1,
    }
    if moe:
        text.update(
            {
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 128,
                "moe_intermediate_size": 64,
            }
        )
    return {"model_type": text["model_type"], "text_config": text}


def _native_model(*, moe: bool):
    # mlx-lm's module-level generation stream is thread-local.  Bind it on the
    # same thread that constructs, realizes, and later executes this fixture.
    bind_generation_streams(("mlx_lm.generate",))
    module = importlib.import_module(
        "mlx_lm.models.qwen3_5_moe" if moe else "mlx_lm.models.qwen3_5"
    )
    model = module.Model(module.ModelArgs.from_dict(_config(moe=moe)))
    if not hasattr(type(model.language_model), "mtp_capability"):
        if os.environ.get("VLLM_MLX_REQUIRE_NATIVE_MTP_TESTS") == "1":
            pytest.fail("designated native-MTP suite requires clean mlx-lm API")
        pytest.skip("requires mlx-lm native Qwen MTP capability branch")
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())

    # Exercise the same exact one-shot sanitize/load handshake as a real
    # checkpoint, using this tiny module's concrete arrays as the checkpoint.
    weights = dict(tree_flatten(model.parameters()))
    sanitized = model.sanitize(dict(weights))
    model.load_weights(list(sanitized.items()), strict=True)
    model.train(False)
    mx.eval(model.parameters())
    assert model.mtp_capability.supported is True
    return model


class _TinyTokenizer:
    bos_token = None
    eos_token_id = 999
    chat_template = "synthetic"

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return [0, 1]

    def decode(self, tokens):
        return "".join(chr(ord("a") + int(token) % 26) for token in tokens)

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        del messages, kwargs
        return [0, 1] if tokenize else "synthetic prompt"


def _tokenizer(*, eos_token_ids=(999,)):
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    return TokenizerWrapper(_TinyTokenizer(), eos_token_ids=eos_token_ids)


def _request(*, temperature=0.0, seed=None):
    return NativeMTPRequestConfig(
        NativeMTPSampling(
            temperature=temperature,
            top_p=0.9 if temperature else 1.0,
            top_k=8 if temperature else 0,
            min_p=0.01 if temperature else 0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            seed=seed,
        ),
        num_draft_tokens=1,
    )


def _language_model(model, tokenizer=None):
    wrapper = MLXLanguageModel("synthetic-qwen")
    wrapper.model = model
    wrapper.tokenizer = tokenizer or _tokenizer()
    wrapper._loaded = True
    return wrapper


def _track_requests(monkeypatch, model):
    requests = []
    original = model.make_mtp_request_cache

    def tracked(*, prompt_cache=None):
        request = original(prompt_cache=prompt_cache)
        requests.append(request)
        return request

    monkeypatch.setattr(model, "make_mtp_request_cache", tracked)
    return requests


def _assert_requests_closed(requests):
    assert requests
    for request in requests:
        assert request.closed is True
        assert request.checkpoint_active is False
        assert request.replay_required is None


def _tokens(outputs):
    return [output.token for output in outputs]


def _assert_target_logprobs(outputs):
    assert outputs
    for output in outputs:
        assert output.logprobs is not None
        assert mx.all(mx.isfinite(output.logprobs)).item()
        assert mx.allclose(
            mx.sum(mx.exp(output.logprobs)), mx.array(1.0), atol=1e-5
        ).item()
        assert mx.argmax(output.logprobs).item() == output.token


@pytest.mark.parametrize("moe", (False, True))
def test_real_native_qwen_mlx_language_model_greedy_parity_and_telemetry(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model)

    dense = list(wrapper.stream_generate([0, 1], max_tokens=4, temperature=0.0))
    native = list(
        wrapper.stream_generate(
            [0, 1], max_tokens=4, native_mtp_request=_request()
        )
    )

    assert _tokens(native) == _tokens(dense)
    _assert_target_logprobs(native)
    assert native[-1].mtp_drafts > 0
    assert 0 <= native[-1].mtp_accepted <= native[-1].mtp_drafts
    assert native[-1].mtp_bypass_reason is None
    _assert_requests_closed(requests)


@pytest.mark.parametrize("moe", (False, True))
def test_real_native_qwen_seeded_stochastic_stream_is_request_repeatable(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model)
    request = _request(temperature=0.7, seed=17)

    first = list(
        wrapper.stream_generate([0, 1], max_tokens=5, native_mtp_request=request)
    )
    second = list(
        wrapper.stream_generate([0, 1], max_tokens=5, native_mtp_request=request)
    )

    assert _tokens(first) == _tokens(second)
    assert (first[-1].mtp_drafts, first[-1].mtp_accepted) == (
        second[-1].mtp_drafts,
        second[-1].mtp_accepted,
    )
    _assert_requests_closed(requests)


@pytest.mark.parametrize("moe", (False, True))
def test_real_native_qwen_max1_eos_stop_and_iterator_close_cleanup(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model)

    max1 = list(
        wrapper.stream_generate([0, 1], max_tokens=1, native_mtp_request=_request())
    )
    assert len(max1) == 1
    assert max1[-1].finish_reason == "length"
    assert (max1[-1].mtp_drafts, max1[-1].mtp_accepted) == (0, 0)

    first_token = max1[0].token
    wrapper.tokenizer = _tokenizer(eos_token_ids=(first_token,))
    eos = list(
        wrapper.stream_generate([0, 1], max_tokens=4, native_mtp_request=_request())
    )
    assert len(eos) == 1
    assert eos[-1].finish_reason == "stop"

    wrapper.tokenizer = _tokenizer()
    stop = list(
        wrapper.stream_generate(
            [0, 1],
            max_tokens=8,
            stop=[max1[0].text],
            native_mtp_request=_request(),
        )
    )
    assert len(stop) == 1
    assert stop[-1].finish_reason == "stop"

    iterator = wrapper.stream_generate(
        [0, 1], max_tokens=8, native_mtp_request=_request()
    )
    next(iterator)
    assert requests[-1].closed is False
    iterator.close()
    assert requests[-1].closed is True
    _assert_requests_closed(requests)


@pytest.mark.anyio
@pytest.mark.parametrize("moe", (False, True))
async def test_real_native_qwen_simple_pure_llm_stream_and_cancel_cleanup(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model)
    engine = SimpleEngine("synthetic-qwen", mtp=True)
    engine._model = wrapper
    engine._loaded = True

    outputs = [
        output
        async for output in engine.stream_generate(
            "synthetic prompt",
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            _native_mtp_request_config=_request(),
        )
    ]
    assert outputs[-1].finished is True
    assert outputs[-1].mtp_drafts > 0
    assert 0 <= outputs[-1].mtp_accepted <= outputs[-1].mtp_drafts
    assert outputs[-1].logprobs is not None
    assert requests[-1].closed is True

    stream = engine.stream_generate(
        "synthetic prompt",
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        _native_mtp_request_config=_request(),
    )
    await anext(stream)
    assert requests[-1].closed is False
    assert engine._active_requests
    await stream.aclose()
    _assert_requests_closed(requests)
    assert engine._active_requests == {}
    assert engine._num_running == 0


@pytest.mark.anyio
@pytest.mark.parametrize("moe", (False, True))
async def test_real_native_qwen_vlm_derived_simple_text_route(monkeypatch, moe):
    import mlx_lm

    wrapper_model = _native_model(moe=moe)
    text_model = wrapper_model.language_model
    requests = _track_requests(monkeypatch, text_model)
    engine = SimpleEngine("synthetic-qwen", force_mllm=True, mtp=True)
    engine._loaded = True
    engine._text_model = text_model
    engine._text_tokenizer = _tokenizer()
    engine._supports_system_kv_cache = False

    outputs = [
        output
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            _native_mtp_request_config=_request(),
        )
    ]

    assert outputs[-1].finished is True
    assert outputs[-1].mtp_drafts > 0
    assert 0 <= outputs[-1].mtp_accepted <= outputs[-1].mtp_drafts
    assert outputs[-1].mtp_bypass_reason is None
    assert outputs[-1].logprobs is not None
    _assert_requests_closed(requests)

    # Hold the worker after its first real native response. Public GeneratorExit
    # must propagate through stream_chat -> tracker -> text route and close the
    # request before either active-request entry is released.
    original_stream_generate = mlx_lm.stream_generate
    release_worker = threading.Event()

    def held_stream(
        model,
        tokenizer,
        prompt,
        *,
        mtp=False,
        mtp_sampling_config=None,
        **kwargs,
    ):
        inner = original_stream_generate(
            model,
            tokenizer,
            prompt,
            mtp=mtp,
            mtp_sampling_config=mtp_sampling_config,
            **kwargs,
        )
        try:
            yield next(inner)
            release_worker.wait(timeout=5)
            yield from inner
        finally:
            inner.close()

    monkeypatch.setattr(mlx_lm, "stream_generate", held_stream)
    stream = engine.stream_chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        _native_mtp_request_config=_request(),
    )
    await anext(stream)
    assert requests[-1].closed is False
    assert engine._active_requests
    close_task = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    release_worker.set()
    await close_task
    _assert_requests_closed(requests)
    assert engine._active_requests == {}
    assert engine._num_running == 0
