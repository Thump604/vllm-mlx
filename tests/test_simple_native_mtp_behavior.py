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
from functools import wraps
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from vllm_mlx.engine.simple import SimpleEngine
from vllm_mlx.mlx_streams import bind_generation_streams
from vllm_mlx.models.llm import MLXLanguageModel
from vllm_mlx.native_mtp_request import NativeMTPRequestConfig, NativeMTPSampling
from vllm_mlx.specprefill_positions import resolve_target_position_adapter
from vllm_mlx.specprefill_selection import (
    SelectionPlan,
    SelectionPolicy,
    SelectionProvenance,
)

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

    def __init__(self, tokens=(0, 1)):
        self.tokens = list(tokens)

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return list(self.tokens)

    def decode(self, tokens):
        return "".join(chr(ord("a") + int(token) % 26) for token in tokens)

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        del messages, kwargs
        return list(self.tokens) if tokenize else "synthetic prompt"


def _tokenizer(*, eos_token_ids=(999,), tokens=(0, 1)):
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    return TokenizerWrapper(_TinyTokenizer(tokens=tokens), eos_token_ids=eos_token_ids)


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
    from mlx_lm.models.cache import NativeMTPRequestCache

    requests = []
    original = model.make_mtp_request_cache
    original_adopt = NativeMTPRequestCache.adopt_sparse_target.__func__

    def tracked(*, prompt_cache=None):
        request = original(prompt_cache=prompt_cache)
        requests.append(request)
        return request

    monkeypatch.setattr(model, "make_mtp_request_cache", tracked)

    def tracked_adopt(cls, actual_model, **kwargs):
        request = original_adopt(cls, actual_model, **kwargs)
        if actual_model is model:
            requests.append(request)
        return request

    monkeypatch.setattr(
        NativeMTPRequestCache,
        "adopt_sparse_target",
        classmethod(tracked_adopt),
    )
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


def _sparse_profile(adapter):
    return SimpleNamespace(
        adapter_id=adapter.adapter_id,
        target_artifact_id="synthetic-qwen",
        target_artifact_hash="a" * 64,
        tokenizer_artifact_hash="b" * 64,
        scorer_artifact_id="synthetic-scorer",
        scorer_artifact_hash="c" * 64,
    )


def _selection_plan(prompt_length, selected_indices):
    policy = SelectionPolicy(
        keep_pct=len(selected_indices) / prompt_length,
        backbone_pct=0.0,
        halo_chunks=0,
        anchor_chunks=1,
        chunk_size=1,
    )
    return SelectionPlan(
        prompt_length=prompt_length,
        policy=policy,
        selected_chunks=selected_indices,
        selected_indices=selected_indices,
        provenance=SelectionProvenance(
            anchor_chunks=(0, prompt_length - 1),
        ),
    )


def _prepare_native_sparse(monkeypatch, model, tokens, selected_indices):
    from mlx_lm.models.cache import make_prompt_cache

    engine = object.__new__(SimpleEngine)
    engine._prefill_step_size = 2
    engine._max_kv_size = None
    plan = _selection_plan(len(tokens), selected_indices)
    monkeypatch.setattr(
        "vllm_mlx.engine.simple.build_selection_plan",
        lambda *_args, **_kwargs: plan,
    )
    adapter = resolve_target_position_adapter(model)
    telemetry = SimpleNamespace(
        profile_selector_version=plan.selector_version,
        selected_tokens=None,
        target_prefill_ms=None,
    )
    cache = make_prompt_cache(model)
    result, forward_context, actual_plan, bootstrap = (
        engine._prepare_sparse_target_prefill(
            target_model=model,
            tokenizer=SimpleNamespace(
                all_special_ids=(),
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
            ),
            tokens=list(tokens),
            importance=mx.ones((len(tokens),)),
            cache=cache,
            telemetry=telemetry,
            keep_pct=plan.policy.keep_pct,
            backbone_pct=0.0,
            chunk_size=1,
            halo_chunks=0,
            anchor_chunks=1,
            profile_key=_sparse_profile(adapter),
            adapter=adapter,
            combined_mtp=True,
        )
    )
    assert actual_plan is plan
    return result, forward_context, bootstrap


def _position_correct_sequential_oracle(
    monkeypatch, model, tokens, selected_indices, *, max_tokens
):
    from mlx_lm.generate import (
        GenerationForward,
        GenerationForwardPhase,
        GenerationForwardPositionAck,
    )

    result, forward_context, bootstrap = _prepare_native_sparse(
        monkeypatch, model, tokens, selected_indices
    )
    assert bootstrap.try_abandon_unclaimed() is True
    generated = [mx.argmax(result.logits[:, -1, :], axis=-1).item()]

    try:
        while len(generated) < max_tokens:
            position = len(tokens) + len(generated) - 1
            input_tokens = mx.array([[generated[-1]]], dtype=mx.uint32)
            ack = GenerationForwardPositionAck(
                (position,),
                model=model,
                cache=bootstrap.target_cache,
                phase=GenerationForwardPhase.DECODE,
            )
            forward = GenerationForward(
                model=model,
                input_tokens=input_tokens,
                cache=bootstrap.target_cache,
                phase=GenerationForwardPhase.DECODE,
                logical_positions=(position,),
                logical_position_ack=ack,
            )
            with forward_context(forward):
                ack._activate()
                try:
                    logits = model(input_tokens, cache=bootstrap.target_cache)
                    ack._require_acknowledged()
                    mx.eval(logits, [entry.state for entry in bootstrap.target_cache])
                finally:
                    ack._finish()
            generated.append(mx.argmax(logits[:, -1, :], axis=-1).item())
    finally:
        forward_context.finish()
    return generated


def _force_mtp_drafts(monkeypatch, model, oracle, *, selected_count, accept):
    original = model.mtp_forward
    next_oracle_index = 1

    def controlled(hidden_states, next_token_ids, mtp_cache):
        nonlocal next_oracle_index
        logits = original(hidden_states, next_token_ids, mtp_cache)
        # Sparse bootstrap first fills the N-1 prompt pairs. Only subsequent
        # calls are speculative drafts whose token choice affects acceptance.
        if mtp_cache[0].offset <= selected_count - 1:
            return logits
        oracle_id = oracle[min(next_oracle_index, len(oracle) - 1)]
        draft_id = oracle_id if accept else (oracle_id + 1) % logits.shape[-1]
        controlled_logits = mx.full(logits.shape, -100.0, dtype=logits.dtype)
        controlled_logits[..., draft_id] = 100.0
        next_oracle_index += 2 if accept else 1
        return controlled_logits

    monkeypatch.setattr(model, "mtp_forward", controlled)


def _configure_public_sparse_engine(monkeypatch, engine, model, tokens, selected):
    plan = _selection_plan(len(tokens), selected)
    adapter = resolve_target_position_adapter(model)
    profile = _sparse_profile(adapter)
    monkeypatch.setattr(
        "vllm_mlx.engine.simple.build_selection_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        "vllm_mlx.specprefill.score_tokens",
        lambda *_args, **_kwargs: mx.ones((len(tokens),)),
    )
    monkeypatch.setattr(
        engine,
        "_admit_sparse_target",
        lambda actual_model, *, combined_mtp=False: (
            profile,
            resolve_target_position_adapter(actual_model),
        ),
    )
    engine._draft_model = object()


@pytest.mark.parametrize("moe", (False, True))
def test_real_qwen_keep_one_sparse_native_mtp_matches_dense_public_stream(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model)
    tokens = (0, 1, 2, 3, 4, 5)
    dense = list(
        wrapper.stream_generate(
            list(tokens), max_tokens=4, native_mtp_request=_request()
        )
    )

    _, forward_context, bootstrap = _prepare_native_sparse(
        monkeypatch, model, tokens, tuple(range(len(tokens)))
    )
    sparse = list(
        wrapper.stream_generate(
            None,
            max_tokens=4,
            native_mtp_request=_request(),
            sparse_bootstrap=bootstrap,
            model_forward_context=forward_context,
        )
    )

    assert _tokens(sparse) == _tokens(dense)
    assert sparse[-1].prompt_tokens == len(tokens)
    assert sparse[-1].mtp_drafts > 0
    request = requests[-1]
    assert request.state.backbone_tokens >= len(tokens)
    assert request.state.mtp_tokens >= len(tokens) - 1
    assert request.state.next_logical_position >= len(tokens)
    _assert_requests_closed(requests)


@pytest.mark.parametrize("moe", (False, True))
def test_real_qwen_noncontiguous_receipts_use_original_immediate_successors(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    tokens = (7, 11, 13, 17, 19, 23)
    selected = (0, 2, 4, 5)
    result, forward_context, bootstrap = _prepare_native_sparse(
        monkeypatch, model, tokens, selected
    )

    assert bootstrap.selected_logical_positions == selected
    assert bootstrap.selected_token_ids == tuple(tokens[index] for index in selected)
    assert bootstrap.immediate_successor_token_ids == (11, 17, 23)
    assert tuple(
        successor
        for receipt in bootstrap.receipts
        for successor in receipt.immediate_successor_token_ids
    ) == (11, 17, 23)
    assert all(receipt.logits is None for receipt in bootstrap.receipts[:-1])
    assert bootstrap.receipts[-1].logits is not None
    assert result.telemetry.selected_tokens == len(selected)
    assert result.telemetry.physical_cache_starts == ((0,), (2,))
    assert bootstrap.try_abandon_unclaimed() is True
    forward_context.finish()


@pytest.mark.parametrize("moe", (False, True))
def test_sparse_native_mtp_first_output_has_exact_physical_counts_and_cursor(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model)
    tokens = (2, 3, 5, 7, 11, 13)
    selected = (0, 2, 4, 5)
    _, forward_context, bootstrap = _prepare_native_sparse(
        monkeypatch, model, tokens, selected
    )
    generation = wrapper.stream_generate(
        None,
        max_tokens=4,
        native_mtp_request=_request(),
        sparse_bootstrap=bootstrap,
        model_forward_context=forward_context,
    )

    first = next(generation)
    request = requests[-1]
    assert first.prompt_tokens == len(tokens)
    assert request.state.backbone_tokens == len(selected)
    assert request.state.mtp_tokens == len(selected) - 1
    assert request.state.next_logical_position == len(tokens) + 1
    generation.close()
    assert bootstrap.try_abandon_unclaimed() is False
    forward_context.finish()
    _assert_requests_closed(requests)


@pytest.mark.parametrize("moe", (False, True))
def test_real_native_qwen_mlx_language_model_greedy_parity_and_telemetry(
    monkeypatch, moe
):
    model = _native_model(moe=moe)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model)

    dense = list(wrapper.stream_generate([0, 1], max_tokens=4, temperature=0.0))
    native = list(
        wrapper.stream_generate([0, 1], max_tokens=4, native_mtp_request=_request())
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
def test_real_native_qwen_max1_eos_stop_and_iterator_close_cleanup(monkeypatch, moe):
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
@pytest.mark.parametrize("accept", (False, True), ids=("reject_replay", "accept"))
async def test_public_simple_llm_noncontiguous_sparse_native_mtp_matches_oracle(
    monkeypatch, accept
):
    tokens = (7, 11, 13, 17, 19, 23)
    selected = (0, 2, 4, 5)
    model = _native_model(moe=False)
    oracle = _position_correct_sequential_oracle(
        monkeypatch, model, tokens, selected, max_tokens=4
    )
    _force_mtp_drafts(
        monkeypatch,
        model,
        oracle,
        selected_count=len(selected),
        accept=accept,
    )
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model, _tokenizer(tokens=tokens))
    engine = SimpleEngine(
        "synthetic-qwen",
        mtp=True,
        specprefill_enabled=True,
        specprefill_threshold=1,
        specprefill_diagnostic_mode=True,
    )
    engine._model = wrapper
    engine._loaded = True
    _configure_public_sparse_engine(monkeypatch, engine, model, tokens, selected)

    outputs = [
        output
        async for output in engine.stream_generate(
            "synthetic prompt",
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            specprefill_policy="sparse",
            specprefill_keep_pct=len(selected) / len(tokens),
            specprefill_backbone_pct=0.0,
            _native_mtp_request_config=_request(),
        )
    ]
    output_ids = [mx.argmax(output.logprobs).item() for output in outputs]

    assert output_ids == oracle[: len(output_ids)]
    assert outputs[-1].specprefill_effective_policy == "sparse"
    assert outputs[-1].specprefill_selected_tokens == len(selected)
    assert outputs[-1].prompt_tokens == len(tokens)
    assert outputs[-1].mtp_drafts > 0
    assert (outputs[-1].mtp_accepted > 0) is accept
    assert all(output.logprobs is not None for output in outputs)
    assert all(mx.all(mx.isfinite(output.logprobs)).item() for output in outputs)
    _assert_requests_closed(requests)


@pytest.mark.anyio
async def test_public_simple_preclaim_failure_abandons_and_dense_falls_back(
    monkeypatch,
):
    tokens = (7, 11, 13, 17, 19, 23)
    selected = (0, 2, 4, 5)
    model = _native_model(moe=False)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model, _tokenizer(tokens=tokens))
    engine = SimpleEngine(
        "synthetic-qwen",
        mtp=True,
        specprefill_enabled=True,
        specprefill_threshold=1,
        specprefill_diagnostic_mode=True,
    )
    engine._model = wrapper
    engine._loaded = True
    _configure_public_sparse_engine(monkeypatch, engine, model, tokens, selected)
    bootstraps = []
    transfers = []
    original_prepare = engine._prepare_sparse_target_prefill

    def capture_prepare(**kwargs):
        prepared = original_prepare(**kwargs)
        bootstraps.append(prepared[-1])
        forward_context = prepared[-3]
        original_transfer = forward_context.transfer_to_native_mtp

        def track_transfer():
            transfers.append("before_native_iteration")
            return original_transfer()

        monkeypatch.setattr(forward_context, "transfer_to_native_mtp", track_transfer)
        return prepared

    monkeypatch.setattr(engine, "_prepare_sparse_target_prefill", capture_prepare)
    original_stream = wrapper.stream_generate

    @wraps(original_stream)
    def fail_sparse_before_claim(**kwargs):
        if kwargs.get("sparse_bootstrap") is not None:
            assert transfers == ["before_native_iteration"]

            def failed():
                raise ValueError("invalid sparse processor config")
                yield  # pragma: no cover

            return failed()
        return original_stream(**kwargs)

    monkeypatch.setattr(wrapper, "stream_generate", fail_sparse_before_claim)
    outputs = [
        output
        async for output in engine.stream_generate(
            "synthetic prompt",
            max_tokens=3,
            temperature=0.0,
            top_p=1.0,
            specprefill_policy="sparse",
            specprefill_keep_pct=len(selected) / len(tokens),
            specprefill_backbone_pct=0.0,
            _native_mtp_request_config=_request(),
        )
    ]

    assert outputs[-1].specprefill_effective_policy == "dense"
    assert outputs[-1].specprefill_fallback_reason == "sparse_execution_failed"
    assert transfers == ["before_native_iteration"]
    assert bootstraps[0].try_abandon_unclaimed() is False
    _assert_requests_closed(requests)


@pytest.mark.anyio
async def test_public_simple_postclaim_failure_is_terminal_without_dense_replay(
    monkeypatch,
):
    tokens = (7, 11, 13, 17, 19, 23)
    selected = (0, 2, 4, 5)
    model = _native_model(moe=False)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model, _tokenizer(tokens=tokens))
    dense_prompts = []
    original_stream = wrapper.stream_generate

    @wraps(original_stream)
    def count_dense(**kwargs):
        if kwargs.get("sparse_bootstrap") is None:
            dense_prompts.append(kwargs.get("prompt"))
        return original_stream(**kwargs)

    monkeypatch.setattr(wrapper, "stream_generate", count_dense)
    engine = SimpleEngine(
        "synthetic-qwen",
        mtp=True,
        specprefill_enabled=True,
        specprefill_threshold=1,
        specprefill_diagnostic_mode=True,
    )
    engine._model = wrapper
    engine._loaded = True
    _configure_public_sparse_engine(monkeypatch, engine, model, tokens, selected)

    def fail_after_adoption(*_args, **_kwargs):
        raise RuntimeError("postclaim MTP failure")

    monkeypatch.setattr(model, "mtp_forward", fail_after_adoption)
    with pytest.raises(RuntimeError, match="postclaim MTP failure"):
        _ = [
            output
            async for output in engine.stream_generate(
                "synthetic prompt",
                max_tokens=3,
                temperature=0.0,
                top_p=1.0,
                specprefill_policy="sparse",
                specprefill_keep_pct=len(selected) / len(tokens),
                specprefill_backbone_pct=0.0,
                _native_mtp_request_config=_request(),
            )
        ]

    assert dense_prompts == []
    _assert_requests_closed(requests)


@pytest.mark.anyio
async def test_public_simple_cancel_before_first_resume_abandons_authority(
    monkeypatch,
):
    tokens = (7, 11, 13, 17, 19, 23)
    selected = (0, 2, 4, 5)
    model = _native_model(moe=False)
    requests = _track_requests(monkeypatch, model)
    wrapper = _language_model(model, _tokenizer(tokens=tokens))
    engine = SimpleEngine(
        "synthetic-qwen",
        mtp=True,
        specprefill_enabled=True,
        specprefill_threshold=1,
        specprefill_diagnostic_mode=True,
    )
    engine._model = wrapper
    engine._loaded = True
    _configure_public_sparse_engine(monkeypatch, engine, model, tokens, selected)
    original_prepare = engine._prepare_sparse_target_prefill
    prepared_event = threading.Event()
    release_worker = threading.Event()
    bootstraps = []

    def held_prepare(**kwargs):
        prepared = original_prepare(**kwargs)
        bootstraps.append(prepared[-1])
        prepared_event.set()
        release_worker.wait(timeout=5)
        return prepared

    monkeypatch.setattr(engine, "_prepare_sparse_target_prefill", held_prepare)
    stream = engine.stream_generate(
        "synthetic prompt",
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        specprefill_policy="sparse",
        specprefill_keep_pct=len(selected) / len(tokens),
        specprefill_backbone_pct=0.0,
        _native_mtp_request_config=_request(),
    )
    next_task = asyncio.create_task(anext(stream))
    assert await asyncio.to_thread(prepared_event.wait, 5)
    next_task.cancel()
    # Let cancellation traverse public stream -> tracker -> sparse child. The
    # child sets its cooperative worker-cancel event and then waits for this
    # held worker, proving release cannot race ahead into bootstrap adoption.
    await asyncio.sleep(0)
    assert not next_task.done()
    release_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await next_task
    await stream.aclose()

    assert bootstraps[0].try_abandon_unclaimed() is False
    assert requests == []
    assert engine._active_requests == {}


@pytest.mark.anyio
async def test_public_simple_generator_exit_after_sparse_sample_closes_request(
    monkeypatch,
):
    tokens = (7, 11, 13, 17, 19, 23)
    selected = (0, 2, 4, 5)
    model = _native_model(moe=False)
    requests = _track_requests(monkeypatch, model)
    engine = SimpleEngine(
        "synthetic-qwen",
        mtp=True,
        specprefill_enabled=True,
        specprefill_threshold=1,
        specprefill_diagnostic_mode=True,
    )
    engine._model = _language_model(model, _tokenizer(tokens=tokens))
    engine._loaded = True
    _configure_public_sparse_engine(monkeypatch, engine, model, tokens, selected)
    original_stream = engine._model.stream_generate
    release_worker = threading.Event()

    @wraps(original_stream)
    def held_stream(**kwargs):
        inner = original_stream(**kwargs)
        try:
            yield next(inner)
            release_worker.wait(timeout=5)
            yield from inner
        finally:
            inner.close()

    monkeypatch.setattr(engine._model, "stream_generate", held_stream)
    stream = engine.stream_generate(
        "synthetic prompt",
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        specprefill_policy="sparse",
        specprefill_keep_pct=len(selected) / len(tokens),
        specprefill_backbone_pct=0.0,
        _native_mtp_request_config=_request(),
    )

    first = await anext(stream)
    assert first.specprefill_effective_policy == "sparse"
    assert requests[-1].closed is False
    close_task = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    release_worker.set()
    await close_task

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


@pytest.mark.anyio
@pytest.mark.parametrize("accept", (False, True), ids=("reject_replay", "accept"))
async def test_public_vlm_text_noncontiguous_sparse_native_mtp_matches_oracle(
    monkeypatch, accept
):
    tokens = (5, 7, 11, 13, 17, 19)
    selected = (0, 2, 4, 5)
    wrapper_model = _native_model(moe=False)
    model = wrapper_model.language_model
    oracle = _position_correct_sequential_oracle(
        monkeypatch, model, tokens, selected, max_tokens=4
    )
    _force_mtp_drafts(
        monkeypatch,
        model,
        oracle,
        selected_count=len(selected),
        accept=accept,
    )
    requests = _track_requests(monkeypatch, model)
    engine = SimpleEngine(
        "synthetic-qwen",
        force_mllm=True,
        mtp=True,
        specprefill_enabled=True,
        specprefill_threshold=1,
        specprefill_diagnostic_mode=True,
    )
    engine._loaded = True
    engine._text_model = model
    engine._text_tokenizer = _tokenizer(tokens=tokens)
    engine._supports_system_kv_cache = False
    _configure_public_sparse_engine(monkeypatch, engine, model, tokens, selected)

    outputs = [
        output
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            specprefill_policy="sparse",
            specprefill_keep_pct=len(selected) / len(tokens),
            specprefill_backbone_pct=0.0,
            _native_mtp_request_config=_request(),
        )
    ]
    output_ids = [mx.argmax(output.logprobs).item() for output in outputs]

    assert output_ids == oracle[: len(output_ids)]
    assert outputs[-1].specprefill_effective_policy == "sparse"
    assert outputs[-1].specprefill_selected_tokens == len(selected)
    assert outputs[-1].prompt_tokens == len(tokens)
    assert outputs[-1].mtp_drafts > 0
    assert (outputs[-1].mtp_accepted > 0) is accept
    assert all(output.logprobs is not None for output in outputs)
    assert all(mx.all(mx.isfinite(output.logprobs)).item() for output in outputs)
    _assert_requests_closed(requests)


# These are intentionally scheduler-level tests rather than direct adapter tests.
# The adapter-only contract is covered in test_cb_native_mtp_adapter.py; this
# slice proves that standard-text CB neither bypasses its public lifecycle nor
# loses output accounting when it drives actual Qwen caches.
def _cb_scheduler(model, tokenizer):
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    return Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            enable_mtp=True,
            enable_prefix_cache=False,
            max_num_seqs=4,
            prefill_step_size=2,
        ),
    )


def _cb_request(request_id, prompt, *, max_tokens=2, seed=17):
    from vllm_mlx.request import Request, SamplingParams

    request = Request(
        request_id=request_id,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    request.native_mtp_config = _request(seed=seed)
    return request


def _run_scheduler_to_terminal(scheduler):
    outputs = []
    # Every native-MTP epoch is one scheduler step.  A small fixed guard makes
    # a broken public lifecycle visible instead of masking it with an endless
    # test loop.
    for _ in range(32):
        result = scheduler.step()
        outputs.extend(result.outputs)
        if not scheduler.has_requests():
            return outputs
    raise AssertionError("native MTP scheduler did not reach a terminal state")


def _live_native_cohort(scheduler):
    """Return the adapter's sole live owner after source-cache consumption."""
    adapter = scheduler.native_mtp_adapter
    assert adapter is not None
    generator = adapter._generator
    assert generator is not None
    cohort = generator._cohort
    assert adapter.closed is False
    assert generator.closed is False
    assert cohort.poisoned is False
    assert cohort._checkpoint is None
    return adapter, generator, cohort


def _direct_native_b1_oracle(model, prompt, *, max_tokens, seed):
    """Collect the two observable boundaries from upstream's public API."""
    from mlx_lm.generate import (
        NativeMTPAdmission,
        NativeMTPBatchGenerator,
        NativeMTPRowSpec,
        NativeMTPSamplingConfig,
    )

    row = NativeMTPRowSpec(
        uid=1,
        prompt=tuple(prompt),
        max_tokens=max_tokens,
        seed=seed,
        eos_token_ids=frozenset(),
        sampling_config=NativeMTPSamplingConfig(temperature=0.0),
    )
    cache = model.make_mtp_request_cache(prompt_cache=None)
    generator = NativeMTPBatchGenerator(
        NativeMTPAdmission.create(model, (row,), (cache,))
    )
    initial, epoch = generator.prefill(prefill_step_size=2)
    ready = epoch.resume()
    decision = ready.decide()
    if decision.accepted_uids:
        emitted, terminal_epoch = decision.accept()
    else:
        emitted, terminal_epoch = decision.reject()
    # max_tokens=2 means this epoch cannot own a surviving row.  Still consume
    # its public cancellation to prove that the direct oracle leaves no cache.
    terminal_epoch.cancel()
    return tuple(initial) + tuple(emitted), cache


def _force_cb_second_token(monkeypatch, model, wanted_by_batch_row, *, prompt_tokens):
    """Force the first post-head MTP draft without changing target logits."""
    original = model.mtp_forward

    def controlled(hidden_states, next_token_ids, mtp_cache):
        logits = original(hidden_states, next_token_ids, mtp_cache)
        # Prompt-pair MTP prefill can have any width.  The first width-one MTP
        # call after it is the public Initial -> Ready draft boundary.  The
        # cohort may split and join its cache owner while pre-filling, so use
        # the actual MTP cursor rather than Python object identity or calls.
        width = int(next_token_ids.shape[1])
        offset = mtp_cache[0].offset
        cursor = int(mx.min(offset).item()) if isinstance(offset, mx.array) else offset
        if width != 1 or cursor < prompt_tokens:
            return logits
        values = [wanted_by_batch_row[index] for index in range(logits.shape[0])]
        forced = mx.full(logits.shape, -100.0, dtype=logits.dtype)
        for index, token in enumerate(values):
            forced[index, -1, token] = 100.0
        return forced

    monkeypatch.setattr(model, "mtp_forward", controlled)


@pytest.mark.parametrize("moe", (False, True), ids=("dense", "recurrent_kv_moe"))
@pytest.mark.parametrize("accept", (False, True), ids=("all_reject", "all_accept"))
def test_actual_qwen_standard_cb_b1_forced_decision_matches_public_oracle(
    monkeypatch, moe, accept
):
    model = _native_model(moe=moe).language_model
    requests = _track_requests(monkeypatch, model)
    prompt = (3, 5, 7, 11)
    wrapper = _language_model(model, _tokenizer(tokens=prompt))
    dense = _tokens(
        list(wrapper.stream_generate(list(prompt), max_tokens=2, temperature=0.0))
    )
    wanted = (
        dense[1]
        if accept
        else (dense[1] + 1) % _config(moe=moe)["text_config"]["vocab_size"]
    )
    _force_cb_second_token(monkeypatch, model, (wanted,), prompt_tokens=len(prompt))

    oracle, oracle_cache = _direct_native_b1_oracle(
        model, prompt, max_tokens=2, seed=17
    )
    scheduler = _cb_scheduler(model, _tokenizer(tokens=prompt))
    request = _cb_request("b1", prompt, max_tokens=2, seed=17)
    scheduler.add_request(request)
    outputs = _run_scheduler_to_terminal(scheduler)

    assert [token for output in outputs for token in output.new_token_ids] == [
        item.token for item in oracle
    ]
    assert [output.logprobs is not None for output in outputs] == [True, True]
    assert outputs[-1].finish_reason == "length"
    assert outputs[-1].completion_tokens == 2
    assert (outputs[-1].mtp_drafts, outputs[-1].mtp_accepted) == (1, int(accept))
    assert scheduler.total_prompt_tokens == len(prompt)
    assert scheduler.total_completion_tokens == 2
    assert scheduler.num_requests_processed == 1
    assert oracle_cache.closed is True
    _assert_requests_closed(requests)


@pytest.mark.parametrize("moe", (False, True), ids=("dense", "recurrent_kv_moe"))
def test_actual_qwen_standard_cb_batched_mixed_decision_is_uid_local(monkeypatch, moe):
    model = _native_model(moe=moe).language_model
    requests = _track_requests(monkeypatch, model)
    prompts = ((2, 3, 5, 7), (11, 13, 17, 19))
    wrapper = _language_model(model)
    dense = [
        _tokens(
            list(wrapper.stream_generate(list(prompt), max_tokens=2, temperature=0.0))
        )
        for prompt in prompts
    ]
    vocab = _config(moe=moe)["text_config"]["vocab_size"]
    wanted = (dense[0][1], (dense[1][1] + 1) % vocab)
    # Build independent B=1 public-lifecycle oracles under exactly the same
    # forced decision.  This is the regression guard for UID-local state: a
    # batched mixed cohort must not borrow its neighbour's outcome or cache.
    expected = {}
    oracle_caches = []
    for uid, (prompt, token) in enumerate(zip(prompts, wanted), start=1):
        with monkeypatch.context() as direct_patch:
            _force_cb_second_token(
                direct_patch, model, (token,), prompt_tokens=len(prompt)
            )
            expected[uid], cache = _direct_native_b1_oracle(
                model, prompt, max_tokens=2, seed=(19, 23)[uid - 1]
            )
            oracle_caches.append(cache)
    _force_cb_second_token(
        monkeypatch,
        model,
        wanted,
        prompt_tokens=len(prompts[0]),
    )

    scheduler = _cb_scheduler(model, _tokenizer())
    first = _cb_request("accepted", prompts[0], max_tokens=2, seed=19)
    second = _cb_request("rejected", prompts[1], max_tokens=2, seed=23)
    scheduler.add_request(first)
    scheduler.add_request(second)
    outputs = _run_scheduler_to_terminal(scheduler)
    by_request = {}
    for output in outputs:
        by_request.setdefault(output.request_id, []).extend(output.new_token_ids)

    assert set(by_request) == {"accepted", "rejected"}
    assert all(len(tokens) == 2 for tokens in by_request.values())
    assert by_request["accepted"] == [item.token for item in expected[1]]
    assert by_request["rejected"] == [item.token for item in expected[2]]
    assert all(output.logprobs is not None for output in outputs)
    final = {output.request_id: output for output in outputs if output.finished}
    assert {key: value.finish_reason for key, value in final.items()} == {
        "accepted": "length",
        "rejected": "length",
    }
    assert final["accepted"].mtp_accepted == 1
    assert final["rejected"].mtp_accepted == 0
    assert final["accepted"].mtp_drafts == final["rejected"].mtp_drafts == 1
    assert scheduler.total_prompt_tokens == sum(map(len, prompts))
    assert scheduler.total_completion_tokens == 4
    assert scheduler.num_requests_processed == 2
    assert all(cache.closed is True for cache in oracle_caches)
    _assert_requests_closed(requests)


@pytest.mark.parametrize("moe", (False, True), ids=("dense", "recurrent_kv_moe"))
def test_actual_qwen_standard_cb_cancellation_and_error_close_request_caches(
    monkeypatch, moe
):
    model = _native_model(moe=moe).language_model
    prompt = (3, 5, 7, 11)

    # A queued cancellation must not create a fresh native request cache.
    before = _track_requests(monkeypatch, model)
    scheduler = _cb_scheduler(model, _tokenizer())
    queued = _cb_request("queued", prompt)
    scheduler.add_request(queued)
    scheduler.abort_request("queued")
    queued_abort = scheduler.step()
    assert queued_abort.outputs == []
    assert queued_abort.finished_request_ids == set()
    assert scheduler.has_requests() is False
    assert before == []

    # Once initial prefill owns the cache, cancellation is cohort-scoped and
    # produces one abort output while closing the request-local owner.
    after = _track_requests(monkeypatch, model)
    scheduler = _cb_scheduler(model, _tokenizer())
    active = _cb_request("active", prompt)
    scheduler.add_request(active)
    first = scheduler.step()
    assert first.outputs
    # Admission transfers B=1 source ownership into a merged cohort, closing
    # the source wrappers before the first public emission is returned.
    _assert_requests_closed(after)
    adapter, generator, cohort = _live_native_cohort(scheduler)
    scheduler.abort_request("active")
    cancelled = scheduler.step()
    assert [
        (item.request_id, item.finished, item.finish_reason, item.completion_tokens)
        for item in cancelled.outputs
    ] == [("active", True, "abort", 1)]
    assert cancelled.finished_request_ids == {"active"}
    assert scheduler.has_requests() is False
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == scheduler.uid_to_request_id == {}
    assert adapter.closed is True
    assert generator.closed is True
    assert cohort.poisoned is True
    assert cohort._checkpoint is None

    # A lifecycle error after initial prefill must turn into a terminal error,
    # not an ordinary BatchGenerator fallback or a leaked cache owner.
    failed = _track_requests(monkeypatch, model)
    scheduler = _cb_scheduler(model, _tokenizer())
    request = _cb_request("failed", prompt)
    scheduler.add_request(request)
    scheduler.step()
    _assert_requests_closed(failed)
    adapter, generator, cohort = _live_native_cohort(scheduler)
    monkeypatch.setattr(
        model,
        "mtp_forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced_mtp_failure")
        ),
    )
    error = scheduler.step()
    assert [
        (item.request_id, item.finish_reason, item.native_mtp_error_reason)
        for item in error.outputs
    ] == [("failed", "error", "forced_mtp_failure")]
    assert scheduler.batch_generator is None
    assert adapter.closed is True
    assert generator.closed is True
    assert cohort.poisoned is True
    assert cohort._checkpoint is None
