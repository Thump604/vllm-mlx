# SPDX-License-Identifier: Apache-2.0
"""Synthetic successful SimpleEngine sparse-only lifecycle tests."""

from contextlib import nullcontext
from types import SimpleNamespace
import time

import mlx.core as mx
import pytest

from vllm_mlx.engine import simple as simple_module
from vllm_mlx.engine.simple import (
    SimpleEngine,
    _SpecPrefillCancelled,
    _build_text_tokenizer,
    _detokenize_sparse_token,
)
from vllm_mlx.specprefill_positions import GEMMA4_DENSE_TARGET, QWEN_DENSE_TARGET


class _Detokenizer:
    def reset(self):
        self._segment = ""

    def add_token(self, token):
        self._segment += {5: "A", 6: "B"}.get(token, f"<{token}>")

    def finalize(self):
        return None

    @property
    def last_segment(self):
        segment = self._segment
        self._segment = ""
        return segment


class _Tokenizer:
    bos_token = None
    eos_token_id = 9
    eos_token_ids = (9, 10)
    all_special_ids = (9, 10)

    @property
    def detokenizer(self):
        return _Detokenizer()


class _RawTokenizer:
    bos_token = None
    eos_token = "<eos>"
    eos_token_id = 9
    chat_template = None

    def get_vocab(self):
        return {}

    def encode(self, *_args, **_kwargs):
        return [1, 2, 3, 4]

    def apply_chat_template(self, *_args, **_kwargs):
        return "rendered prompt"


def _text_tokenizer():
    return _build_text_tokenizer(
        SimpleNamespace(
            tokenizer=_RawTokenizer(),
            detokenizer=_Detokenizer(),
            eos_token_ids=(9, 10),
        )
    )


def test_terminal_sparse_token_finalizes_buffered_segment_for_both_routes():
    class BufferedDetokenizer:
        def __init__(self):
            self.pending = ""
            self.visible = ""

        def add_token(self, token):
            self.pending = {5: "A"}[token]

        def finalize(self):
            self.visible += self.pending
            self.pending = ""

        @property
        def last_segment(self):
            segment = self.visible
            self.visible = ""
            return segment

    text, is_eos = _detokenize_sparse_token(
        BufferedDetokenizer(), 5, frozenset((9,)), terminal=True
    )
    assert text == "A"
    assert is_eos is False


class _ForwardContext:
    def __init__(self, model, cache):
        self.model = model
        self.cache = cache
        self.calls = []
        self.finish_count = 0

    def __call__(self, forward):
        self.calls.append((forward.phase, forward.model, forward.cache))
        return nullcontext()

    def finish(self):
        self.finish_count += 1


def _engine(monkeypatch, *, continuation_tokens=(6,), seed_token=5):
    engine = SimpleEngine("synthetic", specprefill_diagnostic_mode=True)
    engine._draft_model = object()
    tokenizer = _Tokenizer()
    target_model = SimpleNamespace()
    cache = [SimpleNamespace(offset=0)]
    context = _ForwardContext(target_model, cache)
    continuation_calls = []

    def stream_generate(
        *, prompt, prompt_cache, model_forward_context, max_tokens, **_kwargs
    ):
        continuation_calls.append(
            (prompt, prompt_cache, model_forward_context, max_tokens)
        )
        for index, token in enumerate(continuation_tokens):
            with model_forward_context(
                SimpleNamespace(
                    phase="decode",
                    model=target_model,
                    cache=prompt_cache,
                )
            ):
                pass
            yield SimpleNamespace(
                token=token,
                text="wrong-detokenizer-segment",
                finished=index == len(continuation_tokens) - 1,
                finish_reason=(
                    "length" if index == len(continuation_tokens) - 1 else None
                ),
            )

    engine._model = SimpleNamespace(
        model=target_model,
        tokenizer=tokenizer,
        stream_generate=stream_generate,
    )
    engine._supports_sparse_continuation = lambda _continuation: True
    profile_key = SimpleNamespace(adapter_id="qwen_dense")
    engine._admit_sparse_target = lambda _model: (profile_key, QWEN_DENSE_TARGET)
    monkeypatch.setattr(
        "mlx_lm.models.cache.make_prompt_cache", lambda *_args, **_kwargs: cache
    )
    monkeypatch.setattr(
        "vllm_mlx.specprefill.score_tokens",
        lambda *_args, **_kwargs: mx.ones((4,)),
    )
    monkeypatch.setattr(
        simple_module,
        "_sample_with_processors",
        lambda *_args, **_kwargs: (mx.array(seed_token), mx.array([0.0])),
    )
    plan = SimpleNamespace(selected_indices=(0, 3))
    result = SimpleNamespace(
        logits=mx.zeros((1, 1, 16)),
        telemetry=SimpleNamespace(selected_tokens=2, target_prefill_ms=1.0),
    )
    prepare_calls = []

    def prepare(**kwargs):
        prepare_calls.append(kwargs)
        return result, context, plan

    engine._prepare_sparse_target_prefill = prepare
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="sparse",
        coverage="selective",
        has_media=False,
        total_tokens=4,
    )
    return engine, telemetry, cache, context, continuation_calls, prepare_calls


@pytest.mark.anyio
async def test_sparse_seed_and_continuation_share_cache_context_and_detokenizer(
    monkeypatch,
):
    engine, telemetry, cache, context, continuation_calls, prepare_calls = _engine(
        monkeypatch
    )

    outputs = [
        output
        async for output in engine._stream_generate_specprefill(
            "prompt",
            [1, 2, 3, 4],
            max_tokens=2,
            temperature=0.0,
            top_p=1.0,
            telemetry=telemetry,
        )
    ]

    assert [output.new_text for output in outputs] == ["A", "B"]
    assert outputs[-1].text == "AB"
    assert continuation_calls[0][1] is cache
    assert continuation_calls[0][2] is context
    assert context.calls == [("decode", engine._model.model, cache)]
    assert context.finish_count == 1
    assert prepare_calls[0]["profile_key"].adapter_id == "qwen_dense"
    assert prepare_calls[0]["adapter"] is QWEN_DENSE_TARGET
    with pytest.raises(_SpecPrefillCancelled):
        prepare_calls[0]["cancel_check"]()


@pytest.mark.anyio
async def test_sparse_max_one_skips_continuation_and_finishes_context(monkeypatch):
    engine, telemetry, _cache, context, continuation_calls, _prepare_calls = _engine(
        monkeypatch
    )
    outputs = [
        output
        async for output in engine._stream_generate_specprefill(
            "prompt", [1, 2, 3, 4], 1, 0.0, 1.0, telemetry=telemetry
        )
    ]
    assert outputs[-1].text == "A"
    assert outputs[-1].finish_reason == "length"
    assert continuation_calls == []
    assert context.finish_count == 1


@pytest.mark.anyio
async def test_sparse_seed_eos_is_suppressed_and_never_continues(monkeypatch):
    engine, telemetry, _cache, context, continuation_calls, _prepare_calls = _engine(
        monkeypatch, seed_token=10
    )
    outputs = [
        output
        async for output in engine._stream_generate_specprefill(
            "prompt", [1, 2, 3, 4], 4, 0.0, 1.0, telemetry=telemetry
        )
    ]
    assert outputs[-1].text == ""
    assert outputs[-1].finish_reason == "stop"
    assert continuation_calls == []
    assert context.finish_count == 1


@pytest.mark.anyio
async def test_sparse_sampling_failure_is_terminal_and_finishes_context(monkeypatch):
    engine, telemetry, _cache, context, continuation_calls, _prepare_calls = _engine(
        monkeypatch
    )
    monkeypatch.setattr(
        simple_module,
        "_sample_with_processors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sample failed")),
    )
    with pytest.raises(RuntimeError, match="sample failed"):
        _ = [
            output
            async for output in engine._stream_generate_specprefill(
                "prompt", [1, 2, 3, 4], 2, 0.0, 1.0, telemetry=telemetry
            )
        ]
    assert continuation_calls == []
    assert context.finish_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize("failure_phase", ["scorer", "target_prefill"])
async def test_pre_sample_sparse_failures_discard_state_and_restart_dense(
    monkeypatch, failure_phase
):
    engine, telemetry, _cache, _context, _continuation, _prepare = _engine(monkeypatch)
    dense_calls = []

    def dense_stream(**kwargs):
        dense_calls.append(kwargs)
        yield SimpleNamespace(text="dense", finish_reason="stop")

    engine._model.stream_generate = dense_stream
    if failure_phase == "scorer":
        monkeypatch.setattr(
            "vllm_mlx.specprefill.score_tokens",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("scorer failed")
            ),
        )
    else:
        engine._prepare_sparse_target_prefill = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("target prefill failed")
        )

    outputs = [
        output
        async for output in engine._stream_generate_specprefill(
            "prompt", [1, 2, 3, 4], 2, 0.0, 1.0, telemetry=telemetry
        )
    ]

    assert [output.new_text for output in outputs] == ["dense"]
    assert outputs[-1].specprefill_fallback_reason == "sparse_execution_failed"
    assert len(dense_calls) == 1


@pytest.mark.anyio
async def test_logits_processor_failure_after_prefill_is_terminal_without_dense_replay(
    monkeypatch,
):
    engine, telemetry, _cache, context, _continuation, _prepare = _engine(monkeypatch)
    dense_calls = []
    original_stream = engine._model.stream_generate

    def counting_stream(**kwargs):
        dense_calls.append(kwargs)
        return original_stream(**kwargs)

    engine._model.stream_generate = counting_stream

    def failing_processor(_tokens, _logits):
        raise RuntimeError("processor failed")

    def sample_with_processors(tokens, logits, _sampler, processors):
        processors[0](tokens, logits)
        raise AssertionError("processor unexpectedly returned")

    monkeypatch.setattr(
        simple_module, "_sample_with_processors", sample_with_processors
    )

    with pytest.raises(RuntimeError, match="processor failed"):
        _ = [
            output
            async for output in engine._stream_generate_specprefill(
                "prompt",
                [1, 2, 3, 4],
                2,
                0.0,
                1.0,
                telemetry=telemetry,
                logits_processors=[failing_processor],
            )
        ]

    assert dense_calls == []
    assert context.finish_count == 1


@pytest.mark.anyio
async def test_sparse_stop_sequence_signals_worker_cancellation(monkeypatch):
    engine, telemetry, _cache, context, _calls, prepare_calls = _engine(monkeypatch)
    outputs = [
        output
        async for output in engine._stream_generate_specprefill(
            "prompt",
            [1, 2, 3, 4],
            4,
            0.0,
            1.0,
            stop=["B"],
            telemetry=telemetry,
        )
    ]
    assert outputs[-1].finish_reason == "stop"
    with pytest.raises(_SpecPrefillCancelled):
        prepare_calls[0]["cancel_check"]()
    assert context.finish_count == 1


@pytest.mark.anyio
async def test_terminal_stop_normalizes_late_worker_cancellation(monkeypatch):
    engine, telemetry, cache, context, _calls, prepare_calls = _engine(monkeypatch)

    def longer_stream(*, prompt_cache, model_forward_context, **_kwargs):
        yield SimpleNamespace(
            token=6, text="ignored", finished=False, finish_reason=None
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                prepare_calls[0]["cancel_check"]()
            except _SpecPrefillCancelled:
                break
            time.sleep(0.001)
        yield SimpleNamespace(
            token=7, text="ignored", finished=False, finish_reason=None
        )

    engine._model.stream_generate = longer_stream
    outputs = [
        output
        async for output in engine._stream_generate_specprefill(
            "prompt",
            [1, 2, 3, 4],
            4,
            0.0,
            1.0,
            stop=["B"],
            telemetry=telemetry,
        )
    ]
    assert outputs[-1].text == "AB"
    assert outputs[-1].finish_reason == "stop"
    assert context.finish_count == 1
    assert cache


@pytest.mark.anyio
async def test_mllm_text_sparse_route_uses_mlx_lm_wrapper_and_shared_lifecycle(
    monkeypatch,
):
    engine = SimpleEngine(
        "synthetic-gemma4", force_mllm=True, specprefill_diagnostic_mode=True
    )
    engine._draft_model = object()
    engine._supports_system_kv_cache = False
    tokenizer = _text_tokenizer()
    target_model = SimpleNamespace(mtp=None)
    cache = [SimpleNamespace(offset=0)]
    context = _ForwardContext(target_model, cache)
    engine._text_model = target_model
    engine._text_tokenizer = tokenizer
    engine._supports_sparse_continuation = lambda _continuation: True
    profile_key = SimpleNamespace(adapter_id="gemma4_dense")
    engine._admit_sparse_target = lambda _model: (profile_key, GEMMA4_DENSE_TARGET)
    plan = SimpleNamespace(selected_indices=(0, 3))
    result = SimpleNamespace(
        logits=mx.zeros((1, 1, 16)),
        telemetry=SimpleNamespace(selected_tokens=2, target_prefill_ms=1.0),
    )
    engine._prepare_sparse_target_prefill = lambda **_kwargs: (
        result,
        context,
        plan,
    )
    monkeypatch.setattr(
        "mlx_lm.models.cache.make_prompt_cache", lambda *_args, **_kwargs: cache
    )
    monkeypatch.setattr(
        "vllm_mlx.specprefill.score_tokens",
        lambda *_args, **_kwargs: mx.ones((4,)),
    )
    monkeypatch.setattr(
        simple_module,
        "_sample_with_processors",
        lambda *_args, **_kwargs: (mx.array(5), mx.array([0.0])),
    )
    continuation = []

    def stream_generate(model, received_tokenizer, prompt, **kwargs):
        continuation.append((model, received_tokenizer, prompt, kwargs))
        with kwargs["model_forward_context"](
            SimpleNamespace(phase="decode", model=model, cache=kwargs["prompt_cache"])
        ):
            pass
        yield SimpleNamespace(token=6, text="wrong", finish_reason="length")

    monkeypatch.setattr("mlx_lm.stream_generate", stream_generate)

    outputs = [
        output
        async for output in engine._stream_generate_text(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2,
            temperature=0.0,
            top_p=1.0,
            specprefill_policy="sparse",
            specprefill_coverage="selective",
        )
    ]

    assert outputs[-1].text == "AB"
    assert continuation[0][1] is tokenizer
    assert continuation[0][3]["prompt_cache"] is cache
    assert context.calls == [("decode", target_model, cache)]
    assert context.finish_count == 1
    assert tokenizer.eos_token_ids == {9, 10}

    max_one = [
        output
        async for output in engine._stream_generate_text(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            specprefill_policy="sparse",
            specprefill_coverage="selective",
        )
    ]
    assert max_one[-1].text == "A"
    assert max_one[-1].finish_reason == "length"
    assert len(continuation) == 1

    monkeypatch.setattr(
        simple_module,
        "_sample_with_processors",
        lambda *_args, **_kwargs: (mx.array(9), mx.array([0.0])),
    )
    max_one_eos = [
        output
        async for output in engine._stream_generate_text(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            specprefill_policy="sparse",
            specprefill_coverage="selective",
        )
    ]
    assert max_one_eos[-1].text == ""
    assert max_one_eos[-1].finish_reason == "stop"
    assert len(continuation) == 1
    assert context.finish_count == 3
