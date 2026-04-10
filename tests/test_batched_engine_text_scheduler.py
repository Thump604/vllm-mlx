# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine text-scheduler integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.request import RequestOutput


class FakeTextScheduler:
    def __init__(self, outputs=None):
        self.calls = []
        self._outputs = outputs or []

    async def submit(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        for output in self._outputs:
            yield output

    async def stop(self):
        return None

    def get_stats(self):
        return {"running": False, "active_requests": 0}


def _make_engine() -> BatchedEngine:
    engine = BatchedEngine("dummy-model", force_mllm=True, mtp=True)
    engine._loaded = True
    engine._text_model = MagicMock()
    engine._text_scheduler_route_enabled = True
    return engine


def test_scheduler_gate_blocks_top_level_media():
    engine = _make_engine()
    engine._text_scheduler = FakeTextScheduler()

    messages = [{"role": "user", "content": "hello"}]
    assert (
        engine._should_use_text_scheduler(messages, images=["/tmp/image.png"]) is False
    )
    assert (
        engine._should_use_text_scheduler(messages, videos=["/tmp/video.mp4"]) is False
    )


def test_scheduler_gate_allows_system_prompts_tools_and_specprefill():
    engine = _make_engine()
    engine._text_scheduler = FakeTextScheduler()

    assert (
        engine._should_use_text_scheduler(
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hello"},
            ]
        )
        is True
    )
    assert (
        engine._should_use_text_scheduler(
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "noop"}}],
        )
        is True
    )

    engine._specprefill_enabled = True
    assert (
        engine._should_use_text_scheduler([{"role": "user", "content": "hello"}])
        is True
    )


def test_chat_uses_text_scheduler_when_canary_eligible():
    async def _run():
        engine = _make_engine()
        engine._text_scheduler = FakeTextScheduler(
            outputs=[
                GenerationOutput(
                    text="A",
                    new_text="A",
                    prompt_tokens=4,
                    completion_tokens=1,
                    finished=False,
                    finish_reason=None,
                ),
                GenerationOutput(
                    text="ABC",
                    new_text="BC",
                    prompt_tokens=4,
                    completion_tokens=3,
                    finished=True,
                    finish_reason="stop",
                ),
            ]
        )

        output = await engine.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=32,
            temperature=0.2,
            top_p=0.8,
        )

        assert output.text == "ABC"
        assert output.finish_reason == "stop"
        assert len(engine._text_scheduler.calls) == 1
        _, kwargs = engine._text_scheduler.calls[0]
        assert kwargs["max_tokens"] == 32
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.8

    asyncio.run(_run())


def test_chat_forwards_tools_to_text_scheduler():
    async def _run():
        engine = _make_engine()
        engine._text_scheduler = FakeTextScheduler(
            outputs=[
                GenerationOutput(
                    text="{}",
                    new_text="{}",
                    prompt_tokens=6,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )
            ]
        )
        tools = [{"type": "function", "function": {"name": "noop"}}]

        output = await engine.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=8,
            tools=tools,
        )

        assert output.text == "{}"
        assert len(engine._text_scheduler.calls) == 1
        _, kwargs = engine._text_scheduler.calls[0]
        assert kwargs["tools"] == tools

    asyncio.run(_run())


def test_chat_with_top_level_images_stays_off_text_scheduler():
    async def _run():
        engine = _make_engine()
        engine._text_scheduler = FakeTextScheduler()
        engine._apply_chat_template = MagicMock(return_value="prompt")  # type: ignore[method-assign]
        engine.generate = AsyncMock(  # type: ignore[method-assign]
            return_value=GenerationOutput(text="vision", finish_reason="stop")
        )

        output = await engine.chat(
            [{"role": "user", "content": "describe"}],
            images=["/tmp/image.png"],
        )

        assert output.text == "vision"
        assert engine._text_scheduler.calls == []
        engine.generate.assert_awaited_once()
        assert engine.generate.await_args.kwargs["images"] == ["/tmp/image.png"]

    asyncio.run(_run())


def test_hybrid_cache_text_chat_routes_through_serial_mllm_instance():
    """Hybrid-cache models (e.g. Gemma 4 RotatingKVCache) bypass MLLM
    continuous batching due to vllm-mlx #159 and route text-only chat
    through the serial _mllm_instance.chat path."""

    async def _run():
        engine = BatchedEngine("gemma-4-26b-a4b-it-5bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._has_hybrid_cache = True
        engine._mllm_instance = MagicMock()
        engine._mllm_instance.chat = MagicMock(
            return_value=MagicMock(
                text="OK",
                finish_reason="stop",
                prompt_tokens=21,
                completion_tokens=2,
            )
        )
        engine._text_scheduler = FakeTextScheduler()
        engine._text_scheduler_route_enabled = True
        engine._text_model = MagicMock()  # would normally enable text scheduler

        output = await engine.chat(
            [{"role": "user", "content": "Reply OK"}],
            max_tokens=64,
            temperature=1.0,
            top_p=0.95,
        )

        assert output.text == "OK"
        assert output.finish_reason == "stop"
        assert output.prompt_tokens == 21
        assert output.completion_tokens == 2
        engine._mllm_instance.chat.assert_called_once()
        # Hybrid cache route bypasses the text scheduler entirely.
        assert engine._text_scheduler.calls == []

    asyncio.run(_run())


def test_hybrid_cache_text_chat_forwards_kwargs_to_mllm_instance():
    """Hybrid-cache route must forward sampling and template kwargs."""

    async def _run():
        engine = BatchedEngine("gemma-4-26b-a4b-it-5bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._has_hybrid_cache = True
        engine._mllm_instance = MagicMock()
        engine._mllm_instance.chat = MagicMock(
            return_value=MagicMock(
                text="OK",
                finish_reason="stop",
                prompt_tokens=21,
                completion_tokens=2,
            )
        )

        await engine.chat(
            [{"role": "user", "content": "Reply OK"}],
            max_tokens=128,
            temperature=1.0,
            top_p=0.95,
            chat_template_kwargs={"enable_thinking": False},
            raw_output=True,
        )

        engine._mllm_instance.chat.assert_called_once()
        kwargs = engine._mllm_instance.chat.call_args.kwargs
        assert kwargs["max_tokens"] == 128
        assert kwargs["temperature"] == 1.0
        assert kwargs["top_p"] == 0.95
        assert kwargs.get("chat_template_kwargs") == {"enable_thinking": False}
        assert kwargs.get("raw_output") is True

    asyncio.run(_run())


def test_hybrid_cache_text_stream_chat_routes_through_serial_mllm_instance():
    """Hybrid-cache stream_chat path pumps the synchronous mlx_vlm
    stream_chat generator through an asyncio queue and yields
    GenerationOutput chunks back to the caller."""

    async def _run():
        engine = BatchedEngine("gemma-4-26b-a4b-it-5bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._has_hybrid_cache = True
        engine._mllm_instance = MagicMock()

        def _stream_chat(*args, **kwargs):
            yield MagicMock(
                text="O",
                prompt_tokens=15,
                completion_tokens=1,
                finish_reason=None,
            )
            yield MagicMock(
                text="OK",
                prompt_tokens=15,
                completion_tokens=2,
                finish_reason="stop",
            )

        engine._mllm_instance.stream_chat = MagicMock(side_effect=_stream_chat)
        engine._text_scheduler = FakeTextScheduler()
        engine._text_scheduler_route_enabled = True
        engine._text_model = MagicMock()

        outputs = []
        async for output in engine.stream_chat(
            [{"role": "user", "content": "Say OK"}],
            max_tokens=64,
            temperature=1.0,
            top_p=0.95,
        ):
            outputs.append((output.text, output.new_text, output.finished))

        assert outputs == [
            ("O", "O", False),
            ("OK", "K", True),
        ]
        engine._mllm_instance.stream_chat.assert_called_once()
        assert engine._text_scheduler.calls == []

    asyncio.run(_run())


def test_hybrid_cache_routes_only_text_only_requests():
    """Image requests on hybrid-cache models still go through MLLMScheduler."""

    async def _run():
        engine = BatchedEngine("gemma-4-26b-a4b-it-5bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._has_hybrid_cache = True
        engine._mllm_instance = MagicMock()
        engine._mllm_instance.chat = MagicMock()
        engine._apply_chat_template = MagicMock(return_value="prompt")  # type: ignore[method-assign]
        engine.generate = AsyncMock(  # type: ignore[method-assign]
            return_value=GenerationOutput(text="vision", finish_reason="stop")
        )

        output = await engine.chat(
            [{"role": "user", "content": "describe this"}],
            images=["/tmp/image.png"],
        )

        assert output.text == "vision"
        # Hybrid cache gate must NOT fire for image requests.
        engine._mllm_instance.chat.assert_not_called()
        engine.generate.assert_awaited_once()

    asyncio.run(_run())


def test_generate_forwards_full_sampling_params_to_mllm_scheduler():
    async def _run():
        engine = BatchedEngine("gemma-4-26B-A4B-it-6bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._mllm_scheduler = AsyncMock()
        engine._mllm_scheduler.generate.return_value = RequestOutput(
            request_id="req-1",
            output_text="gemma",
            prompt_tokens=9,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )

        output = await engine.generate(
            prompt="hello",
            max_tokens=32,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            stop=["DONE"],
        )

        assert output.text == "gemma"
        kwargs = engine._mllm_scheduler.generate.await_args.kwargs
        assert kwargs["top_k"] == 20
        assert kwargs["min_p"] == 0.0
        assert kwargs["presence_penalty"] == 1.5
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["stop"] == ["DONE"]

    asyncio.run(_run())


def test_stream_generate_forwards_full_sampling_params_to_mllm_scheduler():
    async def _run():
        engine = BatchedEngine("gemma-4-26B-A4B-it-6bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._mllm_scheduler = MagicMock()
        engine._mllm_scheduler.add_request_async = AsyncMock(return_value="req-1")

        async def _stream_outputs(_request_id):
            yield RequestOutput(
                request_id="req-1",
                output_text="A",
                new_text="A",
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
            )
            yield RequestOutput(
                request_id="req-1",
                output_text="AB",
                new_text="B",
                prompt_tokens=4,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
            )

        engine._mllm_scheduler.stream_outputs = _stream_outputs

        outputs = []
        async for output in engine.stream_generate(
            prompt="hello",
            max_tokens=32,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            stop=["DONE"],
        ):
            outputs.append(output.text)

        assert outputs == ["A", "AB"]
        kwargs = engine._mllm_scheduler.add_request_async.await_args.kwargs
        assert kwargs["top_k"] == 20
        assert kwargs["min_p"] == 0.0
        assert kwargs["presence_penalty"] == 1.5
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["stop"] == ["DONE"]

    asyncio.run(_run())


def test_get_stats_exposes_text_scheduler():
    engine = _make_engine()
    engine._text_scheduler = FakeTextScheduler()
    engine._mllm_scheduler = MagicMock()
    engine._mllm_scheduler.get_stats.return_value = {
        "metal_active_memory_gb": 1.5,
        "num_running": 2,
        "num_waiting": 3,
        "num_requests_processed": 4,
        "total_prompt_tokens": 5,
        "total_completion_tokens": 6,
    }

    stats = engine.get_stats()

    assert stats["text_scheduler_route_enabled"] is True
    assert stats["text_scheduler"]["active_requests"] == 0
    assert stats["metal_active_memory_gb"] == 1.5
    assert stats["num_running"] == 2
    assert stats["num_waiting"] == 3
    assert stats["num_requests_processed"] == 4
    assert stats["total_prompt_tokens"] == 5
    assert stats["total_completion_tokens"] == 6


def test_get_stats_survives_text_scheduler_failure():
    engine = _make_engine()
    engine._mllm_scheduler = None
    engine._text_scheduler = MagicMock()
    engine._text_scheduler.get_stats.side_effect = RuntimeError("boom")

    stats = engine.get_stats()

    assert stats["text_scheduler"]["error"] == "boom"


def test_get_stats_handles_quantized_kv_snapshot():
    """Regression: get_stats() must walk legacy and dict-wrapped snapshots.

    A QuantizedSDPACache snapshot entry is shaped
    ((packed, scales, biases), (packed, scales, biases)) — i.e. entry[0] is
    itself a 3-tuple of mlx arrays, not a single mlx array. The previous
    code path raised
        AttributeError: 'tuple' object has no attribute 'nbytes'
    on the very first request once a system prefix had been snapshotted on
    a model with --kv-quantize, which left /v1/status reporting
    'status: degraded' on Qwen 3.5 122B.

    Session 93's rotating-cache snapshot fix wraps new snapshot entries in
    {"state": ..., "meta_state": ...}; stats must still count only backing
    array bytes and ignore string/int metadata.
    """
    engine = _make_engine()
    engine._mllm_scheduler = None
    engine._text_scheduler = None

    class _Arr:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    # Two layers of QuantizedSDPACache state:
    # entry = (k_tuple, v_tuple) where each tuple is (packed, scales, biases)
    quant_layer_a = (
        (_Arr(1024), _Arr(64), _Arr(64)),
        (_Arr(1024), _Arr(64), _Arr(64)),
    )
    quant_layer_b = (
        (_Arr(2048), _Arr(128), _Arr(128)),
        (_Arr(2048), _Arr(128), _Arr(128)),
    )
    # Mix in a plain KVCache layer (single (keys, values) tuple) and an
    # ArraysCache layer (list of arrays) so the recursive helper covers all
    # three shapes _snapshot_system_kv() can store.
    plain_layer = (_Arr(4096), _Arr(4096))
    list_layer = [_Arr(8192), None, _Arr(8192)]

    engine._system_kv_snapshot = [
        {"state": quant_layer_a, "meta_state": ("0", "4", "2", "2")},
        {"state": quant_layer_b, "meta_state": ("0", "4", "4", "4")},
        {"state": plain_layer},
        {"state": list_layer, "meta_state": None},
    ]
    engine._system_kv_token_count = 128
    engine._system_kv_hash = "deadbeef"

    stats = engine.get_stats()

    assert "system_kv_cache" in stats
    assert stats["system_kv_cache"]["tokens"] == 128
    assert stats["system_kv_cache"]["hash"] == "deadbeef"
    # Total backing bytes:
    # quant_layer_a: 2 * (1024 + 64 + 64) = 2304
    # quant_layer_b: 2 * (2048 + 128 + 128) = 4608
    # plain_layer:   2 * 4096                = 8192
    # list_layer:    2 * 8192                = 16384
    # total                                  = 31488 bytes -> 0.0 MB rounded
    expected_bytes = 2304 + 4608 + 8192 + 16384
    assert stats["system_kv_cache"]["memory_mb"] == round(expected_bytes / 1e6, 1)
