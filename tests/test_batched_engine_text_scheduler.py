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


def test_gemma_text_chat_routes_through_text_scheduler():
    """Gemma text-only chat now flows through TextBatchScheduler."""
    async def _run():
        engine = BatchedEngine("gemma-4-26B-A4B-it-6bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_scheduler_route_enabled = True
        engine._text_scheduler = FakeTextScheduler(
            outputs=[
                GenerationOutput(
                    text="gemma",
                    new_text="gemma",
                    prompt_tokens=4,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )
            ]
        )

        output = await engine.chat([{"role": "user", "content": "hello"}], max_tokens=32)

        assert output.text == "gemma"
        assert len(engine._text_scheduler.calls) == 1

    asyncio.run(_run())


def test_gemma_text_chat_forwards_raw_output_kwarg():
    """Gemma text-only chat forwards raw_output to the text scheduler."""
    async def _run():
        raw_text = "<|channel>thought\nplan<channel|>Final answer"
        engine = BatchedEngine("gemma-4-26B-A4B-it-6bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_scheduler_route_enabled = True
        engine._text_scheduler = FakeTextScheduler(
            outputs=[
                GenerationOutput(
                    text=raw_text,
                    new_text=raw_text,
                    prompt_tokens=4,
                    completion_tokens=2,
                    finished=True,
                    finish_reason="stop",
                )
            ]
        )

        output = await engine.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=32,
            raw_output=True,
        )

        assert output.text == raw_text
        _, kwargs = engine._text_scheduler.calls[0]
        assert kwargs.get("raw_output") is True

    asyncio.run(_run())


def test_gemma_text_stream_chat_routes_through_text_scheduler():
    """Gemma text-only stream_chat flows through TextBatchScheduler."""
    async def _run():
        engine = BatchedEngine("gemma-4-26B-A4B-it-6bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_scheduler_route_enabled = True
        engine._text_scheduler = FakeTextScheduler(
            outputs=[
                GenerationOutput(
                    text="gem",
                    new_text="gem",
                    prompt_tokens=4,
                    completion_tokens=1,
                    finished=False,
                ),
                GenerationOutput(
                    text="gemma",
                    new_text="ma",
                    prompt_tokens=4,
                    completion_tokens=2,
                    finished=True,
                    finish_reason="stop",
                ),
            ]
        )

        outputs = []
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=32,
        ):
            outputs.append(output.text)

        assert outputs == ["gem", "gemma"]
        assert len(engine._text_scheduler.calls) == 1

    asyncio.run(_run())


def test_gemma_text_stream_chat_forwards_raw_output_kwarg():
    """Gemma text-only stream_chat forwards raw_output to the text scheduler."""
    async def _run():
        engine = BatchedEngine("gemma-4-26B-A4B-it-6bit", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_scheduler_route_enabled = True
        engine._text_scheduler = FakeTextScheduler(
            outputs=[
                GenerationOutput(
                    text="<|channel>thought\n",
                    new_text="<|channel>thought\n",
                    prompt_tokens=4,
                    completion_tokens=1,
                    finished=False,
                ),
                GenerationOutput(
                    text="<|channel>thought\nplan<channel|>Final answer",
                    new_text="plan<channel|>Final answer",
                    prompt_tokens=4,
                    completion_tokens=2,
                    finished=True,
                    finish_reason="stop",
                ),
            ]
        )

        outputs = []
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=32,
            raw_output=True,
        ):
            outputs.append(output.text)

        assert outputs == [
            "<|channel>thought\n",
            "<|channel>thought\nplan<channel|>Final answer",
        ]
        _, kwargs = engine._text_scheduler.calls[0]
        assert kwargs.get("raw_output") is True

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
