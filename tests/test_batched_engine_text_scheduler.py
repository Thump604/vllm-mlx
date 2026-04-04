# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine text-scheduler integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine


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
