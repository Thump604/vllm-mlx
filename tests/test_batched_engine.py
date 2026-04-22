# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine generate() output."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBatchedEngineGenerate:
    """Test BatchedEngine.generate() output fields."""

    def _make_engine(self):
        """Create a BatchedEngine instance with loading bypassed."""
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")

        engine._loaded = True
        engine._is_mllm = False
        return engine

    def _make_mock_request_output(
        self,
        output_text="Paris",
        output_token_ids=None,
        prompt_tokens=10,
        completion_tokens=3,
        finish_reason="stop",
    ):
        """Build a mock RequestOutput (as returned by AsyncEngineCore)."""
        mock = MagicMock()
        mock.output_text = output_text
        mock.output_token_ids = (
            output_token_ids if output_token_ids is not None else [3681, 374, 279]
        )
        mock.prompt_tokens = prompt_tokens
        mock.completion_tokens = completion_tokens
        mock.finish_reason = finish_reason
        return mock

    @pytest.mark.asyncio
    async def test_tokens_field_is_populated(self):
        """tokens should contain the output token IDs from AsyncEngineCore."""
        engine = self._make_engine()
        token_ids = [3681, 374, 279]
        mock_output = self._make_mock_request_output(output_token_ids=token_ids)

        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=mock_output)
        engine._engine = mock_engine

        result = await engine.generate(
            prompt="What is the capital of France?", max_tokens=10
        )

        assert result.tokens == token_ids

    @pytest.mark.asyncio
    async def test_tokens_field_empty_when_no_tokens_generated(self):
        """tokens should be an empty list when output_token_ids is empty."""
        engine = self._make_engine()
        mock_output = self._make_mock_request_output(output_token_ids=[])

        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=mock_output)
        engine._engine = mock_engine

        result = await engine.generate(prompt="test", max_tokens=10)

        assert result.tokens == []

    @pytest.mark.asyncio
    async def test_other_output_fields_still_populated(self):
        """Existing fields (text, prompt_tokens, etc.) must remain correct."""
        engine = self._make_engine()
        mock_output = self._make_mock_request_output(
            output_text="Paris",
            output_token_ids=[3681],
            prompt_tokens=7,
            completion_tokens=1,
            finish_reason="stop",
        )

        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=mock_output)
        engine._engine = mock_engine

        result = await engine.generate(prompt="Capital of France?", max_tokens=5)

        assert result.text == "Paris"
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 1
        assert result.finish_reason == "stop"


class TestBatchedEngineMLLMCustomLogitsProcessors:
    def _make_engine(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            engine = BatchedEngine("test-mllm")

        engine._loaded = True
        engine._is_mllm = True
        engine._mllm_scheduler = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_generate_forwards_custom_logits_processors(self):
        engine = self._make_engine()
        logits_processors = [MagicMock()]

        mock_output = MagicMock()
        mock_output.output_text = "Hello"
        mock_output.output_token_ids = [42]
        mock_output.prompt_tokens = 10
        mock_output.completion_tokens = 1
        mock_output.finish_reason = "stop"
        engine._mllm_scheduler.generate = AsyncMock(return_value=mock_output)

        result = await engine.generate(
            prompt="test",
            max_tokens=5,
            logits_processors=logits_processors,
        )

        assert result.text == "Hello"
        kwargs = engine._mllm_scheduler.generate.await_args.kwargs
        assert kwargs["logits_processors"] == logits_processors

    @pytest.mark.asyncio
    async def test_stream_generate_forwards_custom_logits_processors(self):
        engine = self._make_engine()
        logits_processors = [MagicMock()]

        async def fake_outputs(_request_id):
            yield MagicMock(
                output_text="Hello",
                new_text="Hello",
                prompt_tokens=10,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )

        engine._mllm_scheduler.add_request_async = AsyncMock(return_value="req-1")
        engine._mllm_scheduler.stream_outputs = fake_outputs

        outputs = [
            output
            async for output in engine.stream_generate(
                prompt="test",
                max_tokens=5,
                logits_processors=logits_processors,
            )
        ]

        assert len(outputs) == 1
        kwargs = engine._mllm_scheduler.add_request_async.await_args.kwargs
        assert kwargs["logits_processors"] == logits_processors
