# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine generate() output."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


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

    @pytest.mark.anyio
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

    @pytest.mark.anyio
    async def test_tokens_field_empty_when_no_tokens_generated(self):
        """tokens should be an empty list when output_token_ids is empty."""
        engine = self._make_engine()
        mock_output = self._make_mock_request_output(output_token_ids=[])

        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=mock_output)
        engine._engine = mock_engine

        result = await engine.generate(prompt="test", max_tokens=10)

        assert result.tokens == []

    @pytest.mark.anyio
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

    def test_accumulate_streamed_text_keeps_last_nonempty_text_on_terminal_stop_chunk(
        self,
    ):
        """Terminal empty chunks must not wipe already accumulated text."""
        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.engine.base import GenerationOutput

        accumulated = ""
        accumulated = BatchedEngine._accumulate_streamed_text(
            accumulated,
            GenerationOutput(
                text="NX-4271-",
                prompt_tokens=10,
                completion_tokens=8,
                new_text="NX-4271-",
                finished=False,
                finish_reason=None,
            ),
        )
        accumulated = BatchedEngine._accumulate_streamed_text(
            accumulated,
            GenerationOutput(
                text="NEMO",
                prompt_tokens=10,
                completion_tokens=11,
                new_text="NEMO",
                finished=False,
                finish_reason=None,
            ),
        )
        accumulated = BatchedEngine._accumulate_streamed_text(
            accumulated,
            GenerationOutput(
                text="",
                prompt_tokens=10,
                completion_tokens=11,
                new_text="",
                finished=True,
                finish_reason="stop",
            ),
        )

        assert accumulated == "NX-4271-NEMO"

    def test_accumulate_streamed_text_treats_nonprefix_text_as_delta(self):
        """Non-prefix text without new_text should be appended, not replace state."""
        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.engine.base import GenerationOutput

        accumulated = "NX-4271-"
        accumulated = BatchedEngine._accumulate_streamed_text(
            accumulated,
            GenerationOutput(
                text="NEMO",
                prompt_tokens=10,
                completion_tokens=11,
                new_text="",
                finished=False,
                finish_reason=None,
            ),
        )

        assert accumulated == "NX-4271-NEMO"
