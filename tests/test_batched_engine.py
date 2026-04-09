# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine generate() output."""

from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
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

    @pytest.mark.anyio
    async def test_stream_chat_text_model_specprefill_phase4_uses_pipelined_handoff(
        self,
    ):
        """SpecPrefill MTP decode must re-enter mlx_lm.stream_generate cleanly."""
        from types import SimpleNamespace

        from vllm_mlx.engine.batched import BatchedEngine

        class DummyTextModel:
            def __init__(self):
                self.mtp_forward = object()

            def make_mtp_cache(self):
                return ["mtp-cache"]

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            engine = BatchedEngine(
                "test-model",
                force_mllm=True,
                mtp=True,
                specprefill_enabled=True,
                specprefill_threshold=1,
                specprefill_keep_pct=0.5,
            )

        engine._loaded = True
        engine._text_model = DummyTextModel()
        engine._draft_model = MagicMock()
        engine._text_tokenizer = MagicMock()
        engine._text_tokenizer.apply_chat_template.return_value = "hello"
        engine._text_tokenizer.encode.return_value = [11, 12, 13, 14]
        engine._text_tokenizer.decode.side_effect = lambda toks: "".join(
            chr(64 + tok) for tok in toks
        )
        engine._text_tokenizer.eos_token_id = 0

        def fake_sampler(_logits):
            return mx.array([1], dtype=mx.int32)

        def fake_stream_generate(
            _model,
            _tokenizer,
            *,
            prompt,
            max_tokens,
            sampler,
            mtp,
            prompt_cache,
        ):
            assert _model is engine._text_model
            assert _tokenizer is engine._text_tokenizer
            assert prompt.tolist() == [1]
            assert max_tokens == 2
            assert sampler is fake_sampler
            assert mtp is True
            assert prompt_cache == ["cache", "mtp-cache"]
            yield SimpleNamespace(text="B", finish_reason="stop")

        with (
            patch("mlx_lm.sample_utils.make_sampler", return_value=fake_sampler),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=["cache"]),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch(
                "vllm_mlx.specprefill.score_tokens",
                return_value=mx.array([0.5]),
            ),
            patch(
                "vllm_mlx.specprefill.select_chunks",
                return_value=mx.array([0], dtype=mx.int32),
            ),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 1, 8)),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
        ):
            chunks = []
            async for chunk in engine._stream_chat_text_model(
                [{"role": "user", "content": "hello"}],
                max_tokens=3,
                temperature=0.7,
                top_p=0.9,
            ):
                chunks.append(chunk.new_text)

        assert chunks == ["A", "B"]
