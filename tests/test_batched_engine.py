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


class TestMLLMModelWrapper:
    """Test MLLM model wrapper behavior for multimodal families."""

    def test_injects_pixel_values_for_gemma4_text_only_calls(self):
        from vllm_mlx.engine.batched import MLLMModelWrapper

        recorded = {}

        class DummyModel:
            model_type = "gemma4"

            def __call__(self, *args, **kwargs):
                recorded["kwargs"] = kwargs
                return mx.zeros((1, 1, 10))

        wrapper = MLLMModelWrapper(DummyModel())
        wrapper(mx.zeros((1, 1, 4)))

        assert recorded["kwargs"]["pixel_values"] is None

    def test_does_not_override_explicit_pixel_values(self):
        from vllm_mlx.engine.batched import MLLMModelWrapper

        recorded = {}
        pixel_values = mx.ones((1, 3, 4, 4))

        class DummyModel:
            model_type = "gemma4"

            def __call__(self, *args, **kwargs):
                recorded["kwargs"] = kwargs
                return mx.zeros((1, 1, 10))

        wrapper = MLLMModelWrapper(DummyModel())
        wrapper(mx.zeros((1, 1, 4)), pixel_values=pixel_values)

        assert recorded["kwargs"]["pixel_values"] is pixel_values

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

    def test_snapshot_restore_round_trips_rotating_kv_meta_state(self):
        """RotatingKVCache snapshots must preserve offset/_idx via meta_state."""
        from mlx_lm.models.cache import RotatingKVCache

        from vllm_mlx.engine.batched import BatchedEngine

        cache = RotatingKVCache(max_size=4, keep=0)
        prefix = mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1)
        cache.update_and_fetch(prefix, prefix)

        snapshot = BatchedEngine._snapshot_cache_entry(cache)
        restored = RotatingKVCache(max_size=16, keep=2)
        BatchedEngine._restore_cache_snapshot_entry(restored, snapshot)

        assert restored.meta_state == cache.meta_state

        step = mx.array([[[[99.0]]]], dtype=mx.float32)
        cache_k, cache_v = cache.update_and_fetch(step, step)
        restored_k, restored_v = restored.update_and_fetch(step, step)

        assert restored.meta_state == cache.meta_state
        assert mx.array_equal(cache_k, restored_k)
        assert mx.array_equal(cache_v, restored_v)

    def test_extract_system_kv_prefix_splits_chatml_prompt(self):
        """System prefix extraction should split at the first user turn marker."""
        from vllm_mlx.engine.batched import BatchedEngine

        prompt = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )

        system_prefix, suffix, prefix_hash = BatchedEngine._extract_system_kv_prefix(
            prompt
        )

        assert system_prefix == "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        assert (
            suffix
            == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        assert prefix_hash is not None and len(prefix_hash) == 16

    def test_extract_system_kv_prefix_returns_full_prompt_without_user_marker(self):
        """Prompts without a supported user marker should not claim a prefix hit."""
        from vllm_mlx.engine.batched import BatchedEngine

        prompt = "Plain completion prompt with no chat markers."

        system_prefix, suffix, prefix_hash = BatchedEngine._extract_system_kv_prefix(
            prompt
        )

        assert system_prefix is None
        assert suffix == prompt
        assert prefix_hash is None

    def test_extract_system_kv_prefix_supports_turn_marker(self):
        """Gemma-style turn markers should also split at the first user turn."""
        from vllm_mlx.engine.batched import BatchedEngine

        prompt = (
            "<|turn|>system\nYou are helpful.<|endturn|>\n"
            "<|turn>user\nHello<|endturn|>\n"
            "<|turn>assistant\n"
        )

        system_prefix, suffix, prefix_hash = BatchedEngine._extract_system_kv_prefix(
            prompt
        )

        assert system_prefix == "<|turn|>system\nYou are helpful.<|endturn|>\n"
        assert suffix == "<|turn>user\nHello<|endturn|>\n<|turn>assistant\n"
        assert prefix_hash is not None and len(prefix_hash) == 16

    def test_extract_system_kv_prefix_hash_stable_across_suffix_changes(self):
        """Changing only the user suffix should keep the system prefix hash stable."""
        from vllm_mlx.engine.batched import BatchedEngine

        prompt_a = (
            "<|im_start|>system\nRules.<|im_end|>\n"
            "<|im_start|>user\nQuestion A<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt_b = (
            "<|im_start|>system\nRules.<|im_end|>\n"
            "<|im_start|>user\nQuestion B<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        prefix_a, suffix_a, hash_a = BatchedEngine._extract_system_kv_prefix(prompt_a)
        prefix_b, suffix_b, hash_b = BatchedEngine._extract_system_kv_prefix(prompt_b)

        assert prefix_a == prefix_b
        assert hash_a == hash_b
        assert suffix_a != suffix_b
