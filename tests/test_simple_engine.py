# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest


class TestSimpleEngineConcurrency:
    """Test SimpleEngine lock behavior with concurrent requests."""

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks concurrent calls."""
        model = MagicMock()
        model.model = model
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            # Simulate some work
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)
        return model

    @pytest.fixture
    def mock_llm_model(self):
        """Create a mock LLM model."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def chat_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.chat = MagicMock(side_effect=chat_side_effect)
        return model

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_generate(self, mock_model):
        """Test that the lock prevents concurrent generate calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            tasks = [
                engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_chat(self, mock_llm_model):
        """Test that the lock prevents concurrent chat calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True

            # Launch multiple concurrent chat calls
            tasks = [
                engine.chat(
                    messages=[{"role": "user", "content": f"test {i}"}], max_tokens=10
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_llm_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_llm_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.anyio
    async def test_chat_with_tools_aggregates_streaming_path(self, mock_llm_model):
        """Tool-enabled non-stream chat should use the streaming path."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="partial",
                tokens=[],
                prompt_tokens=11,
                completion_tokens=1,
                finish_reason=None,
                finished=False,
            )
            yield MagicMock(
                text="<|im_end|><tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"pwd\"}}</tool_call>",
                tokens=[],
                prompt_tokens=11,
                completion_tokens=4,
                finish_reason="stop",
                finished=True,
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

            output = await engine.chat(
                messages=[{"role": "user", "content": "run pwd"}],
                max_tokens=16,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            assert output.text.startswith("<tool_call>")
            assert output.tokens == []
            assert output.prompt_tokens == 11
            assert output.completion_tokens == 4
            assert output.finish_reason == "stop"
            mock_llm_model.chat.assert_not_called()

    @pytest.mark.anyio
    async def test_lock_serializes_stream_generate(self, mock_model):
        """Test that stream_generate uses the same lock as other methods."""
        from vllm_mlx.engine.simple import SimpleEngine

        def stream_generate_side_effect(**kwargs):
            # Yield a few chunks
            for i in range(3):
                chunk = MagicMock()
                chunk.text = f"chunk{i}"
                chunk.prompt_tokens = 5
                chunk.finished = i == 2
                chunk.finish_reason = "stop" if i == 2 else None
                yield chunk

        mock_model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Test that stream_generate acquires the lock
            # by checking if it blocks when lock is already held
            lock_acquired = asyncio.Event()
            stream_started = asyncio.Event()

            async def hold_lock():
                async with engine._generation_lock:
                    lock_acquired.set()
                    # Wait until stream tries to start
                    await asyncio.sleep(0.1)

            async def try_stream():
                # Wait for lock to be held
                await lock_acquired.wait()
                stream_started.set()
                # This should block until hold_lock releases
                result = []
                async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                    result.append(chunk)
                return result

            # Start both tasks
            hold_task = asyncio.create_task(hold_lock())
            stream_task = asyncio.create_task(try_stream())

            # Wait a bit for stream to try to acquire lock
            await asyncio.sleep(0.05)

            # Stream should have started but be blocked on the lock
            assert stream_started.is_set(), "Stream should have attempted to start"

            # Stream task should not be done yet (blocked on lock)
            assert not stream_task.done(), "Stream should be blocked waiting for lock"

            # Let hold_lock finish
            await hold_task

            # Now stream should complete
            result = await stream_task
            assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"

    @pytest.mark.anyio
    async def test_engine_initialization_creates_lock(self):
        """Test that SimpleEngine creates a lock on initialization."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

            assert hasattr(engine, "_generation_lock")
            assert isinstance(engine._generation_lock, asyncio.Lock)

    @pytest.mark.anyio
    async def test_requests_complete_in_order(self, mock_model):
        """Test that concurrent requests complete (may be in any order due to lock)."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            results = await asyncio.gather(
                *[
                    engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                    for i in range(3)
                ]
            )

            # All requests should complete
            assert len(results) == 3
            for result in results:
                assert result.text == "test response"

    @pytest.mark.anyio
    async def test_specprefill_phase4_uses_stream_generate(self):
        """SpecPrefill decode must route through mlx_lm.stream_generate, not model()."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        class DummyModel:
            def __init__(self):
                self.model = self
                self.tokenizer = MagicMock()
                self.stream_generate = MagicMock()

            def __call__(self, *args, **kwargs):
                raise AssertionError(
                    "manual model() decode should not run in Phase 4"
                )

        model = DummyModel()
        model.tokenizer.bos_token = None
        model.tokenizer.eos_token_id = 0
        model.tokenizer.encode.return_value = [11, 12, 13, 14]
        model.tokenizer.decode.side_effect = lambda toks: "".join(
            chr(64 + tok) for tok in toks
        )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine(
                "test-model",
                specprefill_enabled=True,
                specprefill_threshold=1,
                specprefill_keep_pct=0.5,
            )
            engine._model = model
            engine._loaded = True
            engine._draft_model = MagicMock()

            def fake_sampler(logits):
                return mx.array([1], dtype=mx.int32)

            def fake_stream_generate(
                _model,
                _tokenizer,
                *,
                prompt,
                max_tokens,
                sampler,
                prompt_cache,
                prefill_step_size,
            ):
                assert _model is model
                assert _tokenizer is model.tokenizer
                assert prompt.tolist() == [1]
                assert max_tokens == 2
                assert sampler is fake_sampler
                assert prompt_cache == ["cache"]
                assert prefill_step_size == engine._prefill_step_size
                yield SimpleNamespace(text="B", finish_reason="stop")

            with (
                patch("mlx_lm.sample_utils.make_sampler", return_value=fake_sampler),
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=["cache"]),
                patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
                patch("vllm_mlx.specprefill.score_tokens", return_value=mx.array([0.5])),
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
                async for chunk in engine.stream_generate(
                    prompt="hello",
                    max_tokens=3,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    chunks.append(chunk.new_text)

        assert chunks == ["A", "B"]
