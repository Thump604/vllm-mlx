# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

pytestmark = pytest.mark.anyio


class TestSimpleEngineConcurrency:
    """Test SimpleEngine lock behavior with concurrent requests."""

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks concurrent calls."""
        model = MagicMock()
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

    async def test_chat_with_tools_aggregates_streaming_path(self, mock_llm_model):
        """Tool-enabled non-stream chat should use the streaming path."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="partial",
                tokens=[1],
                prompt_tokens=11,
                completion_tokens=1,
                finish_reason=None,
                finished=False,
            )
            yield MagicMock(
                text='<|im_end|><tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                tokens=[7, 8, 9],
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

            assert output.text == '{"name":"bash","arguments":{"command":"pwd"}}'
            assert output.tokens == [7, 8, 9]
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

    def test_get_stats_handles_quantized_system_kv_snapshot(self):
        """Quantized KV snapshots expose nested tuples, not plain arrays."""
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeArray:
            def __init__(self, nbytes):
                self.nbytes = nbytes

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._system_kv_snapshot = [
                (
                    (
                        FakeArray(1_000_000),
                        FakeArray(2_000_000),
                        FakeArray(3_000_000),
                    ),
                    (
                        FakeArray(4_000_000),
                        FakeArray(5_000_000),
                        FakeArray(6_000_000),
                    ),
                )
            ]
            engine._system_kv_token_count = 350
            engine._system_kv_hash = "abc123"

            stats = engine.get_stats()
            cache_stats = engine.get_cache_stats()

        assert stats["system_kv_cache"] == {
            "tokens": 350,
            "hash": "abc123",
            "memory_mb": 21.0,
        }
        assert cache_stats == {"system_kv_cache": stats["system_kv_cache"]}

    def test_clear_runtime_caches_clears_system_kv_snapshot(self):
        """Benchmark cache resets must clear SimpleEngine's system KV snapshot."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._system_kv_snapshot = [("k", "v")]
            engine._system_kv_token_count = 350
            engine._system_kv_hash = "abc123"

            cleared = engine.clear_runtime_caches()
            stats = engine.get_stats()

        assert cleared == {"system_kv_cache": True}
        assert "system_kv_cache" not in stats
        assert engine._system_kv_snapshot is None
        assert engine._system_kv_token_count == 0
        assert engine._system_kv_hash is None

    async def test_active_request_status_and_abort(self):
        """SimpleEngine exposes and cancels active streaming requests by id."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            abort_event = threading.Event()
            engine._active_requests["req-1"] = {
                "request_id": "req-1",
                "status": "running",
                "completion_tokens": 7,
            }
            engine._abort_events["req-1"] = abort_event

            stats = engine.get_stats()
            aborted = await engine.abort_request("req-1")
            missing = await engine.abort_request("missing")

        assert stats["running"] is True
        assert stats["num_running"] == 1
        assert stats["requests"][0]["request_id"] == "req-1"
        assert aborted is True
        assert abort_event.is_set()
        assert engine._active_requests["req-1"]["status"] == "cancelling"
        assert missing is False

    def test_seed_logits_processors_prepends_prompt_tokens(self):
        """Continuation decode processors must see the original prompt prefix."""
        from vllm_mlx.engine.simple import _seed_logits_processors

        seen = {}

        def processor(tokens, logits):
            seen["tokens"] = tokens.tolist()
            return logits

        seeded = _seed_logits_processors(
            mx.array([10, 11], dtype=mx.uint32), [processor]
        )

        logits = mx.zeros((1, 8), dtype=mx.float32)
        seeded[0](mx.array([12, 13], dtype=mx.uint32), logits)

        assert seen["tokens"] == [10, 11, 12, 13]

    @pytest.mark.anyio
    async def test_specprefill_success_preserves_mtp_path(self):
        """Successful sparse prefill should continue through the normal MTP path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_make_sampler(**kwargs):
            captured["sampler_kwargs"] = kwargs

            def _sample(_logprobs):
                return mx.array([17], dtype=mx.uint32)

            return _sample

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured["prompt"] = prompt.tolist()
            captured["kwargs"] = kwargs
            yield SimpleNamespace(text="B", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 99
        tokenizer.encode.return_value = [5, 6, 7]
        tokenizer.decode.side_effect = lambda ids: "".join(
            {17: "A", 99: ""}.get(tok, f"<{tok}>") for tok in ids
        )

        text_model = MagicMock()
        text_model.mtp = object()
        text_model.make_mtp_cache.return_value = ["mtp-cache"]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
            specprefill_threshold=1,
            specprefill_backbone_pct=0.1,
        )
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        engine._draft_model = object()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=["backbone-cache"],
            ),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch(
                "vllm_mlx.specprefill.score_tokens",
                return_value=mx.array([1.0, 0.9, 0.8], dtype=mx.float32),
            ),
            patch("vllm_mlx.specprefill.select_chunks") as mock_select_chunks,
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 3, 32), dtype=mx.float32),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
        ):
            mock_select_chunks.return_value = mx.array([0, 1, 2], dtype=mx.int32)
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=4,
                    temperature=0.6,
                    top_p=0.95,
                )
            ]

        assert [chunk.new_text for chunk in outputs] == ["A", "B"]
        assert captured["sampler_kwargs"] == {
            "temp": 0.6,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
        }
        assert captured["prompt"] == [17]
        assert captured["kwargs"]["mtp"] is True
        assert "draft_model" not in captured["kwargs"]
        assert captured["kwargs"]["prompt_cache"] == ["backbone-cache", "mtp-cache"]
        assert captured["kwargs"]["max_tokens"] == 3
        assert captured["kwargs"]["logits_processors"] is None
        mock_select_chunks.assert_called_once()
        assert mock_select_chunks.call_args.kwargs["backbone_pct"] == pytest.approx(0.1)

    @pytest.mark.anyio
    async def test_run_blocking_serialized_rebinds_worker_generation_streams(self):
        """Worker-thread MLX generation should get fresh thread-local streams."""
        import importlib

        from vllm_mlx.engine.simple import SimpleEngine

        mlx_lm_generate = importlib.import_module("mlx_lm.generate")
        sentinel_stream = object()

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.mlx_streams.mx.default_device", return_value="gpu"),
            patch(
                "vllm_mlx.mlx_streams.mx.new_stream",
                return_value=object(),
            ),
            patch(
                "vllm_mlx.mlx_streams.mx.new_thread_local_stream",
                return_value=sentinel_stream,
            ),
            patch("vllm_mlx.mlx_streams.mx.set_default_stream"),
        ):
            engine = SimpleEngine("test-model")
            observed = await engine._run_blocking_serialized(
                lambda: mlx_lm_generate.generation_stream
            )

        assert observed is sentinel_stream

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
    async def test_start_keeps_text_routing_for_mllm_without_mtp(self):
        """MLLM text-only routing must stay available when MTP is disabled."""
        from vllm_mlx.engine.simple import SimpleEngine

        text_model = MagicMock()
        text_model.mtp = None
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 42

        mock_mllm = MagicMock()
        mock_mllm.model = MagicMock()
        mock_mllm.get_tokenizer.return_value = tokenizer

        with (
            patch(
                "vllm_mlx.models.mllm.MLXMultimodalLM",
                return_value=mock_mllm,
            ),
            patch(
                "vllm_mlx.text_model_from_vlm.build_text_model",
                return_value=text_model,
            ),
        ):
            engine = SimpleEngine("qwen3.6-27b", force_mllm=True, mtp=False)
            await engine.start()

        assert engine._text_model is text_model
        assert engine._text_tokenizer is tokenizer

    @pytest.mark.anyio
    async def test_mllm_text_only_routes_without_mtp(self):
        """Text-only MLLM requests should use the TextModel route even without MTP."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_generate_text(*args, **kwargs):
            yield MagicMock(
                text="Hello",
                new_text="Hello",
                prompt_tokens=5,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._model = MagicMock()
        engine._stream_generate_text = fake_stream_generate_text  # type: ignore[method-assign]

        outputs = []
        async for chunk in engine.stream_chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
        ):
            outputs.append(chunk)

        assert len(outputs) == 1
        assert outputs[0].text == "Hello"
        engine._model.stream_chat.assert_not_called()

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_routes_without_mtp(self):
        """Non-stream text-only MLLM chat must aggregate the TextModel stream route."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="Hello",
                tokens=[1],
                prompt_tokens=5,
                completion_tokens=1,
                finish_reason="stop",
                finished=True,
            )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._model = MagicMock()
        engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

        output = await engine.chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
        )

        assert output.text == "Hello"
        assert output.tokens == [1]
        assert output.prompt_tokens == 5
        assert output.completion_tokens == 1
        assert output.finish_reason == "stop"
        engine._model.chat.assert_not_called()

    @pytest.mark.anyio
    async def test_stream_generate_text_omits_mtp_when_disabled(self):
        """The text route must not force mlx-lm MTP when the engine stage disabled it."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs

    @pytest.mark.anyio
    async def test_stream_generate_text_forwards_logits_processors_and_sampler_args(
        self,
    ):
        """Text routing must preserve request-local decoding controls."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        sampler_calls = []
        penalty_calls = []
        user_processor = MagicMock()
        penalty_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        def fake_make_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return MagicMock()

        def fake_make_logits_processors(**kwargs):
            penalty_calls.append(kwargs)
            return [penalty_processor]

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer

        with (
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                side_effect=fake_make_logits_processors,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    min_p=0.1,
                    presence_penalty=1.5,
                    repetition_penalty=1.2,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert sampler_calls == [{"temp": 0.3, "top_p": 0.8, "top_k": 40, "min_p": 0.1}]
        assert penalty_calls == [{"repetition_penalty": 1.2, "presence_penalty": 1.5}]
        assert captured_kwargs["logits_processors"] == [
            user_processor,
            penalty_processor,
        ]

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_when_logits_processors_active(
        self,
    ):
        """Custom logits processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        user_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is user_processor

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_for_thinking_processor(
        self,
    ):
        """Thinking-budget processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        thinking_proc = MagicMock()

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[thinking_proc],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is thinking_proc

    @pytest.mark.anyio
    async def test_stream_generate_text_passes_num_draft_tokens(self):
        """Text routing should forward configured MTP draft depth."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                )
            ]

        assert outputs[-1].text == "Hello"
        assert captured_kwargs["mtp"] is True
        assert captured_kwargs["num_draft_tokens"] == 4

    @pytest.mark.anyio
    async def test_stream_generate_text_reenables_mtp_after_retired_processor_when_enabled(
        self,
    ):
        """Retired thinking processor handoff is an explicit opt-in path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            if len(calls) == 1:
                processor.is_retired = True
                yield SimpleNamespace(token=11, text="Hello", finish_reason=None)
            else:
                yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [11]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 2
        assert "mtp" not in calls[0]
        assert calls[0]["logits_processors"][0] is processor
        assert calls[1]["mtp"] is True
        assert calls[1]["num_draft_tokens"] == 4
        assert "logits_processors" not in calls[1]

    @pytest.mark.anyio
    async def test_stream_generate_text_keeps_mtp_for_speculation_safe_processor(self):
        """Budget-only thinking processors should not force native MTP off."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class SafeThinkingProcessor:
            speculation_safe = True
            is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = SafeThinkingProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            yield SimpleNamespace(token=11, text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [11]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert len(calls) == 1
        assert calls[0]["mtp"] is True
        assert calls[0]["num_draft_tokens"] == 4
        assert calls[0]["logits_processors"][0] is processor

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_for_retireable_thinking_processor_without_explicit_safety(
        self,
    ):
        """Retireable thinking processors are fail-closed unless they opt into speculation."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class ThinkingAwareLogitsProcessor:
            is_retired = False
            _inner = None

            def __call__(self, tokens, logits):
                return logits

        processor = ThinkingAwareLogitsProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            yield SimpleNamespace(token=11, text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [11]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert len(calls) == 1
        assert "mtp" not in calls[0]
        assert "num_draft_tokens" not in calls[0]
        assert calls[0]["logits_processors"][0] is processor

    @pytest.mark.anyio
    async def test_stream_generate_text_specprefill_reenables_mtp_after_retirement(
        self,
    ):
        """SpecPrefill retirement-to-MTP continuation is explicit opt-in."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [1, 2, 3, 4]
        tokenizer.decode.side_effect = lambda toks: "Hello" if toks == [11] else ""

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
        )
        engine._loaded = True
        engine._draft_model = MagicMock()
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        def fake_sample(tokens, logits, sampler, logits_processors):
            processor.is_retired = True
            return mx.array(11, dtype=mx.uint32), logits

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch(
                "vllm_mlx.specprefill.score_tokens", return_value=mx.array([0.1, 0.2])
            ),
            patch("vllm_mlx.specprefill.select_chunks", return_value=mx.array([0, 1])),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 1, 32)),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
            patch(
                "vllm_mlx.engine.simple._sample_with_processors",
                side_effect=fake_sample,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    specprefill=True,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 1
        assert calls[0]["mtp"] is True
        assert "draft_model" not in calls[0]
        assert calls[0]["num_draft_tokens"] == 4
        assert "logits_processors" not in calls[0]

    @pytest.mark.anyio
    async def test_stream_generate_text_emits_before_generation_completes(self):
        """Text routing should yield the first token before the worker finishes."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        second_token_gate = threading.Event()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            yield SimpleNamespace(text="Hello", finish_reason=None)
            second_token_gate.wait(timeout=1.0)
            yield SimpleNamespace(text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            agen = engine._stream_generate_text(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
                temperature=0.7,
                top_p=0.9,
            )
            first = await asyncio.wait_for(anext(agen), timeout=0.2)
            assert first.new_text == "Hello"
            second_token_gate.set()
            remaining = [chunk async for chunk in agen]

        assert remaining[-1].new_text == " world"
        assert remaining[-1].finished is True

    @pytest.mark.anyio
    async def test_stream_generate_text_honors_stop_sequences(self):
        """Text routing should stop on explicit stop sequences like the LLM path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            yield SimpleNamespace(text="Hello", finish_reason=None)
            yield SimpleNamespace(text=" STOP rest", finish_reason=None)
            yield SimpleNamespace(text=" should_not_emit", finish_reason=None)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["STOP"],
                )
            ]

        assert [chunk.new_text for chunk in outputs] == ["Hello", " STOP rest"]
        assert outputs[-1].finished is True
        assert outputs[-1].finish_reason == "stop"
