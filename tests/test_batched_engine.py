# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine generate() output."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_mlx.engine.batched import _normalize_tool_call_arguments_for_template


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


class TestBatchedEngineCacheRestore:
    def _make_mllm_engine(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            engine = BatchedEngine("test-mllm")

        engine._loaded = True
        engine._is_mllm = True
        return engine

    def test_load_cache_from_disk_bootstraps_mllm_batch_generator(self):
        engine = self._make_mllm_engine()

        prefix_cache = MagicMock()
        prefix_cache.load_from_disk.return_value = 2
        scheduler = MagicMock()
        scheduler.batch_generator = None

        def ensure_batch_generator():
            scheduler.batch_generator = MagicMock(prefix_cache=prefix_cache)

        scheduler._ensure_batch_generator.side_effect = ensure_batch_generator
        scheduler.run_cache_owner_lifecycle_mutation.side_effect = (
            lambda operation, *args: operation(*args)
        )
        engine._mllm_scheduler = scheduler

        loaded = engine.load_cache_from_disk("/tmp/cache")

        assert loaded == 2
        scheduler._ensure_batch_generator.assert_called_once_with()
        prefix_cache.load_from_disk.assert_called_once_with("/tmp/cache")

    @pytest.mark.anyio
    async def test_hybrid_persist_keeps_snapshot_on_owner_and_io_off_thread(self):
        import threading

        engine = self._make_mllm_engine()
        owner_thread = threading.get_ident()
        calls = {}
        snapshot = object()
        prefix_cache = MagicMock()

        def prepare():
            calls["prepare_thread"] = threading.get_ident()
            return snapshot

        def write(path, value):
            calls["write_thread"] = threading.get_ident()
            calls["write_args"] = (path, value)
            return True

        prefix_cache.prepare_hybrid_persistence_snapshot.side_effect = prepare
        prefix_cache.write_hybrid_persistence_snapshot.side_effect = write
        engine._mllm_scheduler = MagicMock(
            batch_generator=MagicMock(prefix_cache=prefix_cache)
        )

        assert await engine.persist_cache_to_disk("/tmp/cache") is True
        assert calls["prepare_thread"] == owner_thread
        assert calls["write_thread"] != owner_thread
        assert calls["write_args"] == ("/tmp/cache", snapshot)

    @pytest.mark.anyio
    async def test_hybrid_restore_keeps_io_off_owner_and_rebuild_on_owner(self):
        import threading

        engine = self._make_mllm_engine()
        owner_thread = threading.get_ident()
        calls = {}
        loaded_snapshot = object()
        prefix_cache = MagicMock()

        def read(path):
            calls["read_thread"] = threading.get_ident()
            calls["read_path"] = path
            return loaded_snapshot

        def restore(value):
            calls["restore_thread"] = threading.get_ident()
            calls["restore_value"] = value
            return 1

        prefix_cache.read_hybrid_persistence_snapshot.side_effect = read
        prefix_cache.restore_hybrid_persistence_snapshot.side_effect = restore
        scheduler = MagicMock(batch_generator=MagicMock(prefix_cache=prefix_cache))
        scheduler.run_cache_owner_lifecycle_mutation.side_effect = (
            lambda operation, *args: operation(*args)
        )
        engine._mllm_scheduler = scheduler

        assert await engine.restore_cache_from_disk("/tmp/cache") == 1
        assert calls["read_thread"] != owner_thread
        assert calls["restore_thread"] == owner_thread
        assert calls["read_path"] == "/tmp/cache"
        assert calls["restore_value"] is loaded_snapshot

    @pytest.mark.anyio
    async def test_hybrid_persist_cancellation_waits_for_disk_publish(self):
        import asyncio
        import threading

        engine = self._make_mllm_engine()
        writer_started = threading.Event()
        release_writer = threading.Event()
        writer_finished = threading.Event()
        prefix_cache = MagicMock()
        prefix_cache.prepare_hybrid_persistence_snapshot.return_value = object()

        def write(*_args):
            writer_started.set()
            release_writer.wait(timeout=2)
            writer_finished.set()
            return True

        prefix_cache.write_hybrid_persistence_snapshot.side_effect = write
        engine._mllm_scheduler = MagicMock(
            batch_generator=MagicMock(prefix_cache=prefix_cache)
        )

        task = asyncio.create_task(engine.persist_cache_to_disk("/tmp/cache"))
        assert await asyncio.to_thread(writer_started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_writer.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert writer_finished.is_set()


class TestBatchedEngineMetalCacheLimit:
    def test_prefers_explicit_mlx_buffer_cache_limit(self, monkeypatch):
        from vllm_mlx.engine.batched import _resolve_metal_buffer_cache_limit

        monkeypatch.setenv("MLX_BUFFER_CACHE_LIMIT", str(2 * 1024**3))

        limit, source = _resolve_metal_buffer_cache_limit(
            max_recommended=16 * 1024**3,
            gpu_memory_utilization=0.5,
        )

        assert limit == 2 * 1024**3
        assert source == "MLX_BUFFER_CACHE_LIMIT"

    def test_scales_cache_limit_to_device_when_env_unset(self, monkeypatch):
        from vllm_mlx.engine.batched import _resolve_metal_buffer_cache_limit

        monkeypatch.delenv("MLX_BUFFER_CACHE_LIMIT", raising=False)

        limit, source = _resolve_metal_buffer_cache_limit(
            max_recommended=16 * 1024**3,
            gpu_memory_utilization=0.5,
        )

        assert limit == 8 * 1024**3
        assert source == "device-scaled"

    def test_ignores_invalid_mlx_buffer_cache_limit(self, monkeypatch):
        from vllm_mlx.engine.batched import _resolve_metal_buffer_cache_limit

        monkeypatch.setenv("MLX_BUFFER_CACHE_LIMIT", "invalid")

        limit, source = _resolve_metal_buffer_cache_limit(
            max_recommended=16 * 1024**3,
            gpu_memory_utilization=0.5,
        )

        assert limit == 8 * 1024**3
        assert source == "device-scaled"

    def test_batched_engine_does_not_hardcode_32gb_cache_limit(self):
        source = Path(__file__).parents[1] / "vllm_mlx" / "engine" / "batched.py"
        content = source.read_text()

        assert "MLX_BUFFER_CACHE_LIMIT" in content
        assert "mx.set_cache_limit(32 * 1024 * 1024 * 1024)" not in content


class TestBatchedEngineAbortRequest:
    @pytest.mark.anyio
    async def test_abort_request_routes_to_mllm_scheduler(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            engine = BatchedEngine("test-mllm")

        engine._mllm_scheduler = MagicMock()
        engine._mllm_scheduler.abort_request.return_value = True

        assert await engine.abort_request("req-1") is True
        engine._mllm_scheduler.abort_request.assert_called_once_with("req-1")

    @pytest.mark.anyio
    async def test_abort_request_routes_to_text_engine(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")

        engine._loaded = True
        engine._is_mllm = False
        engine._engine = MagicMock()
        engine._engine.abort_request.return_value = True

        assert await engine.abort_request("req-1") is True
        engine._engine.abort_request.assert_called_once_with("req-1")

    @pytest.mark.anyio
    async def test_abort_request_routes_to_async_text_engine(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")

        engine._loaded = True
        engine._is_mllm = False
        engine._engine = MagicMock()
        engine._engine.abort_request = AsyncMock(return_value=True)

        assert await engine.abort_request("req-1") is True
        engine._engine.abort_request.assert_awaited_once_with("req-1")

    @pytest.mark.anyio
    async def test_abort_request_returns_false_without_supported_engine(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")

        engine._loaded = True
        engine._is_mllm = False
        engine._engine = None

        assert await engine.abort_request("req-1") is False


class TestToolCallReplayNormalization:
    """Tests for OpenAI tool-call replay normalization before chat templating."""

    def test_parses_function_arguments_string_to_mapping(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Tokyo"
        }
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
            '{"city": "Tokyo"}'
        )

    def test_wraps_non_mapping_arguments_for_template_items(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "echo",
                            "arguments": '["not", "object"]',
                        }
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "value": ["not", "object"]
        }

    def test_closes_dangling_think_before_raw_tool_call(self):
        messages = [
            {
                "role": "assistant",
                "content": (
                    "<think>Need the weather tool.\n"
                    "<tool_call>\n"
                    "<function=get_weather>\n"
                    "<parameter=city>\nParis\n</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)
        content = normalized[0]["content"]

        assert "Need the weather tool.\n</think><tool_call>" in content
        assert "<tool_call>" not in content.split("</think>", 1)[0]
        assert messages[0]["content"].startswith("<think>Need")

    def test_normalizes_pydantic_style_messages_without_stringifying(self):
        class MessageLike:
            def model_dump(self, exclude_none=False):
                assert exclude_none is True
                return {
                    "role": "assistant",
                    "content": "plain response",
                    "unused": None,
                }

        normalized = _normalize_tool_call_arguments_for_template([MessageLike()])

        assert normalized == [{"role": "assistant", "content": "plain response"}]


class TestBatchedEngineStop:
    """stop() must actually release MLX's Metal buffer cache, not just drop
    Python references — otherwise idle-unload frees objects but not memory.
    """

    def _make_engine(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")
        engine._loaded = True
        return engine

    @pytest.mark.anyio
    async def test_stop_calls_mx_clear_cache(self, monkeypatch):
        from vllm_mlx.engine import batched as batched_mod

        calls = {"count": 0}
        monkeypatch.setattr(
            batched_mod.mx,
            "clear_cache",
            lambda: calls.__setitem__("count", calls["count"] + 1),
        )

        engine = self._make_engine()

        await engine.stop()

        assert calls["count"] == 1
        assert engine._model is None
        assert engine._loaded is False
