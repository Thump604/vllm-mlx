# SPDX-License-Identifier: Apache-2.0
"""Regression test for cancellation-safe SimpleEngine serialization."""

from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class SimpleEngineCancelSerializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_does_not_release_lock_before_worker_finishes(self):
        """A cancelled request must not let a second MLX worker overlap."""
        from vllm_mlx.engine.simple import SimpleEngine

        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        model._concurrent_count = 0
        model._max_concurrent = 0

        first_started = threading.Event()
        release_workers = threading.Event()
        call_count = 0
        call_lock = threading.Lock()

        def generate_side_effect(**kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
                model._concurrent_count += 1
                model._max_concurrent = max(
                    model._max_concurrent, model._concurrent_count
                )
                if current_call == 1:
                    first_started.set()

            release_workers.wait(timeout=1.0)

            with call_lock:
                model._concurrent_count -= 1

            result = MagicMock()
            result.text = f"response-{current_call}"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True

            task1 = asyncio.create_task(engine.generate(prompt="first", max_tokens=8))
            await asyncio.to_thread(first_started.wait, 1.0)

            task1.cancel()
            task2 = asyncio.create_task(engine.generate(prompt="second", max_tokens=8))

            await asyncio.sleep(0.05)
            release_workers.set()

            with self.assertRaises(asyncio.CancelledError):
                await task1
            result2 = await task2

            self.assertEqual(result2.text, "response-2")
            self.assertEqual(
                model._max_concurrent,
                1,
                "cancellation released the generation lock before the first worker finished",
            )

    async def test_specprefill_path_does_not_prelock_serialized_runner(self):
        """Specprefill streaming must let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            self.assertFalse(engine._generation_lock.locked())
            return []

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()
            engine._draft_model = MagicMock()
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_specprefill(
                prompt="hello",
                tokens=[1, 2, 3, 4],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].finished)
            self.assertEqual(outputs[0].completion_tokens, 0)

    async def test_text_mtp_path_does_not_prelock_serialized_runner(self):
        """Text-only MTP streaming must let _run_blocking_serialized own the lock."""
        import mlx_lm

        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            self.assertFalse(engine._generation_lock.locked())
            return []

        def fake_stream_generate(*args, **kwargs):
            return iter(())

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True),
            patch(
                "vllm_mlx.engine.simple._bind_worker_generation_streams",
                return_value=None,
            ),
            patch.object(mlx_lm, "stream_generate", side_effect=fake_stream_generate),
        ):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._text_model = MagicMock()
            engine._text_model.make_mtp_cache = MagicMock(return_value=[])
            engine._text_tokenizer = MagicMock()
            engine._text_tokenizer.apply_chat_template = MagicMock(return_value="hello")
            engine._text_tokenizer.bos_token = None
            engine._draft_model = None
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_text(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].finished)
            self.assertEqual(outputs[0].completion_tokens, 0)

    async def test_text_route_stream_cancel_stops_after_next_token_boundary(self):
        """Client disconnect should not let text-route workers drain max_tokens."""
        import mlx_lm

        from vllm_mlx.engine.simple import SimpleEngine

        second_token_allowed = threading.Event()
        second_token_requested = threading.Event()
        consumed_tokens = []

        def fake_stream_generate(*args, **kwargs):
            consumed_tokens.append("A")
            yield SimpleNamespace(text="A", finish_reason=None)

            second_token_requested.set()
            second_token_allowed.wait(timeout=1.0)
            consumed_tokens.append("B")
            yield SimpleNamespace(text="B", finish_reason=None)

            for token in ("C", "D", "E"):
                consumed_tokens.append(token)
                yield SimpleNamespace(text=token, finish_reason=None)

        tokenizer = MagicMock()
        tokenizer.bos_token = None
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.encode.return_value = [1, 2, 3]

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True),
            patch(
                "vllm_mlx.engine.simple._bind_worker_generation_streams",
                return_value=None,
            ),
            patch.object(mlx_lm, "stream_generate", side_effect=fake_stream_generate),
        ):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._text_model = MagicMock()
            engine._text_model.mtp = None
            engine._text_tokenizer = tokenizer
            engine._draft_model = None

            stream = engine._stream_generate_text(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=8,
                temperature=0.7,
                top_p=0.9,
            )
            first = await stream.__anext__()
            self.assertEqual(first.new_text, "A")

            next_task = asyncio.create_task(stream.__anext__())
            await asyncio.to_thread(second_token_requested.wait, 1.0)
            next_task.cancel()
            await asyncio.sleep(0)
            second_token_allowed.set()

            with self.assertRaises(asyncio.CancelledError):
                await next_task

            self.assertEqual(consumed_tokens, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
