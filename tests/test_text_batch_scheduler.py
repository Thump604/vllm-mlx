# SPDX-License-Identifier: Apache-2.0
"""Tests for TextBatchScheduler."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.layers = [MagicMock() for _ in range(4)]
    m.mtp = MagicMock()
    m.mtp_forward = MagicMock()
    return m


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 0
    tok.apply_chat_template = MagicMock(
        return_value="<|im_start|>user\nHello<|im_end|>"
    )
    tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tok.decode = MagicMock(return_value="ABCDE")
    return tok


def test_scheduler_creation(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0, 151643},
    )
    assert scheduler._running is False
    stats = scheduler.get_stats()
    assert stats["active_requests"] == 0
    assert stats["pending_requests"] == 0
    assert stats["deferred_requests"] == 0
    assert stats["prompt_prefix_reuse_foundation_ready"] is True
    assert "prefix_cache" in stats


def test_admission_memory_check(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
        cache_memory_mb=1,
        max_active_tokens=1_000_000,
    )
    assert scheduler._check_memory_budget(100_000) is False
    assert scheduler._check_memory_budget(10) is True


def test_admission_compute_check(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
        cache_memory_mb=99_999,
        max_active_tokens=100,
    )
    scheduler._active_token_count = 90
    assert scheduler._check_compute_budget(20) is False
    assert scheduler._check_compute_budget(5) is True


def test_request_state_tracking():
    from vllm_mlx.text_batch_scheduler import RequestState

    state = RequestState(
        request_id="req-1",
        token_ids=[1, 2, 3],
        max_tokens=100,
        queue=asyncio.Queue(maxsize=256),
    )
    assert state.is_detached is False
    assert time.monotonic() - state.created_at < 1.0
    assert time.monotonic() - state.last_consumed_at < 1.0


def test_eject_marks_detached(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    queue = asyncio.Queue(maxsize=256)
    state = RequestState(
        request_id="req-1",
        token_ids=[1, 2, 3],
        max_tokens=100,
        queue=queue,
    )
    state.uid = 1
    state.admitted = True
    scheduler.requests["req-1"] = state
    scheduler.request_id_to_uid["req-1"] = 1
    scheduler.uid_to_request_id[1] = "req-1"
    scheduler._engine = MagicMock()
    scheduler._eject_request(1, reason="queue_full")
    assert not queue.empty()
    scheduler._engine.remove.assert_called_once_with([1])


def test_stats_include_latency_percentiles(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    for ms in [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]:
        scheduler._latency_samples.append(ms)
    stats = scheduler.get_stats()
    assert "latency_p50_ms" in stats
    assert "latency_p95_ms" in stats
    assert "latency_p99_ms" in stats
    assert stats["latency_p50_ms"] > 0


def test_stats_use_safe_batch_state_not_batch_generator_snapshot(
    mock_model, mock_tokenizer
):
    from vllm_mlx.text_batch_scheduler import TextBatchScheduler

    class SizedBatch:
        def __init__(self, size):
            self.uids = list(range(size))

        def __len__(self):
            return len(self.uids)

    class BatchGeneratorLike:
        def __init__(self):
            self._unprocessed_sequences = [1, 2]
            self._currently_processing = [1]
            self._prompt_batch = SizedBatch(3)
            self._generation_batch = SizedBatch(4)

        def stats(self):
            raise AssertionError("unsafe stats() snapshot should not be used")

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    scheduler._batch_generator = BatchGeneratorLike()
    scheduler._engine = MagicMock()
    scheduler._engine.has_work.return_value = True

    stats = scheduler.get_stats()

    assert stats["batch_state"] == {
        "queued_sequences": 2,
        "currently_processing": 1,
        "prompt_batch_size": 3,
        "generation_batch_size": 4,
    }
    assert stats["engine_has_work"] is True
    assert "batch_generator" not in stats


def test_install_cache_callbacks_supports_prompt_progress_only():
    from vllm_mlx.scheduler import _install_cache_callbacks

    class PromptResponse:
        def __init__(self):
            self.uid = 7
            self.progress = (3, 5)
            self.end_of_segment = True
            self.end_of_prompt = False

    class PromptBatch:
        def __init__(self):
            self.uids = [7]

        def extract_cache(self, idx):
            assert idx == 0
            return ["cache"]

    class BatchGen:
        def __init__(self):
            self._prompt_batch = PromptBatch()
            self._generation_batch = MagicMock(uids=[])

        def _next(self):
            return [PromptResponse()], []

    calls = []

    def save(uid, processed_tokens, extract_cache, end_of_segment, end_of_prompt):
        calls.append(
            (
                uid,
                processed_tokens,
                extract_cache() if extract_cache is not None else None,
                end_of_segment,
                end_of_prompt,
            )
        )

    batch_gen = BatchGen()
    _install_cache_callbacks(batch_gen, prompt_progress_save=save)
    batch_gen._next()

    assert calls == [(7, 3, ["cache"], True, False)]


def test_compute_prefix_boundary_and_segments(mock_model):
    from vllm_mlx.text_batch_scheduler import TextBatchScheduler

    class SimpleTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(
                f"{message['role']}:{message['content']}" for message in messages
            ) + "\nassistant:"

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

        def decode(self, tokens):
            return "".join(chr(token) for token in tokens)

    tokenizer = SimpleTokenizer()
    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "context"},
        {"role": "user", "content": "hello"},
    ]
    prompt = scheduler._apply_chat_template(messages)
    token_ids = scheduler._encode_text(prompt)
    boundary = scheduler._compute_prefix_boundary(messages, None, token_ids)

    assert 0 < boundary < len(token_ids)
    assert scheduler._segment_prompt_tokens(token_ids, boundary) == [
        token_ids[:boundary],
        token_ids[boundary:],
    ]


def test_prepare_request_uses_cached_prefix_segments(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    class BatchableCache:
        @classmethod
        def merge(cls, caches):
            return caches

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    scheduler._prefix_cache.fetch = MagicMock(  # type: ignore[method-assign]
        return_value=([BatchableCache()], [4, 5])
    )
    scheduler._prefix_cache._last_match_type = "prefix"

    state = RequestState(
        request_id="req-1",
        token_ids=[1, 2, 3, 4, 5],
        max_tokens=32,
        queue=asyncio.Queue(maxsize=256),
        segments=[[1, 2, 3], [4, 5]],
        prefix_boundary=3,
    )

    scheduler._prepare_request(state)

    assert state.cached_tokens == 3
    assert state.cache_hit_type == "prefix"
    assert state.prepared_all_tokens == [1, 2, 3]
    assert state.prepared_segments == [[4, 5]]
    assert state.active_token_cost == 2


def test_prepare_request_preserves_boundary_after_partial_cache(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    class BatchableCache:
        @classmethod
        def merge(cls, caches):
            return caches

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    scheduler._prefix_cache.fetch = MagicMock(  # type: ignore[method-assign]
        return_value=([BatchableCache()], [3, 4, 5])
    )
    scheduler._prefix_cache._last_match_type = "prefix"

    state = RequestState(
        request_id="req-2",
        token_ids=[1, 2, 3, 4, 5],
        max_tokens=32,
        queue=asyncio.Queue(maxsize=256),
        segments=[[1, 2, 3, 4], [5]],
        prefix_boundary=4,
    )

    scheduler._prepare_request(state)

    assert state.cached_tokens == 2
    assert state.prepared_all_tokens == [1, 2]
    assert state.prepared_segments == [[3, 4], [5]]
    assert state.active_token_cost == 3


def test_prepare_request_drops_non_batchable_history_cache(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    class NonBatchableCache:
        pass

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    scheduler._prefix_cache.fetch = MagicMock(
        return_value=([NonBatchableCache()], [4, 5])
    )  # type: ignore[method-assign]
    scheduler._prefix_cache._last_match_type = "prefix"

    state = RequestState(
        request_id="req-fallback",
        token_ids=[1, 2, 3, 4, 5],
        max_tokens=32,
        queue=asyncio.Queue(maxsize=256),
        segments=[[1, 2, 3], [4, 5]],
        prefix_boundary=3,
    )

    scheduler._prepare_request(state)

    assert state.prepared_cache is None
    assert state.cached_tokens == 0
    assert state.prepared_all_tokens == []
    assert state.prepared_segments == [[1, 2, 3], [4, 5]]
    assert state.cache_hit_type == "unsupported_history_cache"


def test_prepare_request_marks_cooperative_specprefill_when_boundary_is_cached(
    mock_model, mock_tokenizer
):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    class BatchableCache:
        @classmethod
        def merge(cls, caches):
            return caches

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
        draft_model=MagicMock(),
        specprefill_threshold=2,
        specprefill_keep_pct=0.5,
    )
    scheduler._prefix_cache.fetch = MagicMock(  # type: ignore[method-assign]
        return_value=([BatchableCache()], [4, 5, 6, 7])
    )
    scheduler._prefix_cache._last_match_type = "prefix"

    state = RequestState(
        request_id="req-specprefill",
        token_ids=[1, 2, 3, 4, 5, 6, 7],
        max_tokens=32,
        queue=asyncio.Queue(maxsize=256),
        segments=[[1, 2, 3], [4, 5, 6, 7]],
        prefix_boundary=3,
    )

    scheduler._prepare_request(state)

    assert state.cooperative_specprefill is True
    assert state.cooperative_specprefill_tokens == [4, 5, 6, 7]
    assert state.cooperative_specprefill_position_offset == 3
    assert state.active_token_cost == 1
    assert state.resident_token_cost == 7


def test_prepare_request_defers_cooperative_specprefill_until_prefix_boundary_cached(
    mock_model, mock_tokenizer
):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    class BatchableCache:
        @classmethod
        def merge(cls, caches):
            return caches

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
        draft_model=MagicMock(),
        specprefill_threshold=2,
        specprefill_keep_pct=0.5,
    )
    scheduler._prefix_cache.fetch = MagicMock(  # type: ignore[method-assign]
        return_value=([BatchableCache()], [3, 4, 5, 6, 7])
    )
    scheduler._prefix_cache._last_match_type = "prefix"

    state = RequestState(
        request_id="req-prefix-first",
        token_ids=[1, 2, 3, 4, 5, 6, 7],
        max_tokens=32,
        queue=asyncio.Queue(maxsize=256),
        segments=[[1, 2, 3, 4], [5, 6, 7]],
        prefix_boundary=4,
    )

    scheduler._prepare_request(state)

    assert state.cooperative_specprefill is False
    assert state.prepared_segments == [[3, 4], [5, 6, 7]]
    assert state.active_token_cost == 5


def test_save_prefix_cache_at_boundary(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    scheduler._prefix_cache.store = MagicMock(return_value=True)  # type: ignore[method-assign]

    state = RequestState(
        request_id="req-3",
        token_ids=[1, 2, 3, 4],
        max_tokens=32,
        queue=asyncio.Queue(maxsize=256),
        segments=[[1, 2, 3], [4]],
        prefix_boundary=3,
    )
    scheduler.requests[state.request_id] = state
    scheduler.uid_to_request_id[7] = state.request_id

    scheduler._save_prefix_cache(
        7,
        3,
        lambda: ["cache"],
        True,
        False,
    )

    scheduler._prefix_cache.store.assert_called_once_with(  # type: ignore[attr-defined]
        [1, 2, 3],
        ["cache"],
        evict_prefixes=False,
    )
    assert state.prefix_cache_saved is True


def test_start_request_uses_insert_segments(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    scheduler._engine = MagicMock()
    scheduler._engine.insert_segments.return_value = [11]

    state = RequestState(
        request_id="req-4",
        token_ids=[1, 2, 3, 4, 5],
        max_tokens=32,
        queue=asyncio.Queue(maxsize=256),
        segments=[[1, 2, 3], [4, 5]],
        prefix_boundary=3,
    )
    state.prepared_segments = [[4, 5]]
    state.prepared_all_tokens = [1, 2, 3]
    state.prepared_cache = ["cache"]
    state.active_token_cost = 2

    scheduler._start_request(state)

    scheduler._engine.insert_segments.assert_called_once()
    _, kwargs = scheduler._engine.insert_segments.call_args
    assert kwargs["caches"] == [["cache"]]
    assert kwargs["all_tokens"] == [[1, 2, 3]]


def test_first_token_finish_reason_wraps_state_machine(mock_model, mock_tokenizer):
    from vllm_mlx.cooperative_specprefill import PreseededSequenceStateMachine
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    class FakeStateMachine:
        def make_state(self):
            return "normal"

        def match(self, state, token):
            if token == 9:
                return (None, None, None)
            return ("normal", None, "normal")

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    state = RequestState(
        request_id="req-wrap",
        token_ids=[1, 2, 3],
        max_tokens=8,
        queue=asyncio.Queue(maxsize=256),
    )

    finish_reason, wrapped = scheduler._first_token_finish_reason(
        state,
        5,
        FakeStateMachine(),
    )

    assert finish_reason is None
    assert isinstance(wrapped, PreseededSequenceStateMachine)


def test_cleanup_finished_request_releases_resident_token_budget(
    mock_model, mock_tokenizer
):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    scheduler = TextBatchScheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        gpu_lock=asyncio.Lock(),
        stop_tokens={0},
    )
    state = RequestState(
        request_id="req-budget",
        token_ids=[1, 2, 3, 4, 5],
        max_tokens=16,
        queue=asyncio.Queue(maxsize=256),
    )
    state.admitted = True
    state.active_token_cost = 1
    state.resident_token_cost = 5
    scheduler.requests[state.request_id] = state
    scheduler._active_token_count = 1
    scheduler._current_cache_bytes = scheduler._estimate_kv_bytes(5)

    scheduler._cleanup_finished_request(state)

    assert scheduler._active_token_count == 0
    assert scheduler._current_cache_bytes == 0


def test_loop_failure_finishes_requests_with_error(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import RequestState, TextBatchScheduler

    async def _run():
        scheduler = TextBatchScheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            gpu_lock=asyncio.Lock(),
            stop_tokens={0},
        )
        state = RequestState(
            request_id="req-loop-error",
            token_ids=[1, 2, 3],
            max_tokens=16,
            queue=asyncio.Queue(maxsize=256),
        )
        scheduler.requests[state.request_id] = state
        scheduler._running = True
        scheduler._engine = MagicMock()
        scheduler._engine.has_work.return_value = True
        scheduler._step_engine = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        await scheduler._loop()

        item = await asyncio.wait_for(state.queue.get(), timeout=1)
        assert item.finished is True
        assert item.finish_reason == "error"
        assert state.finish_reason == "scheduler_failed: boom"
        assert scheduler._running is False
        assert scheduler._last_error == "boom"

    asyncio.run(_run())


def test_submit_iterator_cancellation_ejects(mock_model, mock_tokenizer):
    from vllm_mlx.text_batch_scheduler import TextBatchScheduler

    async def _run():
        scheduler = TextBatchScheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            gpu_lock=asyncio.Lock(),
            stop_tokens={0},
        )

        async def fake_start():
            scheduler._running = True

        scheduler.start = fake_start  # type: ignore[method-assign]
        scheduler._eject_request_by_id = MagicMock()  # type: ignore[method-assign]

        agen = scheduler.submit(
            [{"role": "user", "content": "Hello"}],
            max_tokens=32,
        )
        task = asyncio.create_task(agen.__anext__())
        await asyncio.sleep(0)
        assert scheduler.requests

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert scheduler._eject_request_by_id.call_count == 1

    asyncio.run(_run())
