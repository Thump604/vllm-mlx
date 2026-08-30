# SPDX-License-Identifier: Apache-2.0
"""Focused P2 regressions for MLLM request ownership transactions.

These tests cover generator-owned UIDs and abort markers, retryable deferred
removals, scheduler rollback at the generator boundary, response ownership,
and request-lock interleavings.  Admission policy, streaming, and shutdown
are intentionally outside this package.
"""

import asyncio
from collections import deque
from contextlib import nullcontext
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _assert_same_error(action, error, *args):
    with pytest.raises(type(error)) as exc_info:
        action(*args)
    assert exc_info.value is error


def _start_worker(target, *args):
    """Run a worker while preserving exceptions for the test thread."""
    errors = []

    def guarded_target():
        try:
            target(*args)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=guarded_target)
    thread.start()
    return thread, errors


def _assert_workers_done(*workers):
    for thread, errors in workers:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert not errors, f"worker thread raised: {errors[0]!r}" if errors else ""


def _initialize_prefix_ownership_state(generator):
    generator._prefix_checkpoint_lock = threading.RLock()
    generator._request_prefix_checkpoints = {}
    return generator


def _make_insert_generator():
    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

    generator = _initialize_prefix_ownership_state(
        MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    )
    generator.uid_counter = 0
    generator.unprocessed_requests = []
    generator.prefix_cache = None
    return generator


def _make_scheduler(*, max_num_seqs=16):
    from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(max_num_seqs=max_num_seqs)
    scheduler.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda prompt: prompt.split())
    )
    scheduler.waiting = deque()
    scheduler.running = {}
    scheduler.requests = {}
    scheduler.finished_req_ids = set()
    scheduler._state_lock = threading.RLock()
    scheduler._request_lock = scheduler._state_lock
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._pending_generator_removals = set()
    scheduler._detokenizer_pool = {}
    scheduler.output_queues = {}
    scheduler.batch_generator = None
    scheduler._ensure_batch_generator = lambda: None
    scheduler._running = False
    scheduler.num_requests_processed = 0
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler._step_count = 0
    scheduler._clear_cache_interval = 100
    return scheduler


def _add_waiting_requests(scheduler, *request_ids, prompt_tokens=1):
    from vllm_mlx.mllm_scheduler import MLLMRequest

    for request_id in request_ids:
        request = MLLMRequest(request_id=request_id, prompt=request_id)
        request.num_prompt_tokens = prompt_tokens
        scheduler.waiting.append(request)
        scheduler.requests[request_id] = request
    return scheduler


def _move_to_running(scheduler, request_id, uid):
    from vllm_mlx.request import RequestStatus

    request = scheduler.requests[request_id]
    scheduler.waiting.remove(request)
    request.status = RequestStatus.RUNNING
    request.batch_uid = uid
    scheduler.running[request_id] = request
    scheduler.request_id_to_uid[request_id] = uid
    scheduler.uid_to_request_id[uid] = request_id
    return request


def _make_chunked_generator():
    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchStats

    generator = _initialize_prefix_ownership_state(
        MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    )
    generator._next = lambda: []
    generator._pending_error_responses = []
    generator._aborted_request_ids = set()
    generator._aborted_request_uids = set()
    generator._prefill_progress = {}
    generator.active_batch = None
    generator.unprocessed_requests = []
    generator.prefix_cache = None
    generator.language_model = MagicMock()
    generator._stats = MLLMBatchStats()
    generator._think_suffix_len = 0
    generator.max_kv_size = 0
    return generator


class TestMLLMPendingRemovalOwnership:
    def test_process_pending_removals_atomic_swap_and_retry(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        class InjectOnExhaustionIterator:
            def __init__(self, base_iter, callback):
                self._base_iter = base_iter
                self._callback = callback
                self._injected = False

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    return next(self._base_iter)
                except StopIteration:
                    if not self._injected:
                        self._injected = True
                        self._callback()
                    raise

        class InjectOnExhaustionSet(set):
            def __init__(self, values, callback):
                super().__init__(values)
                self._callback = callback

            def __iter__(self):
                return InjectOnExhaustionIterator(
                    super().__iter__(),
                    self._callback,
                )

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._pending_removal_lock = threading.Lock()
        generator._aborted_request_ids = set()
        generator._aborted_request_uids = set()
        removed = []

        def remove(uids):
            removed.extend(uids)

        generator.remove = remove
        generator._pending_removal_uids = InjectOnExhaustionSet(
            {1}, lambda: generator.schedule_removal([2])
        )

        generator.process_pending_removals()
        assert removed == [1]
        assert generator._pending_removal_uids == {2}

        generator.process_pending_removals()
        assert removed == [1, 2]
        assert generator._pending_removal_uids == set()

    def test_process_pending_removals_requeues_failed_removal(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        removal_error = RuntimeError("remove failed")

        class FailingActiveBatch:
            uids = [7, 8]
            requests = [
                SimpleNamespace(uid=7, request_id="same"),
                SimpleNamespace(uid=8, request_id="other"),
            ]

            def __init__(self):
                self.fail = True

            def filter(self, keep_idx):
                if self.fail:
                    self.fail = False
                    raise removal_error
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._pending_removal_lock = threading.Lock()
        generator._pending_removal_uids = {7}
        generator._aborted_request_ids = {"same"}
        generator._aborted_request_uids = {7}
        generator._prefill_progress = {"same": (2, 5)}
        generator.unprocessed_requests = []
        generator.active_batch = FailingActiveBatch()

        _assert_same_error(generator.process_pending_removals, removal_error)
        assert generator._pending_removal_uids == {7}
        assert generator._aborted_request_ids == {"same"}
        assert generator._aborted_request_uids == {7}
        assert generator._prefill_progress == {"same": (2, 5)}

        generator.process_pending_removals()
        assert generator._pending_removal_uids == set()
        assert generator.active_batch.uids == [8]
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()
        assert generator._prefill_progress == {}

    def test_generator_removal_retires_abort_marker_for_exact_uid(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        class FakeActiveBatch:
            def __init__(self):
                self.uids = [92, 95]
                self.requests = [
                    SimpleNamespace(uid=92, request_id="same"),
                    SimpleNamespace(uid=95, request_id="other"),
                ]

            def filter(self, keep_idx):
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = _initialize_prefix_ownership_state(
            MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        )
        generator.active_batch = FakeActiveBatch()
        generator.unprocessed_requests = [SimpleNamespace(uid=91, request_id="same")]
        generator._aborted_request_ids = {"same", "other"}
        generator._aborted_request_uids = {91, 92, 93}
        generator._prefill_progress = {
            "same": (2, 4),
            "other": (1, 3),
        }

        generator.remove([91, 92])
        assert generator.unprocessed_requests == []
        assert generator.active_batch.uids == [95]
        assert [request.request_id for request in generator.active_batch.requests] == [
            "other"
        ]
        assert generator._aborted_request_ids == {"other"}
        assert generator._aborted_request_uids == {93}
        assert generator._prefill_progress == {"other": (1, 3)}

        generator.abort_prefill("same", 93)
        replacement = SimpleNamespace(uid=94, request_id="same")
        retired = SimpleNamespace(uid=93, request_id="same")
        assert generator._consume_prefill_abort(replacement) is False
        assert generator._consume_prefill_abort(retired) is True


class TestMLLMGeneratorInsertOwnership:
    def test_generator_insert_assigns_unique_uids_and_sorts_pending_work(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_insert_generator()
        generator.uid_counter = 4
        existing = MLLMBatchRequest(uid=9, request_id="old", prompt="old", images=["x"])
        image = MLLMBatchRequest(
            uid=-1, request_id="image", prompt="image", images=["y"]
        )
        text = MLLMBatchRequest(uid=-1, request_id="text", prompt="text")
        generator.unprocessed_requests = [existing]

        uids = generator.insert([image, text])

        assert [image.uid, text.uid] == uids == [4, 5]
        assert [request.request_id for request in generator.unprocessed_requests] == [
            "text",
            "old",
            "image",
        ]
        assert generator.uid_counter == 6

    @pytest.mark.parametrize(
        "mode",
        ["duplicate", "already-pending", "already-active", "already-partial"],
    )
    def test_generator_insert_rejects_duplicate_or_owned_request_without_mutation(
        self, mode
    ):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_insert_generator()
        generator.uid_counter = 8
        existing = MLLMBatchRequest(uid=6, request_id="old", prompt="old")
        generator.unprocessed_requests = [existing]
        request = MLLMBatchRequest(uid=3, request_id="new", prompt="new")

        if mode == "duplicate":
            requests = [request, request]
        elif mode == "already-pending":
            generator.unprocessed_requests.append(request)
            requests = [request]
        elif mode == "already-active":
            generator.active_batch = SimpleNamespace(requests=[request], uids=[3])
            requests = [request]
        else:
            generator._partial = {"request": request}
            requests = [request]

        original_queue = generator.unprocessed_requests
        original_uid = request.uid
        with pytest.raises(ValueError):
            generator.insert(requests)

        assert generator.uid_counter == 8
        assert generator.unprocessed_requests is original_queue
        assert request.uid == original_uid

    def test_generator_insert_metadata_failure_is_atomic(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        class FailingMetadataRequest:
            uid = 33
            request_id = "bad"
            videos = audio = None

            @property
            def images(self):
                raise RuntimeError("metadata failed")

        generator = _make_insert_generator()
        generator.uid_counter = 8
        existing = MLLMBatchRequest(uid=7, request_id="old", prompt="old")
        generator.unprocessed_requests = [existing]
        original_queue = generator.unprocessed_requests
        first = MLLMBatchRequest(uid=21, request_id="first", prompt="first")
        second = MLLMBatchRequest(uid=22, request_id="second", prompt="second")
        bad_request = FailingMetadataRequest()
        requests = [first, second, bad_request]
        original_uids = [request.uid for request in requests]

        with pytest.raises(RuntimeError, match="metadata failed"):
            generator.insert(requests)

        assert generator.uid_counter == 8
        assert [request.uid for request in requests] == original_uids
        assert generator.unprocessed_requests is original_queue
        assert [request.uid for request in original_queue] == [7]

    @pytest.mark.parametrize("mode", ["unprocessed", "active", "partial"])
    def test_generator_insert_rejects_live_uid_overlap_without_mutation(self, mode):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_insert_generator()
        generator.uid_counter = 4
        live = MLLMBatchRequest(uid=4, request_id="live", prompt="live")
        request = MLLMBatchRequest(uid=99, request_id="new", prompt="new")

        if mode == "unprocessed":
            generator.unprocessed_requests = [live]
        elif mode == "active":
            generator.active_batch = SimpleNamespace(uids=[4], requests=[live])
        else:
            generator._partial = {"request": live}

        original_queue = generator.unprocessed_requests
        with pytest.raises(RuntimeError, match="overlaps live"):
            generator.insert([request])

        assert request.uid == 99
        assert generator.uid_counter == 4
        assert generator.unprocessed_requests is original_queue
        assert live.uid == 4

    @pytest.mark.parametrize(
        "counter",
        [-1, True, 1.5, "0", None],
        ids=["negative", "bool", "float", "string", "none"],
    )
    def test_generator_insert_rejects_nonnegative_exact_int_uid_counter(self, counter):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_insert_generator()
        generator.uid_counter = counter
        request = MLLMBatchRequest(uid=17, request_id="new", prompt="new")
        original_queue = generator.unprocessed_requests

        with pytest.raises(ValueError, match="nonnegative integer"):
            generator.insert([request])

        assert generator.uid_counter == counter
        assert request.uid == 17
        assert generator.unprocessed_requests is original_queue


class TestMLLMSchedulerOwnership:
    def test_add_request_and_abort_are_atomic_with_surfaced_workers(self):
        class BlockingLock:
            def __init__(self):
                self._lock = threading.Lock()
                self.entered = threading.Event()
                self.release = threading.Event()
                self._first_entry = True

            def __enter__(self):
                self._lock.acquire()
                if self._first_entry:
                    self._first_entry = False
                    self.entered.set()
                    if not self.release.wait(timeout=1.0):
                        self._lock.release()
                        raise AssertionError(
                            "add_request did not release ownership lock"
                        )
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self._lock.release()

        scheduler = _make_scheduler()
        request_lock = BlockingLock()
        scheduler._request_lock = request_lock
        add_result = []
        abort_result = []
        abort_started = threading.Event()

        def add_request():
            add_result.append(scheduler.add_request("prompt", request_id="same"))

        def abort_request():
            abort_started.set()
            abort_result.append(scheduler.abort_request("same"))

        add_worker = _start_worker(add_request)
        assert request_lock.entered.wait(timeout=1.0)
        abort_worker = _start_worker(abort_request)
        assert abort_started.wait(timeout=1.0)
        assert abort_worker[0].is_alive()

        request_lock.release.set()
        _assert_workers_done(add_worker, abort_worker)
        assert add_result == ["same"]
        assert abort_result == [True]
        assert scheduler.requests == {}
        assert list(scheduler.waiting) == []
        assert not scheduler.has_requests()

    def test_add_request_rejects_duplicate_live_id_without_queue_mutation(self):
        scheduler = _make_scheduler()
        scheduler.add_request("original", request_id="same")
        original_request = scheduler.requests["same"]
        original_waiting = list(scheduler.waiting)

        with pytest.raises(ValueError, match="duplicate MLLM request ID"):
            scheduler.add_request("replacement", request_id="same")

        assert scheduler.requests == {"same": original_request}
        assert list(scheduler.waiting) == original_waiting

    def test_scheduler_insert_exception_restores_waiting_state_without_ghosts(self):
        from vllm_mlx.request import RequestStatus

        class PoisonMetadataRequest:
            uid = 7
            request_id = "poison"
            videos = audio = None

            @property
            def images(self):
                raise RuntimeError("metadata failed")

        scheduler = _add_waiting_requests(
            _make_scheduler(max_num_seqs=3),
            "first",
            "second",
            "third",
        )
        generator = _make_insert_generator()
        generator.uid_counter = 8
        poison = PoisonMetadataRequest()
        generator.unprocessed_requests = [poison]
        original_queue = generator.unprocessed_requests
        scheduler.batch_generator = generator

        with pytest.raises(RuntimeError, match="metadata failed"):
            scheduler._schedule_waiting()

        assert [request.request_id for request in scheduler.waiting] == [
            "first",
            "second",
            "third",
        ]
        assert scheduler.running == {}
        assert scheduler.total_prompt_tokens == 0
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        assert all(
            request.status == RequestStatus.WAITING and request.batch_uid is None
            for request in scheduler.requests.values()
        )
        assert generator.uid_counter == 8
        assert generator.unprocessed_requests is original_queue
        assert generator.unprocessed_requests == [poison]

    def test_batch_response_must_match_public_uid_owner(self):
        scheduler = _make_scheduler()
        scheduler.uid_to_request_id = {101: "first"}
        response = SimpleNamespace(
            uid=101,
            request_id="other",
            finish_reason="error",
        )

        with pytest.raises(RuntimeError, match="response owner mismatch"):
            scheduler._process_batch_responses([response])

    @pytest.mark.parametrize("mismatch", ["request", "mapping"])
    def test_batch_response_rejects_internal_uid_owner_mismatch(self, mismatch):
        scheduler = _add_waiting_requests(_make_scheduler(), "first")
        request = _move_to_running(scheduler, "first", uid=101)
        response = SimpleNamespace(
            uid=101,
            request_id="first",
            finish_reason="error",
        )
        if mismatch == "request":
            request.batch_uid = 202
        else:
            scheduler.request_id_to_uid["first"] = 202

        with pytest.raises(RuntimeError, match="UID ownership mismatch"):
            scheduler._process_batch_responses([response])

    def test_step_drains_pending_removal_before_forward_for_other_request(self):
        scheduler = _add_waiting_requests(_make_scheduler(max_num_seqs=1), "running")
        _move_to_running(scheduler, "running", uid=22)
        scheduler._pending_generator_removals = {11}
        events = []

        class BoundedGenerator:
            def schedule_removal(self, uids, request_ids=None):
                events.append(("schedule", list(uids)))

            def process_pending_removals(self):
                events.append(("process", None))

            def has_pending_removals(self):
                return False

            def next(self):
                events.append(("next", None))
                assert events == [
                    ("schedule", [11]),
                    ("process", None),
                    ("next", None),
                ]
                return []

        scheduler.batch_generator = BoundedGenerator()

        output = scheduler.step()

        assert output.has_work is True
        assert events == [
            ("schedule", [11]),
            ("process", None),
            ("next", None),
        ]
        assert scheduler._pending_generator_removals == set()

    def test_schedule_removal_failure_keeps_idle_scheduler_live_until_drain(self):
        scheduler = _make_scheduler(max_num_seqs=1)
        _add_waiting_requests(scheduler, "first", prompt_tokens=2)
        request = _move_to_running(scheduler, "first", uid=41)
        cleanup_error = RuntimeError("schedule removal failed")
        events = []

        def schedule_removal(uids, request_ids=None):
            events.append(("schedule", list(uids)))
            if len(events) == 1:
                raise cleanup_error

        generator = SimpleNamespace(
            abort_prefill=MagicMock(),
            schedule_removal=schedule_removal,
            has_pending_removals=lambda: False,
            process_pending_removals=lambda: events.append(("remove", None)),
            next=MagicMock(),
        )
        scheduler.batch_generator = generator

        _assert_same_error(scheduler.abort_request, cleanup_error, "first")
        assert request.status.name == "FINISHED_ABORTED"
        assert scheduler.requests == {}
        assert scheduler.running == {}
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        assert scheduler._pending_generator_removals == {41}
        assert scheduler.has_requests()

        scheduler.step()
        assert events == [
            ("schedule", [41]),
            ("schedule", [41]),
            ("remove", None),
        ]
        assert scheduler._pending_generator_removals == set()
        assert not scheduler.has_requests()
        generator.next.assert_not_called()

    def test_abort_marker_and_schedule_failures_preserve_retry_backlog(self):
        scheduler = _make_scheduler(max_num_seqs=1)
        _add_waiting_requests(scheduler, "first")
        request = _move_to_running(scheduler, "first", uid=41)
        abort_error = RuntimeError("abort marker failed")
        initial_schedule_error = RuntimeError("initial schedule failed")
        retry_schedule_error = RuntimeError("retry schedule failed")
        events = []
        schedule_attempts = 0

        def abort_prefill(request_id, uid):
            events.append(("abort_prefill", request_id, uid))
            raise abort_error

        def schedule_removal(uids, request_ids=None):
            nonlocal schedule_attempts
            schedule_attempts += 1
            events.append(("schedule", list(uids)))
            if schedule_attempts == 1:
                raise initial_schedule_error
            if schedule_attempts == 2:
                raise retry_schedule_error

        generator = SimpleNamespace(
            abort_prefill=abort_prefill,
            schedule_removal=schedule_removal,
            has_pending_removals=lambda: False,
            process_pending_removals=lambda: events.append(("process", None)),
            next=MagicMock(),
        )
        scheduler.batch_generator = generator

        _assert_same_error(scheduler.abort_request, abort_error, "first")
        assert request.status.name == "FINISHED_ABORTED"
        assert scheduler.requests == {}
        assert scheduler.running == {}
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        assert scheduler._pending_generator_removals == {41}
        assert scheduler.has_requests()
        assert events == [
            ("abort_prefill", "first", 41),
            ("schedule", [41]),
        ]

        with pytest.raises(RuntimeError) as retry_info:
            scheduler.step()
        assert retry_info.value is retry_schedule_error
        assert scheduler._pending_generator_removals == {41}
        assert events == [
            ("abort_prefill", "first", 41),
            ("schedule", [41]),
            ("schedule", [41]),
        ]
        generator.next.assert_not_called()

        scheduler.step()
        assert scheduler._pending_generator_removals == set()
        assert events == [
            ("abort_prefill", "first", 41),
            ("schedule", [41]),
            ("schedule", [41]),
            ("schedule", [41]),
            ("process", None),
        ]
        assert not scheduler.has_requests()
        generator.next.assert_not_called()

    def test_reset_retries_generator_ownership_before_detach(self):
        scheduler = _make_scheduler(max_num_seqs=1)
        _add_waiting_requests(scheduler, "first")
        _move_to_running(scheduler, "first", uid=41)
        scheduler.vision_cache = None
        cleanup_error = RuntimeError("schedule removal failed during reset")
        events = []

        def schedule_removal(uids, request_ids=None):
            events.append(("schedule", list(uids)))
            if len(events) == 1:
                raise cleanup_error

        generator = SimpleNamespace(
            abort_prefill=MagicMock(),
            schedule_removal=schedule_removal,
            process_pending_removals=lambda: events.append(("process", None)),
            close=MagicMock(),
        )
        scheduler.batch_generator = generator

        _assert_same_error(scheduler.reset, cleanup_error)
        assert scheduler.batch_generator is generator
        assert scheduler._pending_generator_removals == {41}
        assert events == [("schedule", [41])]
        generator.close.assert_not_called()

        scheduler.reset()
        assert events == [
            ("schedule", [41]),
            ("schedule", [41]),
            ("process", None),
        ]
        assert scheduler.batch_generator is None
        assert scheduler._pending_generator_removals == set()
        assert scheduler.requests == {}
        assert scheduler.running == {}
        assert scheduler.waiting == deque()
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        generator.close.assert_called_once_with()

    def test_reset_retries_real_generator_removal_before_detach(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        class FailingActiveBatch:
            def __init__(self):
                self.uids = [41, 99]
                self.requests = [
                    SimpleNamespace(uid=41, request_id="first"),
                    SimpleNamespace(uid=99, request_id="keep"),
                ]
                self.fail = True

            def filter(self, keep_idx):
                if self.fail:
                    self.fail = False
                    raise RuntimeError("filter failed during reset")
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        scheduler = _make_scheduler(max_num_seqs=1)
        _add_waiting_requests(scheduler, "first")
        _move_to_running(scheduler, "first", uid=41)
        scheduler.vision_cache = None
        generator = _initialize_prefix_ownership_state(
            MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        )
        generator.active_batch = FailingActiveBatch()
        generator.unprocessed_requests = []
        generator._pending_removal_uids = set()
        generator._pending_removal_lock = threading.Lock()
        generator._aborted_request_ids = {"first"}
        generator._aborted_request_uids = {41}
        generator._prefill_progress = {"first": (2, 5)}
        generator.close = MagicMock()
        scheduler.batch_generator = generator
        monkeypatch.setattr("vllm_mlx.mllm_scheduler.mx.clear_cache", lambda: None)

        with pytest.raises(RuntimeError, match="filter failed during reset"):
            scheduler.reset()

        assert scheduler.batch_generator is generator
        assert generator._pending_removal_uids == {41}
        assert generator._aborted_request_ids == {"first"}
        assert generator._aborted_request_uids == {41}
        assert generator._prefill_progress == {"first": (2, 5)}
        assert generator.active_batch.uids == [41, 99]
        generator.close.assert_not_called()

        scheduler.reset()

        assert scheduler.batch_generator is None
        assert generator._pending_removal_uids == set()
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()
        assert generator._prefill_progress == {}
        assert generator.active_batch.uids == [99]
        assert scheduler.requests == {}
        assert scheduler.running == {}
        assert scheduler.waiting == deque()
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        generator.close.assert_called_once_with()

    def test_real_insert_abort_and_removal_preserve_production_ownership_chain(self):
        scheduler = _add_waiting_requests(_make_scheduler(max_num_seqs=1), "first")
        generator = _make_insert_generator()
        generator.uid_counter = 0
        generator.active_batch = None
        generator._pending_removal_uids = set()
        generator._pending_removal_lock = threading.Lock()
        generator._aborted_request_ids = set()
        generator._aborted_request_uids = set()
        generator._prefill_progress = {}
        scheduler.batch_generator = generator

        scheduled = scheduler._schedule_waiting()

        assert [request.request_id for request in scheduled] == ["first"]
        assert scheduler.request_id_to_uid == {"first": 0}
        assert scheduler.uid_to_request_id == {0: "first"}
        assert scheduler.requests["first"].batch_uid == 0
        assert len(generator.unprocessed_requests) == 1
        assert generator.unprocessed_requests[0].request_id == "first"
        assert generator.unprocessed_requests[0].uid == 0

        request = scheduler.requests["first"]
        assert scheduler.abort_request("first") is True
        assert request.status.name == "FINISHED_ABORTED"
        assert generator._pending_removal_uids == {0}
        assert generator.has_pending_removals()
        assert scheduler.has_requests()

        generator.process_pending_removals()

        assert generator.unprocessed_requests == []
        assert generator._pending_removal_uids == set()
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()
        assert not generator.has_pending_removals()
        assert not scheduler.has_requests()

    def test_step_error_cleanup_retries_scheduler_and_generator_owners(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        class FailingActiveBatch:
            def __init__(self):
                self.uids = [31, 32]
                self.requests = [
                    SimpleNamespace(uid=31, request_id="first"),
                    SimpleNamespace(uid=32, request_id="second"),
                ]
                self.fail = True

            def filter(self, keep_idx):
                if self.fail:
                    self.fail = False
                    raise RuntimeError("filter failed during step-error cleanup")
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

            def notify_all_rows_removed(self):
                pass

        scheduler = _make_scheduler(max_num_seqs=2)
        _add_waiting_requests(scheduler, "first", "second")
        _move_to_running(scheduler, "first", uid=31)
        _move_to_running(scheduler, "second", uid=32)
        generator = _initialize_prefix_ownership_state(
            MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        )
        generator.active_batch = FailingActiveBatch()
        generator.unprocessed_requests = []
        generator._pending_removal_uids = set()
        generator._pending_removal_lock = threading.Lock()
        generator._aborted_request_ids = {"first", "second"}
        generator._aborted_request_uids = {31, 32}
        generator._prefill_progress = {
            "first": (2, 5),
            "second": (3, 6),
        }

        real_schedule_removal = generator.schedule_removal
        schedule_calls = []
        first_schedule_failure = True

        def schedule_removal(uids, request_ids=None):
            nonlocal first_schedule_failure
            schedule_calls.append(list(uids))
            if uids == [31] and first_schedule_failure:
                first_schedule_failure = False
                raise RuntimeError("first schedule failed")
            real_schedule_removal(uids, request_ids=request_ids)

        generator.schedule_removal = schedule_removal
        scheduler.batch_generator = generator

        scheduler._fail_requests_after_step_error(RuntimeError("forward failed"))

        assert scheduler.requests == {}
        assert scheduler.running == {}
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        assert scheduler._pending_generator_removals == {31}
        assert generator._pending_removal_uids == {32}
        assert generator._aborted_request_ids == {"first", "second"}
        assert generator._aborted_request_uids == {31, 32}
        assert generator._prefill_progress == {
            "first": (2, 5),
            "second": (3, 6),
        }
        assert generator.active_batch.uids == [31, 32]
        assert schedule_calls == [[31], [32]]
        assert scheduler.has_requests()

        scheduler.step()

        assert schedule_calls == [[31], [32], [31]]
        assert scheduler._pending_generator_removals == set()
        assert generator._pending_removal_uids == set()
        assert generator.active_batch is None
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()
        assert generator._prefill_progress == {}
        assert not scheduler.has_requests()

    def test_process_loop_drains_real_pending_removal_without_idling(self, monkeypatch):
        import vllm_mlx.mllm_scheduler as scheduler_module
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        scheduler = _make_scheduler()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.unprocessed_requests = []
        generator.active_batch = None
        generator._pending_removal_uids = {7}
        generator._pending_removal_lock = threading.Lock()
        generator._aborted_request_ids = set()
        generator._aborted_request_uids = set()
        generator._prefill_progress = {}
        scheduler.batch_generator = generator
        scheduler._running = True
        monkeypatch.setattr(scheduler_module, "bind_generation_streams", lambda: None)

        real_step = scheduler.step
        step_calls = []
        step_errors = []

        def drain_once():
            step_calls.append(1)
            try:
                return real_step()
            except BaseException as error:
                step_errors.append(error)
                raise
            finally:
                scheduler._running = False

        scheduler.step = drain_once
        asyncio.run(scheduler._process_loop())

        assert step_calls == [1]
        assert step_errors == []
        assert generator._pending_removal_uids == set()
        assert not generator.has_pending_removals()
        assert not scheduler.has_requests()

    def test_step_error_and_concurrent_abort_are_atomic_with_surfaced_workers(self):
        scheduler = _make_scheduler(max_num_seqs=1)
        _add_waiting_requests(scheduler, "first")
        _move_to_running(scheduler, "first", uid=21)
        error_abort_started = threading.Event()
        allow_error_abort = threading.Event()
        original_abort = scheduler._abort_request

        def blocking_abort(request_id):
            error_abort_started.set()
            assert allow_error_abort.wait(timeout=1.0)
            return original_abort(request_id)

        scheduler._abort_request = blocking_abort
        concurrent_started = threading.Event()
        concurrent_result = []

        def concurrent_abort():
            concurrent_started.set()
            concurrent_result.append(scheduler.abort_request("first"))

        error_worker = _start_worker(
            scheduler._fail_requests_after_step_error,
            RuntimeError("forward failed"),
        )
        assert error_abort_started.wait(timeout=1.0)
        abort_worker = _start_worker(concurrent_abort)
        assert concurrent_started.wait(timeout=1.0)
        assert abort_worker[0].is_alive()

        allow_error_abort.set()
        _assert_workers_done(error_worker, abort_worker)
        assert concurrent_result == [False]
        assert scheduler.requests == {}
        assert scheduler.running == {}


class TestMLLMPrefillAbortPolling:
    def test_run_chunked_text_prefill_aborts_before_model_call(self):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            PrefillAbortedError,
        )

        generator = _make_chunked_generator()
        generator.prefill_step_size = 2
        request = MLLMBatchRequest(uid=1, request_id="same", prompt="long text")
        request.input_ids = mx.array([[1, 2, 3, 4, 5]])
        generator.abort_prefill(request.request_id, request.uid)
        replacement = MLLMBatchRequest(
            uid=2, request_id=request.request_id, prompt="replacement"
        )

        assert generator._consume_prefill_abort(replacement) is False
        with pytest.raises(PrefillAbortedError, match="same"):
            generator._run_chunked_text_prefill(request, cache=[])

        generator.language_model.assert_not_called()
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()

    def test_process_prompts_entry_poll_consumes_uid_abort(self, monkeypatch):
        import vllm_mlx.mllm_batch_generator as batch_module
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_chunked_generator()
        generator.prefill_step_size = 2
        generator.sampler = MagicMock()
        generator.model = SimpleNamespace(
            config=SimpleNamespace(image_token_index=None)
        )
        generator._preprocess_request = lambda request: None
        monkeypatch.setattr(
            "mlx_lm.models.cache.make_prompt_cache",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_logits_processors",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_sampler",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(batch_module.mx, "clear_cache", lambda: None)

        request = MLLMBatchRequest(uid=11, request_id="same", prompt="prompt")
        request.input_ids = mx.array([[1, 2, 3]])
        request.is_text_only = True
        generator.abort_prefill(request.request_id, request.uid)
        replacement = MLLMBatchRequest(
            uid=12, request_id=request.request_id, prompt="replacement"
        )
        assert generator._consume_prefill_abort(replacement) is False

        assert generator._process_prompts([request]) is None
        generator.language_model.assert_not_called()
        assert len(generator._pending_error_responses) == 1
        assert generator._pending_error_responses[0].finish_reason == "abort"
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()

    def test_cached_prefix_poll_consumes_abort_before_cached_model_call(
        self, monkeypatch
    ):
        import vllm_mlx.mllm_batch_generator as batch_module
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_chunked_generator()
        generator.prefill_step_size = 2
        generator.sampler = MagicMock()
        generator.model = SimpleNamespace(
            config=SimpleNamespace(image_token_index=None)
        )
        generator._preprocess_request = lambda request: None
        generator._copy_prefix_cache = lambda cache: cache
        generator._prepare_rotating_caches = lambda cache: True
        generator._has_empty_rotating_cache = lambda cache: False
        monkeypatch.setattr(
            "mlx_lm.models.cache.make_prompt_cache",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_logits_processors",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_sampler",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(batch_module.mx, "clear_cache", lambda: None)
        monkeypatch.setattr(batch_module.mx, "stream", lambda _: nullcontext())

        request = MLLMBatchRequest(uid=21, request_id="same", prompt="prompt")
        request.input_ids = mx.array([[1, 2, 3, 4, 5]])
        request.is_text_only = True
        cached_entry = object()

        def fetch(input_ids):
            assert input_ids == [1, 2, 3, 4, 5]
            generator.abort_prefill(request.request_id, request.uid)
            replacement = MLLMBatchRequest(
                uid=22, request_id=request.request_id, prompt="replacement"
            )
            assert generator._consume_prefill_abort(replacement) is False
            return cached_entry, [6, 7, 8, 9, 10]

        generator.prefix_cache = SimpleNamespace(fetch=fetch)

        assert generator._process_prompts([request]) is None
        generator.language_model.assert_not_called()
        assert len(generator._pending_error_responses) == 1
        assert generator._pending_error_responses[0].finish_reason == "abort"
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()


class TestMLLMChunkedOwnership:
    def test_chunked_polling_consumes_uid_abort_and_cleans_partial(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        generator = _make_chunked_generator()
        install_chunked_prefill_mllm(generator, budget=2)
        monkeypatch.setattr(mx, "clear_cache", lambda: None)

        request = MLLMBatchRequest(uid=3, request_id="same", prompt="long text")
        generator._partial = {
            "request": request,
            "cache": [],
            "remaining_ids": mx.array([[3, 4]]),
            "processed": 2,
            "total": 4,
            "cached_count": 0,
            "chunk_count": 1,
        }
        generator._prefill_progress[request.request_id] = (2, 4)
        generator.abort_prefill(request.request_id, request.uid)

        replacement = MLLMBatchRequest(
            uid=4, request_id=request.request_id, prompt="replacement"
        )
        assert generator._consume_prefill_abort(replacement) is False

        responses = generator._next()

        abort_responses = [
            response for response in responses if response.finish_reason == "abort"
        ]
        assert len(abort_responses) == 1
        assert abort_responses[0].uid == request.uid
        assert abort_responses[0].request_id == request.request_id
        assert generator._partial is None
        assert request.request_id not in generator._prefill_progress
        assert request.uid not in generator._aborted_request_uids

    def test_successful_partial_removal_retires_marker_and_progress(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        class ActiveBatch:
            def __init__(self):
                self.uids = [4, 9]
                self.requests = [
                    SimpleNamespace(uid=4, request_id="active"),
                    SimpleNamespace(uid=9, request_id="keep"),
                ]

            def filter(self, keep_idx):
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = _make_chunked_generator()
        install_chunked_prefill_mllm(generator, budget=2)
        clear_cache = MagicMock()
        monkeypatch.setattr(mx, "clear_cache", clear_cache)
        partial_request = MLLMBatchRequest(
            uid=3, request_id="partial", prompt="long text"
        )
        generator._partial = {"request": partial_request}
        generator.active_batch = ActiveBatch()
        generator._aborted_request_ids = {"partial", "active", "keep"}
        generator._aborted_request_uids = {3, 4, 9}
        generator._prefill_progress = {
            "partial": (2, 5),
            "active": (1, 2),
            "keep": (1, 3),
        }

        generator.remove([3, 4])

        assert generator._partial is None
        assert generator.active_batch.uids == [9]
        assert generator._aborted_request_ids == {"keep"}
        assert generator._aborted_request_uids == {9}
        assert generator._prefill_progress == {"keep": (1, 3)}
        clear_cache.assert_called_once_with()

    def test_failed_partial_removal_preserves_ownership_for_retry(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        class FailingActiveBatch:
            def __init__(self):
                self.uids = [4, 9]
                self.requests = [
                    SimpleNamespace(uid=4, request_id="active"),
                    SimpleNamespace(uid=9, request_id="keep"),
                ]
                self.fail = True

            def filter(self, keep_idx):
                if self.fail:
                    self.fail = False
                    raise RuntimeError("filter failed")
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = _make_chunked_generator()
        install_chunked_prefill_mllm(generator, budget=2)
        clear_cache = MagicMock()
        monkeypatch.setattr(mx, "clear_cache", clear_cache)
        partial_request = MLLMBatchRequest(
            uid=3, request_id="partial", prompt="long text"
        )
        partial = {"request": partial_request}
        generator._partial = partial
        generator.active_batch = FailingActiveBatch()
        generator._aborted_request_ids = {"partial", "active"}
        generator._aborted_request_uids = {3, 4}
        generator._prefill_progress = {
            "partial": (2, 5),
            "active": (1, 2),
        }

        with pytest.raises(RuntimeError, match="filter failed"):
            generator.remove([3, 4])
        assert generator._partial is partial
        assert generator.active_batch.uids == [4, 9]
        assert generator._aborted_request_ids == {"partial", "active"}
        assert generator._aborted_request_uids == {3, 4}
        assert generator._prefill_progress == {
            "partial": (2, 5),
            "active": (1, 2),
        }
        clear_cache.assert_not_called()

        generator.remove([3, 4])
        assert generator._partial is None
        assert generator.active_batch.uids == [9]
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()
        assert generator._prefill_progress == {}
        clear_cache.assert_called_once_with()


class TestMLLMExpectedOwnerCleanup:
    def test_remove_finished_request_rejects_nonterminal_owner(self):
        from vllm_mlx.mllm_scheduler import MLLMRequest
        from vllm_mlx.request import RequestStatus

        scheduler = _make_scheduler()
        request = MLLMRequest(request_id="active", prompt="active")
        request.status = RequestStatus.RUNNING
        request.batch_uid = 7
        scheduler.requests = {"active": request}
        scheduler.running = {"active": request}
        scheduler.request_id_to_uid = {"active": 7}
        scheduler.uid_to_request_id = {7: "active"}

        assert scheduler.remove_finished_request("active") is None
        assert scheduler.requests == {"active": request}
        assert scheduler.running == {"active": request}
        assert scheduler.request_id_to_uid == {"active": 7}
        assert scheduler.uid_to_request_id == {7: "active"}

    def test_remove_finished_request_cleans_owner_and_allows_same_id_reuse(
        self, monkeypatch
    ):
        import vllm_mlx.mllm_scheduler as scheduler_module
        from vllm_mlx.mllm_scheduler import MLLMRequest
        from vllm_mlx.request import RequestStatus

        scheduler = _make_scheduler()
        request = MLLMRequest(request_id="same", prompt="old")
        request.status = RequestStatus.FINISHED_ABORTED
        request.batch_uid = 7
        scheduler.requests = {"same": request}
        scheduler.running = {"same": request}
        scheduler.request_id_to_uid = {"same": 7}
        scheduler.uid_to_request_id = {7: "same"}
        monkeypatch.setattr(scheduler_module.mx, "clear_cache", lambda: None)

        assert scheduler.remove_finished_request("same") is request
        assert scheduler.requests == {}
        assert scheduler.running == {}
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}

        replacement_id = scheduler.add_request("replacement", request_id="same")
        replacement = scheduler.requests["same"]
        assert replacement_id == "same"
        assert replacement is not request
        assert replacement.request_id == "same"

    def test_cleanup_finished_skips_replacement_after_stale_queue_interval(
        self, monkeypatch
    ):
        import vllm_mlx.mllm_scheduler as scheduler_module
        from vllm_mlx.mllm_scheduler import MLLMRequest

        scheduler = _make_scheduler()
        old_request = MLLMRequest(request_id="same", prompt="old")
        replacement = MLLMRequest(request_id="same", prompt="replacement")
        scheduler.requests = {"same": replacement}
        scheduler.running = {"same": replacement}
        scheduler.request_id_to_uid = {"same": 8}
        scheduler.uid_to_request_id = {8: "same"}
        monkeypatch.setattr(scheduler_module.mx, "clear_cache", lambda: None)

        scheduler._cleanup_finished(
            {"same"},
            expected_owners={"same": (old_request, 7)},
        )

        assert scheduler.requests == {"same": replacement}
        assert scheduler.running == {"same": replacement}
        assert scheduler.request_id_to_uid == {"same": 8}
        assert scheduler.uid_to_request_id == {8: "same"}


class TestMLLMBatchFilterOwnership:
    def test_filter_is_atomic_across_nested_cache_failure(self):
        from copy import deepcopy

        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        class OwnerCache:
            def __init__(self, label, owners, *, fail):
                self.label = label
                self.owners = list(owners)
                self.values = [f"{label}-{owner}" for owner in owners]
                self.state = {
                    "owners": list(owners),
                    "values": list(self.values),
                    "metadata": {"owner_uids": list(owners)},
                }
                self.fail = fail
                self.filter_calls = 0

            def filter(self, keep_idx):
                indices = [int(index) for index in keep_idx.tolist()]
                self.filter_calls += 1
                self.owners = [self.owners[index] for index in indices]
                self.values = [self.values[index] for index in indices]
                self.state["owners"] = [
                    self.state["owners"][index] for index in indices
                ]
                self.state["values"] = [
                    self.state["values"][index] for index in indices
                ]
                self.state["metadata"]["owner_uids"] = [
                    self.state["metadata"]["owner_uids"][index] for index in indices
                ]
                self.state["staged"] = True
                if self.fail:
                    raise RuntimeError("nested cache filter failed")

        class CacheListLike:
            def __init__(self, *caches):
                self.caches = tuple(caches)
                self.container_state = {
                    "owners": (41, 42),
                    "filter_generation": 0,
                }

            def filter(self, keep_idx):
                indices = [int(index) for index in keep_idx.tolist()]
                self.container_state["owners"] = tuple(
                    self.container_state["owners"][index] for index in indices
                )
                self.container_state["filter_generation"] += 1
                for cache in self.caches:
                    cache.filter(keep_idx)

        requests = [
            MLLMBatchRequest(uid=41, request_id="owner-a", prompt="a"),
            MLLMBatchRequest(uid=42, request_id="owner-b", prompt="b"),
        ]
        owner_by_uid = {request.uid: request.request_id for request in requests}
        failing_cache = OwnerCache("failing", [41, 42], fail=True)
        stable_cache = OwnerCache("stable", [41, 42], fail=False)
        cache_group = CacheListLike(failing_cache, stable_cache)
        batch = MLLMBatch(
            uids=[41, 42],
            request_ids=["owner-a", "owner-b"],
            y=mx.array([410, 420]),
            logprobs=["logprob-a", "logprob-b"],
            max_tokens=[128, 256],
            num_tokens=[3, 5],
            cache=[cache_group],
            requests=requests,
            logits_processors=[["processor-a"], ["processor-b"]],
            samplers=["sampler-a", "sampler-b"],
        )

        original_lists = {
            field: getattr(batch, field)
            for field in (
                "uids",
                "request_ids",
                "logprobs",
                "max_tokens",
                "num_tokens",
                "requests",
                "logits_processors",
                "samplers",
            )
        }
        original_list_values = {
            field: deepcopy(value) for field, value in original_lists.items()
        }
        original_y = batch.y
        original_y_value = batch.y.tolist()
        original_cache_list = batch.cache
        original_cache_group = cache_group
        original_cache_children = cache_group.caches
        original_failing_state = failing_cache.state
        original_stable_state = stable_cache.state
        original_cache_snapshot = deepcopy(
            {
                "group": cache_group.container_state,
                "failing": failing_cache.__dict__,
                "stable": stable_cache.__dict__,
            }
        )
        original_owner_inputs = [
            (request.uid, request.request_id) for request in requests
        ]

        with pytest.raises(RuntimeError, match="nested cache filter failed"):
            batch.filter([1])

        for field, original in original_lists.items():
            assert getattr(batch, field) is original
            assert getattr(batch, field) == original_list_values[field]
        assert batch.y is original_y
        assert batch.y.tolist() == original_y_value
        assert batch.cache is original_cache_list
        assert batch.cache[0] is original_cache_group
        assert batch.cache[0].caches is original_cache_children
        assert failing_cache.state is original_failing_state
        assert stable_cache.state is original_stable_state
        assert {
            "group": cache_group.container_state,
            "failing": failing_cache.__dict__,
            "stable": stable_cache.__dict__,
        } == original_cache_snapshot
        assert owner_by_uid == {41: "owner-a", 42: "owner-b"}
        assert [
            (request.uid, request.request_id) for request in requests
        ] == original_owner_inputs

        failing_cache.fail = False
        batch.filter([1])

        assert batch.uids == [42]
        assert batch.request_ids == ["owner-b"]
        assert batch.logprobs == ["logprob-b"]
        assert batch.max_tokens == [256]
        assert batch.num_tokens == [5]
        assert batch.requests == [requests[1]]
        assert batch.logits_processors == [["processor-b"]]
        assert batch.samplers == ["sampler-b"]
        assert batch.y.tolist() == [420]
        assert dict(zip(batch.uids, batch.request_ids)) == {42: "owner-b"}

        published_group = batch.cache[0]
        assert isinstance(published_group.caches, tuple)
        assert published_group.container_state == {
            "owners": (42,),
            "filter_generation": 1,
        }
        for cache in published_group.caches:
            assert cache.owners == [42]
            assert cache.values == [f"{cache.label}-42"]
            assert cache.state == {
                "owners": [42],
                "values": [f"{cache.label}-42"],
                "metadata": {"owner_uids": [42]},
                "staged": True,
            }
            assert cache.filter_calls == 1
