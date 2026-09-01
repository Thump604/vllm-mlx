# SPDX-License-Identifier: Apache-2.0
"""Real-call-site regressions for MLLM cache-owner lifecycle wiring."""

import ast
import asyncio
from contextlib import nullcontext
from collections import deque
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import MagicMock, call

import pytest
import mlx.core as mx

import vllm_mlx.cache_owner_identity as owner_identity_module
from vllm_mlx.cache_owner_identity import (
    CacheOwnerGovernanceTarget,
    OwnerBindingDecision,
    VerifiedCacheOwnerContext,
    build_loaded_cache_owner_digest,
    verify_loaded_model_cache_owner_context,
)
from vllm_mlx.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
)
from vllm_mlx.mllm_batch_generator import MLLMBatch
from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler, MLLMSchedulerConfig
from vllm_mlx.request import RequestStatus
from vllm_mlx.memory_cache import (
    MemoryCacheConfig,
    build_cache_owner_persistence_identity,
)
from vllm_mlx.scheduler import SchedulerConfig
from vllm_mlx.specprefill_contract import (
    build_draft_compatibility,
    build_identity_manifest,
    canonical_identity_bytes,
)


def _generator(*, owner_required: bool = True):
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.prefix_cache = MagicMock()
    generator._cache_owner_required = owner_required
    generator._cache_owner_requests = {}
    generator._cache_owner_lifecycle_mutating = False
    generator._prefix_checkpoint_lock = threading.RLock()
    generator._request_prefix_checkpoints = {}
    generator._aborted_request_ids = set()
    generator._pending_removal_lock = threading.Lock()
    generator._pending_removal_request_ids = {}
    generator._pending_removal_uids = set()
    generator.unprocessed_requests = []
    generator.active_batch = None
    generator.uid_counter = 0
    generator.completion_batch_size = 4
    generator._pending_error_responses = []
    return generator


def _request(request_id: str = "request-1") -> MLLMBatchRequest:
    return MLLMBatchRequest(uid=-1, request_id=request_id, prompt="hello")


def _configure_mock_insert_transaction(generator, *, minimum_uid: int = 0):
    transaction = MagicMock()
    transaction.__enter__.return_value = minimum_uid
    transaction.__exit__.return_value = False
    generator.insertion_transaction.return_value = transaction


def _batch_for_extend(*, uid: int, y, cache=None, request_id: str | None = None):
    request_id = request_id or f"request-{uid}"
    request = MLLMBatchRequest(uid=uid, request_id=request_id, prompt="hello")
    return MLLMBatch(
        uids=[uid],
        request_ids=[request_id],
        y=y,
        logprobs=[mx.zeros_like(y)],
        max_tokens=[8],
        num_tokens=[0],
        cache=[] if cache is None else cache,
        requests=[request],
    )


def test_batch_extend_shape_failure_is_transactional():
    batch = _batch_for_extend(uid=1, y=mx.array([[1, 2]]))
    other = _batch_for_extend(
        uid=2,
        y=mx.array([[3, 4, 5]]),
    )
    original = {
        "uids": list(batch.uids),
        "request_ids": list(batch.request_ids),
        "y": batch.y,
        "logprobs": list(batch.logprobs),
        "max_tokens": list(batch.max_tokens),
        "num_tokens": list(batch.num_tokens),
        "requests": list(batch.requests),
    }

    with pytest.raises(ValueError):
        batch.extend(other)

    assert batch.uids == original["uids"]
    assert batch.request_ids == original["request_ids"]
    assert batch.y is original["y"]
    assert batch.logprobs == original["logprobs"]
    assert batch.max_tokens == original["max_tokens"]
    assert batch.num_tokens == original["num_tokens"]
    assert batch.requests == original["requests"]


def test_batch_extend_cache_failure_propagates_and_rolls_back():
    class FailingCache:
        def __init__(self, value):
            self.cache = [value]
            self.extend_calls = 0

        def empty(self):
            return False

        def extend(self, other):
            self.extend_calls += 1
            self.cache.extend(other.cache)
            raise RuntimeError("cache extension failed")

    cache = FailingCache("original")
    batch = _batch_for_extend(
        uid=1,
        y=mx.array([1]),
        cache=[cache],
    )
    other = _batch_for_extend(
        uid=2,
        y=mx.array([2]),
        cache=[FailingCache("other")],
    )
    original_uids = list(batch.uids)
    original_y = batch.y

    with pytest.raises(RuntimeError, match="cache extension failed"):
        batch.extend(other)

    assert batch.uids == original_uids
    assert batch.y is original_y
    assert cache.extend_calls == 0
    assert cache.cache == ["original"]


def test_batch_extend_later_cache_failure_rolls_back_every_layer():
    class Cache:
        def __init__(self, value, *, fail=False):
            self.cache = [value]
            self.fail = fail

        def empty(self):
            return False

        def extend(self, other):
            self.cache.extend(other.cache)
            if self.fail:
                raise RuntimeError("later cache extension failed")

    first = Cache("first")
    second = Cache("second", fail=True)
    batch = _batch_for_extend(uid=1, y=mx.array([1]), cache=[first, second])
    other = _batch_for_extend(
        uid=2,
        y=mx.array([2]),
        cache=[Cache("other-first"), Cache("other-second")],
    )

    with pytest.raises(RuntimeError, match="later cache extension failed"):
        batch.extend(other)

    assert first.cache == ["first"]
    assert second.cache == ["second"]
    assert batch.uids == [1]
    assert batch.request_ids == ["request-1"]


def test_batch_extend_base_exception_rolls_back_every_layer():
    class Cache:
        def __init__(self, value, *, fail=False):
            self.cache = [value]
            self.fail = fail

        def empty(self):
            return False

        def extend(self, other):
            self.cache.extend(other.cache)
            if self.fail:
                raise KeyboardInterrupt("cache extension interrupted")

    first = Cache("first")
    second = Cache("second", fail=True)
    batch = _batch_for_extend(uid=1, y=mx.array([1]), cache=[first, second])
    other = _batch_for_extend(
        uid=2,
        y=mx.array([2]),
        cache=[Cache("other-first"), Cache("other-second")],
    )

    with pytest.raises(KeyboardInterrupt, match="cache extension interrupted"):
        batch.extend(other)

    assert first.cache == ["first"]
    assert second.cache == ["second"]
    assert batch.uids == [1]
    assert batch.request_ids == ["request-1"]


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([None], [], "different cache layer counts"),
        ([], [None], "different cache layer counts"),
        ([None], [object()], "mismatched empty cache layers"),
        ([object()], [None], "mismatched empty cache layers"),
    ],
)
def test_batch_extend_rejects_incompatible_cache_topology(left, right, message):
    batch = _batch_for_extend(uid=1, y=mx.array([1]), cache=left)
    other = _batch_for_extend(uid=2, y=mx.array([2]), cache=right)

    with pytest.raises(ValueError, match=message):
        batch.extend(other)

    assert batch.uids == [1]
    assert batch.request_ids == ["request-1"]


def _chunked_generator():
    from vllm_mlx.mllm_batch_generator import MLLMBatchStats

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator._stats = MLLMBatchStats()
    generator._pending_error_responses = []
    generator._aborted_request_ids = set()
    generator._prefill_progress = {}
    generator._prefix_checkpoint_lock = threading.RLock()
    generator._request_prefix_checkpoints = {}
    generator._cache_owner_required = True
    generator._cache_owner_requests = {}
    generator._cache_owner_lifecycle_mutating = False
    generator._cache_owner_context = None
    generator.active_batch = None
    generator.unprocessed_requests = []
    generator.uid_counter = 0
    generator.stop_tokens = set()
    generator._think_suffix_len = 0
    generator.max_kv_size = 0
    generator._allow_mid_batch_extend = True
    generator._require_uniform_mllm_draft = False
    generator._has_empty_rotating_cache = lambda _cache: False
    generator._compatible_pending_requests = (
        lambda requests, limit, reference=None: requests[:limit]
    )
    generator._derive_request_rope_deltas = lambda _request: None
    generator._batch_rope_deltas = lambda _requests: None
    generator._language_model_kwargs = lambda *args, **kwargs: {}
    generator._prefill_checkpoint_plan = lambda *args, **kwargs: (None, None)
    generator._prepare_rotating_caches = lambda _cache: True
    generator._maybe_store_prefix_cache = lambda *args: None
    generator._step = lambda *args, **kwargs: (mx.array([0]), [mx.zeros(4)])
    generator._next = lambda: []
    return generator


def test_chunked_next_preprocess_failure_releases_owner_lease():
    from vllm_mlx.mllm_batch_generator import install_chunked_prefill_mllm

    generator = _chunked_generator()
    release = MagicMock()
    generator.prefix_cache = SimpleNamespace(release_owner_request=release)
    binding = object()
    request = MLLMBatchRequest(uid=1, request_id="chunked-preprocess", prompt="hello")
    generator._cache_owner_requests[request.request_id] = binding
    generator.unprocessed_requests.append(request)
    generator._preprocess_request = MagicMock(
        side_effect=RuntimeError("preprocess failed")
    )

    install_chunked_prefill_mllm(generator, budget=2)

    responses = generator._next()

    assert len(responses) == 1
    assert responses[0].request_id == request.request_id
    assert responses[0].finish_reason == "error"
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    release.assert_called_once_with(binding)


def test_chunked_next_extend_failure_is_transactional_and_releases_new_lease():
    from vllm_mlx.mllm_batch_generator import install_chunked_prefill_mllm

    class MutatingFailingCache:
        def __init__(self, value):
            self.cache = [value]
            self.extend_calls = 0

        def empty(self):
            return False

        def extend(self, other):
            self.extend_calls += 1
            self.cache.extend(other.cache)
            raise RuntimeError("chunked cache extension failed")

    generator = _chunked_generator()
    release = MagicMock()
    generator.prefix_cache = SimpleNamespace(release_owner_request=release)

    active = _batch_for_extend(
        uid=90, y=mx.array([9]), cache=[MutatingFailingCache("active")]
    )
    generator.active_batch = active
    original_uids = list(active.uids)
    original_request_ids = list(active.request_ids)
    active_cache = active.cache[0]

    partial_request = MLLMBatchRequest(
        uid=91,
        request_id="chunked-long",
        prompt="long",
        max_tokens=4,
    )
    partial_state = {
        "request": partial_request,
        "cache": [object()],
        "remaining_ids": mx.array([[1, 2, 3]]),
        "processed": 2,
        "total": 5,
        "cached_count": 0,
        "chunk_count": 1,
        "checkpoint_at": None,
        "checkpoint_key": None,
        "checkpoint_entry": None,
    }

    inline_binding = object()
    inline_request = MLLMBatchRequest(
        uid=92,
        request_id="chunked-inline",
        prompt="short",
    )
    inline_request.input_ids = mx.array([[7]])
    generator._cache_owner_requests[inline_request.request_id] = inline_binding
    generator.unprocessed_requests.append(inline_request)
    new_batch = _batch_for_extend(
        uid=inline_request.uid,
        y=mx.array([7]),
        cache=[MutatingFailingCache("new")],
        request_id=inline_request.request_id,
    )
    generator._process_prompts = MagicMock(return_value=new_batch)
    generator.language_model = lambda tokens, cache, **kwargs: mx.zeros(
        (1, tokens.shape[1], 4)
    )

    install_chunked_prefill_mllm(generator, budget=2)
    generator._partial = partial_state

    responses = generator._next()

    assert any(
        response.request_id == inline_request.request_id
        and response.finish_reason == "error"
        for response in responses
    )
    assert active.uids == original_uids
    assert active.request_ids == original_request_ids
    assert active_cache.extend_calls == 0
    assert active_cache.cache == ["active"]
    assert generator._cache_owner_requests == {}
    release.assert_called_once_with(inline_binding)


def test_mid_batch_partial_preprocess_then_extend_failure_releases_selection_once():
    generator = _chunked_generator()
    generator.completion_batch_size = 4
    release = MagicMock()
    generator.prefix_cache = SimpleNamespace(release_owner_request=release)
    generator.active_batch = _batch_for_extend(uid=90, y=mx.array([9]))
    first = _request("first-fails-preprocess")
    first.uid = 91
    second = _request("second-fails-extend")
    second.uid = 92
    first_binding = object()
    second_binding = object()
    generator._cache_owner_requests = {
        first.request_id: first_binding,
        second.request_id: second_binding,
    }
    generator.unprocessed_requests = [first, second]

    def partially_process_then_fail(requests):
        failed = requests.pop(0)
        generator._release_cache_owner_request(failed.request_id)
        generator._pending_error_responses.append(
            MLLMBatchResponse(
                uid=failed.uid,
                request_id=failed.request_id,
                token=0,
                logprobs=mx.zeros(1),
                finish_reason="error",
            )
        )
        raise RuntimeError("extend preparation failed")

    generator._process_prompts = partially_process_then_fail

    responses = MLLMBatchGenerator._next(generator)

    error_ids = [
        response.request_id
        for response in responses
        if response.finish_reason == "error"
    ]
    assert error_ids.count(first.request_id) == 1
    assert error_ids.count(second.request_id) == 1
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    assert release.call_args_list == [call(first_binding), call(second_binding)]


def test_mid_batch_base_exception_releases_selected_request_before_propagating():
    generator = _chunked_generator()
    generator.completion_batch_size = 4
    release = MagicMock()
    generator.prefix_cache = SimpleNamespace(release_owner_request=release)
    generator.active_batch = _batch_for_extend(uid=90, y=mx.array([9]))
    request = _request("keyboard-mid-batch")
    request.uid = 91
    binding = object()
    generator._cache_owner_requests = {request.request_id: binding}
    generator.unprocessed_requests = [request]
    generator._process_prompts = MagicMock(
        side_effect=KeyboardInterrupt("prompt processing interrupted")
    )

    with pytest.raises(KeyboardInterrupt, match="prompt processing interrupted"):
        MLLMBatchGenerator._next(generator)

    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    release.assert_called_once_with(binding)
    assert [response.request_id for response in generator._pending_error_responses] == [
        request.request_id
    ]


def test_mid_batch_extend_base_exception_rolls_back_and_releases_request():
    class InterruptingCache:
        def __init__(self, value, *, fail=False):
            self.cache = [value]
            self.fail = fail

        def empty(self):
            return False

        def extend(self, other):
            self.cache.extend(other.cache)
            if self.fail:
                raise KeyboardInterrupt("batch extension interrupted")

    generator = _chunked_generator()
    generator.completion_batch_size = 4
    release = MagicMock()
    generator.prefix_cache = SimpleNamespace(release_owner_request=release)
    active_cache = InterruptingCache("active", fail=True)
    generator.active_batch = _batch_for_extend(
        uid=90, y=mx.array([9]), cache=[active_cache]
    )
    request = _request("keyboard-extend")
    request.uid = 91
    binding = object()
    generator._cache_owner_requests = {request.request_id: binding}
    generator.unprocessed_requests = [request]
    generator._process_prompts = MagicMock(
        return_value=_batch_for_extend(
            uid=request.uid,
            y=mx.array([7]),
            cache=[InterruptingCache("new")],
            request_id=request.request_id,
        )
    )

    with pytest.raises(KeyboardInterrupt, match="batch extension interrupted"):
        MLLMBatchGenerator._next(generator)

    assert active_cache.cache == ["active"]
    assert generator.active_batch.uids == [90]
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    release.assert_called_once_with(binding)


def test_chunked_next_success_retains_owner_lease_until_completion():
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.mllm_batch_generator import install_chunked_prefill_mllm

    generator = _chunked_generator()
    generator.prefix_cache = SimpleNamespace()
    generator._fetch_exact_prefix_auxiliary = lambda request_id, tokens: None
    generator._fetch_prefix_cache = lambda request_id, tokens: (None, tokens)
    generator._store_prefix_snapshot = MagicMock()
    generator._preprocess_request = lambda _request: None

    binding = object()
    request = MLLMBatchRequest(
        uid=3,
        request_id="chunked-success",
        prompt="long",
        max_tokens=4,
    )
    request.input_ids = mx.array([[1, 2, 3, 4, 5, 6]])
    request.is_text_only = True
    generator._cache_owner_requests[request.request_id] = binding
    generator.unprocessed_requests.append(request)

    def make_prompt_cache(*args, **kwargs):
        return [KVCache()]

    generator.language_model = lambda tokens, cache, **kwargs: (
        cache[0].update_and_fetch(
            mx.zeros((1, 1, tokens.shape[1], 1)),
            mx.zeros((1, 1, tokens.shape[1], 1)),
        ),
        mx.zeros((1, tokens.shape[1], 4)),
    )[1]

    import mlx_lm.models.cache as mlx_lm_cache

    original_make_prompt_cache = mlx_lm_cache.make_prompt_cache
    mlx_lm_cache.make_prompt_cache = make_prompt_cache
    try:
        install_chunked_prefill_mllm(generator, budget=2)

        assert generator._next() == []
        assert generator._next() == []
        responses = generator._next()
    finally:
        mlx_lm_cache.make_prompt_cache = original_make_prompt_cache

    assert len(responses) == 1
    assert responses[0].request_id == request.request_id
    assert responses[0].finish_reason is None
    assert generator._cache_owner_requests == {request.request_id: binding}


def test_chunked_remove_releases_partial_owner_before_reinsert():
    from vllm_mlx.mllm_batch_generator import install_chunked_prefill_mllm

    generator = _chunked_generator()
    prefix_cache = MagicMock()
    binding = object()
    replacement_binding = object()
    prefix_cache.mint_owner_request.return_value = replacement_binding
    generator.prefix_cache = prefix_cache

    partial_request = MLLMBatchRequest(
        uid=11,
        request_id="chunked-drain-reinsert",
        prompt="long",
    )
    generator._cache_owner_requests[partial_request.request_id] = binding

    install_chunked_prefill_mllm(generator, budget=2)
    generator._partial = {
        "request": partial_request,
        "remaining_ids": mx.array([[1]]),
    }
    generator.remove([partial_request.uid])

    assert generator._partial is None
    assert generator._cache_owner_requests == {}
    prefix_cache.release_owner_request.assert_called_once_with(binding)

    replacement = MLLMBatchRequest(
        uid=-1,
        request_id=partial_request.request_id,
        prompt="retry",
    )
    assert generator.insert([replacement]) == [0]
    assert generator._cache_owner_requests == {
        partial_request.request_id: replacement_binding
    }


def test_chunked_exact_auxiliary_failure_releases_owner_lease():
    from vllm_mlx.mllm_batch_generator import install_chunked_prefill_mllm

    generator = _chunked_generator()
    binding = object()
    release = MagicMock()
    generator.prefix_cache = SimpleNamespace(release_owner_request=release)
    generator._fetch_exact_prefix_auxiliary = lambda request_id, tokens: {
        "last_logits": mx.zeros((1, 4))
    }
    generator._preprocess_request = lambda _request: None
    generator._derive_request_rope_deltas = lambda _request: None
    generator._process_prompts = MagicMock(
        side_effect=RuntimeError("exact auxiliary processing failed")
    )

    request = MLLMBatchRequest(
        uid=12,
        request_id="chunked-exact-auxiliary-failure",
        prompt="cached",
    )
    request.input_ids = mx.array([[1, 2]])
    generator._cache_owner_requests[request.request_id] = binding
    generator.unprocessed_requests.append(request)

    install_chunked_prefill_mllm(generator, budget=2)
    responses = generator._next()

    assert len(responses) == 1
    assert responses[0].request_id == request.request_id
    assert responses[0].finish_reason == "error"
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    release.assert_called_once_with(binding)


def test_insert_mints_owner_request_and_abort_revokes_it():
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding

    request = _request()
    assert generator.insert([request]) == [0]
    generator.prefix_cache.mint_owner_request.assert_called_once_with(0)
    assert generator._cache_owner_requests == {request.request_id: binding}

    generator.abort_prefill(request.request_id)
    generator.prefix_cache.cancel_owner_request.assert_called_once_with(binding)


def test_waiting_abort_does_not_poison_reused_request_id(monkeypatch):
    """A queued abort must not mark a later same-ID request as aborted."""
    from vllm_mlx.mllm_batch_generator import MLLMBatchStats

    generator = _generator(owner_required=False)
    generator.prefix_cache = None
    generator._stats = MLLMBatchStats()
    generator.max_kv_size = 0
    generator.processor = SimpleNamespace()
    generator.prefill_step_size = 512
    generator.language_model = object()
    generator.model = object()
    generator.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
    generator.stop_tokens = set()

    def preprocess(request):
        request.input_ids = mx.array([[1, 2]])
        request.is_text_only = True

    generator._preprocess_request = preprocess
    generator._derive_request_rope_deltas = lambda request: None
    generator._run_chunked_text_prefill = lambda request, cache: mx.array(
        [[[0.0, 1.0]]]
    )
    monkeypatch.setattr(mx, "stream", lambda stream: nullcontext())

    class MergeableCache:
        def merge(self, caches):
            return self

    monkeypatch.setattr(
        "mlx_lm.models.cache.make_prompt_cache",
        lambda *args, **kwargs: [MergeableCache()],
    )
    monkeypatch.setattr(
        "mlx_lm.sample_utils.make_logits_processors", lambda **kwargs: []
    )
    monkeypatch.setattr(
        "mlx_lm.sample_utils.make_sampler",
        lambda **kwargs: generator.sampler,
    )

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.batch_generator = generator
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=1,
        cache_owner_context=None,
    )
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.waiting = deque()
    scheduler.running = {}
    scheduler.requests = {}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler.finished_req_ids = set()
    scheduler.output_queues = {}
    scheduler._detokenizer_pool = {}

    first = MLLMRequest(request_id="reused-id", prompt="first")
    scheduler.requests[first.request_id] = first
    scheduler.waiting.append(first)
    assert scheduler._abort_request_locked(first.request_id)
    assert generator._aborted_request_ids == set()

    replacement = MLLMRequest(request_id="reused-id", prompt="replacement")
    scheduler.requests[replacement.request_id] = replacement
    scheduler.waiting.append(replacement)

    scheduled = scheduler._schedule_waiting()
    assert scheduled == [replacement]
    assert generator.unprocessed_requests[0].request_id == replacement.request_id
    assert (
        MLLMBatchGenerator._process_prompts(generator, generator.unprocessed_requests)
        is not None
    )


def test_running_abort_still_signals_batch_generator_prefill_abort():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    generator = MagicMock()
    request = MLLMRequest(request_id="running-id", prompt="running")
    request.status = RequestStatus.RUNNING
    scheduler.batch_generator = generator
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.waiting = deque()
    scheduler.running = {request.request_id: request}
    scheduler.requests = {request.request_id: request}
    scheduler.request_id_to_uid = {request.request_id: 7}
    scheduler.uid_to_request_id = {7: request.request_id}
    scheduler.finished_req_ids = set()
    scheduler.output_queues = {}
    scheduler._detokenizer_pool = {}

    assert scheduler._abort_request_locked(request.request_id)
    generator.abort_prefill.assert_called_once_with(request.request_id)


def test_running_abort_marker_clears_at_model_thread_removal_for_reuse():
    generator = _generator(owner_required=False)
    request_id = "running-reuse"
    generator.active_batch = _batch_for_extend(
        uid=7,
        y=mx.array([1]),
        request_id=request_id,
    )

    generator.abort_prefill(request_id)
    assert request_id in generator._aborted_request_ids
    generator.schedule_removal([7])
    generator.process_pending_removals()

    assert request_id not in generator._aborted_request_ids
    assert generator.insert([_request(request_id)]) == [0]


def test_partial_abort_marker_clears_after_deferred_removal_for_reuse():
    generator = _generator(owner_required=False)
    request_id = "partial-reuse"
    request = _request(request_id)
    request.uid = 11
    generator._partial = {"request": request}

    generator.abort_prefill(request_id)
    assert request_id in generator._aborted_request_ids
    generator.schedule_removal([request.uid])
    generator.process_pending_removals()

    assert request_id not in generator._aborted_request_ids
    assert generator.insert([_request(request_id)]) == [0]


def test_atomic_prefill_abort_marker_clears_after_removal_for_reuse():
    generator = _generator(owner_required=False)
    generator.prefix_cache.commit_prepared.return_value = True
    request_id = "atomic-reuse"
    request = _request(request_id)
    request.uid = 13
    generator.unprocessed_requests.append(request)

    assert generator._publish_prefill_checkpoint(
        request_id,
        SimpleNamespace(tokens=(1, 2, 3)),
    )
    generator.abort_prefill(request_id)
    assert request_id in generator._aborted_request_ids
    generator.schedule_removal([request.uid])
    generator.process_pending_removals()

    assert request_id not in generator._aborted_request_ids
    assert generator.insert([_request(request_id)]) == [0]


def test_pending_old_uid_blocks_same_id_replacement_and_preserves_other_owner():
    generator = _generator()
    old_binding = object()
    other_binding = object()
    replacement_binding = object()
    generator.prefix_cache.mint_owner_request.side_effect = [
        old_binding,
        other_binding,
        replacement_binding,
    ]
    old = _request("generation-safe")
    other = _request("unrelated")
    assert generator.insert([old, other]) == [0, 1]

    generator.abort_prefill(old.request_id)
    generator._aborted_request_ids.add(other.request_id)
    # Partial prefill retires the owner lease before deferred UID removal.
    generator._release_cache_owner_request(old.request_id)
    generator.schedule_removal([old.uid])
    replacement = _request(old.request_id)

    with pytest.raises(RuntimeError, match="pending removal"):
        generator.insert([replacement])

    assert old.request_id in generator._aborted_request_ids
    assert replacement.uid == -1
    assert generator.uid_counter == 2
    assert replacement not in generator.unprocessed_requests
    assert generator._cache_owner_requests == {other.request_id: other_binding}

    generator.process_pending_removals()

    assert old.request_id not in generator._aborted_request_ids
    assert other.request_id in generator._aborted_request_ids
    assert generator._pending_removal_request_ids == {}
    assert generator._cache_owner_requests == {other.request_id: other_binding}
    assert generator.insert([replacement]) == [2]
    assert generator._cache_owner_requests == {
        other.request_id: other_binding,
        replacement.request_id: replacement_binding,
    }


def test_scheduler_rejects_same_id_until_old_uid_is_removed():
    generator = _generator(owner_required=False)
    old = _batch_for_extend(
        uid=7,
        y=mx.array([1]),
        request_id="scheduler-generation-safe",
    )
    generator.active_batch = old

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.batch_generator = generator
    scheduler.config = MLLMSchedulerConfig(max_num_seqs=1)
    scheduler.processor = SimpleNamespace(encode=lambda prompt: [])
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.waiting = deque()
    request = MLLMRequest(request_id=old.requests[0].request_id, prompt="hello")
    request.status = RequestStatus.RUNNING
    scheduler.running = {request.request_id: request}
    scheduler.requests = dict(scheduler.running)
    scheduler.request_id_to_uid = {request.request_id: old.uids[0]}
    scheduler.uid_to_request_id = {old.uids[0]: request.request_id}
    scheduler.finished_req_ids = set()
    scheduler.output_queues = {}
    scheduler._detokenizer_pool = {}

    request_id = request.request_id
    assert scheduler._abort_request_locked(request_id)
    with pytest.raises(ValueError, match="pending removal"):
        scheduler.add_request("replacement", request_id=request_id)
    assert request_id not in scheduler.requests
    assert list(scheduler.waiting) == []

    generator.process_pending_removals()
    assert scheduler.add_request("replacement", request_id=request_id) == request_id


def test_remove_releases_owner_request_after_scheduler_thread_removal():
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    request = _request()
    generator.insert([request])

    generator.remove([request.uid])

    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert generator._cache_owner_requests == {}


def test_partial_insert_failure_releases_every_minted_owner_request():
    generator = _generator()
    first = object()
    generator.prefix_cache.mint_owner_request.side_effect = [
        first,
        RuntimeError("mint failed"),
    ]

    with pytest.raises(RuntimeError, match="mint failed"):
        generator.insert([_request("request-1"), _request("request-2")])

    generator.prefix_cache.release_owner_request.assert_called_once_with(first)
    assert generator._cache_owner_requests == {}
    assert generator.unprocessed_requests == []
    assert generator.uid_counter == 0


def test_preprocess_failure_releases_owner_request_immediately(monkeypatch):
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    request = _request()
    generator.insert([request])
    monkeypatch.setattr(
        generator,
        "_preprocess_request",
        MagicMock(side_effect=RuntimeError("bad input")),
    )

    assert generator._process_prompts([request]) is None
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert generator._cache_owner_requests == {}


def test_inline_failure_removes_request_and_releases_owner_once():
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    request = _request()
    generator.insert([request])

    generator._fail_inline_requests([request])

    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert [response.request_id for response in generator._pending_error_responses] == [
        request.request_id
    ]


def test_checkpoint_prepare_and_publication_use_same_owner_binding():
    generator = _generator()
    binding = object()
    prepared = SimpleNamespace(tokens=(1, 2, 3))
    cache_state = [object()]
    auxiliary = {"last_logits": object()}
    generator._cache_owner_requests["request-1"] = binding
    generator.prefix_cache.prepare_owner_bound_store.return_value = (
        OwnerBindingDecision(True, "none"),
        prepared,
    )
    generator.prefix_cache.commit_owner_bound_store.return_value = OwnerBindingDecision(
        True, "none"
    )

    entry = generator._prepare_prefix_store(
        "request-1",
        [1, 2, 3],
        cache_state,
        auxiliary=auxiliary,
        persistence_eligible=True,
    )
    assert entry is prepared
    generator.prefix_cache.prepare_owner_bound_store.assert_called_once_with(
        binding,
        [1, 2, 3],
        cache_state,
        auxiliary=auxiliary,
        persistence_eligible=True,
    )

    assert generator._publish_prefill_checkpoint("request-1", entry) is True
    call = generator.prefix_cache.commit_owner_bound_store.call_args
    assert call.args == (prepared,)
    assert call.kwargs["evict_prefixes"] is False
    assert call.kwargs["commit_lock"] is generator._prefix_checkpoint_lock
    assert callable(call.kwargs["commit_guard"])


def test_owner_required_missing_request_binding_fails_closed():
    generator = _generator()

    with pytest.raises(RuntimeError, match="owner-bound request"):
        generator._prepare_prefix_store(
            "missing",
            [1],
            [object()],
        )


def test_legacy_cache_path_remains_explicit_when_owner_binding_is_disabled():
    generator = _generator(owner_required=False)
    entry = SimpleNamespace(tokens=(1, 2, 3))
    generator.prefix_cache.prepare_store.return_value = entry
    generator.prefix_cache.commit_prepared.return_value = True

    prepared = generator._prepare_prefix_store(
        "legacy",
        [1, 2, 3],
        [object()],
    )
    assert prepared is entry
    assert generator._publish_prefill_checkpoint("legacy", prepared) is True
    generator.prefix_cache.prepare_owner_bound_store.assert_not_called()
    generator.prefix_cache.commit_owner_bound_store.assert_not_called()


def test_close_revokes_owner_and_discards_request_handles():
    generator = _generator()
    generator._old_wired_limit = None
    generator._cache_owner_requests["request-1"] = object()

    generator.close()

    generator.prefix_cache.close_owner_identity.assert_called_once_with()
    assert generator._cache_owner_requests == {}


def test_close_excludes_insert_until_owner_identity_is_terminally_closed():
    generator = _generator()
    generator._old_wired_limit = None
    close_entered = threading.Event()
    release_close = threading.Event()
    insert_finished = threading.Event()
    close_errors = []
    insert_errors = []

    def block_identity_close():
        close_entered.set()
        assert release_close.wait(timeout=2)

    generator.prefix_cache.close_owner_identity.side_effect = block_identity_close

    def close_generator():
        try:
            generator.close()
        except BaseException as exc:  # pragma: no cover - assertion below
            close_errors.append(exc)

    def insert_request():
        try:
            generator.insert([_request("after-close")])
        except BaseException as exc:  # pragma: no cover - assertion below
            insert_errors.append(exc)
        finally:
            insert_finished.set()

    close_worker = threading.Thread(target=close_generator)
    insert_worker = threading.Thread(target=insert_request)
    close_worker.start()
    assert close_entered.wait(timeout=2)
    insert_worker.start()
    assert insert_finished.wait(timeout=0.05) is False
    release_close.set()
    close_worker.join(timeout=2)
    insert_worker.join(timeout=2)

    assert close_errors == []
    assert len(insert_errors) == 1
    assert isinstance(insert_errors[0], RuntimeError)
    assert "terminally failed" in str(insert_errors[0])
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}


def test_close_base_exception_releases_all_resources_and_leaves_terminal_owner_state(
    monkeypatch,
):
    generator = _generator()
    generator._old_wired_limit = 123
    synchronize = MagicMock()
    set_wired_limit = MagicMock()
    monkeypatch.setattr(mx, "synchronize", synchronize)
    monkeypatch.setattr(mx, "set_wired_limit", set_wired_limit)
    binding = object()
    generator._cache_owner_requests["request-1"] = binding
    generator.prefix_cache.close_owner_identity.side_effect = KeyboardInterrupt(
        "close interrupted"
    )

    with pytest.raises(KeyboardInterrupt, match="close interrupted"):
        generator.close()

    assert generator._cache_owner_lifecycle_mutating is False
    assert generator._cache_owner_lifecycle_failed is True
    assert generator._cache_owner_requests == {}
    assert generator._old_wired_limit is None
    synchronize.assert_called_once_with(MLLMBatchGenerator._stream)
    set_wired_limit.assert_called_once_with(123)
    generator.prefix_cache.cancel_owner_request.assert_called_once_with(binding)
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)

    insert_finished = threading.Event()
    insert_errors = []

    def insert_after_failed_close():
        try:
            generator.insert([_request("after-failed-close")])
        except BaseException as exc:  # pragma: no cover - assertion below
            insert_errors.append(exc)
        finally:
            insert_finished.set()

    worker = threading.Thread(target=insert_after_failed_close)
    worker.start()
    assert insert_finished.wait(timeout=2)
    worker.join(timeout=2)

    assert len(insert_errors) == 1
    assert isinstance(insert_errors[0], RuntimeError)
    assert "terminally failed" in str(insert_errors[0])


def test_cache_lifecycle_revokes_requests_then_rebinds_verified_context():
    generator = _generator()
    binding = object()
    context = object()
    generator._cache_owner_context = context
    generator._cache_owner_requests["request-1"] = binding

    generator.invalidate_cache_owner_requests()
    generator.rebind_cache_owner_context()

    generator.prefix_cache.cancel_owner_request.assert_called_once_with(binding)
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    generator.prefix_cache.bind_owner_context.assert_called_once_with(context)


def test_owner_bound_fetch_and_completion_release_use_live_request_handle():
    generator = _generator()
    binding = object()
    generator._cache_owner_requests["request-1"] = binding
    generator.prefix_cache.fetch_owner_bound.return_value = (
        OwnerBindingDecision(True, "none"),
        ["cache"],
        [3],
    )

    assert generator._fetch_prefix_cache("request-1", [1, 2, 3]) == (
        ["cache"],
        [3],
    )
    generator.prefix_cache.fetch_owner_bound.assert_called_once_with(binding, [1, 2, 3])

    request = _request()
    request.input_ids = None
    batch = SimpleNamespace(requests=[request])
    generator._maybe_store_prefix_cache(batch, [0])
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert generator._cache_owner_requests == {}


def test_generator_has_no_direct_cache_publication_or_fetch_bypass():
    source = Path(MLLMBatchGenerator.__module__.replace(".", "/") + ".py")
    source = Path(__file__).parents[1] / source
    tree = ast.parse(source.read_text())
    calls: list[tuple[str, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.functions: list[str] = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "prepare_store",
                "commit_prepared",
                "store",
                "fetch",
                "fetch_exact_auxiliary",
                "clone_for_replay",
            }:
                calls.append((self.functions[-1], node.func.attr))
            self.generic_visit(node)

    Visitor().visit(tree)
    assert set(calls) <= {
        ("_prepare_prefix_store", "prepare_store"),
        ("_publish_prefill_checkpoint", "commit_prepared"),
        ("_store_prefix_snapshot", "store"),
        ("_fetch_prefix_cache", "fetch"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_runtime_composition_digest() -> str:
    """Independently hash declared runtime files and installed dependencies."""
    module_names = (
        "vllm_mlx.cache_owner_identity",
        "vllm_mlx.memory_cache",
        "vllm_mlx.mllm_batch_generator",
        "vllm_mlx.mllm_scheduler",
        "vllm_mlx.scheduler",
        "vllm_mlx.engine.batched",
        "vllm_mlx.models.mllm",
    )
    repo_root = Path(__file__).parents[1]
    module_digests = {
        f"module:{module_name}": _sha256(
            repo_root / Path(*module_name.split(".")).with_suffix(".py")
        )
        for module_name in module_names
    }
    dependency_versions = {
        f"dependency:{distribution}": importlib_metadata.version(distribution)
        for distribution in ("mlx", "mlx-lm", "mlx-vlm", "vllm-mlx")
    }
    return hashlib.sha256(
        json.dumps(
            {**module_digests, **dependency_versions},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def test_owner_cache_constructor_recomputes_and_validates_persistence_identity(
    tmp_path, monkeypatch
):
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    context = verify_loaded_model_cache_owner_context(
        model,
        processor,
        str(tmp_path),
        target,
        cache_config=cache_config,
        cache_runtime_identity={"max_kv_size": 1},
        runtime_mode=mode,
    )
    cache_kwargs = dict(
        model=model.language_model,
        config=cache_config,
        tokenizer=processor.tokenizer,
        model_identity=str(tmp_path),
        cache_runtime_identity={"max_kv_size": 1},
        template_renderer=processor,
    )

    cache = MemoryAwarePrefixCache(
        **cache_kwargs,
        cache_owner_context=context,
    )
    assert dict(cache._persistence_identity) == dict(context.persistence_identity)

    mismatched_identity = dict(context.persistence_identity)
    mismatched_identity["model"] = "0" * 64
    mismatched_context = SimpleNamespace(persistence_identity=mismatched_identity)
    with pytest.raises(ValueError, match="verified cache owner provenance"):
        MemoryAwarePrefixCache(
            **cache_kwargs,
            cache_owner_context=mismatched_context,
        )


def test_owner_generic_persistence_requires_exact_identity(tmp_path, monkeypatch):
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache

    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    context = verify_loaded_model_cache_owner_context(
        model,
        processor,
        str(tmp_path),
        target,
        cache_config=cache_config,
        cache_runtime_identity={"max_kv_size": 1},
        runtime_mode=mode,
    )
    cache = MemoryAwarePrefixCache(
        model.language_model,
        cache_config,
        tokenizer=processor.tokenizer,
        model_identity=str(tmp_path),
        cache_runtime_identity={"max_kv_size": 1},
        template_renderer=processor,
        cache_owner_context=context,
    )
    index = {
        "version": 4,
        "model_fingerprint": cache._model_fingerprint,
        "num_entries": 0,
        "total_memory_bytes": 0,
        "entries": [],
    }
    (tmp_path / "index.json").write_text(json.dumps(index))
    cache.invalidate_owner_identity = MagicMock()

    with pytest.raises(RuntimeError, match="strict hybrid persistence"):
        cache.load_from_disk(str(tmp_path))
    cache.invalidate_owner_identity.assert_not_called()

    index["persistence_identity"] = dict(context.persistence_identity)
    (tmp_path / "index.json").write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="strict hybrid persistence"):
        cache.load_from_disk(str(tmp_path))
    cache.invalidate_owner_identity.assert_not_called()


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _patch_fixture_tokenizer_version(monkeypatch):
    original = owner_identity_module.importlib.metadata.version

    def version(distribution):
        if distribution == __name__.split(".", 1)[0]:
            return "fixture-version"
        return original(distribution)

    monkeypatch.setattr(owner_identity_module.importlib.metadata, "version", version)


def _write_loaded_identity_fixture(root: Path):
    files = {
        "artifact-source.json": json.dumps(
            {"model_id": "fixture/model", "revision": "fixture-revision"},
            sort_keys=True,
        ),
        "config.json": "{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"weight": "model-00001-of-00001.safetensors"}},
            sort_keys=True,
        ),
        "model-00001-of-00001.safetensors": "fixture-weights",
        "tokenizer_config.json": "{}",
        "chat_template.jinja": "{{ messages }}",
        "tokenizer.json": "{}",
    }
    for name, contents in files.items():
        (root / name).write_text(contents)

    class FixtureLanguageModel:
        config = SimpleNamespace(
            model_type="fixture-type",
            num_hidden_layers=1,
            hidden_size=8,
            vocab_size=16,
            num_key_value_heads=1,
            head_dim=8,
        )

        def make_cache(self):
            class FixtureCache:
                keys = None
                values = None

            return [FixtureCache()]

    class FixtureModel:
        def __init__(self):
            self.language_model = FixtureLanguageModel()
            self.config = SimpleNamespace(model_type="fixture-type")

    class FixtureTokenizer:
        name_or_path = str(root)
        vocab_size = 1
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        chat_template = "{{ messages }}"

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [len(text)]

        def get_vocab(self):
            return {"token": 0}

    model = FixtureModel()
    processor = SimpleNamespace(
        tokenizer=FixtureTokenizer(), chat_template="{{ messages }}"
    )
    cache_config = MemoryCacheConfig(max_memory_mb=1)
    persistence = dict(
        build_cache_owner_persistence_identity(
            model.language_model,
            processor.tokenizer,
            str(root),
            cache_config,
            {"max_kv_size": 1},
            processor,
        )
    )
    mode = {
        "continuous_batching": True,
        "serialized": False,
        "mtp": False,
        "chunked_prefill": True,
        "rotating_cache": True,
        "ple": False,
        "qsa": False,
        "audio": False,
        "capability_modes": ["media", "text"],
    }
    cache_identity = {
        "model_id": "fixture/model",
        "model_revision": "fixture-revision",
        "artifact_id": root.name,
        "artifact_digest": _sha256(root / "artifact-source.json"),
        "weight_index_digest": _sha256(root / "model.safetensors.index.json"),
        "family": "fixture",
        "architecture": "FixtureModel",
        "model_module": type(model).__module__,
        "language_module": type(model.language_model).__module__,
        "model_type": "fixture-type",
        "config_digest": _sha256(root / "config.json"),
        "tokenizer": {
            "files": [
                {"path": "tokenizer.json", "sha256": _sha256(root / "tokenizer.json")}
            ],
            "config_digest": _sha256(root / "tokenizer_config.json"),
            "implementation": type(processor.tokenizer).__module__.split(".", 1)[0],
            "implementation_version": "fixture-version",
            "added_tokens": [],
            "special_tokens": [],
            "encode_probes": [{"text": "abc", "ids": [3]}],
        },
        "chat_template": {
            "name": "fixture",
            "version": "1",
            "sha256": _sha256(root / "chat_template.jinja"),
        },
        "parser": {"name": "fixture", "version": "1", "sha256": "7" * 64},
        "vision": {
            "enabled": False,
            "config_digest": "8" * 64,
            "processor_files": [],
            "media_token_ids": [],
            "media_mapping_digest": "9" * 64,
        },
        "cache_schema": {
            "name": "fixture-cache",
            "version": "1",
            "sha256": persistence["cache_layout"],
        },
        "mode": mode,
    }
    draft_identity = dict(cache_identity)
    draft_identity.update(
        model_id="fixture/draft",
        model_revision="draft-revision",
        artifact_id="draft-artifact",
        artifact_digest="b" * 64,
    )
    relation = build_draft_compatibility(
        cache_identity, draft_identity, relation="fixture-v1"
    )
    manifest = build_identity_manifest(
        cache_identity,
        {
            "template_digest": "c" * 64,
            "parser_digest": "d" * 64,
            "tool_schema_digest": "e" * 64,
            "renderer": "fixture",
            "version": "1",
        },
        role="target",
        draft_compatibility=relation,
    )
    digest = hashlib.sha256(canonical_identity_bytes(cache_identity)).hexdigest()
    expected_fields = (
        ("role", "target"),
        ("request_protocol_identity", manifest["request_protocol_identity"]),
        ("draft_compatibility", manifest["draft_compatibility"]),
    )
    target = CacheOwnerGovernanceTarget(
        manifest=manifest,
        expected_model_cache_identity_digest=digest,
        expected_loaded_owner_digest=build_loaded_cache_owner_digest(
            persistence_identity=persistence,
            artifact_source={
                "model_id": "fixture/model",
                "revision": "fixture-revision",
            },
            loaded_identity={
                "architecture": "FixtureModel",
                "model_module": type(model).__module__,
                "language_module": type(model.language_model).__module__,
                "model_type": "fixture-type",
            },
            runtime_mode=mode,
            tokenizer_implementation=type(processor.tokenizer).__module__.split(".", 1)[
                0
            ],
            tokenizer_implementation_version="fixture-version",
        ),
        expected_identity_fields=expected_fields,
        expected_persistence_identity=persistence,
        runtime_composition_digest=_test_runtime_composition_digest(),
        cache_namespace="fixture-cache",
        registry_source="test:fixture",
        registry_complete=True,
    )
    return model, processor, target, cache_config, mode


def test_loaded_identity_is_derived_from_live_types_and_artifact_bytes(
    tmp_path, monkeypatch
):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    resolved = verify_loaded_model_cache_owner_context(
        model,
        processor,
        str(tmp_path),
        target,
        cache_config=cache_config,
        cache_runtime_identity={"max_kv_size": 1},
        runtime_mode=mode,
    )
    assert resolved.model_cache_identity_digest == target.expected_loaded_owner_digest

    (tmp_path / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="config.json"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


@pytest.mark.parametrize(
    ("tokenizer_updates", "message"),
    [
        (
            {
                "added_tokens_decoder": {7: "<added>"},
                "all_special_ids": [],
            },
            "added tokens",
        ),
        (
            {
                "added_tokens_decoder": {8: "<special>"},
                "all_special_ids": [8],
            },
            "special tokens",
        ),
    ],
)
def test_loaded_identity_rejects_tokenizer_semantic_mismatch(
    tmp_path, monkeypatch, tokenizer_updates, message
):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    for name, value in tokenizer_updates.items():
        setattr(processor.tokenizer, name, value)

    with pytest.raises(ValueError, match=message):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_capability_mode_mismatch(tmp_path, monkeypatch):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    changed_mode = dict(mode)
    changed_mode["capability_modes"] = ["text"]

    with pytest.raises(ValueError, match="capability_modes"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=changed_mode,
        )


def test_loaded_identity_rejects_weight_mutation(tmp_path, monkeypatch):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    (tmp_path / "model-00001-of-00001.safetensors").write_text("mutated")

    with pytest.raises(ValueError, match="provenance"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_symlinked_model_root(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        model_root
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    symlink = tmp_path / "model-link"
    symlink.symlink_to(model_root, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(symlink),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_target_only_cache_schema_mutation(
    tmp_path, monkeypatch
):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    cache_identity = _plain(target.manifest["model_cache_identity"])
    cache_identity["cache_schema"]["sha256"] = "0" * 64
    changed_manifest = build_identity_manifest(
        cache_identity,
        _plain(target.manifest["request_protocol_identity"]),
        role="target",
        draft_compatibility=_plain(target.manifest["draft_compatibility"]),
    )
    changed_target = CacheOwnerGovernanceTarget(
        manifest=changed_manifest,
        expected_model_cache_identity_digest=hashlib.sha256(
            canonical_identity_bytes(cache_identity)
        ).hexdigest(),
        expected_loaded_owner_digest=target.expected_loaded_owner_digest,
        expected_identity_fields=(
            ("role", "target"),
            (
                "request_protocol_identity",
                changed_manifest["request_protocol_identity"],
            ),
            ("draft_compatibility", changed_manifest["draft_compatibility"]),
        ),
        expected_persistence_identity=target.expected_persistence_identity,
        runtime_composition_digest=target.runtime_composition_digest,
        cache_namespace=target.cache_namespace,
        registry_source="test:mutated-target",
        registry_complete=True,
    )

    with pytest.raises(ValueError, match="cache schema"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            changed_target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_governed_loaded_owner_digest_mismatch(
    tmp_path, monkeypatch
):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    changed_target = CacheOwnerGovernanceTarget(
        manifest=target.manifest,
        expected_model_cache_identity_digest=(
            target.expected_model_cache_identity_digest
        ),
        expected_loaded_owner_digest="0" * 64,
        expected_identity_fields=target.expected_identity_fields,
        expected_persistence_identity=target.expected_persistence_identity,
        runtime_composition_digest=target.runtime_composition_digest,
        cache_namespace=target.cache_namespace,
        registry_source="test:mismatched-loaded-owner",
        registry_complete=True,
    )

    with pytest.raises(ValueError, match="loaded cache owner digest"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            changed_target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_runtime_composition_mismatch(tmp_path, monkeypatch):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    changed_target = CacheOwnerGovernanceTarget(
        manifest=target.manifest,
        expected_model_cache_identity_digest=(
            target.expected_model_cache_identity_digest
        ),
        expected_loaded_owner_digest=target.expected_loaded_owner_digest,
        expected_identity_fields=target.expected_identity_fields,
        expected_persistence_identity=target.expected_persistence_identity,
        runtime_composition_digest="0" * 64,
        cache_namespace=target.cache_namespace,
        registry_source="test:mismatched-runtime",
        registry_complete=True,
    )

    with pytest.raises(ValueError, match="runtime composition"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            changed_target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_scheduler_threads_verified_owner_context(monkeypatch):
    context = MagicMock(spec=VerifiedCacheOwnerContext)
    config = MLLMSchedulerConfig(
        enable_prefix_cache=True,
        cache_owner_context=context,
    )
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.batch_generator = None
    scheduler.config = config
    scheduler.model = object()
    scheduler.processor = object()
    scheduler.mm_processor = object()
    scheduler.stop_tokens = set()
    scheduler._ssd_tier = None
    scheduler._owner_thread_id = threading.get_ident()
    captured = {}

    class FakeGenerator:
        prefix_cache = None

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("vllm_mlx.mllm_scheduler.MLLMBatchGenerator", FakeGenerator)
    monkeypatch.setattr("mlx_lm.sample_utils.make_sampler", lambda **_kwargs: object())

    scheduler._ensure_batch_generator()

    assert captured["cache_owner_context"] is context


@pytest.mark.parametrize("insert_result", [RuntimeError("mint failed"), []])
def test_scheduler_insert_failure_leaves_waiting_state_transactional(insert_result):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="request-1", prompt="hello")
    request.num_prompt_tokens = 3
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(max_num_seqs=1)
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.batch_generator = MagicMock()
    _configure_mock_insert_transaction(scheduler.batch_generator)
    scheduler._ensure_batch_generator = MagicMock()
    if isinstance(insert_result, Exception):
        scheduler.batch_generator.insert.side_effect = insert_result
    else:
        scheduler.batch_generator.insert.return_value = insert_result

    with pytest.raises(
        RuntimeError,
        match="mint failed|atomic UID commit",
    ):
        scheduler._schedule_waiting()

    assert list(scheduler.waiting) == [request]
    assert request.status is RequestStatus.WAITING
    assert request.batch_uid is None
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == {}
    assert scheduler.total_prompt_tokens == 0


@pytest.mark.parametrize("malformed", [None, 7, "7"])
def test_scheduler_malformed_insert_result_rolls_back_real_generator(malformed):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="request-1", prompt="hello")
    request.num_prompt_tokens = 3
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=1,
        cache_owner_context=object(),
    )
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    original_insert = generator.insert

    def insert_then_corrupt(requests):
        original_insert(requests)
        return malformed

    generator.insert = insert_then_corrupt
    scheduler.batch_generator = generator
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(RuntimeError, match="atomic UID commit"):
        scheduler._schedule_waiting()

    assert list(scheduler.waiting) == [request]
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    assert generator.uid_counter == 0
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)


def test_scheduler_base_exception_during_insert_rolls_back_real_generator():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    requests = [
        MLLMRequest(request_id="request-1", prompt="hello"),
        MLLMRequest(request_id="request-2", prompt="hello"),
    ]
    for request in requests:
        request.num_prompt_tokens = 3
    scheduler.waiting = deque(requests)
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=2,
        cache_owner_context=object(),
    )
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.side_effect = [
        binding,
        KeyboardInterrupt("interrupted mint"),
    ]
    scheduler.batch_generator = generator
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(KeyboardInterrupt, match="interrupted mint"):
        scheduler._schedule_waiting()

    assert list(scheduler.waiting) == requests
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    assert generator.uid_counter == 0
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)


def test_scheduler_insert_exception_preserves_preexisting_generator_request():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="request-1", prompt="hello")
    request.num_prompt_tokens = 3
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=1,
        cache_owner_context=object(),
    )
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    generator = _generator()
    existing = _request("request-1")
    existing.uid = 0
    binding = object()
    generator.unprocessed_requests = [existing]
    generator._cache_owner_requests = {existing.request_id: binding}
    generator.uid_counter = 1
    scheduler.batch_generator = generator
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(RuntimeError, match="duplicate owner-bound request ID"):
        scheduler._schedule_waiting()

    assert list(scheduler.waiting) == [request]
    assert generator.unprocessed_requests == [existing]
    assert generator._cache_owner_requests == {existing.request_id: binding}
    assert generator.uid_counter == 1
    generator.prefix_cache.release_owner_request.assert_not_called()


def test_scheduler_malformed_noop_insert_preserves_preexisting_generator_request():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="request-1", prompt="hello")
    request.num_prompt_tokens = 3
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=1,
        cache_owner_context=object(),
    )
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    generator = _generator()
    existing = _request("request-1")
    existing.uid = 0
    binding = object()
    generator.unprocessed_requests = [existing]
    generator._cache_owner_requests = {existing.request_id: binding}
    generator.uid_counter = 1
    generator.insert = MagicMock(return_value=None)
    scheduler.batch_generator = generator
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(RuntimeError, match="atomic UID commit"):
        scheduler._schedule_waiting()

    assert generator.unprocessed_requests == [existing]
    assert generator._cache_owner_requests == {existing.request_id: binding}
    assert generator.uid_counter == 1
    generator.prefix_cache.release_owner_request.assert_not_called()


def test_generator_insert_sort_failure_rolls_back_owner_state():
    class ExplodingMedia:
        def __bool__(self):
            raise RuntimeError("media inspection failed")

    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    request = _request("sort-failure")
    request.images = ExplodingMedia()

    with pytest.raises(RuntimeError, match="media inspection failed"):
        generator.insert([request])

    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    assert generator.uid_counter == 0
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)


def test_generator_insert_failure_cannot_restore_stale_uid_counter():
    class ExplodingMedia:
        def __bool__(self):
            raise RuntimeError("sort failure")

    class NormalFirstLock:
        def __init__(self):
            self.lock = threading.RLock()
            self.failing_waiting = threading.Event()
            self.normal_finished = threading.Event()

        def __enter__(self):
            if threading.current_thread().name == "failing-insert":
                self.failing_waiting.set()
                assert self.normal_finished.wait(timeout=2)
            self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

    generator = _generator(owner_required=False)
    ordered_lock = NormalFirstLock()
    generator._prefix_checkpoint_lock = ordered_lock
    failing_entered = threading.Event()
    failures = []
    normal_uids = []
    failing = _request("failing")
    failing.images = ExplodingMedia()

    def insert_failing():
        failing_entered.set()
        try:
            generator.insert([failing])
        except BaseException as exc:  # pragma: no cover - assertion below
            failures.append(exc)

    def insert_normal():
        normal_uids.extend(generator.insert([_request("normal")]))
        ordered_lock.normal_finished.set()

    failing_worker = threading.Thread(target=insert_failing, name="failing-insert")
    normal_worker = threading.Thread(target=insert_normal)
    failing_worker.start()
    assert failing_entered.wait(timeout=2)
    assert ordered_lock.failing_waiting.wait(timeout=2)
    normal_worker.start()
    failing_worker.join(timeout=2)
    normal_worker.join(timeout=2)

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert str(failures[0]) == "sort failure"
    assert normal_uids == [0]
    assert [
        (request.request_id, request.uid) for request in generator.unprocessed_requests
    ] == [("normal", 0)]
    assert generator.uid_counter == 1


def test_insertion_transaction_releases_lock_after_base_exception():
    generator = _generator(owner_required=False)
    interrupted = threading.Event()
    insert_finished = threading.Event()
    errors = []
    inserted_uids = []

    def interrupt_transaction():
        try:
            with generator.insertion_transaction():
                raise KeyboardInterrupt("transaction interrupted")
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)
        finally:
            interrupted.set()

    def insert_after_interrupt():
        inserted_uids.extend(generator.insert([_request("after-interrupt")]))
        insert_finished.set()

    interrupt_worker = threading.Thread(target=interrupt_transaction)
    interrupt_worker.start()
    assert interrupted.wait(timeout=2)
    insert_worker = threading.Thread(target=insert_after_interrupt)
    insert_worker.start()
    assert insert_finished.wait(timeout=2)
    interrupt_worker.join(timeout=2)
    insert_worker.join(timeout=2)

    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    assert inserted_uids == [0]
    assert generator.uid_counter == 1


def test_scheduler_insert_validation_is_atomic_against_direct_insert():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduled = MLLMRequest(request_id="shared-request", prompt="hello")
    scheduled.num_prompt_tokens = 3
    scheduler.waiting = deque([scheduled])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=1,
        cache_owner_context=None,
    )
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    generator = _generator()
    generator.prefix_cache.mint_owner_request.return_value = object()
    scheduler.batch_generator = generator
    scheduler._ensure_batch_generator = MagicMock()
    scheduler_insert_entered = threading.Event()
    release_scheduler_insert = threading.Event()
    direct_finished = threading.Event()
    scheduler_errors = []
    direct_errors = []

    def malformed_scheduler_insert(_requests):
        scheduler_insert_entered.set()
        assert release_scheduler_insert.wait(timeout=2)
        return None

    generator.insert = malformed_scheduler_insert

    def schedule():
        try:
            scheduler._schedule_waiting()
        except BaseException as exc:  # pragma: no cover - assertion below
            scheduler_errors.append(exc)

    def insert_directly():
        try:
            MLLMBatchGenerator.insert(generator, [_request("shared-request")])
        except BaseException as exc:  # pragma: no cover - assertion below
            direct_errors.append(exc)
        finally:
            direct_finished.set()

    scheduler_worker = threading.Thread(target=schedule)
    direct_worker = threading.Thread(target=insert_directly)
    scheduler_worker.start()
    assert scheduler_insert_entered.wait(timeout=2)
    direct_worker.start()
    assert direct_finished.wait(timeout=0.05) is False
    release_scheduler_insert.set()
    scheduler_worker.join(timeout=2)
    direct_worker.join(timeout=2)

    assert len(scheduler_errors) == 1
    assert isinstance(scheduler_errors[0], RuntimeError)
    assert "atomic UID commit" in str(scheduler_errors[0])
    assert direct_errors == []
    assert [request.request_id for request in generator.unprocessed_requests] == [
        "shared-request"
    ]
    assert set(generator._cache_owner_requests) == {"shared-request"}
    assert generator.uid_counter == 1
    assert list(scheduler.waiting) == [scheduled]
    assert scheduler.running == {}


def test_scheduler_commit_failure_rolls_back_generator_and_scheduler_state():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    first = MLLMRequest(request_id="commit-first", prompt="hello")
    first.num_prompt_tokens = 3
    second = MLLMRequest(request_id="commit-second", prompt="hello")
    second.num_prompt_tokens = "invalid"
    scheduler.waiting = deque([first, second])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=2,
        cache_owner_context=object(),
    )
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    generator = _generator()
    bindings = [object(), object()]
    generator.prefix_cache.mint_owner_request.side_effect = bindings
    scheduler.batch_generator = generator
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(TypeError):
        scheduler._schedule_waiting()

    assert list(scheduler.waiting) == [first, second]
    assert first.status is RequestStatus.WAITING
    assert first.batch_uid is None
    assert second.status is RequestStatus.WAITING
    assert second.batch_uid is None
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == {}
    assert scheduler.total_prompt_tokens == 0
    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    assert generator.uid_counter == 0
    generator.prefix_cache.release_owner_request.assert_has_calls(
        [call(bindings[0]), call(bindings[1])], any_order=True
    )
    assert generator.prefix_cache.release_owner_request.call_count == 2


@pytest.mark.parametrize("insert_result", [[True], [-1]])
def test_scheduler_rejects_malformed_uids_without_state_mutation(insert_result):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="request-1", prompt="hello")
    request.num_prompt_tokens = 3
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(max_num_seqs=1)
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.batch_generator = MagicMock()
    _configure_mock_insert_transaction(scheduler.batch_generator)
    scheduler.batch_generator.insert.return_value = insert_result
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(RuntimeError, match="atomic UID commit"):
        scheduler._schedule_waiting()

    assert list(scheduler.waiting) == [request]
    assert request.status is RequestStatus.WAITING
    assert request.batch_uid is None
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == {}
    assert scheduler.total_prompt_tokens == 0


@pytest.mark.parametrize(
    ("insert_result", "requests", "existing_uids"),
    [
        ([0, 0], ["request-1", "request-2"], {}),
        ([7], ["request-1"], {7: "existing-request"}),
    ],
)
def test_scheduler_rejects_duplicate_or_colliding_uids_transactionally(
    insert_result, requests, existing_uids
):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    waiting = []
    for request_id in requests:
        request = MLLMRequest(request_id=request_id, prompt="hello")
        request.num_prompt_tokens = 3
        waiting.append(request)
    scheduler.waiting = deque(waiting)
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(max_num_seqs=len(waiting))
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = dict(existing_uids)
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.batch_generator = MagicMock()
    _configure_mock_insert_transaction(scheduler.batch_generator)
    scheduler.batch_generator.insert.return_value = insert_result
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(RuntimeError, match="atomic UID commit"):
        scheduler._schedule_waiting()

    assert list(scheduler.waiting) == waiting
    assert all(request.status is RequestStatus.WAITING for request in waiting)
    assert all(request.batch_uid is None for request in waiting)
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == dict(existing_uids)
    assert scheduler.total_prompt_tokens == 0


def test_scheduler_rejects_insert_that_mutates_waiting_queue():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    first = MLLMRequest(request_id="request-1", prompt="hello")
    second = MLLMRequest(request_id="request-2", prompt="hello")
    for request in (first, second):
        request.num_prompt_tokens = 3
    scheduler.waiting = deque([first, second])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(max_num_seqs=2)
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.batch_generator = MagicMock()
    _configure_mock_insert_transaction(scheduler.batch_generator)
    scheduler._ensure_batch_generator = MagicMock()

    def mutate_queue(_requests):
        scheduler.waiting.rotate(1)
        return [10, 11]

    scheduler.batch_generator.insert.side_effect = mutate_queue

    with pytest.raises(RuntimeError, match="atomic UID commit"):
        scheduler._schedule_waiting()

    scheduler.batch_generator.rollback_inserted_requests.assert_called_once_with(
        ["request-1", "request-2"], minimum_uid=0
    )
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == {}
    assert scheduler.total_prompt_tokens == 0


def test_batched_engine_derives_and_threads_verified_owner_context(
    tmp_path, monkeypatch
):
    from vllm_mlx.engine.batched import BatchedEngine

    target = MagicMock(spec=CacheOwnerGovernanceTarget)
    context = MagicMock(spec=VerifiedCacheOwnerContext)
    scheduler_config = SchedulerConfig(
        enable_prefix_cache=True,
        enable_mtp=True,
        chunked_prefill_tokens=2048,
        max_kv_size=4096,
        cache_owner_target=target,
    )
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model = SimpleNamespace(config=SimpleNamespace())
    engine._processor = SimpleNamespace(
        tokenizer=SimpleNamespace(name_or_path=str(tmp_path))
    )
    engine._scheduler_config = scheduler_config
    engine._model_name = str(tmp_path)
    engine._mllm_draft_model = None
    engine._mllm_draft_kind = None
    engine._mllm_draft_block_size = None
    engine._mllm_instance = SimpleNamespace(resolved_model_path=str(tmp_path))
    engine._mllm_scheduler = None
    captured = {}

    class FakeScheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            return None

    monkeypatch.setattr("vllm_mlx.mllm_scheduler.MLLMScheduler", FakeScheduler)
    verify = MagicMock(return_value=context)
    monkeypatch.setattr(
        "vllm_mlx.cache_owner_identity.verify_loaded_model_cache_owner_context",
        verify,
    )
    monkeypatch.setattr(
        "vllm_mlx.mllm_batch_generator.normalize_prefix_cache_chat_template",
        lambda _processor: None,
    )

    asyncio.run(engine._start_mllm())

    config = captured["config"]
    assert config.cache_owner_context is context
    assert verify.call_args.kwargs["runtime_mode"] == {
        "continuous_batching": True,
        "serialized": False,
        "mtp": True,
        "chunked_prefill": True,
        "rotating_cache": True,
        "ple": False,
        "qsa": False,
        "audio": False,
        "capability_modes": ["media", "text"],
    }


def test_mllm_loader_records_and_uses_exact_resolved_model_root(tmp_path, monkeypatch):
    import mlx_vlm
    import mlx_vlm.utils

    from vllm_mlx.models.mllm import MLXMultimodalLM

    observed = []
    model = SimpleNamespace(config=SimpleNamespace())
    processor = object()
    monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda _model_name: tmp_path)
    monkeypatch.setattr(
        mlx_vlm,
        "load",
        lambda path: (observed.append(("load", path)) or (model, processor)),
    )
    monkeypatch.setattr(
        mlx_vlm.utils,
        "load_config",
        lambda path: observed.append(("config", path)) or {},
    )

    wrapper = MLXMultimodalLM("fixture/model", enable_cache=False)
    wrapper.load()

    resolved = str(tmp_path.resolve())
    assert wrapper.resolved_model_path == resolved
    assert observed == [("load", resolved), ("config", resolved)]


@pytest.mark.parametrize("operation", ["load", "clear"])
def test_batched_engine_lifecycle_invalidates_then_rebinds_owner(operation):
    from vllm_mlx.engine.batched import BatchedEngine

    events = MagicMock()
    prefix_cache = MagicMock()
    batch_generator = MagicMock(prefix_cache=prefix_cache)
    events.attach_mock(batch_generator.begin_cache_owner_lifecycle_mutation, "begin")
    events.attach_mock(batch_generator.finish_cache_owner_lifecycle_mutation, "finish")
    events.attach_mock(prefix_cache.load_from_disk, "load")
    events.attach_mock(prefix_cache.clear, "clear")
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.batch_generator = batch_generator
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._ensure_batch_generator = MagicMock()
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = scheduler
    engine._engine = None

    if operation == "load":
        prefix_cache.load_from_disk.return_value = 3
        assert engine.load_cache_from_disk("cache-dir") == 3
        assert events.mock_calls == [
            call.begin(),
            call.load("cache-dir"),
            call.finish(),
        ]
    else:
        engine.clear_prefix_cache()
        assert events.mock_calls == [
            call.begin(),
            call.clear(),
            call.finish(),
        ]


def test_batched_engine_rebinds_owner_when_cache_load_fails():
    from vllm_mlx.engine.batched import BatchedEngine

    events = MagicMock()
    prefix_cache = MagicMock()
    prefix_cache.load_from_disk.side_effect = RuntimeError("corrupt cache")
    batch_generator = MagicMock(prefix_cache=prefix_cache)
    events.attach_mock(batch_generator.begin_cache_owner_lifecycle_mutation, "begin")
    events.attach_mock(
        batch_generator.recover_cache_owner_lifecycle_mutation, "recover"
    )
    events.attach_mock(batch_generator.finish_cache_owner_lifecycle_mutation, "finish")
    events.attach_mock(prefix_cache.load_from_disk, "load")
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.batch_generator = batch_generator
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._ensure_batch_generator = MagicMock()
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = scheduler
    engine._engine = None

    with pytest.raises(RuntimeError, match="corrupt cache"):
        engine.load_cache_from_disk("cache-dir")

    assert events.mock_calls == [
        call.begin(),
        call.load("cache-dir"),
        call.recover(),
    ]
    batch_generator.finish_cache_owner_lifecycle_mutation.assert_not_called()


def test_owner_lifecycle_mutation_rejects_active_generator_work():
    generator = _generator()
    binding = object()
    generator._cache_owner_requests["request-1"] = binding

    with pytest.raises(RuntimeError, match="requires an idle generator"):
        generator.begin_cache_owner_lifecycle_mutation()

    assert generator._cache_owner_requests == {"request-1": binding}
    generator.prefix_cache.cancel_owner_request.assert_not_called()
    generator.prefix_cache.release_owner_request.assert_not_called()


def test_owner_lifecycle_rejects_a_concurrent_second_begin():
    generator = _generator()
    first_entered = threading.Event()
    release_first = threading.Event()
    errors = []

    def hold_first_mutation():
        try:
            generator.begin_cache_owner_lifecycle_mutation()
            first_entered.set()
            assert release_first.wait(timeout=2)
            generator.finish_cache_owner_lifecycle_mutation()
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    worker = threading.Thread(target=hold_first_mutation)
    worker.start()
    assert first_entered.wait(timeout=2)
    with pytest.raises(RuntimeError, match="mutation is active"):
        generator.begin_cache_owner_lifecycle_mutation()
    release_first.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert generator._cache_owner_lifecycle_mutating is False


def test_owner_lifecycle_concurrent_admission_has_one_winner():
    generator = _generator()
    first_phase = threading.Barrier(2)

    class FirstPhaseBarrierLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._counts = {}

        def __enter__(self):
            self._lock.acquire()
            thread_id = threading.get_ident()
            self._counts[thread_id] = self._counts.get(thread_id, 0) + 1
            return self

        def __exit__(self, *_args):
            thread_id = threading.get_ident()
            count = self._counts[thread_id]
            self._lock.release()
            if count == 1:
                first_phase.wait(timeout=2)

    generator._prefix_checkpoint_lock = FirstPhaseBarrierLock()
    successes = []
    errors = []

    def begin():
        try:
            generator.begin_cache_owner_lifecycle_mutation()
            successes.append(threading.get_ident())
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    workers = [threading.Thread(target=begin) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers)
    assert len(successes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "mutation is active" in str(errors[0])
    generator._prefix_checkpoint_lock = threading.RLock()
    generator.finish_cache_owner_lifecycle_mutation()


def test_owner_insert_cannot_cross_lifecycle_admission():
    generator = _generator()
    passed_initial_check = threading.Event()
    allow_insert_to_continue = threading.Event()
    insert_thread_id = []

    class PausingRLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._counts = {}

        def __enter__(self):
            self._lock.acquire()
            thread_id = threading.get_ident()
            self._counts[thread_id] = self._counts.get(thread_id, 0) + 1
            return self

        def __exit__(self, *_args):
            thread_id = threading.get_ident()
            count = self._counts[thread_id]
            self._lock.release()
            if insert_thread_id and thread_id == insert_thread_id[0] and count == 1:
                passed_initial_check.set()
                assert allow_insert_to_continue.wait(timeout=2)

    generator._prefix_checkpoint_lock = PausingRLock()
    errors = []

    def insert():
        insert_thread_id.append(threading.get_ident())
        try:
            generator.insert([_request("racing-insert")])
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    worker = threading.Thread(target=insert)
    worker.start()
    assert passed_initial_check.wait(timeout=2)
    generator.begin_cache_owner_lifecycle_mutation()
    allow_insert_to_continue.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "mutation is active" in str(errors[0])
    assert generator.unprocessed_requests == []
    generator._prefix_checkpoint_lock = threading.RLock()
    generator.finish_cache_owner_lifecycle_mutation()


def test_owner_lifecycle_cannot_enter_during_mint_and_append_transaction():
    generator = _generator()
    mint_entered = threading.Event()
    release_mint = threading.Event()
    lifecycle_finished = threading.Event()
    insert_errors = []
    lifecycle_errors = []

    def mint(_uid):
        mint_entered.set()
        assert release_mint.wait(timeout=2)
        return object()

    generator.prefix_cache.mint_owner_request.side_effect = mint

    def insert():
        try:
            generator.insert([_request("mint-race")])
        except BaseException as exc:  # pragma: no cover - assertion below
            insert_errors.append(exc)

    def begin_lifecycle():
        try:
            generator.begin_cache_owner_lifecycle_mutation()
        except BaseException as exc:  # pragma: no cover - assertion below
            lifecycle_errors.append(exc)
        finally:
            lifecycle_finished.set()

    insert_worker = threading.Thread(target=insert)
    lifecycle_worker = threading.Thread(target=begin_lifecycle)
    insert_worker.start()
    assert mint_entered.wait(timeout=2)
    lifecycle_worker.start()
    assert lifecycle_finished.wait(timeout=0.05) is False
    release_mint.set()
    insert_worker.join(timeout=2)
    lifecycle_worker.join(timeout=2)

    assert insert_errors == []
    assert len(lifecycle_errors) == 1
    assert isinstance(lifecycle_errors[0], RuntimeError)
    assert "idle generator" in str(lifecycle_errors[0])
    assert [request.request_id for request in generator.unprocessed_requests] == [
        "mint-race"
    ]


def test_scheduler_lifecycle_mutations_are_serialized_across_threads():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=None)
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._ensure_batch_generator = MagicMock()
    batch_generator = MagicMock()
    scheduler.batch_generator = batch_generator
    first_entered = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    errors = []

    def first_operation():
        first_entered.set()
        assert release_first.wait(timeout=2)
        return "first"

    def run_first():
        try:
            scheduler.run_cache_owner_lifecycle_mutation(first_operation)
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    def run_second():
        try:
            scheduler.run_cache_owner_lifecycle_mutation(lambda: "second")
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)
        finally:
            second_finished.set()

    first_thread = threading.Thread(target=run_first)
    second_thread = threading.Thread(target=run_second)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()
    assert second_finished.wait(timeout=0.05) is False
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert errors == []
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert batch_generator.begin_cache_owner_lifecycle_mutation.call_count == 2
    assert batch_generator.finish_cache_owner_lifecycle_mutation.call_count == 2


@pytest.mark.parametrize("phase", ["finish", "recover"])
def test_owner_lifecycle_base_exception_clears_gate_and_fails_terminally(phase):
    generator = _generator()
    generator._cache_owner_context = object()
    generator.begin_cache_owner_lifecycle_mutation()
    if phase == "finish":
        generator.prefix_cache.bind_owner_context.side_effect = KeyboardInterrupt(
            "rebind interrupted"
        )
        operation = generator.finish_cache_owner_lifecycle_mutation
    else:
        generator.prefix_cache.clear.side_effect = KeyboardInterrupt(
            "recovery interrupted"
        )
        operation = generator.recover_cache_owner_lifecycle_mutation

    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        operation()

    assert generator._cache_owner_lifecycle_mutating is False
    assert generator._cache_owner_lifecycle_failed is True
    with pytest.raises(RuntimeError, match="terminally failed"):
        generator.begin_cache_owner_lifecycle_mutation()
    with pytest.raises(RuntimeError, match="terminally failed"):
        generator.insert([_request("terminal")])


def test_owner_lifecycle_missing_cache_clears_gate_and_fails_terminally():
    generator = _generator()
    generator.prefix_cache = None
    generator._cache_owner_lifecycle_mutating = True

    with pytest.raises(RuntimeError, match="prefix cache is unavailable"):
        generator.recover_cache_owner_lifecycle_mutation()

    assert generator._cache_owner_lifecycle_mutating is False
    assert generator._cache_owner_lifecycle_failed is True
    with pytest.raises(RuntimeError, match="terminally failed"):
        generator.begin_cache_owner_lifecycle_mutation()


def test_scheduler_lifecycle_wrapper_recovers_after_base_exception():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=None)
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._ensure_batch_generator = MagicMock()
    batch_generator = MagicMock()
    batch_generator.recover_cache_owner_lifecycle_mutation.side_effect = (
        KeyboardInterrupt("recovery interrupted")
    )
    scheduler.batch_generator = batch_generator

    with pytest.raises(KeyboardInterrupt, match="recovery interrupted"):
        scheduler.run_cache_owner_lifecycle_mutation(
            lambda: (_ for _ in ()).throw(KeyboardInterrupt("operation interrupted"))
        )

    batch_generator.begin_cache_owner_lifecycle_mutation.assert_called_once_with()
    batch_generator.recover_cache_owner_lifecycle_mutation.assert_called_once_with()
    batch_generator.finish_cache_owner_lifecycle_mutation.assert_not_called()


def test_owner_replay_dispatch_never_uses_unchecked_clone():
    generator = _generator()
    binding = object()
    generator._cache_owner_requests["request-1"] = binding
    generator.prefix_cache.clone_owner_bound_for_replay.return_value = (
        OwnerBindingDecision(True, "none"),
        ["clone"],
    )
    generator.prefix_cache._clone_for_replay_unchecked = MagicMock(
        side_effect=AssertionError("unchecked clone reached")
    )

    assert generator._clone_prefix_for_replay("request-1", ["state"]) == ["clone"]
    generator.prefix_cache.clone_owner_bound_for_replay.assert_called_once_with(
        binding, ["state"]
    )
    generator.prefix_cache._clone_for_replay_unchecked.assert_not_called()


def test_scheduler_cache_clear_stops_before_mutation_when_owner_is_active():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.vision_cache = None
    prefix_cache = MagicMock()
    batch_generator = MagicMock(prefix_cache=prefix_cache)
    batch_generator.begin_cache_owner_lifecycle_mutation.side_effect = RuntimeError(
        "owner active"
    )
    scheduler.batch_generator = batch_generator
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._ensure_batch_generator = MagicMock()

    with pytest.raises(RuntimeError, match="owner active"):
        scheduler.clear_runtime_caches()

    prefix_cache.clear.assert_not_called()
    batch_generator.finish_cache_owner_lifecycle_mutation.assert_not_called()


def test_scheduler_cache_lifecycle_rejects_off_owner_thread():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=object())
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident() + 1
    scheduler._ensure_batch_generator = MagicMock()
    scheduler.batch_generator = MagicMock()
    operation = MagicMock()

    with pytest.raises(RuntimeError, match="owner thread"):
        scheduler.run_cache_owner_lifecycle_mutation(operation)

    operation.assert_not_called()
    scheduler._ensure_batch_generator.assert_not_called()


@pytest.mark.parametrize("operation", ["start", "stop", "reset"])
def test_owner_enabled_scheduler_lifecycle_rejects_cross_thread(operation):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=object())
    scheduler._owner_thread_id = threading.get_ident()
    errors = []

    def invoke():
        try:
            if operation == "start":
                asyncio.run(scheduler.start())
            elif operation == "stop":
                asyncio.run(scheduler.stop())
            else:
                scheduler.reset()
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "owner thread" in str(errors[0])


def test_owner_disabled_scheduler_lifecycle_allows_cross_thread():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=None)
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._state_lock = threading.RLock()
    scheduler._running = False
    scheduler._stopping = False
    scheduler._processing_task = None
    scheduler.batch_generator = None
    scheduler._ssd_tier = None

    async def process_loop():
        await asyncio.sleep(0)

    scheduler._process_loop = process_loop
    errors = []

    def run_lifecycle():
        async def run():
            await scheduler.start()
            await scheduler._processing_task
            await scheduler.stop()

        try:
            asyncio.run(run())
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    worker = threading.Thread(target=run_lifecycle)
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert scheduler._running is False
    assert scheduler._processing_task is None


@pytest.mark.anyio
async def test_scheduler_start_rejects_while_stop_awaits_task_cancellation():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=None)
    scheduler._owner_thread_id = threading.get_ident()
    scheduler._state_lock = threading.RLock()
    scheduler._running = True
    scheduler._stopping = False
    scheduler.batch_generator = MagicMock()
    scheduler._ssd_tier = None
    cancellation_entered = asyncio.Event()
    release_cancellation = asyncio.Event()

    async def processing_loop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_entered.set()
            await release_cancellation.wait()
            raise

    original_task = asyncio.create_task(processing_loop())
    scheduler._processing_task = original_task
    stop_task = asyncio.create_task(scheduler.stop())
    await asyncio.wait_for(cancellation_entered.wait(), timeout=2)

    with pytest.raises(RuntimeError, match="stop is in progress"):
        await scheduler.start()

    release_cancellation.set()
    await asyncio.wait_for(stop_task, timeout=2)

    assert scheduler._running is False
    assert scheduler._stopping is False
    assert scheduler._processing_task is None
    assert scheduler.batch_generator is None


@pytest.mark.parametrize("operation", ["load", "restore"])
def test_batched_engine_cache_restore_rejects_off_owner_before_construction(operation):
    from vllm_mlx.engine.batched import BatchedEngine

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=object())
    scheduler._owner_thread_id = threading.get_ident() + 1
    scheduler._ensure_batch_generator = MagicMock()
    scheduler.batch_generator = None
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = scheduler
    engine._engine = None

    with pytest.raises(RuntimeError, match="owner thread"):
        if operation == "load":
            engine.load_cache_from_disk("cache-dir")
        else:
            asyncio.run(engine.restore_cache_from_disk("cache-dir"))

    scheduler._ensure_batch_generator.assert_not_called()


def test_scheduler_disabled_owner_allows_cross_thread_admission():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="cross-thread", prompt="hello")
    request.num_prompt_tokens = 3
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(
        max_num_seqs=1,
        cache_owner_context=None,
    )
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler._state_lock = threading.RLock()
    scheduler._owner_thread_id = threading.get_ident()
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator._cache_owner_required = False
    _configure_mock_insert_transaction(scheduler.batch_generator)
    scheduler.batch_generator.insert.return_value = [7]
    scheduler._ensure_batch_generator = MagicMock()
    result = []
    errors = []

    def admit_from_worker():
        try:
            result.append(scheduler._schedule_waiting())
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    worker = threading.Thread(target=admit_from_worker)
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert result == [[request]]
    assert list(scheduler.waiting) == []
    assert scheduler.running == {request.request_id: request}
    assert scheduler.request_id_to_uid == {request.request_id: 7}
    assert scheduler.uid_to_request_id == {7: request.request_id}
    assert request.status is RequestStatus.RUNNING
    assert request.batch_uid == 7


@pytest.mark.parametrize(
    "invoke",
    [
        lambda scheduler: scheduler.add_request("hello"),
        lambda scheduler: scheduler.abort_request("request-1"),
        lambda scheduler: scheduler.step(),
        lambda scheduler: scheduler.clear_runtime_caches(),
    ],
)
def test_owner_enabled_scheduler_public_mutations_reject_cross_thread(invoke):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig(cache_owner_context=object())
    scheduler._owner_thread_id = threading.get_ident()
    errors = []

    def mutate_from_worker():
        try:
            invoke(scheduler)
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    worker = threading.Thread(target=mutate_from_worker)
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "owner thread" in str(errors[0])
