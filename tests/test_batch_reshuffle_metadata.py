# SPDX-License-Identifier: Apache-2.0
"""Focused regressions for eager MLLM batch reshuffle metadata.

Adapted from upstream PR #708 at exact head
369469e4520abc6b8bce3e63d540f8ba42ef106c.  The transaction-failure cases
and published-cache assertions are local adaptations for the request-ownership
atomicity introduced by eb7b0a4f13c75619ada0f15d6d10dd97dae286b0.
"""

import os
import tempfile

import pytest


def _make_batch(cache, *, uid_start=0):
    import mlx.core as mx

    from vllm_mlx.mllm_batch_generator import MLLMBatch

    size = next(
        (
            metadata.size
            for layer in cache
            for metadata in (getattr(layer, "offset", None),)
            if isinstance(metadata, mx.array)
        ),
        1,
    )
    return MLLMBatch(
        uids=list(range(uid_start, uid_start + size)),
        request_ids=[f"request-{uid}" for uid in range(uid_start, uid_start + size)],
        y=mx.zeros((size,), mx.int32),
        logprobs=[mx.zeros((4,)) for _ in range(size)],
        max_tokens=[16] * size,
        num_tokens=[0] * size,
        cache=cache,
        requests=[None] * size,
    )


def _live_kv_cache(left_padding, *, rotating=False):
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    if rotating:
        cache = BatchRotatingKVCache(max_size=8, left_padding=left_padding)
    else:
        cache = BatchKVCache(left_padding)
    values = mx.zeros((len(left_padding), 2, 4, 4), mx.float16)
    cache.update_and_fetch(values, values)
    mx.eval(cache.keys, cache.values, cache.offset, cache.left_padding)
    return cache


def _install_eval_spy(monkeypatch):
    import mlx.core as mx

    captured = []
    real_eval = mx.eval

    def spy(*arrays):
        captured.extend(arrays)
        return real_eval(*arrays)

    monkeypatch.setattr(mx, "eval", spy)
    return captured


def _metadata_arrays(cache):
    import mlx.core as mx

    arrays = []
    stack = list(cache)
    while stack:
        layer = stack.pop()
        if layer is None:
            continue
        children = getattr(layer, "caches", None)
        if children is not None:
            stack.extend(children)
        for name in ("offset", "left_padding", "lengths"):
            value = getattr(layer, name, None)
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def _assert_published_metadata_evaluated(batch, captured):
    metadata = _metadata_arrays(batch.cache)
    assert metadata
    for value in metadata:
        assert any(value is evaluated for evaluated in captured)


def _pending_edges(*arrays):
    import mlx.core as mx

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "graph.dot")
        mx.export_to_dot(path, *arrays)
        with open(path, encoding="utf-8") as graph:
            return graph.read().count("->")


def _all_cache_arrays(cache):
    import mlx.core as mx

    arrays = []
    stack = list(cache)
    while stack:
        layer = stack.pop()
        if layer is None:
            continue
        children = getattr(layer, "caches", None)
        if children is not None:
            stack.extend(children)
        for value in vars(layer).values():
            if isinstance(value, mx.array):
                arrays.append(value)
            elif isinstance(value, (list, tuple)):
                arrays.extend(item for item in value if isinstance(item, mx.array))
    return arrays


class TestReshuffleMetadataPublicPaths:
    def test_filter_evaluates_published_cache_metadata(self, monkeypatch):
        batch = _make_batch([_live_kv_cache([0, 1, 0])])
        original_cache = batch.cache[0]
        captured = _install_eval_spy(monkeypatch)

        batch.filter([1, 2])

        assert batch.cache[0] is not original_cache
        _assert_published_metadata_evaluated(batch, captured)
        assert batch.cache[0].offset.tolist() == [3, 4]
        assert batch.cache[0].left_padding.tolist() == [1, 0]

    def test_extend_evaluates_resulting_cache_metadata(self, monkeypatch):
        batch = _make_batch([_live_kv_cache([0, 1])])
        incoming = _make_batch([_live_kv_cache([2])], uid_start=2)
        captured = _install_eval_spy(monkeypatch)

        batch.extend(incoming)

        _assert_published_metadata_evaluated(batch, captured)
        assert batch.cache[0].offset.tolist() == [4, 3, 2]
        assert batch.cache[0].left_padding.tolist() == [0, 1, 2]

    def test_hybrid_cachelist_children_are_evaluated(self, monkeypatch):
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, CacheList

        recurrent = ArraysCache(1, left_padding=[0, 2])
        recurrent.lengths = mx.array([4, 2])
        recurrent.cache[0] = mx.zeros((2, 4))
        batch = _make_batch(
            [CacheList(_live_kv_cache([0, 1]), recurrent), _live_kv_cache([0, 1])]
        )
        mx.eval(recurrent.left_padding, recurrent.lengths, recurrent.cache[0])
        captured = _install_eval_spy(monkeypatch)

        batch.filter([1])

        _assert_published_metadata_evaluated(batch, captured)
        published_recurrent = batch.cache[0].caches[1]
        assert published_recurrent.left_padding.tolist() == [2]
        assert published_recurrent.lengths.tolist() == [2]

        other_recurrent = ArraysCache(1, left_padding=[3])
        other_recurrent.lengths = mx.array([1])
        other_recurrent.cache[0] = mx.zeros((1, 4))
        incoming = _make_batch(
            [
                CacheList(_live_kv_cache([2]), other_recurrent),
                _live_kv_cache([2]),
            ],
            uid_start=2,
        )
        mx.eval(
            other_recurrent.left_padding,
            other_recurrent.lengths,
            other_recurrent.cache[0],
        )
        captured.clear()

        batch.extend(incoming)

        _assert_published_metadata_evaluated(batch, captured)
        published_recurrent = batch.cache[0].caches[1]
        assert published_recurrent.left_padding.tolist() == [2, 3]
        assert published_recurrent.lengths.tolist() == [2, 1]

    def test_rotating_cache_filter_and_extend_are_evaluated(self, monkeypatch):
        batch = _make_batch([_live_kv_cache([0, 1], rotating=True)])
        incoming = _make_batch([_live_kv_cache([2], rotating=True)], uid_start=2)
        captured = _install_eval_spy(monkeypatch)

        batch.extend(incoming)

        _assert_published_metadata_evaluated(batch, captured)
        captured.clear()

        batch.filter([1, 2])

        _assert_published_metadata_evaluated(batch, captured)
        assert batch.cache[0].offset.tolist() == [3, 2]
        assert batch.cache[0].left_padding.tolist() == [1, 2]

    def test_generator_next_join_leave_churn_has_no_pending_cache_graph(self):
        import mlx.core as mx

        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        assert _pending_edges(mx.zeros((3,)) + 1) > 0
        if MLLMBatchGenerator._stream is None:
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.active_batch = None
        generator.unprocessed_requests = []
        generator._pending_error_responses = []
        generator._prefill_progress = {}
        generator._aborted_request_ids = set()
        generator._aborted_request_uids = set()
        generator._request_prefix_checkpoints = {}
        generator._cache_owner_requests = {}
        generator._cache_owner_required = False
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator.prefix_cache = None
        generator.completion_batch_size = 16
        generator._require_uniform_mllm_draft = False
        generator._allow_mid_batch_extend = True

        def process_prompts(requests):
            caches = [_live_kv_cache([0] * len(requests)) for _ in range(2)]
            batch = _make_batch(caches)
            batch.uids = [request.uid for request in requests]
            batch.request_ids = [request.request_id for request in requests]
            batch.max_tokens = [request.max_tokens for request in requests]
            batch.requests = list(requests)
            batch.logprobs = [mx.zeros((4,)) for _ in requests]
            return batch

        def step(input_tokens, cache, *_args, **_kwargs):
            size = input_tokens.shape[0]
            values = mx.zeros((size, 2, 1, 4), mx.float16)
            for layer in cache:
                layer.update_and_fetch(values, values)
            return mx.ones((size,), mx.int32), [mx.zeros((4,)) for _ in range(size)]

        generator._process_prompts = process_prompts
        generator._step = step

        def request(uid, max_tokens):
            return MLLMBatchRequest(
                uid=uid,
                request_id=f"request-{uid}",
                prompt="x",
                max_tokens=max_tokens,
                is_text_only=True,
            )

        generator.unprocessed_requests.extend(
            request(uid, max_tokens=999) for uid in range(3)
        )
        generator.next()

        measured_retirements = 0
        for cycle in range(6):
            uid = 100 + cycle
            generator.unprocessed_requests.append(request(uid, max_tokens=2))
            responses = generator.next()
            retired = [response.uid for response in responses if response.finish_reason]
            if not retired:
                continue

            measured_retirements += 1
            batch = generator.active_batch
            mx.eval(
                batch.y,
                *(layer.keys for layer in batch.cache),
                *(layer.values for layer in batch.cache),
            )
            cache_arrays = _all_cache_arrays(batch.cache)
            assert cache_arrays
            assert _pending_edges(*cache_arrays) == 0, f"churn cycle {cycle}"

        assert measured_retirements == 5
        assert generator.active_batch.uids == [0, 1, 2, 105]
        assert not generator.unprocessed_requests


class TestReshuffleMetadataTransactions:
    def test_filter_evaluator_failure_does_not_publish(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import MLLMBatch

        batch = _make_batch([_live_kv_cache([0, 1])])
        hook_calls = []
        batch._row_filter_hook = lambda before, after: hook_calls.append(
            (before, after)
        )
        original_cache_list = batch.cache
        original_cache = batch.cache[0]
        original_fields = {
            "uids": batch.uids,
            "request_ids": batch.request_ids,
            "y": batch.y,
            "logprobs": batch.logprobs,
            "max_tokens": batch.max_tokens,
            "num_tokens": batch.num_tokens,
            "requests": batch.requests,
        }
        original_metadata = (original_cache.offset, original_cache.left_padding)
        evaluated = []

        def fail_evaluator(cache):
            evaluated.append(cache)
            assert cache is not original_cache_list
            assert cache[0] is not original_cache
            raise RuntimeError("metadata evaluation failed")

        monkeypatch.setattr(
            MLLMBatch, "_sync_reshuffle_metadata", staticmethod(fail_evaluator)
        )

        with pytest.raises(RuntimeError, match="metadata evaluation failed"):
            batch.filter([1])

        assert batch.cache is original_cache_list
        assert batch.cache[0] is original_cache
        assert original_cache.offset is original_metadata[0]
        assert original_cache.left_padding is original_metadata[1]
        for name, value in original_fields.items():
            assert getattr(batch, name) is value
        assert hook_calls == []
        assert len(evaluated) == 1

    def test_extend_evaluator_failure_rolls_back_cache_and_metadata(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import MLLMBatch

        cache = _live_kv_cache([0, 1])
        batch = _make_batch([cache])
        incoming = _make_batch([_live_kv_cache([2])], uid_start=2)
        original_metadata = (cache.keys, cache.values, cache.offset, cache.left_padding)
        original_fields = {
            "uids": batch.uids,
            "request_ids": batch.request_ids,
            "y": batch.y,
            "logprobs": batch.logprobs,
            "max_tokens": batch.max_tokens,
            "num_tokens": batch.num_tokens,
            "requests": batch.requests,
        }
        evaluated = []

        def fail_evaluator(explicit_cache):
            evaluated.append(explicit_cache)
            assert explicit_cache is batch.cache
            assert cache.keys is not original_metadata[0]
            raise RuntimeError("metadata evaluation failed")

        monkeypatch.setattr(
            MLLMBatch, "_sync_reshuffle_metadata", staticmethod(fail_evaluator)
        )

        with pytest.raises(RuntimeError, match="metadata evaluation failed"):
            batch.extend(incoming)

        assert cache.keys is original_metadata[0]
        assert cache.values is original_metadata[1]
        assert cache.offset is original_metadata[2]
        assert cache.left_padding is original_metadata[3]
        for name, value in original_fields.items():
            assert getattr(batch, name) is value
        assert len(evaluated) == 1
