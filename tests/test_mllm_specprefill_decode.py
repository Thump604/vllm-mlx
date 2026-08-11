# SPDX-License-Identifier: Apache-2.0
"""Synthetic prepared-sparse-row and request-local decode contracts."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

from vllm_mlx.mllm_batch_generator import (
    MLLMBatch,
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchStats,
    MLLMTargetForwardPhase,
    SparseBatchCompatibilityError,
    SparseBatchError,
    _SupportedSparseCacheCheckpoint,
    _apply_first_token_processors,
    _decode_processor_contexts,
    install_mtp_mllm,
)
from vllm_mlx.specprefill_cache import (
    SparseCacheIdentity,
    SparseCacheState,
    SparsePolicyTuning,
)

_TOKENS = (10, 11, 12, 13, 14, 15)
_SELECTED = (0, 1, 4, 5)
_TUNING = SparsePolicyTuning(0.5, 0.0, 0, 1, 2)


class _PreparedCache(KVCache):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        self.keys = mx.zeros((1, 1, offset, 1))
        self.values = mx.zeros((1, 1, offset, 1))
        self.quantize_calls = 0
        self.ssd_calls = 0

    def quantize(self, *_args, **_kwargs):
        self.quantize_calls += 1
        raise AssertionError("sparse adoption must not quantize")

    def write_ssd(self, *_args, **_kwargs):
        self.ssd_calls += 1
        raise AssertionError("sparse adoption must not use SSD")


def _batch_cache(offsets):
    caches = [_PreparedCache(offset) for offset in offsets]
    return BatchKVCache.merge(caches)


def test_sparse_checkpoint_rejects_misaligned_arrays_cache_rows():
    recurrent = ArraysCache(2)
    recurrent.cache[0] = mx.zeros((2, 1))
    with pytest.raises(SparseBatchCompatibilityError, match="ArraysCache metadata"):
        _SupportedSparseCacheCheckpoint.capture(
            [_batch_cache((_state().rows[0].physical_valid_length,)), recurrent],
            _state().rows,
        )


class _ContextRecorder:
    def __init__(self, *, fail_enter=False):
        self.active = False
        self.calls = []
        self.fail_enter = fail_enter

    @contextmanager
    def __call__(self, forward):
        assert forward.phase is MLLMTargetForwardPhase.DECODE
        before = tuple(tuple(entry.offset.tolist()) for entry in forward.cache)
        if self.fail_enter:
            raise RuntimeError("context failed")
        self.active = True
        self.calls.append((forward, before))
        try:
            # Merely entering logical-position context cannot rewrite physical
            # occupancy. The model call below is the only offset advancer.
            assert (
                tuple(tuple(entry.offset.tolist()) for entry in forward.cache) == before
            )
            yield
        finally:
            self.active = False


class _LanguageModel:
    def __init__(self, context, *, fail_after_cache=False, physical_advance=1):
        self.context = context
        self.fail_after_cache = fail_after_cache
        self.physical_advance = physical_advance
        self.calls = 0
        self.mtp_calls = 0

    def __call__(self, input_tokens, *, cache, return_hidden=False):
        assert self.context.active
        self.calls += 1
        count = self.physical_advance
        for entry in cache:
            entry.offset = entry.offset + count
            entry._idx += count
            extension = mx.zeros((entry.keys.shape[0], 1, count, 1))
            entry.keys = mx.concatenate((entry.keys, extension), axis=2)
            entry.values = mx.concatenate((entry.values, extension), axis=2)
        if self.fail_after_cache:
            raise RuntimeError("target decode failed")
        batch, length = input_tokens.shape
        logits = mx.zeros((batch, length, 4))
        logits[:, :, 1] = 5.0
        if return_hidden:
            return logits, mx.ones((batch, length, 2))
        return logits

    def mtp_forward(self, *_args, **_kwargs):
        self.mtp_calls += 1
        raise AssertionError("sparse rows must bypass MTP")


class _PrefixCacheSpy:
    def __init__(self):
        self.fetch_calls = 0
        self.store_calls = 0

    def fetch(self, *_args, **_kwargs):
        self.fetch_calls += 1
        raise AssertionError("sparse rows must not use ordinary prefix lookup")

    def store(self, *_args, **_kwargs):
        self.store_calls += 1
        raise AssertionError("sparse rows must not use ordinary prefix storage")


def _identity(*, scorer_id="scorer@sha256:scorer"):
    return SparseCacheIdentity.from_tokens(
        target_id="target@sha256:target",
        tokenizer_id="tokenizer@sha256:tokenizer",
        scorer_id=scorer_id,
        selector_version="hybrid-chunk-v2",
        tuning=_TUNING,
        tokens=_TOKENS,
        selection_fingerprint="a" * 64,
    )


def _state(*, scorer_id="scorer@sha256:scorer"):
    return SparseCacheState.from_selection(
        _identity(scorer_id=scorer_id), (_SELECTED,), (len(_TOKENS),)
    )


def _request(uid=-1, request_id="request-1", *, max_tokens=2):
    return MLLMBatchRequest(
        uid=uid,
        request_id=request_id,
        prompt="prompt",
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        input_ids=mx.array([_TOKENS]),
        is_text_only=True,
    )


def _generator(*, fail_after_cache=False, fail_context=False, physical_advance=1):
    context = _ContextRecorder(fail_enter=fail_context)
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.language_model = _LanguageModel(
        context,
        fail_after_cache=fail_after_cache,
        physical_advance=physical_advance,
    )
    generator.model = generator.language_model
    generator.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
    generator._target_forward_context = context
    generator._expected_sparse_execution_config = _identity().execution_config
    generator._sparse_checkpoint_scalar_reads = 0
    generator._sparse_checkpoint_host_scalar_reads = 0
    generator.active_batch = None
    generator.unprocessed_requests = []
    generator.uid_counter = 0
    generator._aborted_request_ids = set()
    generator._pending_removal_uids = set()
    generator._pending_removal_lock = __import__("threading").Lock()
    generator._pending_error_responses = []
    generator._prefill_progress = {}
    generator._stats = MLLMBatchStats()
    generator.completion_batch_size = 8
    generator.stop_tokens = set()
    generator.prefix_cache = _PrefixCacheSpy()
    generator._think_suffix_len = 0
    return generator, context


def _adopt(generator, request=None, state=None, cache=None, logits=None):
    request = request or _request()
    state = state or _state()
    cache = cache or [_PreparedCache(len(_SELECTED))]
    logits = logits if logits is not None else mx.array([[[0.0, 1.0, 8.0, 0.0]]])
    uid = generator.adopt_prefilled_sparse_row(request, cache, logits, state)
    return uid, request, cache


def _dense_batch(uid=80):
    request = _request(uid=uid, request_id=f"dense-{uid}")
    return MLLMBatch(
        uids=[uid],
        request_ids=[request.request_id],
        y=mx.array([1]),
        logprobs=[mx.zeros(4)],
        max_tokens=[3],
        num_tokens=[0],
        cache=[_batch_cache([len(_TOKENS)])],
        requests=[request],
    )


def test_adoption_is_atomic_and_emits_only_after_wrapped_decode():
    generator, context = _generator()
    uid, request, _ = _adopt(generator)

    assert uid == request.uid == 0
    assert request.output_tokens == []
    assert generator._pending_error_responses == []
    assert generator.language_model.calls == 0
    assert generator.active_batch.sparse_row_states[0].next_logical_position == 6

    responses = generator._next()

    assert [response.token for response in responses] == [2]
    assert request.output_tokens == [2]
    assert len(context.calls) == 1
    assert context.calls[0][0].sparse_row_states[0].next_logical_position == 6
    assert not context.active
    assert generator.active_batch.sparse_row_states[0].next_logical_position == 7
    assert generator.active_batch.cache[0].offset.tolist() == [5]
    assert generator._sparse_checkpoint_scalar_reads == 5
    assert generator._sparse_checkpoint_host_scalar_reads == 1

    second_responses = generator._next()
    assert [response.token for response in second_responses] == [1]
    assert request.output_tokens == [2, 1]
    assert request.output_tokens.count(2) == 1


def test_prepared_first_token_processors_receive_exact_full_prompt_context():
    generator, _ = _generator()
    observed = []

    def processor(tokens, logits):
        observed.append(tuple(tokens.tolist()))
        forced = mx.full(logits.shape, -100.0)
        forced[:, 3] = 100.0
        return forced

    request = _request()
    request.logits_processors = [processor]

    _adopt(generator, request=request)

    assert observed == [_TOKENS]
    assert generator.active_batch.y.tolist() == [3]
    assert request.output_tokens == []


def test_dense_and_chunked_processor_helpers_use_full_prompt_and_current_y():
    observed = []

    def processor(tokens, logits):
        observed.append(tuple(tokens.tolist()))
        return logits

    logits = mx.zeros((1, 4))
    _apply_first_token_processors(logits, mx.array([_TOKENS]), [processor])
    request = _request()
    request.output_tokens[:] = [20, 21]

    assert observed == [_TOKENS]
    assert _decode_processor_contexts([request], [22]) == [[20, 21, 22]]


def test_mixed_sparse_dense_extend_fails_closed_before_cache_mutation():
    dense = _dense_batch()
    sparse_request = _request(uid=81, request_id="sparse-81")
    sparse_row = _state().rows[0]
    sparse = MLLMBatch(
        uids=[81],
        request_ids=[sparse_request.request_id],
        y=mx.array([2]),
        logprobs=[mx.zeros(4)],
        max_tokens=[3],
        num_tokens=[0],
        cache=[_batch_cache([len(_SELECTED)])],
        requests=[sparse_request],
        sparse_row_states=(sparse_row,),
    )

    with pytest.raises(SparseBatchCompatibilityError, match="cannot share"):
        dense.extend(sparse)

    assert dense.uids == [80]
    assert dense.sparse_row_states == (None,)
    assert dense.cache[0].offset.tolist() == [len(_TOKENS)]


@pytest.mark.parametrize(
    "row_states",
    [
        lambda row: (row, None),
        lambda row: (row, row.append_decode(1)),
    ],
)
def test_direct_terminal_batch_constructor_rejects_invalid_sparse_rows(row_states):
    row = _state().rows[0]
    requests = [
        _request(uid=91, request_id="row-1", max_tokens=1),
        _request(uid=92, request_id="row-2", max_tokens=1),
    ]

    with pytest.raises(SparseBatchCompatibilityError):
        MLLMBatch(
            uids=[91, 92],
            request_ids=[request.request_id for request in requests],
            y=mx.array([2, 2]),
            logprobs=[mx.zeros(4), mx.zeros(4)],
            max_tokens=[1, 1],
            num_tokens=[0, 0],
            cache=[_batch_cache([len(_SELECTED), len(_SELECTED)])],
            requests=requests,
            sparse_row_states=row_states(row),
        )


@pytest.mark.parametrize("y", [mx.array([[2]]), mx.array([2, 2])])
def test_batch_constructor_rejects_non_vector_or_misaligned_y(y):
    request = _request(uid=93, request_id="shape-row")

    with pytest.raises(ValueError, match="row-aligned token vector"):
        MLLMBatch(
            uids=[93],
            request_ids=[request.request_id],
            y=y,
            logprobs=[mx.zeros(4)],
            max_tokens=[1],
            num_tokens=[0],
            cache=[_batch_cache([len(_SELECTED)])],
            requests=[request],
        )


def test_sparse_filter_failure_restores_cache_and_all_row_metadata():
    generator, _ = _generator()
    _adopt(generator)
    batch = generator.active_batch
    original_filter = batch.cache[0].filter

    def failing_filter(keep_idx):
        original_filter(keep_idx)
        raise RuntimeError("filter failed")

    batch.cache[0].filter = failing_filter
    original_uids = list(batch.uids)
    original_offsets = batch.cache[0].offset.tolist()
    original_states = batch.sparse_row_states

    with pytest.raises(RuntimeError, match="filter failed"):
        batch.filter([0])

    assert batch.uids == original_uids
    assert batch.cache[0].offset.tolist() == original_offsets
    assert batch.sparse_row_states == original_states


def test_adoption_extend_failure_restores_active_batch_atomically():
    generator, _ = _generator()
    _adopt(generator)
    original_extend = generator.active_batch.cache[0].extend

    def failing_extend(other):
        original_extend(other)
        raise RuntimeError("extend failed")

    generator.active_batch.cache[0].extend = failing_extend
    original_uids = list(generator.active_batch.uids)
    original_offsets = generator.active_batch.cache[0].offset.tolist()
    request = _request(request_id="sparse-extension")

    with pytest.raises(RuntimeError, match="extend failed"):
        _adopt(generator, request=request)

    assert generator.active_batch.uids == original_uids
    assert generator.active_batch.cache[0].offset.tolist() == original_offsets
    assert len(generator.active_batch.sparse_row_states) == 1
    assert request.uid == -1


def test_extend_failure_does_not_consume_stochastic_sampling_and_retry_is_exact():
    generator, _ = _generator()
    _adopt(generator)
    request = _request(request_id="stochastic-retry")
    request.temperature = 1.0
    processor_calls = []
    request.logits_processors = [
        lambda tokens, logits: processor_calls.append(tuple(tokens.tolist())) or logits
    ]
    sampling_factory_calls = 0
    original_sampling_factory = generator._sampling_for_request

    def tracked_sampling_factory(candidate):
        nonlocal sampling_factory_calls
        sampling_factory_calls += 1
        return original_sampling_factory(candidate)

    generator._sampling_for_request = tracked_sampling_factory
    original_extend = generator.active_batch.cache[0].extend

    def failing_extend(_other):
        raise RuntimeError("extend failed before sampling")

    generator.active_batch.cache[0].extend = failing_extend
    with pytest.raises(RuntimeError, match="before sampling"):
        _adopt(generator, request=request)

    assert sampling_factory_calls == 0
    assert processor_calls == []
    assert generator.active_batch.uids == [0]
    assert request.uid == -1

    generator.active_batch.cache[0].extend = original_extend
    uid, _, _ = _adopt(generator, request=request)

    assert uid == 1
    assert sampling_factory_calls == 1
    assert processor_calls == [_TOKENS]
    assert generator.active_batch.uids == [0, 1]


def test_sampling_failure_rolls_back_preextended_cache_without_publication():
    generator, _ = _generator()
    _adopt(generator)
    request = _request(request_id="sampling-failure")
    request.logits_processors = [
        lambda _tokens, _logits: (_ for _ in ()).throw(RuntimeError("sample failed"))
    ]
    batch = generator.active_batch
    original_offsets = batch.cache[0].offset.tolist()
    original_uids = list(batch.uids)

    with pytest.raises(RuntimeError, match="sample failed"):
        _adopt(generator, request=request)

    assert batch.cache[0].offset.tolist() == original_offsets
    assert batch.uids == original_uids
    assert request.uid == -1


def test_incompatible_sparse_adoption_splits_before_mutating_active_batch():
    generator, _ = _generator()
    _adopt(generator)
    original_uids = list(generator.active_batch.uids)
    original_offsets = generator.active_batch.cache[0].offset.tolist()
    original_states = generator.active_batch.sparse_row_states
    incompatible_request = _request(request_id="request-2")

    with pytest.raises(SparseBatchCompatibilityError):
        _adopt(
            generator,
            request=incompatible_request,
            state=_state(scorer_id="other-scorer@sha256:other"),
        )

    assert generator.active_batch.uids == original_uids
    assert generator.active_batch.cache[0].offset.tolist() == original_offsets
    assert generator.active_batch.sparse_row_states == original_states
    assert incompatible_request.uid == -1


def test_decode_error_restores_physical_cache_and_logical_state_without_output():
    generator, context = _generator(fail_after_cache=True)
    _adopt(generator)
    batch = generator.active_batch
    original_y = batch.y.tolist()
    original_state = batch.sparse_row_states

    with pytest.raises(RuntimeError, match="target decode failed"):
        generator._next()

    assert batch.cache[0].offset.tolist() == [len(_SELECTED)]
    assert batch.sparse_row_states == original_state
    assert batch.y.tolist() == original_y
    assert batch.requests[0].output_tokens == []
    assert not context.active


def test_context_error_is_fail_closed_before_cache_or_logical_advance():
    generator, _ = _generator(fail_context=True)
    _adopt(generator)
    batch = generator.active_batch
    original_state = batch.sparse_row_states

    with pytest.raises(RuntimeError, match="context failed"):
        generator._next()

    assert batch.cache[0].offset.tolist() == [len(_SELECTED)]
    assert batch.sparse_row_states == original_state
    assert batch.requests[0].output_tokens == []


@pytest.mark.parametrize("physical_advance", [0, 2])
def test_wrong_physical_advance_rolls_back_before_logical_commit(physical_advance):
    generator, _ = _generator(physical_advance=physical_advance)
    _adopt(generator)
    batch = generator.active_batch
    original_state = batch.sparse_row_states
    original_keys = batch.cache[0].keys

    with pytest.raises(SparseBatchError, match="advance exactly one"):
        generator._next()

    assert batch.cache[0].offset.tolist() == [len(_SELECTED)]
    assert batch.cache[0].keys is original_keys
    assert batch.sparse_row_states == original_state
    assert batch.requests[0].output_tokens == []


def test_sparse_step_rejects_input_row_mismatch_before_target_context():
    generator, context = _generator()
    _adopt(generator)
    batch = generator.active_batch

    with pytest.raises(SparseBatchCompatibilityError, match="input row count"):
        generator._step(mx.array([[2], [2]]), batch.cache)

    assert generator.language_model.calls == 0
    assert context.calls == []
    assert batch.cache[0].offset.tolist() == [len(_SELECTED)]


def test_sparse_checkpoint_rejects_cache_metadata_row_mismatch():
    generator, context = _generator()
    request = _request(uid=94, request_id="cache-shape")
    row = _state().rows[0]
    generator.active_batch = MLLMBatch(
        uids=[94],
        request_ids=[request.request_id],
        y=mx.array([2]),
        logprobs=[mx.zeros(4)],
        max_tokens=[2],
        num_tokens=[0],
        cache=[_batch_cache([len(_SELECTED), len(_SELECTED)])],
        requests=[request],
        sparse_row_states=(row,),
    )

    with pytest.raises(SparseBatchCompatibilityError, match="metadata"):
        generator._step(mx.array([[2]]), generator.active_batch.cache)

    assert generator.language_model.calls == 0
    assert context.calls == []


def test_every_ordinary_dense_target_call_uses_same_bounded_context_seam():
    generator, context = _generator()
    generator.active_batch = _dense_batch()
    batch = generator.active_batch

    generator._step(batch.y[:, None], batch.cache)

    assert len(context.calls) == 1
    assert context.calls[0][0].sparse_row_states == (None,)
    assert batch.cache[0].offset.tolist() == [len(_TOKENS) + 1]
    assert batch.sparse_row_states == (None,)
    assert not context.active


def test_sparse_finish_bypasses_prefix_lcp_and_transformed_cache_paths():
    generator, _ = _generator()
    request = _request(max_tokens=1)
    _, _, prepared_cache = _adopt(generator, request=request)

    responses = generator._next()

    assert len(responses) == 1
    assert responses[0].finish_reason == "length"
    assert responses[0].prompt_cache is None
    assert generator.prefix_cache.fetch_calls == 0
    assert generator.prefix_cache.store_calls == 0
    assert prepared_cache[0].quantize_calls == 0
    assert prepared_cache[0].ssd_calls == 0
    assert generator.language_model.calls == 0


def test_sparse_eos_first_token_emits_without_target_lookahead():
    generator, _ = _generator()
    generator.stop_tokens = {2}
    _, request, _ = _adopt(generator)
    row_state = generator.active_batch.sparse_row_states[0]

    responses = generator._next()

    assert [(response.token, response.finish_reason) for response in responses] == [
        (2, "stop")
    ]
    assert request.output_tokens == [2]
    assert generator.language_model.calls == 0
    assert row_state.next_logical_position == len(_TOKENS)


def test_sparse_nonpositive_max_tokens_rejected_before_sampling_or_cache_merge():
    generator, _ = _generator()
    request = _request(max_tokens=0)
    observed = []
    request.logits_processors = [
        lambda tokens, logits: observed.append(tokens) or logits
    ]

    with pytest.raises(SparseBatchError, match="greater than zero"):
        _adopt(generator, request=request)

    assert observed == []
    assert generator.active_batch is None


def test_dense_text_arrival_remains_queued_while_sparse_lane_is_active():
    generator, _ = _generator()
    _adopt(generator)
    dense_request = _request(uid=70, request_id="queued-dense", max_tokens=3)
    generator.unprocessed_requests.append(dense_request)

    def forbidden_prefill(_requests):
        raise AssertionError("dense request must remain queued behind sparse lane")

    generator._process_prompts = forbidden_prefill
    responses = generator._next()

    assert [response.token for response in responses] == [2]
    assert generator.unprocessed_requests == [dense_request]
    assert dense_request.output_tokens == []
    assert generator.active_batch.has_sparse_rows


def test_exact_identity_mismatch_rejects_without_queue_or_prefix_side_effects():
    generator, _ = _generator()
    request = _request()
    request.input_ids = mx.array([[10, 11, 12, 13, 14, 99]])

    with pytest.raises(SparseBatchError, match="identity does not match"):
        _adopt(generator, request=request)

    assert generator.active_batch is None
    assert generator.unprocessed_requests == []
    assert generator.prefix_cache.fetch_calls == 0
    assert generator.prefix_cache.store_calls == 0
    assert request.uid == -1


def test_cancellation_removes_sparse_row_before_any_output():
    generator, _ = _generator()
    uid, request, _ = _adopt(generator)

    generator.remove([uid])

    assert generator.active_batch is None
    assert request.output_tokens == []
    assert generator.language_model.calls == 0


def test_native_mtp_bypasses_sparse_rows_before_any_draft_or_verify_call():
    generator, _ = _generator()
    _adopt(generator)
    install_mtp_mllm(generator, generator.language_model)
    batch = generator.active_batch

    batch.y, batch.logprobs = generator._step(
        batch.y[:, None], batch.cache, batch.logits_processors, None, batch.samplers
    )

    assert generator.language_model.mtp_calls == 0
    assert generator.language_model.calls == 1
    assert batch.sparse_row_states[0].next_logical_position == len(_TOKENS) + 1
    assert generator.get_mtp_stats()["bypass_counts"]["sparse_rows"] == 1
