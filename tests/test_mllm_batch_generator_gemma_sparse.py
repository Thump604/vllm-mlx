# SPDX-License-Identifier: Apache-2.0
"""Real-cache tests for the opt-in Gemma sparse CB transaction foundation."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import KVCache, RotatingKVCache

from vllm_mlx.mllm_batch_generator import (
    GemmaSparseBatchConfig,
    GemmaPreparedTargetAttestation,
    MLLMBatch,
    MLLMBatchGenerator,
    MLLMBatchRequest,
    SparseAdoptionError,
    SparseBatchCompatibilityError,
    SparseBatchError,
    install_mtp_mllm,
    prepare_gemma_sparse_target,
)
from vllm_mlx.specprefill_cache import (
    SparseCacheIdentity,
    SparseCacheRowState,
    SparseCacheState,
    SparsePolicyTuning,
)
from vllm_mlx.specprefill_gemma_cache import (
    FULL,
    GEMMA4_E2B,
    GemmaCacheError,
    GemmaCacheTopologyError,
    batch_cache_cursor,
)
from vllm_mlx.specprefill_runtime import TargetProcessorAttestation

_PROMPT = (10, 11, 12, 13)
_TUNING = SparsePolicyTuning(0.5, 0.0, 0, 1, 2)


class _LiveGemmaTopology:
    def __init__(self, *, layer_types=None, previous_kvs=None):
        self.config = SimpleNamespace(
            layer_types=list(layer_types or GEMMA4_E2B.layer_types)
        )
        self.model = SimpleNamespace(
            previous_kvs=list(previous_kvs or GEMMA4_E2B.previous_kvs)
        )


_TARGET_MODEL = _LiveGemmaTopology()
_PROCESSOR = object()
_TARGET_ARTIFACT_PATH = str(Path(__file__).resolve())
_TARGET_HASH = hashlib.sha256(Path(_TARGET_ARTIFACT_PATH).read_bytes()).hexdigest()
_TARGET_ID = f"{GEMMA4_E2B.artifact_id}@sha256:{_TARGET_HASH}"


def _identity(*, target_id=_TARGET_ID):
    return SparseCacheIdentity.from_tokens(
        target_id=target_id,
        tokenizer_id="tokenizer",
        scorer_id="scorer",
        selector_version="test-v1",
        tuning=_TUNING,
        tokens=_PROMPT,
        selection_fingerprint="d" * 64,
    )


def _row(*, target_id=_TARGET_ID, physical=2, logical=4):
    positions = tuple(range(max(0, physical - 1))) + (logical - 1,)
    return SparseCacheRowState(
        identity=_identity(target_id=target_id),
        logical_positions=positions,
        physical_valid_length=physical,
        next_logical_position=logical,
        prefill_physical_length=physical,
    )


def _prepared_attestation(
    *,
    target_model=_TARGET_MODEL,
    processor=_PROCESSOR,
    attested_target_model=None,
    attested_processor=None,
    target_hash=_TARGET_HASH,
    artifact=GEMMA4_E2B,
):
    trusted_identity = TargetProcessorAttestation(
        target_model=(
            target_model if attested_target_model is None else attested_target_model
        ),
        processor=processor if attested_processor is None else attested_processor,
        target_artifact_hash=target_hash,
        tokenizer_artifact_hash="b" * 64,
    )
    return prepare_gemma_sparse_target(
        target_model=target_model,
        processor=processor,
        target_artifact_path=_TARGET_ARTIFACT_PATH,
        artifact=artifact,
        target_identity_attestation=trusted_identity,
    )


def _config(*, target_id=_TARGET_ID, target_model=_TARGET_MODEL):
    return GemmaSparseBatchConfig(
        _prepared_attestation(target_model=target_model),
        _identity(target_id=target_id).execution_config,
    )


def _scalar_cache(*, physical=2):
    cache = []
    for layer_type in GEMMA4_E2B.layer_types[: GEMMA4_E2B.owner_count]:
        entry = (
            KVCache()
            if layer_type == FULL
            else RotatingKVCache(GEMMA4_E2B.sliding_window, keep=0)
        )
        values = mx.arange(physical, dtype=mx.float32).reshape(1, 1, physical, 1)
        entry.update_and_fetch(values, values + 100)
        cache.append(entry)
    mx.eval(*(tensor for entry in cache for tensor in (entry.keys, entry.values)))
    return cache


def _batch_cache(*, physical=2):
    return [entry.merge([entry]) for entry in _scalar_cache(physical=physical)]


def _request(uid, request_id):
    return MLLMBatchRequest(
        uid=uid,
        request_id=request_id,
        prompt="prompt",
        max_tokens=4,
        temperature=0.0,
        input_ids=mx.array(_PROMPT),
        is_text_only=True,
    )


def _batch(uid, request_id, *, row=None, cache=None, config=None):
    row = row or _row()
    return MLLMBatch(
        uids=[uid],
        request_ids=[request_id],
        y=mx.array([7]),
        logprobs=[mx.zeros(8)],
        max_tokens=[4],
        num_tokens=[0],
        cache=cache or _batch_cache(physical=row.physical_valid_length),
        requests=[_request(uid, request_id)],
        sparse_row_states=(row,),
        gemma_sparse_config=config or _config(),
    )


class _OneTokenModel(_LiveGemmaTopology):
    def __init__(self, *, failure=None):
        super().__init__()
        self.failure = failure
        self.calls = 0

    def __call__(self, tokens, *, cache):
        self.calls += 1
        rows = tokens.shape[0]
        values = mx.ones((rows, 1, 1, 1))
        for entry in cache:
            entry.update_and_fetch(values, values)
        if self.failure is not None:
            raise self.failure
        logits = mx.zeros((rows, 1, 8))
        logits[:, :, 3] = 9.0
        return logits


def _generator_for(batch, model):
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.active_batch = batch
    generator.model = model
    generator.language_model = model
    generator.processor = _PROCESSOR
    generator.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
    generator._target_forward_context = lambda _forward: nullcontext()
    generator._sparse_checkpoint_scalar_reads = 0
    generator._sparse_checkpoint_host_scalar_reads = 0
    return generator


def test_prepared_adoption_uses_real_gemma_batch_cache_and_emits_nothing():
    config = _config()
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator._target_forward_context = lambda _forward: nullcontext()
    generator._expected_sparse_execution_config = config.execution_config
    generator._expected_gemma_sparse_config = config
    generator.model = _TARGET_MODEL
    generator.language_model = _TARGET_MODEL
    generator.processor = _PROCESSOR
    generator._mtp_installed = False
    generator._aborted_request_ids = set()
    generator.active_batch = None
    generator.unprocessed_requests = []
    generator.uid_counter = 1
    generator.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
    request = _request(-1, "adopt")
    sparse_state = SparseCacheState((_row(),))

    uid = generator.adopt_prefilled_sparse_row(
        request,
        _scalar_cache(),
        mx.array([[[0.0, 0.0, 0.0, 8.0]]]),
        sparse_state,
    )

    assert uid == 1
    assert generator.active_batch is not None
    assert generator.active_batch.gemma_sparse_config is config
    assert generator.active_batch.y.tolist() == [3]
    assert request.output_tokens == []
    assert generator.unprocessed_requests == []


def test_gemma_decode_context_commits_exactly_one_logical_and_physical_token():
    contexts = []
    model = _OneTokenModel()
    batch = _batch(1, "decode", config=_config(target_model=model))
    generator = _generator_for(batch, model)
    generator._target_forward_context = (
        lambda forward: contexts.append(forward.sparse_row_states) or nullcontext()
    )

    sampled, _ = generator._step(batch.y, batch.cache)

    assert sampled.tolist() == [3]
    assert model.calls == 1
    assert contexts == [(_row(),)]
    assert batch.sparse_row_states[0].next_logical_position == 5
    assert batch.sparse_row_states[0].physical_valid_length == 3
    for entry in batch.cache:
        cursor = batch_cache_cursor(entry, logical_positions=(5,))
        assert cursor.total_writes == (3,)


def test_gemma_vlm_decode_binds_outer_text_and_processor_identities():
    text_model = _OneTokenModel()
    outer_model = SimpleNamespace(language_model=text_model)
    config = _config(target_model=outer_model)
    batch = _batch(1, "vlm-decode", config=config)
    generator = _generator_for(batch, text_model)
    generator.model = outer_model

    sampled, _ = generator._step(batch.y, batch.cache)

    assert sampled.tolist() == [3]
    assert text_model.calls == 1
    assert batch.sparse_row_states[0].physical_valid_length == 3


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("forward failed"), asyncio.CancelledError()],
)
def test_gemma_decode_error_or_cancellation_rolls_back_real_cache(failure):
    model = _OneTokenModel(failure=failure)
    batch = _batch(1, "rollback", config=_config(target_model=model))
    before = tuple(
        batch_cache_cursor(entry, logical_positions=(4,)) for entry in batch.cache
    )
    generator = _generator_for(batch, model)

    with pytest.raises(type(failure)):
        generator._step(batch.y, batch.cache)

    after = tuple(
        batch_cache_cursor(entry, logical_positions=(4,)) for entry in batch.cache
    )
    assert after == before
    assert batch.sparse_row_states == (_row(),)
    assert all("update_and_fetch" not in entry.__dict__ for entry in batch.cache)


def test_gemma_filter_and_extend_are_atomic_and_restore_source():
    destination = _batch(1, "first")
    source = _batch(2, "second")
    source_refs = tuple(
        (entry.keys, entry.values, entry.offset, entry.left_padding)
        for entry in source.cache
    )

    destination.extend(source)

    assert destination.uids == [1, 2]
    assert destination.sparse_row_states == (_row(), _row())
    for entry, refs in zip(source.cache, source_refs, strict=True):
        assert entry.keys is refs[0]
        assert entry.values is refs[1]
        assert entry.offset is refs[2]
        assert entry.left_padding is refs[3]
    destination.filter([1])
    assert destination.uids == [2]
    assert destination.request_ids == ["second"]
    for entry in destination.cache:
        assert batch_cache_cursor(entry, logical_positions=(4,)).total_writes == (2,)


def test_gemma_filter_wrapper_restores_cache_and_rows_on_failure():
    batch = _batch(1, "first")
    batch.extend(_batch(2, "second"))
    before = tuple(
        batch_cache_cursor(entry, logical_positions=(4, 4)) for entry in batch.cache
    )
    original_filter = batch.cache[1].filter

    def failing_filter(indices):
        original_filter(indices)
        raise RuntimeError("filter failed")

    batch.cache[1].filter = failing_filter
    with pytest.raises(RuntimeError, match="filter failed"):
        batch.filter([0])

    after = tuple(
        batch_cache_cursor(entry, logical_positions=(4, 4)) for entry in batch.cache
    )
    assert after == before
    assert batch.uids == [1, 2]
    assert batch.sparse_row_states == (_row(), _row())


def test_gemma_extend_wrapper_restores_both_operands_on_failure():
    destination = _batch(1, "destination")
    source = _batch(2, "source")
    destination_before = tuple(
        batch_cache_cursor(entry, logical_positions=(4,)) for entry in destination.cache
    )
    source_before = tuple(
        batch_cache_cursor(entry, logical_positions=(4,)) for entry in source.cache
    )
    original_extend = destination.cache[1].extend

    def failing_extend(other):
        original_extend(other)
        raise RuntimeError("extend failed")

    destination.cache[1].extend = failing_extend
    with pytest.raises(RuntimeError, match="extend failed"):
        destination.extend(source)

    assert (
        tuple(
            batch_cache_cursor(entry, logical_positions=(4,))
            for entry in destination.cache
        )
        == destination_before
    )
    assert (
        tuple(
            batch_cache_cursor(entry, logical_positions=(4,)) for entry in source.cache
        )
        == source_before
    )
    assert destination.uids == [1]
    assert source.uids == [2]


@pytest.mark.parametrize("cancel_at", ["processor", "sampler"])
def test_gemma_active_adoption_cancellation_restores_extended_cache(cancel_at):
    config = _config()
    active = _batch(1, "active", config=config)
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.active_batch = active
    generator.model = _TARGET_MODEL
    generator.language_model = _TARGET_MODEL
    generator.processor = _PROCESSOR
    generator._target_forward_context = lambda _forward: nullcontext()
    generator._expected_sparse_execution_config = config.execution_config
    generator._expected_gemma_sparse_config = config
    generator._mtp_installed = False
    generator._aborted_request_ids = set()
    generator.unprocessed_requests = []
    generator.uid_counter = 2
    generator._pending_error_responses = []
    before = tuple(
        batch_cache_cursor(entry, logical_positions=(4,)) for entry in active.cache
    )

    def cancel_processor(_tokens, _logits):
        raise asyncio.CancelledError()

    def cancel_sampler(_logprobs):
        raise asyncio.CancelledError()

    generator._sampling_for_request = lambda _request: (
        [cancel_processor] if cancel_at == "processor" else None,
        (
            cancel_sampler
            if cancel_at == "sampler"
            else (lambda logprobs: mx.argmax(logprobs, axis=-1))
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        generator.adopt_prefilled_sparse_row(
            _request(2, "candidate"),
            _scalar_cache(),
            mx.zeros((1, 1, 8)),
            SparseCacheState((_row(),)),
        )

    after = tuple(
        batch_cache_cursor(entry, logical_positions=(4,)) for entry in active.cache
    )
    assert after == before
    assert generator.active_batch is active
    assert active.uids == [1]
    assert active.sparse_row_states == (_row(),)


def test_install_mtp_after_gemma_adoption_bypasses_drafts():
    model = _OneTokenModel()
    config = _config(target_model=model)
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.active_batch = None
    generator.model = model
    generator.language_model = model
    generator.processor = _PROCESSOR
    generator._target_forward_context = lambda _forward: nullcontext()
    generator._expected_sparse_execution_config = config.execution_config
    generator._expected_gemma_sparse_config = config
    generator._mtp_installed = False
    generator._aborted_request_ids = set()
    generator.unprocessed_requests = []
    generator.uid_counter = 1
    generator.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
    generator._sparse_checkpoint_scalar_reads = 0
    generator._sparse_checkpoint_host_scalar_reads = 0
    generator.adopt_prefilled_sparse_row(
        _request(-1, "mtp-after"),
        _scalar_cache(),
        mx.zeros((1, 1, 8)),
        SparseCacheState((_row(),)),
    )

    install_mtp_mllm(generator, model)
    generator._step(generator.active_batch.y[:, None], generator.active_batch.cache)

    assert model.calls == 1
    stats = generator.get_mtp_stats()
    assert stats["attempted"] == 0
    assert stats["bypass_counts"]["sparse_rows"] == 1


def test_gemma_cross_namespace_alias_profile_and_prefix_paths_fail_closed():
    from mlx_lm.models.cache import KVCache as LmKVCache

    config = _config()
    wrong_namespace = [LmKVCache() for _ in range(GEMMA4_E2B.owner_count)]
    with pytest.raises(GemmaCacheTopologyError, match="backend"):
        config.validate_cache(wrong_namespace)

    with pytest.raises(SparseBatchCompatibilityError, match="profile"):
        _batch(1, "profile", row=_row(target_id="other"), config=config)

    batch = _batch(1, "prefix")
    with pytest.raises(SparseBatchError, match="prefix cache"):
        batch.extract_cache(0)

    aliased = _batch(2, "alias", cache=batch.cache)
    before = tuple(entry.offset for entry in batch.cache)
    with pytest.raises(GemmaCacheError, match="overlap"):
        batch.extend(aliased)
    assert all(
        entry.offset is offset
        for entry, offset in zip(batch.cache, before, strict=True)
    )


def test_gemma_uniform_cache_row_mismatch_rejected_at_construction_and_decode():
    with pytest.raises(SparseBatchCompatibilityError, match="row occupancy"):
        _batch(1, "construct-mismatch", row=_row(physical=1), cache=_batch_cache())

    model = _OneTokenModel()
    batch = _batch(1, "decode-mismatch", config=_config(target_model=model))
    for entry in batch.cache:
        entry.offset = mx.array([1])
    generator = _generator_for(batch, model)

    with pytest.raises(SparseBatchCompatibilityError, match="row occupancy"):
        generator._step(batch.y, batch.cache)

    assert model.calls == 0


def test_gemma_prepared_attestation_rejects_forgery_and_wrong_live_identity():
    with pytest.raises(SparseBatchCompatibilityError, match="target_id"):
        _config(target_id="wrong-target")

    with pytest.raises(TypeError, match="prepared runtime"):
        GemmaPreparedTargetAttestation(
            target_model=_TARGET_MODEL,
            text_model=_TARGET_MODEL,
            processor=_PROCESSOR,
            target_artifact_hash=_TARGET_HASH,
            artifact=GEMMA4_E2B,
            layer_types=GEMMA4_E2B.layer_types,
            previous_kvs=GEMMA4_E2B.previous_kvs,
            _token=object(),
        )

    with pytest.raises(SparseBatchCompatibilityError, match="artifact bytes"):
        _prepared_attestation(target_hash="a" * 64)

    with pytest.raises(SparseBatchCompatibilityError, match="loaded target/processor"):
        _prepared_attestation(attested_target_model=object())

    wrong_topology = _LiveGemmaTopology(layer_types=("wrong",))
    with pytest.raises(SparseBatchCompatibilityError, match="layer topology"):
        _prepared_attestation(target_model=wrong_topology)

    config = _config()
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.active_batch = None
    generator.model = object()
    generator.language_model = object()
    generator.processor = _PROCESSOR
    generator._expected_sparse_execution_config = config.execution_config
    with pytest.raises(SparseBatchCompatibilityError, match="generator-owned"):
        generator.set_expected_gemma_sparse_config(config)


def test_gemma_heterogeneous_writes_and_mtp_fail_closed():
    cache = _batch_cache()
    for entry in cache:
        entry.offset = mx.array([2, 3])
        entry.left_padding = mx.array([0, 0])
        entry.keys = mx.concatenate([entry.keys, entry.keys], axis=0)
        entry.values = mx.concatenate([entry.values, entry.values], axis=0)
    with pytest.raises(GemmaCacheError, match="heterogeneous writes"):
        MLLMBatch(
            uids=[1, 2],
            request_ids=["one", "two"],
            y=mx.array([7, 7]),
            logprobs=[mx.zeros(8), mx.zeros(8)],
            max_tokens=[4, 4],
            num_tokens=[0, 0],
            cache=cache,
            requests=[_request(1, "one"), _request(2, "two")],
            sparse_row_states=(_row(), _row()),
            gemma_sparse_config=_config(),
        )

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.active_batch = None
    generator._target_forward_context = lambda _forward: nullcontext()
    generator._expected_sparse_execution_config = _config().execution_config
    generator._expected_gemma_sparse_config = _config()
    generator.model = _TARGET_MODEL
    generator.language_model = _TARGET_MODEL
    generator.processor = _PROCESSOR
    generator._mtp_installed = True
    generator._aborted_request_ids = set()
    generator.unprocessed_requests = []
    generator.uid_counter = 1
    with pytest.raises(SparseAdoptionError, match="MTP"):
        generator.adopt_prefilled_sparse_row(
            _request(-1, "mtp"),
            _scalar_cache(),
            mx.zeros((1, 1, 8)),
            SparseCacheState((_row(),)),
        )
