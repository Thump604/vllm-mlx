from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache as MlxLmKVCache
from mlx_lm.models.cache import RotatingKVCache as MlxLmRotatingKVCache
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

import vllm_mlx.specprefill_gemma_cache as gemma_cache

from vllm_mlx.specprefill_gemma_cache import (
    GEMMA4_E2B,
    GemmaArtifactSpec,
    GemmaCacheCheckpoint,
    GemmaCacheBackend,
    GemmaCacheError,
    GemmaCacheTopologyError,
    GemmaCacheTransactionError,
    GemmaOneTokenTransaction,
    atomic_batch_extend,
    atomic_batch_filter,
    batch_cache_cursor,
    normalize_batch_rotating,
    normalize_scalar_rotating,
    run_atomic_one_token,
    scalar_cache_cursor,
    validate_gemma_cache_topology,
    validate_aligned_scalar_cache,
    validate_homogeneous_batch_lane,
)


# Minimal snapshots of the installed config.json text_config sections.  The
# hashes identify the complete source configs without making tests depend on a
# workstation path.  No model weights are represented or loaded here.
_CONFIG_SNAPSHOTS = (
    {
        "artifact_id": "gemma4-e2b",
        "config_sha256": "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
        "num_hidden_layers": 35,
        "num_kv_shared_layers": 20,
        "sliding_window": 512,
        "layer_types": tuple(
            """
            sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            """.split()
        ),
    },
    {
        "artifact_id": "gemma4-31b",
        "config_sha256": "c58b3d20b54d0ed7e5650c2d6c7d13f0f7bdbc190a0fe428b18bfb2a8d182eb4",
        "num_hidden_layers": 60,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "layer_types": tuple(
            """
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            """.split()
        ),
    },
    {
        "artifact_id": "gemma4-26b-a4b",
        "config_sha256": "431b1c1364c7f043612717de5de557069d23d8ee06a532b887224251043ec495",
        "num_hidden_layers": 30,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "layer_types": tuple(
            """
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention full_attention
            """.split()
        ),
    },
)


def _spec_from_snapshot(config):
    layer_types = tuple(config["layer_types"])
    owner_count = config["num_hidden_layers"] - config["num_kv_shared_layers"]
    previous = list(range(len(layer_types)))
    latest = {}
    for layer, layer_type in enumerate(layer_types):
        if layer < owner_count:
            latest[layer_type] = layer
        else:
            previous[layer] = latest[layer_type]
    return GemmaArtifactSpec(
        artifact_id=config["artifact_id"],
        layer_types=layer_types,
        previous_kvs=tuple(previous),
        owner_count=owner_count,
        sliding_window=config["sliding_window"],
    )


def _token(values, *, rows=1):
    data = mx.array(list(values), dtype=mx.float32).reshape(1, 1, -1, 1)
    return mx.broadcast_to(data, (rows, 1, data.shape[2], 1))


def _append(cache, values, *, rows=1):
    token = _token(values, rows=rows)
    result = cache.update_and_fetch(token, token + 100)
    mx.eval(*result)
    return result


def _artifact_cache(spec, *, batch=False, backend=GemmaCacheBackend.MLX_VLM):
    entries = []
    for layer_type in spec.layer_types[: spec.owner_count]:
        if layer_type == "full_attention":
            entries.append(
                BatchKVCache([0])
                if batch
                else (
                    MlxLmKVCache()
                    if backend is GemmaCacheBackend.MLX_LM
                    else KVCache()
                )
            )
        else:
            entries.append(
                BatchRotatingKVCache(spec.sliding_window, [0])
                if batch
                else (
                    MlxLmRotatingKVCache(spec.sliding_window, keep=0)
                    if backend is GemmaCacheBackend.MLX_LM
                    else RotatingKVCache(spec.sliding_window, keep=0)
                )
            )
    return entries


@pytest.mark.parametrize(
    "config", _CONFIG_SNAPSHOTS, ids=lambda config: config["artifact_id"]
)
def test_certified_artifact_topologies_are_exact(config):
    spec = _spec_from_snapshot(config)
    topology = validate_gemma_cache_topology(
        spec,
        layer_types=spec.layer_types,
        previous_kvs=spec.previous_kvs,
        cache=_artifact_cache(spec),
    )
    assert len(topology.layer_to_slot) == config["num_hidden_layers"]
    assert topology.owner_layers == tuple(range(spec.owner_count))
    assert topology.sliding_window == config["sliding_window"]
    assert len(config["config_sha256"]) == 64


def test_explicit_backends_accept_exact_types_and_reject_crossed_pairs():
    lm_cache = _artifact_cache(GEMMA4_E2B, backend=GemmaCacheBackend.MLX_LM)
    validate_gemma_cache_topology(
        GEMMA4_E2B,
        layer_types=GEMMA4_E2B.layer_types,
        previous_kvs=GEMMA4_E2B.previous_kvs,
        cache=lm_cache,
        backend=GemmaCacheBackend.MLX_LM,
    )
    with pytest.raises(GemmaCacheTopologyError, match="does not match"):
        validate_gemma_cache_topology(
            GEMMA4_E2B,
            layer_types=GEMMA4_E2B.layer_types,
            previous_kvs=GEMMA4_E2B.previous_kvs,
            cache=lm_cache,
            backend=GemmaCacheBackend.MLX_VLM,
        )
    crossed = list(lm_cache)
    crossed[0] = RotatingKVCache(512, keep=0)
    with pytest.raises(GemmaCacheError, match="cross"):
        validate_gemma_cache_topology(
            GEMMA4_E2B,
            layer_types=GEMMA4_E2B.layer_types,
            previous_kvs=GEMMA4_E2B.previous_kvs,
            cache=crossed,
            backend=GemmaCacheBackend.MLX_LM,
        )


def test_e2b_followers_use_last_owner_of_matching_type():
    assert GEMMA4_E2B.previous_kvs[:15] == tuple(range(15))
    assert set(GEMMA4_E2B.previous_kvs[15:]) == {13, 14}
    for layer, owner in enumerate(GEMMA4_E2B.previous_kvs):
        assert GEMMA4_E2B.layer_types[layer] == GEMMA4_E2B.layer_types[owner]


@pytest.mark.parametrize("corruption", ["type", "owner", "window", "count"])
def test_topology_validation_fails_closed(corruption):
    spec = GEMMA4_E2B
    layer_types = list(spec.layer_types)
    previous = list(spec.previous_kvs)
    cache = _artifact_cache(spec)
    if corruption == "type":
        cache[0] = KVCache()
    elif corruption == "owner":
        previous[16] = 14
    elif corruption == "window":
        cache[0].max_size = 1024
    else:
        cache.pop()
    with pytest.raises(GemmaCacheTopologyError):
        validate_gemma_cache_topology(
            spec, layer_types=layer_types, previous_kvs=previous, cache=cache
        )


def test_scalar_cursors_separate_total_resident_circular_and_logical():
    cache = RotatingKVCache(4, keep=0)
    _append(cache, range(6))
    normalize_scalar_rotating(cache)
    cursor = scalar_cache_cursor(cache, logical_position=19)
    assert cursor.total_writes == 6
    assert cursor.resident_tokens == 4
    assert cursor.circular_index == 4
    assert cursor.logical_position == 19


@pytest.mark.parametrize("length", [3, 4, 5])
def test_rotating_boundaries_and_next_decode_preserve_total_offset(length):
    cache = RotatingKVCache(4, keep=0)
    _append(cache, range(length))
    normalize_scalar_rotating(cache)
    before = cache.offset
    transaction = GemmaOneTokenTransaction([cache], logical_positions=[20])
    _append(cache, [99])
    transaction.commit(logical_positions=[21])
    assert cache.offset == before + 1
    assert cache.keys.shape[2] == min(before + 1, 4)
    assert cache._idx <= 4


@pytest.mark.parametrize("initial", [1, 4])
def test_decode_commit_keeps_cache_allocation_and_never_copies_full_window(
    initial, monkeypatch
):
    cache = RotatingKVCache(4, keep=0)
    _append(cache, range(initial))
    normalize_scalar_rotating(cache)
    keys = cache.keys
    copied_lengths = []
    original = mx.contiguous

    def track(array, *args, **kwargs):
        copied_lengths.append(array.shape[2])
        return original(array, *args, **kwargs)

    monkeypatch.setattr(gemma_cache.mx, "contiguous", track)
    transaction = GemmaOneTokenTransaction([cache], logical_positions=[initial])
    _append(cache, [9])
    transaction.commit(logical_positions=[initial + 1])
    assert cache.keys is keys
    assert 4 not in copied_lengths
    assert copied_lengths in ([], [1, 1])


def test_multi_token_concat_normalizes_to_latest_window_without_clamping_writes():
    cache = RotatingKVCache(4, keep=0)
    _append(cache, [0, 1, 2])
    _append(cache, [3, 4, 5])
    assert cache.keys.shape[2] == 6
    normalize_scalar_rotating(cache)
    assert cache.offset == 6
    assert cache._idx == 4
    assert cache.keys.reshape(-1).tolist() == [2, 3, 4, 5]


def test_batch_multi_token_concat_normalizes_without_clamping_any_cursor():
    cache = BatchRotatingKVCache(4, [0, 0])
    _append(cache, [0, 1, 2], rows=2)
    _append(cache, [3, 4, 5], rows=2)
    offsets = cache.offset
    normalize_batch_rotating(cache)
    cursor = batch_cache_cursor(cache, logical_positions=[10, 10])
    assert cursor.total_writes == (6, 6)
    assert cursor.physical_write_cursor == 6
    assert cursor.resident_tokens == 4
    assert cursor.circular_index == 4
    assert cache.offset is offsets
    assert cache.left_padding.tolist() == [-2, -2]
    assert cache.keys[0].reshape(-1).tolist() == [2, 3, 4, 5]


def test_checkpoint_restores_every_scalar_and_batch_field_by_reference():
    scalar = RotatingKVCache(4, keep=0)
    _append(scalar, [1, 2, 3, 4])
    batch = BatchRotatingKVCache(4, [0, 0])
    _append(batch, [1], rows=2)
    scalar_keys, batch_keys = scalar.keys, batch.keys
    scalar_snapshot = (scalar.offset, scalar._idx, scalar.max_size, scalar.keep)
    batch_snapshot = (
        batch.offset,
        batch.left_padding,
        batch._idx,
        batch._offset,
        batch.rotated,
        batch._lengths,
        batch.max_size,
    )
    checkpoint = GemmaCacheCheckpoint([scalar, batch])
    scalar.keys = mx.zeros_like(scalar.keys)
    scalar.offset, scalar._idx, scalar.max_size, scalar.keep = 2, 1, 8, 1
    batch.keys = mx.zeros_like(batch.keys)
    batch.offset = mx.array([7, 8])
    batch.left_padding = mx.array([2, 3])
    batch._idx, batch._offset, batch.rotated = 0, 9, True
    batch._lengths, batch.max_size = mx.array([1, 1]), 8
    checkpoint.restore()
    assert scalar.keys is scalar_keys
    assert (scalar.offset, scalar._idx, scalar.max_size, scalar.keep) == scalar_snapshot
    assert batch.keys is batch_keys
    assert batch.offset is batch_snapshot[0]
    assert batch.left_padding is batch_snapshot[1]
    assert (
        batch._idx,
        batch._offset,
        batch.rotated,
        batch._lengths,
        batch.max_size,
    ) == batch_snapshot[2:]


def test_saturated_one_token_failure_restores_overwritten_slot_and_metadata():
    cache = RotatingKVCache(4, keep=0)
    _append(cache, [0, 1, 2, 3])
    normalize_scalar_rotating(cache)
    before = cache.keys.tolist()

    def forward():
        _append(cache, [99])
        return mx.array([1])

    with pytest.raises(RuntimeError, match="cancelled"):
        run_atomic_one_token(
            [cache],
            logical_positions=[4],
            forward=forward,
            evaluate=lambda _: (_ for _ in ()).throw(RuntimeError("cancelled")),
        )
    mx.eval(cache.keys)
    assert cache.offset == 4
    assert cache._idx == 4
    assert cache.keys.tolist() == before


def test_committed_transaction_discards_journal_and_cannot_be_recommitted():
    cache = KVCache()
    transaction = GemmaOneTokenTransaction([cache], logical_positions=[0])
    _append(cache, [7])
    transaction.commit(logical_positions=[1])
    accepted_keys, accepted_value = cache.keys, cache.keys[..., 0, :].tolist()
    transaction.rollback()
    assert cache.keys is accepted_keys
    assert cache.offset == 1
    assert cache.keys[..., 0, :].tolist() == accepted_value
    with pytest.raises(GemmaCacheTransactionError, match="already committed"):
        transaction.commit(logical_positions=[1])
    assert cache.offset == 1
    assert cache.keys[..., 0, :].tolist() == accepted_value


def test_one_token_guard_rejects_second_update_before_mutation_and_rolls_back():
    cache = RotatingKVCache(4, keep=0)
    _append(cache, [0, 1, 2, 3])
    normalize_scalar_rotating(cache)
    keys, before = cache.keys, cache.keys.tolist()
    transaction = GemmaOneTokenTransaction([cache], logical_positions=[4])
    _append(cache, [1])
    with pytest.raises(GemmaCacheTransactionError, match="more than once"):
        _append(cache, [2])
    transaction.rollback()
    assert cache.keys is keys
    assert cache.keys.tolist() == before
    assert cache.offset == cache._idx == 4


def test_one_token_guard_rejects_multi_token_update_before_mutation():
    cache = KVCache()
    transaction = GemmaOneTokenTransaction([cache], logical_positions=[8])
    with pytest.raises(GemmaCacheTransactionError, match="multi-token"):
        _append(cache, [1, 2])
    transaction.rollback()
    assert cache.offset == 0
    assert cache.keys is None


def test_scalar_and_batch_owner_alignment_is_exact():
    full, rotating = KVCache(), RotatingKVCache(4, keep=0)
    _append(full, [1])
    with pytest.raises(GemmaCacheError, match="physical writes"):
        validate_aligned_scalar_cache([full, rotating], logical_position=1)

    batch_full = BatchKVCache([0, 0])
    batch_rotating = BatchRotatingKVCache(4, [0, 0])
    _append(batch_full, [1], rows=2)
    with pytest.raises(GemmaCacheError, match="physical writes"):
        validate_homogeneous_batch_lane(
            [batch_full, batch_rotating], logical_positions=[1, 1]
        )


def test_batch_cursor_and_lane_reject_heterogeneous_physical_or_logical_rows():
    batch = BatchKVCache([0, 1])
    _append(batch, [1], rows=2)
    cursor = batch_cache_cursor(batch, logical_positions=[5, 5])
    assert cursor.total_writes == (1, 0)
    with pytest.raises(GemmaCacheError, match="heterogeneous writes"):
        validate_homogeneous_batch_lane([batch], logical_positions=[5, 5])
    equal = BatchKVCache([0, 0])
    with pytest.raises(GemmaCacheError, match="equal logical"):
        validate_homogeneous_batch_lane([equal], logical_positions=[5, 6])


def test_batch_one_token_commit_tracks_lifetime_writes_across_wrap():
    left = RotatingKVCache(4, keep=0)
    right = RotatingKVCache(4, keep=0)
    _append(left, range(5))
    _append(right, range(5))
    normalize_scalar_rotating(left)
    normalize_scalar_rotating(right)
    batch = BatchRotatingKVCache.merge([left, right])
    transaction = GemmaOneTokenTransaction([batch], logical_positions=[9, 9])
    _append(batch, [10], rows=2)
    transaction.commit(logical_positions=[10, 10])
    cursor = batch_cache_cursor(batch, logical_positions=[10, 10])
    assert cursor.total_writes == (6, 6)
    assert cursor.circular_index == 1
    assert cursor.rotated is True


def test_batch_rotation_state_canonicalizes_after_a_complete_wrap():
    rows = [RotatingKVCache(4, keep=0), RotatingKVCache(4, keep=0)]
    for row in rows:
        _append(row, range(4))
        normalize_scalar_rotating(row)
    batch = BatchRotatingKVCache.merge(rows)
    assert (batch._idx, batch.rotated) == (4, False)
    logical = 4
    for expected_index in (1, 2, 3):
        transaction = GemmaOneTokenTransaction(
            [batch], logical_positions=[logical, logical]
        )
        _append(batch, [logical], rows=2)
        logical += 1
        transaction.commit(logical_positions=[logical, logical])
        assert (batch._idx, batch.rotated) == (expected_index, True)
    transaction = GemmaOneTokenTransaction(
        [batch], logical_positions=[logical, logical]
    )
    _append(batch, [logical], rows=2)
    logical += 1
    transaction.commit(logical_positions=[logical, logical])
    assert (batch._idx, batch.rotated) == (4, False)


@pytest.mark.parametrize(
    ("offset", "index", "rotated", "message"),
    [
        (3, 3, True, "unsaturated"),
        (4, 4, True, "normalized saturated"),
        (5, 1, False, "wrapped"),
    ],
)
def test_batch_rotation_state_machine_rejects_malformed_states(
    offset, index, rotated, message
):
    cache = BatchRotatingKVCache(4, [0])
    cache.keys = mx.zeros((1, 1, 4, 1))
    cache.values = mx.zeros((1, 1, 4, 1))
    cache.offset = mx.array([offset])
    cache._offset = offset
    cache._idx = index
    cache.rotated = rotated
    with pytest.raises(GemmaCacheError, match=message):
        batch_cache_cursor(cache, logical_positions=[offset])


def test_batch_lane_requires_equal_rotation_state_across_owners():
    def cache(index, rotated):
        entry = BatchRotatingKVCache(4, [0])
        entry.keys = mx.zeros((1, 1, 4, 1))
        entry.values = mx.zeros((1, 1, 4, 1))
        entry.offset = mx.array([5])
        entry._offset = 5
        entry._idx = index
        entry.rotated = rotated
        return entry

    with pytest.raises(GemmaCacheError, match="resident state"):
        validate_homogeneous_batch_lane(
            [cache(4, False), cache(1, True)], logical_positions=[5]
        )


def test_atomic_filter_keeps_selected_rows_and_rolls_back_mid_failure(monkeypatch):
    full = BatchKVCache([0, 0])
    rotating = BatchRotatingKVCache(4, [0, 0])
    _append(full, [1], rows=2)
    _append(rotating, [1], rows=2)
    selected = atomic_batch_filter(
        [full, rotating], [1], logical_positions=[4, 4]
    )
    assert selected == (4,)
    assert full.keys.shape[0] == rotating.keys.shape[0] == 1

    full = BatchKVCache([0, 0])
    rotating = BatchRotatingKVCache(4, [0, 0])
    _append(full, [1], rows=2)
    _append(rotating, [1], rows=2)
    full_keys = full.keys
    monkeypatch.setattr(
        BatchRotatingKVCache,
        "filter",
        lambda self, indices: (_ for _ in ()).throw(RuntimeError("mid-filter")),
    )
    with pytest.raises(RuntimeError, match="mid-filter"):
        atomic_batch_filter([full, rotating], [0], logical_positions=[4, 4])
    assert full.keys is full_keys
    assert full.keys.shape[0] == 2


def test_atomic_filter_rejects_duplicate_owners_before_mutation():
    cache = BatchKVCache([0, 0])
    _append(cache, [1], rows=2)
    keys, offset = cache.keys, cache.offset
    with pytest.raises(GemmaCacheError, match="duplicate owners"):
        atomic_batch_filter([cache, cache], [0], logical_positions=[1, 1])
    assert cache.keys is keys
    assert cache.offset is offset


def test_atomic_extend_restores_source_on_success_and_both_on_failure(monkeypatch):
    destination = [BatchRotatingKVCache(4, [0])]
    source = [BatchRotatingKVCache(4, [0])]
    _append(destination[0], [1], rows=1)
    _append(source[0], [2], rows=1)
    source_keys, source_idx = source[0].keys, source[0]._idx
    atomic_batch_extend(destination, source)
    assert destination[0].keys.shape[0] == 2
    assert source[0].keys is source_keys
    assert source[0]._idx == source_idx

    destination = [BatchKVCache([0]), BatchRotatingKVCache(4, [0])]
    source = [BatchKVCache([0]), BatchRotatingKVCache(4, [0])]
    for entry in destination + source:
        _append(entry, [1], rows=1)
    destination_refs = tuple(entry.keys for entry in destination)
    source_refs = tuple(entry.keys for entry in source)
    monkeypatch.setattr(
        BatchRotatingKVCache,
        "extend",
        lambda self, other: (_ for _ in ()).throw(RuntimeError("mid-extend")),
    )
    with pytest.raises(RuntimeError, match="mid-extend"):
        atomic_batch_extend(destination, source)
    assert tuple(entry.keys for entry in destination) == destination_refs
    assert tuple(entry.keys for entry in source) == source_refs


@pytest.mark.parametrize("duplicate_side", ["destination", "source", "overlap"])
def test_atomic_extend_rejects_aliasing_before_mutation(duplicate_side):
    left = BatchKVCache([0])
    right = BatchKVCache([0])
    if duplicate_side == "destination":
        destination, source = [left, left], [right, BatchKVCache([0])]
    elif duplicate_side == "source":
        destination, source = [left, BatchKVCache([0])], [right, right]
    else:
        destination, source = [left], [left]
    with pytest.raises(GemmaCacheError, match="duplicate|overlap"):
        atomic_batch_extend(destination, source)


@pytest.mark.parametrize("operation", ["normalize", "filter", "extend"])
def test_deferred_cache_failure_restores_all_references(operation, monkeypatch):
    monkeypatch.setattr(
        gemma_cache,
        "_evaluate_cache",
        lambda cache: (_ for _ in ()).throw(RuntimeError("deferred")),
    )
    if operation == "normalize":
        cache = RotatingKVCache(4, keep=0)
        _append(cache, [1, 2, 3, 4, 5])
        keys, offset, index = cache.keys, cache.offset, cache._idx
        with pytest.raises(RuntimeError, match="deferred"):
            normalize_scalar_rotating(cache)
        assert cache.keys is keys
        assert (cache.offset, cache._idx) == (offset, index)
    elif operation == "filter":
        cache = BatchKVCache([0, 0])
        _append(cache, [1], rows=2)
        keys, offset = cache.keys, cache.offset
        with pytest.raises(RuntimeError, match="deferred"):
            atomic_batch_filter([cache], [0], logical_positions=[1, 1])
        assert cache.keys is keys
        assert cache.offset is offset
    else:
        destination, source = [BatchKVCache([0])], [BatchKVCache([0])]
        _append(destination[0], [1])
        _append(source[0], [2])
        destination_keys, source_keys = destination[0].keys, source[0].keys
        with pytest.raises(RuntimeError, match="deferred"):
            atomic_batch_extend(destination, source)
        assert destination[0].keys is destination_keys
        assert source[0].keys is source_keys


def test_eos_or_cancel_rollback_is_idempotent():
    cache = RotatingKVCache(4, keep=0)
    _append(cache, [0, 1, 2, 3])
    normalize_scalar_rotating(cache)
    transaction = GemmaOneTokenTransaction([cache], logical_positions=[4])
    _append(cache, [5])
    transaction.rollback()
    transaction.rollback()
    assert cache.offset == 4
    assert cache._idx == 4


def test_malformed_saturated_prefix_is_rejected_before_negative_dimension_path():
    def malformed():
        cache = RotatingKVCache(4, keep=0)
        cache.keys = mx.zeros((1, 1, 3, 1))
        cache.values = mx.zeros((1, 1, 3, 1))
        cache.offset = 6
        cache._idx = 3
        return cache

    # mlx-lm's next one-token growth would calculate max_size - offset == -2.
    with pytest.raises(ValueError, match="Negative dimensions"):
        _append(malformed(), [7])
    with pytest.raises(GemmaCacheError, match="short resident buffer"):
        GemmaOneTokenTransaction([malformed()], logical_positions=[6])


def test_batch_malformed_prefix_and_window_mismatch_fail_closed():
    malformed = BatchRotatingKVCache(4, [0])
    malformed.keys = mx.zeros((1, 1, 3, 1))
    malformed.values = mx.zeros((1, 1, 3, 1))
    malformed.offset = mx.array([6])
    malformed._offset = 6
    malformed._idx = 3
    with pytest.raises(GemmaCacheError, match="short buffer"):
        GemmaOneTokenTransaction([malformed], logical_positions=[6])

    with pytest.raises(GemmaCacheError, match="windows differ"):
        atomic_batch_extend(
            [BatchRotatingKVCache(4, [0])],
            [BatchRotatingKVCache(8, [0])],
        )
