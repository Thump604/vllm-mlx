"""Synthetic real-cache coverage for native-Qwen CB transactions."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

from vllm_mlx.native_mtp_batch import (
    NativeMTPBatch,
    NativeMTPBatchError,
    NativeMTPBatchPosition,
    NativeMTPBatchRow,
)
from vllm_mlx.native_mtp_request import NativeMTPRequestConfig, NativeMTPSampling


@dataclass(frozen=True)
class _Capability:
    supported: bool = True
    reason: str = "supported"
    num_layers: int = 1


class _Model:
    def __init__(self, *, capability=None, linear_layers=1, attention_layers=1):
        self._capability = capability or _Capability()
        self.layers = [
            *[type("_Linear", (), {"is_linear": True})() for _ in range(linear_layers)],
            *[
                type("_Attention", (), {"is_linear": False})()
                for _ in range(attention_layers)
            ],
        ]
        self.mtp = type("_MTP", (), {"layers": [object()]})()
        self.model = type("_Core", (), {"pipeline_size": 1})()

    @property
    def mtp_capability(self):
        return self._capability


class _FreshCapabilityModel(_Model):
    @property
    def mtp_capability(self):
        value = self._capability
        return _Capability(value.supported, value.reason, value.num_layers)


def _sampling(seed):
    return NativeMTPRequestConfig(
        sampling=NativeMTPSampling(
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            seed=seed,
        ),
        num_draft_tokens=2,
    )


def _advance_kv(cache, count, *, batch_size=1, value=1.0):
    keys = mx.full((batch_size, 1, count, 8), value, dtype=mx.float32)
    cache.update_and_fetch(keys, keys + 10)


def _row(model, uid, *, position=0, seed=None):
    recurrent = ArraysCache(2)
    recurrent[0] = mx.array([[float(uid), 1.0]])
    recurrent[1] = mx.array([[float(uid), 2.0]])
    attention = KVCache()
    head = KVCache()
    if position:
        _advance_kv(attention, position, value=float(uid + 1))
        _advance_kv(head, position, value=float(uid + 2))
    return NativeMTPBatchRow(
        uid=uid,
        target_model=model,
        capability=model.mtp_capability,
        backbone_cache=[recurrent, attention],
        mtp_cache=[head],
        position=NativeMTPBatchPosition(position, position, position),
        request_config=_sampling(uid if seed is None else seed),
    )


def _advance_batch(batch, count=1, *, recurrent_delta=1.0):
    batch_size = len(batch.uids)
    for entry in (*batch.backbone_cache, *batch.mtp_cache):
        if type(entry) is BatchKVCache:
            _advance_kv(entry, count, batch_size=batch_size)
        elif type(entry) is ArraysCache:
            entry.cache = [
                value + recurrent_delta if value is not None else None
                for value in entry.cache
            ]


def _positions(batch, delta):
    return {
        uid: NativeMTPBatchPosition(
            position.logical_cursor + delta,
            position.backbone_tokens + delta,
            position.mtp_tokens + delta,
        )
        for uid, position in batch.positions
    }


def test_full_verified_batch_commit_advances_all_rows_atomically():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 11, position=2), _row(model, 22, position=2)])
    assert all(
        type(entry) in {BatchKVCache, ArraysCache}
        for entry in (*batch.backbone_cache, *batch.mtp_cache)
    )

    batch.checkpoint()
    _advance_batch(batch, 2, recurrent_delta=2)
    sealed = _positions(batch, 2)
    batch.seal_verified(sealed)
    batch.commit(sealed)

    assert dict(batch.positions) == sealed
    assert batch.checkpoint_active is False
    assert batch.backbone_cache[1]._idx == 4
    assert batch.mtp_cache[0]._idx == 4


def test_partial_recurrent_rejection_rolls_back_partitions_replays_and_rejoins():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1), _row(model, 2)])
    initial = [value + mx.zeros(value.shape) for value in batch.backbone_cache[0].cache]

    batch.checkpoint()
    _advance_batch(batch, 3, recurrent_delta=30)
    verified = _positions(batch, 3)
    batch.seal_verified(verified)
    targets = {
        1: NativeMTPBatchPosition(0, 0, 0),
        2: NativeMTPBatchPosition(2, 2, 2),
    }
    plan = batch.reject_partial(targets)

    assert plan.groups == ((1,), (2,))
    assert batch.backbone_cache[1]._idx == 0
    assert batch.mtp_cache[0]._idx == 0
    assert all(
        mx.array_equal(actual, expected).item()
        for actual, expected in zip(batch.backbone_cache[0].cache, initial)
    )

    partitions = batch.partition_replay_groups()
    assert len(partitions) == 2
    for partition in partitions:
        target = targets[partition.uids[0]]
        while dict(partition.positions)[partition.uids[0]] != target:
            _advance_batch(partition, 1, recurrent_delta=1)
            partition.replay_retained(_positions(partition, 1))
        assert partition.replay_targets == ()

    joined = NativeMTPBatch.join(partitions)
    assert joined.uids == (1, 2)
    assert dict(joined.positions) == targets
    assert joined.backbone_cache[1].offset.tolist() == [0, 2]
    assert joined.mtp_cache[0].offset.tolist() == [0, 2]
    recurrent = joined.backbone_cache[0].cache[0]
    assert recurrent[:, 1].tolist() == [1.0, 3.0]


def test_all_zero_partial_rejection_can_rejoin_without_replay():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1), _row(model, 2)])
    batch.checkpoint()
    _advance_batch(batch, 2, recurrent_delta=9)
    batch.seal_verified(_positions(batch, 2))
    targets = {uid: NativeMTPBatchPosition(0, 0, 0) for uid in batch.uids}

    plan = batch.reject_partial(targets)

    assert plan.groups == ((1, 2),)
    assert batch.replay_targets == ()
    partitions = batch.partition_replay_groups()
    assert partitions == (batch,)
    joined = NativeMTPBatch.join(partitions)
    assert joined.uids == (1, 2)
    assert dict(joined.positions) == targets


def test_partial_commit_is_forbidden_even_if_attention_was_trimmed():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1)])
    batch.checkpoint()
    _advance_batch(batch, 2, recurrent_delta=2)
    batch.seal_verified(_positions(batch, 2))
    batch.backbone_cache[1].trim(1)
    batch.mtp_cache[0].trim(1)

    with pytest.raises(
        NativeMTPBatchError,
        match="native_mtp_batch_partial_commit_requires_replay",
    ):
        batch.commit({1: NativeMTPBatchPosition(1, 1, 1)})

    batch.reject_partial({1: NativeMTPBatchPosition(1, 1, 1)})
    assert batch.backbone_cache[1]._idx == 0
    assert batch.mtp_cache[0]._idx == 0


def test_filter_and_compatible_join_preserve_positions_and_rng_ownership():
    model = _Model()
    batch = NativeMTPBatch(
        [
            _row(model, 1, position=1),
            _row(model, 2, position=2),
            _row(model, 3, position=3),
        ]
    )
    batch.filter((3, 1))
    assert batch.uids == (3, 1)
    assert batch.backbone_cache[1].offset.tolist() == [3, 1]
    assert batch.mtp_cache[0].offset.tolist() == [3, 1]

    other = NativeMTPBatch([_row(model, 4, position=2)])
    joined = NativeMTPBatch.join((batch, other))
    assert joined.uids == (3, 1, 4)
    assert joined.backbone_cache[1].offset.tolist() == [3, 1, 2]
    assert batch.closed and other.closed
    assert batch.backbone_cache == [] and other.mtp_cache == []


def test_filter_failure_restores_every_cache_and_uid(monkeypatch):
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1, position=1), _row(model, 2, position=1)])
    original_filter = BatchKVCache.filter
    calls = 0

    def fail_second_attention(entry, indices):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic filter failure")
        return original_filter(entry, indices)

    monkeypatch.setattr(BatchKVCache, "filter", fail_second_attention)
    with pytest.raises(RuntimeError, match="synthetic filter failure"):
        batch.filter((2,))

    assert batch.uids == (1, 2)
    assert batch.backbone_cache[0].batch_size == 2
    assert batch.backbone_cache[1].offset.tolist() == [1, 1]
    assert batch.mtp_cache[0].offset.tolist() == [1, 1]


def test_cancellation_rolls_back_active_transaction_then_removes_only_uid():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1, position=1), _row(model, 2, position=1)])
    initial = batch.backbone_cache[0].cache[0] + mx.zeros(
        batch.backbone_cache[0].cache[0].shape
    )
    batch.checkpoint()
    _advance_batch(batch, 2, recurrent_delta=8)

    batch.finish(1, "cancelled")

    assert batch.uids == (2,)
    assert batch.backbone_cache[1].offset.tolist() == [1]
    assert batch.mtp_cache[0].offset.tolist() == [1]
    assert mx.array_equal(batch.backbone_cache[0].cache[0], initial[1:2]).item()
    batch.finish(2, "eos")
    assert batch.closed
    assert batch.backbone_cache == [] and batch.mtp_cache == []


def test_rollback_failure_poison_closes_batch():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1)])
    batch.checkpoint()
    _advance_batch(batch, 1)
    batch.backbone_cache[1] = BatchKVCache([0])

    with pytest.raises(NativeMTPBatchError, match="native_mtp_batch_rollback_failed"):
        batch.rollback()
    assert batch.poisoned and batch.closed
    assert batch.uids == ()
    assert batch.backbone_cache == [] and batch.mtp_cache == []
    assert batch.checkpoint_active is False
    assert batch.replay_targets == ()
    assert batch.target_model is None
    assert batch.capability is None


def test_target_capability_topology_and_aliasing_fail_closed():
    model = _Model()
    other = _Model(capability=model.mtp_capability)
    row = _row(model, 1)
    with pytest.raises(NativeMTPBatchError, match="target_identity_mismatch"):
        NativeMTPBatch([row, _row(other, 2)])

    forged = _row(model, 3)
    object.__setattr__(forged, "capability", _Capability(num_layers=2))
    with pytest.raises(NativeMTPBatchError, match="capability_identity_mismatch"):
        NativeMTPBatch([forged])

    wrong = _row(model, 4)
    wrong.backbone_cache.reverse()
    with pytest.raises(TypeError, match="backbone_cache_type_mismatch"):
        NativeMTPBatch([wrong])

    aliased = _row(model, 5)
    aliased.mtp_cache[0] = aliased.backbone_cache[1]
    with pytest.raises(ValueError, match="cache_entries_must_be_unique"):
        NativeMTPBatch([aliased])


def test_fresh_qwen_capability_snapshots_bind_by_value_and_recheck_current_state():
    model = _FreshCapabilityModel()
    first = model.mtp_capability
    second = model.mtp_capability
    assert first == second and first is not second

    batch = NativeMTPBatch([_row(model, 1)])
    batch.checkpoint()
    batch.rollback()
    model._capability = _Capability(False, "native_mtp_weights_not_loaded", 1)
    with pytest.raises(NativeMTPBatchError, match="native_mtp_capability_changed"):
        batch.next_rng_key(1)


@pytest.mark.parametrize("pipeline_owner", ("target", "backbone"))
def test_direct_and_nested_pipeline_parallelism_fail_closed(pipeline_owner):
    model = _Model()
    if pipeline_owner == "target":
        model.pipeline_size = 2
    else:
        model.model.pipeline_size = 2
    with pytest.raises(
        NativeMTPBatchError,
        match="native_mtp_pipeline_parallelism_unsupported",
    ):
        NativeMTPBatch([_row(model, 1)])


def test_truly_empty_recurrent_cache_admits_canonical_zero_padding_for_exact_rows():
    model = _Model()
    rows = [_row(model, uid) for uid in (1, 2, 3)]
    for row in rows:
        row.backbone_cache[0] = ArraysCache(2)

    batch = NativeMTPBatch(rows)

    recurrent = batch.backbone_cache[0]
    assert recurrent.empty()
    assert recurrent.batch_size == 3
    assert recurrent.left_padding.tolist() == [0, 0, 0]
    batch.checkpoint()
    batch.rollback()
    assert recurrent.left_padding.tolist() == [0, 0, 0]


def test_alignment_uses_one_host_scalar_read_per_transaction_boundary():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1), _row(model, 2)])
    assert (batch.alignment_checks, batch.alignment_host_syncs) == (1, 1)

    batch.checkpoint()
    assert (batch.alignment_checks, batch.alignment_host_syncs) == (2, 2)
    _advance_batch(batch, 1)
    sealed = _positions(batch, 1)
    batch.seal_verified(sealed)
    assert (batch.alignment_checks, batch.alignment_host_syncs) == (3, 3)
    batch.commit(sealed)
    assert (batch.alignment_checks, batch.alignment_host_syncs) == (3, 3)
    assert batch.alignment_check_ms >= 0


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("prefix_reused", "native_mtp_prefix_reuse_unsupported"),
        ("has_media", "native_mtp_media_unsupported"),
        ("specprefill_selected", "native_mtp_specprefill_composition_unsupported"),
        ("external_assistant_selected", "native_mtp_external_assistant_conflict"),
    ],
)
def test_unqualified_compositions_fail_before_batch_cache_creation(field, reason):
    model = _Model()
    values = {
        "uid": 1,
        "target_model": model,
        "capability": model.mtp_capability,
        "backbone_cache": [ArraysCache(2), KVCache()],
        "mtp_cache": [KVCache()],
        "position": NativeMTPBatchPosition(0, 0, 0),
        "request_config": _sampling(1),
        field: True,
    }
    with pytest.raises(NativeMTPBatchError, match=f"^{reason}$"):
        NativeMTPBatchRow(**values)


def test_request_rng_streams_do_not_cross_between_uids_or_after_filter_join():
    model = _Model()
    batch = NativeMTPBatch([_row(model, 1, seed=77), _row(model, 2, seed=77)])
    first_one = batch.next_rng_key(1)
    second_one = batch.next_rng_key(1)
    first_two = batch.next_rng_key(2)
    mx.eval(first_one, second_one, first_two)
    assert mx.array_equal(first_one, first_two).item()
    assert not mx.array_equal(first_one, second_one).item()

    batch.filter((2,))
    other = NativeMTPBatch([_row(model, 3, seed=77)])
    joined = NativeMTPBatch.join((batch, other))
    second_two = joined.next_rng_key(2)
    first_three = joined.next_rng_key(3)
    mx.eval(second_two, first_three)
    assert mx.array_equal(second_one, second_two).item()
    assert mx.array_equal(first_one, first_three).item()


def test_join_rejects_cross_model_and_active_checkpoint_without_consuming_sources():
    first = NativeMTPBatch([_row(_Model(), 1)])
    second = NativeMTPBatch([_row(_Model(), 2)])
    with pytest.raises(NativeMTPBatchError, match="native_mtp_batch_join_incompatible"):
        NativeMTPBatch.join((first, second))
    assert not first.closed and not second.closed

    model = _Model()
    left = NativeMTPBatch([_row(model, 3)])
    right = NativeMTPBatch([_row(model, 4)])
    left.checkpoint()
    with pytest.raises(NativeMTPBatchError, match="native_mtp_batch_checkpoint_active"):
        NativeMTPBatch.join((left, right))
    assert not left.closed and not right.closed
