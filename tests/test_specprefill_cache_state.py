# SPDX-License-Identifier: Apache-2.0
"""Pure-Python invariants for request-local SpecPrefill cache state."""

from __future__ import annotations

import threading

import pytest

from vllm_mlx.specprefill_cache import (
    ExactSparseCacheStore,
    SparseCacheIdentity,
    SparseCacheRowState,
    SparseCacheState,
    SparseCacheStateError,
    SparseCacheTransformUnsupported,
    SparsePolicyTuning,
)


def _identity(
    tokens: tuple[int, ...] = (10, 11, 12, 13),
    *,
    selection_fingerprint: str = "a" * 64,
) -> SparseCacheIdentity:
    return SparseCacheIdentity.from_tokens(
        target_id="gemma-4-31b-bf16@sha256:target",
        tokenizer_id="gemma-tokenizer@sha256:tokenizer",
        scorer_id="gemma-4-e2b@sha256:scorer",
        selector_version="hybrid-chunk-v1",
        tuning=SparsePolicyTuning(
            keep_pct=0.7,
            backbone_pct=0.1,
            halo_chunks=1,
            anchor_chunks=1,
            chunk_size=32,
        ),
        tokens=tokens,
        selection_fingerprint=selection_fingerprint,
    )


def _state(*, rows: int = 2) -> SparseCacheState:
    identities = (
        _identity(selection_fingerprint="a" * 64),
        _identity(selection_fingerprint="b" * 64),
    )[:rows]
    return SparseCacheState.from_selection(
        identities,
        selected_logical_positions=((0, 2, 3), (0, 1, 3))[:rows],
        next_logical_positions=(4, 4)[:rows],
    )


def test_sparse_cache_tuning_requires_boundary_anchors():
    with pytest.raises(SparseCacheStateError, match="anchor_chunks must be positive"):
        SparsePolicyTuning(
            keep_pct=0.7,
            backbone_pct=0.1,
            halo_chunks=1,
            anchor_chunks=0,
            chunk_size=32,
        )


def test_identity_includes_artifacts_policy_full_prompt_and_selection():
    base = _identity()
    assert base.full_token_hash == SparseCacheIdentity.hash_tokens((10, 11, 12, 13))
    assert base != _identity(tokens=(10, 11, 12, 14))
    assert base != _identity(selection_fingerprint="b" * 64)
    assert base != SparseCacheIdentity.from_tokens(
        target_id="gemma-4-31b-bf16@sha256:target",
        tokenizer_id="gemma-tokenizer@sha256:tokenizer",
        scorer_id="other-scorer@sha256:scorer",
        selector_version="hybrid-chunk-v1",
        tuning=base.tuning,
        tokens=(10, 11, 12, 13),
        selection_fingerprint="a" * 64,
    )


def test_state_tracks_per_row_logical_positions_separately_from_physical_lengths():
    state = _state()
    assert state.physical_valid_lengths == (3, 3)
    assert state.logical_positions == ((0, 2, 3), (0, 1, 3))
    assert state.next_logical_positions == (4, 4)

    advanced = state.append_decode((2, 1))
    assert advanced.physical_valid_lengths == (5, 4)
    assert advanced.logical_positions == ((0, 2, 3, 4, 5), (0, 1, 3, 4))
    assert advanced.next_logical_positions == (6, 5)
    assert state.physical_valid_lengths == (3, 3)


def test_filter_clone_and_extend_are_immutable_and_atomic():
    state = _state()
    clone = state.clone()
    assert clone == state
    assert clone is not state

    filtered = state.filter((1,))
    assert filtered.logical_positions == ((0, 1, 3),)
    assert state.row_count == 2

    extended = filtered.extend(filtered)
    assert extended.row_count == 2
    assert extended.logical_positions == ((0, 1, 3), (0, 1, 3))

    with pytest.raises(SparseCacheStateError, match="unique"):
        state.filter((0, 0))
    with pytest.raises(SparseCacheStateError, match="outside"):
        state.filter((2,))
    incompatible = SparseCacheIdentity.from_tokens(
        target_id="other-target@sha256:target",
        tokenizer_id="gemma-tokenizer@sha256:tokenizer",
        scorer_id="gemma-4-e2b@sha256:scorer",
        selector_version="hybrid-chunk-v1",
        tuning=_identity().tuning,
        tokens=(1, 2, 3),
        selection_fingerprint="a" * 64,
    )
    with pytest.raises(SparseCacheStateError, match="execution config"):
        state.extend(SparseCacheState.from_selection(incompatible, ((0, 1),), (2,)))
    incompatible_row = SparseCacheState.from_selection(
        incompatible, ((0, 1),), (2,)
    ).rows[0]
    with pytest.raises(SparseCacheStateError, match="execution config"):
        SparseCacheState(rows=(state.rows[0], incompatible_row))
    assert state.row_count == 2


def test_compatible_mixed_prompt_rows_extend_and_filter_without_identity_loss():
    left = SparseCacheState.from_selection(
        _identity(tokens=(10, 11, 12, 13), selection_fingerprint="a" * 64),
        ((0, 2, 3),),
        (4,),
    )
    right = SparseCacheState.from_selection(
        _identity(tokens=(20, 21, 22, 23, 24, 25), selection_fingerprint="b" * 64),
        ((0, 1, 4, 5),),
        (6,),
    )
    batch = left.extend(right)
    assert batch.row_count == 2
    assert batch.identities == (left.rows[0].identity, right.rows[0].identity)
    assert batch.physical_valid_lengths == (3, 4)

    filtered = batch.filter((1,))
    assert filtered.identities == (right.rows[0].identity,)
    assert filtered.logical_positions == ((0, 1, 4, 5),)


def test_rollback_requires_contiguous_decode_suffix_and_is_atomic():
    advanced = _state().append_decode((2, 1))
    restored = advanced.rollback((2, 1))
    assert restored == _state()
    with pytest.raises(SparseCacheStateError, match="one value per row"):
        advanced.rollback((1,))
    assert advanced.physical_valid_lengths == (5, 4)

    with pytest.raises(SparseCacheStateError, match="contiguous decode suffix"):
        SparseCacheRowState(
            identity=_identity(),
            logical_positions=(0, 2, 3),
            physical_valid_length=3,
            next_logical_position=5,
            prefill_physical_length=2,
        )

    sparse_only = _state(rows=1)
    with pytest.raises(SparseCacheStateError, match="prefill boundary"):
        sparse_only.rollback(1)
    assert sparse_only.logical_positions == ((0, 2, 3),)

    corrupt = SparseCacheState(
        rows=(
            SparseCacheRowState(
                identity=_identity(),
                logical_positions=(0, 2, 3),
                physical_valid_length=3,
                next_logical_position=4,
                prefill_physical_length=3,
            ),
        ),
    )
    assert corrupt.rollback(0) == corrupt


def test_trim_requires_explicit_cursor_when_sparse_suffix_makes_it_ambiguous():
    sparse_only = _state(rows=1)
    with pytest.raises(SparseCacheStateError, match="prefill boundary"):
        sparse_only.trim(1)

    decoded = sparse_only.append_decode(2)
    trimmed = decoded.trim(1)
    assert trimmed.logical_positions == ((0, 2, 3, 4),)
    assert trimmed.physical_valid_lengths == (4,)
    assert trimmed.next_logical_positions == (5,)


def test_state_rejects_partial_or_mismatched_row_metadata():
    with pytest.raises(SparseCacheStateError, match="does not match"):
        SparseCacheRowState(
            identity=_identity(),
            logical_positions=(0, 1),
            physical_valid_length=1,
            next_logical_position=2,
            prefill_physical_length=1,
        )
    with pytest.raises(SparseCacheStateError, match="one value per row"):
        SparseCacheState.from_selection(_identity(), ((0, 1),), (2, 3))
    with pytest.raises(SparseCacheStateError, match="strictly increasing"):
        SparseCacheState.from_selection(_identity(), ((0, 2, 2),), (3,))


def test_exact_store_rejects_lcp_supersequence_and_different_selection_hits():
    store: ExactSparseCacheStore[str] = ExactSparseCacheStore()
    state = _state(rows=1)
    store.store(state, "opaque-cache-payload", clone_payload=lambda payload: payload)
    hit = store.lookup(state.rows[0].identity)
    assert hit is not None
    assert hit.payload == "opaque-cache-payload"
    assert hit.state == state
    assert hit.state is not state
    assert store.lookup(_identity(tokens=(10, 11, 12))) is None
    assert store.lookup(_identity(tokens=(10, 11, 12, 13, 14))) is None
    assert store.lookup(_identity(selection_fingerprint="b" * 64)) is None
    with pytest.raises(SparseCacheStateError, match="exactly one request row"):
        store.store(_state(rows=2), "must-not-be-shared", clone_payload=lambda p: p)

    mutable_store: ExactSparseCacheStore[list[str]] = ExactSparseCacheStore()
    mutable_store.store(state, ["request-local"], clone_payload=list)
    first_hit = mutable_store.lookup(state.rows[0].identity)
    assert first_hit is not None
    first_hit.payload.append("mutated")
    second_hit = mutable_store.lookup(state.rows[0].identity)
    assert second_hit is not None
    assert second_hit.payload == ["request-local"]


def test_exact_store_clones_payload_outside_its_lock():
    store: ExactSparseCacheStore[list[str]] = ExactSparseCacheStore()
    state = _state(rows=1)
    clone_completed = threading.Event()

    def clone_payload(payload: list[str]) -> list[str]:
        worker = threading.Thread(
            target=lambda: (
                store.discard(state.rows[0].identity),
                clone_completed.set(),
            )
        )
        worker.start()
        try:
            assert clone_completed.wait(
                timeout=1
            ), "payload clone ran while store lock was held"
        finally:
            worker.join(timeout=1)
        return list(payload)

    store.store(state, ["payload"], clone_payload=clone_payload)
    assert store.lookup(state.rows[0].identity) is not None


def test_quantized_and_ssd_transforms_fail_closed_until_atomic_serializers_exist():
    state = _state(rows=1)
    for method in (state.for_quantized_storage, state.for_ssd_storage):
        with pytest.raises(SparseCacheTransformUnsupported, match="not supported"):
            method()
