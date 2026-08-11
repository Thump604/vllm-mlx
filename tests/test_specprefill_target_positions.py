# SPDX-License-Identifier: Apache-2.0
"""Host-only position contracts for sparse target execution."""

import pytest

from vllm_mlx.specprefill_cache import (
    SparseCacheIdentity,
    SparseCacheState,
    SparsePolicyTuning,
)
from vllm_mlx.specprefill_positions import (
    GEMMA4_A4B_TARGET,
    GEMMA4_DENSE_TARGET,
    QWEN_DENSE_TARGET,
    QWEN35_TEXT_HYBRID_TARGET,
    QWEN35_VLM_HYBRID_TARGET,
    QWEN35_VLM_MOE_TARGET,
    TargetPositionError,
    TargetPositionFamily,
    decode_plan,
    gemma_previous_kv_cache_map,
    mtp_decode_plan,
    resolve_target_position_adapter,
    sparse_prefill_plan,
    target_position_adapter,
    verify_plan,
)


class _Config:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        for name, value in kwargs.items():
            setattr(self, name, value)


class _Model:
    def __init__(self, config=None, *, args=None, language_model=None, vision=False):
        self.config = config
        self.args = args
        self.language_model = language_model
        if vision:
            self.vision_tower = object()


def test_known_target_resolver_is_config_and_wrapper_bounded():
    assert (
        resolve_target_position_adapter(_Model(_Config("qwen3"))) is QWEN_DENSE_TARGET
    )
    assert (
        resolve_target_position_adapter(_Model(_Config("qwen3_5_moe")))
        is QWEN35_TEXT_HYBRID_TARGET
    )
    assert (
        resolve_target_position_adapter(
            _Model(
                _Config("qwen3_5"),
                language_model=_Model(_Config("qwen3_5")),
                vision=True,
            )
        )
        is QWEN35_VLM_HYBRID_TARGET
    )
    # This matches clean mlx-lm qwen3_5.Model: ``args`` plus a
    # ``language_model`` text wrapper, with no vision tower.
    assert (
        resolve_target_position_adapter(
            _Model(
                args=_Config("qwen3_5"),
                language_model=_Model(_Config("qwen3_5")),
            )
        )
        is QWEN35_TEXT_HYBRID_TARGET
    )
    assert (
        resolve_target_position_adapter(_Model(_Config("gemma4", num_experts=16)))
        is GEMMA4_A4B_TARGET
    )
    with pytest.raises(TargetPositionError, match="unsupported target model layout"):
        resolve_target_position_adapter(_Model(_Config("unknown")))


def _identity(tokens: tuple[int, ...], fingerprint: str = "a" * 64):
    return SparseCacheIdentity.from_tokens(
        target_id="target@sha256:target",
        tokenizer_id="tokenizer@sha256:tokenizer",
        scorer_id="scorer@sha256:scorer",
        selector_version="hybrid-chunk-v1",
        tuning=SparsePolicyTuning(
            keep_pct=0.7,
            backbone_pct=0.1,
            halo_chunks=1,
            anchor_chunks=1,
            chunk_size=32,
        ),
        tokens=tokens,
        selection_fingerprint=fingerprint,
    )


def _state(
    selected: tuple[tuple[int, ...], ...] = ((0, 2, 4, 5),),
    cursors: tuple[int, ...] = (6,),
) -> SparseCacheState:
    identities = tuple(
        _identity(tuple(range(cursor)), chr(ord("a") + index) * 64)
        for index, cursor in enumerate(cursors)
    )
    return SparseCacheState.from_selection(identities, selected, cursors)


def test_adapter_matrix_is_explicit_about_position_transport_and_architecture():
    assert QWEN_DENSE_TARGET.supports_noncontiguous_prefill is False
    assert QWEN35_TEXT_HYBRID_TARGET.supports_noncontiguous_prefill is False
    assert QWEN35_TEXT_HYBRID_TARGET.supports_partial_rope
    assert QWEN35_VLM_HYBRID_TARGET.position_id_planes == 3
    assert QWEN35_VLM_HYBRID_TARGET.supports_heterogeneous_batch_rows
    assert QWEN35_VLM_MOE_TARGET.uses_q_norm
    assert GEMMA4_DENSE_TARGET.supports_shared_kv
    assert GEMMA4_DENSE_TARGET.supports_partial_rope
    assert GEMMA4_A4B_TARGET.supports_shared_kv
    assert target_position_adapter(TargetPositionFamily.GEMMA4_A4B) is GEMMA4_A4B_TARGET
    with pytest.raises(TargetPositionError, match="unsupported target position family"):
        target_position_adapter("attribute_guessing_is_not_an_adapter")


def test_keep_ratio_one_plan_is_dense_position_equivalent_for_all_families():
    dense = _state(selected=((0, 1, 2, 3),), cursors=(4,))
    for adapter in (
        QWEN_DENSE_TARGET,
        QWEN35_TEXT_HYBRID_TARGET,
        QWEN35_VLM_HYBRID_TARGET,
        QWEN35_VLM_MOE_TARGET,
        GEMMA4_DENSE_TARGET,
        GEMMA4_A4B_TARGET,
    ):
        prefill = sparse_prefill_plan(adapter, dense)
        assert prefill.is_dense_equivalent
        prefill.require_executable()

        decode = decode_plan(adapter, dense)
        assert decode.logical_positions == ((4,),)
        assert decode.physical_cache_lengths == (4,)
        decode.require_executable()


def test_sparse_qwen_vlm_plan_uses_exact_three_plane_batch_shape():
    state = _state(
        selected=((0, 2, 4, 5), (1, 3, 4, 7)),
        cursors=(6, 8),
    )
    plan = sparse_prefill_plan(QWEN35_VLM_HYBRID_TARGET, state)

    plan.require_executable()
    assert plan.qwen35_mrope_position_ids() == (
        ((0, 2, 4, 5), (1, 3, 4, 7)),
        ((0, 2, 4, 5), (1, 3, 4, 7)),
        ((0, 2, 4, 5), (1, 3, 4, 7)),
    )


def test_sparse_qwen_vlm_moe_has_same_explicit_mrope_contract():
    plan = sparse_prefill_plan(QWEN35_VLM_MOE_TARGET, _state())
    plan.require_executable()
    assert plan.qwen35_mrope_position_ids() == (
        ((0, 2, 4, 5),),
        ((0, 2, 4, 5),),
        ((0, 2, 4, 5),),
    )


def test_sparse_prefill_fails_closed_for_cache_offset_and_current_gemma_paths():
    sparse_state = _state()
    for adapter in (QWEN_DENSE_TARGET, QWEN35_TEXT_HYBRID_TARGET, GEMMA4_DENSE_TARGET):
        with pytest.raises(
            TargetPositionError, match="no request-local|cannot execute"
        ):
            sparse_prefill_plan(adapter, sparse_state).require_executable()

    # Gemma's owner attention replaces the passed offset with cache.offset;
    # shared-KV forwarding therefore does not make sparse decode representable.
    with pytest.raises(TargetPositionError, match="cannot execute"):
        decode_plan(GEMMA4_DENSE_TARGET, sparse_state).gemma4_offsets()


def test_heterogeneous_rows_are_planned_but_only_verified_transport_executes_them():
    state = _state(
        selected=((0, 2, 4, 5), (1, 3, 4, 7)),
        cursors=(6, 8),
    )
    qwen = sparse_prefill_plan(QWEN35_VLM_HYBRID_TARGET, state)
    qwen.require_executable()

    gemma = sparse_prefill_plan(GEMMA4_DENSE_TARGET, state)
    with pytest.raises(TargetPositionError, match="non-contiguous"):
        gemma.require_executable()

    # Even normal Gemma decode cannot join rows whose logical cursors differ
    # until a vector-offset transport is proven in the underlying attention.
    dense_rows = _state(selected=((0, 1), (0, 1)), cursors=(2, 3))
    with pytest.raises(TargetPositionError, match="heterogeneous-row"):
        decode_plan(GEMMA4_DENSE_TARGET, dense_rows).require_executable()


def test_verify_rollback_contract_keeps_logical_cursor_independent_of_physical_kv():
    state = _state()
    plan = verify_plan(QWEN35_VLM_HYBRID_TARGET, state, 2)
    assert plan.logical_positions == ((6, 7),)
    assert plan.physical_cache_lengths == (4,)
    plan.require_executable()

    advanced = state.append_decode(2)
    assert advanced.logical_positions == ((0, 2, 4, 5, 6, 7),)
    assert advanced.physical_valid_lengths == (6,)
    assert advanced.rollback(2) == state


def test_mtp_pairs_logical_positions_but_not_target_assistant_physical_lengths():
    target = _state(selected=((0, 2, 4, 5),), cursors=(6,))
    assistant = _state(selected=((0, 1, 2, 3, 4, 5),), cursors=(6,))
    mtp = mtp_decode_plan(
        QWEN35_VLM_HYBRID_TARGET,
        target,
        QWEN35_VLM_HYBRID_TARGET,
        assistant,
    )
    assert mtp.target.logical_positions == ((6,),)
    assert mtp.assistant.logical_positions == ((6,),)
    assert mtp.physical_cache_lengths == ((4,), (6,))

    different_cursor = _state(selected=((0, 2, 4, 5),), cursors=(7,))
    with pytest.raises(TargetPositionError, match="logical positions must agree"):
        mtp_decode_plan(
            QWEN35_VLM_HYBRID_TARGET,
            target,
            QWEN35_VLM_HYBRID_TARGET,
            different_cursor,
        )


def test_gemma_shared_kv_owner_mapping_is_explicit_and_validated():
    assert gemma_previous_kv_cache_map((0, 1, 2, 1, 2)) == {
        0: 0,
        1: 1,
        2: 2,
        3: 1,
        4: 2,
    }
    with pytest.raises(TargetPositionError, match="earlier layer"):
        gemma_previous_kv_cache_map((0, 2))
