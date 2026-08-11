# SPDX-License-Identifier: Apache-2.0
"""Pure-Python semantic contracts for the sparse selector."""

import pytest

from vllm_mlx.specprefill_selection import (
    RotatingTailRequirement,
    SelectionPlan,
    SelectionPolicy,
    SelectionProvenance,
    build_selection_plan_from_chunk_scores,
)


def _policy(**overrides) -> SelectionPolicy:
    values = {
        "keep_pct": 0.25,
        "backbone_pct": 0.0,
        "halo_chunks": 0,
        "anchor_chunks": 0,
        "chunk_size": 4,
    }
    values.update(overrides)
    return SelectionPolicy(**values)


def _plan(*, scores=(0.1, 0.2, 0.3, 0.4, 0.5), **kwargs):
    return build_selection_plan_from_chunk_scores(
        prompt_length=20,
        chunk_scores=scores,
        policy=_policy(**kwargs.pop("policy", {})),
        **kwargs,
    )


def test_caller_supplied_control_positions_are_mandatory_and_provenanced():
    plan = _plan(control_token_indices=(14, 5, 5))

    assert plan.control_anchor_indices == (5, 14)
    assert plan.control_anchor_chunks == (1, 3)
    assert {1, 3}.issubset(plan.selected_chunks)
    assert {5, 14}.issubset(plan.selected_indices)


def test_control_positions_are_prompt_local_and_never_tokenizer_guessed():
    with pytest.raises(ValueError, match="prompt-local"):
        _plan(control_token_indices=(20,))
    with pytest.raises(ValueError, match="prompt-local"):
        _plan(control_token_indices=(True,))


def test_rotating_tail_is_mandatory_before_plan_fingerprint_and_reuse():
    tail_five = _plan(rotating_tail_requirement=RotatingTailRequirement(5))
    tail_six = _plan(rotating_tail_requirement=RotatingTailRequirement(6))

    # Both tail sizes touch chunk 3 at this granularity, but their cache
    # semantics differ and therefore so must their exact-cache fingerprint.
    assert tail_five.rotating_tail_chunks == (3, 4)
    assert tail_six.rotating_tail_chunks == (3, 4)
    assert {3, 4}.issubset(tail_five.selected_chunks)
    assert tail_five.fingerprint != tail_six.fingerprint


def test_policy_drift_changes_fingerprint_even_when_final_chunks_match():
    all_chunks = _plan(policy={"keep_pct": 1.0, "halo_chunks": 0})
    different_halo = _plan(policy={"keep_pct": 1.0, "halo_chunks": 4})
    different_backbone = _plan(policy={"keep_pct": 1.0, "backbone_pct": 1.0})

    assert all_chunks.selected_chunks == different_halo.selected_chunks
    assert all_chunks.selected_chunks == different_backbone.selected_chunks
    assert all_chunks.fingerprint != different_halo.fingerprint
    assert all_chunks.fingerprint != different_backbone.fingerprint


def test_halo_boundary_collision_and_stratified_backbone_are_provenanced():
    plan = _plan(
        scores=(0.99, 0.2, 0.3, 0.4, 0.5),
        control_token_indices=(4,),
        policy={"keep_pct": 0.6, "halo_chunks": 2, "backbone_pct": 0.4},
    )

    # The high-scoring first chunk has no negative halo. Chunk 1 is both a
    # control anchor and a halo candidate; each source remains observable.
    assert plan.backbone_chunks == (1, 3)
    assert plan.control_anchor_chunks == (1,)
    assert plan.importance_chunks == (0,)
    assert plan.halo_chunks == (2,)
    assert {0, 1, 2, 3}.issubset(plan.selected_chunks)


def test_tail_requirement_rejects_partial_or_unprovenanced_plans():
    requirement = RotatingTailRequirement(5)
    with pytest.raises(ValueError, match="rotating_tail_chunks"):
        SelectionPlan(
            prompt_length=20,
            policy=_policy(),
            selected_chunks=(0,),
            selected_indices=(0, 1, 2, 3),
            rotating_tail_requirement=requirement,
        )


def test_rotating_tail_requires_an_integer_window_size():
    with pytest.raises(ValueError, match="positive integer"):
        RotatingTailRequirement(5.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("keep_pct", True),
        ("keep_pct", float("nan")),
        ("keep_pct", "0.25"),
        ("backbone_pct", True),
        ("backbone_pct", float("inf")),
        ("backbone_pct", "0.0"),
        ("halo_chunks", True),
        ("halo_chunks", 1.0),
        ("anchor_chunks", True),
        ("anchor_chunks", 1.0),
        ("chunk_size", True),
        ("chunk_size", 4.0),
    ),
)
def test_policy_rejects_boolean_and_coerced_numeric_values(field, value):
    values = {
        "keep_pct": 0.25,
        "backbone_pct": 0.0,
        "halo_chunks": 0,
        "anchor_chunks": 0,
        "chunk_size": 4,
    }
    values[field] = value
    with pytest.raises(ValueError):
        SelectionPolicy(**values)


def test_builder_requires_an_explicit_selection_policy():
    with pytest.raises(TypeError, match="SelectionPolicy"):
        build_selection_plan_from_chunk_scores(
            prompt_length=4,
            chunk_scores=(1.0,),
            policy=None,
        )


def test_budget_rejected_halo_is_not_recorded_as_retained_provenance():
    plan = _plan(
        scores=(0.99, 0.2, 0.3, 0.4, 0.5),
        control_token_indices=(4,),
        policy={"halo_chunks": 2, "backbone_pct": 0.4},
    )

    assert plan.selected_chunks == (0, 1, 3)
    assert plan.importance_chunks == (0,)
    assert plan.halo_chunks == ()


def test_direct_plan_construction_cannot_omit_policy_mandatory_chunks():
    policy = _policy(anchor_chunks=1)
    with pytest.raises(ValueError, match="retained in selected_chunks"):
        SelectionPlan(
            prompt_length=20,
            policy=policy,
            selected_chunks=(0,),
            selected_indices=(0, 1, 2, 3),
            provenance=SelectionProvenance(anchor_chunks=(0, 4)),
        )


def test_direct_plan_construction_cannot_forge_control_anchor_mapping():
    with pytest.raises(ValueError, match="control_anchor_chunks"):
        SelectionPlan(
            prompt_length=20,
            policy=_policy(),
            selected_chunks=(1,),
            selected_indices=(4, 5, 6, 7),
            provenance=SelectionProvenance(control_anchor_indices=(4,)),
        )


def test_direct_plan_construction_cannot_forge_a_tail_without_requirement():
    with pytest.raises(ValueError, match="require a rotating_tail_requirement"):
        SelectionPlan(
            prompt_length=20,
            policy=_policy(),
            selected_chunks=(3,),
            selected_indices=(12, 13, 14, 15),
            provenance=SelectionProvenance(rotating_tail_chunks=(3,)),
        )


def test_unicode_selector_version_is_fingerprintable():
    ascii_plan = _plan(policy={"selector_version": "selector-v2"})
    unicode_plan = _plan(policy={"selector_version": "selector-β"})

    assert len(unicode_plan.fingerprint) == 64
    assert ascii_plan.fingerprint != unicode_plan.fingerprint
