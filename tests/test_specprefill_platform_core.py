# SPDX-License-Identifier: Apache-2.0
"""Focused non-model contracts for the SpecPrefill platform core."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.specprefill import (
    ARCHITECTURE_ADAPTERS,
    SPECPREFILL_SELECTOR_VERSION,
    SpecPrefillCoverage,
    SpecPrefillPolicy,
    _AttentionCapture,
    _build_layer_to_cache_map,
    _gemma4_extract_queries,
    _qwen_extract_queries,
    build_selection_plan,
    resolve_specprefill_adapter,
    resolve_specprefill_decision,
    select_chunks,
)


def test_production_policy_requires_declared_selective_coverage():
    allowed = resolve_specprefill_decision(
        "auto", "selective", production=True, threshold_met=True, admission_allowed=True
    )
    assert allowed.requested_policy is SpecPrefillPolicy.AUTO
    assert allowed.effective_policy is SpecPrefillPolicy.SPARSE

    for coverage in (SpecPrefillCoverage.EXHAUSTIVE, SpecPrefillCoverage.UNKNOWN):
        denied = resolve_specprefill_decision("auto", coverage, production=True)
        assert denied.effective_policy is SpecPrefillPolicy.DENSE
        assert denied.fallback_reason == "coverage_not_selective"
        diagnostic_auto = resolve_specprefill_decision(
            "auto", coverage, production=False
        )
        assert diagnostic_auto.effective_policy is SpecPrefillPolicy.DENSE


def test_explicit_sparse_is_diagnostic_only_in_production():
    production = resolve_specprefill_decision("sparse", "selective", production=True)
    assert production.effective_policy is SpecPrefillPolicy.DENSE
    assert production.fallback_reason == "sparse_forcing_diagnostic_only"

    diagnostic = resolve_specprefill_decision("sparse", "exhaustive", production=False)
    assert diagnostic.effective_policy is SpecPrefillPolicy.SPARSE


def test_non_text_policy_falls_back_before_sparse_admission():
    decision = resolve_specprefill_decision(
        "auto", "selective", production=False, text_only=False
    )
    assert decision.effective_policy is SpecPrefillPolicy.DENSE
    assert decision.fallback_reason == "media_request"


def test_adapter_registry_is_explicit_for_installed_qwen_and_gemma_families():
    assert resolve_specprefill_adapter("qwen3").name == "qwen-dense"
    assert resolve_specprefill_adapter("qwen3_5").name == "qwen3.5-3.6-hybrid-moe"
    assert (
        resolve_specprefill_adapter("qwen3_5_moe_text").name == "qwen3.5-3.6-hybrid-moe"
    )
    assert resolve_specprefill_adapter("gemma4").name == "gemma4-shared-kv"
    assert "gemma4" in ARCHITECTURE_ADAPTERS
    with pytest.raises(ValueError, match="Unsupported SpecPrefill model_type"):
        resolve_specprefill_adapter("unclassified-model")


def test_gemma_shared_kv_mapping_compacts_to_unique_owner_caches():
    # E2B has 35 decoder layers but only 15 KV owners; the prompt cache is
    # sized for those 15 owners rather than all decoder layers.
    previous_kvs = list(range(15)) + [index % 15 for index in range(20)]
    model = SimpleNamespace(
        layers=[object() for _ in previous_kvs],
        model=SimpleNamespace(previous_kvs=previous_kvs),
    )
    mapping = _build_layer_to_cache_map(model)
    assert len(mapping) == 35
    assert set(mapping.values()) == set(range(15))
    for layer_idx, owner_idx in enumerate(previous_kvs):
        assert mapping[layer_idx] == owner_idx

    # Owners need not be adjacent decoder layers; cache slots are compact in
    # first-owner order rather than the decoder-layer number.
    sparse_owners = [0, 0, 2, 0, 2]
    sparse_model = SimpleNamespace(
        layers=[object() for _ in sparse_owners],
        previous_kvs=sparse_owners,
    )
    assert _build_layer_to_cache_map(sparse_model) == {0: 0, 1: 0, 2: 1, 3: 0, 4: 1}


def test_capture_preserves_shared_kv_offset_arguments():
    recorded = []

    class Attention:
        def __call__(self, x, mask=None, cache=None, **kwargs):
            recorded.append((mask, cache, kwargs))
            return x

    captured = []

    def extractor(attn, x, cache=None, **kwargs):
        captured.append((cache, kwargs))
        return x

    wrapper = _AttentionCapture(Attention(), 0, [[]], extractor)
    offset = mx.array(17)
    shared_kv = (mx.array([1]), mx.array([2]))
    wrapper(mx.zeros((1, 1, 1)), shared_kv=shared_kv, offset=offset)
    assert recorded[0][2]["shared_kv"] is shared_kv
    assert int(recorded[0][2]["offset"].item()) == 17
    assert captured[0][1]["shared_kv"] is shared_kv


def test_explicit_qwen_and_gemma_extractors_apply_normalization_and_offset():
    class Norm:
        def __call__(self, x):
            return x * 2

    class Rope:
        def __call__(self, x, offset=0):
            return x + offset

    class Attention:
        n_heads = 2
        q_norm = Norm()
        rope = Rope()

        def q_proj(self, x):
            return x

    x = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8)
    expected = (x.reshape(1, 2, 2, 4) * 2).transpose(0, 2, 1, 3) + 9
    cache = SimpleNamespace(offset=9)
    for extractor in (_qwen_extract_queries, _gemma4_extract_queries):
        actual = extractor(Attention(), x, cache=cache)
        mx.eval(actual)
        assert actual.tolist() == expected.tolist()


def test_hybrid_selector_retains_importance_halo_anchors_and_backbone():
    importance = mx.array(
        [0.1] * 4
        + [0.2] * 4
        + [0.3] * 4
        + [0.99] * 4
        + [0.4] * 4
        + [0.5] * 4
        + [0.6] * 4
        + [0.7] * 4
    )
    plan = build_selection_plan(
        importance,
        keep_pct=0.5,
        chunk_size=4,
        backbone_pct=0.125,
        halo_chunks=1,
        anchor_chunks=1,
    )
    assert plan.selector_version == SPECPREFILL_SELECTOR_VERSION
    assert plan.anchor_chunks == (0, 7)
    assert plan.backbone_chunks == (4,)
    assert 3 in plan.importance_chunks
    assert {2, 3, 4}.issubset(plan.selected_chunks)
    assert plan.selected_indices == tuple(sorted(plan.selected_indices))
    assert select_chunks(
        importance,
        keep_pct=0.5,
        chunk_size=4,
        backbone_pct=0.125,
        halo_chunks=1,
        anchor_chunks=1,
    ).tolist() == list(plan.selected_indices)


def test_selector_is_stable_and_validates_invalid_importance():
    importance = mx.array([1.0] * 16)
    first = build_selection_plan(importance, keep_pct=0.5, chunk_size=4)
    second = build_selection_plan(importance, keep_pct=0.5, chunk_size=4)
    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.selected_chunks[0] == 0

    different_policy = build_selection_plan(importance, keep_pct=0.75, chunk_size=4)
    assert first.fingerprint != different_policy.fingerprint

    with pytest.raises(ValueError, match="finite"):
        build_selection_plan(mx.array([0.0, float("nan")]), chunk_size=1)
    with pytest.raises(ValueError, match="keep_pct"):
        build_selection_plan(importance, keep_pct=0.0)
