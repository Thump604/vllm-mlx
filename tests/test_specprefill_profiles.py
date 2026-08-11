# SPDX-License-Identifier: Apache-2.0
"""MLX-free contracts for calibrated SpecPrefill profile resolution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_mlx.specprefill_profiles import (
    EMPTY_SPECPREFILL_PROFILE_REGISTRY,
    SpecPrefillCalibration,
    SpecPrefillCell,
    SpecPrefillEngine,
    SpecPrefillProfile,
    SpecPrefillProfileKey,
    SpecPrefillProfileRegistry,
    SpecPrefillProfileTier,
    SpecPrefillQualificationEvidence,
    SpecPrefillTuning,
)


def _hash(letter: str) -> str:
    return letter * 64


def _key(
    *,
    engine: SpecPrefillEngine = SpecPrefillEngine.SIMPLE,
    cell: SpecPrefillCell = SpecPrefillCell.SPARSE_ONLY,
) -> SpecPrefillProfileKey:
    return SpecPrefillProfileKey(
        target_artifact_id="gemma-4-31b-bf16",
        target_artifact_hash=_hash("a"),
        tokenizer_artifact_hash=_hash("b"),
        scorer_artifact_id="gemma-4-e2b-bf16",
        scorer_artifact_hash=_hash("c"),
        adapter_id="gemma4-shared-kv",
        engine=engine,
        cell=cell,
    )


def _calibration(*, keep_pct: float = 0.7) -> SpecPrefillCalibration:
    return SpecPrefillCalibration(
        selector_version="hybrid-chunk-v1",
        tuning=SpecPrefillTuning(
            keep_pct=keep_pct,
            backbone_pct=0.1,
            halo_chunks=1,
            anchor_chunks=1,
            chunk_size=32,
        ),
        crossover_tokens=8192,
        max_context_tokens=131072,
        residency_limit_bytes=80 * 1024**3,
        min_ttft_improvement_pct=20.0,
        max_total_latency_regression_pct=5.0,
        max_decode_throughput_regression_pct=5.0,
        required_context_tokens=(8192, 16384, 32768, 65536, 98304, 130048),
        required_concurrency_levels=(1,),
        max_p95_inter_token_latency_regression_pct=5.0,
        min_prefill_heavy_throughput_improvement_pct=10.0,
    )


def test_profile_tuning_requires_boundary_anchors():
    with pytest.raises(ValueError, match="anchor_chunks must be positive"):
        SpecPrefillTuning(
            keep_pct=0.7,
            backbone_pct=0.1,
            halo_chunks=1,
            anchor_chunks=0,
            chunk_size=32,
        )


def _evidence(
    *,
    key: SpecPrefillProfileKey | None = None,
    selector_version: str = "hybrid-chunk-v1",
    **overrides: object,
) -> SpecPrefillQualificationEvidence:
    values: dict[str, object] = {
        "report_id": "run/specprefill/gemma4-31b-simple-sparse/summary.json",
        "report_sha256": _hash("9"),
        "key": key or _key(),
        "selector_version": selector_version,
        "tested_context_tokens": (8192, 16384, 32768, 65536, 98304, 130048),
        "tested_concurrency_levels": (1,),
        "deterministic_baseline_successes": 12,
        "preserved_baseline_successes": 12,
        "fabricated_or_source_corruption_count": 0,
        "quality_noninferiority_ci_lower_points": -1.5,
        "median_ttft_improvement_pct": 22.0,
        "median_total_latency_regression_pct": 4.0,
        "decode_throughput_regression_pct": 4.0,
        "oom_count": 0,
        "swap_escalation_count": 0,
        "unbounded_retry_count": 0,
        "peak_resident_bytes": 70 * 1024**3,
        "admission_safety_reserve_pct": 10.0,
        "cb_p95_inter_token_latency_regression_pct": None,
        "cb_aggregate_throughput_regression_pct": None,
        "cb_prefill_heavy_throughput_improvement_pct": None,
        "mtp_evidence_id": None,
        "mtp_evidence_sha256": None,
        "mtp_drafts": 0,
        "mtp_accepted": 0,
    }
    values.update(overrides)
    return SpecPrefillQualificationEvidence(**values)


def _profile(
    *,
    key: SpecPrefillProfileKey | None = None,
    tier: SpecPrefillProfileTier = SpecPrefillProfileTier.PRODUCTION,
    keep_pct: float = 0.7,
    evidence: SpecPrefillQualificationEvidence | None = None,
) -> SpecPrefillProfile:
    return SpecPrefillProfile(
        key=key or _key(),
        tier=tier,
        calibration=_calibration(keep_pct=keep_pct),
        qualification_evidence=evidence,
    )


def test_empty_seed_registry_fails_closed_for_every_artifact():
    decision = EMPTY_SPECPREFILL_PROFILE_REGISTRY.resolve(
        _key(), prompt_tokens=16384, residency_bytes=1
    )
    assert not decision.eligible
    assert not decision.production_certified
    assert decision.fallback_reason == "profile_not_registered"


def test_production_requires_explicit_profile_calibration_and_certification():
    registry = SpecPrefillProfileRegistry((_profile(),))
    decision = registry.resolve(_key(), prompt_tokens=16384, residency_bytes=1)
    assert not decision.eligible
    assert decision.fallback_reason == "uncalibrated_profile"

    certified = _profile(evidence=_evidence())
    decision = SpecPrefillProfileRegistry((certified,)).resolve(
        _key(), prompt_tokens=16384, residency_bytes=1
    )
    assert decision.eligible
    assert decision.production_certified
    assert decision.tuning == certified.calibration.tuning


def test_artifact_adapter_engine_and_cell_are_independent_eligibility_cells():
    simple_sparse = _profile(evidence=_evidence())
    cb_sparse_key = _key(engine=SpecPrefillEngine.CONTINUOUS_BATCHING)
    simple_combined_key = _key(cell=SpecPrefillCell.COMBINED_MTP)
    registry = SpecPrefillProfileRegistry((simple_sparse,))

    assert registry.resolve(_key(), prompt_tokens=16384, residency_bytes=1).eligible
    assert (
        registry.resolve(
            cb_sparse_key, prompt_tokens=16384, residency_bytes=1
        ).fallback_reason
        == "profile_not_registered"
    )
    assert (
        registry.resolve(
            simple_combined_key, prompt_tokens=16384, residency_bytes=1
        ).fallback_reason
        == "profile_not_registered"
    )


def test_calibrated_bounds_and_residency_have_explicit_dense_fallbacks():
    registry = SpecPrefillProfileRegistry((_profile(evidence=_evidence()),))
    assert (
        registry.resolve(_key(), prompt_tokens=8191, residency_bytes=1).fallback_reason
        == "below_calibrated_crossover"
    )
    assert (
        registry.resolve(
            _key(), prompt_tokens=131073, residency_bytes=1
        ).fallback_reason
        == "above_calibrated_max_context"
    )
    assert (
        registry.resolve(
            _key(), prompt_tokens=16384, residency_bytes=80 * 1024**3 + 1
        ).fallback_reason
        == "residency_limit_exceeded"
    )


def test_255k_calibration_requires_both_qwen_extension_ladder_rungs():
    required = (8192, 16384, 32768, 65536, 98304, 130048, 196608, 261120)
    with pytest.raises(ValueError, match="context ladder"):
        replace(
            _calibration(),
            max_context_tokens=261120,
            required_context_tokens=required[:-2] + (261120,),
        )
    with pytest.raises(ValueError, match="context ladder"):
        replace(
            _calibration(),
            max_context_tokens=261120,
            required_context_tokens=required[:-1],
        )
    calibration = replace(
        _calibration(),
        max_context_tokens=261120,
        required_context_tokens=required,
    )
    assert calibration.required_context_tokens == required


def test_diagnostic_tuning_is_explicit_but_cannot_claim_production_certification():
    diagnostic = _profile(
        tier=SpecPrefillProfileTier.DIAGNOSTIC,
        keep_pct=0.35,
    )
    registry = SpecPrefillProfileRegistry((diagnostic,))
    decision = registry.resolve(
        _key(), prompt_tokens=16384, residency_bytes=1, diagnostic=True
    )
    assert decision.eligible
    assert not decision.production_certified
    assert decision.tuning.keep_pct == 0.35
    assert (
        registry.resolve(_key(), prompt_tokens=16384, residency_bytes=1).fallback_reason
        == "profile_not_registered"
    )


def test_no_universal_keep_ratio_and_selector_or_hash_drift_cannot_match():
    first = _profile(keep_pct=0.55, evidence=_evidence())
    second_key = SpecPrefillProfileKey(
        target_artifact_id="qwen3.6-35b-a3b",
        target_artifact_hash=_hash("d"),
        tokenizer_artifact_hash=_hash("e"),
        scorer_artifact_id="qwen3.5-4b",
        scorer_artifact_hash=_hash("f"),
        adapter_id="qwen3.5-3.6-hybrid-moe",
        engine=SpecPrefillEngine.SIMPLE,
        cell=SpecPrefillCell.SPARSE_ONLY,
    )
    second = _profile(
        key=second_key,
        keep_pct=0.8,
        evidence=_evidence(key=second_key),
    )
    registry = SpecPrefillProfileRegistry((first, second))
    assert (
        registry.resolve(_key(), prompt_tokens=16384, residency_bytes=1).tuning.keep_pct
        == 0.55
    )
    assert (
        registry.resolve(
            second_key, prompt_tokens=16384, residency_bytes=1
        ).tuning.keep_pct
        == 0.8
    )

    selector_drift = SpecPrefillProfile(
        key=_key(),
        tier=SpecPrefillProfileTier.PRODUCTION,
        calibration=replace(_calibration(), selector_version="hybrid-chunk-v2"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        SpecPrefillProfileRegistry((first, selector_drift))

    hash_drift = replace(_key(), target_artifact_hash=_hash("0"))
    assert (
        registry.resolve(
            hash_drift, prompt_tokens=16384, residency_bytes=1
        ).fallback_reason
        == "profile_not_registered"
    )


def test_production_certification_is_evidence_derived_and_rejects_drift_or_failures():
    with pytest.raises(TypeError, match="production_certified"):
        SpecPrefillProfile(
            key=_key(),
            tier=SpecPrefillProfileTier.PRODUCTION,
            calibration=_calibration(),
            production_certified=True,
        )

    with pytest.raises(ValueError, match="key"):
        _profile(
            evidence=_evidence(key=replace(_key(), target_artifact_hash=_hash("0")))
        )
    with pytest.raises(ValueError, match="selector"):
        _profile(evidence=_evidence(selector_version="hybrid-chunk-v2"))
    with pytest.raises(ValueError, match="baseline"):
        _profile(evidence=_evidence(preserved_baseline_successes=11))
    with pytest.raises(ValueError, match="source corruption"):
        _profile(evidence=_evidence(fabricated_or_source_corruption_count=1))
    with pytest.raises(ValueError, match="quality"):
        _profile(evidence=_evidence(quality_noninferiority_ci_lower_points=-2.1))
    with pytest.raises(ValueError, match="TTFT"):
        _profile(evidence=_evidence(median_ttft_improvement_pct=19.9))
    with pytest.raises(ValueError, match="OOM"):
        _profile(evidence=_evidence(oom_count=1))
    with pytest.raises(ValueError, match="swap"):
        _profile(evidence=_evidence(swap_escalation_count=1))
    with pytest.raises(ValueError, match="context ladder"):
        _profile(
            evidence=_evidence(
                tested_context_tokens=(8192, 16384, 32768, 98304, 130048)
            )
        )
    with pytest.raises(ValueError, match="reserve"):
        _profile(evidence=_evidence(admission_safety_reserve_pct=9.9))
    with pytest.raises(ValueError, match="resident peak"):
        _profile(evidence=_evidence(peak_resident_bytes=81 * 1024**3))


def test_cb_and_combined_profiles_require_route_specific_evidence():
    cb_key = _key(engine=SpecPrefillEngine.CONTINUOUS_BATCHING)
    with pytest.raises(ValueError, match="CB"):
        _profile(key=cb_key, evidence=_evidence(key=cb_key))

    cb_evidence = _evidence(
        key=cb_key,
        tested_concurrency_levels=(1, 2, 4, 8),
        cb_p95_inter_token_latency_regression_pct=4.0,
        cb_aggregate_throughput_regression_pct=0.0,
        cb_prefill_heavy_throughput_improvement_pct=12.0,
    )
    cb_calibration = replace(_calibration(), required_concurrency_levels=(1, 2, 4, 8))
    cb_profile = SpecPrefillProfile(
        key=cb_key,
        tier=SpecPrefillProfileTier.PRODUCTION,
        calibration=cb_calibration,
        qualification_evidence=cb_evidence,
    )
    assert (
        SpecPrefillProfileRegistry((cb_profile,))
        .resolve(cb_key, prompt_tokens=16384, residency_bytes=1)
        .eligible
    )
    with pytest.raises(ValueError, match="concurrency ladder"):
        SpecPrefillProfile(
            key=cb_key,
            tier=SpecPrefillProfileTier.PRODUCTION,
            calibration=cb_calibration,
            qualification_evidence=replace(
                cb_evidence, tested_concurrency_levels=(1, 2, 8)
            ),
        )

    combined_key = _key(cell=SpecPrefillCell.COMBINED_MTP)
    with pytest.raises(ValueError, match="MTP"):
        _profile(key=combined_key, evidence=_evidence(key=combined_key))
