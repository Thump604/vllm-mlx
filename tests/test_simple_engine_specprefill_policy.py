# SPDX-License-Identifier: Apache-2.0
"""Mocked SimpleEngine policy and telemetry contracts for SpecPrefill."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.simple import SimpleEngine, _request_can_compose_mtp
from vllm_mlx.specprefill_profiles import (
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
from vllm_mlx.specprefill_positions import QWEN_DENSE_TARGET


def _hash(letter: str) -> str:
    return letter * 64


def _profile_key(*, cell: SpecPrefillCell) -> SpecPrefillProfileKey:
    return SpecPrefillProfileKey(
        target_artifact_id="test-model@bf16",
        target_artifact_hash=_hash("a"),
        tokenizer_artifact_hash=_hash("b"),
        scorer_artifact_id="test-scorer@bf16",
        scorer_artifact_hash=_hash("c"),
        adapter_id="qwen_dense",
        engine=SpecPrefillEngine.SIMPLE,
        cell=cell,
    )


def _calibration(
    *, selector_version: str, tuning: SpecPrefillTuning
) -> SpecPrefillCalibration:
    return SpecPrefillCalibration(
        selector_version=selector_version,
        tuning=tuning,
        crossover_tokens=4,
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


def _profile(
    *,
    cell: SpecPrefillCell,
    diagnostic: bool = False,
    selector_version: str,
    tuning: SpecPrefillTuning,
) -> SpecPrefillProfile:
    key = _profile_key(cell=cell)
    calibration = _calibration(selector_version=selector_version, tuning=tuning)
    if diagnostic:
        return SpecPrefillProfile(
            key=key,
            tier=SpecPrefillProfileTier.DIAGNOSTIC,
            calibration=calibration,
        )
    return SpecPrefillProfile(
        key=key,
        tier=SpecPrefillProfileTier.PRODUCTION,
        calibration=calibration,
        qualification_evidence=SpecPrefillQualificationEvidence(
            report_id="run/specprefill/simple-policy/summary.json",
            report_sha256=_hash("d"),
            key=key,
            selector_version=selector_version,
            tested_context_tokens=(8192, 16384, 32768, 65536, 98304, 130048),
            tested_concurrency_levels=(1,),
            deterministic_baseline_successes=4,
            preserved_baseline_successes=4,
            fabricated_or_source_corruption_count=0,
            quality_noninferiority_ci_lower_points=-1.0,
            median_ttft_improvement_pct=25.0,
            median_total_latency_regression_pct=2.0,
            decode_throughput_regression_pct=2.0,
            oom_count=0,
            swap_escalation_count=0,
            unbounded_retry_count=0,
            peak_resident_bytes=70 * 1024**3,
            admission_safety_reserve_pct=10.0,
            cb_p95_inter_token_latency_regression_pct=None,
            cb_aggregate_throughput_regression_pct=None,
            cb_prefill_heavy_throughput_improvement_pct=None,
            mtp_evidence_id=(
                "run/specprefill/simple-combined-mtp.json"
                if cell is SpecPrefillCell.COMBINED_MTP
                else None
            ),
            mtp_evidence_sha256=(
                _hash("e") if cell is SpecPrefillCell.COMBINED_MTP else None
            ),
            mtp_drafts=(10 if cell is SpecPrefillCell.COMBINED_MTP else 0),
            mtp_accepted=(8 if cell is SpecPrefillCell.COMBINED_MTP else 0),
        ),
    )


def _engine(
    *,
    diagnostic: bool = False,
    mtp: bool = False,
    registered: bool = False,
    estimated_residency: bool = True,
) -> SimpleEngine:
    sparse_profile = _profile(
        cell=SpecPrefillCell.SPARSE_ONLY,
        diagnostic=diagnostic,
        selector_version="test-sparse-selector-v1",
        tuning=SpecPrefillTuning(0.55, 0.1, 1, 1, 32),
    )
    combined_profile = _profile(
        cell=SpecPrefillCell.COMBINED_MTP,
        diagnostic=diagnostic,
        selector_version="test-combined-selector-v1",
        tuning=SpecPrefillTuning(0.65, 0.2, 2, 3, 64),
    )
    engine = SimpleEngine(
        "test-model",
        mtp=mtp,
        specprefill_threshold=4,
        specprefill_diagnostic_mode=diagnostic,
        specprefill_profile_registry=(
            SpecPrefillProfileRegistry((sparse_profile, combined_profile))
            if registered
            else None
        ),
        specprefill_sparse_profile_key=(sparse_profile.key if registered else None),
        specprefill_combined_profile_key=(combined_profile.key if registered else None),
        specprefill_estimated_residency_bytes=(
            70 * 1024**3 if registered and estimated_residency else None
        ),
    )
    engine._draft_model = object()
    return engine


def test_production_default_and_legacy_intent_are_dense_without_selective_coverage():
    engine = _engine()

    omitted = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy=None,
        coverage=None,
        has_media=False,
        total_tokens=8,
    )
    assert omitted.decision.effective_policy.value == "dense"
    assert omitted.decision.fallback_reason == "coverage_not_selective"

    legacy_disabled = engine._resolve_specprefill_telemetry(
        legacy=False,
        policy=None,
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert legacy_disabled.decision.requested_policy.value == "dense"
    assert legacy_disabled.decision.effective_policy.value == "dense"

    legacy_enabled = engine._resolve_specprefill_telemetry(
        legacy=True,
        policy=None,
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert legacy_enabled.decision.requested_policy.value == "sparse"
    assert legacy_enabled.decision.effective_policy.value == "dense"
    assert legacy_enabled.decision.fallback_reason == "profile_not_registered"


def test_auto_selective_engages_only_with_value_admission_and_media_is_dense():
    engine = _engine()

    eligible = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert eligible.decision.effective_policy.value == "dense"
    assert eligible.decision.fallback_reason == "profile_not_registered"

    registered = _engine(registered=True)
    eligible = registered._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert eligible.decision.effective_policy.value == "sparse"
    assert eligible.as_output_kwargs()["specprefill_engaged"] is True
    assert (
        eligible.as_output_kwargs()["specprefill_selector_version"]
        == "test-sparse-selector-v1"
    )
    assert eligible.profile_tuning == SpecPrefillTuning(0.55, 0.1, 1, 1, 32)
    assert eligible.selected_tokens is None

    media = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=True,
        total_tokens=8,
    )
    assert media.decision.effective_policy.value == "dense"
    assert media.decision.fallback_reason == "media_request"
    assert media.selected_tokens == 8


def test_explicit_sparse_and_selector_overrides_are_diagnostic_only():
    production = _engine()
    diagnostic = _engine(diagnostic=True)

    assert (
        production._resolve_specprefill_telemetry(
            legacy=None,
            policy="sparse",
            coverage="selective",
            has_media=False,
            total_tokens=8,
        ).decision.fallback_reason
        == "sparse_forcing_diagnostic_only"
    )
    assert (
        diagnostic._resolve_specprefill_telemetry(
            legacy=None,
            policy="sparse",
            coverage="exhaustive",
            has_media=False,
            total_tokens=8,
        ).decision.effective_policy.value
        == "sparse"
    )
    # Preserve diagnostic legacy forcing. Production maps the same legacy
    # intent through an evidence-backed selective auto profile instead.
    assert (
        diagnostic._resolve_specprefill_telemetry(
            legacy=True,
            policy=None,
            coverage="unknown",
            has_media=False,
            total_tokens=8,
        ).decision.effective_policy.value
        == "sparse"
    )

    controls = {"specprefill_keep_pct": 0.2, "specprefill_backbone_pct": 0.1}
    parsed = diagnostic._specprefill_controls(controls)
    assert parsed["keep_pct"] == 0.2
    assert parsed["backbone_pct"] == 0.1
    assert controls == {}


def test_diagnostic_auto_requires_an_exact_diagnostic_profile():
    unregistered = _engine(diagnostic=True)
    denied = unregistered._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert denied.decision.fallback_reason == "profile_not_registered"

    registered = _engine(diagnostic=True, registered=True)
    eligible = registered._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert eligible.decision.effective_policy.value == "sparse"
    assert eligible.profile_tuning == SpecPrefillTuning(0.55, 0.1, 1, 1, 32)


def test_profile_cell_is_selected_from_the_actual_request_mtp_route():
    engine = _engine(mtp=True, registered=True)

    sparse_only = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
        combined_mtp=False,
    )
    combined = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
        combined_mtp=True,
    )

    assert sparse_only.profile_selector_version == "test-sparse-selector-v1"
    assert sparse_only.profile_tuning == SpecPrefillTuning(0.55, 0.1, 1, 1, 32)
    assert combined.profile_selector_version == "test-combined-selector-v1"
    assert combined.profile_tuning == SpecPrefillTuning(0.65, 0.2, 2, 3, 64)


def test_retiring_processor_selects_combined_cell_but_permanent_processor_does_not(
    monkeypatch,
):
    class RetiringProcessor:
        is_retired = False

    class PermanentProcessor:
        pass

    monkeypatch.setenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME", "1")
    assert _request_can_compose_mtp(True, [RetiringProcessor()]) is True
    assert _request_can_compose_mtp(True, [PermanentProcessor()]) is False

    engine = _engine(mtp=True, registered=True)
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
        combined_mtp=_request_can_compose_mtp(True, [RetiringProcessor()]),
    )
    assert telemetry.profile_selector_version == "test-combined-selector-v1"


def test_profile_keys_must_match_the_simple_engine_cell():
    sparse_key = _profile_key(cell=SpecPrefillCell.SPARSE_ONLY)
    combined_key = _profile_key(cell=SpecPrefillCell.COMBINED_MTP)

    with pytest.raises(ValueError, match="sparse_only"):
        SimpleEngine("test-model", specprefill_sparse_profile_key=combined_key)
    with pytest.raises(ValueError, match="SimpleEngine"):
        SimpleEngine(
            "test-model",
            specprefill_sparse_profile_key=replace(
                sparse_key, engine=SpecPrefillEngine.CONTINUOUS_BATCHING
            ),
        )


def test_registered_profile_requires_an_estimated_residency_value():
    engine = _engine(registered=True, estimated_residency=False)
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert telemetry.decision.effective_policy.value == "dense"
    assert telemetry.decision.fallback_reason == "profile_residency_not_estimated"


def test_long_context_has_no_hidden_ceiling_and_production_overrides_fail():
    uncapped = _engine(registered=True)
    long_context = uncapped._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=127 * 1024,
    )
    assert long_context.decision.effective_policy.value == "sparse"

    capped = SimpleEngine(
        "test-model",
        specprefill_threshold=4,
        specprefill_max_tokens=16,
    )
    capped._draft_model = object()
    denied = capped._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=17,
    )
    assert denied.decision.fallback_reason == "admission_denied"

    with pytest.raises(ValueError, match="diagnostic-only"):
        uncapped._specprefill_controls(
            {"specprefill_keep_pct": 0.2, "specprefill_has_media": False}
        )


@pytest.mark.anyio
async def test_production_profile_forwards_every_calibrated_selector_control(
    monkeypatch,
):
    engine = _engine(registered=True)
    engine._loaded = True
    engine._model = SimpleNamespace(
        tokenizer=SimpleNamespace(
            bos_token=None,
            encode=lambda *_args, **_kwargs: list(range(8)),
        )
    )
    received: dict[str, object] = {}

    async def fake_sparse_stream(*_args, **kwargs):
        received.update(
            {
                name: kwargs[name]
                for name in (
                    "specprefill_keep_pct",
                    "specprefill_backbone_pct",
                    "specprefill_chunk_size",
                    "specprefill_halo_chunks",
                    "specprefill_anchor_chunks",
                )
            }
        )
        yield GenerationOutput(text="ok")

    monkeypatch.setattr(engine, "_stream_generate_specprefill", fake_sparse_stream)
    outputs = [
        output
        async for output in engine._stream_generate_impl(
            "prompt",
            max_tokens=1,
            specprefill_policy="auto",
            specprefill_coverage="selective",
        )
    ]

    assert outputs[-1].text == "ok"
    assert received == {
        "specprefill_keep_pct": 0.55,
        "specprefill_backbone_pct": 0.1,
        "specprefill_chunk_size": 32,
        "specprefill_halo_chunks": 1,
        "specprefill_anchor_chunks": 1,
    }


@pytest.mark.anyio
async def test_direct_native_mtp_routes_dense_until_composition_is_qualified(
    monkeypatch,
):
    engine = _engine(mtp=True, registered=True)
    engine._loaded = True
    engine._model = SimpleNamespace(
        tokenizer=SimpleNamespace(
            bos_token=None,
            encode=lambda *_args, **_kwargs: list(range(8)),
        ),
        stream_generate=lambda **_kwargs: iter(
            [SimpleNamespace(text="dense", finish_reason="stop")]
        ),
    )
    received: dict[str, object] = {}

    async def fake_sparse_stream(*_args, **kwargs):
        received["selector_version"] = kwargs["telemetry"].profile_selector_version
        yield GenerationOutput(text="ok")

    monkeypatch.setattr(engine, "_stream_generate_specprefill", fake_sparse_stream)
    outputs = [
        output
        async for output in engine._stream_generate_impl(
            "prompt",
            max_tokens=1,
            specprefill_policy="auto",
            specprefill_coverage="selective",
        )
    ]

    assert outputs[-1].text == "dense"
    assert received == {}
    assert outputs[-1].specprefill_effective_policy == "dense"
    assert outputs[-1].specprefill_fallback_reason == "mtp_composition_not_implemented"


def test_sparse_eos_accepts_tokenizer_multi_id_contract():
    tokenizer = SimpleNamespace(eos_token_id=1, eos_token_ids=(1, 7))
    assert SimpleEngine._eos_token_ids(tokenizer) == frozenset((1, 7))


def test_sparse_cache_identity_rejects_selector_version_drift():
    engine = _engine(registered=True)
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    target = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    with pytest.raises(ValueError, match="selector version"):
        engine._prepare_sparse_target_prefill(
            target_model=target,
            tokenizer=SimpleNamespace(all_special_ids=()),
            tokens=list(range(8)),
            importance=mx.ones((8,)),
            cache=[],
            telemetry=telemetry,
            keep_pct=0.55,
            backbone_pct=0.1,
            chunk_size=32,
            halo_chunks=1,
            anchor_chunks=1,
            profile_key=engine._active_specprefill_profile_key(False),
            adapter=QWEN_DENSE_TARGET,
        )


@pytest.mark.anyio
async def test_zero_max_tokens_never_enters_sparse_seed_path(monkeypatch):
    engine = _engine(diagnostic=True)
    engine._loaded = True
    engine._model = SimpleNamespace(
        tokenizer=SimpleNamespace(
            bos_token=None,
            encode=lambda *_args, **_kwargs: list(range(8)),
        ),
        stream_generate=lambda **_kwargs: iter(()),
    )
    entered_sparse = False

    async def fake_sparse_stream(*_args, **_kwargs):
        nonlocal entered_sparse
        entered_sparse = True
        yield GenerationOutput(text="unexpected")

    monkeypatch.setattr(engine, "_stream_generate_specprefill", fake_sparse_stream)
    outputs = [
        output
        async for output in engine._stream_generate_impl(
            "prompt",
            max_tokens=0,
            specprefill_policy="sparse",
            specprefill_coverage="selective",
        )
    ]
    assert entered_sparse is False
    assert outputs[-1].specprefill_effective_policy == "dense"
    assert outputs[-1].specprefill_fallback_reason == "sparse_no_completion_requested"


@pytest.mark.anyio
async def test_target_prefill_failure_before_sampling_restarts_dense(monkeypatch):
    engine = _engine(diagnostic=True)
    engine._loaded = True
    engine._model = SimpleNamespace(
        model=object(),
        tokenizer=SimpleNamespace(eos_token_id=0),
        stream_generate=lambda **_kwargs: iter(
            [SimpleNamespace(text="dense", finish_reason="stop")]
        ),
    )
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="sparse",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )

    def fail_sparse_prefill(*_args, **_kwargs):
        raise RuntimeError("target sparse prefill failed")

    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda *_a, **_k: [])
    monkeypatch.setattr("vllm_mlx.specprefill.score_tokens", lambda *_a, **_k: object())
    monkeypatch.setattr(engine, "_supports_sparse_continuation", lambda *_a: True)
    monkeypatch.setattr(
        engine,
        "_admit_sparse_target",
        lambda *_a: (SimpleNamespace(adapter_id="qwen_dense"), QWEN_DENSE_TARGET),
    )
    monkeypatch.setattr(
        "vllm_mlx.engine.simple._new_sparse_detokenizer", lambda *_a: object()
    )
    monkeypatch.setattr(engine, "_prepare_sparse_target_prefill", fail_sparse_prefill)
    outputs = [
        output
        async for output in engine._stream_generate_specprefill(
            "prompt",
            list(range(8)),
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            telemetry=telemetry,
        )
    ]

    assert telemetry.scorer_ms is not None
    assert outputs[-1].new_text == "dense"
    assert telemetry.decision.effective_policy.value == "dense"
    assert telemetry.decision.fallback_reason == "sparse_execution_failed"


def test_sparse_fallback_retains_completed_phase_timing():
    engine = _engine(diagnostic=True)
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="sparse",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    telemetry.scorer_ms = 12.5
    telemetry.target_prefill_ms = 30.0
    telemetry.fallback("sparse_execution_failed")

    metadata = telemetry.as_output_kwargs()
    assert metadata["specprefill_effective_policy"] == "dense"
    assert metadata["specprefill_scorer_ms"] == 12.5
    assert metadata["specprefill_target_prefill_ms"] == 30.0


@pytest.mark.anyio
async def test_combined_mtp_and_specprefill_metadata_are_independent():
    engine = _engine(mtp=True, registered=True)
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
        combined_mtp=True,
    )
    telemetry.selected_tokens = 4
    telemetry.scorer_ms = 3.5
    telemetry.target_prefill_ms = 8.0

    output = GenerationOutput(
        text="ok",
        mtp_drafts=6,
        mtp_accepted=5,
        **telemetry.as_output_kwargs(),
    )

    assert output.mtp_drafts == 6
    assert output.mtp_accepted == 5
    assert output.specprefill_effective_policy == "sparse"
    assert output.specprefill_engaged is True
    assert output.specprefill_selector_version == "test-combined-selector-v1"
    assert output.specprefill_total_tokens == 8
    assert output.specprefill_selected_tokens == 4
