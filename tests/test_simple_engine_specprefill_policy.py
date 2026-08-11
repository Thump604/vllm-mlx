# SPDX-License-Identifier: Apache-2.0
"""Mocked SimpleEngine policy and telemetry contracts for SpecPrefill."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.simple import SimpleEngine


def _engine(*, diagnostic: bool = False) -> SimpleEngine:
    engine = SimpleEngine(
        "test-model",
        specprefill_threshold=4,
        specprefill_diagnostic_mode=diagnostic,
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
    assert legacy_enabled.decision.fallback_reason == "sparse_forcing_diagnostic_only"


def test_auto_selective_engages_only_with_value_admission_and_media_is_dense():
    engine = _engine()

    eligible = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
    )
    assert eligible.decision.effective_policy.value == "sparse"
    assert eligible.as_output_kwargs()["specprefill_engaged"] is True
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

    controls = {"specprefill_keep_pct": 0.2, "specprefill_backbone_pct": 0.1}
    parsed = diagnostic._specprefill_controls(controls)
    assert parsed["keep_pct"] == 0.2
    assert parsed["backbone_pct"] == 0.1
    assert controls == {}


def test_long_context_has_no_hidden_ceiling_and_production_overrides_fail():
    uncapped = _engine()
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
async def test_sparse_failure_restarts_dense_and_reports_fallback(monkeypatch):
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
    monkeypatch.setattr(
        "vllm_mlx.specprefill.select_chunks",
        lambda *_a, **_k: SimpleNamespace(shape=(4,)),
    )
    monkeypatch.setattr("vllm_mlx.specprefill.sparse_prefill", fail_sparse_prefill)
    monkeypatch.setattr("vllm_mlx.specprefill.cleanup_rope", lambda *_a, **_k: None)
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

    assert outputs[-1].text == "dense"
    assert outputs[-1].specprefill_requested_policy == "sparse"
    assert outputs[-1].specprefill_effective_policy == "dense"
    assert outputs[-1].specprefill_engaged is False
    assert outputs[-1].specprefill_fallback_reason == "sparse_execution_failed"
    assert outputs[-1].specprefill_total_tokens == 8
    assert outputs[-1].specprefill_selected_tokens == 8
    assert outputs[-1].specprefill_scorer_ms is not None
    assert outputs[-1].specprefill_target_prefill_ms is None


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
    engine = _engine()
    telemetry = engine._resolve_specprefill_telemetry(
        legacy=None,
        policy="auto",
        coverage="selective",
        has_media=False,
        total_tokens=8,
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
    assert output.specprefill_selector_version == "hybrid-chunk-v1"
    assert output.specprefill_total_tokens == 8
    assert output.specprefill_selected_tokens == 4
