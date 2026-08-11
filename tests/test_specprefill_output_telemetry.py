"""Engine output telemetry remains independent across prefill and decode."""

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.request import RequestOutput


def _assert_feature_telemetry(output):
    assert output.mtp_drafts == 10
    assert output.mtp_accepted == 7
    assert output.specprefill_engaged is True
    assert output.specprefill_requested_policy == "auto"
    assert output.specprefill_effective_policy == "sparse"
    assert output.specprefill_coverage == "selective"
    assert output.specprefill_selector_version == "hybrid-v1"
    assert output.specprefill_total_tokens == 8192
    assert output.specprefill_selected_tokens == 3072
    assert output.specprefill_scorer_ms == 12.5
    assert output.specprefill_target_prefill_ms == 40.0


def test_generation_output_carries_independent_feature_telemetry():
    output = GenerationOutput(
        text="ok",
        mtp_drafts=10,
        mtp_accepted=7,
        specprefill_requested_policy="auto",
        specprefill_effective_policy="sparse",
        specprefill_coverage="selective",
        specprefill_engaged=True,
        specprefill_selector_version="hybrid-v1",
        specprefill_total_tokens=8192,
        specprefill_selected_tokens=3072,
        specprefill_scorer_ms=12.5,
        specprefill_target_prefill_ms=40.0,
    )

    _assert_feature_telemetry(output)


def test_request_output_carries_independent_feature_telemetry():
    output = RequestOutput(
        request_id="request-1",
        mtp_drafts=10,
        mtp_accepted=7,
        specprefill_requested_policy="auto",
        specprefill_effective_policy="sparse",
        specprefill_coverage="selective",
        specprefill_engaged=True,
        specprefill_selector_version="hybrid-v1",
        specprefill_total_tokens=8192,
        specprefill_selected_tokens=3072,
        specprefill_scorer_ms=12.5,
        specprefill_target_prefill_ms=40.0,
    )

    _assert_feature_telemetry(output)
