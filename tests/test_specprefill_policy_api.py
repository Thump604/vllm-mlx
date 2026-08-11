"""Public API contract for request-safe SpecPrefill policy controls."""

import pytest
from pydantic import ValidationError
from types import SimpleNamespace
import asyncio
import json

from vllm_mlx.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
    GenerationMetadata,
)


def _chat(**overrides):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


@pytest.mark.parametrize(
    ("legacy", "policy"),
    [(False, "dense"), (True, "sparse")],
)
def test_legacy_and_policy_may_agree(legacy, policy):
    request = _chat(specprefill=legacy, specprefill_policy=policy)

    assert request.specprefill is legacy
    assert request.specprefill_policy == policy


@pytest.mark.parametrize(
    ("legacy", "policy"),
    [(False, "auto"), (False, "sparse"), (True, "auto"), (True, "dense")],
)
def test_conflicting_legacy_and_policy_are_rejected(legacy, policy):
    with pytest.raises(ValidationError, match="conflicts with specprefill_policy"):
        _chat(specprefill=legacy, specprefill_policy=policy)


@pytest.mark.parametrize("field", ["specprefill_keep_pct", "specprefill_backbone_pct"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_tuning_percentages_are_bounded(field, value):
    with pytest.raises(ValidationError):
        CompletionRequest(model="test-model", prompt="hello", **{field: value})


def test_policy_and_coverage_values_are_closed_enums():
    with pytest.raises(ValidationError):
        _chat(specprefill_policy="sometimes")
    with pytest.raises(ValidationError):
        _chat(specprefill_coverage="maybe")


def test_generation_metadata_exposes_independent_specprefill_and_mtp_fields():
    metadata = GenerationMetadata(
        mtp_drafts=12,
        mtp_accepted=8,
        specprefill_requested_policy="auto",
        specprefill_effective_policy="sparse",
        specprefill_coverage="selective",
        specprefill_engaged=True,
        specprefill_selector_version="hybrid-v1",
        specprefill_total_tokens=10_000,
        specprefill_selected_tokens=3_000,
        specprefill_scorer_ms=125.5,
        specprefill_target_prefill_ms=810.0,
    )

    dumped = metadata.model_dump(exclude_none=True)
    assert dumped["mtp_drafts"] == 12
    assert dumped["mtp_accepted"] == 8
    assert dumped["specprefill_engaged"] is True
    assert dumped["specprefill_selected_tokens"] == 3_000


def test_chat_invocation_forwards_policy_coverage_and_media_state():
    from vllm_mlx.server import _prepare_chat_completion_invocation

    engine = SimpleNamespace(is_mllm=False, preserve_native_tool_format=False)
    request = _chat(
        specprefill_policy="auto",
        specprefill_coverage="selective",
    )

    prepared = _prepare_chat_completion_invocation(engine, request, 16)

    assert prepared.chat_kwargs["specprefill_policy"] == "auto"
    assert prepared.chat_kwargs["specprefill_coverage"] == "selective"
    assert prepared.chat_kwargs["specprefill_has_media"] is False


def test_server_metadata_combines_prefill_decode_and_watchdog():
    from vllm_mlx.server import _generation_metadata

    processor = SimpleNamespace(
        _no_final_content_token_limit=20,
        watchdog_was_enforced=False,
    )
    output = SimpleNamespace(
        mtp_drafts=12,
        mtp_accepted=8,
        specprefill_requested_policy="auto",
        specprefill_effective_policy="sparse",
        specprefill_coverage="selective",
        specprefill_engaged=True,
        specprefill_selector_version="hybrid-v1",
        specprefill_fallback_reason=None,
        specprefill_total_tokens=10_000,
        specprefill_selected_tokens=3_000,
        specprefill_scorer_ms=125.5,
        specprefill_target_prefill_ms=810.0,
    )

    metadata = _generation_metadata(processor, output)

    assert metadata.no_final_content_watchdog_tokens == 20
    assert metadata.mtp_drafts == 12
    assert metadata.specprefill_engaged is True


def test_stream_terminal_chunk_includes_feature_metadata():
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.server import stream_chat_completion

    class Engine:
        model_name = "test-model"
        preserve_native_tool_format = False

        async def stream_chat(self, **kwargs):
            yield GenerationOutput(
                text="ok",
                new_text="ok",
                finished=True,
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=1,
                mtp_drafts=2,
                mtp_accepted=1,
                specprefill_requested_policy="auto",
                specprefill_effective_policy="sparse",
                specprefill_coverage="selective",
                specprefill_engaged=True,
            )

    async def collect():
        return [
            chunk
            async for chunk in stream_chat_completion(
                Engine(),
                [{"role": "user", "content": "hello"}],
                _chat(stream=True),
            )
        ]

    chunks = asyncio.run(collect())
    terminal = next(
        json.loads(chunk.removeprefix("data: "))
        for chunk in chunks
        if chunk.startswith("data: {") and '"finish_reason":"stop"' in chunk
    )
    assert terminal["generation_metadata"]["mtp_drafts"] == 2
    assert terminal["generation_metadata"]["specprefill_engaged"] is True
