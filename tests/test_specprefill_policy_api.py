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


def _completion(**overrides):
    payload = {"model": "test-model", "prompt": "hello"}
    payload.update(overrides)
    return CompletionRequest(**payload)


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
    with pytest.raises(ValidationError, match="conflicts with specprefill_policy"):
        _completion(specprefill=legacy, specprefill_policy=policy)


def test_completion_conflicting_policy_body_returns_http_422_without_engine_load():
    from fastapi.testclient import TestClient
    import vllm_mlx.server as server

    original_api_key = server._api_key
    server._api_key = None
    try:
        response = TestClient(server.app).post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "specprefill": False,
                "specprefill_policy": "sparse",
            },
        )
    finally:
        server._api_key = original_api_key

    assert response.status_code == 422
    assert "conflicts with specprefill_policy" in response.text


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


def test_completion_nonstreaming_forwards_contract_and_returns_terminal_metadata(
    monkeypatch,
):
    from vllm_mlx.engine.base import GenerationOutput
    import vllm_mlx.server as server

    class Engine:
        seen_kwargs = None

        async def generate(self, **kwargs):
            self.seen_kwargs = kwargs
            return GenerationOutput(
                text="ok",
                new_text="ok",
                finished=True,
                finish_reason="stop",
                prompt_tokens=20,
                completion_tokens=2,
                mtp_drafts=4,
                mtp_accepted=3,
                specprefill_requested_policy="auto",
                specprefill_effective_policy="sparse",
                specprefill_coverage="selective",
                specprefill_engaged=True,
                specprefill_selector_version="hybrid-v1",
                specprefill_fallback_reason=None,
                specprefill_total_tokens=20,
                specprefill_selected_tokens=8,
                specprefill_scorer_ms=1.5,
                specprefill_target_prefill_ms=2.5,
            )

    engine = Engine()

    async def acquire(*_args, **_kwargs):
        return engine

    async def release(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "_model_name", "test-model")
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_engine_for_request", release)
    response = asyncio.run(
        server.create_completion(
            _completion(specprefill_policy="auto", specprefill_coverage="selective"),
            raw_request=None,
        )
    )

    assert engine.seen_kwargs["specprefill_policy"] == "auto"
    assert engine.seen_kwargs["specprefill_coverage"] == "selective"
    assert response.generation_metadata is not None
    assert response.generation_metadata.mtp_drafts == 4
    assert response.generation_metadata.mtp_accepted == 3
    assert response.generation_metadata.specprefill_selected_tokens == 8
    assert response.generation_metadata.specprefill_target_prefill_ms == 2.5


def test_completion_stream_terminal_includes_independent_feature_metadata():
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.server import stream_completion

    class Engine:
        async def stream_generate(self, **_kwargs):
            yield GenerationOutput(
                text="ok",
                new_text="ok",
                finished=True,
                finish_reason="stop",
                prompt_tokens=11,
                completion_tokens=1,
                mtp_drafts=2,
                mtp_accepted=1,
                specprefill_requested_policy="auto",
                specprefill_effective_policy="dense",
                specprefill_coverage="exhaustive",
                specprefill_engaged=False,
                specprefill_selector_version="hybrid-v1",
                specprefill_fallback_reason="coverage_not_selective",
                specprefill_total_tokens=11,
                specprefill_selected_tokens=0,
                specprefill_scorer_ms=0.0,
                specprefill_target_prefill_ms=0.0,
            )

    async def collect():
        return [
            chunk
            async for chunk in stream_completion(
                Engine(), "hello", _completion(stream=True), max_tokens=2
            )
        ]

    chunks = asyncio.run(collect())
    terminal = next(
        json.loads(chunk.removeprefix("data: "))
        for chunk in chunks
        if chunk.startswith("data: {") and '"finish_reason": "stop"' in chunk
    )
    metadata = terminal["generation_metadata"]
    assert metadata["mtp_drafts"] == 2
    assert metadata["mtp_accepted"] == 1
    assert metadata["specprefill_effective_policy"] == "dense"
    assert metadata["specprefill_engaged"] is False
    assert metadata["specprefill_fallback_reason"] == "coverage_not_selective"


def test_completion_prompt_lists_aggregate_counts_without_misreporting_route():
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.server import _aggregate_completion_generation_metadata

    outputs = [
        GenerationOutput(
            text="first",
            mtp_drafts=2,
            mtp_accepted=1,
            specprefill_requested_policy="auto",
            specprefill_effective_policy="sparse",
            specprefill_coverage="selective",
            specprefill_engaged=True,
            specprefill_total_tokens=10,
            specprefill_selected_tokens=4,
            specprefill_scorer_ms=1.0,
            specprefill_target_prefill_ms=2.0,
        ),
        GenerationOutput(
            text="second",
            mtp_drafts=3,
            mtp_accepted=2,
            specprefill_requested_policy="auto",
            specprefill_effective_policy="dense",
            specprefill_coverage="exhaustive",
            specprefill_engaged=False,
            specprefill_total_tokens=20,
            specprefill_selected_tokens=0,
            specprefill_scorer_ms=0.0,
            specprefill_target_prefill_ms=0.0,
        ),
    ]

    metadata = _aggregate_completion_generation_metadata(outputs)

    assert metadata is not None
    assert metadata.mtp_drafts == 5
    assert metadata.mtp_accepted == 3
    assert metadata.specprefill_total_tokens == 30
    assert metadata.specprefill_selected_tokens == 4
    assert metadata.specprefill_effective_policy is None
    assert metadata.specprefill_engaged is None
