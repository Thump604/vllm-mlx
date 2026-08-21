# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the versioned capacity-envelope evidence contract."""

import pytest

from vllm_mlx.bench_serve import (
    CapacityConfig,
    CapacityThresholds,
    _capacity_hardware_fingerprint,
    _capacity_memory_from_status,
    _capacity_telemetry_delta,
    classify_capacity_error,
    compare_capacity_outputs,
    parse_sse_line,
    run_capacity_envelope,
    sha256_json,
    summarize_capacity_cell,
    summarize_capacity_envelope,
)


def _sample(**overrides):
    value = {
        "error": "",
        "error_kind": None,
        "ttft_ms": 10.0,
        "chunk_gap_ms": 2.0,
        "e2e_latency_ms": 20.0,
        "queue_time_ms": None,
        "prompt_tokens": 16,
        "completion_tokens": 8,
        "correctness": {"status": "not_run"},
        "memory": {"source": "unavailable"},
    }
    value.update(overrides)
    return value


def test_partial_failure_fails_cell_and_does_not_inflate_rps():
    cell = summarize_capacity_cell(
        prompt_tokens=16,
        output_tokens=8,
        cache_mode="warm",
        concurrency=2,
        samples=[_sample(), _sample(error="timed out", error_kind="timeout")],
        batch_durations_ms=[1000.0],
        thresholds=CapacityThresholds(),
    )

    assert cell["failure_count"] == 1
    assert cell["timeout_count"] == 1
    assert cell["requests_per_s"] == 1.0
    assert cell["sustainable"] is False


def test_unavailable_observations_remain_null(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr("subprocess.run", unavailable)

    hardware = _capacity_hardware_fingerprint()
    assert hardware["gpu_cores"] is None
    assert hardware["bandwidth_gbs"] is None
    assert hardware["chip"] is None
    assert hardware["memory_gb"] is None
    assert _capacity_memory_from_status({}) == {"source": "unavailable"}
    assert _capacity_telemetry_delta({}, {})["available"] is False


def test_sse_timing_inputs_are_explicitly_chunks_with_optional_token_ids():
    parsed = parse_sse_line(
        'data: {"choices":[{"delta":{"content":"several tokens"},'
        '"logprobs":{"content":[{"token_id":7},{"token_id":8}]}}],'
        '"metrics":{"queue_time_ms":3.5}}'
    )

    assert parsed["content"] == "several tokens"
    assert parsed["token_ids"] == [7, 8]
    assert parsed["queue_time_ms"] == 3.5


def test_workload_hash_is_order_stable_and_output_parity_prefers_token_ids():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    comparison = compare_capacity_outputs(
        {"token_ids": [1, 2], "content": "ignored"},
        {"token_ids": [1, 2], "content": "different"},
    )
    assert comparison["status"] == "match"
    assert comparison["method"] == "token_ids"


def test_envelope_reports_throughput_per_gib_not_concurrency_per_gib():
    config = CapacityConfig(
        concurrencies=(1,),
        prompt_tokens=(16,),
        output_tokens=(8,),
        cache_modes=("warm",),
    )
    summary = summarize_capacity_envelope(
        [{"concurrency": 1, "sustainable": True, "output_throughput_tps": 64.0}],
        config=config,
        memory_gb=16.0,
    )
    assert summary["throughput_per_gib"] == 4.0
    assert summary["sustainable_concurrency_per_gib"] == 0.0625


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("request timeout", "timeout"),
        ("Metal out of memory", "oom"),
        ("HTTP 500", "error"),
    ],
)
def test_error_classification(message, kind):
    assert classify_capacity_error(message) == kind


class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


@pytest.mark.anyio
async def test_cold_cell_requires_verified_reset(monkeypatch):
    monkeypatch.setattr("vllm_mlx.bench_serve.httpx.AsyncClient", _DummyClient)
    monkeypatch.setattr(
        "vllm_mlx.bench_serve.auto_detect_runtime",
        lambda *args: _async_value({"model_id": "test-model"}),
    )
    monkeypatch.setattr(
        "vllm_mlx.bench_serve._fetch_post_run_status",
        lambda *args: _async_value({}),
    )
    monkeypatch.setattr(
        "vllm_mlx.bench_serve.clear_runtime_cache",
        lambda *args: _async_value({"ok": False, "status_code": 404, "error": ""}),
    )

    with pytest.raises(RuntimeError, match="could not be verified"):
        await run_capacity_envelope(
            model="test-model",
            config=CapacityConfig(
                concurrencies=(1,),
                prompt_tokens=(8,),
                output_tokens=(4,),
                cache_modes=("cold",),
                repetitions=1,
                warmup=3,
            ),
        )


@pytest.mark.anyio
async def test_cold_cell_never_warms_after_reset(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("vllm_mlx.bench_serve.httpx.AsyncClient", _DummyClient)
    monkeypatch.setattr(
        "vllm_mlx.bench_serve.auto_detect_runtime",
        lambda *args: _async_value({"model_id": "test-model"}),
    )
    monkeypatch.setattr(
        "vllm_mlx.bench_serve._fetch_post_run_status",
        lambda *args: _async_value({}),
    )
    monkeypatch.setattr(
        "vllm_mlx.bench_serve.scrape_metrics",
        lambda *args: _async_value({}),
    )
    monkeypatch.setattr(
        "vllm_mlx.bench_serve._capacity_hardware_fingerprint",
        lambda: {"memory_gb": None},
    )
    monkeypatch.setattr(
        "vllm_mlx.bench_serve.clear_runtime_cache",
        lambda *args: _async_value(
            {
                "ok": True,
                "status_code": 200,
                "response": {
                    "status": "cleared",
                    "engine_cache": {"prefix_cache": True},
                },
            }
        ),
    )

    async def fake_request(*args, **kwargs):
        calls.append(kwargs)
        return _sample(
            finish_reason="stop",
            content="ok",
            telemetry_delta={"available": False, "values": None},
        )

    monkeypatch.setattr("vllm_mlx.bench_serve._run_capacity_request", fake_request)
    result = await run_capacity_envelope(
        model="test-model",
        output_path=str(tmp_path / "capacity.json"),
        config=CapacityConfig(
            concurrencies=(2,),
            prompt_tokens=(8,),
            output_tokens=(4,),
            cache_modes=("cold",),
            repetitions=1,
            warmup=3,
        ),
    )

    assert len(calls) == 2
    assert result["cells"][0]["cache_actions"][0]["warmup_requests"] == 0


async def _async_value(value):
    return value
