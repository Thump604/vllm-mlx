# SPDX-License-Identifier: Apache-2.0
"""Portable integration tests for recorded model and hardware fit fixtures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from vllm_mlx.hardware import HardwareInventory, SourceRecord
from vllm_mlx.model_family_adapters import (
    CallerProfileInput,
    KvProfileInputs,
    adapt_model_config,
)
from vllm_mlx.model_fit import (
    ArtifactFile,
    ArtifactResidencyInput,
    MemoryMarginInput,
    estimate_artifact_residency,
    estimate_dense_gqa_kv_cache,
    estimate_memory_margin,
    select_serving_context,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "model_configs"
_GIB = 1024**3


def _fixture(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _config(name: str) -> dict[str, object]:
    config = _fixture(name)["config"]
    assert isinstance(config, dict)
    return config


def _profile() -> KvProfileInputs:
    return KvProfileInputs(
        element_bytes=CallerProfileInput(2, "fixture:bf16-kv"),
        context_tokens=CallerProfileInput(32_768, "fixture:32k-policy"),
        concurrency=CallerProfileInput(1, "fixture:single-request"),
        quantization_overhead_bytes=CallerProfileInput(
            0, "fixture:no-packing-overhead"
        ),
    )


def _m2_ultra_inventory() -> HardwareInventory:
    total_memory = 128 * _GIB
    return HardwareInventory(
        operating_system="Darwin",
        architecture="arm64",
        is_apple_silicon=True,
        chip_string="Apple M2 Ultra",
        machine_model=None,
        machine_name=None,
        total_unified_memory_bytes=total_memory,
        reported_cpu_configuration=None,
        logical_cpu_cores=None,
        physical_cpu_cores=None,
        gpu_core_count=60,
        macos_version=None,
        macos_build=None,
        sources=(
            SourceRecord(
                field="total_unified_memory_bytes",
                source="recorded_fixture",
                locator="tests/model-fit/m2-ultra-128gb",
                status="reported",
                value=total_memory,
            ),
            SourceRecord(
                field="gpu_core_count",
                source="recorded_fixture",
                locator="tests/model-fit/m2-ultra-128gb",
                status="reported",
                value=60,
            ),
        ),
    )


def test_provider_fixture_metadata_is_pinned_to_recorded_revision_and_digest():
    qwen = _fixture("qwen3-5-hybrid.json")["fixture_metadata"]
    laguna = _fixture("laguna-s-2-1.json")["fixture_metadata"]

    assert qwen == {
        "source_repository": "mlx-community/Qwen3.6-35B-A3B-8bit",
        "source_revision": "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
        "base_model_repository": "Qwen/Qwen3.6-35B-A3B",
        "source_config_sha256": "5da49c3a84c5a4a751b720b62b9f8703bdb8a0796b5fc55df3278f8b9d6ce296",
    }
    assert laguna == {
        "source_repository": "poolside/Laguna-S-2.1",
        "source_revision": "a50e85e7e0aae7b0a504d156bd36a616ec9fea38",
        "local_config_sha256": "8440d3ec23e275aa62bba1371c20cee4a72906fdc33ca37966ba7cd83472847b",
    }


def test_dense_fixture_adapter_feeds_exact_generic_kv_calculation():
    adapted = adapt_model_config(_config("dense-gqa.json"), kv_profile=_profile())

    assert adapted.status == "ready"
    assert adapted.dense_gqa_kv_input is not None
    estimate = estimate_dense_gqa_kv_cache(adapted.dense_gqa_kv_input)

    assert estimate.status == "estimated"
    assert estimate.bytes_per_token == 131_072
    assert estimate.total_bytes == 4 * _GIB
    assert {item.classification for item in adapted.input_provenance} == {
        "provider_fact",
        "derived",
        "caller_profile",
    }


def test_laguna_recorded_artifact_and_hardware_keep_incomplete_margin_unknown():
    artifact_fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "model_fit"
            / "laguna-s-2-1-artifact.json"
        ).read_text()
    )
    files = tuple(
        ArtifactFile(item["name"], item["size_bytes"])
        for item in artifact_fixture["weight_files"]
    )
    expected_names = tuple(
        f"model-{index:05d}-of-00013.safetensors" for index in range(1, 14)
    )
    residency = estimate_artifact_residency(ArtifactResidencyInput(files))

    assert len(files) == 13
    assert tuple(item.name for item in files) == expected_names
    assert all(item.size_bytes > 0 for item in files)
    assert residency.status == "exact"
    assert residency.artifact_bytes == 66_147_820_723
    assert artifact_fixture["indexed_tensor_bytes"] == 66_147_556_864

    margin = estimate_memory_margin(
        MemoryMarginInput(
            hardware=_m2_ultra_inventory(),
            weight_residency_bytes=residency.artifact_bytes,
            kv_cache_bytes=None,
            system_reserve_bytes=16 * _GIB,
            runtime_overhead_bytes=4 * _GIB,
            temporary_buffers_bytes=2 * _GIB,
        )
    )

    assert margin.status == "unknown"
    assert margin.margin_bytes is None
    assert margin.hardware_memory_source is not None
    assert margin.hardware_memory_source.source == "recorded_fixture"
    assert any("kv_cache_bytes" in reason for reason in margin.reasons)


def test_dense_fixture_context_selection_remains_explainable():
    adapted = adapt_model_config(_config("dense-gqa.json"), kv_profile=_profile())
    assert adapted.context_limits is not None

    context = select_serving_context(adapted.context_limits)

    assert context.serving_context_tokens == 32_768
    assert context.advertised_context_tokens == 131_072


def test_hybrid_fixtures_keep_unsupported_kv_unknown_without_heuristics():
    qwen = adapt_model_config(_config("qwen3-5-hybrid.json"))
    laguna = adapt_model_config(_config("laguna-s-2-1.json"))

    assert qwen.status == "ready"
    assert qwen.dense_gqa_kv_input is None
    assert qwen.predictive_kv_reason is not None
    assert "not modeled" in qwen.predictive_kv_reason

    assert laguna.status == "ready"
    assert laguna.dense_gqa_kv_input is None
    assert laguna.predictive_kv_reason is not None
    assert "not modeled" in laguna.predictive_kv_reason


def test_laguna_manual_32k_policy_does_not_rewrite_provider_context_or_kv_window():
    adapted = adapt_model_config(_config("laguna-s-2-1.json"))
    assert adapted.context_limits is not None

    recommended = select_serving_context(
        replace(adapted.context_limits, policy_context_tokens=32_768)
    )

    assert recommended.serving_context_tokens == 32_768
    assert recommended.advertised_context_tokens == 1_048_576
    assert recommended.kv_window_tokens == 512
    assert recommended.output_token_cap is None


def test_portable_model_fixtures_contain_no_absolute_local_paths():
    payloads = (
        _fixture("dense-gqa.json"),
        _fixture("qwen3-5-hybrid.json"),
        _fixture("laguna-s-2-1.json"),
        json.loads(
            (
                Path(__file__).parent
                / "fixtures"
                / "model_fit"
                / "laguna-s-2-1-artifact.json"
            ).read_text()
        ),
    )

    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from strings(key)
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    local_paths = [
        value
        for payload in payloads
        for value in strings(payload)
        if Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or urlsplit(value).scheme == "file"
    ]
    assert local_paths == []
