# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for pure model-fit calculations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from vllm_mlx.hardware import HardwareInventory, SourceRecord
from vllm_mlx.model_fit import (
    ArtifactFile,
    ArtifactResidencyInput,
    ContextLimitsInput,
    ConversionWorkspaceInput,
    DenseGqaKvInput,
    MeasuredPeakMemory,
    MemoryMarginInput,
    estimate_artifact_residency,
    estimate_conversion_workspace,
    estimate_dense_gqa_kv_cache,
    estimate_memory_margin,
    select_serving_context,
)


def hardware_with_memory(total_bytes: int | None) -> HardwareInventory:
    return HardwareInventory(
        operating_system="Darwin",
        architecture="arm64",
        is_apple_silicon=True,
        chip_string="Apple fixture",
        machine_model=None,
        machine_name=None,
        total_unified_memory_bytes=total_bytes,
        reported_cpu_configuration=None,
        logical_cpu_cores=None,
        physical_cpu_cores=None,
        gpu_core_count=None,
        macos_version=None,
        macos_build=None,
        sources=(
            SourceRecord(
                field="total_unified_memory_bytes",
                source="fixture",
                locator="tests.hardware_with_memory",
                status="reported" if total_bytes is not None else "unknown",
                value=total_bytes,
            ),
        ),
    )


def test_laguna_artifact_facts_sum_exact_weight_bytes_without_architecture_guessing():
    # Laguna's indexed tensor bytes are an artifact fact. Its hybrid/recurrent
    # layout is intentionally not passed to the generic dense/GQA estimator.
    result = estimate_artifact_residency(
        ArtifactResidencyInput(
            weight_files=(
                ArtifactFile("model-00001.safetensors", 33_000_000_000),
                ArtifactFile("model-00002.safetensors", 33_147_556_864),
            )
        )
    )

    assert result.status == "exact"
    assert result.artifact_bytes == 66_147_556_864
    assert [component.bytes for component in result.components] == [
        33_000_000_000,
        33_147_556_864,
    ]


def test_artifact_residency_rejects_missing_invalid_or_duplicate_inventory_entries():
    empty = estimate_artifact_residency(ArtifactResidencyInput(weight_files=()))
    invalid = estimate_artifact_residency(
        ArtifactResidencyInput(
            weight_files=(
                ArtifactFile("weights.safetensors", -1),
                ArtifactFile("weights.safetensors", 1),
            )
        )
    )

    assert empty.status == "unknown"
    assert "no exact weight-file inventory" in empty.reasons[0]
    assert invalid.status == "unknown"
    assert any("invalid byte size" in reason for reason in invalid.reasons)
    assert any("duplicate" in reason for reason in invalid.reasons)


def test_dense_gqa_kv_formula_and_explicit_quantization_overhead():
    result = estimate_dense_gqa_kv_cache(
        DenseGqaKvInput(
            architecture_kind="gqa",
            layer_count=2,
            kv_head_count=4,
            head_dimension=8,
            element_bytes=2,
            context_tokens=16,
            concurrency=3,
            quantization_overhead_bytes=96,
        )
    )

    assert result.status == "estimated"
    assert result.bytes_per_token == 2 * 2 * 4 * 8 * 2
    assert result.components[0].bytes == result.bytes_per_token * 16 * 3
    assert result.components[1].name == "quantization_packing_overhead_bytes"
    assert result.components[1].bytes == 96
    assert result.total_bytes == result.components[0].bytes + 96


def test_kv_estimate_returns_unknown_for_missing_invalid_and_unsupported_architectures():
    missing = estimate_dense_gqa_kv_cache(
        DenseGqaKvInput("dense", 1, None, 8, 2, 16, 1)
    )
    unsupported = estimate_dense_gqa_kv_cache(
        DenseGqaKvInput("hybrid_recurrent", 1, 1, 8, 2, 16, 1)
    )

    assert missing.status == "unknown"
    assert "kv_head_count" in missing.reasons[0]
    assert unsupported.status == "unknown"
    assert "does not support architecture" in unsupported.reasons[0]


def test_context_selection_keeps_context_output_and_kv_window_distinct():
    result = select_serving_context(
        ContextLimitsInput(
            advertised_context_tokens=262_144,
            architecture_context_tokens=131_072,
            policy_context_tokens=65_536,
            output_token_cap=16_384,
            kv_window_tokens=4_096,
        )
    )

    assert result.status == "estimated"
    assert result.serving_context_tokens == 65_536
    assert result.output_token_cap == 16_384
    assert result.kv_window_tokens == 4_096
    assert "do not select serving context" in result.assumptions[1]


def test_context_selection_requires_a_valid_context_candidate():
    result = select_serving_context(ContextLimitsInput(None, None, None, 1024, 4096))
    invalid = select_serving_context(ContextLimitsInput(0, None, None, None, None))

    assert result.status == "unknown"
    assert "no advertised" in result.reasons[0]
    assert invalid.status == "unknown"
    assert "positive integer" in invalid.reasons[0]


def test_context_selection_rejects_policy_only_cap_as_model_capability():
    result = select_serving_context(ContextLimitsInput(None, None, 4096, 1024, None))

    assert result.status == "unknown"
    assert "policy alone cannot establish capability" in result.reasons[0]


def test_conversion_workspace_is_explicit_component_sum_without_multiplier():
    result = estimate_conversion_workspace(
        ConversionWorkspaceInput(
            source_artifact_bytes=100,
            expected_output_bytes=80,
            temporary_workspace_bytes=20,
            manifest_bytes=3,
            reserve_bytes=7,
        )
    )

    assert result.status == "estimated"
    assert result.required_bytes == 210
    assert [component.name for component in result.components] == [
        "source_artifact_bytes",
        "expected_output_bytes",
        "temporary_workspace_bytes",
        "manifest_bytes",
        "reserve_bytes",
    ]


def test_conversion_workspace_returns_unknown_for_missing_component():
    result = estimate_conversion_workspace(
        ConversionWorkspaceInput(100, None, 20, 3, 7)
    )

    assert result.status == "unknown"
    assert result.required_bytes is None
    assert result.reasons == ("expected_output_bytes must be a non-negative integer",)


def test_memory_margin_reports_positive_and_negative_derived_values():
    positive = estimate_memory_margin(
        MemoryMarginInput(
            hardware=hardware_with_memory(1_000),
            weight_residency_bytes=300,
            kv_cache_bytes=100,
            system_reserve_bytes=100,
            runtime_overhead_bytes=50,
            temporary_buffers_bytes=50,
        )
    )
    negative = estimate_memory_margin(
        MemoryMarginInput(
            hardware=hardware_with_memory(1_000),
            weight_residency_bytes=900,
            kv_cache_bytes=100,
            system_reserve_bytes=100,
            runtime_overhead_bytes=50,
            temporary_buffers_bytes=50,
        )
    )

    assert positive.status == "estimated"
    assert positive.usable_memory_bytes == 900
    assert positive.peak_bytes == 500
    assert positive.margin_bytes == 400
    assert negative.status == "estimated"
    assert negative.margin_bytes == -200


def test_measured_peak_overrides_selected_margin_and_preserves_derived_comparison():
    result = estimate_memory_margin(
        MemoryMarginInput(
            hardware=hardware_with_memory(1_000),
            weight_residency_bytes=300,
            kv_cache_bytes=100,
            system_reserve_bytes=100,
            runtime_overhead_bytes=50,
            temporary_buffers_bytes=50,
            measured_peak=MeasuredPeakMemory(650, "run/fixture/peak.json"),
        )
    )

    assert result.status == "measured"
    assert result.derived_peak_bytes == 500
    assert result.derived_margin_bytes == 400
    assert result.measured_peak_bytes == 650
    assert result.measured_margin_bytes == 250
    assert result.measured_evidence_reference == "run/fixture/peak.json"
    assert result.hardware_memory_source is not None
    assert result.hardware_memory_source.locator == "tests.hardware_with_memory"
    assert result.peak_bytes == 650
    assert result.margin_bytes == 250
    assert result.components[-1].classification == "measured"


def test_memory_margin_returns_unknown_without_hardware_fact_or_explicit_inputs():
    missing_hardware = estimate_memory_margin(
        MemoryMarginInput(
            hardware=hardware_with_memory(None),
            weight_residency_bytes=1,
            kv_cache_bytes=1,
            system_reserve_bytes=1,
            runtime_overhead_bytes=1,
            temporary_buffers_bytes=1,
        )
    )
    invalid_measurement = estimate_memory_margin(
        MemoryMarginInput(
            hardware=hardware_with_memory(100),
            weight_residency_bytes=1,
            kv_cache_bytes=1,
            system_reserve_bytes=1,
            runtime_overhead_bytes=1,
            temporary_buffers_bytes=1,
            measured_peak=MeasuredPeakMemory(-1, ""),
        )
    )

    assert missing_hardware.status == "unknown"
    assert "HardwareInventory" in missing_hardware.reasons[0]
    assert invalid_measurement.status == "unknown"
    assert len(invalid_measurement.reasons) == 2


def test_measured_peak_requires_a_nonempty_string_evidence_reference():
    def estimate(reference: Any):
        return estimate_memory_margin(
            MemoryMarginInput(
                hardware=hardware_with_memory(100),
                weight_residency_bytes=1,
                kv_cache_bytes=1,
                system_reserve_bytes=1,
                runtime_overhead_bytes=1,
                temporary_buffers_bytes=1,
                measured_peak=MeasuredPeakMemory(5, cast(str, reference)),
            )
        )

    assert estimate("   ").status == "unknown"
    assert estimate(42).status == "unknown"


def test_boolean_artifact_size_is_rejected():
    assert (
        estimate_artifact_residency(
            ArtifactResidencyInput((ArtifactFile("weights.safetensors", True),))
        ).status
        == "unknown"
    )


@pytest.mark.parametrize(
    "field",
    (
        "layer_count",
        "kv_head_count",
        "head_dimension",
        "element_bytes",
        "context_tokens",
        "concurrency",
        "quantization_overhead_bytes",
    ),
)
def test_boolean_kv_inputs_are_rejected(field):
    value = DenseGqaKvInput("dense", 1, 1, 1, 1, 1, 1)

    assert (
        estimate_dense_gqa_kv_cache(replace(value, **{field: True})).status == "unknown"
    )


@pytest.mark.parametrize(
    "field",
    (
        "advertised_context_tokens",
        "architecture_context_tokens",
        "policy_context_tokens",
        "output_token_cap",
        "kv_window_tokens",
    ),
)
def test_boolean_context_inputs_are_rejected(field):
    value = ContextLimitsInput(8192, 8192, 4096, 1024, 4096)

    assert select_serving_context(replace(value, **{field: True})).status == "unknown"


@pytest.mark.parametrize(
    "field",
    (
        "source_artifact_bytes",
        "expected_output_bytes",
        "temporary_workspace_bytes",
        "manifest_bytes",
        "reserve_bytes",
    ),
)
def test_boolean_conversion_inputs_are_rejected(field):
    value = ConversionWorkspaceInput(1, 1, 1, 1, 1)

    assert (
        estimate_conversion_workspace(replace(value, **{field: True})).status
        == "unknown"
    )


@pytest.mark.parametrize(
    "field",
    (
        "weight_residency_bytes",
        "kv_cache_bytes",
        "system_reserve_bytes",
        "runtime_overhead_bytes",
        "temporary_buffers_bytes",
    ),
)
def test_boolean_memory_component_inputs_are_rejected(field):
    value = MemoryMarginInput(hardware_with_memory(100), 1, 1, 1, 1, 1)

    assert estimate_memory_margin(replace(value, **{field: True})).status == "unknown"


def test_boolean_measured_peak_and_hardware_provenance_are_rejected():
    base = MemoryMarginInput(hardware_with_memory(100), 1, 1, 1, 1, 1)
    boolean_total = replace(
        base,
        hardware=replace(
            base.hardware,
            total_unified_memory_bytes=True,
            sources=(
                SourceRecord(
                    "total_unified_memory_bytes",
                    "fixture",
                    "boolean-total",
                    "reported",
                    True,
                ),
            ),
        ),
    )
    boolean_source = replace(
        base,
        hardware=replace(
            base.hardware,
            total_unified_memory_bytes=1,
            sources=(
                SourceRecord(
                    "total_unified_memory_bytes",
                    "fixture",
                    "boolean-source",
                    "reported",
                    True,
                ),
            ),
        ),
    )

    assert (
        estimate_memory_margin(
            replace(base, measured_peak=MeasuredPeakMemory(True, "run/fixture.json"))
        ).status
        == "unknown"
    )
    assert estimate_memory_margin(boolean_total).status == "unknown"
    assert estimate_memory_margin(boolean_source).status == "unknown"
