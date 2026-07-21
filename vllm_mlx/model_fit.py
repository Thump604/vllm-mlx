# SPDX-License-Identifier: Apache-2.0
"""Pure, explainable model-fit calculations.

This module deliberately has no model, registry, scheduler, cache, CLI, or
serving dependencies.  It calculates only from explicit caller-provided facts
and assumptions.  Unsupported architectures and incomplete inputs are returned
as ``unknown`` results instead of being converted into a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vllm_mlx.hardware import HardwareInventory, SourceRecord

EstimateStatus = Literal["exact", "estimated", "measured", "unknown"]
ComponentClass = Literal["provider_fact", "derived", "assumption", "measured"]


@dataclass(frozen=True)
class ByteComponent:
    """One named byte value used by an estimate."""

    name: str
    bytes: int
    classification: ComponentClass


@dataclass(frozen=True)
class TokenComponent:
    """One named token limit kept separate from the other context limits."""

    name: str
    tokens: int
    classification: ComponentClass


@dataclass(frozen=True)
class ArtifactFile:
    """An exact recognized model-weight artifact, with no inferred size."""

    name: str
    size_bytes: int


@dataclass(frozen=True)
class ArtifactResidencyInput:
    """Exact weight-file inventory used for an artifact residency calculation."""

    weight_files: tuple[ArtifactFile, ...]


@dataclass(frozen=True)
class ArtifactResidencyEstimate:
    """Exact byte sum for known model-weight artifacts."""

    status: EstimateStatus
    artifact_bytes: int | None
    components: tuple[ByteComponent, ...]
    assumptions: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DenseGqaKvInput:
    """Complete dense/GQA KV-cache inputs.

    ``quantization_overhead_bytes`` is an explicit total for scales, biases,
    packing, or group metadata.  It is not inferred from a quantization label.
    """

    architecture_kind: str
    layer_count: int | None
    kv_head_count: int | None
    head_dimension: int | None
    element_bytes: int | None
    context_tokens: int | None
    concurrency: int | None
    quantization_overhead_bytes: int | None = 0


@dataclass(frozen=True)
class KvCacheEstimate:
    """Dense/GQA KV-cache calculation with explicit base and packing values."""

    status: EstimateStatus
    total_bytes: int | None
    bytes_per_token: int | None
    components: tuple[ByteComponent, ...]
    assumptions: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ContextLimitsInput:
    """Independent context and output bounds supplied by provider and policy."""

    advertised_context_tokens: int | None
    architecture_context_tokens: int | None
    policy_context_tokens: int | None
    output_token_cap: int | None
    kv_window_tokens: int | None


@dataclass(frozen=True)
class ContextSelectionEstimate:
    """Selected serving context without conflating output or KV-window limits."""

    status: EstimateStatus
    advertised_context_tokens: int | None
    architecture_context_tokens: int | None
    policy_context_tokens: int | None
    serving_context_tokens: int | None
    output_token_cap: int | None
    kv_window_tokens: int | None
    components: tuple[TokenComponent, ...]
    assumptions: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ConversionWorkspaceInput:
    """Explicit disk components for a conversion operation."""

    source_artifact_bytes: int | None
    expected_output_bytes: int | None
    temporary_workspace_bytes: int | None
    manifest_bytes: int | None
    reserve_bytes: int | None


@dataclass(frozen=True)
class ConversionWorkspaceEstimate:
    """Required conversion disk from an exact component sum."""

    status: EstimateStatus
    required_bytes: int | None
    components: tuple[ByteComponent, ...]
    assumptions: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MeasuredPeakMemory:
    """A separately collected peak-memory observation.

    The value overrides the selected margin while the derived result remains in
    the output for direct comparison.
    """

    peak_bytes: int
    evidence_reference: str


@dataclass(frozen=True)
class MemoryMarginInput:
    """Inputs for a memory-margin calculation over a hardware inventory."""

    hardware: HardwareInventory
    weight_residency_bytes: int | None
    kv_cache_bytes: int | None
    system_reserve_bytes: int | None
    runtime_overhead_bytes: int | None
    temporary_buffers_bytes: int | None
    measured_peak: MeasuredPeakMemory | None = None


@dataclass(frozen=True)
class MemoryMarginEstimate:
    """Derived and optionally measured peak-memory and margin values."""

    status: EstimateStatus
    usable_memory_bytes: int | None
    peak_bytes: int | None
    margin_bytes: int | None
    derived_peak_bytes: int | None
    derived_margin_bytes: int | None
    measured_peak_bytes: int | None
    measured_margin_bytes: int | None
    measured_evidence_reference: str | None
    hardware_memory_source: SourceRecord | None
    components: tuple[ByteComponent, ...]
    assumptions: tuple[str, ...]
    reasons: tuple[str, ...]


def estimate_artifact_residency(
    value: ArtifactResidencyInput,
) -> ArtifactResidencyEstimate:
    """Return the exact sum of supplied weight artifacts, never parameters."""
    reasons: list[str] = []
    if not value.weight_files:
        reasons.append("no exact weight-file inventory was supplied")

    names: set[str] = set()
    components: list[ByteComponent] = []
    for artifact in value.weight_files:
        if not isinstance(artifact.name, str) or not artifact.name:
            reasons.append("weight-file inventory contains an invalid file name")
            continue
        if artifact.name in names:
            reasons.append(
                f"weight-file inventory contains duplicate {artifact.name!r}"
            )
            continue
        names.add(artifact.name)
        if not _is_nonnegative_integer(artifact.size_bytes):
            reasons.append(f"weight-file {artifact.name!r} has an invalid byte size")
            continue
        components.append(
            ByteComponent(artifact.name, artifact.size_bytes, "provider_fact")
        )

    if reasons:
        return ArtifactResidencyEstimate(
            status="unknown",
            artifact_bytes=None,
            components=tuple(components),
            assumptions=(),
            reasons=tuple(reasons),
        )
    return ArtifactResidencyEstimate(
        status="exact",
        artifact_bytes=sum(component.bytes for component in components),
        components=tuple(components),
        assumptions=(),
        reasons=(),
    )


def estimate_dense_gqa_kv_cache(value: DenseGqaKvInput) -> KvCacheEstimate:
    """Estimate dense/GQA KV memory from complete, explicit architecture facts."""
    reasons: list[str] = []
    if value.architecture_kind not in {"dense", "gqa"}:
        reasons.append(
            f"generic KV estimation does not support architecture {value.architecture_kind!r}"
        )

    required = (
        ("layer_count", value.layer_count),
        ("kv_head_count", value.kv_head_count),
        ("head_dimension", value.head_dimension),
        ("element_bytes", value.element_bytes),
        ("context_tokens", value.context_tokens),
        ("concurrency", value.concurrency),
    )
    for name, raw in required:
        if not _is_positive_integer(raw):
            reasons.append(f"{name} must be a positive integer")
    if not _is_nonnegative_integer(value.quantization_overhead_bytes):
        reasons.append("quantization_overhead_bytes must be a non-negative integer")
    if reasons:
        return KvCacheEstimate(
            status="unknown",
            total_bytes=None,
            bytes_per_token=None,
            components=(),
            assumptions=(),
            reasons=tuple(reasons),
        )

    assert value.layer_count is not None
    assert value.kv_head_count is not None
    assert value.head_dimension is not None
    assert value.element_bytes is not None
    assert value.context_tokens is not None
    assert value.concurrency is not None
    assert value.quantization_overhead_bytes is not None
    bytes_per_token = (
        2
        * value.layer_count
        * value.kv_head_count
        * value.head_dimension
        * value.element_bytes
    )
    tensor_bytes = bytes_per_token * value.context_tokens * value.concurrency
    components = (
        ByteComponent("key_value_tensor_bytes", tensor_bytes, "derived"),
        ByteComponent(
            "quantization_packing_overhead_bytes",
            value.quantization_overhead_bytes,
            "provider_fact",
        ),
    )
    return KvCacheEstimate(
        status="estimated",
        total_bytes=sum(component.bytes for component in components),
        bytes_per_token=bytes_per_token,
        components=components,
        assumptions=(
            "dense/GQA KV uses two tensors per layer: key and value",
            "quantization packing overhead is caller-supplied rather than inferred",
        ),
        reasons=(),
    )


def select_serving_context(value: ContextLimitsInput) -> ContextSelectionEstimate:
    """Select an explicit serving context without treating output or KV as context."""
    limits: tuple[tuple[str, int | None, ComponentClass], ...] = (
        ("advertised_context_tokens", value.advertised_context_tokens, "provider_fact"),
        (
            "architecture_context_tokens",
            value.architecture_context_tokens,
            "provider_fact",
        ),
        ("policy_context_tokens", value.policy_context_tokens, "assumption"),
        ("output_token_cap", value.output_token_cap, "assumption"),
        ("kv_window_tokens", value.kv_window_tokens, "provider_fact"),
    )
    reasons: list[str] = []
    components: list[TokenComponent] = []
    for name, raw, classification in limits:
        if raw is None:
            continue
        if not _is_positive_integer(raw):
            reasons.append(f"{name} must be a positive integer when supplied")
            continue
        assert isinstance(raw, int) and not isinstance(raw, bool)
        components.append(TokenComponent(name, raw, classification))

    selected_candidates = (
        value.advertised_context_tokens,
        value.architecture_context_tokens,
        value.policy_context_tokens,
    )
    provider_candidates = (
        value.advertised_context_tokens,
        value.architecture_context_tokens,
    )
    if not any(candidate is not None for candidate in provider_candidates):
        reasons.append(
            "no advertised or architecture context fact was supplied; policy alone cannot establish capability"
        )
    if reasons:
        return ContextSelectionEstimate(
            status="unknown",
            advertised_context_tokens=value.advertised_context_tokens,
            architecture_context_tokens=value.architecture_context_tokens,
            policy_context_tokens=value.policy_context_tokens,
            serving_context_tokens=None,
            output_token_cap=value.output_token_cap,
            kv_window_tokens=value.kv_window_tokens,
            components=tuple(components),
            assumptions=(),
            reasons=tuple(reasons),
        )

    candidates = tuple(
        candidate for candidate in selected_candidates if candidate is not None
    )
    return ContextSelectionEstimate(
        status="estimated",
        advertised_context_tokens=value.advertised_context_tokens,
        architecture_context_tokens=value.architecture_context_tokens,
        policy_context_tokens=value.policy_context_tokens,
        serving_context_tokens=min(candidates),
        output_token_cap=value.output_token_cap,
        kv_window_tokens=value.kv_window_tokens,
        components=tuple(components),
        assumptions=(
            "serving context is the minimum of explicit advertised, architecture, and policy limits",
            "output_token_cap and kv_window_tokens do not select serving context",
        ),
        reasons=(),
    )


def estimate_conversion_workspace(
    value: ConversionWorkspaceInput,
) -> ConversionWorkspaceEstimate:
    """Return a conversion workspace requirement from explicit byte components."""
    named_values: tuple[tuple[str, int | None, ComponentClass], ...] = (
        ("source_artifact_bytes", value.source_artifact_bytes, "provider_fact"),
        ("expected_output_bytes", value.expected_output_bytes, "assumption"),
        ("temporary_workspace_bytes", value.temporary_workspace_bytes, "assumption"),
        ("manifest_bytes", value.manifest_bytes, "provider_fact"),
        ("reserve_bytes", value.reserve_bytes, "assumption"),
    )
    reasons: list[str] = []
    components: list[ByteComponent] = []
    for name, raw, classification in named_values:
        if not _is_nonnegative_integer(raw):
            reasons.append(f"{name} must be a non-negative integer")
            continue
        assert isinstance(raw, int) and not isinstance(raw, bool)
        components.append(ByteComponent(name, raw, classification))
    if reasons:
        return ConversionWorkspaceEstimate(
            status="unknown",
            required_bytes=None,
            components=tuple(components),
            assumptions=(),
            reasons=tuple(reasons),
        )
    return ConversionWorkspaceEstimate(
        status="estimated",
        required_bytes=sum(component.bytes for component in components),
        components=tuple(components),
        assumptions=(
            "conversion workspace is an explicit source + output + temporary + manifests + reserve sum",
        ),
        reasons=(),
    )


def estimate_memory_margin(value: MemoryMarginInput) -> MemoryMarginEstimate:
    """Calculate a derived margin and optionally select measured peak evidence."""
    reasons: list[str] = []
    total_memory = value.hardware.total_unified_memory_bytes
    if not _is_positive_integer(total_memory):
        reasons.append("HardwareInventory has no positive total_unified_memory_bytes")
    hardware_memory_source = next(
        (
            source
            for source in value.hardware.sources
            if source.field == "total_unified_memory_bytes"
            and source.status == "reported"
            and type(source.value) is int
            and source.value == total_memory
        ),
        None,
    )
    if total_memory is not None and hardware_memory_source is None:
        reasons.append(
            "HardwareInventory total_unified_memory_bytes has no matching reported source"
        )
    named_values: tuple[tuple[str, int | None, ComponentClass], ...] = (
        ("weight_residency_bytes", value.weight_residency_bytes, "provider_fact"),
        ("kv_cache_bytes", value.kv_cache_bytes, "derived"),
        ("system_reserve_bytes", value.system_reserve_bytes, "assumption"),
        ("runtime_overhead_bytes", value.runtime_overhead_bytes, "assumption"),
        ("temporary_buffers_bytes", value.temporary_buffers_bytes, "assumption"),
    )
    components: list[ByteComponent] = []
    for name, raw, classification in named_values:
        if not _is_nonnegative_integer(raw):
            reasons.append(f"{name} must be a non-negative integer")
            continue
        assert isinstance(raw, int) and not isinstance(raw, bool)
        components.append(ByteComponent(name, raw, classification))
    if value.measured_peak is not None:
        if not _is_nonnegative_integer(value.measured_peak.peak_bytes):
            reasons.append("measured peak_bytes must be a non-negative integer")
        if not isinstance(value.measured_peak.evidence_reference, str) or not (
            value.measured_peak.evidence_reference.strip()
        ):
            reasons.append("measured peak requires an evidence_reference")
    if reasons:
        return MemoryMarginEstimate(
            status="unknown",
            usable_memory_bytes=None,
            peak_bytes=None,
            margin_bytes=None,
            derived_peak_bytes=None,
            derived_margin_bytes=None,
            measured_peak_bytes=None,
            measured_margin_bytes=None,
            measured_evidence_reference=None,
            hardware_memory_source=hardware_memory_source,
            components=tuple(components),
            assumptions=(),
            reasons=tuple(reasons),
        )

    assert total_memory is not None
    assert value.weight_residency_bytes is not None
    assert value.kv_cache_bytes is not None
    assert value.system_reserve_bytes is not None
    assert value.runtime_overhead_bytes is not None
    assert value.temporary_buffers_bytes is not None
    components.insert(
        0, ByteComponent("total_unified_memory_bytes", total_memory, "provider_fact")
    )
    derived_peak = (
        value.weight_residency_bytes
        + value.kv_cache_bytes
        + value.runtime_overhead_bytes
        + value.temporary_buffers_bytes
    )
    usable_memory = total_memory - value.system_reserve_bytes
    derived_margin = usable_memory - derived_peak
    if value.measured_peak is None:
        return MemoryMarginEstimate(
            status="estimated",
            usable_memory_bytes=usable_memory,
            peak_bytes=derived_peak,
            margin_bytes=derived_margin,
            derived_peak_bytes=derived_peak,
            derived_margin_bytes=derived_margin,
            measured_peak_bytes=None,
            measured_margin_bytes=None,
            measured_evidence_reference=None,
            hardware_memory_source=hardware_memory_source,
            components=tuple(components),
            assumptions=(
                "system reserve, runtime overhead, and temporary buffers are explicit caller inputs",
            ),
            reasons=(),
        )

    measured_peak = value.measured_peak.peak_bytes
    measured_margin = usable_memory - measured_peak
    components.append(ByteComponent("measured_peak_bytes", measured_peak, "measured"))
    return MemoryMarginEstimate(
        status="measured",
        usable_memory_bytes=usable_memory,
        peak_bytes=measured_peak,
        margin_bytes=measured_margin,
        derived_peak_bytes=derived_peak,
        derived_margin_bytes=derived_margin,
        measured_peak_bytes=measured_peak,
        measured_margin_bytes=measured_margin,
        measured_evidence_reference=value.measured_peak.evidence_reference.strip(),
        hardware_memory_source=hardware_memory_source,
        components=tuple(components),
        assumptions=(
            "system reserve, runtime overhead, and temporary buffers are explicit caller inputs",
            "measured peak overrides the selected result while preserving the derived comparison",
        ),
        reasons=(),
    )


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
