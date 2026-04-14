# SPDX-License-Identifier: Apache-2.0
"""Runtime registry and state loaders for the Phase 1 contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

PriorityClass = Literal[
    "admin",
    "interactive",
    "discussion",
    "benchmark",
    "background",
]
ExecutionClass = Literal["draft_only", "shared_candidate", "solo_only"]
ArchitectureKind = Literal["dense", "open_moe"]

_PRIORITY_CLASSES = {"admin", "interactive", "discussion", "benchmark", "background"}
_EXECUTION_CLASSES = {"draft_only", "shared_candidate", "solo_only"}
_ARCHITECTURES = {"dense", "open_moe"}
_CONTENTION_STRATEGIES = {
    "fail",
    "wait",
    "preempt",
    "wait_then_fail",
    "wait_then_preempt",
}


def _expect_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _expect_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _expect_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, label)


def _optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _bool_or_default(value: Any, label: str, default: bool = False) -> bool:
    parsed = _optional_bool(value, label)
    return default if parsed is None else parsed


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _expect_priority(value: Any, label: str) -> PriorityClass:
    priority = _expect_str(value, label)
    if priority not in _PRIORITY_CLASSES:
        raise ValueError(f"{label} has unsupported priority class '{priority}'")
    return priority  # type: ignore[return-value]


def _expect_execution_class(value: Any, label: str) -> ExecutionClass:
    execution_class = _expect_str(value, label)
    if execution_class not in _EXECUTION_CLASSES:
        raise ValueError(f"{label} has unsupported execution class '{execution_class}'")
    return execution_class  # type: ignore[return-value]


def _expect_architecture(value: Any, label: str) -> ArchitectureKind:
    architecture = _expect_str(value, label)
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"{label} has unsupported architecture '{architecture}'")
    return architecture  # type: ignore[return-value]


@dataclass(frozen=True)
class ContentionPolicyConfig:
    strategy: str = "wait_then_fail"
    wait_timeout_s: float | None = 30.0
    preempt_after_s: float | None = None


@dataclass(frozen=True)
class PolicyDefaultsConfig:
    memory_budget_gb: float | None = None
    contention_policy: ContentionPolicyConfig = field(
        default_factory=ContentionPolicyConfig
    )


@dataclass(frozen=True)
class SpecPrefillConfig:
    enabled: bool = False
    threshold: int | None = None
    keep_pct: float | None = None


@dataclass(frozen=True)
class ServingProfileConfig:
    force_mllm: bool | None = None
    continuous_batching: bool | None = None
    prefill_step_size: int | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    enable_auto_tool_choice: bool | None = None
    enable_thinking_default: bool | None = None
    force_nonempty_content: bool | None = None
    specprefill: SpecPrefillConfig | None = None


@dataclass(frozen=True)
class DraftModelConfig:
    id: str
    source: str
    estimated_memory_gb: float | None = None


@dataclass(frozen=True)
class RegistryModelConfig:
    id: str
    display_name: str
    source: str
    family: str
    architecture: ArchitectureKind
    execution_class: ExecutionClass
    estimated_memory_gb: float
    serving_profile: ServingProfileConfig
    draft_model: DraftModelConfig | None = None
    multimodal: bool = False
    supports_mtp: bool = False
    supports_reasoning: bool = False
    supports_tools: bool = False
    supports_specprefill: bool = False
    supports_kv_quant: bool = False
    supports_continuous_batching: bool = False
    preferred_engine: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SamplingProfileConfig:
    temperature: float | None = None
    top_p: float | None = None
    enable_thinking: bool | None = None


@dataclass(frozen=True)
class RequestPolicyConfig:
    max_tokens: int | None = None
    timeout_s: float | None = None


@dataclass(frozen=True)
class ModelPresetConfig:
    id: str
    display_name: str
    model_id: str
    priority_class: PriorityClass
    performance_bias: str | None = None
    sampling_profile: SamplingProfileConfig = field(
        default_factory=SamplingProfileConfig
    )
    request_policy: RequestPolicyConfig = field(default_factory=RequestPolicyConfig)


@dataclass(frozen=True)
class ServicePresetConfig:
    id: str
    display_name: str
    services: dict[str, bool]
    mcp_bundles: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscussionProfileConfig:
    id: str
    display_name: str
    concurrent_pool: tuple[str, ...]
    solo_pool: tuple[str, ...]
    human_seats: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegistryDocument:
    schema_version: int
    policy_defaults: PolicyDefaultsConfig
    models: dict[str, RegistryModelConfig]
    model_presets: dict[str, ModelPresetConfig]
    service_presets: dict[str, ServicePresetConfig]
    discussion_profiles: dict[str, DiscussionProfileConfig]


@dataclass(frozen=True)
class RuntimeExecutionState:
    active_model: str | None = None
    queue_policy: str | None = None


@dataclass(frozen=True)
class RuntimeState:
    schema_version: int
    active_model_preset: str | None = None
    active_service_presets: tuple[str, ...] = ()
    active_backend: str | None = None
    resident_models: tuple[str, ...] = ()
    execution: RuntimeExecutionState = field(default_factory=RuntimeExecutionState)
    updated_at: str | None = None
    updated_by: str | None = None


@dataclass(frozen=True)
class ResolvedRuntimeState:
    model_preset: ModelPresetConfig | None
    model: RegistryModelConfig | None
    service_presets: tuple[ServicePresetConfig, ...]


def _parse_policy_defaults(raw: Any) -> PolicyDefaultsConfig:
    if raw is None:
        return PolicyDefaultsConfig()
    mapping = _expect_mapping(raw, "policy_defaults")
    contention_raw = mapping.get("contention_policy") or {}
    contention = _expect_mapping(contention_raw, "policy_defaults.contention_policy")
    strategy = (
        _optional_str(
            contention.get("strategy"), "policy_defaults.contention_policy.strategy"
        )
        or "wait_then_fail"
    )
    if strategy not in _CONTENTION_STRATEGIES:
        raise ValueError(
            "policy_defaults.contention_policy.strategy has unsupported value "
            f"'{strategy}'"
        )
    return PolicyDefaultsConfig(
        memory_budget_gb=_optional_float(
            mapping.get("memory_budget_gb"), "policy_defaults.memory_budget_gb"
        ),
        contention_policy=ContentionPolicyConfig(
            strategy=strategy,
            wait_timeout_s=_optional_float(
                contention.get("wait_timeout_s"),
                "policy_defaults.contention_policy.wait_timeout_s",
            ),
            preempt_after_s=_optional_float(
                contention.get("preempt_after_s"),
                "policy_defaults.contention_policy.preempt_after_s",
            ),
        ),
    )


def _parse_specprefill(raw: Any, label: str) -> SpecPrefillConfig | None:
    if raw is None:
        return None
    mapping = _expect_mapping(raw, label)
    enabled = mapping.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"{label}.enabled must be a boolean")
    return SpecPrefillConfig(
        enabled=enabled,
        threshold=_optional_int(mapping.get("threshold"), f"{label}.threshold"),
        keep_pct=_optional_float(mapping.get("keep_pct"), f"{label}.keep_pct"),
    )


def _parse_serving_profile(raw: Any, label: str) -> ServingProfileConfig:
    mapping = _expect_mapping(raw, label)
    return ServingProfileConfig(
        force_mllm=_optional_bool(mapping.get("force_mllm"), f"{label}.force_mllm"),
        continuous_batching=_optional_bool(
            mapping.get("continuous_batching"), f"{label}.continuous_batching"
        ),
        prefill_step_size=_optional_int(
            mapping.get("prefill_step_size"), f"{label}.prefill_step_size"
        ),
        tool_call_parser=_optional_str(
            mapping.get("tool_call_parser"), f"{label}.tool_call_parser"
        ),
        reasoning_parser=_optional_str(
            mapping.get("reasoning_parser"), f"{label}.reasoning_parser"
        ),
        enable_auto_tool_choice=_optional_bool(
            mapping.get("enable_auto_tool_choice"),
            f"{label}.enable_auto_tool_choice",
        ),
        enable_thinking_default=_optional_bool(
            mapping.get("enable_thinking_default"),
            f"{label}.enable_thinking_default",
        ),
        force_nonempty_content=_optional_bool(
            mapping.get("force_nonempty_content"),
            f"{label}.force_nonempty_content",
        ),
        specprefill=_parse_specprefill(
            mapping.get("specprefill"), f"{label}.specprefill"
        ),
    )


def _parse_draft_model(raw: Any, label: str) -> DraftModelConfig | None:
    if raw is None:
        return None
    mapping = _expect_mapping(raw, label)
    return DraftModelConfig(
        id=_expect_str(mapping.get("id"), f"{label}.id"),
        source=_expect_str(mapping.get("source"), f"{label}.source"),
        estimated_memory_gb=_optional_float(
            mapping.get("estimated_memory_gb"), f"{label}.estimated_memory_gb"
        ),
    )


def _parse_model_entry(raw: Any, index: int) -> RegistryModelConfig:
    label = f"models[{index}]"
    mapping = _expect_mapping(raw, label)
    estimated_memory_gb = _optional_float(
        mapping.get("estimated_memory_gb"), f"{label}.estimated_memory_gb"
    )
    if estimated_memory_gb is None or estimated_memory_gb <= 0:
        raise ValueError(f"{label}.estimated_memory_gb must be a positive number")
    if "serving_profile" not in mapping:
        raise ValueError(f"{label}.serving_profile is required")
    serving_profile = _parse_serving_profile(
        mapping.get("serving_profile"), f"{label}.serving_profile"
    )
    return RegistryModelConfig(
        id=_expect_str(mapping.get("id"), f"{label}.id"),
        display_name=_expect_str(mapping.get("display_name"), f"{label}.display_name"),
        source=_expect_str(mapping.get("source"), f"{label}.source"),
        family=_expect_str(mapping.get("family"), f"{label}.family"),
        architecture=_expect_architecture(
            mapping.get("architecture"), f"{label}.architecture"
        ),
        execution_class=_expect_execution_class(
            mapping.get("execution_class"), f"{label}.execution_class"
        ),
        estimated_memory_gb=estimated_memory_gb,
        serving_profile=serving_profile,
        draft_model=_parse_draft_model(
            mapping.get("draft_model"), f"{label}.draft_model"
        ),
        multimodal=_bool_or_default(mapping.get("multimodal"), f"{label}.multimodal"),
        supports_mtp=_bool_or_default(
            mapping.get("supports_mtp"), f"{label}.supports_mtp"
        ),
        supports_reasoning=_bool_or_default(
            mapping.get("supports_reasoning"), f"{label}.supports_reasoning"
        ),
        supports_tools=_bool_or_default(
            mapping.get("supports_tools"), f"{label}.supports_tools"
        ),
        supports_specprefill=_bool_or_default(
            mapping.get("supports_specprefill"), f"{label}.supports_specprefill"
        ),
        supports_kv_quant=_bool_or_default(
            mapping.get("supports_kv_quant"), f"{label}.supports_kv_quant"
        ),
        supports_continuous_batching=_bool_or_default(
            mapping.get("supports_continuous_batching"),
            f"{label}.supports_continuous_batching",
        ),
        preferred_engine=_optional_str(
            mapping.get("preferred_engine"), f"{label}.preferred_engine"
        ),
        notes=_optional_str(mapping.get("notes"), f"{label}.notes"),
    )


def _parse_sampling_profile(raw: Any, label: str) -> SamplingProfileConfig:
    mapping = _expect_mapping(raw or {}, label)
    return SamplingProfileConfig(
        temperature=_optional_float(mapping.get("temperature"), f"{label}.temperature"),
        top_p=_optional_float(mapping.get("top_p"), f"{label}.top_p"),
        enable_thinking=_optional_bool(
            mapping.get("enable_thinking"), f"{label}.enable_thinking"
        ),
    )


def _parse_request_policy(raw: Any, label: str) -> RequestPolicyConfig:
    mapping = _expect_mapping(raw or {}, label)
    return RequestPolicyConfig(
        max_tokens=_optional_int(mapping.get("max_tokens"), f"{label}.max_tokens"),
        timeout_s=_optional_float(mapping.get("timeout_s"), f"{label}.timeout_s"),
    )


def _parse_model_preset(raw: Any, index: int) -> ModelPresetConfig:
    label = f"model_presets[{index}]"
    mapping = _expect_mapping(raw, label)
    return ModelPresetConfig(
        id=_expect_str(mapping.get("id"), f"{label}.id"),
        display_name=_expect_str(mapping.get("display_name"), f"{label}.display_name"),
        model_id=_expect_str(mapping.get("model_id"), f"{label}.model_id"),
        priority_class=_expect_priority(
            mapping.get("priority_class"), f"{label}.priority_class"
        ),
        performance_bias=_optional_str(
            mapping.get("performance_bias"), f"{label}.performance_bias"
        ),
        sampling_profile=_parse_sampling_profile(
            mapping.get("sampling_profile"), f"{label}.sampling_profile"
        ),
        request_policy=_parse_request_policy(
            mapping.get("request_policy"), f"{label}.request_policy"
        ),
    )


def _parse_service_preset(raw: Any, index: int) -> ServicePresetConfig:
    label = f"service_presets[{index}]"
    mapping = _expect_mapping(raw, label)
    services_raw = _expect_mapping(mapping.get("services") or {}, f"{label}.services")
    services = {}
    for name, enabled in services_raw.items():
        services[_expect_str(name, f"{label}.services key")] = _bool_or_default(
            enabled, f"{label}.services[{name}]"
        )
    bundles = tuple(
        _expect_str(item, f"{label}.mcp_bundles[{idx}]")
        for idx, item in enumerate(
            _expect_list(mapping.get("mcp_bundles") or [], f"{label}.mcp_bundles")
        )
    )
    return ServicePresetConfig(
        id=_expect_str(mapping.get("id"), f"{label}.id"),
        display_name=_expect_str(mapping.get("display_name"), f"{label}.display_name"),
        services=services,
        mcp_bundles=bundles,
    )


def _parse_discussion_profile(raw: Any, index: int) -> DiscussionProfileConfig:
    label = f"discussion_profiles[{index}]"
    mapping = _expect_mapping(raw, label)
    return DiscussionProfileConfig(
        id=_expect_str(mapping.get("id"), f"{label}.id"),
        display_name=_expect_str(mapping.get("display_name"), f"{label}.display_name"),
        concurrent_pool=tuple(
            _expect_str(item, f"{label}.concurrent_pool[{idx}]")
            for idx, item in enumerate(
                _expect_list(mapping.get("concurrent_pool"), f"{label}.concurrent_pool")
            )
        ),
        solo_pool=tuple(
            _expect_str(item, f"{label}.solo_pool[{idx}]")
            for idx, item in enumerate(
                _expect_list(mapping.get("solo_pool"), f"{label}.solo_pool")
            )
        ),
        human_seats=tuple(
            _expect_str(item, f"{label}.human_seats[{idx}]")
            for idx, item in enumerate(
                _expect_list(mapping.get("human_seats") or [], f"{label}.human_seats")
            )
        ),
    )


def _check_unique(section: str, values: dict[str, Any], item_id: str) -> None:
    if item_id in values:
        raise ValueError(f"Duplicate {section} id '{item_id}'")


def _validate_registry_document(document: RegistryDocument) -> None:
    for preset in document.model_presets.values():
        if preset.model_id not in document.models:
            raise ValueError(
                f"model_presets[{preset.id}] references unknown model '{preset.model_id}'"
            )

    for profile in document.discussion_profiles.values():
        for model_id in profile.concurrent_pool:
            model = document.models.get(model_id)
            if model is None:
                raise ValueError(
                    f"discussion_profiles[{profile.id}] references unknown model '{model_id}'"
                )
            if model.execution_class != "shared_candidate":
                raise ValueError(
                    f"discussion_profiles[{profile.id}] concurrent_pool requires "
                    f"shared_candidate models, got '{model.execution_class}' for "
                    f"'{model_id}'"
                )
        for model_id in profile.solo_pool:
            model = document.models.get(model_id)
            if model is None:
                raise ValueError(
                    f"discussion_profiles[{profile.id}] references unknown model '{model_id}'"
                )
            if model.execution_class != "solo_only":
                raise ValueError(
                    f"discussion_profiles[{profile.id}] solo_pool requires solo_only "
                    f"models, got '{model.execution_class}' for '{model_id}'"
                )


def load_registry_document(config_path: str | Path) -> RegistryDocument:
    """Load the approved registry.yaml contract document."""
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    mapping = _expect_mapping(raw, "registry.yaml")
    schema_version = mapping.get("schema_version")
    if schema_version != 1:
        raise ValueError("registry.yaml schema_version must be 1")

    models_raw = _expect_list(mapping.get("models"), "models")
    model_presets_raw = _expect_list(mapping.get("model_presets"), "model_presets")
    service_presets_raw = _expect_list(
        mapping.get("service_presets") or [], "service_presets"
    )
    discussion_raw = _expect_list(
        mapping.get("discussion_profiles") or [], "discussion_profiles"
    )

    models: dict[str, RegistryModelConfig] = {}
    for index, item in enumerate(models_raw):
        model = _parse_model_entry(item, index)
        _check_unique("models", models, model.id)
        models[model.id] = model

    model_presets: dict[str, ModelPresetConfig] = {}
    for index, item in enumerate(model_presets_raw):
        preset = _parse_model_preset(item, index)
        _check_unique("model_presets", model_presets, preset.id)
        model_presets[preset.id] = preset

    service_presets: dict[str, ServicePresetConfig] = {}
    for index, item in enumerate(service_presets_raw):
        preset = _parse_service_preset(item, index)
        _check_unique("service_presets", service_presets, preset.id)
        service_presets[preset.id] = preset

    discussion_profiles: dict[str, DiscussionProfileConfig] = {}
    for index, item in enumerate(discussion_raw):
        profile = _parse_discussion_profile(item, index)
        _check_unique("discussion_profiles", discussion_profiles, profile.id)
        discussion_profiles[profile.id] = profile

    document = RegistryDocument(
        schema_version=1,
        policy_defaults=_parse_policy_defaults(mapping.get("policy_defaults")),
        models=models,
        model_presets=model_presets,
        service_presets=service_presets,
        discussion_profiles=discussion_profiles,
    )
    _validate_registry_document(document)
    return document


def load_runtime_state(state_path: str | Path) -> RuntimeState:
    """Load the mutable runtime-state.json document."""
    raw = json.loads(Path(state_path).read_text())
    mapping = _expect_mapping(raw, "runtime-state.json")
    schema_version = mapping.get("schema_version")
    if schema_version != 1:
        raise ValueError("runtime-state.json schema_version must be 1")

    execution_raw = _expect_mapping(mapping.get("execution") or {}, "execution")
    return RuntimeState(
        schema_version=1,
        active_model_preset=_optional_str(
            mapping.get("active_model_preset"), "active_model_preset"
        ),
        active_service_presets=tuple(
            _expect_str(item, f"active_service_presets[{idx}]")
            for idx, item in enumerate(
                _expect_list(
                    mapping.get("active_service_presets") or [],
                    "active_service_presets",
                )
            )
        ),
        active_backend=_optional_str(mapping.get("active_backend"), "active_backend"),
        resident_models=tuple(
            _expect_str(item, f"resident_models[{idx}]")
            for idx, item in enumerate(
                _expect_list(mapping.get("resident_models") or [], "resident_models")
            )
        ),
        execution=RuntimeExecutionState(
            active_model=_optional_str(
                execution_raw.get("active_model"), "execution.active_model"
            ),
            queue_policy=_optional_str(
                execution_raw.get("queue_policy"), "execution.queue_policy"
            ),
        ),
        updated_at=_optional_str(mapping.get("updated_at"), "updated_at"),
        updated_by=_optional_str(mapping.get("updated_by"), "updated_by"),
    )


def resolve_runtime_state(
    registry: RegistryDocument, state: RuntimeState
) -> ResolvedRuntimeState:
    """Validate and resolve runtime-state.json against the registry document."""
    model_preset = None
    model = None
    if state.active_model_preset is not None:
        model_preset = registry.model_presets.get(state.active_model_preset)
        if model_preset is None:
            raise ValueError(
                "runtime-state.json active_model_preset references unknown "
                f"preset '{state.active_model_preset}'"
            )
        model = registry.models[model_preset.model_id]

    service_presets = []
    for preset_id in state.active_service_presets:
        preset = registry.service_presets.get(preset_id)
        if preset is None:
            raise ValueError(
                "runtime-state.json active_service_presets references unknown "
                f"preset '{preset_id}'"
            )
        service_presets.append(preset)

    for model_id in state.resident_models:
        if model_id not in registry.models:
            raise ValueError(
                f"runtime-state.json resident_models references unknown model '{model_id}'"
            )

    if state.execution.active_model is not None:
        if state.execution.active_model not in registry.models:
            raise ValueError(
                "runtime-state.json execution.active_model references unknown "
                f"model '{state.execution.active_model}'"
            )
        if (
            state.resident_models
            and state.execution.active_model not in state.resident_models
        ):
            raise ValueError(
                "runtime-state.json execution.active_model must appear in "
                "resident_models when resident_models is non-empty"
            )

    return ResolvedRuntimeState(
        model_preset=model_preset,
        model=model,
        service_presets=tuple(service_presets),
    )
