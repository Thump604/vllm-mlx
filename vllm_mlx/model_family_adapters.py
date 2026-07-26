# SPDX-License-Identifier: Apache-2.0
"""Pure, config-structure adapters for the first model-fit families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeGuard

from vllm_mlx.model_fit import ContextLimitsInput, DenseGqaKvInput

AdapterStatus = Literal["ready", "unknown"]
InputClassification = Literal["provider_fact", "derived", "caller_profile"]


@dataclass(frozen=True)
class ConfigFact:
    """One config fact with an RFC 6901 pointer into the model config."""

    name: str
    value: object
    pointer: str
    classification: Literal["provider_fact", "derived"] = "provider_fact"


@dataclass(frozen=True)
class CallerProfileInput:
    """A profile-selected value that is intentionally not a model config fact."""

    value: int
    reference: str


@dataclass(frozen=True)
class KvProfileInputs:
    """Explicit caller/profile inputs required for a generic KV estimate."""

    element_bytes: CallerProfileInput
    context_tokens: CallerProfileInput
    concurrency: CallerProfileInput
    quantization_overhead_bytes: CallerProfileInput


@dataclass(frozen=True)
class EstimatorInputProvenance:
    """Provenance for one non-null field in a generated estimator input."""

    field: str
    value: object
    classification: InputClassification
    references: tuple[str, ...]


@dataclass(frozen=True)
class ModelFamilyAdapterResult:
    """Config facts and optional generic estimator inputs, without policy."""

    adapter_id: str
    status: AdapterStatus
    provider_facts: tuple[ConfigFact, ...]
    context_limits: ContextLimitsInput | None
    dense_gqa_kv_input: DenseGqaKvInput | None
    input_provenance: tuple[EstimatorInputProvenance, ...]
    predictive_kv_reason: str | None
    reasons: tuple[str, ...]


_DENSE_GQA_MODEL_TYPES = frozenset({"llama", "mistral", "qwen2"})


def adapt_model_config(
    config: Mapping[str, object], *, kv_profile: KvProfileInputs | None = None
) -> ModelFamilyAdapterResult:
    """Adapt declared config structure without model-path or name inference."""
    if not isinstance(config, Mapping):
        return _unknown("unknown", "config must be a mapping")
    dense = _adapt_dense_gqa(config, kv_profile)
    if dense is not None:
        return dense
    return _unknown(
        "unknown",
        f"no P2.4 adapter supports declared model_type {config.get('model_type')!r}",
    )


def _adapt_dense_gqa(
    config: Mapping[str, object], kv_profile: KvProfileInputs | None
) -> ModelFamilyAdapterResult | None:
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or model_type not in _DENSE_GQA_MODEL_TYPES:
        return None
    if "text_config" in config:
        return _unknown(
            "dense_gqa",
            "dense/GQA adapter does not consume nested text_config wrappers",
        )
    if config.get("sliding_window") not in (None, 0):
        return _unknown(
            "dense_gqa", "dense/GQA adapter does not support sliding-window attention"
        )

    values, reasons = _positive_fields(
        config,
        "",
        (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        ),
    )
    if reasons:
        return _unknown("dense_gqa", *reasons)
    if values["num_attention_heads"] < values["num_key_value_heads"]:
        return _unknown(
            "dense_gqa", "/num_attention_heads must be >= /num_key_value_heads"
        )

    facts = (
        ConfigFact("model_type", model_type, "/model_type"),
        *(_config_fact(config, name) for name in values),
    )
    context, context_provenance = _context_input(
        values["max_position_embeddings"], "/max_position_embeddings", kv_profile
    )
    if kv_profile is None:
        return ModelFamilyAdapterResult(
            "dense_gqa",
            "ready",
            facts,
            context,
            None,
            context_provenance,
            "generic predictive KV requires explicit caller/profile KV inputs",
            (),
        )
    profile_reasons = _validate_kv_profile(kv_profile)
    if profile_reasons:
        return _unknown("dense_gqa", *profile_reasons)
    architecture_kind = (
        "dense"
        if values["num_attention_heads"] == values["num_key_value_heads"]
        else "gqa"
    )
    kv_input = DenseGqaKvInput(
        architecture_kind=architecture_kind,
        layer_count=values["num_hidden_layers"],
        kv_head_count=values["num_key_value_heads"],
        head_dimension=values["head_dim"],
        element_bytes=kv_profile.element_bytes.value,
        context_tokens=kv_profile.context_tokens.value,
        concurrency=kv_profile.concurrency.value,
        quantization_overhead_bytes=kv_profile.quantization_overhead_bytes.value,
    )
    kv_provenance = (
        EstimatorInputProvenance(
            "dense_gqa_kv_input.architecture_kind",
            architecture_kind,
            "derived",
            ("config:/num_attention_heads", "config:/num_key_value_heads"),
        ),
        EstimatorInputProvenance(
            "dense_gqa_kv_input.layer_count",
            values["num_hidden_layers"],
            "provider_fact",
            ("config:/num_hidden_layers",),
        ),
        EstimatorInputProvenance(
            "dense_gqa_kv_input.kv_head_count",
            values["num_key_value_heads"],
            "provider_fact",
            ("config:/num_key_value_heads",),
        ),
        EstimatorInputProvenance(
            "dense_gqa_kv_input.head_dimension",
            values["head_dim"],
            "provider_fact",
            ("config:/head_dim",),
        ),
        *_profile_provenance(kv_profile),
    )
    return ModelFamilyAdapterResult(
        "dense_gqa",
        "ready",
        facts,
        context,
        kv_input,
        context_provenance + kv_provenance,
        None,
        (),
    )


def _context_input(
    advertised: int,
    pointer: str,
    kv_profile: KvProfileInputs | None,
    *,
    kv_window_tokens: int | None = None,
) -> tuple[ContextLimitsInput, tuple[EstimatorInputProvenance, ...]]:
    provenance = [
        EstimatorInputProvenance(
            "context_limits.advertised_context_tokens",
            advertised,
            "provider_fact",
            (f"config:{pointer}",),
        )
    ]
    policy_context = None
    if kv_profile is not None:
        policy_context = kv_profile.context_tokens.value
        provenance.append(
            _caller_provenance(
                "context_limits.policy_context_tokens",
                kv_profile.context_tokens,
                "context_tokens",
            )
        )
    if kv_window_tokens is not None:
        provenance.append(
            EstimatorInputProvenance(
                "context_limits.kv_window_tokens",
                kv_window_tokens,
                "provider_fact",
                ("config:/sliding_window",),
            )
        )
    return ContextLimitsInput(
        advertised, None, policy_context, None, kv_window_tokens
    ), tuple(provenance)


def _profile_provenance(
    profile: KvProfileInputs,
) -> tuple[EstimatorInputProvenance, ...]:
    return (
        _caller_provenance(
            "dense_gqa_kv_input.element_bytes", profile.element_bytes, "element_bytes"
        ),
        _caller_provenance(
            "dense_gqa_kv_input.context_tokens",
            profile.context_tokens,
            "context_tokens",
        ),
        _caller_provenance(
            "dense_gqa_kv_input.concurrency", profile.concurrency, "concurrency"
        ),
        _caller_provenance(
            "dense_gqa_kv_input.quantization_overhead_bytes",
            profile.quantization_overhead_bytes,
            "quantization_overhead_bytes",
        ),
    )


def _caller_provenance(
    field: str, value: CallerProfileInput, name: str
) -> EstimatorInputProvenance:
    return EstimatorInputProvenance(
        field, value.value, "caller_profile", (f"profile:{name}", value.reference)
    )


def _validate_kv_profile(profile: KvProfileInputs) -> tuple[str, ...]:
    reasons: list[str] = []
    for name, item, positive in (
        ("element_bytes", profile.element_bytes, True),
        ("context_tokens", profile.context_tokens, True),
        ("concurrency", profile.concurrency, True),
        ("quantization_overhead_bytes", profile.quantization_overhead_bytes, False),
    ):
        valid = (
            _is_positive_integer(item.value)
            if positive
            else _is_nonnegative_integer(item.value)
        )
        if not valid:
            reasons.append(
                f"profile {name} must be a {'positive' if positive else 'non-negative'} integer"
            )
        if not isinstance(item.reference, str) or not item.reference.strip():
            reasons.append(f"profile {name} requires a non-empty provenance reference")
    return tuple(reasons)


def _positive_fields(
    source: Mapping[str, object], prefix: str, names: tuple[str, ...]
) -> tuple[dict[str, int], list[str]]:
    values: dict[str, int] = {}
    reasons: list[str] = []
    for name in names:
        value = source.get(name)
        if not _is_positive_integer(value):
            reasons.append(f"{_pointer(prefix, name)} must be a positive integer")
        else:
            values[name] = value
    return values, reasons


def _config_fact(
    source: Mapping[str, object], name: str, prefix: str = ""
) -> ConfigFact:
    return ConfigFact(name, source[name], _pointer(prefix, name))


def _unknown(adapter_id: str, *reasons: str) -> ModelFamilyAdapterResult:
    return ModelFamilyAdapterResult(
        adapter_id, "unknown", (), None, None, (), None, tuple(reasons)
    )


def _pointer(prefix: str, name: str) -> str:
    escaped = name.replace("~", "~0").replace("/", "~1")
    return f"{prefix}/{escaped}" if prefix else f"/{escaped}"


def _is_positive_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
