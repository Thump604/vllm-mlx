# SPDX-License-Identifier: Apache-2.0
"""Closed-vocabulary PR3B serving mappings for legacy profile imports."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from vllm_mlx._model_profile_compat_types import (
    CompatibilityIssue,
    LegacySourceInput,
    ProvenanceKind,
)
from vllm_mlx._model_profile_serving_vocab import (
    _ACTIVATION_FIELDS,
    _CLOSED_SERVING_FIELDS,
    _FEATURE_FIELDS,
    _FEATURE_NAMES,
    _FEATURE_SETTING_NAMES,
    _REGISTRY_FIELDS_REQUIRING_TARGET_CONTRACT,
    _REQUEST_FIELDS,
    _SAMPLING_FIELDS,
    _SERVING_TOP_LEVEL_FIELDS,
)

Assignment = Callable[[str, Any, LegacySourceInput, ProvenanceKind], None]


def import_registry(
    source: LegacySourceInput, assign: Assignment, issues: list[CompatibilityIssue]
) -> None:
    """Map registry serving data without treating registry keys as identity."""
    values = _import_serving_payload(source.payload, source, assign, issues)
    for field in _REGISTRY_FIELDS_REQUIRING_TARGET_CONTRACT:
        if field in values:
            issues.append(
                CompatibilityIssue(
                    code="registry_field_requires_target_contract",
                    severity="warning",
                    pointer="/serving",
                    sources=(source.location,),
                    detail=(
                        f"registry field {field!r} has no lossless ModelProfile "
                        "v1 destination"
                    ),
                )
            )


def import_serving(
    source: LegacySourceInput, assign: Assignment, issues: list[CompatibilityIssue]
) -> None:
    """Map a source through the closed serving vocabulary."""
    _import_serving_payload(source.payload, source, assign, issues)


def _import_serving_payload(
    payload: Mapping[str, Any],
    source: LegacySourceInput,
    assign: Assignment,
    issues: list[CompatibilityIssue],
) -> dict[str, Any]:
    values = _merge_nested_serving_values(payload, source, issues)
    _report_unknown_serving_fields(values, source, issues)
    for name in ("engine", "route"):
        assign(f"/serving/{name}", values.get(name), source, "maintainer_policy")
    _import_closed_sections(values, source, assign, issues)
    _import_direct_features(values, source, assign, issues)
    _import_direct_limits(values, source, assign, issues)
    return values


def _import_closed_sections(
    values: Mapping[str, Any],
    source: LegacySourceInput,
    assign: Assignment,
    issues: list[CompatibilityIssue],
) -> None:
    for name in (
        "template",
        "parsers",
        "sampling",
        "limits",
        "features",
        "activation_policy",
        "request_policy",
    ):
        value = values.get(name)
        if value is not None and not isinstance(value, Mapping):
            if name in {"activation_policy", "request_policy"}:
                _report_invalid_policy_shape(name, source, issues)
            else:
                _report_invalid_serving_shape(name, source, issues)
            continue
        normalized = _normalize_closed_section(name, value, source, issues)
        for key, item in normalized.items():
            assign(f"/serving/{name}/{key}", item, source, "maintainer_policy")


def _import_direct_features(
    values: Mapping[str, Any],
    source: LegacySourceInput,
    assign: Assignment,
    issues: list[CompatibilityIssue],
) -> None:
    if "continuous_batching" in values:
        batching = values["continuous_batching"]
        if _assign_boolean_feature(
            "continuous_batching", batching, source, assign, issues
        ):
            assign(
                "/serving/engine",
                "batched" if batching else "simple",
                source,
                "maintainer_policy",
            )
    if "enable_mtp" in values:
        _assign_boolean_feature("mtp", values["enable_mtp"], source, assign, issues)
    if any(
        key in values
        for key in (
            "specprefill",
            "specprefill_enabled",
            "specprefill_draft_model",
            "specprefill_threshold",
            "specprefill_keep_pct",
            "specprefill_backbone_pct",
        )
    ):
        _import_direct_specprefill(values, source, assign, issues)


def _import_direct_specprefill(
    values: Mapping[str, Any],
    source: LegacySourceInput,
    assign: Assignment,
    issues: list[CompatibilityIssue],
) -> None:
    enabled = values.get("specprefill_enabled", values.get("specprefill"))
    settings = (
        {"draft_model": values["specprefill_draft_model"]}
        if "specprefill_draft_model" in values
        else {}
    )
    for key in (
        "specprefill_threshold",
        "specprefill_keep_pct",
        "specprefill_backbone_pct",
    ):
        if key in values:
            issues.append(
                CompatibilityIssue(
                    code="unsupported_feature_setting",
                    severity="warning",
                    pointer="/serving/features/specprefill/settings",
                    sources=(source.location,),
                    detail=f"{key} has no v1 feature-settings field.",
                )
            )
    if enabled is None:
        issues.append(
            CompatibilityIssue(
                code="invalid_feature_declaration",
                severity="error",
                pointer="/serving/features/specprefill",
                sources=(source.location,),
                detail="direct SpecPrefill settings require a boolean enablement field",
            )
        )
        return
    _assign_boolean_feature(
        "specprefill",
        enabled,
        source,
        assign,
        issues,
        settings,
    )


def _import_direct_limits(
    values: Mapping[str, Any],
    source: LegacySourceInput,
    assign: Assignment,
    issues: list[CompatibilityIssue],
) -> None:
    for legacy, pointer in (
        ("max_tokens", "/serving/limits/max_output_tokens"),
        ("max_request_tokens", "/serving/limits/max_request_output_tokens"),
        ("max_kv_size", "/serving/limits/max_kv_size"),
    ):
        value = values.get(legacy)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            issues.append(
                CompatibilityIssue(
                    code="invalid_source_field_shape",
                    severity="error",
                    pointer=pointer,
                    sources=(source.location,),
                    detail=f"legacy limit {legacy!r} must be a positive integer",
                )
            )
            continue
        assign(pointer, value, source, "maintainer_policy")


def _merge_nested_serving_values(
    payload: Mapping[str, Any],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> dict[str, Any]:
    values = dict(payload)
    nested = payload.get("serving")
    if nested is not None and not isinstance(nested, Mapping):
        _report_invalid_serving_shape("", source, issues)
        return values
    for key, value in _mapping_or_empty(nested).items():
        if key in values and values[key] != value:
            issues.append(
                CompatibilityIssue(
                    code="same_source_conflict",
                    severity="error",
                    pointer=f"/serving/{key}",
                    sources=(source.location,),
                    detail=(
                        f"top-level value {values[key]!r} conflicts with "
                        f"nested serving value {value!r}; nested value retained"
                    ),
                )
            )
        values[key] = value
    return values


def _assign_boolean_feature(
    name: str,
    value: Any,
    source: LegacySourceInput,
    assign: Assignment,
    issues: list[CompatibilityIssue],
    settings: Mapping[str, Any] | None = None,
) -> bool:
    if not isinstance(value, bool):
        issues.append(
            CompatibilityIssue(
                code="invalid_feature_declaration",
                severity="error",
                pointer=f"/serving/features/{name}",
                sources=(source.location,),
                detail="legacy feature declarations must be boolean to map into v1.",
            )
        )
        return False
    feature: dict[str, Any] = {
        "mode": "enabled_by_default" if value else "guarded_off",
        "control": "none",
        "reason": (
            "enabled in the imported legacy configuration; override support is "
            "not established"
            if value
            else "disabled in the imported legacy configuration; feature support "
            "is not established"
        ),
    }
    if settings:
        feature["settings"] = deepcopy(dict(settings))
    assign(f"/serving/features/{name}", feature, source, "maintainer_policy")
    return True


def _report_unknown_serving_fields(
    values: Mapping[str, Any],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> None:
    for field in sorted(set(values) - _SERVING_TOP_LEVEL_FIELDS):
        issues.append(
            CompatibilityIssue(
                code="unknown_serving_field",
                severity="warning",
                pointer=f"/serving/{field}",
                sources=(source.location,),
                detail="legacy serving field has no declared ModelProfile v1 mapping",
            )
        )


def _normalize_closed_section(
    name: str,
    value: Any,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> dict[str, Any]:
    raw = _mapping_or_empty(value)
    for field in sorted(set(raw) - _CLOSED_SERVING_FIELDS[name]):
        issues.append(
            CompatibilityIssue(
                code="unknown_serving_field",
                severity="warning",
                pointer=f"/serving/{name}/{field}",
                sources=(source.location,),
                detail="field is outside the closed ModelProfile v1 serving vocabulary",
            )
        )
    if name == "sampling":
        return _normalize_sampling(raw, source, issues)
    if name == "features":
        return _normalize_features(raw, source, issues)
    if name == "activation_policy":
        return _normalize_activation_policy(raw, source, issues)
    if name == "request_policy":
        return _normalize_request_policy(raw, source, issues)
    return {
        key: deepcopy(raw[key]) for key in _CLOSED_SERVING_FIELDS[name] if key in raw
    }


def _normalize_sampling(
    raw: Mapping[str, Any],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("provider_defaults", "profile_defaults"):
        if key not in raw:
            continue
        item = raw[key]
        if not isinstance(item, Mapping):
            issues.append(
                CompatibilityIssue(
                    code="invalid_source_field_shape",
                    severity="error",
                    pointer=f"/serving/sampling/{key}",
                    sources=(source.location,),
                    detail="sampling defaults must be an object",
                )
            )
            continue
        for field in sorted(set(item) - _SAMPLING_FIELDS):
            issues.append(
                CompatibilityIssue(
                    code="unknown_serving_field",
                    severity="warning",
                    pointer=f"/serving/sampling/{key}/{field}",
                    sources=(source.location,),
                    detail="field is outside the closed v1 sampling vocabulary",
                )
            )
        result[key] = {
            field: deepcopy(field_value)
            for field, field_value in item.items()
            if field in _SAMPLING_FIELDS
        }
    return result


def _normalize_features(
    raw: Mapping[str, Any],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(set(raw) - _FEATURE_NAMES):
        issues.append(
            CompatibilityIssue(
                code="unknown_feature_declaration",
                severity="error",
                pointer=f"/serving/features/{name}",
                sources=(source.location,),
                detail="the legacy feature is outside the closed v1 feature vocabulary.",
            )
        )
    for name in sorted(set(raw) & _FEATURE_NAMES):
        declaration = raw[name]
        if not isinstance(declaration, Mapping):
            _report_invalid_feature(name, source, issues)
            continue
        unknown_fields = sorted(set(declaration) - _FEATURE_FIELDS)
        for field in unknown_fields:
            issues.append(
                CompatibilityIssue(
                    code="unknown_serving_field",
                    severity="warning",
                    pointer=f"/serving/features/{name}/{field}",
                    sources=(source.location,),
                    detail="field is outside the closed v1 feature vocabulary",
                )
            )
        feature = {
            field: deepcopy(field_value)
            for field, field_value in declaration.items()
            if field in _FEATURE_FIELDS
        }
        if "settings" in feature:
            settings = feature["settings"]
            if not isinstance(settings, Mapping):
                _report_invalid_feature(name, source, issues, field="settings")
                continue
            for setting in sorted(set(settings) - _FEATURE_SETTING_NAMES):
                issues.append(
                    CompatibilityIssue(
                        code="unsupported_feature_setting",
                        severity="warning",
                        pointer=f"/serving/features/{name}/settings/{setting}",
                        sources=(source.location,),
                        detail=(
                            "the legacy setting is outside the closed v1 feature "
                            "vocabulary."
                        ),
                    )
                )
            feature["settings"] = {
                field: deepcopy(field_value)
                for field, field_value in settings.items()
                if field in _FEATURE_SETTING_NAMES
            }
        result[name] = feature
    return result


def _normalize_activation_policy(
    raw: Mapping[str, Any],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> dict[str, Any]:
    if "owner_override_fields" not in raw:
        return {}
    fields = raw["owner_override_fields"]
    if not isinstance(fields, list):
        _report_invalid_policy_field(
            "activation_policy", "owner_override_fields", fields, source, issues
        )
        return {}
    return {
        "owner_override_fields": _filter_policy_fields(
            "activation_policy",
            "owner_override_fields",
            fields,
            _ACTIVATION_FIELDS,
            source,
            issues,
        )
    }


def _normalize_request_policy(
    raw: Mapping[str, Any],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "required_fields" in raw:
        required = raw["required_fields"]
        if not isinstance(required, Mapping):
            _report_invalid_policy_field(
                "request_policy", "required_fields", required, source, issues
            )
        else:
            result["required_fields"] = {
                field: deepcopy(field_value)
                for field, field_value in required.items()
                if field in _REQUEST_FIELDS
            }
            for field in set(required) - _REQUEST_FIELDS:
                _report_invalid_policy_field(
                    "request_policy", "required_fields", field, source, issues
                )
    for collection in ("allowed_fields", "forbidden_fields"):
        if collection not in raw:
            continue
        fields = raw[collection]
        if not isinstance(fields, list):
            _report_invalid_policy_field(
                "request_policy", collection, fields, source, issues
            )
            continue
        result[collection] = _filter_policy_fields(
            "request_policy",
            collection,
            fields,
            _REQUEST_FIELDS,
            source,
            issues,
        )
    return result


def _filter_policy_fields(
    policy: str,
    collection: str,
    fields: list[Any],
    allowed: frozenset[str],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> list[str]:
    accepted: list[str] = []
    for field in fields:
        if isinstance(field, str) and field in allowed:
            accepted.append(field)
        else:
            _report_invalid_policy_field(policy, collection, field, source, issues)
    return accepted


def _report_invalid_feature(
    name: str,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
    *,
    field: str | None = None,
) -> None:
    suffix = f"/{field}" if field else ""
    issues.append(
        CompatibilityIssue(
            code="invalid_feature_declaration",
            severity="error",
            pointer=f"/serving/features/{name}{suffix}",
            sources=(source.location,),
            detail=(
                "feature settings must be an object"
                if field
                else "a named feature declaration must be an object"
            ),
        )
    )


def _report_invalid_policy_field(
    policy: str,
    collection: str,
    field: Any,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> None:
    issues.append(
        CompatibilityIssue(
            code="invalid_policy_field",
            severity="error",
            pointer=f"/serving/{policy}/{collection}",
            sources=(source.location,),
            detail=f"field {field!r} is outside the closed v1 policy vocabulary",
        )
    )


def _report_invalid_serving_shape(
    name: str,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> None:
    pointer = f"/serving/{name}" if name else "/serving"
    issues.append(
        CompatibilityIssue(
            code="invalid_source_field_shape",
            severity="error",
            pointer=pointer,
            sources=(source.location,),
            detail="legacy serving field must be an object",
        )
    )


def _report_invalid_policy_shape(
    name: str,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> None:
    issues.append(
        CompatibilityIssue(
            code="invalid_policy_shape",
            severity="error",
            pointer=f"/serving/{name}",
            sources=(source.location,),
            detail="policy must be an object",
        )
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
