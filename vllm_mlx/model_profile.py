# SPDX-License-Identifier: Apache-2.0
"""ModelProfile v1 canonical hashing and behavior-independent validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import rfc8785
from jsonschema.validators import validator_for
from referencing import Registry, Resource

_SUBJECT_FIELDS = (
    "identity",
    "artifact",
    "capabilities",
    "serving",
    "hardware_fit",
    "provenance",
)


@dataclass(frozen=True)
class ModelProfileValidationIssue:
    """One schema or semantic validation failure."""

    code: str
    pointer: str
    detail: str


class ModelProfileValidationError(ValueError):
    """Raised when a ModelProfile fails one or more deterministic checks."""

    def __init__(self, issues: Sequence[ModelProfileValidationIssue]) -> None:
        self.issues = tuple(issues)
        summary = "; ".join(
            f"{issue.pointer or '/'}: {issue.detail}" for issue in self.issues
        )
        super().__init__(summary)


def canonical_subject(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the immutable fields covered by ``subject_digest``."""
    return {field: profile[field] for field in _SUBJECT_FIELDS if field in profile}


def compute_subject_digest(profile: Mapping[str, Any]) -> str:
    """Compute the RFC 8785 SHA-256 digest for the immutable profile subject."""
    encoded = rfc8785.dumps(canonical_subject(profile))
    return hashlib.sha256(encoded).hexdigest()


def collect_model_profile_issues(
    profile: Mapping[str, Any], schema: Mapping[str, Any]
) -> tuple[ModelProfileValidationIssue, ...]:
    """Collect JSON Schema and cross-field ModelProfile v1 failures."""
    issues: list[ModelProfileValidationIssue] = []
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema, format_checker=validator_class.FORMAT_CHECKER)
    for error in sorted(validator.iter_errors(profile), key=_schema_error_key):
        issues.append(
            ModelProfileValidationIssue(
                code="schema_invalid",
                pointer=_json_pointer(error.absolute_path),
                detail=error.message,
            )
        )
    if issues:
        return tuple(issues)

    _check_subject_digest(profile, issues)
    _check_limits(profile, issues)
    _check_request_policy(profile, issues)
    _check_feature_controls(profile, issues)
    _check_qualification(profile, issues)
    _check_provenance(profile, issues)
    return tuple(issues)


def validate_model_profile(
    profile: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    """Raise with all deterministic failures; return ``None`` when valid."""
    issues = collect_model_profile_issues(profile, schema)
    if issues:
        raise ModelProfileValidationError(issues)


def collect_import_result_issues(
    result: Mapping[str, Any],
    import_schema: Mapping[str, Any],
    profile_schema: Mapping[str, Any],
) -> tuple[ModelProfileValidationIssue, ...]:
    """Validate a bounded import envelope and any complete profile it carries."""
    issues: list[ModelProfileValidationIssue] = []
    validator_class = validator_for(import_schema)
    validator_class.check_schema(import_schema)
    registry = Registry().with_resource(
        str(profile_schema["$id"]), Resource.from_contents(profile_schema)
    )
    validator = validator_class(
        import_schema,
        registry=registry,
        format_checker=validator_class.FORMAT_CHECKER,
    )
    for error in sorted(validator.iter_errors(result), key=_schema_error_key):
        issues.append(
            ModelProfileValidationIssue(
                code="import_schema_invalid",
                pointer=_json_pointer(error.absolute_path),
                detail=error.message,
            )
        )
    if issues or not result.get("complete"):
        return tuple(issues)
    profile = result.get("profile")
    if not isinstance(profile, Mapping):
        return tuple(issues)
    for issue in collect_model_profile_issues(profile, profile_schema):
        issues.append(
            ModelProfileValidationIssue(
                code=issue.code,
                pointer=f"/profile{issue.pointer}",
                detail=issue.detail,
            )
        )
    return tuple(issues)


def validate_import_result(
    result: Mapping[str, Any],
    import_schema: Mapping[str, Any],
    profile_schema: Mapping[str, Any],
) -> None:
    """Raise when an import envelope or its complete profile is invalid."""
    issues = collect_import_result_issues(result, import_schema, profile_schema)
    if issues:
        raise ModelProfileValidationError(issues)


def _schema_error_key(error: Any) -> tuple[str, str]:
    return (_json_pointer(error.absolute_path), error.message)


def _json_pointer(parts: Sequence[Any]) -> str:
    if not parts:
        return ""
    escaped = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def _issue(
    issues: list[ModelProfileValidationIssue], code: str, pointer: str, detail: str
) -> None:
    issues.append(
        ModelProfileValidationIssue(code=code, pointer=pointer, detail=detail)
    )


def _check_subject_digest(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    try:
        expected = compute_subject_digest(profile)
    except rfc8785.CanonicalizationError as error:
        _issue(
            issues,
            "subject_not_canonicalizable",
            "/subject_digest",
            f"profile subject cannot be represented by RFC 8785: {error}",
        )
        return
    if profile["subject_digest"].lower() != expected:
        _issue(
            issues,
            "subject_digest_mismatch",
            "/subject_digest",
            f"expected {expected}",
        )


def _check_limits(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    limits = profile["serving"]["limits"]
    advertised = limits["advertised_context"]
    serving = limits["serving_context"]
    output = limits["max_output_tokens"]
    request_output = limits["max_request_output_tokens"]
    max_kv = limits.get("max_kv_size")
    if serving > advertised:
        _issue(
            issues,
            "serving_context_exceeds_advertised",
            "/serving/limits/serving_context",
            "serving context must not exceed advertised context",
        )
    if output > serving:
        _issue(
            issues,
            "output_exceeds_context",
            "/serving/limits/max_output_tokens",
            "default output must not exceed serving context",
        )
    if request_output > serving:
        _issue(
            issues,
            "request_output_exceeds_context",
            "/serving/limits/max_request_output_tokens",
            "request output cap must not exceed serving context",
        )
    if output > request_output:
        _issue(
            issues,
            "default_output_exceeds_request_cap",
            "/serving/limits/max_output_tokens",
            "default output must not exceed the request output cap",
        )
    if max_kv is not None and max_kv < serving:
        _issue(
            issues,
            "kv_smaller_than_context",
            "/serving/limits/max_kv_size",
            "max KV size must support the selected serving context",
        )


def _check_request_policy(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    policy = profile["serving"]["request_policy"]
    required = set(policy["required_fields"])
    allowed = set(policy["allowed_fields"])
    forbidden = set(policy["forbidden_fields"])
    overlap = (required | allowed) & forbidden
    if overlap:
        _issue(
            issues,
            "request_policy_conflict",
            "/serving/request_policy",
            f"forbidden fields also required or allowed: {sorted(overlap)}",
        )


def _check_feature_controls(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    serving = profile["serving"]
    request_policy = serving["request_policy"]
    request_fields = set(request_policy["required_fields"]) | set(
        request_policy["allowed_fields"]
    )
    activation_fields = set(serving["activation_policy"]["owner_override_fields"])
    for name, feature in serving["features"].items():
        mode = feature["mode"]
        control = feature["control"]
        field = feature.get("control_field")
        pointer = f"/serving/features/{name}"
        if mode == "available_per_request":
            if control != "request" or field not in request_fields:
                _issue(
                    issues,
                    "request_feature_not_allowed",
                    pointer,
                    "per-request feature must name an allowed or required request field",
                )
        elif mode == "available_on_activation":
            if control != "activation" or field not in activation_fields:
                _issue(
                    issues,
                    "activation_feature_not_allowed",
                    pointer,
                    "activation feature must name an allowed activation override",
                )
        elif mode in {"guarded_off", "deferred", "not_supported"}:
            if control != "none" or field is not None:
                _issue(
                    issues,
                    "inactive_feature_has_control",
                    pointer,
                    "inactive feature modes cannot expose activation or request control",
                )
        elif control == "request" and field not in request_fields:
            _issue(
                issues,
                "request_feature_not_allowed",
                pointer,
                "request control field is not allowed by request policy",
            )
        elif control == "activation" and field not in activation_fields:
            _issue(
                issues,
                "activation_feature_not_allowed",
                pointer,
                "activation control field is not allowed by activation policy",
            )


def _check_qualification(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    qualification = profile["qualification"]
    evidence = qualification["evidence"]
    subject_digest = profile["subject_digest"].lower()
    for index, record in enumerate(evidence):
        if record["subject_digest"].lower() != subject_digest:
            _issue(
                issues,
                "evidence_digest_mismatch",
                f"/qualification/evidence/{index}/subject_digest",
                "evidence is not bound to this profile subject",
            )
    if qualification["status"] == "qualified" and not any(
        record["result"] == "pass"
        and record["subject_digest"].lower() == subject_digest
        for record in evidence
    ):
        _issue(
            issues,
            "qualification_without_bound_pass",
            "/qualification/status",
            "qualified requires passing evidence bound to this subject digest",
        )


def _check_provenance(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    records = profile["provenance"]["records"]
    if "hardware_fit" in profile and not any(
        any(
            path == "/hardware_fit" or path.startswith("/hardware_fit/")
            for path in record["field_paths"]
        )
        for record in records
    ):
        _issue(
            issues,
            "hardware_fit_without_provenance",
            "/hardware_fit",
            "hardware-fit claims require provenance coverage",
        )
