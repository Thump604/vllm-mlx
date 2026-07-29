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
    "backend",
    "capabilities",
    "serving",
    "hardware_fit",
    "provenance",
)

_IDENTITY_PROVIDER_FIELDS = (
    "/identity/provider",
    "/identity/repository_id",
    "/identity/requested_revision",
    "/identity/resolved_revision",
)

_PROVENANCE_KINDS_BY_PREFIX = (
    ("/identity/", {"provider_fact", "derived_recommendation", "maintainer_policy"}),
    ("/artifact/", {"provider_fact", "derived_recommendation"}),
    ("/backend/", {"provider_fact", "derived_recommendation", "measured_result"}),
    (
        "/capabilities/",
        {"provider_fact", "derived_recommendation", "measured_result"},
    ),
    (
        "/serving/",
        {"provider_fact", "derived_recommendation", "maintainer_policy"},
    ),
)

_FEATURE_MODE_CONTROL_RULES = {
    "available_per_request": (
        "request",
        "request_feature_not_allowed",
        "per-request feature must name an allowed or required request field",
    ),
    "available_on_activation": (
        "activation",
        "activation_feature_not_allowed",
        "activation feature must name an allowed activation override",
    ),
}

_INACTIVE_FEATURE_MODES = frozenset({"guarded_off", "deferred", "not_supported"})

_CONTROL_FIELD_RULES = {
    "request": (
        "request_feature_not_allowed",
        "request control field is not allowed by request policy",
    ),
    "activation": (
        "activation_feature_not_allowed",
        "activation control field is not allowed by activation policy",
    ),
}


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
    _check_backend(profile, issues)
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


def _check_backend(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    """Validate optional, immutable backend compatibility facts.

    Profiles created before this contract intentionally remain valid.  A profile
    that opts in must be explicit about whether a text loader may discard
    unmatched weights; silent fallback is not a compatibility contract.
    """
    backend = profile.get("backend")
    if backend is None:
        return
    policy = backend["weight_policy"]
    allowed_prefixes = policy["allowed_unmatched_weight_prefixes"]
    if policy["mode"] == "strict" and allowed_prefixes:
        _issue(
            issues,
            "strict_loader_allows_unmatched_weights",
            "/backend/weight_policy/allowed_unmatched_weight_prefixes",
            "strict loader policy cannot allow unmatched weight prefixes",
        )
    if backend["loader_route"] == "mlx_vlm" and backend["backend_id"] != "mlx-vlm":
        _issue(
            issues,
            "backend_route_mismatch",
            "/backend",
            "mlx_vlm loader route requires backend_id mlx-vlm",
        )
    if backend["loader_route"] == "mlx_lm" and backend["backend_id"] != "mlx-lm":
        _issue(
            issues,
            "backend_route_mismatch",
            "/backend",
            "mlx_lm loader route requires backend_id mlx-lm",
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
    fields_by_control = {
        "request": request_fields,
        "activation": activation_fields,
    }
    for name, feature in serving["features"].items():
        pointer = f"/serving/features/{name}"
        issue = _feature_control_issue(feature, fields_by_control)
        if issue is not None:
            _issue(issues, issue[0], pointer, issue[1])


def _feature_control_issue(
    feature: Mapping[str, Any], fields_by_control: Mapping[str, set[str]]
) -> tuple[str, str] | None:
    mode = feature["mode"]
    control = feature["control"]
    field = feature.get("control_field")
    mode_rule = _FEATURE_MODE_CONTROL_RULES.get(mode)
    if mode_rule is not None:
        expected_control, code, detail = mode_rule
        if (
            control != expected_control
            or field not in fields_by_control[expected_control]
        ):
            return code, detail
        return None
    if mode in _INACTIVE_FEATURE_MODES:
        if control != "none" or field is not None:
            return (
                "inactive_feature_has_control",
                "inactive feature modes cannot expose activation or request control",
            )
        return None
    control_rule = _CONTROL_FIELD_RULES.get(control)
    if control_rule is not None and field not in fields_by_control[control]:
        return control_rule
    return None


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
    pointers = _provenance_subject_pointers(profile)
    _check_provenance_records(records, pointers, issues)
    _check_provenance_coverage(profile, records, pointers, issues)
    _check_measured_hardware_fit(profile, issues)


def _provenance_subject_pointers(profile: Mapping[str, Any]) -> list[str]:
    pointers = [
        pointer
        for section in _provenance_subject_sections(profile)
        for pointer in _leaf_pointers(profile[section], f"/{section}")
    ]
    pointers.extend(_leaf_pointers(profile.get("hardware_fit", []), "/hardware_fit"))
    return pointers


def _check_provenance_records(
    records: Sequence[Mapping[str, Any]],
    pointers: Sequence[str],
    issues: list[ModelProfileValidationIssue],
) -> None:
    for index, record in enumerate(records):
        for path in record["field_paths"]:
            if not any(_path_covers(path, pointer) for pointer in pointers):
                _issue(
                    issues,
                    "invalid_provenance_path",
                    f"/provenance/records/{index}/field_paths",
                    f"{path} does not identify a profile subject field",
                )
        _check_provenance_metadata(record, index, issues)


def _check_provenance_coverage(
    profile: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    pointers: Sequence[str],
    issues: list[ModelProfileValidationIssue],
) -> None:
    for pointer in pointers:
        allowed_kinds = _allowed_provenance_kinds(profile, pointer)
        covering = [
            (index, record)
            for index, record in enumerate(records)
            if any(_path_covers(path, pointer) for path in record["field_paths"])
        ]
        if not any(record["kind"] in allowed_kinds for _, record in covering):
            _issue(
                issues,
                "provenance_kind_mismatch",
                pointer,
                f"requires provenance kind in {sorted(allowed_kinds)}",
            )
        for index, record in covering:
            if record["kind"] not in allowed_kinds:
                _issue(
                    issues,
                    "invalid_provenance_claim",
                    f"/provenance/records/{index}",
                    f"{record['kind']} cannot establish {pointer}",
                )


def _check_measured_hardware_fit(
    profile: Mapping[str, Any], issues: list[ModelProfileValidationIssue]
) -> None:
    for index, fit in enumerate(profile.get("hardware_fit", [])):
        if (
            fit["method"] == "measured"
            and fit.get("measured_peak_memory_bytes") is None
        ):
            _issue(
                issues,
                "measured_hardware_fit_without_measurement",
                f"/hardware_fit/{index}/measured_peak_memory_bytes",
                "measured hardware fit requires a measured peak-memory value",
            )


def _provenance_subject_sections(profile: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the immutable profile sections that require provenance."""
    sections = ("identity", "artifact", "capabilities", "serving")
    return (*sections, "backend") if "backend" in profile else sections


def _leaf_pointers(value: Any, pointer: str) -> list[str]:
    if isinstance(value, Mapping):
        if not value:
            return [pointer]
        return [
            child
            for key, item in value.items()
            for child in _leaf_pointers(
                item, f"{pointer}/{_escape_pointer_token(str(key))}"
            )
        ]
    if isinstance(value, list):
        if not value or all(not isinstance(item, (Mapping, list)) for item in value):
            return [pointer]
        return [
            child
            for index, item in enumerate(value)
            for child in _leaf_pointers(item, f"{pointer}/{index}")
        ]
    return [pointer]


def _path_covers(path: str, pointer: str) -> bool:
    return path == pointer or pointer.startswith(f"{path}/")


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _check_provenance_metadata(
    record: Mapping[str, Any],
    index: int,
    issues: list[ModelProfileValidationIssue],
) -> None:
    pointer = f"/provenance/records/{index}"
    if record.get("observed_at") is None:
        _issue(
            issues,
            "provenance_timestamp_missing",
            f"{pointer}/observed_at",
            "provenance records require an observation timestamp",
        )
    kind = record["kind"]
    if kind == "provider_fact" and not (record.get("revision") or record.get("sha256")):
        _issue(
            issues,
            "provider_provenance_unbound",
            pointer,
            "provider facts require an immutable revision or source hash",
        )
    elif kind == "derived_recommendation" and not record.get("rule_id"):
        _issue(
            issues,
            "derived_provenance_rule_missing",
            f"{pointer}/rule_id",
            "derived recommendations require a resolver rule ID",
        )
    elif kind in {"measured_result", "maintainer_policy"} and not record.get("sha256"):
        _issue(
            issues,
            "provenance_source_hash_missing",
            f"{pointer}/sha256",
            f"{kind} requires a hashed source artifact",
        )


def _allowed_provenance_kinds(profile: Mapping[str, Any], pointer: str) -> set[str]:
    if pointer.startswith(_IDENTITY_PROVIDER_FIELDS):
        if profile["identity"]["provider"] == "local":
            return {"derived_recommendation", "maintainer_policy"}
        return {"provider_fact"}
    if pointer.startswith("/serving/sampling/provider_defaults"):
        if not profile["serving"]["sampling"]["provider_defaults"]:
            return {"provider_fact", "derived_recommendation", "maintainer_policy"}
        return {"provider_fact"}
    if pointer.startswith("/hardware_fit/"):
        index = int(pointer.split("/", 3)[2])
        method = profile["hardware_fit"][index]["method"]
        return {"measured_result" if method == "measured" else "derived_recommendation"}
    for prefix, allowed_kinds in _PROVENANCE_KINDS_BY_PREFIX:
        if pointer.startswith(prefix):
            return allowed_kinds
    raise ValueError(f"no provenance policy for {pointer}")
