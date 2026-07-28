# SPDX-License-Identifier: Apache-2.0
"""Validation for bounded legacy ModelProfile import results."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from jsonschema.validators import validator_for
from referencing import Registry, Resource

from vllm_mlx.model_profile import (
    ModelProfileValidationError,
    ModelProfileValidationIssue,
    collect_model_profile_issues,
)


def collect_import_result_issues(
    result: Mapping[str, Any],
    import_schema: Mapping[str, Any],
    profile_schema: Mapping[str, Any],
) -> tuple[ModelProfileValidationIssue, ...]:
    """Validate an import envelope and any complete profile it carries.

    Both schema mappings must be valid JSON Schemas, and ``profile_schema``
    must define ``$id``. Schema configuration errors propagate to the caller;
    only instance-validation failures are returned as issues.
    """
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
    """Raise when an import envelope or its complete profile is invalid.

    The schema preconditions and configuration-error behavior are identical to
    :func:`collect_import_result_issues`.
    """
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
