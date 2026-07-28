# SPDX-License-Identifier: Apache-2.0
"""Qualification-evidence import for incomplete ModelProfile fragments."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Any, Callable, Mapping

from vllm_mlx._model_profile_compat_types import (
    CompatibilityIssue,
    LegacySourceInput,
    ProvenanceKind,
)

Assignment = Callable[[str, Any, LegacySourceInput, ProvenanceKind], None]

_EVIDENCE_FIELDS = (
    "evidence_id",
    "kind",
    "location",
    "artifact_sha256",
    "result",
    "hardware_fingerprint",
    "workload_id",
    "subject_digest",
    "created_at",
)
_EVIDENCE_STRING_FIELDS = frozenset(
    {"evidence_id", "kind", "location", "hardware_fingerprint", "workload_id"}
)
_EVIDENCE_RESULTS = frozenset({"pass", "fail", "incomplete"})
_QUALIFICATION_STATUSES = frozenset({"not_qualified", "qualified", "failed", "blocked"})
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}" r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def import_qualification(
    source: LegacySourceInput, assign: Assignment, issues: list[CompatibilityIssue]
) -> None:
    """Import recorded evidence without qualifying, finalizing, or promoting it."""
    payload = source.payload
    evidence = _normalize_evidence(payload.get("evidence"), source, issues)
    if evidence is not None:
        assign("/qualification/evidence", evidence, source, "measured_result")
    _report_weak_qualification_signals(payload, source, issues)
    _import_status(
        payload.get("qualification_status"), evidence, source, assign, issues
    )


def _normalize_evidence(
    value: Any,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        _invalid_evidence(
            "/qualification/evidence", "evidence must be an array", source, issues
        )
        return None

    records: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        normalized = _normalize_evidence_record(record, index, source, issues)
        if normalized is not None:
            records.append(normalized)
    if records or not value:
        return records
    return None


def _normalize_evidence_record(
    value: Any,
    index: int,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> dict[str, Any] | None:
    pointer = f"/qualification/evidence/{index}"
    if not isinstance(value, Mapping):
        _invalid_evidence(pointer, "evidence records must be objects", source, issues)
        return None
    fields = set(value)
    missing = sorted(set(_EVIDENCE_FIELDS) - fields)
    unknown = sorted(fields - set(_EVIDENCE_FIELDS))
    if missing or unknown:
        detail_parts = []
        if missing:
            detail_parts.append(f"missing required fields: {', '.join(missing)}")
        if unknown:
            detail_parts.append(f"unknown fields: {', '.join(unknown)}")
        _invalid_evidence(pointer, "; ".join(detail_parts), source, issues)
        return None
    for field in _EVIDENCE_STRING_FIELDS:
        if not _is_non_empty_string(value[field]):
            _invalid_evidence(
                f"{pointer}/{field}",
                "field must be a non-empty string",
                source,
                issues,
            )
            return None
    if not _is_sha256(value["artifact_sha256"]):
        _invalid_evidence(
            f"{pointer}/artifact_sha256",
            "field must be a SHA-256 digest",
            source,
            issues,
        )
        return None
    if not _is_sha256(value["subject_digest"]):
        _invalid_evidence(
            f"{pointer}/subject_digest",
            "field must be a SHA-256 digest",
            source,
            issues,
        )
        return None
    if not isinstance(value["result"], str) or value["result"] not in _EVIDENCE_RESULTS:
        _invalid_evidence(
            f"{pointer}/result",
            "result must be one of: pass, fail, incomplete",
            source,
            issues,
        )
        return None
    if not is_rfc3339_datetime(value["created_at"]):
        _invalid_evidence(
            f"{pointer}/created_at",
            "field must be an RFC 3339 date-time with an offset",
            source,
            issues,
        )
        return None
    return {field: deepcopy(value[field]) for field in _EVIDENCE_FIELDS}


def _import_status(
    value: Any,
    evidence: list[dict[str, Any]] | None,
    source: LegacySourceInput,
    assign: Assignment,
    issues: list[CompatibilityIssue],
) -> None:
    if value is None:
        return
    if not isinstance(value, str) or value not in _QUALIFICATION_STATUSES:
        _invalid_evidence(
            "/qualification/status",
            "qualification_status must be one of: not_qualified, qualified, failed, blocked",
            source,
            issues,
            code="invalid_qualification_status",
        )
        return
    if value == "qualified" and not _has_passing_evidence(evidence):
        _invalid_evidence(
            "/qualification/status",
            "qualified requires at least one structurally valid passing evidence record",
            source,
            issues,
            code="qualification_without_passing_evidence",
        )
        return
    assign("/qualification/status", value, source, "measured_result")


def _report_weak_qualification_signals(
    payload: Mapping[str, Any],
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> None:
    if "status" in payload:
        _weak_signal(
            "generic_status_ignored",
            "/qualification/status",
            "generic status is not qualification evidence; use qualification_status with evidence",
            source,
            issues,
        )
    for field in ("production_ready", "qualified", "is_qualified", "qualification"):
        if isinstance(payload.get(field), bool):
            _weak_signal(
                "qualification_boolean_ignored",
                "/qualification",
                f"{field} boolean is not qualification evidence",
                source,
                issues,
            )


def _weak_signal(
    code: str,
    pointer: str,
    detail: str,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
) -> None:
    issues.append(
        CompatibilityIssue(
            code=code,
            severity="warning",
            pointer=pointer,
            sources=(source.location,),
            detail=detail,
        )
    )


def _invalid_evidence(
    pointer: str,
    detail: str,
    source: LegacySourceInput,
    issues: list[CompatibilityIssue],
    *,
    code: str = "invalid_qualification_evidence",
) -> None:
    issues.append(
        CompatibilityIssue(
            code=code,
            severity="error",
            pointer=pointer,
            sources=(source.location,),
            detail=detail,
        )
    )


def _has_passing_evidence(evidence: list[dict[str, Any]] | None) -> bool:
    if evidence is None:
        return False
    return any(record["result"] == "pass" for record in evidence)


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def is_rfc3339_datetime(value: Any) -> bool:
    """Return whether a value is an offset-qualified RFC 3339 date-time."""
    if not isinstance(value, str) or _RFC3339_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None
