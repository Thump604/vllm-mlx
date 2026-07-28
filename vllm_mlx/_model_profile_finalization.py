# SPDX-License-Identifier: Apache-2.0
"""Explicit, behavior-independent finalization for legacy profile imports."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from vllm_mlx._model_profile_compat_types import (
    ModelProfileFinalizationError,
    ModelProfileImportResult,
)
from vllm_mlx.model_profile import validate_model_profile
from vllm_mlx.model_profile_import import validate_import_result


def finalize_import(
    imported: ModelProfileImportResult,
    completed_profile: Mapping[str, Any],
    *,
    profile_schema: Mapping[str, Any],
    import_schema: Mapping[str, Any],
) -> ModelProfileImportResult:
    """Validate an explicit completion without changing imported facts."""
    if imported.complete:
        raise ModelProfileFinalizationError("import result is already complete")
    validate_import_result(imported.as_dict(), import_schema, profile_schema)
    if not isinstance(imported.profile, Mapping):
        raise ModelProfileFinalizationError(
            "incomplete import does not contain a profile fragment"
        )

    blockers = tuple(
        issue
        for issue in imported.issues
        if issue.severity == "error" and issue.code != "missing_required_fact"
    )
    if blockers:
        codes = ", ".join(sorted({issue.code for issue in blockers}))
        raise ModelProfileFinalizationError(
            f"import has unresolved non-missing errors: {codes}"
        )

    candidate = deepcopy(dict(completed_profile))
    mismatch = _first_changed_fact(imported.profile, candidate)
    if mismatch is not None:
        raise ModelProfileFinalizationError(
            f"completed profile changes imported fact at {mismatch}"
        )
    _require_preserved_provenance(imported.profile, candidate)

    validate_model_profile(candidate, profile_schema)
    result = ModelProfileImportResult(
        complete=True,
        sources=imported.sources,
        profile=candidate,
        issues=tuple(issue for issue in imported.issues if issue.severity == "warning"),
    )
    validate_import_result(result.as_dict(), import_schema, profile_schema)
    return result


def _first_changed_fact(
    imported: Any,
    candidate: Any,
    pointer: str = "",
) -> str | None:
    if isinstance(imported, Mapping):
        if not isinstance(candidate, Mapping):
            return pointer or "/"
        for key, value in imported.items():
            child = _join_pointer(pointer, str(key))
            if child == "/provenance/records":
                continue
            if key not in candidate:
                return child
            mismatch = _first_changed_fact(value, candidate[key], child)
            if mismatch is not None:
                return mismatch
        return None
    if imported != candidate:
        return pointer or "/"
    return None


def _require_preserved_provenance(
    imported: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    imported_records = _provenance_records(imported)
    candidate_records = _provenance_records(candidate)
    for index, record in enumerate(imported_records):
        if record not in candidate_records:
            raise ModelProfileFinalizationError(
                "completed profile removes or changes imported provenance record "
                f"/provenance/records/{index}"
            )


def _provenance_records(profile: Mapping[str, Any]) -> Sequence[Any]:
    provenance = profile.get("provenance")
    if not isinstance(provenance, Mapping):
        return ()
    records = provenance.get("records")
    return records if isinstance(records, list) else ()


def _join_pointer(parent: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"
