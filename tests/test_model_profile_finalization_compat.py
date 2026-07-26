# SPDX-License-Identifier: Apache-2.0
"""Focused tests for explicit legacy ModelProfile finalization."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from vllm_mlx._model_profile_compat_types import (
    CompatibilityIssue,
    LegacySourceInput,
    ModelProfileImportResult,
)
from vllm_mlx.model_profile import ModelProfileValidationError
from vllm_mlx.model_profile_compat import (
    ModelProfileFinalizationError,
    finalize_legacy_model_profile,
)

ROOT = Path(__file__).parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


@pytest.fixture
def profile_schema() -> dict:
    return _load("schemas/model-profile-v1.schema.json")


@pytest.fixture
def import_schema() -> dict:
    return _load("schemas/model-profile-import-result-v1.schema.json")


@pytest.fixture
def completed_profile() -> dict:
    return _load("schemas/examples/model-profile-v1.example.json")


def _source() -> LegacySourceInput:
    return LegacySourceInput.from_mapping(
        "acquisition",
        {
            "location": "/manifests/acquisition.json",
            "sha256": "a" * 64,
            "payload": {},
        },
    )


def _issue(code: str, severity: str = "error") -> CompatibilityIssue:
    return CompatibilityIssue(
        code=code,
        severity=severity,
        pointer="/profile_id",
        sources=("/manifests/acquisition.json",),
        detail="test issue",
    )


def _incomplete(profile: dict, *issues: CompatibilityIssue) -> ModelProfileImportResult:
    return ModelProfileImportResult(
        complete=False,
        sources=(_source(),),
        profile=profile,
        issues=issues or (_issue("missing_required_fact"),),
    )


def _matching_fragment(completed_profile: dict) -> dict:
    return {
        "schema_version": 1,
        "identity": {
            "provider": completed_profile["identity"]["provider"],
            "artifact_id": completed_profile["identity"]["artifact_id"],
        },
        "provenance": {
            "records": [deepcopy(completed_profile["provenance"]["records"][0])]
        },
    }


def test_finalizes_without_mutating_inputs_and_retains_warnings(
    completed_profile, profile_schema, import_schema
):
    imported = _incomplete(
        _matching_fragment(completed_profile),
        _issue("missing_required_fact"),
        _issue("legacy_hint_ignored", "warning"),
    )
    imported_before = imported.as_dict()
    candidate_before = deepcopy(completed_profile)

    result = finalize_legacy_model_profile(
        imported,
        completed_profile,
        profile_schema=profile_schema,
        import_schema=import_schema,
    )

    assert result.complete is True
    assert result.profile == completed_profile
    assert result.profile is not completed_profile
    assert result.sources == imported.sources
    assert [issue.code for issue in result.issues] == ["legacy_hint_ignored"]
    assert imported.as_dict() == imported_before
    assert completed_profile == candidate_before


def test_rejects_an_already_complete_import(
    completed_profile, profile_schema, import_schema
):
    imported = ModelProfileImportResult(
        complete=True,
        sources=(_source(),),
        profile=completed_profile,
        issues=(),
    )

    with pytest.raises(ModelProfileFinalizationError, match="already complete"):
        finalize_legacy_model_profile(
            imported,
            completed_profile,
            profile_schema=profile_schema,
            import_schema=import_schema,
        )


def test_rejects_non_missing_import_errors(
    completed_profile, profile_schema, import_schema
):
    imported = _incomplete(
        _matching_fragment(completed_profile),
        _issue("conflicting_value"),
    )

    with pytest.raises(ModelProfileFinalizationError, match="conflicting_value"):
        finalize_legacy_model_profile(
            imported,
            completed_profile,
            profile_schema=profile_schema,
            import_schema=import_schema,
        )


def test_rejects_changes_to_imported_facts(
    completed_profile, profile_schema, import_schema
):
    imported = _incomplete(_matching_fragment(completed_profile))
    candidate = deepcopy(completed_profile)
    candidate["identity"]["provider"] = "local"

    with pytest.raises(ModelProfileFinalizationError, match="/identity/provider"):
        finalize_legacy_model_profile(
            imported,
            candidate,
            profile_schema=profile_schema,
            import_schema=import_schema,
        )


def test_rejects_removed_or_changed_imported_provenance(
    completed_profile, profile_schema, import_schema
):
    imported = _incomplete(_matching_fragment(completed_profile))
    candidate = deepcopy(completed_profile)
    candidate["provenance"]["records"].pop(0)

    with pytest.raises(ModelProfileFinalizationError, match="provenance record"):
        finalize_legacy_model_profile(
            imported,
            candidate,
            profile_schema=profile_schema,
            import_schema=import_schema,
        )


def test_semantic_validation_rejects_digest_or_evidence_mismatch(
    completed_profile, profile_schema, import_schema
):
    imported = _incomplete(_matching_fragment(completed_profile))
    candidate = deepcopy(completed_profile)
    candidate["serving"]["engine"] = "different"

    with pytest.raises(ModelProfileValidationError) as error:
        finalize_legacy_model_profile(
            imported,
            candidate,
            profile_schema=profile_schema,
            import_schema=import_schema,
        )

    assert {issue.code for issue in error.value.issues} >= {"subject_digest_mismatch"}


def test_semantic_validation_rejects_unbound_qualification_evidence(
    completed_profile, profile_schema, import_schema
):
    imported = _incomplete(_matching_fragment(completed_profile))
    candidate = deepcopy(completed_profile)
    candidate["qualification"]["evidence"] = [
        {
            "evidence_id": "unbound",
            "kind": "bench_serve",
            "location": "/run/unbound.json",
            "artifact_sha256": "b" * 64,
            "result": "pass",
            "hardware_fingerprint": "apple-silicon",
            "workload_id": "smoke",
            "subject_digest": "c" * 64,
            "created_at": "2026-07-26T05:00:00Z",
        }
    ]

    with pytest.raises(ModelProfileValidationError) as error:
        finalize_legacy_model_profile(
            imported,
            candidate,
            profile_schema=profile_schema,
            import_schema=import_schema,
        )

    assert {issue.code for issue in error.value.issues} >= {"evidence_digest_mismatch"}


def test_schema_validation_rejects_incomplete_candidate(
    completed_profile, profile_schema, import_schema
):
    imported = _incomplete(_matching_fragment(completed_profile))
    candidate = deepcopy(completed_profile)
    del candidate["serving"]

    with pytest.raises(ModelProfileValidationError):
        finalize_legacy_model_profile(
            imported,
            candidate,
            profile_schema=profile_schema,
            import_schema=import_schema,
        )
