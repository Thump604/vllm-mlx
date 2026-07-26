# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the bounded legacy qualification-import slice."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from vllm_mlx._model_profile_compat import _import_legacy_sources
from vllm_mlx.model_profile_compat import import_legacy_model_profile
from vllm_mlx.model_profile_import import collect_import_result_issues

ROOT = Path(__file__).parents[1]


def _source(location: str, payload: dict) -> dict:
    return {"location": location, "sha256": "a" * 64, "payload": payload}


def _evidence(result: str = "pass", **overrides: object) -> dict:
    record = {
        "evidence_id": "qwen35b-cli-20260726",
        "kind": "bench_serve",
        "location": "/run/qwen35b-cli.json",
        "artifact_sha256": "b" * 64,
        "result": result,
        "hardware_fingerprint": "apple-m4-max-128gb",
        "workload_id": "coding-cli-v1",
        "subject_digest": "c" * 64,
        "created_at": "2026-07-26T04:00:00Z",
    }
    record.update(overrides)
    return record


def _issues(result, code: str) -> list:
    return [issue for issue in result.issues if issue.code == code]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_imports_bound_evidence_without_mutating_input_or_promoting():
    qualification = _source(
        "/manifests/qualification.json",
        {
            "qualification_status": "qualified",
            "evidence": [_evidence("pass"), _evidence("fail", evidence_id="retry")],
        },
    )
    original = deepcopy(qualification)

    result = import_legacy_model_profile(qualification=qualification)

    assert qualification == original
    assert result.complete is False
    assert result.profile["qualification"] == {
        "evidence": original["payload"]["evidence"],
        "status": "qualified",
    }
    assert result.profile["provenance"]["records"] == [
        {
            "field_paths": ["/qualification/evidence", "/qualification/status"],
            "kind": "measured_result",
            "source": "/manifests/qualification.json",
            "revision": None,
            "sha256": "a" * 64,
            "rule_id": "model-profile-compat-v1:qualification",
            "observed_at": "2026-07-26T04:00:00Z",
        }
    ]
    assert not _issues(result, "qualification_without_passing_evidence")
    serialized = result.as_dict()
    assert "payload" not in serialized["sources"][0]
    assert (
        collect_import_result_issues(
            serialized,
            _load("schemas/model-profile-import-result-v1.schema.json"),
            _load("schemas/model-profile-v1.schema.json"),
        )
        == ()
    )


@pytest.mark.parametrize("record_result", ["pass", "fail", "incomplete"])
def test_preserves_each_structurally_valid_evidence_result(record_result):
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {
                "qualification_status": "failed",
                "evidence": [_evidence(record_result)],
            },
        )
    )

    assert result.profile["qualification"] == {
        "evidence": [_evidence(record_result)],
        "status": "failed",
    }


@pytest.mark.parametrize(
    ("record", "pointer"),
    [
        ({"evidence_id": "missing-fields"}, "/qualification/evidence/0"),
        (_evidence(artifact_sha256="bad"), "/qualification/evidence/0/artifact_sha256"),
        (
            _evidence(subject_digest="not-a-digest"),
            "/qualification/evidence/0/subject_digest",
        ),
        (_evidence(result="unknown"), "/qualification/evidence/0/result"),
        (_evidence(created_at="2026-07-26"), "/qualification/evidence/0/created_at"),
        (_evidence(extra="not-allowed"), "/qualification/evidence/0"),
    ],
)
def test_rejects_invalid_evidence_records(record, pointer):
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {"qualification_status": "qualified", "evidence": [record]},
        )
    )

    assert "qualification" not in result.profile
    assert any(
        issue.pointer == pointer
        for issue in _issues(result, "invalid_qualification_evidence")
    )
    assert _issues(result, "qualification_without_passing_evidence")


def test_preserves_valid_history_when_an_adjacent_record_is_invalid():
    valid = _evidence("pass", evidence_id="valid-pass")
    invalid = _evidence("fail", evidence_id="invalid-fail", artifact_sha256="bad")

    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {"qualification_status": "qualified", "evidence": [invalid, valid]},
        )
    )

    assert result.profile["qualification"] == {
        "evidence": [valid],
        "status": "qualified",
    }
    assert any(
        issue.pointer == "/qualification/evidence/0/artifact_sha256"
        for issue in _issues(result, "invalid_qualification_evidence")
    )


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-07-26 04:00:00+00:00",
        "2026-07-26T04:00:00",
        "2026-07-26T04:00:00+0000",
    ],
)
def test_rejects_timestamps_outside_the_declared_rfc3339_shape(created_at):
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {"evidence": [_evidence(created_at=created_at)]},
        )
    )

    assert "qualification" not in result.profile
    assert any(
        issue.pointer == "/qualification/evidence/0/created_at"
        for issue in _issues(result, "invalid_qualification_evidence")
    )


def test_provenance_uses_the_first_schema_valid_evidence_timestamp():
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {
                "evidence": [
                    _evidence(created_at="not-a-timestamp"),
                    _evidence(evidence_id="valid", created_at="2026-07-26T04:01:00Z"),
                ]
            },
        )
    )

    assert result.profile["qualification"]["evidence"] == [
        _evidence(evidence_id="valid", created_at="2026-07-26T04:01:00Z")
    ]
    assert result.profile["provenance"]["records"][0]["observed_at"] == (
        "2026-07-26T04:01:00Z"
    )


def test_empty_evidence_history_is_preserved_without_implying_qualification():
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {"qualification_status": "not_qualified", "evidence": []},
        )
    )

    assert result.profile["qualification"] == {
        "evidence": [],
        "status": "not_qualified",
    }


def test_weak_signals_and_generic_status_never_establish_qualification():
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {
                "status": "succeeded",
                "production_ready": True,
                "qualified": True,
                "qualification": False,
            },
        )
    )

    assert "qualification" not in result.profile
    assert _issues(result, "generic_status_ignored")
    assert len(_issues(result, "qualification_boolean_ignored")) == 3


@pytest.mark.parametrize("status", ["not_qualified", "failed", "blocked"])
def test_non_promoting_statuses_are_preserved(status):
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json", {"qualification_status": status}
        )
    )

    assert result.complete is False
    assert result.profile["qualification"] == {"status": status}
    assert not _issues(result, "invalid_qualification_status")


def test_qualified_requires_passing_evidence_but_preserves_valid_history():
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {
                "qualification_status": "qualified",
                "evidence": [
                    _evidence("fail"),
                    _evidence("incomplete", evidence_id="retry"),
                ],
            },
        )
    )

    assert result.profile["qualification"] == {
        "evidence": [_evidence("fail"), _evidence("incomplete", evidence_id="retry")]
    }
    assert _issues(result, "qualification_without_passing_evidence")


@pytest.mark.parametrize("status", ["pass", True, ["qualified"]])
def test_rejects_unknown_or_untyped_qualification_status(status):
    result = import_legacy_model_profile(
        qualification=_source(
            "/manifests/qualification.json",
            {"qualification_status": status, "evidence": [_evidence()]},
        )
    )

    assert result.profile["qualification"] == {"evidence": [_evidence()]}
    assert _issues(result, "invalid_qualification_status")


def test_multiple_qualification_sources_report_deterministic_conflicts():
    first = _source(
        "/manifests/qualification-first.json",
        {"qualification_status": "blocked", "evidence": [_evidence("incomplete")]},
    )
    second = _source(
        "/manifests/qualification-second.json",
        {"qualification_status": "failed", "evidence": [_evidence("fail")]},
    )

    result = _import_legacy_sources(
        (("qualification", first), ("qualification", second))
    )

    assert result.profile["qualification"]["status"] == "blocked"
    conflicts = _issues(result, "conflicting_value")
    assert {(issue.pointer, issue.sources) for issue in conflicts} == {
        (
            "/qualification/evidence",
            (
                "/manifests/qualification-first.json",
                "/manifests/qualification-second.json",
            ),
        ),
        (
            "/qualification/status",
            (
                "/manifests/qualification-first.json",
                "/manifests/qualification-second.json",
            ),
        ),
    }
    records = result.profile["provenance"]["records"]
    assert [record["source"] for record in records] == [
        "/manifests/qualification-first.json"
    ]
