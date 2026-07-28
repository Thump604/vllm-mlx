# SPDX-License-Identifier: Apache-2.0
"""Contract tests for bounded legacy ModelProfile import results."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vllm_mlx.model_profile import (
    ModelProfileValidationError,
    compute_subject_digest,
)
from vllm_mlx.model_profile_import import (
    collect_import_result_issues,
    validate_import_result,
)

ROOT = Path(__file__).parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _source() -> dict:
    return {
        "kind": "acquisition",
        "location": "/manifests/acquisition.json",
        "sha256": "a" * 64,
    }


def test_incomplete_import_result_requires_explicit_issues():
    result = {
        "schema_version": 1,
        "complete": False,
        "sources": [_source()],
        "profile": {"schema_version": 1},
        "issues": [
            {
                "code": "missing_required_fact",
                "severity": "error",
                "pointer": "/identity/resolved_revision",
                "sources": ["/manifests/acquisition.json"],
                "detail": "an immutable revision is required",
            }
        ],
    }

    assert (
        collect_import_result_issues(
            result,
            _load("schemas/model-profile-import-result-v1.schema.json"),
            _load("schemas/model-profile-v1.schema.json"),
        )
        == ()
    )


def test_complete_import_result_requires_a_valid_complete_profile():
    profile = _load("schemas/examples/model-profile-v1.example.json")
    result = {
        "schema_version": 1,
        "complete": True,
        "sources": [_source()],
        "profile": profile,
        "issues": [],
    }
    import_schema = _load("schemas/model-profile-import-result-v1.schema.json")
    profile_schema = _load("schemas/model-profile-v1.schema.json")

    assert collect_import_result_issues(result, import_schema, profile_schema) == ()

    invalid = copy.deepcopy(result)
    invalid["issues"] = [
        {
            "code": "unresolved_conflict",
            "severity": "error",
            "pointer": "/artifact",
            "sources": ["/manifests/acquisition.json"],
            "detail": "artifact facts conflict",
        }
    ]
    assert any(
        issue.code == "import_schema_invalid"
        for issue in collect_import_result_issues(
            invalid, import_schema, profile_schema
        )
    )


def test_complete_import_result_prefixes_profile_semantic_failures():
    profile = _load("schemas/examples/model-profile-v1.example.json")
    profile["serving"]["limits"]["serving_context"] = (
        profile["serving"]["limits"]["advertised_context"] + 1
    )
    profile["subject_digest"] = compute_subject_digest(profile)
    result = {
        "schema_version": 1,
        "complete": True,
        "sources": [_source()],
        "profile": profile,
        "issues": [],
    }

    issues = collect_import_result_issues(
        result,
        _load("schemas/model-profile-import-result-v1.schema.json"),
        _load("schemas/model-profile-v1.schema.json"),
    )

    assert any(
        issue.code == "serving_context_exceeds_advertised"
        and issue.pointer == "/profile/serving/limits/serving_context"
        for issue in issues
    )


def test_validate_import_result_raises_with_collected_issues():
    profile = _load("schemas/examples/model-profile-v1.example.json")
    result = {
        "schema_version": 1,
        "complete": True,
        "sources": [_source()],
        "profile": profile,
        "issues": [],
    }
    import_schema = _load("schemas/model-profile-import-result-v1.schema.json")
    profile_schema = _load("schemas/model-profile-v1.schema.json")

    validate_import_result(result, import_schema, profile_schema)

    invalid = copy.deepcopy(result)
    invalid["profile"]["serving"]["limits"]["serving_context"] = (
        invalid["profile"]["serving"]["limits"]["advertised_context"] + 1
    )
    invalid["profile"]["subject_digest"] = compute_subject_digest(invalid["profile"])
    with pytest.raises(ModelProfileValidationError) as caught:
        validate_import_result(invalid, import_schema, profile_schema)

    assert any(
        issue.code == "serving_context_exceeds_advertised"
        and issue.pointer == "/profile/serving/limits/serving_context"
        for issue in caught.value.issues
    )
