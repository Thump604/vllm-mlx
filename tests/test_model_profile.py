# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ModelProfile v1 validation and canonical hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from vllm_mlx.model_profile import (
    ModelProfileValidationError,
    canonical_subject,
    collect_model_profile_issues,
    compute_subject_digest,
    validate_model_profile,
)

ROOT = Path(__file__).parents[1]
PROFILE_SCHEMA_PATH = ROOT / "schemas" / "model-profile-v1.schema.json"
EXAMPLE_PATH = ROOT / "schemas" / "examples" / "model-profile-v1.example.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def profile_schema() -> dict:
    return _load(PROFILE_SCHEMA_PATH)


@pytest.fixture
def example_profile() -> dict:
    return _load(EXAMPLE_PATH)


def _refresh_digest(profile: dict) -> None:
    profile["subject_digest"] = compute_subject_digest(profile)


def test_canonical_subject_excludes_metadata_qualification_and_extensions(
    example_profile,
):
    subject = canonical_subject(example_profile)

    assert set(subject) == {
        "identity",
        "artifact",
        "capabilities",
        "serving",
        "hardware_fit",
        "provenance",
    }
    assert "profile_id" not in subject
    assert "profile_revision" not in subject
    assert "description" not in subject
    assert "qualification" not in subject
    assert "extensions" not in subject


def test_subject_digest_is_deterministic_and_changes_for_subject_fields(
    example_profile,
):
    digest = compute_subject_digest(example_profile)
    assert digest == compute_subject_digest(copy.deepcopy(example_profile))

    changed = copy.deepcopy(example_profile)
    changed["artifact"]["model_type"] = "changed"
    assert compute_subject_digest(changed) != digest

    metadata_only = copy.deepcopy(example_profile)
    metadata_only["profile_revision"] += 1
    metadata_only["description"] = "Changed catalog description"
    metadata_only["qualification"]["reason"] = "Changed evidence summary"
    metadata_only["extensions"] = {"example.org": {"catalog_label": "changed"}}
    assert compute_subject_digest(metadata_only) == digest


def test_subject_digest_has_independent_rfc8785_vector():
    profile = {"identity": {"b": 1, "a": 2}}
    canonical_bytes = b'{"identity":{"a":2,"b":1}}'

    assert hashlib.sha256(canonical_bytes).hexdigest() == (
        "01e208d1b2aef5c8a62c2c6cb7350a6dfada51760f2b0a70abe2c788fc52a237"
    )
    assert (
        compute_subject_digest(profile) == hashlib.sha256(canonical_bytes).hexdigest()
    )


def test_canonical_example_validates_after_assigning_computed_digest(
    example_profile, profile_schema
):
    validate_model_profile(example_profile, profile_schema)


def test_schema_failures_aggregate(profile_schema):
    invalid = {"schema_version": "one", "unexpected": True}

    issues = collect_model_profile_issues(invalid, profile_schema)

    assert len(issues) >= 2
    assert all(issue.code == "schema_invalid" for issue in issues)
    assert {issue.pointer for issue in issues} >= {"/schema_version", ""}


def test_schema_format_validation_rejects_invalid_evidence_timestamp(
    example_profile, profile_schema
):
    example_profile["qualification"] = {
        "status": "failed",
        "evidence": [
            {
                "evidence_id": "run-1",
                "kind": "qualification",
                "location": "/runs/run-1.json",
                "artifact_sha256": "a" * 64,
                "result": "fail",
                "hardware_fingerprint": "apple-m2",
                "workload_id": "smoke",
                "subject_digest": example_profile["subject_digest"],
                "created_at": "not-a-timestamp",
            }
        ],
    }

    issues = collect_model_profile_issues(example_profile, profile_schema)

    assert any(
        issue.code == "schema_invalid"
        and issue.pointer == "/qualification/evidence/0/created_at"
        for issue in issues
    )


def test_all_limit_relationships_are_checked(example_profile, profile_schema):
    limits = example_profile["serving"]["limits"]
    limits.update(
        {
            "advertised_context": 10,
            "serving_context": 20,
            "max_output_tokens": 30,
            "max_request_output_tokens": 25,
            "max_kv_size": 5,
        }
    )
    _refresh_digest(example_profile)

    codes = {
        issue.code
        for issue in collect_model_profile_issues(example_profile, profile_schema)
    }

    assert codes >= {
        "serving_context_exceeds_advertised",
        "output_exceeds_context",
        "request_output_exceeds_context",
        "default_output_exceeds_request_cap",
        "kv_smaller_than_context",
    }


def test_request_policy_forbidden_overlap_is_rejected(example_profile, profile_schema):
    policy = example_profile["serving"]["request_policy"]
    policy["required_fields"] = {"temperature": True}
    policy["allowed_fields"] = ["max_tokens"]
    policy["forbidden_fields"] = ["temperature", "max_tokens"]
    _refresh_digest(example_profile)

    issues = collect_model_profile_issues(example_profile, profile_schema)

    assert any(issue.code == "request_policy_conflict" for issue in issues)


def test_per_request_and_activation_features_must_link_to_matching_policy(
    example_profile, profile_schema
):
    features = example_profile["serving"]["features"]
    features["streaming"]["control_field"] = "reasoning_effort"
    features["continuous_batching"]["control_field"] = "limits.serving_context"
    _refresh_digest(example_profile)

    codes = [
        issue.code
        for issue in collect_model_profile_issues(example_profile, profile_schema)
    ]

    assert codes.count("request_feature_not_allowed") == 1
    assert codes.count("activation_feature_not_allowed") == 1


def test_inactive_features_cannot_expose_controls(example_profile, profile_schema):
    feature = example_profile["serving"]["features"]["prefix_cache"]
    feature["control"] = "activation"
    feature["control_field"] = "features.prefix_cache"
    example_profile["serving"]["activation_policy"]["owner_override_fields"].append(
        "features.prefix_cache"
    )
    _refresh_digest(example_profile)

    issues = collect_model_profile_issues(example_profile, profile_schema)

    assert any(issue.code == "schema_invalid" for issue in issues)


def test_qualification_evidence_must_bind_to_subject_digest(
    example_profile, profile_schema
):
    example_profile["qualification"] = {
        "status": "qualified",
        "evidence": [
            {
                "evidence_id": "run-1",
                "kind": "qualification",
                "location": "/runs/run-1.json",
                "artifact_sha256": "a" * 64,
                "result": "pass",
                "hardware_fingerprint": "apple-m2",
                "workload_id": "smoke",
                "subject_digest": "b" * 64,
                "created_at": "2026-07-21T00:00:00Z",
            }
        ],
    }
    _refresh_digest(example_profile)

    codes = {
        issue.code
        for issue in collect_model_profile_issues(example_profile, profile_schema)
    }

    assert "evidence_digest_mismatch" in codes
    assert "qualification_without_bound_pass" in codes


def test_hardware_fit_requires_provenance_coverage(example_profile, profile_schema):
    example_profile["provenance"]["records"][1]["field_paths"].remove("/hardware_fit")
    _refresh_digest(example_profile)

    issues = collect_model_profile_issues(example_profile, profile_schema)

    assert any(
        issue.code == "provenance_kind_mismatch"
        and issue.pointer.startswith("/hardware_fit/0")
        for issue in issues
    )


def test_maintainer_policy_cannot_establish_provider_and_measured_facts(
    example_profile, profile_schema
):
    example_profile["hardware_fit"][0]["method"] = "measured"
    example_profile["hardware_fit"][0]["measured_peak_memory_bytes"] = 5_000_000_000
    example_profile["provenance"]["records"] = [
        {
            "field_paths": [
                "/identity",
                "/artifact",
                "/capabilities",
                "/serving",
                "/hardware_fit",
            ],
            "kind": "maintainer_policy",
            "source": "local-policy.yaml",
            "revision": None,
            "sha256": "a" * 64,
            "rule_id": "local-policy-v1",
            "observed_at": "2026-07-21T00:00:00Z",
        }
    ]
    _refresh_digest(example_profile)

    issues = collect_model_profile_issues(example_profile, profile_schema)
    pointers = {issue.pointer for issue in issues}
    codes = {issue.code for issue in issues}

    assert "/identity/resolved_revision" in pointers
    assert any(pointer.startswith("/artifact/") for pointer in pointers)
    assert any(pointer.startswith("/capabilities/") for pointer in pointers)
    assert any(
        pointer.startswith("/serving/sampling/provider_defaults/")
        for pointer in pointers
    )
    assert any(pointer.startswith("/hardware_fit/0/") for pointer in pointers)
    assert "invalid_provenance_claim" in codes


def test_provenance_rejects_nonexistent_descendants_and_unbound_sources(
    example_profile, profile_schema
):
    provider = example_profile["provenance"]["records"][0]
    provider["field_paths"] = [
        "/identity/resolved_revision/not-a-real-field",
        "/artifact",
        "/serving/sampling/provider_defaults",
    ]
    provider["revision"] = None
    provider["sha256"] = None
    derived = example_profile["provenance"]["records"][1]
    derived["rule_id"] = None
    derived["observed_at"] = None
    _refresh_digest(example_profile)

    codes = {
        issue.code
        for issue in collect_model_profile_issues(example_profile, profile_schema)
    }

    assert "invalid_provenance_path" in codes
    assert "provider_provenance_unbound" in codes
    assert "derived_provenance_rule_missing" in codes
    assert "provenance_timestamp_missing" in codes


def test_measured_hardware_fit_requires_a_measurement(example_profile, profile_schema):
    fit = example_profile["hardware_fit"][0]
    fit["method"] = "measured"
    fit["measured_peak_memory_bytes"] = None
    evidence_record = example_profile["provenance"]["records"][1]
    evidence_record["kind"] = "measured_result"
    evidence_record["sha256"] = "e" * 64
    evidence_record["rule_id"] = None
    _refresh_digest(example_profile)

    issues = collect_model_profile_issues(example_profile, profile_schema)

    assert any(
        issue.code == "measured_hardware_fit_without_measurement" for issue in issues
    )
    assert any(
        issue.code == "measured_hardware_fit_without_bound_evidence" for issue in issues
    )


def test_measured_hardware_fit_requires_bound_passing_qualification_evidence(
    example_profile, profile_schema
):
    fit = example_profile["hardware_fit"][0]
    fit["method"] = "measured"
    fit["measured_peak_memory_bytes"] = 5_000_000_000
    example_profile["provenance"]["records"][1]["field_paths"].remove("/hardware_fit")
    example_profile["provenance"]["records"].append(
        {
            "field_paths": ["/hardware_fit"],
            "kind": "measured_result",
            "source": "/runs/run-1.json",
            "revision": None,
            "sha256": "e" * 64,
            "rule_id": None,
            "observed_at": "2026-07-21T00:00:00Z",
        }
    )
    _refresh_digest(example_profile)

    evidence = {
        "evidence_id": "run-1",
        "kind": "qualification",
        "location": "/runs/run-1.json",
        "artifact_sha256": "a" * 64,
        "result": "pass",
        "hardware_fingerprint": "apple-m2",
        "workload_id": "smoke",
        "subject_digest": example_profile["subject_digest"],
        "created_at": "2026-07-21T00:00:00Z",
    }
    example_profile["qualification"]["evidence"] = [evidence]

    validate_model_profile(example_profile, profile_schema)


def test_local_only_profile_uses_local_and_derived_provenance(
    example_profile, profile_schema
):
    identity = example_profile["identity"]
    identity.update(
        {
            "provider": "local",
            "repository_id": None,
            "requested_revision": None,
            "resolved_revision": None,
        }
    )
    example_profile["artifact"]["source_uri"] = "local://example-model"
    example_profile["serving"]["sampling"]["provider_defaults"] = {}
    example_profile["provenance"]["records"][0] = {
        "field_paths": ["/artifact"],
        "kind": "derived_recommendation",
        "source": "/manifests/conversion.json",
        "revision": None,
        "sha256": "a" * 64,
        "rule_id": "conversion-manifest-v1",
        "observed_at": "2026-07-21T00:00:00Z",
    }
    example_profile["provenance"]["records"].append(
        {
            "field_paths": [
                "/identity",
                "/serving/sampling/provider_defaults",
            ],
            "kind": "maintainer_policy",
            "source": "/catalog/local-example.json",
            "revision": None,
            "sha256": "b" * 64,
            "rule_id": "local-catalog-v1",
            "observed_at": "2026-07-21T00:00:00Z",
        }
    )
    _refresh_digest(example_profile)

    validate_model_profile(example_profile, profile_schema)


def test_provenance_paths_use_rfc6901_escaping(example_profile, profile_schema):
    example_profile["serving"]["template"]["default_kwargs"] = {"a/b~c": True}
    _refresh_digest(example_profile)

    validate_model_profile(example_profile, profile_schema)


def test_noncanonical_numeric_subject_becomes_validation_issue(
    example_profile, profile_schema
):
    example_profile["serving"]["features"]["continuous_batching"]["settings"][
        "max_concurrency"
    ] = 2**60

    issues = collect_model_profile_issues(example_profile, profile_schema)

    assert any(issue.code == "subject_not_canonicalizable" for issue in issues)


def test_validate_model_profile_raises_all_semantic_failures(
    example_profile, profile_schema
):
    example_profile["serving"]["request_policy"]["forbidden_fields"] = ["stream"]
    example_profile["serving"]["features"]["streaming"]["control_field"] = (
        "reasoning_effort"
    )
    _refresh_digest(example_profile)

    with pytest.raises(ModelProfileValidationError) as raised:
        validate_model_profile(example_profile, profile_schema)

    assert {issue.code for issue in raised.value.issues} >= {
        "request_policy_conflict",
        "request_feature_not_allowed",
    }
