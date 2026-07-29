# SPDX-License-Identifier: Apache-2.0
"""Pure contract tests for profile-directed backend loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.loader_compat import (
    LoaderCompatibilityError,
    build_load_receipt,
    resolve_loader_policy,
)
from vllm_mlx.model_profile import (
    ModelProfileValidationError,
    canonical_subject,
    compute_subject_digest,
    validate_model_profile,
)

ROOT = Path(__file__).parents[1]


def _profile() -> dict:
    profile = json.loads(
        (ROOT / "schemas/examples/model-profile-v1.example.json").read_text()
    )
    profile["backend"] = {
        "backend_id": "mlx-lm",
        "loader_route": "mlx_lm",
        "weight_policy": {
            "mode": "strict",
            "allowed_unmatched_weight_prefixes": [],
        },
        "dependency_constraints": {
            "mlx": ">=0.32.0",
            "mlx-lm": ">=0.31.3",
        },
    }
    profile["provenance"]["records"].append(
        {
            "field_paths": ["/backend"],
            "kind": "derived_recommendation",
            "source": "tests/test_loader_compat.py",
            "revision": "1",
            "sha256": "b" * 64,
            "rule_id": "test-loader-contract-v1",
            "observed_at": "2026-07-29T00:00:00Z",
        }
    )
    profile["subject_digest"] = compute_subject_digest(profile)
    for record in profile["qualification"]["evidence"]:
        record["subject_digest"] = profile["subject_digest"]
    return profile


def _schema() -> dict:
    return json.loads((ROOT / "schemas/model-profile-v1.schema.json").read_text())


def test_backend_contract_is_subject_bound_and_valid():
    profile = _profile()

    validate_model_profile(profile, _schema())
    assert "backend" in canonical_subject(profile)


def test_backend_policy_change_changes_subject_digest():
    profile = _profile()
    original = compute_subject_digest(profile)
    profile["backend"]["weight_policy"]["mode"] = "allowlisted_extras"
    profile["backend"]["weight_policy"]["allowed_unmatched_weight_prefixes"] = [
        "vision_tower."
    ]

    assert compute_subject_digest(profile) != original


def test_strict_backend_rejects_unmatched_weight_allowlist():
    profile = _profile()
    profile["backend"]["weight_policy"]["allowed_unmatched_weight_prefixes"] = [
        "vision_tower."
    ]
    profile["subject_digest"] = compute_subject_digest(profile)
    for record in profile["qualification"]["evidence"]:
        record["subject_digest"] = profile["subject_digest"]

    with pytest.raises(ModelProfileValidationError, match="strict loader policy"):
        validate_model_profile(profile, _schema())


def test_loader_policy_requires_explicit_backend_contract():
    profile = _profile()
    profile.pop("backend")

    with pytest.raises(LoaderCompatibilityError, match="does not declare"):
        resolve_loader_policy(profile)


def test_load_receipt_records_policy_without_inventing_load_result(tmp_path):
    profile = _profile()

    receipt = build_load_receipt(
        profile,
        tmp_path,
        installed_versions={"mlx": "0.32.0", "mlx-lm": "0.31.3"},
    )

    assert receipt["artifact_path"] == str(tmp_path.resolve())
    assert receipt["loader_policy"]["loader_route"] == "mlx_lm"
    assert receipt["observed_unmatched_weight_keys"] is None
    assert receipt["load_result"] == "not_started"
