# SPDX-License-Identifier: Apache-2.0
"""P4.2 tests for the injected ModelProfile catalog loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.catalog import CatalogValidationError, load_catalog
from vllm_mlx.model_profile import compute_subject_digest

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "schemas" / "model-profile-v1.schema.json"
EXAMPLE_PATH = ROOT / "schemas" / "examples" / "model-profile-v1.example.json"


def _profile(**changes):
    profile = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for key, value in changes.items():
        profile[key] = value
    profile["subject_digest"] = compute_subject_digest(profile)
    return profile


def _write(root: Path, relative: str, profile: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile), encoding="utf-8")


def test_loads_valid_profiles_in_deterministic_order_and_gets_exact_revision(tmp_path):
    _write(
        tmp_path, "zeta/profile.json", _profile(profile_id="zeta", profile_revision=1)
    )
    _write(tmp_path, "alpha/v2.json", _profile(profile_id="alpha", profile_revision=2))
    _write(tmp_path, "alpha/v1.json", _profile(profile_id="alpha", profile_revision=1))

    catalog = load_catalog(tmp_path, schema_path=SCHEMA_PATH)

    profiles = catalog.list_profiles()
    assert [(p["profile_id"], p["profile_revision"]) for p in profiles] == [
        ("alpha", 1),
        ("alpha", 2),
        ("zeta", 1),
    ]
    assert catalog.get("alpha")["profile_revision"] == 2
    assert catalog.get("alpha", 1)["profile_revision"] == 1


def test_returns_defensive_copies(tmp_path):
    _write(tmp_path, "profile.json", _profile(profile_id="copy-safe"))
    catalog = load_catalog(tmp_path, schema_path=SCHEMA_PATH)

    returned = catalog.get("copy-safe")
    returned["identity"]["artifact_id"] = "mutated"
    returned_list = catalog.list_profiles()
    returned_list[0]["profile_id"] = "mutated"

    assert catalog.get("copy-safe")["identity"]["artifact_id"] != "mutated"
    assert catalog.get("copy-safe")["profile_id"] == "copy-safe"


def test_rejects_duplicate_profile_id_and_revision(tmp_path):
    _write(tmp_path, "one.json", _profile(profile_id="duplicate", profile_revision=1))
    _write(tmp_path, "two.json", _profile(profile_id="duplicate", profile_revision=1))

    with pytest.raises(CatalogValidationError, match="duplicate profile identity"):
        load_catalog(tmp_path, schema_path=SCHEMA_PATH)


def test_rejects_stale_subject_digest(tmp_path):
    profile = _profile(profile_id="stale")
    profile["subject_digest"] = "0" * 64
    _write(tmp_path, "stale.json", profile)

    with pytest.raises(CatalogValidationError, match="subject_digest"):
        load_catalog(tmp_path, schema_path=SCHEMA_PATH)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("profile_id",), "/tmp/profile"),
        (("identity", "artifact_id"), "/tmp/artifact"),
        (("artifact", "source_uri"), "/tmp/model"),
        (("artifact", "source_uri"), "file:///tmp/model"),
        (("artifact", "source_uri"), r"C:\\models\\model"),
    ],
)
def test_rejects_local_absolute_identity_or_source_uri(tmp_path, field_path, value):
    profile = _profile(profile_id="path-safe")
    target = profile
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    if field_path != ("profile_id",):
        profile["subject_digest"] = compute_subject_digest(profile)
    _write(tmp_path, "path.json", profile)

    with pytest.raises(CatalogValidationError, match="local absolute path"):
        load_catalog(tmp_path, schema_path=SCHEMA_PATH)


def test_schema_and_semantic_validation_are_applied(tmp_path):
    profile = _profile(profile_id="invalid")
    profile["serving"]["limits"]["serving_context"] = 1
    _write(tmp_path, "invalid.json", profile)

    with pytest.raises(CatalogValidationError, match="invalid ModelProfile"):
        load_catalog(tmp_path, schema_path=SCHEMA_PATH)


def test_missing_profile_or_revision_is_explicit(tmp_path):
    _write(tmp_path, "profile.json", _profile(profile_id="present", profile_revision=2))
    catalog = load_catalog(tmp_path, schema_path=SCHEMA_PATH)

    with pytest.raises(KeyError):
        catalog.get("missing")
    with pytest.raises(KeyError):
        catalog.get("present", 1)
