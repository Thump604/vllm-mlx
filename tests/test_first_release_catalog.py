# SPDX-License-Identifier: Apache-2.0
"""P4.4 portable first-release catalog and hardware-envelope fixtures."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from vllm_mlx.catalog import load_catalog
from vllm_mlx.model_profile import compute_subject_digest

ROOT = Path(__file__).parents[1]
PROFILE_ROOT = ROOT / "catalog" / "profiles"
HARDWARE_ROOT = ROOT / "catalog" / "hardware"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_first_release_profiles_load_through_existing_catalog_loader():
    catalog = load_catalog(PROFILE_ROOT)

    assert [profile["profile_id"] for profile in catalog.list_profiles()] == [
        "laguna-s-2.1-mlx-q4",
        "qwen3.6-35b-a3b-8bit",
    ]


def test_profile_identities_and_artifact_hashes_are_immutable_fixtures():
    qwen = load_catalog(PROFILE_ROOT).get("qwen3.6-35b-a3b-8bit", 1)
    laguna = load_catalog(PROFILE_ROOT).get("laguna-s-2.1-mlx-q4", 1)

    assert qwen["identity"]["resolved_revision"] == (
        "e06a74e6236a60c8367e1a3214e83d8b61b637b0"
    )
    assert qwen["artifact"]["hashes"] == {
        "config_sha256": "0c37843fa49b0faf20d02582b66b6bed236f42db6f2b54bd6f84dffad3a3b365",
        "tokenizer_sha256": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
        "chat_template_sha256": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
        "generation_config_sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "weights_manifest_sha256": "09e48724ae52eaf03ecb2b042530dfbbf058c038e857bb8d751747b680ae09ab",
    }
    assert laguna["identity"]["resolved_revision"] == (
        "a50e85e7e0aae7b0a504d156bd36a616ec9fea38"
    )
    assert laguna["artifact"]["hashes"] == {
        "config_sha256": "8440d3ec23e275aa62bba1371c20cee4a72906fdc33ca37966ba7cd83472847b",
        "tokenizer_sha256": "ff04405d2d1e1b6c77a8be25f0fce9371003a558b055c23248d9e8ca1d956d92",
        "chat_template_sha256": "2d3c724b3c2e9eb71fe9ccc5423ff268a370a8bfa89e9238b6de14fe000825c8",
        "generation_config_sha256": "2deeac08584c9177028e108a994e37dffd06acf61ca429dc064f76fee52e2bea",
        "weights_manifest_sha256": "e0974415a8e0a449f2b80e39fb6cc7ee9594ecff9c6aed00a93f96427678000a",
    }
    assert qwen["subject_digest"] == compute_subject_digest(qwen)
    assert laguna["subject_digest"] == compute_subject_digest(laguna)


def test_laguna_is_artifact_only_and_not_exposed_by_structural_fixture():
    profile = load_catalog(PROFILE_ROOT).get("laguna-s-2.1-mlx-q4", 1)

    assert profile["qualification"] == {
        "status": "not_qualified",
        "reason": "Artifact-only onboarding fixture; live qualification and exposure are intentionally separate.",
        "evidence": [],
    }
    assert profile["serving"]["limits"]["serving_context"] == 32768
    assert profile["serving"]["sampling"]["profile_defaults"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "min_p": 0.0,
    }
    assert all(
        feature["mode"] in {"deferred", "not_supported"}
        for feature in profile["serving"]["features"].values()
    )


def test_qwen_fixture_uses_the_multimodal_route_declared_by_its_artifact():
    profile = load_catalog(PROFILE_ROOT).get("qwen3.6-35b-a3b-8bit", 1)

    assert profile["capabilities"]["modalities"] == ["text", "image", "video"]
    assert profile["serving"]["route"] == "multimodal"


def test_hardware_envelopes_are_explicit_without_invented_model_fit():
    envelopes = [_read(path) for path in sorted(HARDWARE_ROOT.glob("*.json"))]

    assert {item["hardware_profile_id"] for item in envelopes} == {
        "apple-silicon-64gb",
        "apple-silicon-128gb",
    }
    for envelope in envelopes:
        assert envelope["platform"] == "Darwin/arm64"
        assert all(
            fit["status"] == "unknown" and fit["method"] == "not_measured"
            for fit in envelope["model_fit"]
        )
        assert all(
            fit["serving_context"] is None and fit["max_concurrency"] is None
            for fit in envelope["model_fit"]
        )


def test_catalog_fixtures_contain_no_local_absolute_paths():
    payloads = [
        *(_read(path) for path in PROFILE_ROOT.glob("*.json")),
        *(_read(path) for path in HARDWARE_ROOT.glob("*.json")),
    ]

    json_pointer_roots = tuple(
        f"/{name}"
        for name in (
            "identity",
            "artifact",
            "capabilities",
            "serving",
            "hardware_fit",
            "qualification",
            "provenance",
        )
    )

    def is_local_path(value: str) -> bool:
        if value.startswith(json_pointer_roots):
            return False
        return (
            Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or (
                urlsplit(value).scheme == "file"
                and urlsplit(value).path.startswith("/")
            )
        )

    local_paths = [
        value
        for payload in payloads
        for value in _strings(payload)
        if is_local_path(value)
    ]
    assert local_paths == []
