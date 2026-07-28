# SPDX-License-Identifier: Apache-2.0
"""Focused regression tests for workflow-to-profile evidence binding."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from vllm_mlx.model_profile import compute_subject_digest
from vllm_mlx.model_profile_workflow import (
    finalize_workflow_profile,
    load_workflow_profile_evidence,
)

ROOT = Path(__file__).parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _workflow_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    revision = "0" * 40
    acquisition = {
        "kind": "vllm-mlx-model-artifact",
        "operation_id": "a" * 64,
        "model_id": "example-org/example-model",
        "revision": "main",
        "resolved_revision": revision,
        "created_at": "2026-07-28T00:00:00Z",
        "inspection": {
            "total_size_bytes": 4294967296,
            "model_family": {
                "model_type": "example",
                "architectures": ["ExampleForCausalLM"],
                "torch_dtype": "float16",
                "quantization": {
                    "method": "affine",
                    "bits": 4,
                    "group_size": 64,
                },
            },
        },
    }
    acquisition_path = tmp_path / "acquisition.json"
    _write_json(acquisition_path, acquisition)
    acquisition_sha256 = hashlib.sha256(acquisition_path.read_bytes()).hexdigest()

    conversion = {
        "kind": "vllm-mlx-conversion",
        "operation_id": "b" * 64,
        "status": "succeeded",
        "completed_at": "2026-07-28T00:01:00Z",
        "identity": {"source": {"acquisition_manifest_sha256": acquisition_sha256}},
        "recipe": {"q_bits": 4, "q_group_size": 64, "q_mode": "affine"},
        "environment": {"python": "3.12", "mlx_lm": "0.30.0"},
        "output_inspection": deepcopy(acquisition["inspection"]),
        "artifact_validation": {"artifact_sha256": "c" * 64},
    }
    conversion_path = tmp_path / "conversion.json"
    _write_json(conversion_path, conversion)

    registration = {
        "kind": "vllm-mlx-model-registration",
        "created_at": "2026-07-28T00:02:00Z",
        "model_id": "example-model-mlx-q4",
        "artifact_id": "example-model-mlx-q4",
        "served_model_name": "example-model",
        "serving_defaults": {"temperature": 0.7, "top_p": 0.9},
        "parser_policy": {"tool_call_parser": None, "reasoning_parser": None},
        "source_manifests": {
            "acquisition": {"payload": acquisition},
            "conversion": {"payload": conversion},
        },
    }
    registration_path = tmp_path / "registration.json"
    _write_json(registration_path, registration)
    return acquisition_path, conversion_path, registration_path


def test_loads_hash_bound_workflow_evidence_without_promotion(tmp_path):
    acquisition, conversion, registration = _workflow_manifests(tmp_path)

    evidence = load_workflow_profile_evidence(
        acquisition_manifest=acquisition,
        conversion_manifest=conversion,
        registration_manifest=registration,
    )

    output = evidence.as_dict()
    assert output["promotion_required"] is True
    assert output["production_ready"] is False
    assert output["import"]["complete"] is False
    assert output["acquisition"]["resolved_revision"] == "0" * 40
    assert output["conversion"]["artifact_sha256"] == "c" * 64
    assert output["conversion"]["recipe"] == {
        "q_bits": 4,
        "q_group_size": 64,
        "q_mode": "affine",
    }
    assert [item["kind"] for item in output["sources"]] == [
        "acquisition",
        "conversion",
        "registration",
    ]
    assert all(len(item["sha256"]) == 64 for item in output["sources"])


@pytest.mark.parametrize(
    "defect", ["conversion_link", "registration_copy", "artifact_digest"]
)
def test_rejects_unbound_or_incomplete_workflow_manifests(tmp_path, defect):
    acquisition, conversion, registration = _workflow_manifests(tmp_path)
    conversion_payload = json.loads(conversion.read_text())
    registration_payload = json.loads(registration.read_text())
    if defect == "conversion_link":
        conversion_payload["identity"]["source"]["acquisition_manifest_sha256"] = (
            "d" * 64
        )
        _write_json(conversion, conversion_payload)
    elif defect == "registration_copy":
        registration_payload["source_manifests"]["conversion"]["payload"][
            "status"
        ] = "failed"
        _write_json(registration, registration_payload)
    else:
        conversion_payload["artifact_validation"] = {}
        _write_json(conversion, conversion_payload)
        registration_payload["source_manifests"]["conversion"][
            "payload"
        ] = conversion_payload
        _write_json(registration, registration_payload)

    with pytest.raises(ValueError):
        load_workflow_profile_evidence(
            acquisition_manifest=acquisition,
            conversion_manifest=conversion,
            registration_manifest=registration,
        )


def test_finalizes_only_an_explicit_profile_candidate(tmp_path):
    acquisition, conversion, registration = _workflow_manifests(tmp_path)
    evidence = load_workflow_profile_evidence(
        acquisition_manifest=acquisition,
        conversion_manifest=conversion,
        registration_manifest=registration,
    )
    candidate = _load("schemas/examples/model-profile-v1.example.json")
    candidate["identity"].update(
        {
            "repository_id": "example-org/example-model",
            "requested_revision": "main",
            "resolved_revision": "0" * 40,
            "artifact_id": "example-model-mlx-q4",
            "served_model_name": "example-model",
        }
    )
    candidate["artifact"].update(
        {
            "source_uri": "example-org/example-model",
            "format": "mlx",
            "model_type": "example",
            "architectures": ["ExampleForCausalLM"],
            "dtype": "float16",
            "size_bytes": 4294967296,
        }
    )
    candidate["artifact"]["quantization"].update(
        {
            "method": "affine",
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "source": "conversion_recipe",
        }
    )
    candidate["serving"]["sampling"]["profile_defaults"] = {
        "temperature": 0.7,
        "top_p": 0.9,
    }
    candidate["provenance"]["records"].extend(
        deepcopy(evidence.imported.profile["provenance"]["records"])
    )
    candidate["subject_digest"] = compute_subject_digest(candidate)

    result = finalize_workflow_profile(
        evidence,
        candidate,
        profile_schema=_load("schemas/model-profile-v1.schema.json"),
        import_schema=_load("schemas/model-profile-import-result-v1.schema.json"),
    )

    assert result.complete is True
    assert result.profile == candidate
