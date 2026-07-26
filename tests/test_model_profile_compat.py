# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the first legacy ModelProfile compatibility slice."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from vllm_mlx import _model_profile_compat_types, model_profile_compat
from vllm_mlx._model_profile_compat import _import_legacy_sources
from vllm_mlx.model_profile_compat import (
    LegacySourceInput,
    import_legacy_model_profile,
)
from vllm_mlx.model_profile_import import collect_import_result_issues

ROOT = Path(__file__).parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _source(location: str, payload: dict) -> dict:
    return {"location": location, "sha256": "a" * 64, "payload": payload}


def test_import_does_not_mutate_inputs_and_records_provenance():
    acquisition = _source(
        "/manifests/acquisition.json",
        {
            "model_id": "mlx-community/Qwen3-4B-4bit",
            "revision": "main",
            "resolved_revision": "b" * 40,
            "inspection": {
                "inspected_at": "2026-01-01T00:00:00Z",
                "total_size_bytes": 1234,
                "model_family": {
                    "model_type": "qwen3",
                    "architectures": ["Qwen3ForCausalLM"],
                    "torch_dtype": "bfloat16",
                    "quantization": {"bits": 4, "group_size": 64},
                },
            },
        },
    )
    conversion = _source(
        "/manifests/conversion.json",
        {
            "status": "succeeded",
            "completed_at": "2026-01-02T00:00:00Z",
            "resolved_revision": "main",
            "recipe": {"q_bits": 4, "q_group_size": 64, "q_mode": "affine"},
            "output_inspection": acquisition["payload"]["inspection"],
        },
    )
    registration = _source(
        "/manifests/registration.json",
        {
            "created_at": "2026-01-03T00:00:00Z",
            "artifact_id": "qwen3-4b-4bit",
            "served_model_name": "qwen3",
            "preset_alias": "fast-qwen",
            "serving_defaults": {
                "temperature": 0.6,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "parser_policy": {"tool_call_parser": "qwen3_coder"},
        },
    )
    original = deepcopy((acquisition, conversion, registration))

    result = import_legacy_model_profile(
        acquisition=acquisition, conversion=conversion, registration=registration
    )

    assert (acquisition, conversion, registration) == original
    assert result.complete is False
    assert result.profile["identity"]["resolved_revision"] == "b" * 40
    assert result.profile["provenance"]["records"] == [
        {
            "field_paths": [
                "/identity/provider",
                "/identity/repository_id",
                "/identity/requested_revision",
                "/identity/resolved_revision",
                "/artifact/source_uri",
                "/artifact/model_type",
                "/artifact/architectures",
                "/artifact/dtype",
                "/artifact/size_bytes",
                "/artifact/quantization/bits",
                "/artifact/quantization/group_size",
            ],
            "kind": "provider_fact",
            "source": "/manifests/acquisition.json",
            "revision": "b" * 40,
            "sha256": "a" * 64,
            "rule_id": None,
            "observed_at": "2026-01-01T00:00:00Z",
        },
        {
            "field_paths": [
                "/artifact/format",
                "/artifact/quantization/mode",
                "/artifact/quantization/source",
            ],
            "kind": "derived_recommendation",
            "source": "/manifests/conversion.json",
            "revision": None,
            "sha256": "a" * 64,
            "rule_id": "model-profile-compat-v1:conversion",
            "observed_at": "2026-01-02T00:00:00Z",
        },
        {
            "field_paths": [
                "/identity/artifact_id",
                "/identity/served_model_name",
                "/identity/aliases",
                "/serving/sampling/profile_defaults",
                "/serving/template/default_kwargs",
                "/serving/parsers/tool",
            ],
            "kind": "maintainer_policy",
            "source": "/manifests/registration.json",
            "revision": None,
            "sha256": "a" * 64,
            "rule_id": "model-profile-compat-v1:registration",
            "observed_at": "2026-01-03T00:00:00Z",
        },
    ]
    assert any(issue.code == "missing_required_fact" for issue in result.issues)
    assert (
        collect_import_result_issues(
            result.as_dict(),
            _load("schemas/model-profile-import-result-v1.schema.json"),
            _load("schemas/model-profile-v1.schema.json"),
        )
        == ()
    )


def test_registration_feature_flags_do_not_establish_feature_states():
    result = import_legacy_model_profile(
        registration=_source(
            "/manifests/registration.json",
            {
                "artifact_id": "artifact",
                "served_model_name": "served",
                "feature_flags": ["prefix_cache"],
                "production_ready": True,
            },
        )
    )

    declaration = next(
        issue for issue in result.issues if issue.code == "untyped_feature_declaration"
    )
    assert declaration.pointer == "/serving/features/prefix_cache"
    missing = {
        issue.pointer
        for issue in result.issues
        if issue.code == "missing_required_fact"
    }
    assert "/serving/features/kvq4" in missing
    assert "/serving/features/streaming" in missing
    assert any(issue.code == "qualification_boolean_ignored" for issue in result.issues)


def test_failed_conversion_is_quarantined_without_artifact_claims():
    result = import_legacy_model_profile(
        conversion=_source(
            "/manifests/conversion.json",
            {
                "status": "failed",
                "recipe": {"q_bits": 4},
                "returncode": 1,
            },
        )
    )

    assert result.profile == {"schema_version": 1}
    issue = next(
        issue for issue in result.issues if issue.code == "conversion_output_unverified"
    )
    assert issue.pointer == "/artifact"
    assert not any(
        record["field_paths"] == ["/artifact/format"]
        for record in result.profile.get("provenance", {}).get("records", [])
    )


def test_private_engine_retains_first_conflicting_value_in_source_order():
    conversion = _source(
        "/manifests/conversion.json",
        {
            "status": "succeeded",
            "output_inspection": {
                "model_family": {
                    "model_type": "conversion-model",
                    "architectures": ["ConversionModel"],
                }
            },
        },
    )
    acquisition = _source(
        "/manifests/acquisition.json",
        {
            "model_id": "org/acquisition-model",
            "inspection": {
                "model_family": {
                    "model_type": "acquisition-model",
                    "architectures": ["AcquisitionModel"],
                }
            },
        },
    )

    result = _import_legacy_sources(
        (("conversion", conversion), ("acquisition", acquisition))
    )

    assert tuple(source.kind for source in result.sources) == (
        "conversion",
        "acquisition",
    )
    assert result.profile["artifact"]["model_type"] == "conversion-model"
    conflict = next(
        issue
        for issue in result.issues
        if issue.code == "conflicting_value" and issue.pointer == "/artifact/model_type"
    )
    assert conflict.sources == (
        "/manifests/conversion.json",
        "/manifests/acquisition.json",
    )


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        ({"location": "", "sha256": "a" * 64, "payload": {}}, ValueError, "location"),
        (
            {"location": "/source", "sha256": "bad", "payload": {}},
            ValueError,
            "SHA-256",
        ),
        (
            {"location": "/source", "sha256": "a" * 64, "payload": []},
            TypeError,
            "payload",
        ),
    ],
)
def test_legacy_source_input_validates_identity_and_payload(value, exception, message):
    with pytest.raises(exception, match=message):
        LegacySourceInput.from_mapping("acquisition", value)


def test_legacy_source_input_rejects_wrong_kind_and_non_source_dispatch():
    source = LegacySourceInput.from_mapping(
        "acquisition", _source("/source", {"model_id": "org/model"})
    )

    with pytest.raises(ValueError, match="expected conversion source"):
        import_legacy_model_profile(conversion=source)
    with pytest.raises(ValueError, match="at least one legacy source"):
        import_legacy_model_profile()


def test_as_dict_omits_payloads_and_deep_copies_nested_values():
    result = import_legacy_model_profile(
        registration=_source(
            "/manifests/registration.json",
            {"artifact_id": "artifact", "served_model_name": "served"},
        )
    )

    serialized = result.as_dict()
    assert "payload" not in serialized["sources"][0]
    serialized["profile"]["identity"]["artifact_id"] = "changed"
    serialized["issues"][0]["sources"].append("/unrelated")

    assert result.profile["identity"]["artifact_id"] == "artifact"
    assert "/unrelated" not in result.issues[0].sources


def test_dispatcher_exposes_only_three_keyword_only_inputs():
    signature = inspect.signature(import_legacy_model_profile)
    assert model_profile_compat.__all__ == (
        "LegacySourceInput",
        "CompatibilityIssue",
        "ModelProfileImportResult",
        "import_legacy_model_profile",
    )
    assert model_profile_compat.LegacySourceInput is (
        _model_profile_compat_types.LegacySourceInput
    )
    assert list(signature.parameters) == ["acquisition", "conversion", "registration"]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
