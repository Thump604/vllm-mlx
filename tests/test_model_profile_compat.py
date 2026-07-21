# SPDX-License-Identifier: Apache-2.0
"""Focused compatibility tests using representative legacy workflow records."""

from __future__ import annotations

from copy import deepcopy

from vllm_mlx.model_profile_compat import import_legacy_model_profile


def _source(location: str, payload: dict) -> dict:
    return {"location": location, "sha256": "a" * 64, "payload": payload}


def test_import_maps_legacy_records_without_mutating_inputs():
    acquisition = _source(
        "/manifests/acquisition.json",
        {
            "kind": "vllm-mlx-model-artifact",
            "model_id": "mlx-community/Qwen3-4B-4bit",
            "revision": "main",
            "inspection": {
                "revision": "b" * 40,
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
            "kind": "vllm-mlx-conversion",
            "backend": "mlx-lm",
            "status": "succeeded",
            "recipe": {"q_bits": 4, "q_group_size": 64, "q_mode": "affine"},
            "output_inspection": acquisition["payload"]["inspection"],
        },
    )
    registration = _source(
        "/manifests/registration.json",
        {
            "kind": "vllm-mlx-model-registration",
            "model_id": "qwen3-4b-4bit",
            "served_model_name": "qwen3",
            "preset_alias": "fast-qwen",
            "serving_defaults": {
                "temperature": 0.6,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "parser_policy": {
                "tool_call_parser": "qwen3_coder",
                "reasoning_parser": "qwen3",
            },
            "production_ready": False,
        },
    )
    original = deepcopy((acquisition, conversion, registration))

    result = import_legacy_model_profile(
        acquisition=acquisition, conversion=conversion, registration=registration
    )

    assert (acquisition, conversion, registration) == original
    assert result.complete is False
    assert result.profile == {
        "schema_version": 1,
        "identity": {
            "provider": "huggingface",
            "repository_id": "mlx-community/Qwen3-4B-4bit",
            "requested_revision": "main",
            "served_model_name": "qwen3",
            "aliases": ["fast-qwen"],
        },
        "artifact": {
            "source_uri": "mlx-community/Qwen3-4B-4bit",
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "dtype": "bfloat16",
            "size_bytes": 1234,
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "source": "conversion_recipe",
            },
            "format": "mlx",
        },
        "serving": {
            "sampling": {"profile_defaults": {"temperature": 0.6}},
            "template": {"default_kwargs": {"enable_thinking": True}},
            "parsers": {"tool": "qwen3_coder", "reasoning": "qwen3"},
        },
        "provenance": result.profile["provenance"],
    }
    assert any(
        issue.pointer == "/artifact/hashes/config_sha256" for issue in result.issues
    )
    assert any(
        issue.pointer == "/identity/resolved_revision" for issue in result.issues
    )
    assert any(
        issue.code == "ambiguous_registration_model_id" for issue in result.issues
    )
    assert any(issue.code == "qualification_boolean_ignored" for issue in result.issues)
    assert result.as_dict()["sources"][0]["kind"] == "acquisition"


def test_import_reports_conflicts_and_never_promotes_qualification_booleans():
    acquisition = _source(
        "/manifests/acquisition.json",
        {
            "model_id": "org/model",
            "inspection": {
                "model_family": {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                }
            },
        },
    )
    registration = _source(
        "/manifests/registration.json",
        {
            "model_id": "artifact",
            "served_model_name": "served",
            "production_ready": True,
        },
    )
    qualification = _source(
        "/manifests/qualification.json",
        {
            "model_id": "served",
            "status": "succeeded",
            "returncode": 0,
            "production_ready": True,
        },
    )

    result = import_legacy_model_profile(
        acquisition=acquisition, registration=registration, qualification=qualification
    )

    assert "qualification" not in result.profile
    assert all(
        not (
            issue.pointer == "/qualification/status"
            and issue.code != "missing_required_fact"
        )
        for issue in result.issues
    )
    evidence_issue = next(
        issue
        for issue in result.issues
        if issue.code == "qualification_evidence_missing"
    )
    assert evidence_issue.sources == ("/manifests/qualification.json",)


def test_cli_and_registry_mappings_keep_registry_identity_separate():
    registry = _source(
        "/config/models.yaml",
        {"name": "registry-key", "source": "/models/qwen", "continuous_batching": True},
    )
    cli_server = _source(
        "/commands/serve.json",
        {
            "serving": {
                "engine": "simple",
                "route": "openai_chat",
                "template": {
                    "source": "tokenizer",
                    "sha256": "c" * 64,
                    "default_kwargs": {},
                },
                "limits": {
                    "max_output_tokens": 1024,
                    "max_request_output_tokens": 2048,
                    "unknown_limit": 7,
                },
                "features": {"not_a_v1_feature": {"mode": "enabled_by_default"}},
                "activation_policy": {
                    "owner_override_fields": [
                        "limits.serving_context",
                        "identity.provider",
                    ]
                },
                "request_policy": {
                    "required_fields": {"unsupported": True},
                    "allowed_fields": ["temperature", "unsupported"],
                    "forbidden_fields": [],
                },
                "outside": "ignored",
            },
            "continuous_batching": True,
            "enable_mtp": True,
            "specprefill_enabled": True,
            "specprefill_threshold": 16,
            "specprefill_draft_model": "org/draft",
        },
    )

    result = import_legacy_model_profile(registry_entry=registry, cli_server=cli_server)

    assert "identity" not in result.profile
    assert result.profile["serving"]["engine"] == "batched"
    assert result.profile["serving"]["route"] == "openai_chat"
    assert result.profile["serving"]["template"]["sha256"] == "c" * 64
    assert "outside" not in result.profile["serving"]
    assert result.profile["serving"]["features"]["continuous_batching"] == {
        "mode": "enabled_by_default",
        "control": "none",
        "reason": (
            "enabled in the imported legacy configuration; override support is "
            "not established"
        ),
    }
    assert result.profile["serving"]["features"]["mtp"]["control"] == "none"
    assert result.profile["serving"]["features"]["specprefill"]["settings"] == {
        "draft_model": "org/draft"
    }
    conflict = next(
        issue for issue in result.issues if issue.code == "conflicting_value"
    )
    assert conflict.pointer == "/serving/engine"
    assert conflict.sources == ("/config/models.yaml", "/commands/serve.json")
    assert any(issue.code == "unknown_feature_declaration" for issue in result.issues)
    assert any(issue.code == "unsupported_feature_setting" for issue in result.issues)
    assert any(
        issue.code == "registry_field_requires_target_contract"
        for issue in result.issues
    )
    unknown_pointers = {
        issue.pointer
        for issue in result.issues
        if issue.code == "unknown_serving_field"
    }
    assert "/serving/outside" in unknown_pointers
    assert "/serving/limits/unknown_limit" in unknown_pointers
    assert any(issue.code == "invalid_policy_field" for issue in result.issues)
    assert result.profile["serving"]["activation_policy"] == {
        "owner_override_fields": ["limits.serving_context"]
    }
    assert result.profile["serving"]["request_policy"] == {
        "required_fields": {},
        "allowed_fields": ["temperature"],
        "forbidden_fields": [],
    }


def test_registration_feature_flags_are_reported_and_all_feature_states_are_required():
    result = import_legacy_model_profile(
        registration=_source(
            "/manifests/registration.json",
            {
                "model_id": "artifact",
                "served_model_name": "served",
                "feature_flags": ["prefix_cache"],
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


def test_failed_conversion_is_quarantined_without_artifact_claims():
    result = import_legacy_model_profile(
        conversion=_source(
            "/manifests/conversion.json",
            {
                "status": "failed",
                "backend": "mlx-lm",
                "recipe": {"q_bits": 4},
                "returncode": 1,
            },
        )
    )

    assert "artifact" not in result.profile
    issue = next(
        issue for issue in result.issues if issue.code == "conversion_output_unverified"
    )
    assert issue.pointer == "/artifact"


def test_invalid_boolean_and_same_source_conflict_do_not_silently_change_engine():
    result = import_legacy_model_profile(
        registry_entry=_source(
            "/config/models.yaml",
            {
                "continuous_batching": "false",
                "engine": "simple",
                "serving": {"engine": "batched"},
                "preload": True,
                "estimated_memory_gb": 8,
            },
        )
    )

    assert result.profile["serving"]["engine"] == "batched"
    assert "features" not in result.profile["serving"]
    codes = {issue.code for issue in result.issues}
    assert "invalid_feature_declaration" in codes
    assert "same_source_conflict" in codes
    assert "registry_field_requires_target_contract" in codes


def test_malformed_policy_shapes_are_reported_without_membership_errors():
    malformed_members = import_legacy_model_profile(
        cli_server=_source(
            "/commands/serve.json",
            {
                "activation_policy": {
                    "owner_override_fields": [[], "limits.max_kv_size"]
                },
                "request_policy": {
                    "required_fields": [],
                    "allowed_fields": [{}, "temperature"],
                    "forbidden_fields": [],
                },
            },
        )
    )
    malformed_container = import_legacy_model_profile(
        cli_server=_source(
            "/commands/serve-invalid.json",
            {"activation_policy": "invalid"},
        )
    )

    assert any(
        issue.code == "invalid_policy_field" for issue in malformed_members.issues
    )
    assert any(
        issue.code == "invalid_policy_shape" for issue in malformed_container.issues
    )
    serving = malformed_members.profile["serving"]
    assert serving["activation_policy"]["owner_override_fields"] == [
        "limits.max_kv_size"
    ]
    assert serving["request_policy"] == {
        "required_fields": {},
        "allowed_fields": ["temperature"],
        "forbidden_fields": [],
    }
