# SPDX-License-Identifier: Apache-2.0
"""Fixture-backed tests for pure model-family adapter extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.model_family_adapters import (
    CallerProfileInput,
    KvProfileInputs,
    adapt_model_config,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "model_configs"


def _payload(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _config(name: str) -> dict[str, object]:
    config = _payload(name)["config"]
    assert isinstance(config, dict)
    return config


def _profile(**overrides: object) -> KvProfileInputs:
    values = {
        "element_bytes": CallerProfileInput(2, "profile/cache-dtype"),
        "context_tokens": CallerProfileInput(32_768, "profile/context"),
        "concurrency": CallerProfileInput(3, "profile/concurrency"),
        "quantization_overhead_bytes": CallerProfileInput(96, "profile/kv-overhead"),
    }
    values.update(overrides)
    return KvProfileInputs(**values)


def _fact(result, name: str):
    return next(fact for fact in result.provider_facts if fact.name == name)


def _provenance(result, field: str):
    return next(item for item in result.input_provenance if item.field == field)


def test_fixture_metadata_is_not_part_of_provider_config_extraction():
    dense = _payload("dense-gqa.json")

    assert dense["fixture_metadata"] == {
        "fixture_kind": "synthetic",
        "purpose": "generic dense/GQA adapter contract; not a provider artifact",
    }
    assert "fixture_metadata" not in _config("dense-gqa.json")


def test_qwen_fixture_metadata_is_not_part_of_provider_config_extraction():
    qwen = _payload("qwen3-5-hybrid.json")

    assert qwen["fixture_metadata"] == {
        "source_repository": "mlx-community/Qwen3.6-35B-A3B-8bit",
        "source_revision": "e06a74e6236a60c8367e1a3214e83d8b61b637b0",
        "base_model_repository": "Qwen/Qwen3.6-35B-A3B",
        "source_config_sha256": "5da49c3a84c5a4a751b720b62b9f8703bdb8a0796b5fc55df3278f8b9d6ce296",
    }
    assert all(
        "fixture_metadata" not in fact.pointer
        for fact in adapt_model_config(_config("qwen3-5-hybrid.json")).provider_facts
    )


def test_laguna_fixture_metadata_is_not_part_of_provider_config_extraction():
    laguna = _payload("laguna-s-2-1.json")

    assert laguna["fixture_metadata"] == {
        "source_repository": "poolside/Laguna-S-2.1",
        "source_revision": "a50e85e7e0aae7b0a504d156bd36a616ec9fea38",
        "local_config_sha256": "8440d3ec23e275aa62bba1371c20cee4a72906fdc33ca37966ba7cd83472847b",
    }
    assert all(
        "fixture_metadata" not in fact.pointer
        for fact in adapt_model_config(_config("laguna-s-2-1.json")).provider_facts
    )


def test_dense_gqa_requires_explicit_profile_inputs_and_preserves_provenance():
    config = _config("dense-gqa.json")
    without_profile = adapt_model_config(config)
    result = adapt_model_config(config, kv_profile=_profile())

    assert without_profile.status == "ready"
    assert without_profile.dense_gqa_kv_input is None
    assert "caller/profile" in without_profile.predictive_kv_reason
    assert result.adapter_id == "dense_gqa"
    assert result.dense_gqa_kv_input is not None
    assert result.dense_gqa_kv_input.architecture_kind == "gqa"
    assert result.dense_gqa_kv_input.context_tokens == 32_768
    assert result.dense_gqa_kv_input.concurrency == 3
    assert _fact(result, "num_key_value_heads").pointer == "/num_key_value_heads"
    assert _provenance(result, "dense_gqa_kv_input.layer_count").references == (
        "config:/num_hidden_layers",
    )
    assert _provenance(result, "dense_gqa_kv_input.element_bytes").references == (
        "profile:element_bytes",
        "profile/cache-dtype",
    )
    assert (
        _provenance(result, "dense_gqa_kv_input.concurrency").classification
        == "caller_profile"
    )
    assert _provenance(result, "context_limits.policy_context_tokens").references == (
        "profile:context_tokens",
        "profile/context",
    )


def test_dense_adapter_distinguishes_mha_from_gqa_by_declared_head_counts():
    config = _config("dense-gqa.json")
    config["num_key_value_heads"] = 32

    result = adapt_model_config(config, kv_profile=_profile())

    assert result.status == "ready"
    assert result.dense_gqa_kv_input is not None
    assert result.dense_gqa_kv_input.architecture_kind == "dense"


def test_dense_adapter_rejects_invalid_head_relationship_and_synthetic_fields():
    config = _config("dense-gqa.json")
    config["num_attention_heads"] = 4
    config["num_key_value_heads"] = 8

    result = adapt_model_config(config, kv_profile=_profile())

    assert result.status == "unknown"
    assert any("must be >=" in reason for reason in result.reasons)
    assert "cache_layout" not in config
    assert "kv_cache_element_bytes" not in config


def test_qwen_nested_config_requires_exact_declared_full_attention_schedule():
    result = adapt_model_config(_config("qwen3-5-hybrid.json"))

    assert result.adapter_id == "qwen3_5_hybrid"
    assert result.status == "ready"
    assert result.context_limits is not None
    assert result.context_limits.advertised_context_tokens == 262_144
    assert result.dense_gqa_kv_input is None
    assert "linear-attention" in result.predictive_kv_reason
    assert (
        _fact(result, "linear_value_head_dim").pointer
        == "/text_config/linear_value_head_dim"
    )
    assert _fact(result, "num_experts_per_tok").value == 8
    assert _fact(result, "mtp_num_hidden_layers").value == 1


def test_qwen_hybrid_rejects_irrelevant_dense_kv_profile():
    result = adapt_model_config(_config("qwen3-5-hybrid.json"), kv_profile=_profile())

    assert result.status == "unknown"
    assert any("do not apply" in reason for reason in result.reasons)


def test_qwen_schedule_mismatch_fails_unknown():
    config = _config("qwen3-5-hybrid.json")
    text = config["text_config"]
    assert isinstance(text, dict)
    layer_types = text["layer_types"]
    assert isinstance(layer_types, list)
    layer_types[3] = "linear_attention"

    result = adapt_model_config(config)

    assert result.status == "unknown"
    assert any("full_attention_interval" in reason for reason in result.reasons)


def test_laguna_requires_complete_identity_structure_and_retains_unknown_kv():
    result = adapt_model_config(_config("laguna-s-2-1.json"))

    assert result.adapter_id == "laguna_s_2_1"
    assert result.status == "ready"
    assert result.context_limits is not None
    assert result.context_limits.advertised_context_tokens == 1_048_576
    assert result.context_limits.kv_window_tokens == 512
    assert result.dense_gqa_kv_input is None
    assert "mixed global/sliding" in result.predictive_kv_reason
    assert _fact(result, "architecture").pointer == "/architectures/0"
    assert _fact(result, "shared_expert_intermediate_size").value == 1024
    assert _fact(result, "mlp_only_layers").value == [0]
    assert _fact(result, "global_attention_layer_count").value == 12
    assert _fact(result, "sliding_attention_layer_count").value == 36


def test_laguna_rejects_irrelevant_dense_kv_profile():
    result = adapt_model_config(_config("laguna-s-2-1.json"), kv_profile=_profile())

    assert result.status == "unknown"
    assert any("do not apply" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architectures", ["OtherForCausalLM"]),
        ("layer_types", ["full_attention"] * 48),
        ("mlp_layer_types", ["sparse"] * 48),
        ("gating_types", ["per_head"] * 47),
        ("num_attention_heads_per_layer", [48] * 48),
        ("mlp_only_layers", []),
        ("num_key_value_heads", 4),
        ("head_dim", 256),
        ("max_position_embeddings", 262_144),
        ("sliding_window", 1024),
        ("num_experts_per_tok", 8),
        ("shared_expert_intermediate_size", 512),
    ],
)
def test_laguna_adversarial_identity_variants_fail_unknown(field: str, value: object):
    config = _config("laguna-s-2-1.json")
    config[field] = value

    result = adapt_model_config(config)

    assert result.status == "unknown"
    assert result.dense_gqa_kv_input is None


def test_unknown_model_type_does_not_use_path_or_name_inference():
    config = _config("dense-gqa.json")
    config["model_type"] = "unknown_dense_gqa"
    config["model_path"] = "models/Laguna-S-2.1-Qwen3.6"
    config["repository_id"] = "org/qwen3_5_moe"

    result = adapt_model_config(config, kv_profile=_profile())

    assert result.adapter_id == "unknown"
    assert result.status == "unknown"
    assert result.provider_facts == ()


@pytest.mark.parametrize(
    ("fixture_name", "path"),
    [
        ("dense-gqa.json", ("num_hidden_layers",)),
        ("dense-gqa.json", ("num_attention_heads",)),
        ("dense-gqa.json", ("num_key_value_heads",)),
        ("dense-gqa.json", ("head_dim",)),
        ("dense-gqa.json", ("max_position_embeddings",)),
        ("qwen3-5-hybrid.json", ("text_config", "linear_num_key_heads")),
        ("qwen3-5-hybrid.json", ("text_config", "num_experts_per_tok")),
        ("qwen3-5-hybrid.json", ("text_config", "mtp_num_hidden_layers")),
        ("laguna-s-2-1.json", ("num_experts",)),
        ("laguna-s-2-1.json", ("num_experts_per_tok",)),
        ("laguna-s-2-1.json", ("shared_expert_intermediate_size",)),
    ],
)
def test_boolean_values_never_count_as_config_numeric_facts(
    fixture_name: str, path: tuple[str, ...]
):
    config = _config(fixture_name)
    target: dict[str, object] = config
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = True

    result = adapt_model_config(config)

    assert result.status == "unknown"
    assert any("positive integer" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "profile",
    [
        _profile(element_bytes=CallerProfileInput(True, "profile/cache-dtype")),
        _profile(context_tokens=CallerProfileInput(True, "profile/context")),
        _profile(concurrency=CallerProfileInput(True, "profile/concurrency")),
        _profile(
            quantization_overhead_bytes=CallerProfileInput(True, "profile/kv-overhead")
        ),
        _profile(concurrency=CallerProfileInput(1, " ")),
    ],
)
def test_profile_inputs_require_valid_values_and_provenance(profile: KvProfileInputs):
    result = adapt_model_config(_config("dense-gqa.json"), kv_profile=profile)

    assert result.status == "unknown"
    assert result.dense_gqa_kv_input is None
