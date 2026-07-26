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
