# SPDX-License-Identifier: Apache-2.0
"""Precedence-matrix tests for pure ModelProfile serving resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.model_profile import compute_subject_digest
from vllm_mlx.model_profile_resolution import (
    ModelProfileResolutionError,
    resolve_effective_serving_configuration,
)

ROOT = Path(__file__).parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


@pytest.mark.parametrize(
    ("provider", "profile_default", "fallback", "expected", "source"),
    [
        (0.9, 0.6, 0.1, 0.6, "profile_default"),
        (0.9, None, 0.1, 0.9, "provider_default"),
        (None, None, 0.1, 0.1, "runtime_fallback"),
    ],
)
def test_sampling_precedence_matrix(
    provider, profile_default, fallback, expected, source
):
    profile = _load("schemas/examples/model-profile-v1.example.json")
    provider_defaults = profile["serving"]["sampling"]["provider_defaults"]
    profile_defaults = profile["serving"]["sampling"]["profile_defaults"]
    provider_defaults.pop("temperature", None)
    profile_defaults.pop("temperature", None)
    if provider is not None:
        provider_defaults["temperature"] = provider
    if profile_default is not None:
        profile_defaults["temperature"] = profile_default
    profile["subject_digest"] = compute_subject_digest(profile)

    effective = resolve_effective_serving_configuration(
        profile,
        profile_schema=_load("schemas/model-profile-v1.schema.json"),
        runtime_fallbacks={"temperature": fallback},
    )

    assert effective.values["sampling"]["temperature"] == expected
    assert effective.sources["/sampling/temperature"] == source


def test_runtime_fallback_cannot_target_non_sampling_contract_fields():
    profile = _load("schemas/examples/model-profile-v1.example.json")

    with pytest.raises(ModelProfileResolutionError, match="not schema sampling"):
        resolve_effective_serving_configuration(
            profile,
            profile_schema=_load("schemas/model-profile-v1.schema.json"),
            runtime_fallbacks={"engine": "batched"},
        )


def test_every_effective_leaf_has_a_source():
    profile = _load("schemas/examples/model-profile-v1.example.json")
    profile["serving"]["template"]["default_kwargs"] = {"a/b~c": True}
    profile["subject_digest"] = compute_subject_digest(profile)
    effective = resolve_effective_serving_configuration(
        profile,
        profile_schema=_load("schemas/model-profile-v1.schema.json"),
    )

    def leaf_pointers(value, pointer=""):
        if isinstance(value, dict):
            if not value:
                return {pointer}
            return {
                child
                for key, item in value.items()
                for child in leaf_pointers(
                    item,
                    f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}",
                )
            }
        if isinstance(value, list):
            return {pointer}
        return {pointer}

    assert set(effective.sources) == leaf_pointers(effective.values)
    assert "/template/default_kwargs/a~1b~0c" in effective.sources


def test_guarded_feature_cannot_be_activated_even_if_policy_is_malformed():
    profile = _load("schemas/examples/model-profile-v1.example.json")
    profile["serving"]["activation_policy"]["owner_override_fields"].append(
        "features.prefix_cache"
    )
    profile["subject_digest"] = compute_subject_digest(profile)

    with pytest.raises(ModelProfileResolutionError, match="not available"):
        resolve_effective_serving_configuration(
            profile,
            profile_schema=_load("schemas/model-profile-v1.schema.json"),
            activation_overrides={"features.prefix_cache": True},
        )
