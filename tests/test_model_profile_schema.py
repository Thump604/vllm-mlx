# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[1]
SCHEMA = json.loads((ROOT / "schemas/model-profile-v1.schema.json").read_text())
EXAMPLE = json.loads(
    (ROOT / "schemas/examples/model-profile-v1.example.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def _errors(profile):
    return sorted(VALIDATOR.iter_errors(profile), key=lambda error: list(error.path))


def test_model_profile_example_matches_v1_schema():
    Draft202012Validator.check_schema(SCHEMA)
    assert _errors(EXAMPLE) == []


def test_activation_control_requires_activation_field():
    profile = copy.deepcopy(EXAMPLE)
    feature = profile["serving"]["features"]["continuous_batching"]
    feature.update(
        mode="available_on_activation",
        control="activation",
        control_field="stream",
    )

    assert _errors(profile)


def test_request_control_requires_request_field():
    profile = copy.deepcopy(EXAMPLE)
    feature = profile["serving"]["features"]["continuous_batching"]
    feature.update(
        mode="available_per_request",
        control="request",
        control_field="features.continuous_batching",
    )

    assert _errors(profile)


def test_inactive_feature_cannot_expose_control():
    profile = copy.deepcopy(EXAMPLE)
    feature = profile["serving"]["features"]["continuous_batching"]
    feature.update(
        mode="deferred",
        control="request",
        control_field="stream",
    )

    assert _errors(profile)
