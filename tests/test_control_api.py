# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from vllm_mlx.control_api import (
    CONTROL_API_VERSION,
    CONTROL_ERROR_CODES,
    CONTROL_OPERATIONS,
    ControlApiCompatibilityError,
    build_control_api_descriptor,
    canonical_idempotency_digest,
    parse_activation_request,
    parse_api_version,
    parse_idempotent_request,
    parse_profile_mutation_request,
    parse_profile_reference,
    require_compatible_client,
)


def test_control_api_descriptor_matches_versioned_schema():
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas/control-api-v1.schema.json").read_text()
    )
    validator = validator_for(schema)
    validator.check_schema(schema)
    validator(schema).validate(build_control_api_descriptor())


def test_control_operations_are_unique_and_cover_first_product_workflow():
    operation_ids = [operation.operation_id for operation in CONTROL_OPERATIONS]
    assert len(operation_ids) == len(set(operation_ids))
    assert set(operation_ids) == {
        "catalog.list",
        "capabilities.get",
        "profile.get",
        "model.install",
        "model.activate",
        "model.stop",
        "model.remove",
        "operation.get",
        "operation.cancel",
        "runtime.status",
        "runtime.diagnostics",
    }
    assert all(operation.idempotent for operation in CONTROL_OPERATIONS)
    assert len(CONTROL_ERROR_CODES) == len(set(CONTROL_ERROR_CODES))


def test_control_descriptor_returns_independent_operation_records():
    first = build_control_api_descriptor()
    first["operations"][0]["operation_id"] = "changed"
    second = build_control_api_descriptor()
    assert second["operations"][0]["operation_id"] == "capabilities.get"


@pytest.mark.parametrize("version", ["1.0"])
def test_supported_control_client_versions(version):
    require_compatible_client(version)


@pytest.mark.parametrize(
    ("version", "message"),
    [
        ("0.9", "major version"),
        ("2.0", "major version"),
        ("1.1", "newer than server"),
        ("1", "MAJOR.MINOR"),
        ("01.0", "canonical"),
        ("1.x", "MAJOR.MINOR"),
    ],
)
def test_incompatible_or_malformed_control_client_versions(version, message):
    with pytest.raises(ControlApiCompatibilityError, match=message):
        require_compatible_client(version)


def test_parse_control_api_version_matches_published_version():
    assert parse_api_version(CONTROL_API_VERSION) == (1, 0)


def test_profile_reference_and_activation_request_are_strict():
    reference = {
        "profile_id": "laguna-s-2.1",
        "profile_revision": 3,
        "subject_digest": "A" * 64,
    }
    assert parse_profile_reference(reference).subject_digest == "a" * 64
    assert parse_activation_request(
        {
            "profile": reference,
            "idempotency_key": "activate-laguna-3",
            "overrides": {"limits.serving_context": 32768},
        },
        allowed_override_fields={"limits.serving_context"},
    ) == {
        "profile": {**reference, "subject_digest": "a" * 64},
        "overrides": {"limits.serving_context": 32768},
        "idempotency_key": "activate-laguna-3",
    }

    with pytest.raises(ValueError, match="unknown profile reference"):
        parse_profile_reference({**reference, "path": "/private/model"})
    with pytest.raises(ValueError, match="unknown activation request"):
        parse_activation_request(
            {
                "profile": reference,
                "idempotency_key": "activate-laguna-3",
                "model": "other",
            }
        )
    with pytest.raises(ValueError, match="positive integer"):
        parse_profile_reference({**reference, "profile_revision": True})


def test_profile_mutations_require_exact_identity_and_idempotency():
    reference = {
        "profile_id": "laguna-s-2.1",
        "profile_revision": 3,
        "subject_digest": "a" * 64,
    }
    assert (
        parse_profile_mutation_request(
            {"profile": reference, "idempotency_key": "install-laguna-3"},
            route_profile_id="laguna-s-2.1",
        )["profile"]
        == reference
    )
    with pytest.raises(ValueError, match="missing profile mutation"):
        parse_profile_mutation_request(
            {"profile": reference}, route_profile_id="laguna-s-2.1"
        )
    with pytest.raises(ValueError, match="8 to 128"):
        parse_profile_mutation_request(
            {"profile": reference, "idempotency_key": "short"},
            route_profile_id="laguna-s-2.1",
        )
    with pytest.raises(ValueError, match="route profile_id"):
        parse_profile_mutation_request(
            {"profile": reference, "idempotency_key": "install-laguna-3"},
            route_profile_id="other-model",
        )


def test_activation_overrides_are_bounded_by_profile_policy():
    request = {
        "profile": {
            "profile_id": "laguna-s-2.1",
            "profile_revision": 3,
            "subject_digest": "a" * 64,
        },
        "idempotency_key": "activate-laguna-3",
        "overrides": {"features.mtp": True},
    }
    with pytest.raises(ValueError, match="not allowed"):
        parse_activation_request(request, allowed_override_fields=set())
    with pytest.raises(ValueError, match="invalid activation"):
        parse_activation_request(
            {**request, "overrides": {"sampling.temperature": 0.7}}
        )
    with pytest.raises(ValueError, match="invalid activation"):
        parse_activation_request(
            {**request, "overrides": {"limits.serving_context": True}}
        )
    with pytest.raises(ValueError, match="invalid activation"):
        parse_activation_request({**request, "overrides": {"features.mtp": 1}})


def test_idempotency_is_scoped_to_operation_and_canonical_request():
    first = {
        "idempotency_key": "activate-laguna-3",
        "profile": {"profile_id": "laguna", "profile_revision": 3},
    }
    reordered = {
        "profile": {"profile_revision": 3, "profile_id": "laguna"},
        "idempotency_key": "a-different-retry-key",
    }
    assert canonical_idempotency_digest("model.activate", first) == (
        canonical_idempotency_digest("model.activate", reordered)
    )
    assert canonical_idempotency_digest("model.activate", first) != (
        canonical_idempotency_digest(
            "model.install", first, route_parameters={"profile_id": "laguna"}
        )
    )
    cancel = {"idempotency_key": "cancel-operation"}
    assert canonical_idempotency_digest(
        "operation.cancel", cancel, route_parameters={"operation_id": "op-1"}
    ) != canonical_idempotency_digest(
        "operation.cancel", cancel, route_parameters={"operation_id": "op-2"}
    )
    with pytest.raises(ValueError, match="route parameters"):
        canonical_idempotency_digest("operation.cancel", cancel)


def test_stop_and_cancel_request_has_only_idempotency_key():
    assert parse_idempotent_request({"idempotency_key": "stop-current-model"}) == {
        "idempotency_key": "stop-current-model"
    }
    with pytest.raises(ValueError, match="requires only"):
        parse_idempotent_request(
            {"idempotency_key": "stop-current-model", "force": True}
        )
