# SPDX-License-Identifier: Apache-2.0
"""Versioned HTTP control-client contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import re
from typing import Any, Mapping

import rfc8785

from . import __version__

CONTROL_API_VERSION = "1.0"
MINIMUM_CLIENT_VERSION = "1.0"
PROFILE_SCHEMA_VERSION = "1"
LIFECYCLE_SCHEMA_VERSION = "1"


class ControlApiCompatibilityError(ValueError):
    """Raised when a client version cannot safely consume this control API."""


@dataclass(frozen=True)
class ControlOperation:
    """One stable product operation exposed by a control transport."""

    operation_id: str
    operation_version: str
    method: str
    path: str
    mutating: bool
    idempotent: bool


CONTROL_OPERATIONS = (
    ControlOperation(
        "capabilities.get", "1.0", "GET", "/api/v1/control/capabilities", False, True
    ),
    ControlOperation(
        "catalog.list", "1.0", "GET", "/api/v1/control/catalog", False, True
    ),
    ControlOperation(
        "profile.get",
        "1.0",
        "GET",
        "/api/v1/control/catalog/{profile_id}",
        False,
        True,
    ),
    ControlOperation(
        "model.install",
        "1.0",
        "POST",
        "/api/v1/control/models/{profile_id}/install",
        True,
        True,
    ),
    ControlOperation(
        "model.activate", "1.0", "PUT", "/api/v1/control/active", True, True
    ),
    ControlOperation(
        "model.stop", "1.0", "POST", "/api/v1/control/active/stop", True, True
    ),
    ControlOperation(
        "model.remove",
        "1.0",
        "POST",
        "/api/v1/control/models/{profile_id}/remove",
        True,
        True,
    ),
    ControlOperation(
        "operation.get",
        "1.0",
        "GET",
        "/api/v1/control/operations/{operation_id}",
        False,
        True,
    ),
    ControlOperation(
        "operation.cancel",
        "1.0",
        "POST",
        "/api/v1/control/operations/{operation_id}/cancel",
        True,
        True,
    ),
    ControlOperation(
        "runtime.status", "1.0", "GET", "/api/v1/control/status", False, True
    ),
    ControlOperation(
        "runtime.diagnostics",
        "1.0",
        "GET",
        "/api/v1/control/diagnostics",
        False,
        True,
    ),
)

CONTROL_ERROR_CODES = (
    "invalid_request",
    "profile_not_found",
    "operation_not_found",
    "profile_revision_stale",
    "profile_subject_mismatch",
    "lifecycle_conflict",
    "runtime_unavailable",
    "operation_not_cancellable",
    "idempotency_conflict",
)


@dataclass(frozen=True)
class ProfileReference:
    """Exact immutable profile identity required by mutating operations."""

    profile_id: str
    profile_revision: int
    subject_digest: str


def parse_profile_reference(value: Mapping[str, Any]) -> ProfileReference:
    """Validate an exact profile reference and reject hidden request fields."""
    if not isinstance(value, Mapping):
        raise ValueError("profile reference must be an object")
    required = {"profile_id", "profile_revision", "subject_digest"}
    unknown = set(value) - required
    missing = required - set(value)
    if unknown:
        raise ValueError(f"unknown profile reference fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing profile reference fields: {sorted(missing)}")
    profile_id = value["profile_id"]
    revision = value["profile_revision"]
    digest = value["subject_digest"]
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile_id must be a non-empty string")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("profile_revision must be a positive integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ):
        raise ValueError("subject_digest must be a SHA-256 hex digest")
    return ProfileReference(profile_id, revision, digest.lower())


def _parse_idempotency_key(value: object) -> str:
    if not isinstance(value, str) or not 8 <= len(value) <= 128:
        raise ValueError("idempotency_key must contain 8 to 128 characters")
    return value


def parse_profile_mutation_request(
    value: Mapping[str, Any], *, route_profile_id: str
) -> dict[str, Any]:
    """Validate an install/removal request bound to one exact profile subject."""
    if not isinstance(value, Mapping):
        raise ValueError("profile mutation request must be an object")
    allowed = {"profile", "idempotency_key"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown profile mutation request fields: {sorted(unknown)}")
    missing = allowed - set(value)
    if missing:
        raise ValueError(f"missing profile mutation request fields: {sorted(missing)}")
    profile = parse_profile_reference(value["profile"])
    if profile.profile_id != route_profile_id:
        raise ValueError("route profile_id does not match request profile_id")
    return {
        "profile": asdict(profile),
        "idempotency_key": _parse_idempotency_key(value["idempotency_key"]),
    }


def parse_idempotent_request(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate a body used by stop and cancellation operations."""
    if not isinstance(value, Mapping):
        raise ValueError("idempotent request must be an object")
    if set(value) != {"idempotency_key"}:
        raise ValueError("idempotent request requires only idempotency_key")
    return {"idempotency_key": _parse_idempotency_key(value["idempotency_key"])}


def canonical_idempotency_digest(
    operation_id: str,
    request: Mapping[str, Any],
    *,
    route_parameters: Mapping[str, str] | None = None,
) -> str:
    """Bind an idempotency record to one operation and canonical request body."""
    operation = next(
        (item for item in CONTROL_OPERATIONS if item.operation_id == operation_id), None
    )
    if operation is None:
        raise ValueError(f"unknown control operation: {operation_id}")
    required_parameters = set(re.findall(r"\{([^}]+)\}", operation.path))
    parameters = dict(route_parameters or {})
    if set(parameters) != required_parameters:
        raise ValueError(
            f"route parameters must match {sorted(required_parameters)} for {operation_id}"
        )
    body = dict(request)
    body.pop("idempotency_key", None)
    encoded = rfc8785.dumps(
        {
            "operation_id": operation_id,
            "route_parameters": parameters,
            "request": body,
        }
    )
    return sha256(encoded).hexdigest()


def parse_activation_request(
    value: Mapping[str, Any], *, allowed_override_fields: set[str] | None = None
) -> dict[str, Any]:
    """Validate an exact, idempotent v1 activation request."""
    if not isinstance(value, Mapping):
        raise ValueError("activation request must be an object")
    allowed = {"profile", "overrides", "idempotency_key"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown activation request fields: {sorted(unknown)}")
    required = {"profile", "idempotency_key"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"activation request missing fields: {sorted(missing)}")
    overrides = value.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("activation overrides must be an object")
    invalid_shapes = []
    for name, override in overrides.items():
        if not isinstance(name, str):
            invalid_shapes.append(name)
        elif name.startswith("limits."):
            if (
                not isinstance(override, int)
                or isinstance(override, bool)
                or override < 1
            ):
                invalid_shapes.append(name)
        elif name.startswith("features."):
            if not isinstance(override, bool):
                invalid_shapes.append(name)
        else:
            invalid_shapes.append(name)
    invalid_shapes = sorted(str(name) for name in invalid_shapes)
    if invalid_shapes:
        raise ValueError(f"invalid activation overrides: {invalid_shapes}")
    if allowed_override_fields is not None:
        disallowed = sorted(set(overrides) - allowed_override_fields)
        if disallowed:
            raise ValueError(f"activation overrides are not allowed: {disallowed}")
    return {
        "profile": asdict(parse_profile_reference(value["profile"])),
        "overrides": dict(overrides),
        "idempotency_key": _parse_idempotency_key(value["idempotency_key"]),
    }


def parse_api_version(value: str) -> tuple[int, int]:
    """Parse the deliberately small ``MAJOR.MINOR`` control API version."""
    if not isinstance(value, str):
        raise ControlApiCompatibilityError("control API version must be a string")
    parts = value.split(".")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise ControlApiCompatibilityError(
            "control API version must use MAJOR.MINOR numeric form"
        )
    major, minor = (int(part) for part in parts)
    if str(major) != parts[0] or str(minor) != parts[1]:
        raise ControlApiCompatibilityError(
            "control API version must use canonical MAJOR.MINOR form"
        )
    return major, minor


def require_compatible_client(client_version: str) -> None:
    """Reject clients outside the server's explicitly supported API interval."""
    client = parse_api_version(client_version)
    minimum = parse_api_version(MINIMUM_CLIENT_VERSION)
    current = parse_api_version(CONTROL_API_VERSION)
    if client[0] != current[0]:
        raise ControlApiCompatibilityError(
            f"client major version {client[0]} is incompatible with server major "
            f"version {current[0]}"
        )
    if client < minimum:
        raise ControlApiCompatibilityError(
            f"client version {client_version} is older than the minimum supported "
            f"version {MINIMUM_CLIENT_VERSION}"
        )
    if client > current:
        raise ControlApiCompatibilityError(
            f"client version {client_version} is newer than server version "
            f"{CONTROL_API_VERSION}"
        )


def build_control_api_descriptor() -> dict[str, Any]:
    """Return the canonical capability document for control clients."""
    return {
        "kind": "vllm-mlx-control-api",
        "api_version": CONTROL_API_VERSION,
        "minimum_client_version": MINIMUM_CLIENT_VERSION,
        "runtime_version": __version__,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "error_codes": list(CONTROL_ERROR_CODES),
        "operations": [asdict(operation) for operation in CONTROL_OPERATIONS],
    }
