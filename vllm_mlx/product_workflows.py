# SPDX-License-Identifier: Apache-2.0
"""Golden first-product workflows over catalog and control-client contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping, Protocol

from .catalog import ModelProfileCatalog


class ProductWorkflowError(RuntimeError):
    """Raised when a product workflow cannot prove its expected state."""

    def __init__(self, message: str, *, recovery: Mapping[str, Any] | None = None):
        self.recovery = dict(recovery or {})
        super().__init__(message)


class ProductControlClient(Protocol):
    def install(
        self, profile: Mapping[str, Any], idempotency_key: str
    ) -> dict[str, Any]: ...

    def activate(
        self,
        profile: Mapping[str, Any],
        idempotency_key: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def operation(self, operation_id: str) -> dict[str, Any]: ...

    def cancel_operation(
        self, operation_id: str, idempotency_key: str
    ) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...

    def chat(
        self,
        *,
        model: str,
        message: str,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


def install_to_chat(
    client: ProductControlClient,
    catalog: ModelProfileCatalog,
    *,
    profile_id: str,
    profile_revision: int,
    install_idempotency_key: str,
    activate_idempotency_key: str,
    message: str,
    activation_overrides: Mapping[str, Any] | None = None,
    max_operation_polls: int = 100,
) -> dict[str, Any]:
    """Install, activate, verify, and chat with one exact catalog profile."""
    common = _install_and_activate(
        client,
        catalog,
        profile_id=profile_id,
        profile_revision=profile_revision,
        install_idempotency_key=install_idempotency_key,
        activate_idempotency_key=activate_idempotency_key,
        activation_overrides=activation_overrides,
        max_operation_polls=max_operation_polls,
    )
    profile = common["profile"]
    chat = client.chat(
        model=profile["identity"]["served_model_name"],
        message=message,
        stream=False,
    )
    return {**common, "workflow": "install_to_chat", "chat": chat}


def install_to_code(
    client: ProductControlClient,
    catalog: ModelProfileCatalog,
    *,
    profile_id: str,
    profile_revision: int,
    install_idempotency_key: str,
    activate_idempotency_key: str,
    coding_client: str,
    activation_overrides: Mapping[str, Any] | None = None,
    runtime_api_key_configured: bool = False,
    max_operation_polls: int = 100,
) -> dict[str, Any]:
    """Install, activate, verify, and emit deterministic coding configuration."""
    from .product_cli import build_coding_setup

    common = _install_and_activate(
        client,
        catalog,
        profile_id=profile_id,
        profile_revision=profile_revision,
        install_idempotency_key=install_idempotency_key,
        activate_idempotency_key=activate_idempotency_key,
        activation_overrides=activation_overrides,
        max_operation_polls=max_operation_polls,
    )
    endpoint = common["runtime_status"].get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise ProductWorkflowError("runtime status has no active inference endpoint")
    configuration = build_coding_setup(
        coding_client,
        common["profile"]["identity"]["served_model_name"],
        endpoint,
        runtime_api_key_configured=runtime_api_key_configured,
    )
    return {
        **common,
        "workflow": "install_to_code",
        "coding_configuration": configuration,
    }


def _install_and_activate(
    client: ProductControlClient,
    catalog: ModelProfileCatalog,
    *,
    profile_id: str,
    profile_revision: int,
    install_idempotency_key: str,
    activate_idempotency_key: str,
    activation_overrides: Mapping[str, Any] | None,
    max_operation_polls: int,
) -> dict[str, Any]:
    profile = catalog.get(profile_id, profile_revision)
    reference = {
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "subject_digest": profile["subject_digest"],
    }
    install = _await_operation(
        client,
        client.install(reference, install_idempotency_key),
        expected_profile=reference,
        cancellation_key=_cancellation_key(install_idempotency_key),
        max_polls=max_operation_polls,
    )
    activate = _await_operation(
        client,
        client.activate(
            reference,
            activate_idempotency_key,
            overrides=activation_overrides,
        ),
        expected_profile=reference,
        cancellation_key=_cancellation_key(activate_idempotency_key),
        max_polls=max_operation_polls,
    )
    status = client.status()
    if status.get("active_profile") != reference:
        raise ProductWorkflowError(
            "runtime active profile does not match the requested profile subject"
        )
    if status.get("state") != "loaded" or status.get("healthy") is not True:
        raise ProductWorkflowError("runtime is not loaded and healthy")
    if not isinstance(status.get("endpoint"), str) or not status["endpoint"]:
        raise ProductWorkflowError("runtime status has no active inference endpoint")
    return {
        "profile": deepcopy(profile),
        "profile_reference": reference,
        "install_operation": install,
        "activate_operation": activate,
        "runtime_status": status,
    }


def _await_operation(
    client: ProductControlClient,
    record: Mapping[str, Any],
    *,
    expected_profile: Mapping[str, Any],
    cancellation_key: str,
    max_polls: int,
) -> dict[str, Any]:
    if max_polls < 1:
        raise ValueError("max_operation_polls must be positive")
    current = dict(record)
    operation_id = current.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise ProductWorkflowError("control operation record has no operation_id")
    for _ in range(max_polls):
        _validate_operation_identity(current, operation_id, expected_profile)
        status = current.get("status")
        if status == "succeeded":
            return current
        if status in {"failed", "cancelled"}:
            raise ProductWorkflowError(
                f"control operation {operation_id} ended with status {status}"
            )
        if status not in {"queued", "running"}:
            raise ProductWorkflowError(
                f"control operation {operation_id} has invalid status {status!r}"
            )
        current = dict(client.operation(operation_id))
    _validate_operation_identity(current, operation_id, expected_profile)
    recovery: dict[str, Any] = {
        "operation_id": operation_id,
        "last_record": current,
        "cancellation_idempotency_key": cancellation_key,
    }
    try:
        cancellation = client.cancel_operation(operation_id, cancellation_key)
        _validate_operation_identity(cancellation, operation_id, expected_profile)
        if cancellation.get("status") not in {"succeeded", "failed", "cancelled"}:
            raise ProductWorkflowError(
                f"cancellation for {operation_id} did not return a terminal record"
            )
        recovery["cancellation"] = cancellation
    except Exception as exc:
        recovery["cancellation_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    raise ProductWorkflowError(
        f"control operation {operation_id} did not finish within {max_polls} polls",
        recovery=recovery,
    )


def _cancellation_key(operation_key: str) -> str:
    digest = hashlib.sha256(operation_key.encode()).hexdigest()
    return f"cancel-{digest}"


def _validate_operation_identity(
    record: Mapping[str, Any],
    operation_id: str,
    expected_profile: Mapping[str, Any],
) -> None:
    if record.get("operation_id") != operation_id:
        raise ProductWorkflowError(
            f"control operation response substituted operation {operation_id}"
        )
    if record.get("profile") != dict(expected_profile):
        raise ProductWorkflowError(
            f"control operation {operation_id} is not bound to the requested profile"
        )
