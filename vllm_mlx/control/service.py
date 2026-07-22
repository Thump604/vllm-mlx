# SPDX-License-Identifier: Apache-2.0
"""Application service for the versioned product control API."""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any, Awaitable, Callable, Mapping, Protocol, cast

from vllm_mlx.catalog import ModelProfileCatalog
from vllm_mlx.control_api import canonical_idempotency_digest
from vllm_mlx.lifecycle_contract import LifecycleContractError

logger = logging.getLogger(__name__)


class ProductControlError(RuntimeError):
    """Stable service error translated by the HTTP control adapter."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ProductOperationState(Protocol):
    """Existing lifecycle state surface used for durable product operations."""

    def create_control_operation(
        self,
        *,
        operation_kind: str,
        idempotency_key: str,
        request_digest: str,
        profile: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]: ...

    def get_control_operation(self, operation_id: str) -> dict[str, Any]: ...

    def update_control_operation(
        self,
        operation_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def bind_control_idempotency(
        self,
        *,
        operation_kind: str,
        idempotency_key: str,
        request_digest: str,
        operation_id: str,
    ) -> bool: ...


class ProductRuntimeAdapter(Protocol):
    """Only server-owned boundary allowed to mutate artifacts or residency."""

    async def install(self, profile: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def activate(
        self, profile: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    async def stop(self) -> Mapping[str, Any]: ...

    async def remove(self, profile: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def status(self) -> Mapping[str, Any]: ...

    def diagnostics(self) -> Mapping[str, Any]: ...

    def operation_is_cancellable(self, operation_kind: str) -> bool: ...


class ProductControlService:
    """Coordinate catalog identity, durable operations, and one runtime adapter."""

    def __init__(
        self,
        catalog: ModelProfileCatalog,
        operation_state: ProductOperationState,
        runtime: ProductRuntimeAdapter,
    ) -> None:
        self.catalog = catalog
        self.operation_state = operation_state
        self.runtime = runtime
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def list_catalog(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.catalog.list_profiles())

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], self.catalog.get(profile_id))
        except KeyError as exc:
            raise ProductControlError(
                "profile_not_found", f"catalog profile not found: {profile_id}"
            ) from exc

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        try:
            return self.operation_state.get_control_operation(operation_id)
        except KeyError as exc:
            raise ProductControlError(
                "operation_not_found", f"operation not found: {operation_id}"
            ) from exc

    def status(self) -> dict[str, Any]:
        return deepcopy(dict(self.runtime.status()))

    def diagnostics(self) -> dict[str, Any]:
        return deepcopy(dict(self.runtime.diagnostics()))

    def install(
        self,
        request: Mapping[str, Any],
        *,
        route_profile_id: str,
    ) -> dict[str, Any]:
        profile = self._exact_profile(request["profile"])
        digest = canonical_idempotency_digest(
            "model.install",
            request,
            route_parameters={"profile_id": route_profile_id},
        )
        return self._submit(
            "model.install",
            str(request["idempotency_key"]),
            digest,
            profile,
            lambda: self.runtime.install(profile),
        )

    def activate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        profile = self._exact_profile(request["profile"])
        if profile["qualification"]["status"] != "qualified":
            raise ProductControlError(
                "profile_not_qualified",
                f"profile is not qualified for product activation: {profile['profile_id']}",
            )
        overrides = dict(request.get("overrides", {}))
        allowed = set(profile["serving"]["activation_policy"]["owner_override_fields"])
        disallowed = sorted(set(overrides) - allowed)
        if disallowed:
            raise ProductControlError(
                "invalid_request",
                f"activation overrides are not allowed: {disallowed}",
            )
        digest = canonical_idempotency_digest("model.activate", request)
        return self._submit(
            "model.activate",
            str(request["idempotency_key"]),
            digest,
            profile,
            lambda: self.runtime.activate(profile, overrides),
        )

    def stop(self, request: Mapping[str, Any]) -> dict[str, Any]:
        digest = canonical_idempotency_digest("model.stop", request)
        return self._submit(
            "model.stop",
            str(request["idempotency_key"]),
            digest,
            None,
            self.runtime.stop,
        )

    def remove(
        self,
        request: Mapping[str, Any],
        *,
        route_profile_id: str,
    ) -> dict[str, Any]:
        profile = self._exact_profile(request["profile"])
        digest = canonical_idempotency_digest(
            "model.remove",
            request,
            route_parameters={"profile_id": route_profile_id},
        )
        return self._submit(
            "model.remove",
            str(request["idempotency_key"]),
            digest,
            profile,
            lambda: self.runtime.remove(profile),
        )

    async def cancel(
        self, operation_id: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        record = self.get_operation(operation_id)
        digest = canonical_idempotency_digest(
            "operation.cancel",
            request,
            route_parameters={"operation_id": operation_id},
        )
        try:
            replayed = self.operation_state.bind_control_idempotency(
                operation_kind="operation.cancel",
                idempotency_key=str(request["idempotency_key"]),
                request_digest=digest,
                operation_id=operation_id,
            )
        except LifecycleContractError as exc:
            if "different request" in str(exc):
                raise ProductControlError("idempotency_conflict", str(exc)) from exc
            raise ProductControlError("lifecycle_conflict", str(exc)) from exc
        if replayed:
            return self.get_operation(operation_id)
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        operation_kind = str(record["operation_kind"])
        if not self.runtime.operation_is_cancellable(operation_kind):
            raise ProductControlError(
                "operation_not_cancellable",
                f"operation {operation_id} is in a non-cancellable phase",
            )
        task = self._tasks.get(operation_id)
        if task is None or task.done():
            raise ProductControlError(
                "operation_not_cancellable",
                f"operation {operation_id} has no cancellable in-process task",
            )
        self.operation_state.update_control_operation(
            operation_id,
            status="cancelled",
            error={"code": "operation_cancelled", "message": "operation cancelled"},
        )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return self.get_operation(operation_id)

    def _exact_profile(self, reference: Mapping[str, Any]) -> dict[str, Any]:
        profile_id = reference["profile_id"]
        revision = reference["profile_revision"]
        try:
            profile = self.catalog.get(str(profile_id), int(revision))
        except KeyError as exc:
            raise ProductControlError(
                "profile_not_found",
                f"catalog profile not found: {profile_id} revision {revision}",
            ) from exc
        if profile["subject_digest"] != str(reference["subject_digest"]).lower():
            raise ProductControlError(
                "profile_subject_mismatch",
                "profile subject digest does not match the catalog",
            )
        return cast(dict[str, Any], profile)

    def _submit(
        self,
        operation_kind: str,
        idempotency_key: str,
        request_digest: str,
        profile: Mapping[str, Any] | None,
        work: Callable[[], Awaitable[Mapping[str, Any]]],
    ) -> dict[str, Any]:
        reference = (
            {
                "profile_id": profile["profile_id"],
                "profile_revision": profile["profile_revision"],
                "subject_digest": profile["subject_digest"],
            }
            if profile is not None
            else None
        )
        try:
            record, replayed = self.operation_state.create_control_operation(
                operation_kind=operation_kind,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                profile=reference,
            )
        except LifecycleContractError as exc:
            if "different request" in str(exc):
                raise ProductControlError("idempotency_conflict", str(exc)) from exc
            raise ProductControlError("lifecycle_conflict", str(exc)) from exc
        if replayed:
            return record
        task = asyncio.create_task(self._run(record["operation_id"], work))
        self._tasks[record["operation_id"]] = task
        task.add_done_callback(
            lambda completed: self._forget_task(record["operation_id"], completed)
        )
        return record

    def _forget_task(self, operation_id: str, _completed: asyncio.Task[None]) -> None:
        self._tasks.pop(operation_id, None)

    async def _run(
        self,
        operation_id: str,
        work: Callable[[], Awaitable[Mapping[str, Any]]],
    ) -> None:
        self.operation_state.update_control_operation(operation_id, status="running")
        try:
            result = await work()
        except asyncio.CancelledError:
            self.operation_state.update_control_operation(
                operation_id,
                status="cancelled",
                error={"code": "operation_cancelled", "message": "operation cancelled"},
            )
            raise
        except Exception:
            logger.exception("product control operation %s failed", operation_id)
            self.operation_state.update_control_operation(
                operation_id,
                status="failed",
                error={
                    "code": "runtime_unavailable",
                    "message": "operation failed; inspect protected runtime diagnostics",
                },
            )
        else:
            self.operation_state.update_control_operation(
                operation_id, status="succeeded", result=dict(result)
            )
