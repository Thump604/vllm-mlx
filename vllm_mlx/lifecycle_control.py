# SPDX-License-Identifier: Apache-2.0
"""Crash-durable lifecycle state tracking for the concrete residency manager."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, TextIO, cast

import fcntl

from .lifecycle_contract import (
    ConfiguredProfileRef,
    LifecycleContractError,
    LifecycleEvent,
    LifecycleSnapshot,
    ProcessState,
    ResolvedProcessRef,
    transition,
    validate_snapshot,
)

CONTROL_STATE_VERSION = 1
_ACTIVE_STATE_PATHS: set[str] = set()
_ACTIVE_STATE_PATHS_LOCK = threading.Lock()


def model_spec_digest(spec: object) -> str:
    """Return a stable digest for dataclass-based resident construction inputs."""
    try:
        payload = asdict(cast(Any, spec))
    except (TypeError, ValueError) as exc:
        raise LifecycleContractError("model spec must be a dataclass instance") from exc
    # Default-valued fields added for dynamic activation do not change engine
    # construction and therefore must not invalidate legacy lifecycle state.
    for field_name, default in (
        ("trust_remote_code", False),
        ("mllm_draft_model", None),
        ("mllm_draft_kind", None),
        ("mllm_draft_block_size", None),
    ):
        if payload.get(field_name) == default:
            payload.pop(field_name, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_payload(snapshot: LifecycleSnapshot) -> dict[str, Any]:
    return {
        "configured_profile": (
            asdict(snapshot.configured_profile)
            if snapshot.configured_profile is not None
            else None
        ),
        "resolved_process": (
            asdict(snapshot.resolved_process)
            if snapshot.resolved_process is not None
            else None
        ),
        "process_state": ProcessState(snapshot.process_state).value,
        "active_leases": snapshot.active_leases,
    }


def _snapshot_from_payload(payload: object) -> LifecycleSnapshot:
    if not isinstance(payload, dict):
        raise LifecycleContractError("lifecycle control snapshot must be an object")
    configured = payload.get("configured_profile")
    resolved = payload.get("resolved_process")
    try:
        snapshot = LifecycleSnapshot(
            configured_profile=(
                ConfiguredProfileRef(**configured)
                if isinstance(configured, dict)
                else None
            ),
            resolved_process=(
                ResolvedProcessRef(**resolved) if isinstance(resolved, dict) else None
            ),
            process_state=ProcessState(payload.get("process_state", "unloaded")),
            active_leases=payload.get("active_leases", 0),
        )
    except (TypeError, ValueError) as exc:
        raise LifecycleContractError(
            f"invalid lifecycle control snapshot: {exc}"
        ) from exc
    validate_snapshot(snapshot)
    return snapshot


class LifecycleControlState:
    """Persist pure lifecycle transitions driven by ``ResidencyManager``."""

    def __init__(self, state_path: str | Path) -> None:
        self._state_path = Path(state_path).expanduser()
        self._state_key = str(self._state_path.resolve())
        self._lock_handle: TextIO | None = None
        self._model_key: str | None = None
        self._snapshot = LifecycleSnapshot()
        self._recovery: dict[str, Any] | None = None
        self._product_control: dict[str, dict[str, Any]] = {
            "operations": {},
            "idempotency": {},
        }
        self._acquire_owner_lock()
        try:
            if self._state_path.exists():
                self._load()
        except BaseException:
            self.close()
            raise

    @property
    def model_key(self) -> str | None:
        return self._model_key

    @property
    def is_open(self) -> bool:
        return self._lock_handle is not None

    def configure(self, spec: object) -> None:
        model_key = getattr(spec, "model_key", None)
        model_name = getattr(spec, "model_name", None)
        if not isinstance(model_key, str) or not model_key:
            raise LifecycleContractError("model spec requires a non-empty model_key")
        if not isinstance(model_name, str) or not model_name:
            raise LifecycleContractError("model spec requires a non-empty model_name")
        digest = model_spec_digest(spec)
        profile_id = getattr(spec, "profile_id", None) or model_name
        profile_revision = getattr(spec, "profile_revision", None) or (
            f"launch-{digest[:16]}"
        )
        config_digest = getattr(spec, "config_digest", None) or digest
        profile = ConfiguredProfileRef(profile_id, profile_revision)
        resolved = ResolvedProcessRef(profile_id, profile_revision, config_digest)
        same_identity = (
            self._model_key == model_key
            and self._snapshot.configured_profile == profile
            and self._snapshot.resolved_process == resolved
        )
        if self._snapshot.configured_profile is not None and not same_identity:
            if (
                self._snapshot.process_state != ProcessState.UNLOADED
                or self._snapshot.active_leases
            ):
                raise LifecycleContractError(
                    "cannot replace lifecycle configuration while a model is active"
                )
        if not same_identity:
            snapshot = transition(
                LifecycleSnapshot(), LifecycleEvent.CONFIGURE, profile=profile
            )
            previous = (self._snapshot, self._model_key, self._recovery)
            try:
                self._snapshot = transition(
                    snapshot, LifecycleEvent.RESOLVE, resolved=resolved
                )
                self._model_key = model_key
                self._recovery = None
                self._persist()
            except BaseException:
                self._snapshot, self._model_key, self._recovery = previous
                raise

    def clear_configuration(self) -> None:
        """Remove a dormant candidate that never became resident."""
        if (
            self._snapshot.process_state
            not in {
                ProcessState.UNLOADED,
                ProcessState.FAILED,
            }
            or self._snapshot.active_leases
        ):
            raise LifecycleContractError(
                "cannot clear lifecycle configuration while a model is active"
            )
        previous = (self._snapshot, self._model_key, self._recovery)
        try:
            self._snapshot = LifecycleSnapshot()
            self._model_key = None
            self._recovery = {"reason": "failed_candidate_cleared"}
            self._persist()
        except BaseException:
            self._snapshot, self._model_key, self._recovery = previous
            raise

    def reconcile(self, state: ProcessState, active_leases: int) -> None:
        if self._snapshot.configured_profile is None:
            raise LifecycleContractError("lifecycle control is not configured")
        leases = active_leases if state == ProcessState.LOADED else 0
        reconciled = LifecycleSnapshot(
            configured_profile=self._snapshot.configured_profile,
            resolved_process=self._snapshot.resolved_process,
            process_state=state,
            active_leases=leases,
        )
        validate_snapshot(reconciled)
        if reconciled != self._snapshot:
            previous = (self._snapshot, self._recovery)
            try:
                self._recovery = {
                    "reason": "persisted_and_resident_state_disagreed",
                    "previous": _snapshot_payload(self._snapshot),
                    "reconciled": _snapshot_payload(reconciled),
                }
                self._snapshot = reconciled
                self._persist()
            except BaseException:
                self._snapshot, self._recovery = previous
                raise

    def apply(self, event: LifecycleEvent) -> None:
        previous = self._snapshot
        try:
            self._snapshot = transition(self._snapshot, event)
            self._persist()
        except BaseException:
            self._snapshot = previous
            raise

    def status(self) -> dict[str, Any]:
        return {
            "kind": "vllm-mlx-lifecycle-control-state",
            "version": CONTROL_STATE_VERSION,
            "model_key": self._model_key,
            "snapshot": _snapshot_payload(self._snapshot),
            "recovery": self._recovery,
            "product_control": deepcopy(self._product_control),
            "owner": {"pid": os.getpid(), "acquired_at": self._owner_acquired_at},
        }

    def create_product_operation(
        self,
        *,
        operation_kind: str,
        idempotency_key: str,
        request_digest: str,
        profile: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        """Create one durable queued operation or replay its exact prior record."""
        ledger_key = f"{operation_kind}:{idempotency_key}"
        binding = self._product_control["idempotency"].get(ledger_key)
        if binding is not None:
            if binding["request_digest"] != request_digest:
                raise LifecycleContractError(
                    "idempotency key was already used for a different request"
                )
            return self.get_product_operation(binding["operation_id"]), True

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        operation_id = uuid.uuid4().hex
        record = {
            "operation_id": operation_id,
            "operation_kind": operation_kind,
            "status": "queued",
            "profile": deepcopy(profile),
            "created_at": timestamp,
            "updated_at": timestamp,
            "result": None,
            "error": None,
            "recovery": None,
        }
        self._product_control["operations"][operation_id] = record
        self._product_control["idempotency"][ledger_key] = {
            "operation_id": operation_id,
            "request_digest": request_digest,
        }
        self._persist()
        return deepcopy(record), False

    def bind_product_idempotency(
        self,
        *,
        operation_kind: str,
        idempotency_key: str,
        request_digest: str,
        operation_id: str,
    ) -> bool:
        """Bind an idempotent command to an existing durable operation."""
        if operation_id not in self._product_control["operations"]:
            raise KeyError(operation_id)
        ledger_key = f"{operation_kind}:{idempotency_key}"
        binding = self._product_control["idempotency"].get(ledger_key)
        if binding is not None:
            if (
                binding["request_digest"] != request_digest
                or binding["operation_id"] != operation_id
            ):
                raise LifecycleContractError(
                    "idempotency key was already used for a different request"
                )
            return True
        self._product_control["idempotency"][ledger_key] = {
            "operation_id": operation_id,
            "request_digest": request_digest,
        }
        self._persist()
        return False

    def get_product_operation(self, operation_id: str) -> dict[str, Any]:
        """Return one durable product operation record."""
        record = self._product_control["operations"].get(operation_id)
        if record is None:
            raise KeyError(operation_id)
        return deepcopy(record)

    def update_product_operation(
        self,
        operation_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a legal product-operation state transition."""
        if status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"invalid product operation status: {status}")
        record = self._product_control["operations"].get(operation_id)
        if record is None:
            raise KeyError(operation_id)
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            if record["status"] != status:
                raise LifecycleContractError(
                    "terminal product operation status cannot change"
                )
            return deepcopy(record)
        record.update(
            status=status,
            updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            result=deepcopy(result),
            error=deepcopy(error),
        )
        self._persist()
        return deepcopy(record)

    def close(self) -> None:
        handle = self._lock_handle
        if handle is None:
            return
        self._lock_handle = None
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        with _ACTIVE_STATE_PATHS_LOCK:
            _ACTIVE_STATE_PATHS.discard(self._state_key)

    def _acquire_owner_lock(self) -> None:
        with _ACTIVE_STATE_PATHS_LOCK:
            if self._state_key in _ACTIVE_STATE_PATHS:
                raise LifecycleContractError(
                    f"lifecycle state is already owned by this process: {self._state_path}"
                )
            _ACTIVE_STATE_PATHS.add(self._state_key)
        lock_path = self._state_path.with_suffix(f"{self._state_path.suffix}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            with _ACTIVE_STATE_PATHS_LOCK:
                _ACTIVE_STATE_PATHS.discard(self._state_key)
            raise LifecycleContractError(
                f"lifecycle state is owned by another process: {self._state_path}"
            ) from exc
        self._lock_handle = handle
        self._owner_acquired_at = time.time()

    def _persist(self) -> None:
        _write_json_atomic(self._state_path, self.status())

    def _load(self) -> None:
        try:
            payload = json.loads(self._state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleContractError(
                f"invalid lifecycle control state at {self._state_path}: {exc}"
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "vllm-mlx-lifecycle-control-state"
            or payload.get("version") != CONTROL_STATE_VERSION
        ):
            raise LifecycleContractError(
                f"unsupported lifecycle control state at {self._state_path}"
            )
        self._snapshot = _snapshot_from_payload(payload.get("snapshot"))
        model_key = payload.get("model_key")
        if model_key is None and self._snapshot == LifecycleSnapshot():
            self._model_key = None
        elif isinstance(model_key, str) and model_key:
            self._model_key = model_key
        else:
            raise LifecycleContractError("lifecycle control model_key is invalid")
        recovery = payload.get("recovery")
        self._recovery = recovery if isinstance(recovery, dict) else None
        product_control = payload.get("product_control")
        if product_control is not None:
            if (
                not isinstance(product_control, dict)
                or not isinstance(product_control.get("operations"), dict)
                or not isinstance(product_control.get("idempotency"), dict)
            ):
                raise LifecycleContractError(
                    f"invalid product control state at {self._state_path}"
                )
            self._product_control = deepcopy(product_control)
        interrupted = False
        for record in self._product_control["operations"].values():
            if record.get("status") in {"queued", "running"}:
                record.update(
                    status="failed",
                    updated_at=datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    error={
                        "code": "runtime_unavailable",
                        "message": "operation was interrupted by control-plane restart",
                    },
                    recovery={"reason": "control_plane_restart"},
                )
                interrupted = True
        if interrupted:
            self._persist()
