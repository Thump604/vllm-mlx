# SPDX-License-Identifier: Apache-2.0
"""Crash-durable lifecycle state tracking for the concrete residency manager."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict
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
            "owner": {"pid": os.getpid(), "acquired_at": self._owner_acquired_at},
        }

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
        model_key = payload.get("model_key")
        if not isinstance(model_key, str) or not model_key:
            raise LifecycleContractError("lifecycle control model_key is invalid")
        self._model_key = model_key
        self._snapshot = _snapshot_from_payload(payload.get("snapshot"))
        recovery = payload.get("recovery")
        self._recovery = recovery if isinstance(recovery, dict) else None
