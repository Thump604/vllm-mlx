# SPDX-License-Identifier: Apache-2.0
"""Pure lifecycle state and transition contract.

This module describes configured, resolved, resident, and leased state without
loading a model or depending on either lifecycle manager implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class LifecycleContractError(ValueError):
    """Raised when a snapshot or requested lifecycle transition is invalid."""


class ProcessState(str, Enum):
    """Resident-process state shared with the existing lifecycle vocabulary."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    FAILED = "failed"


class LifecycleEvent(str, Enum):
    """Events accepted by the pure lifecycle transition function."""

    CONFIGURE = "configure"
    RESOLVE = "resolve"
    BEGIN_LOAD = "begin_load"
    LOAD_SUCCEEDED = "load_succeeded"
    LOAD_FAILED = "load_failed"
    LOAD_CANCELLED = "load_cancelled"
    ACQUIRE = "acquire"
    RELEASE = "release"
    BEGIN_UNLOAD = "begin_unload"
    UNLOAD_SUCCEEDED = "unload_succeeded"
    UNLOAD_FAILED = "unload_failed"
    CLEAR = "clear"


@dataclass(frozen=True)
class ConfiguredProfileRef:
    """Immutable identity of the profile selected for a lifecycle instance."""

    profile_id: str
    profile_revision: str


@dataclass(frozen=True)
class ResolvedProcessRef:
    """Immutable effective process configuration bound to one profile revision."""

    profile_id: str
    profile_revision: str
    config_digest: str


@dataclass(frozen=True)
class LifecycleSnapshot:
    """Orthogonal configured, resolved, resident, and request-lease state."""

    configured_profile: ConfiguredProfileRef | None = None
    resolved_process: ResolvedProcessRef | None = None
    process_state: ProcessState = ProcessState.UNLOADED
    active_leases: int = 0


def validate_snapshot(snapshot: LifecycleSnapshot) -> None:
    """Validate a snapshot or raise a stable lifecycle contract error."""
    if not isinstance(snapshot, LifecycleSnapshot):
        raise LifecycleContractError("snapshot must be a LifecycleSnapshot")
    process_state = normalize_process_state(snapshot.process_state)
    if (
        not isinstance(snapshot.active_leases, int)
        or isinstance(snapshot.active_leases, bool)
        or snapshot.active_leases < 0
    ):
        raise LifecycleContractError("active_leases must be a non-negative integer")

    profile = snapshot.configured_profile
    resolved = snapshot.resolved_process
    if profile is not None:
        _validate_profile_ref(profile)
    if resolved is not None:
        _validate_resolved_ref(resolved)
        if profile is None:
            raise LifecycleContractError(
                "resolved process configuration requires a configured profile"
            )
        if (
            resolved.profile_id != profile.profile_id
            or resolved.profile_revision != profile.profile_revision
        ):
            raise LifecycleContractError(
                "resolved process configuration identity does not match the configured profile"
            )

    if (
        process_state
        in {
            ProcessState.LOADING,
            ProcessState.LOADED,
            ProcessState.UNLOADING,
        }
        and resolved is None
    ):
        raise LifecycleContractError(
            f"{process_state.value} process state requires a resolved process configuration"
        )
    if snapshot.active_leases and process_state != ProcessState.LOADED:
        raise LifecycleContractError("active leases are allowed only while loaded")


def normalize_process_state(value: object) -> ProcessState:
    """Normalize the explicit existing Runtime lifecycle boundary."""
    if isinstance(value, ProcessState):
        return value
    from vllm_mlx.lifecycle import ResidentState

    if isinstance(value, ResidentState):
        return ProcessState(value.value)
    raise LifecycleContractError("process_state must be a ProcessState-compatible enum")


def transition(
    snapshot: LifecycleSnapshot,
    event: LifecycleEvent,
    *,
    profile: ConfiguredProfileRef | None = None,
    resolved: ResolvedProcessRef | None = None,
) -> LifecycleSnapshot:
    """Apply one legal event and return a new validated snapshot."""
    validate_snapshot(snapshot)
    if not isinstance(event, LifecycleEvent):
        raise LifecycleContractError("event must be a LifecycleEvent")
    _validate_event_payload(event, profile=profile, resolved=resolved)

    if event == LifecycleEvent.CONFIGURE:
        _require_configurable(snapshot, event)
        assert profile is not None
        result = LifecycleSnapshot(configured_profile=profile)
    elif event == LifecycleEvent.RESOLVE:
        _require_configurable(snapshot, event)
        if snapshot.configured_profile is None:
            raise LifecycleContractError("resolve requires a configured profile")
        if snapshot.resolved_process is not None:
            raise LifecycleContractError(
                "resolve cannot replace an existing process configuration; configure or clear first"
            )
        assert resolved is not None
        result = replace(
            snapshot, resolved_process=resolved, process_state=ProcessState.UNLOADED
        )
    elif event == LifecycleEvent.BEGIN_LOAD:
        _require_state(snapshot, event, ProcessState.UNLOADED, ProcessState.FAILED)
        if snapshot.resolved_process is None:
            raise LifecycleContractError(
                "begin_load requires a resolved process configuration"
            )
        result = replace(snapshot, process_state=ProcessState.LOADING)
    elif event == LifecycleEvent.LOAD_SUCCEEDED:
        _require_state(snapshot, event, ProcessState.LOADING)
        result = replace(snapshot, process_state=ProcessState.LOADED)
    elif event == LifecycleEvent.LOAD_FAILED:
        _require_state(snapshot, event, ProcessState.LOADING)
        result = replace(snapshot, process_state=ProcessState.FAILED)
    elif event == LifecycleEvent.LOAD_CANCELLED:
        _require_state(snapshot, event, ProcessState.LOADING)
        result = replace(snapshot, process_state=ProcessState.UNLOADED)
    elif event == LifecycleEvent.ACQUIRE:
        _require_state(snapshot, event, ProcessState.LOADED)
        result = replace(snapshot, active_leases=snapshot.active_leases + 1)
    elif event == LifecycleEvent.RELEASE:
        _require_state(snapshot, event, ProcessState.LOADED)
        if snapshot.active_leases == 0:
            raise LifecycleContractError("release cannot underflow active leases")
        result = replace(snapshot, active_leases=snapshot.active_leases - 1)
    elif event == LifecycleEvent.BEGIN_UNLOAD:
        _require_state(snapshot, event, ProcessState.LOADED)
        if snapshot.active_leases:
            raise LifecycleContractError("begin_unload requires zero active leases")
        result = replace(snapshot, process_state=ProcessState.UNLOADING)
    elif event == LifecycleEvent.UNLOAD_SUCCEEDED:
        _require_state(snapshot, event, ProcessState.UNLOADING)
        result = replace(snapshot, process_state=ProcessState.UNLOADED)
    elif event == LifecycleEvent.UNLOAD_FAILED:
        _require_state(snapshot, event, ProcessState.UNLOADING)
        result = replace(snapshot, process_state=ProcessState.LOADED)
    else:
        assert event == LifecycleEvent.CLEAR
        _require_configurable(snapshot, event)
        result = LifecycleSnapshot()

    validate_snapshot(result)
    return result


def _validate_profile_ref(value: ConfiguredProfileRef) -> None:
    if not isinstance(value, ConfiguredProfileRef):
        raise LifecycleContractError(
            "configured_profile must be a ConfiguredProfileRef"
        )
    _require_nonempty("configured profile_id", value.profile_id)
    _require_nonempty("configured profile_revision", value.profile_revision)


def _validate_resolved_ref(value: ResolvedProcessRef) -> None:
    if not isinstance(value, ResolvedProcessRef):
        raise LifecycleContractError("resolved_process must be a ResolvedProcessRef")
    _require_nonempty("resolved profile_id", value.profile_id)
    _require_nonempty("resolved profile_revision", value.profile_revision)
    _require_nonempty("resolved config_digest", value.config_digest)


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleContractError(f"{name} must be a non-empty string")


def _validate_event_payload(
    event: LifecycleEvent,
    *,
    profile: ConfiguredProfileRef | None,
    resolved: ResolvedProcessRef | None,
) -> None:
    if event == LifecycleEvent.CONFIGURE:
        if profile is None or resolved is not None:
            raise LifecycleContractError("configure requires only a profile payload")
        _validate_profile_ref(profile)
    elif event == LifecycleEvent.RESOLVE:
        if resolved is None or profile is not None:
            raise LifecycleContractError("resolve requires only a resolved payload")
        _validate_resolved_ref(resolved)
    elif profile is not None or resolved is not None:
        raise LifecycleContractError(f"{event.value} does not accept a payload")


def _require_configurable(snapshot: LifecycleSnapshot, event: LifecycleEvent) -> None:
    _require_state(snapshot, event, ProcessState.UNLOADED, ProcessState.FAILED)
    if snapshot.active_leases:
        raise LifecycleContractError(f"{event.value} requires zero active leases")


def _require_state(
    snapshot: LifecycleSnapshot, event: LifecycleEvent, *allowed: ProcessState
) -> None:
    process_state = normalize_process_state(snapshot.process_state)
    if process_state not in allowed:
        expected = ", ".join(state.value for state in allowed)
        raise LifecycleContractError(
            f"{event.value} requires process state {expected}; current state is {process_state.value}"
        )
