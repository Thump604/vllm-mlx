# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure configured/resolved/resident/leased lifecycle contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from enum import Enum

import pytest

from vllm_mlx.lifecycle_contract import (
    ConfiguredProfileRef,
    LifecycleContractError,
    LifecycleEvent,
    LifecycleSnapshot,
    ProcessState,
    ResolvedProcessRef,
    normalize_process_state,
    transition,
    validate_snapshot,
)


def _profile(profile_id: str = "qwen-35b") -> ConfiguredProfileRef:
    return ConfiguredProfileRef(profile_id, "profile-sha256")


def _resolved(profile_id: str = "qwen-35b") -> ResolvedProcessRef:
    return ResolvedProcessRef(profile_id, "profile-sha256", "config-sha256")


def _configured() -> LifecycleSnapshot:
    return transition(LifecycleSnapshot(), LifecycleEvent.CONFIGURE, profile=_profile())


def _ready() -> LifecycleSnapshot:
    snapshot = _configured()
    snapshot = transition(snapshot, LifecycleEvent.RESOLVE, resolved=_resolved())
    snapshot = transition(snapshot, LifecycleEvent.BEGIN_LOAD)
    return transition(snapshot, LifecycleEvent.LOAD_SUCCEEDED)


def test_complete_lifecycle_preserves_configuration_across_unload_then_clears():
    configured = _configured()
    resolved = transition(configured, LifecycleEvent.RESOLVE, resolved=_resolved())
    loading = transition(resolved, LifecycleEvent.BEGIN_LOAD)
    loaded = transition(loading, LifecycleEvent.LOAD_SUCCEEDED)
    leased = transition(loaded, LifecycleEvent.ACQUIRE)
    released = transition(leased, LifecycleEvent.RELEASE)
    unloading = transition(released, LifecycleEvent.BEGIN_UNLOAD)
    unloaded = transition(unloading, LifecycleEvent.UNLOAD_SUCCEEDED)
    cleared = transition(unloaded, LifecycleEvent.CLEAR)

    assert configured.resolved_process is None
    assert resolved.process_state == ProcessState.UNLOADED
    assert loading.process_state == ProcessState.LOADING
    assert loaded.process_state == ProcessState.LOADED
    assert leased.active_leases == 1
    assert released.active_leases == 0
    assert unloading.process_state == ProcessState.UNLOADING
    assert unloaded.configured_profile == _profile()
    assert unloaded.resolved_process == _resolved()
    assert cleared == LifecycleSnapshot()


def test_load_failure_preserves_identity_and_allows_retry_or_reconfiguration():
    snapshot = transition(_configured(), LifecycleEvent.RESOLVE, resolved=_resolved())
    failed = transition(
        transition(snapshot, LifecycleEvent.BEGIN_LOAD), LifecycleEvent.LOAD_FAILED
    )

    retrying = transition(failed, LifecycleEvent.BEGIN_LOAD)
    replaced = transition(
        failed,
        LifecycleEvent.CONFIGURE,
        profile=ConfiguredProfileRef("laguna", "other-profile-sha"),
    )

    assert failed.process_state == ProcessState.FAILED
    assert failed.resolved_process == _resolved()
    assert retrying.process_state == ProcessState.LOADING
    assert replaced.process_state == ProcessState.UNLOADED
    assert replaced.resolved_process is None


def test_load_cancellation_returns_to_unloaded_with_configuration_preserved():
    resolved = transition(_configured(), LifecycleEvent.RESOLVE, resolved=_resolved())
    loading = transition(resolved, LifecycleEvent.BEGIN_LOAD)

    cancelled = transition(loading, LifecycleEvent.LOAD_CANCELLED)

    assert cancelled.process_state == ProcessState.UNLOADED
    assert cancelled.configured_profile == _profile()
    assert cancelled.resolved_process == _resolved()


def test_unload_failure_rolls_back_to_loaded_without_creating_a_lease():
    unloading = transition(_ready(), LifecycleEvent.BEGIN_UNLOAD)

    restored = transition(unloading, LifecycleEvent.UNLOAD_FAILED)

    assert restored.process_state == ProcessState.LOADED
    assert restored.active_leases == 0


def test_clear_from_failed_removes_configuration_and_failure_state():
    resolved = transition(_configured(), LifecycleEvent.RESOLVE, resolved=_resolved())
    failed = transition(
        transition(resolved, LifecycleEvent.BEGIN_LOAD), LifecycleEvent.LOAD_FAILED
    )

    assert transition(failed, LifecycleEvent.CLEAR) == LifecycleSnapshot()


@pytest.mark.parametrize(
    "snapshot",
    [
        LifecycleSnapshot(resolved_process=_resolved()),
        LifecycleSnapshot(
            configured_profile=_profile(),
            resolved_process=_resolved("other"),
        ),
        LifecycleSnapshot(process_state=ProcessState.LOADING),
        LifecycleSnapshot(process_state=ProcessState.LOADED),
        LifecycleSnapshot(process_state=ProcessState.UNLOADING),
        LifecycleSnapshot(process_state=ProcessState.UNLOADED, active_leases=1),
        LifecycleSnapshot(active_leases=-1),
        LifecycleSnapshot(active_leases=True),
    ],
)
def test_invalid_snapshots_fail_closed(snapshot: LifecycleSnapshot):
    with pytest.raises(LifecycleContractError):
        validate_snapshot(snapshot)


@pytest.mark.parametrize(
    ("event", "snapshot"),
    [
        (LifecycleEvent.RESOLVE, LifecycleSnapshot()),
        (LifecycleEvent.BEGIN_LOAD, _configured()),
        (LifecycleEvent.LOAD_SUCCEEDED, _configured()),
        (LifecycleEvent.LOAD_FAILED, _configured()),
        (LifecycleEvent.LOAD_CANCELLED, _configured()),
        (LifecycleEvent.ACQUIRE, _configured()),
        (LifecycleEvent.RELEASE, _ready()),
        (
            LifecycleEvent.BEGIN_UNLOAD,
            transition(_ready(), LifecycleEvent.ACQUIRE),
        ),
        (LifecycleEvent.UNLOAD_SUCCEEDED, _ready()),
        (LifecycleEvent.UNLOAD_FAILED, _ready()),
        (LifecycleEvent.CLEAR, _ready()),
    ],
)
def test_illegal_events_raise_stable_contract_error(
    event: LifecycleEvent, snapshot: LifecycleSnapshot
):
    kwargs = {"resolved": _resolved()} if event == LifecycleEvent.RESOLVE else {}
    with pytest.raises(LifecycleContractError):
        transition(snapshot, event, **kwargs)


def test_resolve_rejects_config_identity_mismatch():
    with pytest.raises(LifecycleContractError, match="identity does not match"):
        transition(
            _configured(),
            LifecycleEvent.RESOLVE,
            resolved=_resolved("other-model"),
        )


def test_resolve_cannot_silently_replace_existing_process_configuration():
    resolved = transition(_configured(), LifecycleEvent.RESOLVE, resolved=_resolved())
    replacement = ResolvedProcessRef(
        "qwen-35b", "profile-sha256", "different-config-sha256"
    )

    with pytest.raises(LifecycleContractError, match="cannot replace"):
        transition(resolved, LifecycleEvent.RESOLVE, resolved=replacement)


@pytest.mark.parametrize(
    ("event", "kwargs"),
    [
        (LifecycleEvent.CONFIGURE, {}),
        (LifecycleEvent.CONFIGURE, {"profile": _profile(), "resolved": _resolved()}),
        (LifecycleEvent.RESOLVE, {}),
        (LifecycleEvent.RESOLVE, {"profile": _profile(), "resolved": _resolved()}),
        (LifecycleEvent.BEGIN_LOAD, {"profile": _profile()}),
    ],
)
def test_event_payload_contract_is_explicit(event: LifecycleEvent, kwargs: dict):
    with pytest.raises(LifecycleContractError):
        transition(_configured(), event, **kwargs)


@pytest.mark.parametrize(
    "value",
    [
        ConfiguredProfileRef("", "revision"),
        ConfiguredProfileRef("model", " "),
    ],
)
def test_configured_identity_requires_nonempty_fields(value: ConfiguredProfileRef):
    with pytest.raises(LifecycleContractError):
        transition(LifecycleSnapshot(), LifecycleEvent.CONFIGURE, profile=value)


@pytest.mark.parametrize(
    "value",
    [
        ResolvedProcessRef("", "revision", "digest"),
        ResolvedProcessRef("model", " ", "digest"),
        ResolvedProcessRef("model", "revision", ""),
    ],
)
def test_resolved_identity_requires_nonempty_fields(value: ResolvedProcessRef):
    with pytest.raises(LifecycleContractError):
        transition(_configured(), LifecycleEvent.RESOLVE, resolved=value)


def test_snapshots_and_identity_refs_are_immutable():
    snapshot = _ready()

    with pytest.raises(FrozenInstanceError):
        snapshot.active_leases = 2
    assert snapshot.configured_profile is not None
    with pytest.raises(FrozenInstanceError):
        snapshot.configured_profile.profile_id = "changed"


def test_validate_rejects_non_enum_process_state():
    snapshot = replace(LifecycleSnapshot(), process_state="loaded")

    with pytest.raises(LifecycleContractError, match="ProcessState"):
        validate_snapshot(snapshot)


def test_existing_resident_state_has_an_explicit_compatibility_boundary():
    from vllm_mlx.lifecycle import ResidentState

    assert normalize_process_state(ResidentState.LOADED) == ProcessState.LOADED
    snapshot = LifecycleSnapshot(
        configured_profile=_profile(),
        resolved_process=_resolved(),
        process_state=ResidentState.LOADED,
    )
    validate_snapshot(snapshot)


def test_foreign_lookalike_enum_is_rejected():
    class ForeignState(str, Enum):
        LOADED = "loaded"

    snapshot = replace(LifecycleSnapshot(), process_state=ForeignState.LOADED)

    with pytest.raises(LifecycleContractError, match="ProcessState-compatible"):
        validate_snapshot(snapshot)
