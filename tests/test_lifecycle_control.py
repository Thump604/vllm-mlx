# SPDX-License-Identifier: Apache-2.0
"""Integration tests for persistent control in the concrete residency manager."""

from __future__ import annotations

import asyncio
import json

import pytest

import vllm_mlx.lifecycle_control as lifecycle_control
from vllm_mlx.lifecycle import (
    ModelSpec,
    ResidencyManager,
    bind_model_spec_to_profile,
)
from vllm_mlx.lifecycle_contract import LifecycleContractError
from vllm_mlx.model_profile import compute_subject_digest


class FakeEngine:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1


def _spec(name: str = "laguna") -> ModelSpec:
    return ModelSpec(
        model_key=name,
        model_name=f"{name}-model",
        profile_id=name,
        profile_revision="rev-1",
        config_digest="digest-1",
    )


def test_profile_binding_carries_immutable_identity_into_lifecycle(
    tmp_path, monkeypatch
):
    profile = {
        "profile_id": "laguna-s-2.1",
        "profile_revision": 3,
        "identity": {"served_model_name": "laguna-s-2.1"},
    }
    profile["subject_digest"] = compute_subject_digest(profile)
    bound = bind_model_spec_to_profile(
        ModelSpec(model_key="laguna", model_name="/models/laguna"), profile
    )
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))
    manager = ResidencyManager(lambda spec: FakeEngine())
    manager.register_model(bound)

    control = manager.get_status("laguna")["control"]["snapshot"]
    assert control["configured_profile"] == {
        "profile_id": "laguna-s-2.1",
        "profile_revision": "3",
    }
    assert control["resolved_process"]["config_digest"] == profile["subject_digest"]
    manager._control.close()


def test_profile_binding_rejects_stale_or_conflicting_identity():
    profile = {
        "profile_id": "laguna-s-2.1",
        "profile_revision": 3,
        "identity": {"served_model_name": "laguna-s-2.1"},
    }
    profile["subject_digest"] = compute_subject_digest(profile)

    with pytest.raises(ValueError, match="conflicts"):
        bind_model_spec_to_profile(
            ModelSpec(
                model_key="laguna",
                model_name="/models/laguna",
                profile_id="other",
            ),
            profile,
        )
    profile["subject_digest"] = "0" * 64
    with pytest.raises(ValueError, match="stale"):
        bind_model_spec_to_profile(
            ModelSpec(model_key="laguna", model_name="/models/laguna"), profile
        )


@pytest.mark.anyio
async def test_manager_persists_activate_acquire_release_stop(tmp_path, monkeypatch):
    state_path = tmp_path / "lifecycle.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))
    engines: list[FakeEngine] = []

    async def factory(spec):
        engine = FakeEngine()
        engines.append(engine)
        return engine

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    engine = await manager.acquire("laguna")
    status = manager.get_status("laguna")

    assert engine is engines[0]
    assert status["control"]["snapshot"]["active_leases"] == 1

    await manager.release("laguna")
    assert await manager.unload("laguna") is True
    status = manager.get_status("laguna")

    assert status["control"]["snapshot"]["process_state"] == "unloaded"
    assert engines[0].started == 1
    assert engines[0].stopped == 1
    assert (
        json.loads(state_path.read_text())["snapshot"] == status["control"]["snapshot"]
    )
    await manager.shutdown()


@pytest.mark.anyio
async def test_unload_rejects_active_request_with_persistent_control(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))

    async def factory(spec):
        return FakeEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    await manager.acquire("laguna")

    with pytest.raises(RuntimeError, match="active requests"):
        await manager.unload("laguna")

    await manager.release("laguna")
    await manager.unload("laguna")
    await manager.shutdown()


def test_persistent_control_rejects_second_model(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))
    manager = ResidencyManager(lambda spec: FakeEngine())
    manager.register_model(_spec())

    with pytest.raises(RuntimeError, match="one configured resident"):
        manager.register_model(_spec("other"))
    manager._control.close()


@pytest.mark.anyio
async def test_restart_registers_spec_and_reconciles_stale_loaded_state(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))

    async def factory(spec):
        return FakeEngine()

    first = ResidencyManager(factory)
    first.register_model(_spec())
    await first.ensure_loaded("laguna")

    with pytest.raises(LifecycleContractError, match="already owned"):
        ResidencyManager(factory)

    # Simulate process death: OS releases the ownership lock while persisted
    # state still says loaded; the in-process fake engine is no longer relevant.
    first._control.close()

    restarted = ResidencyManager(factory)
    restarted.register_model(_spec())
    status = restarted.get_status("laguna")

    assert status["control"]["snapshot"]["process_state"] == "unloaded"
    assert status["control"]["recovery"]["reason"] == (
        "persisted_and_resident_state_disagreed"
    )
    await restarted.shutdown()


@pytest.mark.anyio
async def test_cancelled_load_persists_unloaded_state(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))
    entered = asyncio.Event()

    class BlockingEngine(FakeEngine):
        async def start(self):
            entered.set()
            await asyncio.Event().wait()

    async def factory(spec):
        return BlockingEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    task = asyncio.create_task(manager.ensure_loaded("laguna"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (
        manager.get_status("laguna")["control"]["snapshot"]["process_state"]
        == "unloaded"
    )
    await manager.shutdown()


@pytest.mark.anyio
async def test_explicit_unload_cancels_inflight_load(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))
    entered = asyncio.Event()

    class BlockingEngine(FakeEngine):
        async def start(self):
            entered.set()
            await asyncio.Event().wait()

    async def factory(spec):
        return BlockingEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    load = asyncio.create_task(manager.ensure_loaded("laguna"))
    await entered.wait()

    assert await manager.unload("laguna") is False
    with pytest.raises(asyncio.CancelledError):
        await load
    assert (
        manager.get_status("laguna")["control"]["snapshot"]["process_state"]
        == "unloaded"
    )
    await manager.shutdown()


@pytest.mark.anyio
async def test_persistence_failures_roll_back_resident_and_control_state(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))

    async def factory(spec):
        return FakeEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    original_write = lifecycle_control._write_json_atomic

    def fail_write(path, payload):
        raise OSError("disk unavailable")

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        await manager.ensure_loaded("laguna")
    assert manager._residents["laguna"].state.value == "unloaded"
    assert manager._control.status()["snapshot"]["process_state"] == "unloaded"

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    await manager.ensure_loaded("laguna")

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        await manager.acquire("laguna")
    assert manager._residents["laguna"].active_requests == 0
    assert manager._control.status()["snapshot"]["active_leases"] == 0

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    await manager.acquire("laguna")

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        await manager.release("laguna")
    assert manager._residents["laguna"].active_requests == 1
    assert manager._control.status()["snapshot"]["active_leases"] == 1

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    await manager.release("laguna")

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        await manager.unload("laguna")
    assert manager._residents["laguna"].state.value == "loaded"
    assert manager._control.status()["snapshot"]["process_state"] == "loaded"

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    await manager.shutdown()


@pytest.mark.anyio
async def test_cancelled_shutdown_releases_control_after_unload(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))
    stop_entered = asyncio.Event()
    allow_stop = asyncio.Event()

    class BlockingStopEngine(FakeEngine):
        async def stop(self):
            stop_entered.set()
            await allow_stop.wait()
            await super().stop()

    async def factory(spec):
        return BlockingStopEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    await manager.ensure_loaded("laguna")
    shutdown = asyncio.create_task(manager.shutdown())
    await stop_entered.wait()
    shutdown.cancel()
    allow_stop.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown

    restarted = ResidencyManager(factory)
    restarted.register_model(_spec())
    assert restarted.get_status("laguna")["state"] == "unloaded"
    with pytest.raises(RuntimeError, match="shutting down"):
        await manager.ensure_loaded("laguna")
    await restarted.shutdown()


@pytest.mark.anyio
async def test_cancelled_shutdown_during_load_cleanup_propagates_and_releases_owner(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))
    start_entered = asyncio.Event()
    stop_entered = asyncio.Event()
    allow_stop = asyncio.Event()

    class BlockingLoadCleanupEngine(FakeEngine):
        async def start(self):
            start_entered.set()
            await asyncio.Event().wait()

        async def stop(self):
            stop_entered.set()
            await allow_stop.wait()
            await super().stop()

    async def factory(spec):
        return BlockingLoadCleanupEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    load = asyncio.create_task(manager.ensure_loaded("laguna"))
    await start_entered.wait()
    shutdown = asyncio.create_task(manager.shutdown())
    await stop_entered.wait()
    shutdown.cancel()
    allow_stop.set()

    with pytest.raises(asyncio.CancelledError):
        await shutdown
    with pytest.raises(asyncio.CancelledError):
        await load

    restarted = ResidencyManager(factory)
    restarted.register_model(_spec())
    with pytest.raises(RuntimeError, match="shutting down"):
        await manager.ensure_loaded("laguna")
    await restarted.shutdown()


@pytest.mark.anyio
async def test_shutdown_rejects_concurrent_acquire_and_releases_owner(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))
    stop_entered = asyncio.Event()
    allow_stop = asyncio.Event()

    class BlockingStopEngine(FakeEngine):
        async def stop(self):
            stop_entered.set()
            await allow_stop.wait()
            await super().stop()

    async def factory(spec):
        return BlockingStopEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    await manager.ensure_loaded("laguna")
    shutdown = asyncio.create_task(manager.shutdown())
    await stop_entered.wait()

    with pytest.raises(RuntimeError, match="shutting down"):
        await manager.acquire("laguna")

    allow_stop.set()
    await shutdown
    restarted = ResidencyManager(factory)
    restarted.register_model(_spec())
    await restarted.shutdown()


@pytest.mark.anyio
async def test_shutdown_detaches_stale_manager_from_replacement_state(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))

    async def factory(spec):
        return FakeEngine()

    first = ResidencyManager(factory)
    first.register_model(_spec())
    await first.ensure_loaded("laguna")
    await first.shutdown()

    replacement = ResidencyManager(factory)
    replacement.register_model(_spec())
    await replacement.ensure_loaded("laguna")
    replacement_state = state_path.read_text()

    assert "control" not in first.get_status("laguna")
    assert state_path.read_text() == replacement_state
    await replacement.shutdown()


def _fail_next_persisted_state(monkeypatch, target_state):
    original_write = lifecycle_control._write_json_atomic
    failed = False

    def fail_once(path, payload):
        nonlocal failed
        if not failed and payload["snapshot"]["process_state"] == target_state:
            failed = True
            raise OSError(f"cannot persist {target_state}")
        original_write(path, payload)

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", fail_once)
    return original_write


@pytest.mark.anyio
async def test_load_success_persistence_failure_requires_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))

    async def factory(spec):
        return FakeEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    original_write = _fail_next_persisted_state(monkeypatch, "loaded")

    with pytest.raises(OSError, match="cannot persist loaded"):
        await manager.ensure_loaded("laguna")
    assert manager._residents["laguna"].state.value == "loaded"
    assert manager._control.status()["snapshot"]["process_state"] == "loading"

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    status = manager.get_status("laguna")
    assert status["control"]["snapshot"]["process_state"] == "loaded"
    assert status["control_sync_error"] is None
    await manager.shutdown()


@pytest.mark.anyio
async def test_load_failure_persistence_failure_requires_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))

    async def factory(spec):
        raise ValueError("engine failed")

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    original_write = _fail_next_persisted_state(monkeypatch, "failed")

    with pytest.raises(OSError, match="cannot persist failed"):
        await manager.ensure_loaded("laguna")
    assert manager._residents["laguna"].state.value == "failed"
    assert manager._control.status()["snapshot"]["process_state"] == "loading"

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    assert (
        manager.get_status("laguna")["control"]["snapshot"]["process_state"] == "failed"
    )
    manager._control.close()


@pytest.mark.anyio
async def test_unload_terminal_persistence_failures_require_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))

    async def factory(spec):
        return FakeEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    await manager.ensure_loaded("laguna")
    original_write = _fail_next_persisted_state(monkeypatch, "unloaded")

    with pytest.raises(OSError, match="cannot persist unloaded"):
        await manager.unload("laguna")
    assert manager._residents["laguna"].state.value == "unloaded"
    assert manager._control.status()["snapshot"]["process_state"] == "unloading"

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    assert (
        manager.get_status("laguna")["control"]["snapshot"]["process_state"]
        == "unloaded"
    )
    await manager.shutdown()

    class FailingStopEngine(FakeEngine):
        async def stop(self):
            raise ValueError("stop failed")

    async def failing_factory(spec):
        return FailingStopEngine()

    second_path = tmp_path / "second.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(second_path))
    manager = ResidencyManager(failing_factory)
    manager.register_model(_spec())
    await manager.ensure_loaded("laguna")
    original_write = _fail_next_persisted_state(monkeypatch, "loaded")

    with pytest.raises(OSError, match="cannot persist loaded"):
        await manager.unload("laguna")
    assert manager._residents["laguna"].state.value == "loaded"
    assert manager._control.status()["snapshot"]["process_state"] == "unloading"

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    assert (
        manager.get_status("laguna")["control"]["snapshot"]["process_state"] == "loaded"
    )
    manager._control.close()


@pytest.mark.anyio
async def test_cancelled_load_persistence_failure_requires_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "state.json"))
    entered = asyncio.Event()

    class BlockingEngine(FakeEngine):
        async def start(self):
            entered.set()
            await asyncio.Event().wait()

    async def factory(spec):
        return BlockingEngine()

    manager = ResidencyManager(factory)
    manager.register_model(_spec())
    load = asyncio.create_task(manager.ensure_loaded("laguna"))
    await entered.wait()
    original_write = _fail_next_persisted_state(monkeypatch, "unloaded")
    load.cancel()

    with pytest.raises(OSError, match="cannot persist unloaded"):
        await load
    assert manager._residents["laguna"].state.value == "unloaded"
    assert manager._control.status()["snapshot"]["process_state"] == "loading"

    monkeypatch.setattr(lifecycle_control, "_write_json_atomic", original_write)
    assert (
        manager.get_status("laguna")["control"]["snapshot"]["process_state"]
        == "unloaded"
    )
    await manager.shutdown()


def test_invalid_persisted_state_fails_closed(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text("not-json")
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))

    with pytest.raises(LifecycleContractError, match="invalid lifecycle control state"):
        ResidencyManager(lambda spec: FakeEngine())


@pytest.mark.anyio
async def test_server_lazy_residency_exposes_persistent_control(tmp_path, monkeypatch):
    import vllm_mlx.server as srv

    async def factory(spec):
        return FakeEngine()

    monkeypatch.setenv(
        "VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "server-state.json")
    )
    monkeypatch.setattr(srv, "_engine_factory", factory)
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setattr(srv, "_residency_manager", None)
    monkeypatch.setattr(srv, "_default_model_key", None)
    monkeypatch.setattr(srv, "_lifespan_active", False)

    srv.load_model("org/laguna", lazy_load_model=True)
    engine = await srv._acquire_default_engine()
    assert isinstance(engine, FakeEngine)
    status = srv._get_lifecycle_status()
    assert status["control"]["snapshot"]["active_leases"] == 1

    await srv._release_default_engine()
    assert await srv._residency_manager.unload("default") is True
    await srv._residency_manager.shutdown()
