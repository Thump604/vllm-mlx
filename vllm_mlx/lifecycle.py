# SPDX-License-Identifier: Apache-2.0
"""Model lifecycle / residency management for vllm-mlx."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from .engine.base import BaseEngine, suspend_cancellation
from .lifecycle_contract import LifecycleEvent, ProcessState
from .lifecycle_control import LifecycleControlState
from .model_profile import compute_subject_digest


class ResidentState(str, Enum):
    """Runtime residency state for a configured model."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable engine construction inputs for a resident model."""

    model_key: str
    model_name: str
    use_batching: bool = False
    scheduler_config: Any | None = None
    stream_interval: int = 1
    max_tokens: int = 32768
    force_mllm: bool = False
    mtp: bool = False
    prefill_step_size: int = 2048
    specprefill_enabled: bool = False
    specprefill_threshold: int = 8192
    specprefill_keep_pct: float = 0.3
    specprefill_backbone_pct: float = 0.0
    specprefill_draft_model: str | None = None
    profile_id: str | None = None
    profile_revision: str | None = None
    config_digest: str | None = None


def bind_model_spec_to_profile(
    spec: ModelSpec, profile: Mapping[str, Any]
) -> ModelSpec:
    """Bind a resolved serving spec to one immutable ModelProfile subject."""
    profile_id = profile.get("profile_id")
    profile_revision = profile.get("profile_revision")
    stored_digest = profile.get("subject_digest")
    computed_digest = str(compute_subject_digest(profile))
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("model profile_id is missing")
    if not isinstance(profile_revision, (str, int)):
        raise ValueError("model profile_revision is missing")
    if not isinstance(stored_digest, str) or stored_digest.lower() != computed_digest:
        raise ValueError("model profile subject_digest is missing or stale")
    for field_name, existing, expected in (
        ("profile_id", spec.profile_id, profile_id),
        ("profile_revision", spec.profile_revision, str(profile_revision)),
        ("config_digest", spec.config_digest, computed_digest),
    ):
        if existing is not None and existing != expected:
            raise ValueError(f"model spec {field_name} conflicts with profile")
    return replace(
        spec,
        profile_id=profile_id,
        profile_revision=str(profile_revision),
        config_digest=computed_digest,
    )


@dataclass
class ResidentModel:
    """Runtime state for a single resident model."""

    spec: ModelSpec
    state: ResidentState = ResidentState.UNLOADED
    engine: BaseEngine | None = None
    active_requests: int = 0
    last_used_at: float | None = None
    loaded_at: float | None = None
    last_error: str | None = None
    estimated_memory_bytes: int | None = None
    _load_waiters: int = field(default=0, repr=False)
    _load_waiter_task: asyncio.Task[BaseEngine] | None = field(default=None, repr=False)
    _prepare_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _abandoned_loading_task: asyncio.Task[BaseEngine] | None = field(
        default=None, repr=False
    )
    _loading_task: asyncio.Task[BaseEngine] | None = field(default=None, repr=False)
    _unloading_task: asyncio.Task[bool] | None = field(default=None, repr=False)


class ResidencyManager:
    """Single-flight lifecycle manager for resident models."""

    def __init__(
        self,
        engine_factory: Callable[[ModelSpec], Awaitable[BaseEngine]],
        *,
        on_engine_loaded: (
            Callable[[ModelSpec, BaseEngine], Awaitable[None] | None] | None
        ) = None,
        on_engine_unloading: (
            Callable[[ModelSpec, BaseEngine], Awaitable[None] | None] | None
        ) = None,
        time_fn: Callable[[], float] | None = None,
        auto_unload_idle_seconds: float = 0,
    ) -> None:
        self._engine_factory = engine_factory
        self._on_engine_loaded = on_engine_loaded
        self._on_engine_unloading = on_engine_unloading
        self._time_fn = time_fn or __import__("time").time
        self.auto_unload_idle_seconds = auto_unload_idle_seconds
        self._residents: dict[str, ResidentModel] = {}
        self._lock = asyncio.Lock()
        self._shutting_down = False
        self._control_sync_error: str | None = None
        state_path = os.environ.get("VLLM_MLX_LIFECYCLE_STATE_PATH")
        self._control = LifecycleControlState(state_path) if state_path else None
        self._persistent_control_enabled = self._control is not None

    def register_model(self, spec: ModelSpec) -> str:
        """Register a model spec, or replace a dormant resident entry."""
        if self._control is not None and self._control.model_key not in {
            None,
            spec.model_key,
        }:
            raise RuntimeError(
                "Persistent lifecycle control permits one configured resident model"
            )
        existing = self._residents.get(spec.model_key)
        if existing is not None:
            is_dormant = (
                existing.engine is None
                and existing.active_requests == 0
                and existing._load_waiters == 0
                and existing._loading_task is None
                and existing._unloading_task is None
                and existing.state in {ResidentState.UNLOADED, ResidentState.FAILED}
            )
            if not is_dormant:
                raise RuntimeError(
                    f"Cannot replace resident model '{spec.model_key}' while it is live"
                )

        self._residents[spec.model_key] = ResidentModel(spec=spec)
        if self._control is not None:
            self._control.configure(spec)
            self._control.reconcile(ProcessState.UNLOADED, 0)
        return spec.model_key

    def get_engine(self, model_key: str) -> BaseEngine | None:
        """Get the currently loaded engine, if any."""
        return self._resident(model_key).engine

    def get_status(self, model_key: str) -> dict[str, Any]:
        """Return a serializable snapshot of resident state."""
        resident = self._resident(model_key)
        if self._control is not None and self._control.model_key == model_key:
            self._reconcile_control(resident)
        status: dict[str, Any] = {
            "model_key": resident.spec.model_key,
            "model_name": resident.spec.model_name,
            "state": resident.state.value,
            "active_requests": resident.active_requests,
            "last_used_at": resident.last_used_at,
            "loaded_at": resident.loaded_at,
            "last_error": resident.last_error,
            "estimated_memory_bytes": resident.estimated_memory_bytes,
            "auto_unload_idle_seconds": self.auto_unload_idle_seconds,
        }
        if self._control is not None and self._control.model_key == model_key:
            status["control"] = self._control.status()
            status["control_sync_error"] = self._control_sync_error
        return status

    async def ensure_loaded(self, model_key: str) -> BaseEngine:
        """Load and start a resident engine if needed."""
        while True:
            task: asyncio.Task[BaseEngine] | None = None
            unloading_task: asyncio.Task[bool] | None = None

            async with self._lock:
                if self._shutting_down:
                    raise RuntimeError("Residency manager is shutting down")
                resident = self._resident(model_key)
                self._reconcile_control(resident)
                if (
                    resident.state == ResidentState.LOADED
                    and resident.engine is not None
                ):
                    return resident.engine

                if resident._unloading_task is not None:
                    unloading_task = resident._unloading_task
                else:
                    if resident._loading_task is None:
                        previous_state = resident.state
                        previous_error = resident.last_error
                        resident.state = ResidentState.LOADING
                        resident.last_error = None
                        try:
                            self._apply_control(LifecycleEvent.BEGIN_LOAD, model_key)
                        except BaseException:
                            resident.state = previous_state
                            resident.last_error = previous_error
                            raise
                        resident._loading_task = asyncio.create_task(
                            self._load_engine(resident)
                        )
                        resident._load_waiters = 0
                        resident._load_waiter_task = resident._loading_task
                        resident._abandoned_loading_task = None
                    task = resident._loading_task
                    resident._load_waiters += 1
                    resident._load_waiter_task = task

            if unloading_task is not None:
                await asyncio.shield(unloading_task)
                continue

            if task is None:
                raise RuntimeError(f"No load task available for resident {model_key}")
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                cancelling = getattr(current_task, "cancelling", None)
                if (
                    task.done()
                    and task.cancelled()
                    and (cancelling is None or cancelling() == 0)
                ):
                    async with self._lock:
                        resident = self._resident(model_key)
                        if resident._abandoned_loading_task is task:
                            continue
                raise
            finally:
                await self._release_load_waiter(model_key, task)

    async def acquire(
        self,
        model_key: str,
        *,
        count_activity: bool = True,
    ) -> BaseEngine:
        """Acquire a resident engine for request processing."""
        while True:
            engine = await self.ensure_loaded(model_key)
            async with self._lock:
                if self._shutting_down:
                    raise RuntimeError("Residency manager is shutting down")
                resident = self._resident(model_key)
                self._reconcile_control(resident)
                if (
                    resident.engine is not engine
                    or resident.state != ResidentState.LOADED
                    or resident._unloading_task is not None
                ):
                    continue
                resident.active_requests += 1
                try:
                    self._apply_control(LifecycleEvent.ACQUIRE, model_key)
                except BaseException:
                    resident.active_requests -= 1
                    raise
                if count_activity:
                    resident.last_used_at = self._time_fn()
                return engine

    async def release(self, model_key: str, *, count_activity: bool = True) -> None:
        """Release a previously acquired resident engine."""
        async with self._lock:
            resident = self._resident(model_key)
            self._reconcile_control(resident)
            if resident.active_requests > 0:
                self._apply_control(LifecycleEvent.RELEASE, model_key)
                resident.active_requests -= 1
            if count_activity:
                resident.last_used_at = self._time_fn()

    async def unload_if_idle(self, model_key: str) -> bool:
        """Unload a resident engine if it has been idle past the threshold."""
        if self.auto_unload_idle_seconds <= 0:
            return False

        while True:
            unloading_task: asyncio.Task[bool] | None = None
            async with self._lock:
                resident = self._resident(model_key)
                self._reconcile_control(resident)

                if resident._loading_task is not None:
                    return False

                if resident._unloading_task is not None:
                    unloading_task = resident._unloading_task
                else:
                    if (
                        resident.state != ResidentState.LOADED
                        or resident.engine is None
                        or resident.active_requests > 0
                        or resident.last_used_at is None
                    ):
                        return False

                    idle_for = self._time_fn() - resident.last_used_at
                    if idle_for < self.auto_unload_idle_seconds:
                        return False

                    resident.state = ResidentState.UNLOADING
                    try:
                        self._apply_control(LifecycleEvent.BEGIN_UNLOAD, model_key)
                    except BaseException:
                        resident.state = ResidentState.LOADED
                        raise
                    resident._unloading_task = asyncio.create_task(
                        self._unload_engine(resident)
                    )
                    unloading_task = resident._unloading_task

            if unloading_task is None:
                return False
            return await asyncio.shield(unloading_task)

    async def unload(self, model_key: str) -> bool:
        """Explicitly unload one idle resident without applying idle-time policy."""
        while True:
            loading_task: asyncio.Task[BaseEngine] | None = None
            unloading_task: asyncio.Task[bool] | None = None
            async with self._lock:
                resident = self._resident(model_key)
                self._reconcile_control(resident)
                if resident.active_requests:
                    raise RuntimeError(
                        f"Cannot unload resident model '{model_key}' with active requests"
                    )
                if resident._loading_task is not None:
                    resident._loading_task.cancel()
                    loading_task = resident._loading_task
                elif resident._unloading_task is not None:
                    unloading_task = resident._unloading_task
                elif (
                    resident.engine is None or resident.state == ResidentState.UNLOADED
                ):
                    resident.state = ResidentState.UNLOADED
                    return False
                else:
                    resident.state = ResidentState.UNLOADING
                    try:
                        self._apply_control(LifecycleEvent.BEGIN_UNLOAD, model_key)
                    except BaseException:
                        resident.state = ResidentState.LOADED
                        raise
                    resident._unloading_task = asyncio.create_task(
                        self._unload_engine(resident)
                    )
                    unloading_task = resident._unloading_task

            if loading_task is not None:
                with suppress(asyncio.CancelledError):
                    await loading_task
                continue
            if unloading_task is not None:
                return await asyncio.shield(unloading_task)

    def list_status(self) -> list[dict[str, Any]]:
        """Return snapshots for every registered resident."""
        return [self.get_status(model_key) for model_key in sorted(self._residents)]

    async def shutdown(self) -> None:
        """Stop all loaded residents."""
        async with self._lock:
            if self._shutting_down:
                raise RuntimeError("Residency manager is already shutting down")
            self._shutting_down = True

        failures: list[str] = []
        try:
            for model_key in list(self._residents):
                while True:
                    loading_task: asyncio.Task[BaseEngine] | None = None
                    unloading_task: asyncio.Task[bool] | None = None

                    async with self._lock:
                        resident = self._resident(model_key)
                        self._reconcile_control(resident)
                        if resident._loading_task is not None:
                            resident._loading_task.cancel()
                            loading_task = resident._loading_task
                        elif (
                            resident.engine is None
                            or resident.state == ResidentState.UNLOADED
                        ):
                            break
                        else:
                            if resident._unloading_task is None:
                                resident.state = ResidentState.UNLOADING
                                try:
                                    self._apply_control(
                                        LifecycleEvent.BEGIN_UNLOAD, model_key
                                    )
                                except BaseException:
                                    resident.state = ResidentState.LOADED
                                    raise
                                resident._unloading_task = asyncio.create_task(
                                    self._unload_engine(resident)
                                )
                            unloading_task = resident._unloading_task

                    if loading_task is not None:
                        try:
                            await asyncio.shield(loading_task)
                        except asyncio.CancelledError:
                            current = asyncio.current_task()
                            cancelling = getattr(current, "cancelling", None)
                            if cancelling is not None and cancelling() > 0:
                                with suspend_cancellation():
                                    with suppress(asyncio.CancelledError):
                                        await loading_task
                                self._close_control_if_fully_unloaded()
                                raise
                        continue

                    if unloading_task is not None:
                        try:
                            unloaded = await asyncio.shield(unloading_task)
                        except asyncio.CancelledError:
                            with suspend_cancellation():
                                unloaded = await unloading_task
                            if unloaded:
                                self._close_control_if_fully_unloaded()
                            raise
                        if not unloaded:
                            async with self._lock:
                                resident = self._resident(model_key)
                                error = (
                                    resident.last_error or "resident remained loaded"
                                )
                            failures.append(
                                f"Failed to unload resident model '{model_key}' during shutdown: {error}"
                            )
                            break
                        break

            if failures:
                if len(failures) == 1:
                    raise RuntimeError(failures[0])
                raise RuntimeError("; ".join(failures))
            if not self._close_control_if_fully_unloaded():
                raise RuntimeError("Residency manager did not fully unload")
        except BaseException:
            if not self._persistent_control_enabled or (
                self._control is not None and self._control.is_open
            ):
                self._shutting_down = False
            raise

    def _close_control_if_fully_unloaded(self) -> bool:
        fully_unloaded = all(
            resident.engine is None
            and resident.state == ResidentState.UNLOADED
            and resident._loading_task is None
            and resident._unloading_task is None
            for resident in self._residents.values()
        )
        if fully_unloaded and self._control is not None:
            self._control.close()
            self._control = None
        return fully_unloaded

    async def _load_engine(self, resident: ResidentModel) -> BaseEngine:
        """Create and start a resident engine."""
        engine: BaseEngine | None = None
        try:
            engine = await self._engine_factory(resident.spec)
            await self._prepare_engine_start(resident, engine)
            await engine.start()
            await self._run_hook(self._on_engine_loaded, resident.spec, engine)
        except asyncio.CancelledError:
            await self._cleanup_cancelled_load(resident, engine)
            raise
        except Exception as exc:
            async with self._lock:
                abandoned = resident._abandoned_loading_task is asyncio.current_task()
            if abandoned:
                await self._cleanup_cancelled_load(resident, engine)
                raise asyncio.CancelledError() from exc
            if engine is not None:
                with suppress(Exception):
                    await engine.stop()
            async with self._lock:
                resident.state = ResidentState.FAILED
                resident.last_error = str(exc)
                resident._abandoned_loading_task = None
                resident._loading_task = None
                self._apply_terminal_control(
                    LifecycleEvent.LOAD_FAILED, resident.spec.model_key
                )
            raise

        try:
            async with self._lock:
                resident.engine = engine
                resident.state = ResidentState.LOADED
                resident.loaded_at = self._time_fn()
                resident.last_used_at = resident.loaded_at
                resident.last_error = None
                resident._abandoned_loading_task = None
                resident._loading_task = None
                self._apply_terminal_control(
                    LifecycleEvent.LOAD_SUCCEEDED, resident.spec.model_key
                )
        except asyncio.CancelledError:
            await self._cleanup_cancelled_load(resident, engine)
            raise

        return engine

    async def _unload_engine(self, resident: ResidentModel) -> bool:
        """Stop and drop a resident engine."""
        engine = resident.engine
        if engine is None:
            async with self._lock:
                resident.state = ResidentState.UNLOADED
                resident._unloading_task = None
            return False

        try:
            await self._run_hook(self._on_engine_unloading, resident.spec, engine)
            await engine.stop()
        except asyncio.CancelledError:
            async with self._lock:
                resident.state = ResidentState.LOADED
                resident._unloading_task = None
                self._apply_terminal_control(
                    LifecycleEvent.UNLOAD_FAILED, resident.spec.model_key
                )
            raise
        except Exception as exc:
            async with self._lock:
                resident.engine = engine
                resident.state = ResidentState.LOADED
                resident.last_error = str(exc)
                resident._unloading_task = None
                self._apply_terminal_control(
                    LifecycleEvent.UNLOAD_FAILED, resident.spec.model_key
                )
            return False

        async with self._lock:
            resident.engine = None
            resident.state = ResidentState.UNLOADED
            resident.loaded_at = None
            resident.last_error = None
            resident._unloading_task = None
            self._apply_terminal_control(
                LifecycleEvent.UNLOAD_SUCCEEDED, resident.spec.model_key
            )

        return True

    def _resident(self, model_key: str) -> ResidentModel:
        try:
            return self._residents[model_key]
        except KeyError as exc:
            raise KeyError(f"Resident model '{model_key}' is not registered") from exc

    async def _run_hook(
        self,
        hook: Callable[[ModelSpec, BaseEngine], Awaitable[None] | None] | None,
        spec: ModelSpec,
        engine: BaseEngine,
    ) -> None:
        if hook is None:
            return

        result = hook(spec, engine)
        if inspect.isawaitable(result):
            await result

    async def _prepare_engine_start(
        self,
        resident: ResidentModel,
        engine: BaseEngine,
    ) -> None:
        """Run blocking startup work away from the serving event loop."""
        prepare_for_start = getattr(engine, "prepare_for_start", None)
        if prepare_for_start is None:
            return

        uses_default_prepare = getattr(engine, "_uses_default_prepare_for_start", None)
        if callable(uses_default_prepare) and uses_default_prepare():
            # Keep default engine prepare on the event-loop thread so MLX
            # thread-local stream ownership matches subsequent streaming calls.
            prepare_for_start()
            return

        prepare_task = asyncio.create_task(asyncio.to_thread(prepare_for_start))
        async with self._lock:
            resident._prepare_task = prepare_task

        try:
            await asyncio.shield(prepare_task)
        except asyncio.CancelledError:
            with suspend_cancellation():
                while not prepare_task.done():
                    try:
                        await asyncio.shield(prepare_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
            raise
        finally:
            async with self._lock:
                if resident._prepare_task is prepare_task:
                    resident._prepare_task = None

    async def _cleanup_cancelled_load(
        self,
        resident: ResidentModel,
        engine: BaseEngine | None,
    ) -> None:
        """Stop a partially loaded engine and unwind resident state."""
        with suspend_cancellation():
            if engine is not None:
                with suppress(Exception):
                    await engine.stop()
            async with self._lock:
                resident.engine = None
                resident.state = ResidentState.UNLOADED
                resident.loaded_at = None
                resident.last_error = None
                # Keep the abandoned-load marker until a new load task replaces it
                # so late waiters on the old task can still recognize a retryable
                # cancellation instead of inheriting CancelledError.
                resident._loading_task = None
                self._apply_terminal_control(
                    LifecycleEvent.LOAD_CANCELLED, resident.spec.model_key
                )

    def _apply_control(self, event: LifecycleEvent, model_key: str) -> None:
        if self._control is not None and self._control.model_key == model_key:
            self._control.apply(event)

    def _apply_terminal_control(self, event: LifecycleEvent, model_key: str) -> None:
        try:
            self._apply_control(event, model_key)
            self._control_sync_error = None
        except BaseException as exc:
            self._control_sync_error = str(exc)
            raise

    def _reconcile_control(self, resident: ResidentModel) -> None:
        if self._control is None or self._control.model_key != resident.spec.model_key:
            return
        try:
            self._control.reconcile(
                ProcessState(resident.state.value), resident.active_requests
            )
            self._control_sync_error = None
        except BaseException as exc:
            self._control_sync_error = str(exc)
            raise

    async def _release_load_waiter(
        self,
        model_key: str,
        task: asyncio.Task[BaseEngine],
    ) -> None:
        """Drop one waiter from a shared load, canceling abandoned solo loads."""
        task_to_cancel: asyncio.Task[BaseEngine] | None = None

        async with self._lock:
            resident = self._resident(model_key)
            if resident._load_waiter_task is not task or resident._load_waiters <= 0:
                return

            resident._load_waiters -= 1
            if resident._load_waiters == 0:
                resident._load_waiter_task = None
                if resident._loading_task is task and not task.done():
                    resident._abandoned_loading_task = task
                    task_to_cancel = task

        if task_to_cancel is None:
            return

        with suspend_cancellation():
            task_to_cancel.cancel()
            with suppress(asyncio.CancelledError):
                await task_to_cancel
