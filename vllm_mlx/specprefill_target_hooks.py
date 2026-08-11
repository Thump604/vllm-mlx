# SPDX-License-Identifier: Apache-2.0
"""Install-once, request-local target RoPE transport for SpecPrefill.

The old SpecPrefill executor temporarily replaced every target attention
module's ``rope`` object, then left a different wrapper installed for decode.
That makes a target model's positional semantics depend on whichever request
last entered prefill.  This module makes the one allowed model mutation at
installation time: each eligible RoPE class receives a stable dispatcher keyed
by its original instance. Every call thereafter is governed by a
:class:`contextvars.ContextVar` session owned by the request that is currently
forwarding the model.

The wrapper never writes cache offsets.  Cache offsets remain physical KV
coordinates for cache allocation and attention masks; the session supplies the
independent logical RoPE coordinates.  No active session means an exact
delegation to the original RoPE module.
"""

from __future__ import annotations

import contextvars
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import mlx.core as mx

from .specprefill_positions import (
    PositionPhase,
    TargetPositionAdapter,
    TargetPositionError,
    TargetPositionPlan,
)


class TargetPositionHookError(TargetPositionError):
    """Target position wrappers are missing, unsafe, or used inconsistently."""


@dataclass(frozen=True)
class TargetPositionSession:
    """Immutable logical/physical coordinates for one target forward phase.

    ``logical_positions`` are per-row coordinates for one bounded target
    forward quantum. ``physical_starts`` are the corresponding physical cache
    lengths before its first token in each row. The scheduler validates those
    host values while constructing the request/cache-local plan; each RoPE
    layer then reuses precomputed positions without reading device metadata.
    """

    logical_positions: tuple[tuple[int, ...], ...]
    physical_starts: tuple[int, ...]
    phase: PositionPhase
    position_tensor: mx.array = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.logical_positions:
            raise TargetPositionHookError(
                "target-position session needs at least one row"
            )
        if len(self.logical_positions) != len(self.physical_starts):
            raise TargetPositionHookError(
                "target-position logical rows and physical starts must match"
            )
        token_count: int | None = None
        for positions, start in zip(
            self.logical_positions, self.physical_starts, strict=True
        ):
            if not positions:
                raise TargetPositionHookError(
                    "target-position rows must include at least one position"
                )
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise TargetPositionHookError(
                    "target-position physical starts must be non-negative integers"
                )
            previous = -1
            for position in positions:
                if (
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or position < 0
                ):
                    raise TargetPositionHookError(
                        "target logical positions must be non-negative integers"
                    )
                if position <= previous:
                    raise TargetPositionHookError(
                        "target logical positions must be strictly increasing"
                    )
                previous = position
            if token_count is None:
                token_count = len(positions)
            elif len(positions) != token_count:
                raise TargetPositionHookError(
                    "target-position sessions require equal row token counts; "
                    "scheduler must split unequal rows into separate lanes"
                )
        # Materialize once per bounded target-forward quantum. Every installed
        # RoPE layer reuses this device tensor; do not recreate host positions
        # or synchronize cache metadata in a per-layer dispatcher.
        position_tensor = mx.array(self.logical_positions, dtype=mx.float32)
        mx.eval(position_tensor)
        object.__setattr__(self, "position_tensor", position_tensor)

    @property
    def batch_size(self) -> int:
        return len(self.logical_positions)

    @property
    def token_count(self) -> int:
        return len(self.logical_positions[0])

    @classmethod
    def from_plan(cls, plan: TargetPositionPlan) -> "TargetPositionSession":
        """Build an execution session from the cache-local position contract.

        The caller intentionally does not invoke ``plan.require_executable()``:
        that method describes pre-hook native target capability.  This hook is
        the separately tested transport that makes scalar-offset Qwen and Gemma
        sparse positions executable without mutating their cache metadata.
        """
        return cls(
            logical_positions=plan.logical_positions,
            physical_starts=plan.physical_cache_lengths,
            phase=plan.phase,
        )

    def positions_for_forward(self, sequence_length: int) -> mx.array:
        """Return precomputed ``(batch, sequence)`` coordinates for this quantum."""
        if sequence_length != self.token_count:
            raise TargetPositionHookError(
                "target-position session covers one forward quantum; create a new "
                "session for the next sparse-prefill chunk"
            )
        return self.position_tensor


@dataclass(frozen=True)
class _RoPEExecutionMetadata:
    """Immutable, device-ready RoPE data reused by every target layer call."""

    dims: int
    traditional: bool
    pre_scale: float
    inverse_frequencies: mx.array


@dataclass(frozen=True)
class _InstalledRope:
    """One original RoPE object registered with its hook-owning target model."""

    hooks_ref: weakref.ReferenceType["TargetPositionHooks"]
    layer_index: int
    metadata: _RoPEExecutionMetadata


@dataclass(frozen=True)
class _RoPEClassDispatch:
    """Original and installed class callables for topology/tamper verification."""

    original_call: Any
    dispatcher: Any


class _IdentityWeakRegistry:
    """Weak registry for MLX modules, which are intentionally unhashable."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}
        self._lock = threading.RLock()

    def get(self, value: Any) -> Any | None:
        with self._lock:
            key = id(value)
            entry = self._entries.get(key)
            if entry is None:
                return None
            reference, stored = entry
            if reference() is value:
                return stored
            self._entries.pop(key, None)
            return None

    def set(self, value: Any, stored: Any) -> None:
        with self._lock:
            key = id(value)

            def _cleanup(reference: weakref.ReferenceType[Any]) -> None:
                with self._lock:
                    entry = self._entries.get(key)
                    if entry is not None and entry[0] is reference:
                        self._entries.pop(key, None)

            try:
                reference = weakref.ref(value, _cleanup)
            except TypeError as exc:
                raise TargetPositionHookError(
                    "target position registry objects must support weak references"
                ) from exc
            self._entries[key] = (reference, stored)

    def pop(self, value: Any) -> None:
        with self._lock:
            key = id(value)
            entry = self._entries.get(key)
            if entry is not None and entry[0]() is value:
                self._entries.pop(key, None)


_ROPE_DISPATCH_LOCK = threading.RLock()
_ROPE_DISPATCHED_CLASSES: dict[type, _RoPEClassDispatch] = {}
_ROPE_HOOKS = _IdentityWeakRegistry()


def _install_rope_dispatch(
    rope: Any, hooks: "TargetPositionHooks", layer_index: int
) -> None:
    """Install one class dispatcher without replacing the rope child module.

    Assigning a plain proxy to ``attention.rope`` makes MLX remove the original
    child module and its state subtree.  Instead the original instance stays
    installed and its class receives one stable dispatcher.  A weak instance
    registry selects request-local behaviour only for the target model's RoPE
    objects; every other instance delegates to the original implementation.
    """
    # Capture the complete device-ready execution contract before changing the
    # class.  If introspection fails, this instance and every class topology
    # remain untouched.  Dispatch never re-reads mutable module attributes.
    metadata = _rope_execution_metadata(rope)
    rope_type = type(rope)
    with _ROPE_DISPATCH_LOCK:
        class_dispatch = _ROPE_DISPATCHED_CLASSES.get(rope_type)
        if class_dispatch is None:
            original_call = rope_type.__call__

            def _dispatch(self, *args, **kwargs):
                with _ROPE_DISPATCH_LOCK:
                    installed = _ROPE_HOOKS.get(self)
                if installed is None:
                    return original_call(self, *args, **kwargs)
                owner = installed.hooks_ref()
                if owner is None:
                    with _ROPE_DISPATCH_LOCK:
                        _ROPE_HOOKS.pop(self)
                    return original_call(self, *args, **kwargs)
                session = owner._active_session.get()
                if session is None:
                    return original_call(self, *args, **kwargs)
                if not args:
                    x = kwargs.get("x")
                else:
                    x = args[0]
                if x is None:
                    raise TargetPositionHookError(
                        "target RoPE call is missing its input"
                    )
                offset = kwargs.get("offset", args[1] if len(args) > 1 else 0)
                _validate_scalar_offset(offset, session)
                return _apply_request_positions(
                    x,
                    session.positions_for_forward(x.shape[2]),
                    installed.metadata,
                )

            try:
                rope_type.__call__ = _dispatch
            except (AttributeError, TypeError) as exc:
                raise TargetPositionHookError(
                    f"cannot install a stable dispatcher for RoPE class {rope_type}"
                ) from exc
            class_dispatch = _RoPEClassDispatch(original_call, _dispatch)
            _ROPE_DISPATCHED_CLASSES[rope_type] = class_dispatch
        elif rope_type.__call__ is not class_dispatch.dispatcher:
            raise TargetPositionHookError("target RoPE class dispatcher was modified")
        existing = _ROPE_HOOKS.get(rope)
        if existing is not None and existing.hooks_ref() is not hooks:
            raise TargetPositionHookError("target RoPE instance is already registered")
        _ROPE_HOOKS.set(
            rope,
            _InstalledRope(
                weakref.ref(hooks),
                layer_index,
                metadata,
            ),
        )


def _uninstall_rope_dispatch(rope: Any, hooks: "TargetPositionHooks") -> None:
    """Rollback a failed multi-layer installation without touching child topology."""
    with _ROPE_DISPATCH_LOCK:
        installed = _ROPE_HOOKS.get(rope)
        if installed is not None and installed.hooks_ref() is hooks:
            _ROPE_HOOKS.pop(rope)


class TargetPositionHooks:
    """Install-once target RoPE dispatch with concurrent request-local sessions."""

    def __init__(self, model: Any, adapter: TargetPositionAdapter):
        self._model_ref = weakref.ref(model)
        self.adapter = adapter
        self._active_session: contextvars.ContextVar[TargetPositionSession | None] = (
            contextvars.ContextVar(
                f"specprefill_target_positions_{id(self)}", default=None
            )
        )

        layers = _target_attention_layers(model)
        if not layers:
            raise TargetPositionHookError(
                "target model has no rope-bearing attention layers"
            )
        originals: list[tuple[int, Any, Any]] = []
        for layer_index, attention in layers:
            rope = getattr(attention, "rope", None)
            if rope is None:
                continue
            originals.append((layer_index, attention, rope))
        if not originals:
            raise TargetPositionHookError(
                "target model does not expose patchable `.rope` modules; "
                "use its explicit family position API instead"
            )

        installed: list[tuple[int, Any, Any]] = []
        try:
            for layer_index, attention, rope in originals:
                _install_rope_dispatch(rope, self, layer_index)
                installed.append((layer_index, attention, rope))
        except Exception:
            for _layer_index, _attention, rope in reversed(installed):
                _uninstall_rope_dispatch(rope, self)
            raise
        self._wrappers = tuple(installed)

    @classmethod
    def for_model(
        cls, model: Any, adapter: TargetPositionAdapter
    ) -> "TargetPositionHooks":
        """Return the sole hook set for ``model`` and verify stable topology."""
        with _TARGET_HOOK_REGISTRY_LOCK:
            try:
                hooks = _TARGET_HOOK_REGISTRY.get(model)
            except TypeError as exc:
                raise TargetPositionHookError(
                    "target models must support weak references for position hooks"
                ) from exc
            if hooks is None:
                hooks = cls(model, adapter)
                _TARGET_HOOK_REGISTRY.set(model, hooks)
            elif hooks.adapter != adapter:
                raise TargetPositionHookError(
                    "target model already has hooks for a different position adapter"
                )
            hooks._verify_installed()
            return hooks

    @property
    def active_session(self) -> TargetPositionSession | None:
        return self._active_session.get()

    def _verify_installed(self) -> None:
        model = self._model_ref()
        if model is None:
            raise TargetPositionHookError("target model is no longer available")
        for layer_index, attention, rope in self._wrappers:
            del layer_index
            if getattr(attention, "rope", None) is not rope:
                raise TargetPositionHookError(
                    "target RoPE wrapper topology was modified"
                )
            with _ROPE_DISPATCH_LOCK:
                installed = _ROPE_HOOKS.get(rope)
                class_dispatch = _ROPE_DISPATCHED_CLASSES.get(type(rope))
            if installed is None or installed.hooks_ref() is not self:
                raise TargetPositionHookError(
                    "target RoPE dispatcher registration was modified"
                )
            if (
                class_dispatch is None
                or type(rope).__call__ is not class_dispatch.dispatcher
            ):
                raise TargetPositionHookError(
                    "target RoPE class dispatcher was modified"
                )

    @contextmanager
    def session(
        self, session: TargetPositionSession
    ) -> Iterator[TargetPositionSession]:
        """Activate one request's positions for its complete target forward/eval.

        Sessions are context-local, so concurrent scheduler tasks can use the
        same installed model without one request's logical coordinates leaking
        into another.  Nested activation in the same context is rejected: a
        target forward has exactly one authoritative logical-position source.
        """
        if not isinstance(session, TargetPositionSession):
            raise TargetPositionHookError("target hooks require TargetPositionSession")
        self._verify_installed()
        if self._active_session.get() is not None:
            raise TargetPositionHookError(
                "target context already has an active session"
            )
        token = self._active_session.set(session)
        try:
            yield session
        finally:
            self._active_session.reset(token)

    @contextmanager
    def session_for_plan(
        self, plan: TargetPositionPlan
    ) -> Iterator[TargetPositionSession]:
        """Activate a cache-local plan for the caller's entire model forward."""
        with self.session(TargetPositionSession.from_plan(plan)) as session:
            yield session


_TARGET_HOOK_REGISTRY = _IdentityWeakRegistry()
_TARGET_HOOK_REGISTRY_LOCK = threading.RLock()


def _target_attention_layers(model: Any) -> list[tuple[int, Any]]:
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TargetPositionHookError("target model does not expose decoder layers")
    return [
        (index, layer.self_attn)
        for index, layer in enumerate(layers)
        if getattr(layer, "self_attn", None) is not None
    ]


def _validate_scalar_offset(offset: Any, session: TargetPositionSession) -> None:
    """Validate scalar offsets without synchronizing batched cache metadata.

    Batched caches may pass an MLX offset vector. The scheduler/cache-state
    transition admitted its physical starts before this session was created;
    the vector remains owned by the target's mask/cache code and cannot alter
    this dispatcher’s precomputed logical coordinates. Therefore the per-layer
    RoPE dispatcher must not call ``tolist``/``item``/``eval`` to rediscover it.
    Scalar Python offsets remain useful single-row guards.
    """
    if isinstance(offset, bool):
        raise TargetPositionHookError("target cache offset cannot be boolean")
    if isinstance(offset, int):
        if any(start != offset for start in session.physical_starts):
            raise TargetPositionHookError(
                "scalar target cache offset disagrees with request-local physical starts"
            )
    elif not isinstance(offset, mx.array):
        raise TargetPositionHookError(
            "target cache offset must be a Python int or an MLX array"
        )


def _apply_request_positions(
    x: mx.array,
    positions: mx.array,
    metadata: _RoPEExecutionMetadata,
) -> mx.array:
    """Apply a native RoPE-equivalent rotation at per-row logical positions."""
    if x.ndim != 4:
        raise TargetPositionHookError(
            "request-local target RoPE expects (batch, heads, sequence, dims)"
        )
    batch_size, _heads, sequence_length, head_dim = x.shape
    if positions.ndim != 2 or positions.shape != (batch_size, sequence_length):
        raise TargetPositionHookError(
            "request-local target RoPE positions must have host shape (batch, sequence)"
        )
    dims = metadata.dims
    if dims <= 0 or dims > head_dim or dims % 2:
        raise TargetPositionHookError(
            "target RoPE dimensions must be positive, even, and <= head size"
        )

    angles = (
        positions[:, None, :, None] * metadata.inverse_frequencies[None, None, None, :]
    )
    cos_a = mx.cos(angles).astype(x.dtype)
    sin_a = mx.sin(angles).astype(x.dtype)
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    if metadata.pre_scale != 1.0:
        x_rot = x_rot * metadata.pre_scale
    if metadata.traditional:
        paired = x_rot.reshape(*x_rot.shape[:-1], dims // 2, 2)
        first, second = paired[..., 0], paired[..., 1]
        rotated = mx.stack(
            (first * cos_a - second * sin_a, first * sin_a + second * cos_a), axis=-1
        ).reshape(*x_rot.shape)
    else:
        first, second = x_rot[..., : dims // 2], x_rot[..., dims // 2 :]
        rotated = mx.concatenate(
            (first * cos_a - second * sin_a, first * sin_a + second * cos_a), axis=-1
        )
    return mx.concatenate((rotated, x_pass), axis=-1)


def _rope_execution_metadata(rope: Any) -> _RoPEExecutionMetadata:
    """Capture/evaluate immutable RoPE settings once at hook installation."""
    dims = _rope_dimensions(rope)
    inverse_frequencies = _inverse_frequencies(rope, dims)
    mx.eval(inverse_frequencies)
    return _RoPEExecutionMetadata(
        dims=dims,
        traditional=_rope_traditional(rope),
        pre_scale=_rope_pre_scale(rope),
        inverse_frequencies=inverse_frequencies,
    )


def _rope_dimensions(rope: Any) -> int:
    for name in ("dims", "_dims", "dim"):
        value = getattr(rope, name, None)
        if value is not None:
            return int(value)
    raise TargetPositionHookError(f"cannot determine RoPE dimensions from {type(rope)}")


def _rope_traditional(rope: Any) -> bool:
    for name in ("traditional", "_traditional"):
        value = getattr(rope, name, None)
        if value is not None:
            return bool(value)
    return False


def _inverse_frequencies(rope: Any, dims: int) -> mx.array:
    """Use model-provided scaled frequencies when available, else native params."""
    freqs = getattr(rope, "_freqs", None)
    if freqs is None:
        freqs = getattr(rope, "freqs", None)
    if freqs is not None:
        freqs = mx.array(freqs).astype(mx.float32)
        if freqs.size != dims // 2:
            raise TargetPositionHookError(
                "target RoPE frequency count does not match dims"
            )
        return 1.0 / freqs
    base = getattr(rope, "base", getattr(rope, "_base", None))
    if base is None:
        raise TargetPositionHookError(
            "target RoPE has no public frequencies or base for request-local positions"
        )
    scale = getattr(rope, "scale", 1.0)
    if not isinstance(scale, (float, int)) or float(scale) <= 0:
        raise TargetPositionHookError("target RoPE scale must be a positive scalar")
    indices = mx.arange(0, dims, 2, dtype=mx.float32)
    return 1.0 / (float(base) ** (indices / dims)) / float(scale)


def _rope_pre_scale(rope: Any) -> float:
    """Match MLX custom RoPE amplitude scaling for scaled-frequency variants."""
    mscale = getattr(rope, "mscale", None)
    if mscale is not None:
        if not isinstance(mscale, (float, int)) or float(mscale) <= 0:
            raise TargetPositionHookError(
                "target RoPE mscale must be a positive scalar"
            )
        return float(mscale)
    # MLX SuScaled/Yarn-style modules expose `_scale` alongside `dim`; plain
    # nn.RoPE instead exposes public `scale` and must not be treated here.
    scale = getattr(rope, "_scale", None)
    if scale is not None and getattr(rope, "dim", None) is not None:
        if not isinstance(scale, (float, int)) or float(scale) <= 0:
            raise TargetPositionHookError("target RoPE pre-scale must be positive")
        return float(scale)
    return 1.0
