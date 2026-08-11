# SPDX-License-Identifier: Apache-2.0
"""Transaction foundation for native-Qwen continuous batching.

This module owns cache and request state only.  It deliberately does not
install a scheduler hook or execute model forwards.  A future continuous-
batching consumer must drive the explicit checkpoint, seal, commit, rollback,
and sequential-replay boundaries defined here.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

from .native_mtp_request import NativeMTPRequestConfig


class NativeMTPBatchError(RuntimeError):
    """A native-MTP batch transaction cannot continue safely."""


@dataclass(frozen=True, slots=True)
class NativeMTPBatchPosition:
    """Immutable logical and physical cursors for one UID."""

    logical_cursor: int
    backbone_tokens: int
    mtp_tokens: int

    def __post_init__(self) -> None:
        values = (self.logical_cursor, self.backbone_tokens, self.mtp_tokens)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("native MTP batch positions must be integers")
        if any(value < 0 for value in values):
            raise ValueError("native MTP batch positions must be non-negative")


@dataclass(frozen=True, slots=True)
class NativeMTPBatchRow:
    """Fresh B=1 request ownership admitted into one native-MTP batch."""

    uid: int
    target_model: Any
    capability: Any
    backbone_cache: list[Any]
    mtp_cache: list[Any]
    position: NativeMTPBatchPosition
    request_config: NativeMTPRequestConfig
    prefix_reused: bool = False
    has_media: bool = False
    specprefill_selected: bool = False
    external_assistant_selected: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.uid, bool) or not isinstance(self.uid, int) or self.uid < 0:
            raise ValueError("native MTP batch UID must be a non-negative integer")
        if not isinstance(self.position, NativeMTPBatchPosition):
            raise TypeError("native MTP batch row requires NativeMTPBatchPosition")
        if not isinstance(self.request_config, NativeMTPRequestConfig):
            raise TypeError("native MTP batch row requires NativeMTPRequestConfig")
        if not isinstance(self.backbone_cache, list) or not isinstance(
            self.mtp_cache, list
        ):
            raise TypeError("native MTP batch caches must be caller-owned lists")
        if self.backbone_cache is self.mtp_cache:
            raise ValueError("native MTP backbone and head caches must be distinct")
        incompatibilities = (
            (self.prefix_reused, "native_mtp_prefix_reuse_unsupported"),
            (self.has_media, "native_mtp_media_unsupported"),
            (
                self.specprefill_selected,
                "native_mtp_specprefill_composition_unsupported",
            ),
            (
                self.external_assistant_selected,
                "native_mtp_external_assistant_conflict",
            ),
        )
        for selected, reason in incompatibilities:
            if selected:
                raise NativeMTPBatchError(reason)


@dataclass(frozen=True, slots=True)
class NativeMTPReplayPlan:
    """Rollback result grouped by equal sequential replay work."""

    targets: tuple[tuple[int, NativeMTPBatchPosition], ...]
    groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class NativeMTPCapabilityFingerprint:
    """Stable value identity for a loader-produced capability snapshot."""

    supported: bool
    reason: str
    num_layers: int


def _capability_fingerprint(capability: Any) -> NativeMTPCapabilityFingerprint:
    supported = getattr(capability, "supported", None)
    reason = getattr(capability, "reason", None)
    num_layers = getattr(capability, "num_layers", None)
    if (
        not isinstance(supported, bool)
        or not isinstance(reason, str)
        or isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers < 0
    ):
        raise NativeMTPBatchError("native_mtp_capability_invalid")
    return NativeMTPCapabilityFingerprint(supported, reason, num_layers)


@dataclass(slots=True)
class _RowRuntime:
    row: NativeMTPBatchRow
    position: NativeMTPBatchPosition
    rng_key: Any
    replay_target: NativeMTPBatchPosition | None = None


@dataclass(frozen=True, slots=True)
class _AttentionCheckpoint:
    index: int
    entry: BatchKVCache
    idx: int
    offset: Any
    left_padding: Any


@dataclass(frozen=True, slots=True)
class _ArraysCheckpoint:
    index: int
    entry: ArraysCache
    state: tuple[Any, ...]
    left_padding: Any
    lengths: Any


@dataclass(frozen=True, slots=True)
class _CacheCheckpoint:
    attention: tuple[_AttentionCheckpoint, ...]
    arrays: tuple[_ArraysCheckpoint, ...]


@dataclass(frozen=True, slots=True)
class _BatchCheckpoint:
    positions: tuple[tuple[int, NativeMTPBatchPosition], ...]
    backbone: _CacheCheckpoint
    mtp: _CacheCheckpoint


def _copy_small_array(value: Any) -> Any:
    if value is None:
        return None
    return value + mx.zeros(value.shape, dtype=value.dtype)


def _iter_arrays(value: Any) -> Iterable[Any]:
    if value is None:
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_arrays(item)
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        yield value


def _force_eval(values: Iterable[Any]) -> None:
    arrays = list(_iter_arrays(tuple(values)))
    if arrays:
        mx.eval(arrays)


def _cache_eval_values(entry: Any) -> tuple[Any, ...]:
    """Return realized cache values without dereferencing empty BatchKV state."""

    if type(entry) is BatchKVCache:
        return (
            entry.keys,
            entry.values,
            entry.offset,
            entry.left_padding,
        )
    if type(entry) is ArraysCache:
        return (*entry.cache, entry.left_padding, entry.lengths)
    raise NativeMTPBatchError("native_mtp_batch_cache_type_mismatch")


def _model_layers(model: Any) -> tuple[Any, ...]:
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise NativeMTPBatchError("native_mtp_cache_topology_unavailable")
    return tuple(layers)


def _mtp_layers(model: Any) -> tuple[Any, ...]:
    layers = getattr(getattr(model, "mtp", None), "layers", None)
    if layers is None:
        raise NativeMTPBatchError("native_mtp_cache_topology_unavailable")
    return tuple(layers)


class NativeMTPBatch:
    """Request-owned native-Qwen cache state for a true batched consumer."""

    def __init__(self, rows: Sequence[NativeMTPBatchRow]):
        if not rows:
            raise ValueError("native MTP batch requires at least one row")
        self._closed = False
        self._poisoned = False
        self.alignment_checks = 0
        self.alignment_host_syncs = 0
        self.alignment_check_ms = 0.0
        self._checkpoint: _BatchCheckpoint | None = None
        self._sealed: tuple[tuple[int, NativeMTPBatchPosition], ...] | None = None
        self._rows = self._validate_rows(rows)
        self.target_model = next(iter(self._rows.values())).row.target_model
        self.capability = next(iter(self._rows.values())).row.capability
        self.capability_fingerprint = _capability_fingerprint(self.capability)
        self._topology = self._topology_for(self.target_model)
        self.backbone_cache, self.mtp_cache = self._merge_admission_rows()
        self._assert_aligned()

    @property
    def uids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def checkpoint_active(self) -> bool:
        return self._checkpoint is not None

    @property
    def positions(self) -> tuple[tuple[int, NativeMTPBatchPosition], ...]:
        return tuple((uid, runtime.position) for uid, runtime in self._rows.items())

    @property
    def replay_targets(self) -> tuple[tuple[int, NativeMTPBatchPosition], ...]:
        return tuple(
            (uid, runtime.replay_target)
            for uid, runtime in self._rows.items()
            if runtime.replay_target is not None
        )

    @staticmethod
    def _topology_for(model: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        backbone = tuple(
            "arrays" if bool(getattr(layer, "is_linear", False)) else "kv"
            for layer in _model_layers(model)
        )
        mtp = tuple("kv" for _ in _mtp_layers(model))
        return backbone, mtp

    @classmethod
    def _validate_rows(
        cls, rows: Sequence[NativeMTPBatchRow]
    ) -> OrderedDict[int, _RowRuntime]:
        if any(not isinstance(row, NativeMTPBatchRow) for row in rows):
            raise TypeError("native MTP batch requires NativeMTPBatchRow entries")
        uids = tuple(row.uid for row in rows)
        if len(set(uids)) != len(uids):
            raise ValueError("native MTP batch UIDs must be unique")
        model = rows[0].target_model
        capability = rows[0].capability
        capability_fingerprint = _capability_fingerprint(capability)
        current_capability = getattr(model, "mtp_capability", None)
        if (
            current_capability != capability
            or _capability_fingerprint(current_capability) != capability_fingerprint
        ):
            raise NativeMTPBatchError("native_mtp_capability_identity_mismatch")
        if not capability.supported:
            reason = getattr(capability, "reason", None)
            raise NativeMTPBatchError(reason or "native_mtp_unsupported")
        if (
            getattr(model, "pipeline_size", 1) != 1
            or getattr(getattr(model, "model", None), "pipeline_size", 1) != 1
        ):
            raise NativeMTPBatchError("native_mtp_pipeline_parallelism_unsupported")

        topology = cls._topology_for(model)
        entry_ids: list[int] = []
        result: OrderedDict[int, _RowRuntime] = OrderedDict()
        for row in rows:
            if row.target_model is not model:
                raise NativeMTPBatchError("native_mtp_target_identity_mismatch")
            if (
                row.capability != capability
                or _capability_fingerprint(row.capability) != capability_fingerprint
            ):
                raise NativeMTPBatchError("native_mtp_capability_identity_mismatch")
            current_capability = getattr(row.target_model, "mtp_capability", None)
            if (
                current_capability != row.capability
                or _capability_fingerprint(current_capability) != capability_fingerprint
            ):
                raise NativeMTPBatchError("native_mtp_capability_identity_mismatch")
            cls._validate_row_cache(row, topology)
            entry_ids.extend(id(entry) for entry in row.backbone_cache)
            entry_ids.extend(id(entry) for entry in row.mtp_cache)
            seed = row.request_config.sampling.seed
            if seed is None:
                seed = secrets.randbits(63)
            result[row.uid] = _RowRuntime(
                row=row,
                position=row.position,
                rng_key=mx.random.key(seed),
            )
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("native_mtp_cache_entries_must_be_unique")
        return result

    @staticmethod
    def _validate_row_cache(
        row: NativeMTPBatchRow,
        topology: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> None:
        backbone_topology, mtp_topology = topology
        if len(row.backbone_cache) != len(backbone_topology):
            raise ValueError("native_mtp_backbone_cache_topology_mismatch")
        if len(row.mtp_cache) != len(mtp_topology):
            raise ValueError("native_mtp_head_cache_topology_mismatch")
        for index, (kind, entry) in enumerate(
            zip(backbone_topology, row.backbone_cache)
        ):
            expected = ArraysCache if kind == "arrays" else KVCache
            if type(entry) is not expected:
                raise TypeError(
                    f"native_mtp_backbone_cache_type_mismatch: entry={index}"
                )
            NativeMTPBatch._validate_b1_entry(entry)
        for index, entry in enumerate(row.mtp_cache):
            if type(entry) is not KVCache:
                raise TypeError(f"native_mtp_head_cache_type_mismatch: entry={index}")
            NativeMTPBatch._validate_b1_entry(entry)
        for name, entries, expected in (
            ("backbone", row.backbone_cache, row.position.backbone_tokens),
            ("mtp", row.mtp_cache, row.position.mtp_tokens),
        ):
            for index, entry in enumerate(entries):
                if type(entry) is KVCache and entry.offset != expected:
                    raise NativeMTPBatchError(
                        f"native_mtp_{name}_cache_position_mismatch: entry={index}"
                    )

    @staticmethod
    def _validate_b1_entry(entry: Any) -> None:
        if type(entry) is ArraysCache:
            if entry.batch_size != 1:
                raise NativeMTPBatchError("native_mtp_row_cache_must_be_b1")
            if entry.left_padding is not None or entry.lengths is not None:
                raise NativeMTPBatchError("native_mtp_arrays_cache_not_finalized")
            return
        if entry.keys is not None and entry.keys.shape[0] != 1:
            raise NativeMTPBatchError("native_mtp_row_cache_must_be_b1")

    def _merge_admission_rows(self) -> tuple[list[Any], list[Any]]:
        rows = tuple(runtime.row for runtime in self._rows.values())
        backbone = self._merge_entries(tuple(row.backbone_cache for row in rows))
        mtp = self._merge_entries(tuple(row.mtp_cache for row in rows))
        _force_eval(_cache_eval_values(entry) for entry in (*backbone, *mtp))
        return backbone, mtp

    @staticmethod
    def _merge_entries(row_entries: Sequence[Sequence[Any]]) -> list[Any]:
        merged: list[Any] = []
        for layer_entries in zip(*row_entries):
            entry_type = type(layer_entries[0])
            if any(type(entry) is not entry_type for entry in layer_entries):
                raise NativeMTPBatchError("native_mtp_cache_topology_mismatch")
            merged.append(entry_type.merge(list(layer_entries)))
        return merged

    def _ensure_live(self) -> None:
        if self._closed:
            raise NativeMTPBatchError("native_mtp_batch_closed")
        if self._poisoned:
            raise NativeMTPBatchError("native_mtp_batch_poisoned")
        current = getattr(self.target_model, "mtp_capability", None)
        try:
            current_fingerprint = _capability_fingerprint(current)
        except NativeMTPBatchError as exc:
            raise NativeMTPBatchError("native_mtp_capability_changed") from exc
        if (
            current != self.capability
            or current_fingerprint != self.capability_fingerprint
        ):
            raise NativeMTPBatchError("native_mtp_capability_changed")

    def _position_map(
        self, positions: Mapping[int, NativeMTPBatchPosition]
    ) -> OrderedDict[int, NativeMTPBatchPosition]:
        if set(positions) != set(self._rows):
            raise ValueError("native MTP positions must cover the exact active UIDs")
        result: OrderedDict[int, NativeMTPBatchPosition] = OrderedDict()
        for uid in self._rows:
            position = positions[uid]
            if not isinstance(position, NativeMTPBatchPosition):
                raise TypeError("native MTP positions must be immutable positions")
            result[uid] = position
        return result

    def _assert_aligned(
        self,
        positions: Mapping[int, NativeMTPBatchPosition] | None = None,
    ) -> None:
        started = time.perf_counter()
        self.alignment_checks += 1
        if positions is None:
            positions = {uid: runtime.position for uid, runtime in self._rows.items()}
        expected_backbone = mx.array(
            [positions[uid].backbone_tokens for uid in self._rows]
        )
        expected_mtp = mx.array([positions[uid].mtp_tokens for uid in self._rows])
        predicates = []
        for name, entries, expected in (
            ("backbone", self.backbone_cache, expected_backbone),
            ("mtp", self.mtp_cache, expected_mtp),
        ):
            for entry in entries:
                if type(entry) is BatchKVCache:
                    if entry.offset.shape[0] != len(self._rows):
                        raise NativeMTPBatchError(
                            f"native_mtp_{name}_cache_row_mismatch"
                        )
                    predicates.append(mx.all(entry.offset == expected))
                elif type(entry) is ArraysCache:
                    if entry.batch_size != len(self._rows):
                        raise NativeMTPBatchError(
                            "native_mtp_arrays_cache_row_mismatch"
                        )
                    if entry.lengths is not None:
                        raise NativeMTPBatchError(
                            "native_mtp_arrays_cache_not_finalized"
                        )
                    if entry.left_padding is not None:
                        if (
                            entry.left_padding.shape[0] != len(self._rows)
                            or not entry.empty()
                        ):
                            raise NativeMTPBatchError(
                                "native_mtp_arrays_cache_not_finalized"
                            )
                        predicates.append(mx.all(entry.left_padding == 0))
                else:
                    raise NativeMTPBatchError("native_mtp_batch_cache_type_mismatch")
        try:
            if predicates:
                self.alignment_host_syncs += 1
                if not bool(mx.all(mx.stack(predicates)).item()):
                    raise NativeMTPBatchError(
                        "native_mtp_batch_cache_position_mismatch"
                    )
        finally:
            self.alignment_check_ms += (time.perf_counter() - started) * 1000

    @staticmethod
    def _capture_entries(entries: Sequence[Any]) -> _CacheCheckpoint:
        attention: list[_AttentionCheckpoint] = []
        arrays: list[_ArraysCheckpoint] = []
        staged: list[Any] = []
        for index, entry in enumerate(entries):
            if type(entry) is BatchKVCache:
                if entry._right_padding is not None:
                    raise NativeMTPBatchError("native_mtp_batch_cache_not_finalized")
                offset = _copy_small_array(entry.offset)
                left_padding = _copy_small_array(entry.left_padding)
                attention.append(
                    _AttentionCheckpoint(
                        index=index,
                        entry=entry,
                        idx=entry._idx,
                        offset=offset,
                        left_padding=left_padding,
                    )
                )
                staged.extend((offset, left_padding))
            elif type(entry) is ArraysCache:
                if entry.lengths is not None or (
                    entry.left_padding is not None
                    and (entry.left_padding.shape[0] == 0 or not entry.empty())
                ):
                    raise NativeMTPBatchError("native_mtp_arrays_cache_not_finalized")
                state = tuple(_copy_small_array(value) for value in entry.cache)
                left_padding = _copy_small_array(entry.left_padding)
                arrays.append(
                    _ArraysCheckpoint(
                        index=index,
                        entry=entry,
                        state=state,
                        left_padding=left_padding,
                        lengths=None,
                    )
                )
                staged.extend(value for value in state if value is not None)
                if left_padding is not None:
                    staged.append(left_padding)
            else:
                raise NativeMTPBatchError("native_mtp_batch_cache_type_mismatch")
        _force_eval(staged)
        return _CacheCheckpoint(tuple(attention), tuple(arrays))

    def checkpoint(self) -> None:
        self._ensure_live()
        if self._checkpoint is not None:
            raise NativeMTPBatchError("native_mtp_batch_checkpoint_already_active")
        if any(runtime.replay_target is not None for runtime in self._rows.values()):
            raise NativeMTPBatchError("native_mtp_batch_replay_required")
        self._assert_aligned()
        self._checkpoint = _BatchCheckpoint(
            positions=self.positions,
            backbone=self._capture_entries(self.backbone_cache),
            mtp=self._capture_entries(self.mtp_cache),
        )

    def seal_verified(self, positions: Mapping[int, NativeMTPBatchPosition]) -> None:
        self._ensure_live()
        if self._checkpoint is None:
            raise NativeMTPBatchError("native_mtp_batch_checkpoint_missing")
        if self._sealed is not None:
            raise NativeMTPBatchError("native_mtp_batch_verification_already_sealed")
        sealed = self._position_map(positions)
        starts = dict(self._checkpoint.positions)
        for uid, position in sealed.items():
            start = starts[uid]
            if (
                position.logical_cursor < start.logical_cursor
                or position.backbone_tokens < start.backbone_tokens
                or position.mtp_tokens < start.mtp_tokens
            ):
                raise ValueError("native MTP sealed positions cannot move backwards")
        deltas = {
            (
                position.logical_cursor - starts[uid].logical_cursor,
                position.backbone_tokens - starts[uid].backbone_tokens,
                position.mtp_tokens - starts[uid].mtp_tokens,
            )
            for uid, position in sealed.items()
        }
        if len(deltas) != 1:
            raise NativeMTPBatchError("native_mtp_verified_rows_must_be_homogeneous")
        _force_eval(
            _cache_eval_values(entry)
            for entry in (*self.backbone_cache, *self.mtp_cache)
        )
        self._assert_aligned(sealed)
        self._sealed = tuple(sealed.items())

    def commit(self, positions: Mapping[int, NativeMTPBatchPosition]) -> None:
        self._ensure_live()
        if self._checkpoint is None:
            raise NativeMTPBatchError("native_mtp_batch_checkpoint_missing")
        if self._sealed is None:
            raise NativeMTPBatchError("native_mtp_batch_verification_not_sealed")
        committed = tuple(self._position_map(positions).items())
        if committed != self._sealed:
            raise NativeMTPBatchError("native_mtp_batch_partial_commit_requires_replay")
        for uid, position in committed:
            self._rows[uid].position = position
        self._checkpoint = None
        self._sealed = None

    @staticmethod
    def _restore_entries(entries: list[Any], checkpoint: _CacheCheckpoint) -> None:
        staged: list[Any] = []
        for snapshot in checkpoint.attention:
            if entries[snapshot.index] is not snapshot.entry:
                raise NativeMTPBatchError("native_mtp_batch_cache_entry_replaced")
            entry = snapshot.entry
            if entry._idx < snapshot.idx:
                raise NativeMTPBatchError("native_mtp_batch_cache_advanced_backwards")
            entry._idx = snapshot.idx
            entry.offset = _copy_small_array(snapshot.offset)
            entry.left_padding = _copy_small_array(snapshot.left_padding)
            entry._right_padding = None
            staged.extend((entry.offset, entry.left_padding))
        for snapshot in checkpoint.arrays:
            if entries[snapshot.index] is not snapshot.entry:
                raise NativeMTPBatchError("native_mtp_batch_cache_entry_replaced")
            entry = snapshot.entry
            entry.cache = [
                _copy_small_array(value) if value is not None else None
                for value in snapshot.state
            ]
            entry.left_padding = snapshot.left_padding
            entry.lengths = snapshot.lengths
            staged.extend(value for value in entry.cache if value is not None)
        _force_eval(staged)

    def rollback(self) -> None:
        self._ensure_live()
        if self._checkpoint is None:
            raise NativeMTPBatchError("native_mtp_batch_checkpoint_missing")
        checkpoint = self._checkpoint
        try:
            self._restore_entries(self.backbone_cache, checkpoint.backbone)
            self._restore_entries(self.mtp_cache, checkpoint.mtp)
            for uid, position in checkpoint.positions:
                self._rows[uid].position = position
            self._checkpoint = None
            self._sealed = None
            self._assert_aligned()
        except BaseException as exc:
            self._release_poisoned()
            raise NativeMTPBatchError("native_mtp_batch_rollback_failed") from exc

    def _release_poisoned(self) -> None:
        """Drop every request/cache/RNG reference after an inexact restore."""

        self._poisoned = True
        self._closed = True
        self._rows.clear()
        self.backbone_cache = []
        self.mtp_cache = []
        self._checkpoint = None
        self._sealed = None
        self.capability = None
        self.capability_fingerprint = None
        self.target_model = None

    def reject_partial(
        self, targets: Mapping[int, NativeMTPBatchPosition]
    ) -> NativeMTPReplayPlan:
        self._ensure_live()
        if self._checkpoint is None or self._sealed is None:
            raise NativeMTPBatchError("native_mtp_batch_verification_not_sealed")
        replay_targets = self._position_map(targets)
        starts = dict(self._checkpoint.positions)
        sealed = dict(self._sealed)
        strict = False
        grouped: OrderedDict[tuple[int, int, int], list[int]] = OrderedDict()
        for uid, target in replay_targets.items():
            start = starts[uid]
            end = sealed[uid]
            if not (
                start.logical_cursor <= target.logical_cursor <= end.logical_cursor
                and start.backbone_tokens
                <= target.backbone_tokens
                <= end.backbone_tokens
                and start.mtp_tokens <= target.mtp_tokens <= end.mtp_tokens
            ):
                raise ValueError("native MTP replay target exceeds verified positions")
            strict |= target != end
            delta = (
                target.logical_cursor - start.logical_cursor,
                target.backbone_tokens - start.backbone_tokens,
                target.mtp_tokens - start.mtp_tokens,
            )
            grouped.setdefault(delta, []).append(uid)
        if not strict:
            raise ValueError("full verified acceptance must use native MTP commit")
        self.rollback()
        for uid, target in replay_targets.items():
            runtime = self._rows[uid]
            runtime.replay_target = None if target == runtime.position else target
        return NativeMTPReplayPlan(
            targets=tuple(replay_targets.items()),
            groups=tuple(tuple(uids) for uids in grouped.values()),
        )

    def replay_retained(self, positions: Mapping[int, NativeMTPBatchPosition]) -> None:
        self._ensure_live()
        if self._checkpoint is not None:
            raise NativeMTPBatchError("native_mtp_batch_checkpoint_active")
        advanced = self._position_map(positions)
        deltas = set()
        for uid, position in advanced.items():
            runtime = self._rows[uid]
            target = runtime.replay_target
            if target is None:
                raise NativeMTPBatchError("native_mtp_batch_replay_not_required")
            current = runtime.position
            delta = (
                position.logical_cursor - current.logical_cursor,
                position.backbone_tokens - current.backbone_tokens,
                position.mtp_tokens - current.mtp_tokens,
            )
            if any(value not in (0, 1) for value in delta) or delta == (0, 0, 0):
                raise ValueError("native MTP replay must advance one token boundary")
            if (
                position.logical_cursor > target.logical_cursor
                or position.backbone_tokens > target.backbone_tokens
                or position.mtp_tokens > target.mtp_tokens
            ):
                raise ValueError("native MTP replay exceeds accepted positions")
            deltas.add(delta)
        if len(deltas) != 1:
            raise NativeMTPBatchError("native_mtp_replay_rows_must_be_homogeneous")
        self._assert_aligned(advanced)
        _force_eval(
            _cache_eval_values(entry)
            for entry in (*self.backbone_cache, *self.mtp_cache)
        )
        for uid, position in advanced.items():
            runtime = self._rows[uid]
            runtime.position = position
            if position == runtime.replay_target:
                runtime.replay_target = None

    def next_rng_key(self, uid: int) -> Any:
        self._ensure_live()
        try:
            runtime = self._rows[uid]
        except KeyError as exc:
            raise KeyError(f"unknown native MTP batch UID: {uid}") from exc
        keys = mx.random.split(runtime.rng_key)
        runtime.rng_key = keys[0]
        return keys[1]

    @staticmethod
    def _structural_snapshot(entries: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
        snapshots = []
        for entry in entries:
            if type(entry) is BatchKVCache:
                snapshots.append(
                    (
                        entry,
                        entry.keys,
                        entry.values,
                        entry.offset,
                        entry.left_padding,
                        entry._idx,
                        entry._right_padding,
                    )
                )
            elif type(entry) is ArraysCache:
                snapshots.append(
                    (entry, list(entry.cache), entry.left_padding, entry.lengths)
                )
            else:
                raise NativeMTPBatchError("native_mtp_batch_cache_type_mismatch")
        return tuple(snapshots)

    @staticmethod
    def _restore_structure(snapshots: Sequence[tuple[Any, ...]]) -> None:
        staged = []
        for snapshot in snapshots:
            entry = snapshot[0]
            if type(entry) is BatchKVCache:
                (
                    entry.keys,
                    entry.values,
                    entry.offset,
                    entry.left_padding,
                    entry._idx,
                    entry._right_padding,
                ) = snapshot[1:]
                staged.extend(
                    value
                    for value in (entry.offset, entry.left_padding)
                    if value is not None
                )
            else:
                entry.cache = list(snapshot[1])
                entry.left_padding = snapshot[2]
                entry.lengths = snapshot[3]
                staged.extend(value for value in entry.cache if value is not None)
        _force_eval(staged)

    def filter(self, uids: Sequence[int]) -> None:
        self._ensure_live()
        if self._checkpoint is not None:
            raise NativeMTPBatchError("native_mtp_batch_checkpoint_active")
        if not uids:
            raise ValueError("native MTP batch filter cannot produce an empty batch")
        if len(set(uids)) != len(uids) or any(uid not in self._rows for uid in uids):
            raise ValueError("native MTP batch filter requires unique active UIDs")
        indices = [self.uids.index(uid) for uid in uids]
        snapshots = self._structural_snapshot((*self.backbone_cache, *self.mtp_cache))
        old_rows = self._rows
        try:
            for entry in (*self.backbone_cache, *self.mtp_cache):
                entry.filter(indices)
            _force_eval(
                _cache_eval_values(entry)
                for entry in (*self.backbone_cache, *self.mtp_cache)
            )
            self._rows = OrderedDict((uid, old_rows[uid]) for uid in uids)
            self._assert_aligned()
        except BaseException:
            try:
                self._restore_structure(snapshots)
                self._rows = old_rows
            except BaseException as restore_error:
                self._release_poisoned()
                raise NativeMTPBatchError(
                    "native_mtp_batch_filter_restore_failed"
                ) from restore_error
            raise

    @staticmethod
    def _extract_entry(entry: Any, index: int) -> Any:
        if type(entry) is BatchKVCache and entry.keys is None:
            return KVCache()
        if type(entry) is ArraysCache and entry.empty():
            return ArraysCache(len(entry.cache))
        extracted = entry.extract(index)
        if type(entry) is ArraysCache:
            extracted.left_padding = None
            extracted.lengths = None
        return extracted

    def _extracted_row_caches(self, index: int) -> tuple[list[Any], list[Any]]:
        return (
            [self._extract_entry(entry, index) for entry in self.backbone_cache],
            [self._extract_entry(entry, index) for entry in self.mtp_cache],
        )

    @classmethod
    def join(cls, batches: Sequence["NativeMTPBatch"]) -> "NativeMTPBatch":
        if not batches:
            raise ValueError("native MTP batch join requires at least one batch")
        for batch in batches:
            batch._ensure_live()
            if batch._checkpoint is not None:
                raise NativeMTPBatchError("native_mtp_batch_checkpoint_active")
        model = batches[0].target_model
        capability = batches[0].capability
        topology = batches[0]._topology
        if any(
            batch.target_model is not model
            or batch.capability != capability
            or batch.capability_fingerprint != batches[0].capability_fingerprint
            or batch._topology != topology
            for batch in batches
        ):
            raise NativeMTPBatchError("native_mtp_batch_join_incompatible")
        uids = tuple(uid for batch in batches for uid in batch.uids)
        if len(set(uids)) != len(uids):
            raise ValueError("native MTP batch join UIDs must be unique")

        rows: list[NativeMTPBatchRow] = []
        runtime_by_uid: dict[int, _RowRuntime] = {}
        for batch in batches:
            for index, uid in enumerate(batch.uids):
                backbone, mtp = batch._extracted_row_caches(index)
                runtime = batch._rows[uid]
                rows.append(
                    NativeMTPBatchRow(
                        uid=uid,
                        target_model=model,
                        capability=capability,
                        backbone_cache=backbone,
                        mtp_cache=mtp,
                        position=runtime.position,
                        request_config=runtime.row.request_config,
                    )
                )
                runtime_by_uid[uid] = runtime
        joined = cls(rows)
        for uid, runtime in joined._rows.items():
            source = runtime_by_uid[uid]
            runtime.rng_key = source.rng_key
            runtime.replay_target = source.replay_target
        for batch in batches:
            batch._closed = True
            batch._rows.clear()
            batch.backbone_cache = []
            batch.mtp_cache = []
        return joined

    def partition_replay_groups(self) -> tuple["NativeMTPBatch", ...]:
        self._ensure_live()
        if self._checkpoint is not None:
            raise NativeMTPBatchError("native_mtp_batch_checkpoint_active")
        groups: OrderedDict[tuple[int, int, int], list[int]] = OrderedDict()
        for uid, runtime in self._rows.items():
            target = runtime.replay_target
            current = runtime.position
            remaining = (
                (0, 0, 0)
                if target is None
                else (
                    target.logical_cursor - current.logical_cursor,
                    target.backbone_tokens - current.backbone_tokens,
                    target.mtp_tokens - current.mtp_tokens,
                )
            )
            groups.setdefault(remaining, []).append(uid)
        if len(groups) == 1:
            return (self,)

        partitions = []
        for uids in groups.values():
            indices = [self.uids.index(uid) for uid in uids]
            rows = []
            for index, uid in zip(indices, uids):
                backbone, mtp = self._extracted_row_caches(index)
                runtime = self._rows[uid]
                rows.append(
                    NativeMTPBatchRow(
                        uid=uid,
                        target_model=self.target_model,
                        capability=self.capability,
                        backbone_cache=backbone,
                        mtp_cache=mtp,
                        position=runtime.position,
                        request_config=runtime.row.request_config,
                    )
                )
            partition = type(self)(rows)
            for uid in uids:
                partition._rows[uid].rng_key = self._rows[uid].rng_key
                partition._rows[uid].replay_target = self._rows[uid].replay_target
            partitions.append(partition)
        self._closed = True
        self._rows.clear()
        self.backbone_cache = []
        self.mtp_cache = []
        return tuple(partitions)

    def finish(self, uid: int, reason: str) -> None:
        if reason not in {"eos", "length", "cancelled", "error"}:
            raise ValueError(f"unsupported native MTP batch finish reason: {reason}")
        self._ensure_live()
        if uid not in self._rows:
            raise KeyError(f"unknown native MTP batch UID: {uid}")
        if self._checkpoint is not None:
            self.rollback()
        if len(self._rows) == 1:
            self._rows.clear()
            self.backbone_cache = []
            self.mtp_cache = []
            self._closed = True
            return
        self.filter(tuple(active_uid for active_uid in self.uids if active_uid != uid))

    def close(self, reason: str = "cancelled") -> None:
        if reason not in {"eos", "length", "cancelled", "error"}:
            raise ValueError(f"unsupported native MTP batch finish reason: {reason}")
        if self._closed:
            return
        if self._checkpoint is not None:
            self.rollback()
        self._rows.clear()
        self.backbone_cache = []
        self.mtp_cache = []
        self._closed = True
