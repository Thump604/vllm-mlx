"""Transactional cache primitives for Gemma 4 sparse prefill.

This module deliberately does not integrate with an engine.  It defines the
small, fail-closed cache contract needed by a later Simple/CB integration.
Snapshots retain array references (no full-KV copy) and restore *all* mutable
cache metadata.  Logical positions are caller-owned and never inferred from a
rotating cache's physical cursors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)


class GemmaCacheError(RuntimeError):
    """Base error for an unsupported or inconsistent Gemma cache."""


class GemmaCacheTopologyError(GemmaCacheError):
    """The model/cache topology does not match a certified artifact layout."""


class GemmaCacheTransactionError(GemmaCacheError):
    """A cache mutation did not satisfy the declared transaction contract."""


FULL = "full_attention"
SLIDING = "sliding_attention"


@dataclass(frozen=True)
class GemmaArtifactSpec:
    artifact_id: str
    layer_types: tuple[str, ...]
    previous_kvs: tuple[int, ...]
    owner_count: int
    sliding_window: int

    @property
    def layer_count(self) -> int:
        return len(self.layer_types)


def _repeated_layers(sliding_count: int, groups: int) -> tuple[str, ...]:
    return (tuple([SLIDING] * sliding_count + [FULL])) * groups


def _shared_previous_kvs(
    layer_types: Sequence[str], owner_count: int
) -> tuple[int, ...]:
    previous = list(range(len(layer_types)))
    latest_by_type: dict[str, int] = {}
    for index, layer_type in enumerate(layer_types):
        if index < owner_count:
            latest_by_type[layer_type] = index
        else:
            previous[index] = latest_by_type[layer_type]
    return tuple(previous)


_E2B_LAYERS = _repeated_layers(4, 7)
GEMMA4_E2B = GemmaArtifactSpec(
    artifact_id="gemma4-e2b",
    layer_types=_E2B_LAYERS,
    previous_kvs=_shared_previous_kvs(_E2B_LAYERS, 15),
    owner_count=15,
    sliding_window=512,
)
_DENSE_31B_LAYERS = _repeated_layers(5, 10)
GEMMA4_31B = GemmaArtifactSpec(
    artifact_id="gemma4-31b",
    layer_types=_DENSE_31B_LAYERS,
    previous_kvs=tuple(range(60)),
    owner_count=60,
    sliding_window=1024,
)
_A4B_LAYERS = _repeated_layers(5, 5)
GEMMA4_26B_A4B = GemmaArtifactSpec(
    artifact_id="gemma4-26b-a4b",
    layer_types=_A4B_LAYERS,
    previous_kvs=tuple(range(30)),
    owner_count=30,
    sliding_window=1024,
)

GEMMA4_ARTIFACTS = {
    spec.artifact_id: spec for spec in (GEMMA4_E2B, GEMMA4_31B, GEMMA4_26B_A4B)
}


@dataclass(frozen=True)
class GemmaCacheTopology:
    artifact_id: str
    layer_to_slot: tuple[int, ...]
    owner_layers: tuple[int, ...]
    sliding_window: int


def _is_full_cache(cache: Any) -> bool:
    return type(cache) in (KVCache, BatchKVCache)


def _is_rotating_cache(cache: Any) -> bool:
    return type(cache) in (RotatingKVCache, BatchRotatingKVCache)


def validate_gemma_cache_topology(
    spec: GemmaArtifactSpec,
    *,
    layer_types: Sequence[str],
    previous_kvs: Sequence[int],
    cache: Sequence[Any],
) -> GemmaCacheTopology:
    """Validate a cache against one of the bounded, observed Gemma layouts."""

    if GEMMA4_ARTIFACTS.get(spec.artifact_id) != spec:
        raise GemmaCacheTopologyError(f"uncertified Gemma artifact: {spec.artifact_id}")
    if tuple(layer_types) != spec.layer_types:
        raise GemmaCacheTopologyError("decoder layer types do not match artifact")
    if tuple(previous_kvs) != spec.previous_kvs:
        raise GemmaCacheTopologyError("shared-KV owner mapping does not match artifact")
    if len(cache) != spec.owner_count:
        raise GemmaCacheTopologyError(
            f"expected {spec.owner_count} cache owners, got {len(cache)}"
        )
    if len({id(entry) for entry in cache}) != len(cache):
        raise GemmaCacheTopologyError("cache owner entries must not alias")

    owner_layers = tuple(dict.fromkeys(spec.previous_kvs))
    if owner_layers != tuple(range(spec.owner_count)):
        raise GemmaCacheTopologyError("owners must be the compact leading layer range")
    for layer, owner in enumerate(spec.previous_kvs):
        if not 0 <= owner < spec.owner_count or owner > layer:
            raise GemmaCacheTopologyError(f"invalid owner {owner} for layer {layer}")
        if spec.layer_types[owner] != spec.layer_types[layer]:
            raise GemmaCacheTopologyError(f"owner type mismatch for layer {layer}")

    batch_kind: bool | None = None
    for owner, entry in enumerate(cache):
        is_batch = type(entry) in (BatchKVCache, BatchRotatingKVCache)
        if batch_kind is None:
            batch_kind = is_batch
        elif is_batch != batch_kind:
            raise GemmaCacheTopologyError("scalar and batch caches cannot be mixed")
        layer_type = spec.layer_types[owner]
        if layer_type == FULL and not _is_full_cache(entry):
            raise GemmaCacheTopologyError(f"owner {owner} requires a full KV cache")
        if layer_type == SLIDING:
            if not _is_rotating_cache(entry):
                raise GemmaCacheTopologyError(
                    f"owner {owner} requires a rotating KV cache"
                )
            if entry.max_size != spec.sliding_window:
                raise GemmaCacheTopologyError(
                    f"owner {owner} window {entry.max_size} != {spec.sliding_window}"
                )
            if type(entry) is RotatingKVCache and entry.keep != 0:
                raise GemmaCacheTopologyError("Gemma rotating caches require keep=0")
    return GemmaCacheTopology(
        artifact_id=spec.artifact_id,
        layer_to_slot=spec.previous_kvs,
        owner_layers=owner_layers,
        sliding_window=spec.sliding_window,
    )


@dataclass(frozen=True)
class ScalarCacheCursor:
    total_writes: int
    resident_tokens: int
    circular_index: int
    logical_position: int


@dataclass(frozen=True)
class BatchCacheCursor:
    total_writes: tuple[int, ...]
    physical_write_cursor: int
    resident_tokens: int
    circular_index: int
    rotated: bool
    logical_positions: tuple[int, ...]


def scalar_cache_cursor(cache: Any, *, logical_position: int) -> ScalarCacheCursor:
    if logical_position < 0:
        raise GemmaCacheError("logical position must be non-negative")
    if type(cache) is KVCache:
        return ScalarCacheCursor(cache.offset, cache.offset, cache.offset, logical_position)
    if type(cache) is RotatingKVCache:
        _validate_scalar_rotating(cache, allow_oversized=True)
        return ScalarCacheCursor(
            cache.offset, min(cache.offset, cache.max_size), cache._idx, logical_position
        )
    raise GemmaCacheError(f"unsupported scalar cache: {type(cache).__name__}")


def _host_ints(value: Any) -> tuple[int, ...]:
    mx.eval(value)
    return tuple(int(item) for item in value.tolist())


def batch_cache_cursor(
    cache: Any, *, logical_positions: Sequence[int]
) -> BatchCacheCursor:
    logical = tuple(int(position) for position in logical_positions)
    if not logical or min(logical) < 0:
        raise GemmaCacheError("logical positions must be non-empty and non-negative")
    if type(cache) is BatchKVCache:
        total = _host_ints(cache.offset)
        physical, resident, index = cache._idx, cache._idx, cache._idx
        rotated = False
    elif type(cache) is BatchRotatingKVCache:
        _validate_batch_rotating(cache)
        total = _host_ints(cache.offset)
        physical = cache._offset
        resident, index = min(cache._offset, cache.max_size), cache._idx
        rotated = cache.rotated
    else:
        raise GemmaCacheError(f"unsupported batch cache: {type(cache).__name__}")
    if len(total) != len(logical):
        raise GemmaCacheError("logical-position row count does not match cache")
    return BatchCacheCursor(total, physical, resident, index, rotated, logical)


def validate_homogeneous_batch_lane(
    cache: Sequence[Any], *, logical_positions: Sequence[int]
) -> None:
    logical = tuple(int(value) for value in logical_positions)
    if not logical or len(set(logical)) != 1:
        raise GemmaCacheError("Gemma sparse cache lane requires equal logical positions")
    reference: tuple[int, ...] | None = None
    physical: int | None = None
    rotating_state: tuple[int, int, bool] | None = None
    for entry in cache:
        cursor = batch_cache_cursor(entry, logical_positions=logical)
        if len(set(cursor.total_writes)) != 1:
            raise GemmaCacheError("Gemma sparse cache lane has heterogeneous writes")
        if reference is None:
            reference = cursor.total_writes
        elif cursor.total_writes != reference:
            raise GemmaCacheError("Gemma cache owners disagree on physical writes")
        if physical is None:
            physical = cursor.physical_write_cursor
        elif cursor.physical_write_cursor != physical:
            raise GemmaCacheError("Gemma cache owners disagree on write cursor")
        if type(entry) is BatchRotatingKVCache:
            state = (
                cursor.resident_tokens,
                cursor.circular_index,
                cursor.rotated,
            )
            if rotating_state is None:
                rotating_state = state
            elif state != rotating_state:
                raise GemmaCacheError("Gemma rotating owners disagree on resident state")


def validate_aligned_scalar_cache(
    cache: Sequence[Any], *, logical_position: int
) -> None:
    cursors = tuple(
        scalar_cache_cursor(entry, logical_position=logical_position) for entry in cache
    )
    if not cursors:
        raise GemmaCacheError("Gemma scalar cache cannot be empty")
    if len({cursor.total_writes for cursor in cursors}) != 1:
        raise GemmaCacheError("Gemma scalar owners disagree on physical writes")
    rotating = tuple(
        (cursor.resident_tokens, cursor.circular_index)
        for entry, cursor in zip(cache, cursors)
        if type(entry) is RotatingKVCache
    )
    if rotating and len(set(rotating)) != 1:
        raise GemmaCacheError("Gemma rotating owners disagree on resident state")


@dataclass
class _Snapshot:
    cache: Any
    fields: dict[str, Any]

    @classmethod
    def capture(cls, cache: Any) -> "_Snapshot":
        cache_type = type(cache)
        common = ("keys", "values", "offset")
        extras: tuple[str, ...]
        if cache_type is KVCache:
            extras = ()
        elif cache_type is RotatingKVCache:
            extras = ("keep", "max_size", "_idx")
        elif cache_type is BatchKVCache:
            extras = ("left_padding", "_idx", "_right_padding")
        elif cache_type is BatchRotatingKVCache:
            extras = (
                "left_padding",
                "max_size",
                "_idx",
                "_offset",
                "rotated",
                "_lengths",
            )
        else:
            raise GemmaCacheError(f"unsupported cache: {cache_type.__name__}")
        return cls(cache, {name: getattr(cache, name) for name in common + extras})

    def restore(self) -> None:
        for name, value in self.fields.items():
            setattr(self.cache, name, value)


class GemmaCacheCheckpoint:
    """Reference-only checkpoint for a complete compact Gemma cache."""

    def __init__(self, cache: Sequence[Any]):
        if not cache:
            raise GemmaCacheError("cache checkpoint cannot be empty")
        self._cache = tuple(cache)
        self._snapshots = tuple(_Snapshot.capture(entry) for entry in cache)
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def restore(self) -> None:
        if not self._active:
            return
        if tuple(self._cache) != tuple(snapshot.cache for snapshot in self._snapshots):
            raise GemmaCacheTransactionError("cache topology changed during transaction")
        for snapshot in self._snapshots:
            snapshot.restore()
        self._active = False

    def seal(self) -> None:
        if not self._active:
            raise GemmaCacheTransactionError("checkpoint is already closed")
        self._active = False


def _cache_tensors(cache: Sequence[Any]) -> tuple[Any, ...]:
    names = (
        "keys",
        "values",
        "offset",
        "left_padding",
        "_right_padding",
        "_lengths",
    )
    return tuple(
        value
        for entry in cache
        for name in names
        if (value := getattr(entry, name, None)) is not None
    )


def _evaluate_cache(cache: Sequence[Any]) -> None:
    tensors = _cache_tensors(cache)
    if tensors:
        mx.eval(*tensors)


class GemmaCachePairCheckpoint:
    """Checkpoint both operands because batch extend mutates either operand."""

    def __init__(self, destination: Sequence[Any], source: Sequence[Any]):
        if len(destination) != len(source):
            raise GemmaCacheError("cache pair topology differs")
        self.destination = GemmaCacheCheckpoint(destination)
        self.source = GemmaCacheCheckpoint(source)

    def restore(self) -> None:
        self.destination.restore()
        self.source.restore()


def _validate_scalar_rotating(
    cache: RotatingKVCache, *, allow_oversized: bool = False
) -> None:
    if cache.offset < 0 or cache._idx < 0 or cache.max_size <= 0 or cache.keep != 0:
        raise GemmaCacheError("invalid scalar rotating-cache metadata")
    if (cache.keys is None) != (cache.values is None):
        raise GemmaCacheError("rotating cache has only one KV tensor")
    if cache.keys is None:
        if cache.offset != 0 or cache._idx != 0:
            raise GemmaCacheError("empty rotating cache has non-zero cursors")
        return
    if cache.keys.shape[2] != cache.values.shape[2]:
        raise GemmaCacheError("rotating KV resident lengths differ")
    resident = cache.keys.shape[2]
    if cache.offset >= cache.max_size and resident < cache.max_size:
        raise GemmaCacheError("saturated rotating cache has a short resident buffer")
    if resident == 0 or cache._idx > resident:
        raise GemmaCacheError("invalid rotating-cache circular index")
    if not allow_oversized and resident > cache.max_size:
        raise GemmaCacheError("rotating cache is not normalized")
    if cache.offset < cache.max_size and cache._idx != cache.offset:
        raise GemmaCacheError("unsaturated rotating cache cursors disagree")


def normalize_scalar_rotating(cache: RotatingKVCache) -> None:
    """Normalize resident storage without changing the lifetime write cursor."""

    if type(cache) is not RotatingKVCache:
        raise GemmaCacheError("normalization requires RotatingKVCache")
    _validate_scalar_rotating(cache, allow_oversized=True)
    if cache.keys is None:
        return
    checkpoint = GemmaCacheCheckpoint([cache])
    try:
        total_writes = cache.offset
        if total_writes < cache.max_size:
            # Preserve allocator spare capacity.  The cache's logical state is
            # bounded by ``offset/_idx``; slicing here would force a growth
            # allocation on the very next decode token.
            cache._idx = total_writes
        else:
            ordered_keys = cache._temporal_order(cache.keys)
            ordered_values = cache._temporal_order(cache.values)
            cache.keys = mx.contiguous(ordered_keys[..., -cache.max_size :, :])
            cache.values = mx.contiguous(ordered_values[..., -cache.max_size :, :])
            cache._idx = cache.max_size
        cache.offset = total_writes
        _evaluate_cache([cache])
        _validate_scalar_rotating(cache)
        checkpoint.seal()
    except BaseException:
        checkpoint.restore()
        raise


def _validate_batch_rotating(
    cache: BatchRotatingKVCache, *, allow_oversized: bool = False
) -> None:
    if cache.max_size <= 0 or cache._offset < 0 or cache._idx < 0:
        raise GemmaCacheError("invalid batch rotating-cache metadata")
    if (cache.keys is None) != (cache.values is None):
        raise GemmaCacheError("batch rotating cache has only one KV tensor")
    rows = len(cache.offset)
    if len(cache.left_padding) != rows:
        raise GemmaCacheError("batch rotating row metadata differs")
    if cache.keys is None:
        if cache._offset != 0 or cache._idx != 0 or cache.rotated:
            raise GemmaCacheError("empty batch rotating cache has non-zero cursors")
        return
    if cache.keys.shape[0] != rows or cache.values.shape[0] != rows:
        raise GemmaCacheError("batch rotating tensor rows differ from metadata")
    if cache.keys.shape[2] != cache.values.shape[2]:
        raise GemmaCacheError("batch rotating KV resident lengths differ")
    resident = cache.keys.shape[2]
    if cache._offset >= cache.max_size and resident < cache.max_size:
        raise GemmaCacheError("saturated batch rotating cache has a short buffer")
    if (not allow_oversized and resident > cache.max_size) or cache._idx > resident:
        raise GemmaCacheError("invalid batch rotating resident/circular state")
    if allow_oversized and resident > cache.max_size:
        return
    if cache._offset < cache.max_size:
        if cache.rotated or cache._idx != cache._offset:
            raise GemmaCacheError("invalid unsaturated batch rotating state")
    elif cache._idx == cache.max_size:
        if cache.rotated:
            raise GemmaCacheError("normalized saturated cache cannot be rotated")
    elif not (0 < cache._idx < cache.max_size and cache.rotated):
        raise GemmaCacheError("invalid wrapped batch rotating state")


def normalize_batch_rotating(cache: BatchRotatingKVCache) -> None:
    """Normalize a finalized batch cache while preserving lifetime cursors."""

    if type(cache) is not BatchRotatingKVCache:
        raise GemmaCacheError("normalization requires BatchRotatingKVCache")
    if cache._lengths is not None:
        raise GemmaCacheError("right-padded batch cache must be finalized first")
    _validate_batch_rotating(cache, allow_oversized=True)
    if cache.keys is None:
        return
    checkpoint = GemmaCacheCheckpoint([cache])
    try:
        row_offsets = cache.offset
        physical_writes = cache._offset
        resident = min(physical_writes, cache.max_size)
        if cache.keys.shape[2] > cache.max_size or cache.rotated:
            cache._temporal_order()
            dropped = cache.keys.shape[2] - resident
            cache.keys = mx.contiguous(cache.keys[..., -resident:, :])
            cache.values = mx.contiguous(cache.values[..., -resident:, :])
            if dropped:
                cache.left_padding = cache.left_padding - dropped
            cache._idx = resident
            cache.rotated = False
        cache.offset = row_offsets
        cache._offset = physical_writes
        _evaluate_cache([cache])
        _validate_batch_rotating(cache)
        checkpoint.seal()
    except BaseException:
        checkpoint.restore()
        raise


class GemmaOneTokenTransaction:
    """Atomic one-token adoption for a compact scalar or batch cache."""

    def __init__(self, cache: Sequence[Any], *, logical_positions: Sequence[int]):
        if not cache:
            raise GemmaCacheError("transaction cache cannot be empty")
        if len({id(entry) for entry in cache}) != len(cache):
            raise GemmaCacheError("transaction cache contains duplicate owners")
        self.cache = tuple(cache)
        self._state = "active"
        self.logical_before = tuple(int(value) for value in logical_positions)
        for entry in cache:
            if type(entry) is RotatingKVCache:
                _validate_scalar_rotating(entry)
            elif type(entry) is BatchRotatingKVCache:
                _validate_batch_rotating(entry)
        self.checkpoint = GemmaCacheCheckpoint(cache)
        self._write_journal = tuple(self._capture_overwrite(entry) for entry in cache)
        mx.eval(
            *(
                tensor
                for journal in self._write_journal
                if journal is not None
                for tensor in journal[1:]
            )
        )
        self._batch = type(cache[0]) in (BatchKVCache, BatchRotatingKVCache)
        if self._batch:
            validate_homogeneous_batch_lane(cache, logical_positions=self.logical_before)
            self.before = tuple(
                batch_cache_cursor(entry, logical_positions=self.logical_before)
                for entry in cache
            )
        else:
            if len(self.logical_before) != 1:
                raise GemmaCacheError("scalar transaction requires one logical position")
            validate_aligned_scalar_cache(
                cache, logical_position=self.logical_before[0]
            )
            self.before = tuple(
                scalar_cache_cursor(entry, logical_position=self.logical_before[0])
                for entry in cache
            )
        self._guard_originals: list[tuple[Any, bool, Any]] = []
        self._update_counts = [0] * len(self.cache)
        self._install_guards()

    def _install_guards(self) -> None:
        for index, entry in enumerate(self.cache):
            had_instance_override = "update_and_fetch" in entry.__dict__
            instance_override = entry.__dict__.get("update_and_fetch")
            original = entry.update_and_fetch

            def guarded(keys, values, *, _index=index, _original=original):
                if keys.shape[2] != 1:
                    raise GemmaCacheTransactionError(
                        "one-token transaction received a multi-token update"
                    )
                if self._update_counts[_index] != 0:
                    raise GemmaCacheTransactionError(
                        "cache owner was updated more than once"
                    )
                self._update_counts[_index] = 1
                return _original(keys, values)

            self._guard_originals.append(
                (entry, had_instance_override, instance_override)
            )
            entry.update_and_fetch = guarded

    def _remove_guards(self) -> None:
        if not self._guard_originals:
            return
        for entry, had_instance_override, instance_override in self._guard_originals:
            if had_instance_override:
                entry.update_and_fetch = instance_override
            else:
                entry.__dict__.pop("update_and_fetch", None)
        self._guard_originals.clear()

    @staticmethod
    def _capture_overwrite(cache: Any) -> tuple[int, Any, Any] | None:
        """Copy only the one resident slot a decode quantum may overwrite."""

        if cache.keys is None:
            return None
        if type(cache) is KVCache:
            index = cache.offset
        elif type(cache) is BatchKVCache:
            index = cache._idx
        elif type(cache) is RotatingKVCache:
            index = 0 if cache._idx == cache.max_size else cache._idx
        elif type(cache) is BatchRotatingKVCache:
            index = 0 if cache._idx == cache.max_size else cache._idx
        else:  # pragma: no cover - constructor rejects other cache types
            raise GemmaCacheError(f"unsupported cache: {type(cache).__name__}")
        if index >= cache.keys.shape[2]:
            return None
        return (
            index,
            mx.contiguous(cache.keys[..., index : index + 1, :]),
            mx.contiguous(cache.values[..., index : index + 1, :]),
        )

    def rollback(self) -> None:
        if self._state != "active":
            return
        self._remove_guards()
        self.checkpoint.restore()
        for entry, journal in zip(self.cache, self._write_journal):
            if journal is None:
                continue
            index, keys, values = journal
            entry.keys[..., index : index + 1, :] = keys
            entry.values[..., index : index + 1, :] = values
        mx.eval(
            *(
                tensor
                for entry in self.cache
                for tensor in (entry.keys, entry.values)
                if tensor is not None
            )
        )
        self._write_journal = ()
        self._state = "rolled_back"

    def commit(self, *, logical_positions: Sequence[int]) -> None:
        if self._state != "active":
            raise GemmaCacheTransactionError(
                f"transaction is already {self._state}"
            )
        logical_after = tuple(int(value) for value in logical_positions)
        if logical_after != tuple(value + 1 for value in self.logical_before):
            self.rollback()
            raise GemmaCacheTransactionError("logical positions did not advance by one")
        try:
            if self._update_counts != [1] * len(self.cache):
                raise GemmaCacheTransactionError(
                    "each cache owner must be updated exactly once"
                )
            # Completing a full circular lap leaves mlx-vlm's flag set even
            # though the physical array is again in temporal order.  Clear
            # only this metadata bit; no window normalization/copy is needed.
            for entry in self.cache:
                if (
                    type(entry) is BatchRotatingKVCache
                    and entry._idx == entry.max_size
                    and entry.rotated
                ):
                    entry.rotated = False
            if self._batch:
                validate_homogeneous_batch_lane(
                    self.cache, logical_positions=logical_after
                )
                after = tuple(
                    batch_cache_cursor(entry, logical_positions=logical_after)
                    for entry in self.cache
                )
                for entry, old, new in zip(self.cache, self.before, after):
                    if new.total_writes != tuple(value + 1 for value in old.total_writes):
                        raise GemmaCacheTransactionError(
                            "batch physical writes did not advance by one"
                        )
                    if new.physical_write_cursor != old.physical_write_cursor + 1:
                        raise GemmaCacheTransactionError(
                            "batch write cursor did not advance by one"
                        )
                    expected_index = old.circular_index + 1
                    if (
                        type(entry) is BatchRotatingKVCache
                        and old.circular_index == entry.max_size
                    ):
                        expected_index = 1
                    if new.circular_index != expected_index:
                        raise GemmaCacheTransactionError(
                            "batch circular index transition is invalid"
                        )
                    expected_rotated = (
                        type(entry) is BatchRotatingKVCache
                        and new.physical_write_cursor >= entry.max_size
                        and new.circular_index < entry.max_size
                    )
                    if new.rotated != expected_rotated:
                        raise GemmaCacheTransactionError(
                            "batch rotation transition is invalid"
                        )
                    expected_resident = (
                        min(old.physical_write_cursor + 1, entry.max_size)
                        if type(entry) is BatchRotatingKVCache
                        else old.resident_tokens + 1
                    )
                    if new.resident_tokens != expected_resident:
                        raise GemmaCacheTransactionError(
                            "batch resident transition is invalid"
                        )
            else:
                validate_aligned_scalar_cache(
                    self.cache, logical_position=logical_after[0]
                )
                after = tuple(
                    scalar_cache_cursor(entry, logical_position=logical_after[0])
                    for entry in self.cache
                )
                for entry, old, new in zip(self.cache, self.before, after):
                    if new.total_writes != old.total_writes + 1:
                        raise GemmaCacheTransactionError(
                            "scalar physical writes did not advance by one"
                        )
                    expected_index = old.circular_index + 1
                    if (
                        type(entry) is RotatingKVCache
                        and old.circular_index == entry.max_size
                    ):
                        expected_index = 1
                    if new.circular_index != expected_index:
                        raise GemmaCacheTransactionError(
                            "scalar circular index transition is invalid"
                        )
                    expected_resident = (
                        min(old.total_writes + 1, entry.max_size)
                        if type(entry) is RotatingKVCache
                        else old.resident_tokens + 1
                    )
                    if new.resident_tokens != expected_resident:
                        raise GemmaCacheTransactionError(
                            "scalar resident transition is invalid"
                        )
            _evaluate_cache(self.cache)
            self._remove_guards()
            self.checkpoint.seal()
            self._write_journal = ()
            self._state = "committed"
        except BaseException:
            self.rollback()
            raise


def atomic_batch_filter(
    cache: Sequence[Any],
    indices: Any,
    *,
    logical_positions: Sequence[int],
) -> tuple[int, ...]:
    if len({id(entry) for entry in cache}) != len(cache):
        raise GemmaCacheError("filter cache contains duplicate owners")
    checkpoint = GemmaCacheCheckpoint(cache)
    try:
        for entry in cache:
            if type(entry) not in (BatchKVCache, BatchRotatingKVCache):
                raise GemmaCacheError("filter requires batch cache entries")
            entry.filter(indices)
        selected = tuple(logical_positions[int(index)] for index in indices)
        _evaluate_cache(cache)
        validate_homogeneous_batch_lane(cache, logical_positions=selected)
        checkpoint.seal()
        return selected
    except BaseException:
        checkpoint.restore()
        raise


def atomic_batch_extend(destination: Sequence[Any], source: Sequence[Any]) -> None:
    destination_ids = [id(entry) for entry in destination]
    source_ids = [id(entry) for entry in source]
    if len(set(destination_ids)) != len(destination_ids):
        raise GemmaCacheError("destination cache contains duplicate owners")
    if len(set(source_ids)) != len(source_ids):
        raise GemmaCacheError("source cache contains duplicate owners")
    if set(destination_ids) & set(source_ids):
        raise GemmaCacheError("destination and source caches overlap")
    checkpoint = GemmaCachePairCheckpoint(destination, source)
    try:
        for target, addition in zip(destination, source):
            if type(target) is not type(addition) or type(target) not in (
                BatchKVCache,
                BatchRotatingKVCache,
            ):
                raise GemmaCacheError("extend cache types differ")
            if _is_rotating_cache(target) and target.max_size != addition.max_size:
                raise GemmaCacheError("extend rotating windows differ")
            target.extend(addition)
        _evaluate_cache((*destination, *source))
        # BatchRotatingKVCache.extend temporalizes its source.  Restore source
        # even on success so cache ownership remains request-local.
        checkpoint.source.restore()
        checkpoint.destination.seal()
    except BaseException:
        checkpoint.restore()
        raise


def run_atomic_one_token(
    cache: Sequence[Any],
    *,
    logical_positions: Sequence[int],
    forward: Callable[[], Any],
    evaluate: Callable[[Any], None] = mx.eval,
) -> Any:
    """Run/evaluate one lazy forward and publish cache mutation atomically."""

    transaction = GemmaOneTokenTransaction(
        cache, logical_positions=logical_positions
    )
    try:
        result = forward()
        evaluate(result)
        transaction.commit(
            logical_positions=tuple(value + 1 for value in logical_positions)
        )
        return result
    except BaseException:
        transaction.rollback()
        raise


__all__ = [
    "BatchCacheCursor",
    "GEMMA4_26B_A4B",
    "GEMMA4_31B",
    "GEMMA4_ARTIFACTS",
    "GEMMA4_E2B",
    "GemmaArtifactSpec",
    "GemmaCacheCheckpoint",
    "GemmaCacheError",
    "GemmaCachePairCheckpoint",
    "GemmaCacheTopology",
    "GemmaCacheTopologyError",
    "GemmaCacheTransactionError",
    "GemmaOneTokenTransaction",
    "ScalarCacheCursor",
    "atomic_batch_extend",
    "atomic_batch_filter",
    "batch_cache_cursor",
    "normalize_scalar_rotating",
    "normalize_batch_rotating",
    "run_atomic_one_token",
    "scalar_cache_cursor",
    "validate_gemma_cache_topology",
    "validate_aligned_scalar_cache",
    "validate_homogeneous_batch_lane",
]
