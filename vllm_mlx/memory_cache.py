# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware prefix cache for vllm-mlx.

This module provides a prefix cache implementation that tracks memory usage
and evicts entries based on memory pressure rather than entry count.

Key features:
- Automatic memory limit detection based on available system RAM
- Accurate memory tracking for MLX array caches
- LRU eviction triggered by memory thresholds
- No unnecessary deep copies (MLX arrays are immutable)

Example:
    config = MemoryCacheConfig(max_memory_percent=0.25)
    cache = MemoryAwarePrefixCache(model, config)

    # Fetch returns reference (no copy) - safe because MLX arrays are immutable
    kv_cache, remaining = cache.fetch(tokens)

    # Store tracks memory automatically
    cache.store(tokens, kv_cache)
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import logging
import math
import threading
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .cache_owner_identity import (
    CacheOwnerIdentity,
    ModelCacheOwnerBinding,
    ModelCacheRequestBinding,
    OwnerBindingDecision,
    PREPARED_STORE_VERSION,
    PreparedOwnerBoundCacheEntry,
    VerifiedCacheOwnerContext,
)

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails
# Bump this when the cache on-disk format or KV semantics change.
# Loading a cache with a different version is rejected automatically.
_CACHE_PERSIST_VERSION = 5
_HYBRID_CACHE_LAYOUT_ABI = "arrays-kv-v2"


def _get_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes, or 0 if detection fails.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not installed, using fallback memory limit")
        return 0
    except Exception as e:
        logger.warning(f"Failed to detect available memory: {e}")
        return 0


def _array_memory(arr) -> int:
    """
    Estimate array memory from shape+dtype without triggering lazy eval.

    Accessing .nbytes on a lazy MLX array forces evaluation of the entire
    computation graph, causing a VRAM spike. This function uses shape and
    dtype metadata (which are always available without eval) to compute
    the same value.

    Args:
        arr: An MLX array or similar object.

    Returns:
        Estimated memory in bytes.
    """
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        if hasattr(dtype, "size"):
            return math.prod(arr.shape) * dtype.size
    # Fallback for non-MLX arrays or objects without shape/dtype
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    return 0


def _nested_array_memory(value: Any) -> int:
    """Sum ``_array_memory`` over an arbitrarily nested state structure.

    Cache ``state`` payloads are not always a flat ``(keys, values)`` pair:
    CacheList yields a list of sub-cache states and PoolingCache yields
    ``(buf_kv, buf_gate, pooled)`` with possible ``None`` members. Unpacking
    those as two values raised, was swallowed, and the entry was accounted as
    zero bytes — so the dashboard showed 0% cache memory and, far worse, the
    byte-based LRU eviction never fired for such models.
    """
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_nested_array_memory(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_array_memory(v) for v in value)
    return _array_memory(value)


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """
    Estimate memory usage of a KV cache in bytes.

    This function inspects MLX arrays in the cache and calculates their
    total memory footprint using shape+dtype metadata to avoid triggering
    lazy evaluation (which would cause a VRAM spike).

    Args:
        cache: List of layer cache objects, each containing keys/values tensors.

    Returns:
        Estimated memory usage in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        # Handle different cache object types
        # Check dict first since dicts have .keys() method that would match below
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            # Extracted state dict.  The payload may be a flat (keys, values)
            # pair or an arbitrarily nested container/mapping — walk it
            # recursively like the .state-property branch below.
            total_bytes += _nested_array_memory(layer_cache["state"])
        elif getattr(layer_cache, "preserve_auxiliary_kv_state", False):
            # Some attention caches carry state required for replay beyond
            # keys/values (for example QSA's raw index keys and text/MRoPE
            # positions).  Their explicit state protocol is authoritative:
            # pricing only keys/values would let an entry exceed the hard cap.
            total_bytes += _nested_array_memory(layer_cache.state)
        # Handle QuantizedKVCache: keys/values are tuples of (data, scales, biases)
        elif hasattr(layer_cache, "keys") and isinstance(
            getattr(layer_cache, "keys", None), (list, tuple)
        ):
            for arr in layer_cache.keys:
                total_bytes += _array_memory(arr)
            for arr in layer_cache.values:
                total_bytes += _array_memory(arr)
            continue
        elif hasattr(layer_cache, "caches") and isinstance(
            getattr(layer_cache, "caches", None), (list, tuple)
        ):
            # Container caches (CacheList): price the children the same way
            # storage snapshots them — recursively, per child class.  Walking
            # the container's .state instead would price sliced views while
            # detachment copies the children's raw buffers.
            total_bytes += estimate_kv_cache_memory(list(layer_cache.caches))
        elif (
            hasattr(layer_cache, "keys")
            and hasattr(layer_cache, "values")
            and not callable(getattr(layer_cache, "keys", None))
        ):
            # keys/values-carrying caches (KVCache, RotatingKVCache,
            # ChunkedKVCache, batch variants).  Price the RAW buffers: that
            # is exactly what detachment copies.  Pricing the offset-sliced
            # .state view here under-counts ring/chunked layers whose padded
            # buffer cannot be sliced without breaking their semantics —
            # measured 20-30% resident-vs-accounted drift, enough to breach
            # the byte cap.
            total_bytes += _array_memory(layer_cache.keys)
            total_bytes += _array_memory(layer_cache.values)
            # Batch variants carry per-row metadata arrays (offset is an
            # mx.array there); detachment copies them, so price them too.
            for attr in ("left_padding", "lengths", "offset"):
                extra = getattr(layer_cache, attr, None)
                if extra is not None and hasattr(extra, "shape"):
                    total_bytes += _array_memory(extra)
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            # Stateful caches without keys/values (ArraysCache, MambaCache).
            # Walk the state recursively: the payload may nest containers or
            # mappings, and the old two-way unpack silently measured those
            # as 0.
            try:
                total_bytes += _nested_array_memory(layer_cache.state)
            except (TypeError, ValueError):
                pass
            # Detachment also copies these metadata arrays on state-carrying
            # layers; price them so accounting equals snapshot residency.
            for attr in ("left_padding", "lengths"):
                extra = getattr(layer_cache, attr, None)
                if extra is not None and hasattr(extra, "shape"):
                    total_bytes += _array_memory(extra)

    return total_bytes


@dataclass(frozen=True)
class MemoryCacheConfig:
    """
    Configuration for memory-aware prefix cache.

    Attributes:
        max_memory_mb: Maximum memory in MB. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Hard limit on number of entries (safety net).
        enable_memory_tracking: Whether to track per-entry memory.
        kv_quantize: Whether to quantize KV cache layers for reduced memory.
        kv_bits: Number of bits for KV cache quantization.
        kv_group_size: Group size for KV cache quantization.
        kv_min_quantize_tokens: Minimum sequence length for quantization to apply.
        min_prefix_tokens: Minimum cached prefix length eligible for reuse.
    """

    max_memory_mb: int | None = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT
    max_entries: int = 1000  # Safety limit
    enable_memory_tracking: bool = True
    kv_quantize: bool = False
    kv_bits: int = 8
    kv_group_size: int = 64
    kv_min_quantize_tokens: int = 256
    min_prefix_tokens: int = 128

    def __post_init__(self) -> None:
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.kv_min_quantize_tokens < 0:
            raise ValueError(
                f"kv_min_quantize_tokens must be >= 0, got {self.kv_min_quantize_tokens}"
            )
        if self.min_prefix_tokens < 1:
            raise ValueError(
                f"min_prefix_tokens must be >= 1, got {self.min_prefix_tokens}"
            )

    def compute_memory_limit(self) -> int:
        """
        Compute the memory limit in bytes.

        Returns:
            Memory limit in bytes.
        """
        if self.max_memory_mb is not None:
            return self.max_memory_mb * _BYTES_PER_MB

        available = _get_available_memory()
        if available > 0:
            limit = int(available * self.max_memory_percent)
            return max(limit, _MIN_MEMORY_BYTES)

        # Fallback: assume 8GB system, use configured percent
        fallback_total = 8 * 1024 * _BYTES_PER_MB
        return int(fallback_total * self.max_memory_percent)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    entry_count: int = 0
    # Entries refused by store() (over-limit, undetachable, or pipeline
    # failure).  Fail-closed trades a leak for a cache miss; this makes
    # those misses observable instead of silent.
    store_rejections: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "current_memory_mb": round(self.current_memory_bytes / _BYTES_PER_MB, 2),
            "max_memory_mb": round(self.max_memory_bytes / _BYTES_PER_MB, 2),
            "memory_utilization": round(self.memory_utilization, 4),
            "entry_count": self.entry_count,
            "store_rejections": self.store_rejections,
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with memory tracking."""

    tokens: tuple[int, ...]
    cache: list[Any]
    memory_bytes: int
    auxiliary: dict[str, Any] | None = None
    persistence_eligible: bool = False

    @classmethod
    def create(
        cls,
        tokens: list[int],
        cache: list[Any],
        auxiliary: dict[str, Any] | None = None,
        persistence_eligible: bool = False,
    ) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        if auxiliary:
            memory += sum(getattr(value, "nbytes", 0) for value in auxiliary.values())
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
            auxiliary=auxiliary,
            persistence_eligible=persistence_eligible,
        )


def _is_cache_layer_trimmable(layer_cache: Any) -> bool:
    """Return whether a cache layer can safely be rewound for partial reuse."""
    if getattr(layer_cache, "preserve_auxiliary_kv_state", False):
        # Generic rewind reconstructs KV-like layers from keys/values only.
        # Auxiliary-state caches require a class-specific coordinated trim;
        # until that protocol exists, admit exact hits but reject partial reuse.
        return False
    if isinstance(layer_cache, _QuantizedCacheWrapper):
        if "max_size" in layer_cache.orig_attrs:
            return False
        return hasattr(layer_cache, "offset") and hasattr(layer_cache, "keys")

    # _trim_cache_offset does not currently rewind container children.
    if hasattr(layer_cache, "caches"):
        return False

    is_trimmable = getattr(layer_cache, "is_trimmable", None)
    if callable(is_trimmable):
        try:
            return bool(is_trimmable())
        except Exception:
            logger.debug(
                "Failed to check cache layer trimmability for %s",
                type(layer_cache).__name__,
                exc_info=True,
            )
            return False

    # Compatibility fallback for simple KV-like cache implementations.
    return hasattr(layer_cache, "offset") and hasattr(layer_cache, "keys")


def _trim_cache_offset(cache: list[Any], trim_by: int) -> list[Any]:
    """Create copies of cache layers with the last ``trim_by`` positions removed.

    This is used when returning a cached KV state to the scheduler so that
    the last N positions are "freed" and the model will recompute them on the
    next forward pass (preventing duplicate KV entries).

    For plain KVCache: reduces offset (surplus data beyond offset is harmless
    since merge slices to ``keys[:, :, :offset, :]``).

    For RotatingKVCache: actually trims the circular buffer — reducing offset
    alone breaks ``size()`` / ``_temporal_order`` invariants.

    Supports KVCache, RotatingKVCache, and _QuantizedCacheWrapper.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import RotatingKVCache

    trimmed: list[Any] = []
    eval_targets: list[Any] = []
    for layer_cache in cache:
        if isinstance(layer_cache, _QuantizedCacheWrapper):
            # Shallow copy with reduced offset
            tc = _QuantizedCacheWrapper.__new__(_QuantizedCacheWrapper)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            tc.bits = layer_cache.bits
            tc.group_size = layer_cache.group_size
            tc.orig_type = layer_cache.orig_type
            tc.orig_attrs = layer_cache.orig_attrs
            trimmed.append(tc)
        elif isinstance(layer_cache, RotatingKVCache):
            if layer_cache.keys is None or trim_by <= 0:
                trimmed.append(layer_cache)
                continue
            # RotatingKVCache: must trim buffer, not just offset.
            # The buffer stores the last min(offset, max_size) tokens in a
            # circular arrangement.  Trimming excess positions from the END
            # means removing the newest entries (chronologically last).
            old_offset = layer_cache.offset
            new_offset = max(old_offset - trim_by, 0)
            old_size = min(old_offset, layer_cache.max_size)
            entries_to_keep = max(0, old_size - trim_by)

            orig_cls = type(layer_cache)
            tc = orig_cls.__new__(orig_cls)
            tc.offset = new_offset
            tc.max_size = layer_cache.max_size
            tc.keep = getattr(layer_cache, "keep", 0)
            tc.step = getattr(layer_cache, "step", layer_cache.max_size)

            if entries_to_keep <= 0:
                # All buffer content is beyond the trim point — clear
                tc.keys = None
                tc.values = None
                tc._idx = 0
                tc.offset = 0
            elif entries_to_keep < old_size:
                # Reorder to temporal order, keep the oldest entries
                ordered_k = layer_cache._temporal_order(layer_cache.keys)
                ordered_v = layer_cache._temporal_order(layer_cache.values)
                kept_k = ordered_k[:, :, :entries_to_keep, :]
                kept_v = ordered_v[:, :, :entries_to_keep, :]

                if new_offset >= tc.max_size:
                    # Invariant: when offset >= max_size, buffer must be
                    # full (keys.shape[2] == max_size).  Left-pad with
                    # zeros to restore the full buffer.  Zeros represent
                    # positions evicted long ago; _idx = max_size so
                    # _temporal_order returns as-is and _update_in_place
                    # rotates to overwrite zeros first.
                    pad_n = tc.max_size - entries_to_keep
                    pad_k = mx.zeros(
                        (kept_k.shape[0], kept_k.shape[1], pad_n, kept_k.shape[3]),
                        dtype=kept_k.dtype,
                    )
                    pad_v = mx.zeros(
                        (kept_v.shape[0], kept_v.shape[1], pad_n, kept_v.shape[3]),
                        dtype=kept_v.dtype,
                    )
                    tc.keys = mx.concatenate([pad_k, kept_k], axis=2)
                    tc.values = mx.concatenate([pad_v, kept_v], axis=2)
                    tc._idx = tc.max_size
                else:
                    if entries_to_keep < new_offset:
                        # Buffer has fewer entries than offset requires.
                        # This happens when old_offset > max_size (rotating)
                        # and the trim brought new_offset below max_size.
                        # Pad with zeros on the left to maintain the invariant
                        # size() == keys.shape[2], preventing merge crashes.
                        pad_n = new_offset - entries_to_keep
                        pad_k = mx.zeros(
                            (
                                kept_k.shape[0],
                                kept_k.shape[1],
                                pad_n,
                                kept_k.shape[3],
                            ),
                            dtype=kept_k.dtype,
                        )
                        pad_v = mx.zeros(
                            (
                                kept_v.shape[0],
                                kept_v.shape[1],
                                pad_n,
                                kept_v.shape[3],
                            ),
                            dtype=kept_v.dtype,
                        )
                        tc.keys = mx.concatenate([pad_k, kept_k], axis=2)
                        tc.values = mx.concatenate([pad_v, kept_v], axis=2)
                        tc._idx = new_offset
                    else:
                        tc.keys = kept_k
                        tc.values = kept_v
                        tc._idx = entries_to_keep
                eval_targets.extend([tc.keys, tc.values])
            else:
                # No entries removed (trim_by == 0 already handled above,
                # this covers entries_to_keep == old_size edge case)
                tc.keys = layer_cache.keys
                tc.values = layer_cache.values
                tc._idx = layer_cache._idx
            trimmed.append(tc)
        elif (
            hasattr(layer_cache, "offset")
            and hasattr(layer_cache, "keys")
            and not isinstance(layer_cache.keys, (list, tuple))
        ):
            orig_cls = type(layer_cache)
            tc = orig_cls.__new__(orig_cls)
            new_offset = max(layer_cache.offset - trim_by, 0)
            keys = layer_cache.keys
            values = layer_cache.values
            # Slice the arrays down to new_offset rather than just shrinking the
            # offset pointer.  Sharing the original (over-sized) array across
            # requests lets attention paths that read the full underlying
            # buffer (e.g. Gemma 4's KV-shared layers, which read cache.state
            # directly instead of going through update_and_fetch) see stale
            # tokens from the previous owner — issue #384.
            if (
                keys is not None
                and hasattr(keys, "shape")
                and len(keys.shape) >= 3
                and new_offset < keys.shape[-2]
            ):
                tc.keys = keys[..., :new_offset, :]
                tc.values = values[..., :new_offset, :]
            else:
                tc.keys = keys
                tc.values = values
            tc.offset = new_offset
            # Preserve type-specific attrs (max_size, keep, step, _idx)
            for attr in ("max_size", "keep", "step", "_idx"):
                if hasattr(layer_cache, attr):
                    setattr(tc, attr, getattr(layer_cache, attr))
            trimmed.append(tc)
        else:
            trimmed.append(layer_cache)

    if eval_targets:
        mx.eval(*eval_targets)

    return trimmed


def _needs_kv_trim(layer: Any) -> bool:
    """Check if a cache layer has oversized KV arrays (duck-typed, no MLX import)."""
    if getattr(layer, "preserve_auxiliary_kv_state", False):
        # The layer's state getter owns coordinated trimming of keys/values
        # and any auxiliary arrays. Rebuilding it as a plain KVCache would
        # silently discard those arrays.
        return False
    keys = getattr(layer, "keys", None)
    offset = getattr(layer, "offset", None)
    if keys is None or offset is None:
        return False
    if isinstance(keys, (list, tuple)):
        return False  # QuantizedKVCache — skip
    shape = getattr(keys, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    return 0 < offset < shape[2]


def _trim_to_offset(cache: list[Any]) -> list[Any]:
    """Trim KV arrays to their actual used size (offset) before storage.

    KV arrays are often pre-allocated larger than needed (e.g. 4096 slots
    when only 100 are used).  This slices them down to ``offset`` LAZILY:
    the slices materialize in ``_detach_cache_for_storage``'s single
    batched eval, and only for entries the size preflight has accepted.
    Evaluating here would defeat the preflight — a rejected over-limit
    entry would still incur an entry-sized allocation.

    Args:
        cache: List of cache layer objects (KVCache or other types).

    Returns:
        New list with KVCache layers trimmed to their offset.
        Non-KVCache layers are passed through unchanged.
    """
    if not any(_needs_kv_trim(layer) for layer in cache):
        return cache

    from mlx_lm.models.cache import KVCache

    trimmed = []
    for layer in cache:
        if isinstance(layer, KVCache) and layer.keys is not None:
            offset = layer.offset
            if offset <= 0 or offset >= layer.keys.shape[2]:
                trimmed.append(layer)
                continue
            tc = KVCache()
            tc.keys = layer.keys[:, :, :offset, :]
            tc.values = layer.values[:, :, :offset, :]
            tc.offset = offset
            trimmed.append(tc)
        else:
            trimmed.append(layer)

    return trimmed


class UndetachableCacheError(Exception):
    """Raised when a cache entry cannot be safely detached for storage.

    ``store()`` treats this as a rejection: storing the entry by reference
    would reintroduce the aliasing / lazy-graph retention leak, so the safe
    response is to not store it at all (a skipped store costs one cache
    miss; a leaked store pins live batch buffers until eviction).
    """


def _bears_arrays(value: Any, _seen: set[int] | None = None) -> bool:
    """True if ``value`` (object or nested container) holds any array,
    duck-typed as having both ``shape`` and ``dtype``.  Used to decide
    whether an unrecognized cache layer is safe to pass through unchanged
    (nothing to pin) or must fail the store (unknown retention risk)."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return False
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return True
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return False
    _seen.add(id(value))
    if isinstance(value, dict):
        return any(_bears_arrays(v, _seen) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_bears_arrays(v, _seen) for v in value)
    # Object attributes: instance dict plus any __slots__ up the MRO.
    # vars() (not dir()) so properties are never executed here.
    attrs = list(vars(value).values()) if hasattr(value, "__dict__") else []
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if hasattr(value, name):
                attrs.append(getattr(value, name))
    return any(_bears_arrays(v, _seen) for v in attrs)


def _collect_mx_array_ids(
    value: Any, mx: Any, out: set[int], _seen: set[int] | None = None
) -> None:
    """Collect ``id()`` of every real ``mx.array`` reachable from ``value``.

    Same traversal as ``_bears_arrays`` (containers, instance dicts,
    ``__slots__``; properties are never executed).  Only genuine MLX arrays
    are collected: duck-typed array-likes pass through detachment by
    reference deliberately and must not trip the identity postcondition."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if isinstance(value, mx.array):
        out.add(id(value))
        return
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return
    _seen.add(id(value))
    if isinstance(value, dict):
        for v in value.values():
            _collect_mx_array_ids(v, mx, out, _seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _collect_mx_array_ids(v, mx, out, _seen)
        return
    attrs = list(vars(value).values()) if hasattr(value, "__dict__") else []
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if hasattr(value, name):
                attrs.append(getattr(value, name))
    for v in attrs:
        _collect_mx_array_ids(v, mx, out, _seen)


def _detach_cache_for_storage(
    cache: list[Any], _eval_targets: list[Any] | None = None
) -> list[Any]:
    """Materialize and detach cache arrays before storage.

    Per-request caches handed to ``store()`` are built from lazy slices of
    live batch arrays (``extract_cache`` / ``_trim_cache_offset``), and
    hybrid layers such as ``ArraysCache`` expose their mutable state
    container by reference via ``.state``.  Storing those references has two
    failure modes:

    - the stored entry aliases containers/buffers the batch generator keeps
      mutating (same class of bug as the SimpleEngine snapshot aliasing,
      #575), and
    - unevaluated arrays retain their entire lazy computation graph, pinning
      every upstream batch-wide buffer.  Under sustained traffic this leaks
      Metal buffer handles roughly proportional to generated tokens per
      stored entry, until the process hits the device resource limit
      (``[metal::malloc] Resource limit (N) exceeded``) and aborts.

    Force a compact, evaluated copy of every MLX array so the stored entry owns
    exactly its own data and nothing else.

    Fail-closed: raises ``UndetachableCacheError`` if any layer cannot be
    safely snapshotted — either an unrecognized layer type that carries
    arrays, or a recognized layer whose snapshot fails.  Array-free unknown
    layers pass through unchanged (nothing to pin).
    """
    try:
        import mlx.core as mx
    except ImportError:
        # No-MLX environments (e.g. the CI unit-test lane) have no MLX
        # arrays to materialize; container/attribute snapshotting below
        # still applies, arrays pass through _detach unchanged.
        mx = None

    # Recursive calls (CacheList children) share the caller's target list
    # so the whole entry still materializes in one batched eval.
    is_root = _eval_targets is None
    eval_targets: list[Any] = [] if is_root else _eval_targets

    def _detach(arr: Any) -> Any:
        # Only real MLX arrays are detached: they alone carry a lazy graph
        # and Metal buffers.  Duck-typed array-likes (test doubles, numpy)
        # have nothing to pin and pass through unchanged.
        if mx is None or not isinstance(arr, mx.array):
            return arr
        # ``arr + 0`` guarantees a freshly allocated buffer.  mx.contiguous
        # is documented to copy only when necessary and may share the input
        # buffer for an already-row-contiguous input, which would let the
        # stored snapshot alias live storage.  ``+ 0`` promotes bool to
        # int32, so cast back — the cast reads the already-fresh buffer, the
        # allocation guarantee holds either way.
        out = arr + 0
        if out.dtype != arr.dtype:
            out = out.astype(arr.dtype)
        eval_targets.append(out)
        return out

    def _detach_container(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(_detach_container(v) for v in value)
        if isinstance(value, list):
            return [_detach_container(v) for v in value]
        if isinstance(value, dict):
            # Nested mapping state (e.g. {"ssm": array}) — rebuild the
            # mapping so the snapshot owns its container, preserving the
            # mapping type.  A mapping type whose constructor rejects an
            # iterable of pairs raises and fails the store closed.
            return type(value)((k, _detach_container(v)) for k, v in value.items())
        if (
            value is not None
            and not (hasattr(value, "shape") and hasattr(value, "dtype"))
            and _bears_arrays(value)
        ):
            # A non-array object (or unsupported container, e.g. a set)
            # smuggling arrays inside a state payload: copying around it
            # would alias whatever it holds.  Fail closed.
            raise UndetachableCacheError(
                f"state payload holds arrays inside an unsupported "
                f"{type(value).__name__}"
            )
        return _detach(value)

    def _detach_layer(layer: Any) -> Any:
        if layer is None:
            return layer
        if isinstance(layer, dict):
            if "state" in layer:
                snap_dict = dict(layer)
                snap_dict["state"] = _detach_container(layer["state"])
                rest = {k: v for k, v in snap_dict.items() if k != "state"}
                if _bears_arrays(rest):
                    raise UndetachableCacheError(
                        "dict layer carries arrays outside its 'state' field"
                    )
                return snap_dict
            if _bears_arrays(layer):
                raise UndetachableCacheError(
                    "dict layer without 'state' carries arrays"
                )
            return layer
        # copy.copy (not __new__ + __dict__.update) so classes using
        # __slots__ (e.g. _QuantizedCacheWrapper) snapshot correctly too.
        if getattr(layer, "preserve_auxiliary_kv_state", False):
            # Cache implementations opt into this protocol when replay needs
            # state beyond the ordinary keys/values pair.  Copying through
            # the complete state getter/setter keeps those arrays coordinated
            # and lets the verified postcondition catch omissions.
            if not hasattr(layer, "state"):
                raise UndetachableCacheError(
                    f"{type(layer).__name__} declares auxiliary KV state "
                    "without exposing a state protocol"
                )
            snap = copy.copy(layer)
            snap.state = _detach_container(layer.state)
            return snap
        if hasattr(layer, "keys") and not callable(getattr(layer, "keys")):
            # KVCache / RotatingKVCache / _QuantizedCacheWrapper style.
            snap = copy.copy(layer)
            snap.keys = _detach_container(layer.keys)
            snap.values = _detach_container(layer.values)
            # Batch variants also carry per-row metadata arrays that the
            # batch generator rebinds (offset is an mx.array there).
            for attr in ("left_padding", "lengths", "offset"):
                extra = getattr(snap, attr, None)
                if extra is not None and hasattr(extra, "shape"):
                    setattr(snap, attr, _detach(extra))
            return snap
        if hasattr(layer, "caches") and isinstance(
            getattr(layer, "caches"), (list, tuple)
        ):
            # Container caches (e.g. ``CacheList``): their ``state`` setter
            # writes through to the child caches in place, so going through
            # the setter would mutate the caller's children.  Snapshot the
            # children recursively instead.
            snap = copy.copy(layer)
            children = _detach_cache_for_storage(
                list(layer.caches), _eval_targets=eval_targets
            )
            snap.caches = (
                tuple(children) if isinstance(layer.caches, tuple) else children
            )
            return snap
        if hasattr(layer, "state"):
            # Hybrid state-container layers (e.g. ``ArraysCache``): ``.state``
            # returns the live mutable list — clone the container and detach
            # its arrays instead of aliasing it.
            snap = copy.copy(layer)
            snap.state = _detach_container(layer.state)
            for attr in ("left_padding", "lengths"):
                if getattr(snap, attr, None) is not None:
                    setattr(snap, attr, _detach(getattr(snap, attr)))
            return snap
        if _bears_arrays(layer):
            raise UndetachableCacheError(
                f"unrecognized cache layer type {type(layer).__name__} "
                "carries arrays"
            )
        return layer

    detached: list[Any] = []
    for layer in cache:
        try:
            detached.append(_detach_layer(layer))
        except UndetachableCacheError:
            raise
        except Exception as e:
            raise UndetachableCacheError(
                f"failed to snapshot {type(layer).__name__}: "
                f"{type(e).__name__}: {e}"
            ) from e

    if is_root and mx is not None:
        # Verified postcondition, not assumed branch coverage: no MLX array
        # in the snapshot may be the same object as one in the live input.
        # Branch dispatch above is heuristic (every mlx_lm ``_BaseCache``
        # subclass inherits a default ``state`` property returning ``[]``,
        # so the state branch matches layers whose arrays live elsewhere
        # and would "detach" an empty state, aliasing the rest).  Checking
        # object identity after the fact closes that hole for any layer
        # shape, present or future.
        live_ids: set[int] = set()
        snap_ids: set[int] = set()
        for layer in cache:
            _collect_mx_array_ids(layer, mx, live_ids)
        for layer in detached:
            _collect_mx_array_ids(layer, mx, snap_ids)
        if live_ids & snap_ids:
            raise UndetachableCacheError(
                "snapshot still aliases live cache arrays (layer carries "
                "arrays its state/keys view does not expose)"
            )

    if is_root and eval_targets:
        try:
            mx.eval(*eval_targets)
        except Exception as e:
            # e.g. stream-affinity RuntimeError when arrays belong to a
            # stream whose thread has exited.  store() must reject, not
            # raise — several call sites have no enclosing try.
            raise UndetachableCacheError(
                f"failed to materialize snapshot: {type(e).__name__}: {e}"
            ) from e

    return detached


class _QuantizedCacheWrapper:
    """Lightweight wrapper storing quantized KV arrays + original cache metadata.

    Unlike ``QuantizedKVCache``, this preserves enough info to reconstruct
    the *original* cache type (KVCache, RotatingKVCache, etc.) on dequantize.
    """

    __slots__ = (
        "keys",
        "values",
        "offset",
        "bits",
        "group_size",
        "orig_type",
        "orig_attrs",
    )

    def __init__(self, layer: Any, bits: int, group_size: int):
        import mlx.core as mx

        self.keys = mx.quantize(layer.keys, group_size=group_size, bits=bits)
        self.values = mx.quantize(layer.values, group_size=group_size, bits=bits)
        self.offset = layer.offset
        self.bits = bits
        self.group_size = group_size
        self.orig_type = type(layer)
        # Preserve RotatingKVCache-specific attrs
        self.orig_attrs = {}
        for attr in ("max_size", "keep", "step", "_idx"):
            if hasattr(layer, attr):
                self.orig_attrs[attr] = getattr(layer, attr)


def _quantize_cache(cache: list[Any], bits: int = 8, group_size: int = 64) -> list[Any]:
    """Quantize KV cache layers to reduce memory.

    Plain KVCache and the exact Qwen4 QSAKVCache contract are supported.
    RotatingKVCache (sliding window) is left as-is because its internal
    _idx/rotation state is tightly coupled with update_and_fetch logic and
    cannot survive quantize/dequantize roundtrip. RotatingKVCache is typically
    small (max_size=1024) so skipping it is fine.
    """
    from mlx_lm.models.cache import KVCache

    try:
        from mlx_vlm.models.qwen4_exp.language import QSAKVCache
    except ImportError:
        QSAKVCache = None

    quantized = []
    for layer in cache:
        if type(layer) is KVCache and getattr(layer, "keys", None) is not None:
            quantized.append(_QuantizedCacheWrapper(layer, bits, group_size))
        elif QSAKVCache is not None and type(layer) is QSAKVCache:
            if bits != 8 or group_size != 64:
                quantized.append(layer)
                continue
            if (
                layer.keys is None
                or layer.values is None
                or layer.index_keys is None
                or layer.index_position_ids is None
            ):
                raise ValueError("Qwen4 QSA cache state is incomplete")
            quantized.append(layer.to_quantized(group_size=group_size, bits=bits))
        else:
            quantized.append(layer)
    return quantized


def _dequantize_cache(cache: list[Any]) -> list[Any]:
    """Dequantize _QuantizedCacheWrapper layers and copy non-quantized layers.

    All layers are copied (never returned by reference) so that the model's
    ``update_and_fetch`` mutations don't corrupt the stored cache entry.
    """
    import mlx.core as mx

    try:
        from mlx_vlm.models.qwen4_exp.language import (
            QSAKVCache,
            QSAQuantizedKVCache,
        )
    except ImportError:
        QSAKVCache = None
        QSAQuantizedKVCache = None

    result = []
    for layer in cache:
        if QSAQuantizedKVCache is not None and type(layer) is QSAQuantizedKVCache:
            if layer.bits != 8 or layer.group_size != 64:
                raise ValueError("unsupported Qwen4 QSA quantization contract")
            if layer.keys is None or layer.values is None:
                raise ValueError("quantized Qwen4 QSA K/V state is incomplete")
            if layer.index_keys is None or layer.index_position_ids is None:
                raise ValueError("quantized Qwen4 QSA indexer state is incomplete")
            restored = QSAKVCache()
            restored.keys = mx.dequantize(*layer.keys, group_size=64, bits=8)
            restored.values = mx.dequantize(*layer.values, group_size=64, bits=8)
            restored.offset = layer.offset
            restored.index_keys = mx.array(layer.index_keys)
            restored.index_position_ids = mx.array(layer.index_position_ids)
            result.append(restored)
            continue
        if isinstance(layer, _QuantizedCacheWrapper):
            # Reconstruct original cache type from quantized data
            orig_cls = layer.orig_type
            kv = orig_cls.__new__(orig_cls)
            kv.keys = mx.dequantize(
                *layer.keys, group_size=layer.group_size, bits=layer.bits
            )
            kv.values = mx.dequantize(
                *layer.values, group_size=layer.group_size, bits=layer.bits
            )
            kv.offset = layer.offset
            # Slice the dequantized arrays down to offset so that readers
            # which bypass offset (e.g. Gemma 4 KV-shared layers reading
            # cache.state directly) cannot see stale tokens from a previous
            # request.  Mirrors the plain-KVCache slice in
            # _trim_cache_offset — see issue #384.
            if (
                kv.keys is not None
                and hasattr(kv.keys, "shape")
                and len(kv.keys.shape) >= 3
                and kv.offset < kv.keys.shape[-2]
            ):
                kv.keys = kv.keys[..., : kv.offset, :]
                kv.values = kv.values[..., : kv.offset, :]
            # Restore type-specific attrs (max_size, keep, step, _idx)
            for attr, val in layer.orig_attrs.items():
                setattr(kv, attr, val)
            result.append(kv)
        elif hasattr(layer, "keys") and hasattr(layer, "offset"):
            # Deep-copy non-quantized cache layers (e.g. RotatingKVCache)
            # so model's in-place mutations don't corrupt stored entries
            orig_cls = type(layer)
            kv = orig_cls.__new__(orig_cls)
            kv.keys = mx.array(layer.keys) if layer.keys is not None else None
            kv.values = mx.array(layer.values) if layer.values is not None else None
            kv.offset = layer.offset
            for attr in ("max_size", "keep", "step", "_idx"):
                if hasattr(layer, attr):
                    setattr(kv, attr, getattr(layer, attr))
            result.append(kv)
        else:
            result.append(layer)
    return result


def _compute_model_fingerprint(model: Any) -> str:
    """Compute a fingerprint from model architecture for cache compatibility.

    Used to reject disk-persisted caches created by a different model or
    a different quantisation of the same model.  The fingerprint is a
    short hex digest of (num_layers, hidden_size, vocab_size, num_kv_heads,
    head_dim) — lightweight and deterministic.
    """
    import hashlib

    parts: list[str] = []
    # Walk model.config / model.args / direct attributes
    for cfg_attr in ("config", "args", "model_config"):
        cfg = getattr(model, cfg_attr, None)
        if cfg is not None:
            break
    if cfg is None:
        cfg = model  # fallback: attributes on the model itself

    for key in (
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "model_type",
    ):
        val = getattr(cfg, key, None)
        if val is not None:
            parts.append(f"{key}={val}")

    fingerprint = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    logger.debug(f"[model_fingerprint] {fingerprint} ({', '.join(parts)})")
    return fingerprint


def _identity_attr(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _model_text_config(model: Any) -> Any:
    """Return the language config that owns cache geometry.

    Text-only models expose these fields directly. VLM wrappers such as
    Qwen3.5/Qwen3.8 keep them under ``text_config``.
    """
    cfg = None
    for cfg_attr in ("config", "args", "model_config"):
        cfg = getattr(model, cfg_attr, None)
        if cfg is not None:
            break
    cfg = cfg if cfg is not None else model
    text_config = _identity_attr(cfg, "text_config")
    return text_config if text_config is not None else cfg


def _compute_model_persistence_fingerprint(
    model: Any, artifact_identity: str | None = None
) -> str:
    """Return a strict, stable identity for restart-persisted cache state."""
    cfg = None
    for cfg_attr in ("config", "args", "model_config"):
        cfg = getattr(model, cfg_attr, None)
        if cfg is not None:
            break
    cfg = cfg if cfg is not None else model
    keys = (
        "_commit_hash",
        "revision",
        "_name_or_path",
        "name_or_path",
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rope_theta",
        "rope_scaling",
        "sliding_window",
        "layer_types",
        "tie_word_embeddings",
        "torch_dtype",
        "quantization",
        "quantization_config",
    )
    artifact: Any = artifact_identity
    if artifact_identity:
        path = Path(artifact_identity).expanduser()
        if path.exists():
            if path.is_symlink():
                raise ValueError("model artifact root must not be a symlink")
            resolved = path.resolve()
            files = []
            candidates = (
                [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
            )
            for candidate in candidates:
                if candidate.is_symlink():
                    raise ValueError(
                        "model artifact must not contain symlinks: "
                        f"{candidate.relative_to(resolved)}"
                    )
                if not candidate.is_file():
                    continue
                before = candidate.stat()
                digest = hashlib.sha256()
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                after = candidate.stat()
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError("model artifact changed while fingerprinting")
                files.append(
                    {
                        "path": (
                            candidate.name
                            if resolved.is_file()
                            else candidate.relative_to(resolved).as_posix()
                        ),
                        "size": after.st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            artifact = {"resolved_path": str(resolved), "files": files}
        else:
            revision = _identity_attr(cfg, "_commit_hash") or _identity_attr(
                cfg, "revision"
            )
            if not revision and "@" not in artifact_identity:
                # A floating remote repository name can resolve to different
                # weights after restart. Disable persistence unless the loaded
                # configuration or caller supplies an immutable revision.
                return ""
            artifact = {
                "reference": artifact_identity,
                "revision": revision,
            }
    identity = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "artifact": artifact,
        "config": {
            key: value
            for key in keys
            if (value := _identity_attr(cfg, key)) is not None
        },
    }
    try:
        payload = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    except Exception as exc:
        raise ValueError(f"model identity is not serializable: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _compute_tokenizer_persistence_fingerprint(
    tokenizer: Any, template_renderer: Any | None = None
) -> str:
    """Hash tokenizer vocabulary and formatting semantics used by cache keys."""
    if tokenizer is None:
        return ""
    digest = hashlib.sha256()
    header = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "renderer_class": (
            f"{type(template_renderer).__module__}."
            f"{type(template_renderer).__qualname__}"
            if template_renderer is not None
            else None
        ),
        "renderer_chat_template": getattr(template_renderer, "chat_template", None),
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":"), default=str).encode()
    )
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if not isinstance(vocab, dict) or not vocab:
            return ""
        for token, token_id in sorted(vocab.items(), key=lambda item: item[0]):
            digest.update(str(token).encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
            digest.update(str(token_id).encode())
            digest.update(b"\0")
    else:
        return ""
    return digest.hexdigest()


def _cache_layer_topology(layer: Any) -> tuple[str, int] | None:
    """Describe one cache layer using its exact replay-state contract."""
    layer_type = f"{type(layer).__module__}.{type(layer).__qualname__}"
    state = getattr(layer, "state", None)
    if getattr(layer, "preserve_auxiliary_kv_state", False):
        if not isinstance(state, (list, tuple)):
            return None
        return layer_type, len(state)
    if isinstance(state, list):
        return layer_type, len(state)
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        return layer_type, 2
    return None


def _cache_topology(model: Any) -> tuple[tuple[str, int], ...] | None:
    """Describe the ordered empty-cache topology without sequence state."""
    make_cache = getattr(model, "make_cache", None)
    if not callable(make_cache):
        return None
    try:
        cache = make_cache()
    except Exception:
        logger.debug("Failed to construct cache topology", exc_info=True)
        return None
    topology = tuple(_cache_layer_topology(layer) for layer in cache)
    if not topology or any(layer is None for layer in topology):
        return None
    return topology


def _cache_topology_fingerprint(
    topology: tuple[tuple[str, int], ...] | None,
    config: MemoryCacheConfig | None = None,
    runtime_identity: dict[str, Any] | None = None,
) -> str:
    if not topology:
        return ""
    config_identity = None
    if config is not None:
        config_identity = {
            "kv_quantize": config.kv_quantize,
            "kv_bits": config.kv_bits,
            "kv_group_size": config.kv_group_size,
            "kv_min_quantize_tokens": config.kv_min_quantize_tokens,
        }
    payload = json.dumps(
        {
            "abi": _HYBRID_CACHE_LAYOUT_ABI,
            "topology": topology,
            "cache_config": config_identity,
            "runtime": runtime_identity or {},
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_cache_owner_persistence_identity(
    model: Any,
    tokenizer: Any,
    model_identity: str,
    cache_config: MemoryCacheConfig,
    cache_runtime_identity: Mapping[str, Any],
    template_renderer: Any,
) -> Mapping[str, str]:
    """Derive the exact loaded model/tokenizer/cache identity used by owners."""

    if not isinstance(model_identity, str) or not model_identity:
        raise ValueError("cache owner model identity must be non-empty")
    identity = {
        "model": _compute_model_persistence_fingerprint(model, model_identity),
        "tokenizer": _compute_tokenizer_persistence_fingerprint(
            tokenizer, template_renderer
        ),
        "cache_layout": _cache_topology_fingerprint(
            _cache_topology(model),
            cache_config,
            dict(cache_runtime_identity),
        ),
    }
    if any(not value for value in identity.values()):
        raise ValueError("complete model/tokenizer/cache provenance is required")
    return identity


def _mx_array_to_numpy(value: Any) -> tuple[Any, str | None]:
    """Materialize one MLX value to numpy on the calling model-owner thread."""
    import numpy as np

    original_dtype = str(value.dtype).rsplit(".", 1)[-1]
    if original_dtype == "bfloat16":
        import mlx.core as mx

        converted = value.astype(mx.float32)
        mx.eval(converted)
        return np.array(converted), original_dtype
    try:
        return np.array(value), None
    except RuntimeError as exc:
        if "buffer format string" not in str(exc):
            raise
        import mlx.core as mx

        converted = value.astype(mx.float32)
        mx.eval(converted)
        return np.array(converted), original_dtype


def _snapshot_hybrid_layer(layer: Any):
    from .cache_persistence import HybridLayerSnapshot

    layer_type = type(layer).__name__
    if getattr(layer, "preserve_auxiliary_kv_state", False):
        qualified_name = f"{type(layer).__module__}.{type(layer).__qualname__}"
        state = getattr(layer, "state", None)
        if (
            qualified_name != "mlx_vlm.models.qwen4_exp.language.QSAKVCache"
            or not isinstance(state, tuple)
            or len(state) != 4
            or any(value is None for value in state)
        ):
            raise ValueError(f"unsupported persisted {layer_type} state")
        tensors = {}
        original_dtypes = []
        for index, value in enumerate(state):
            array, original_dtype = _mx_array_to_numpy(value)
            tensors[f"state_{index}"] = array
            original_dtypes.append(original_dtype)
        return HybridLayerSnapshot(
            layer_type,
            tensors,
            {
                "num_arrays": len(state),
                "state_original_dtypes": original_dtypes,
                "state_container": "tuple",
            },
        )
    if layer_type in {"KVCache", "RotatingKVCache"}:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or isinstance(keys, (tuple, list)):
            raise ValueError(f"unsupported persisted {layer_type} state")
        keys_np, keys_dtype = _mx_array_to_numpy(keys)
        values_np, values_dtype = _mx_array_to_numpy(values)
        metadata: dict[str, Any] = {"offset": int(layer.offset)}
        if keys_dtype is not None:
            metadata["keys_original_dtype"] = keys_dtype
        if values_dtype is not None:
            metadata["values_original_dtype"] = values_dtype
        for attr in ("max_size", "keep", "step", "_idx"):
            if hasattr(layer, attr):
                value = getattr(layer, attr)
                if isinstance(value, (bool, int, float, str)) or value is None:
                    metadata[attr] = value
        return HybridLayerSnapshot(
            layer_type, {"keys": keys_np, "values": values_np}, metadata
        )
    if layer_type == "ArraysCache" and isinstance(getattr(layer, "state", None), list):
        tensors = {}
        original_dtypes = []
        for index, value in enumerate(layer.state):
            if value is None:
                raise ValueError("ArraysCache contains an uninitialized state")
            array, original_dtype = _mx_array_to_numpy(value)
            tensors[f"state_{index}"] = array
            original_dtypes.append(original_dtype)
        state_count = len(tensors)
        metadata_arrays = []
        metadata_original_dtypes = {}
        for attr in ("left_padding", "lengths"):
            value = getattr(layer, attr, None)
            if value is not None:
                array, original_dtype = _mx_array_to_numpy(value)
                tensors[attr] = array
                metadata_arrays.append(attr)
                metadata_original_dtypes[attr] = original_dtype
        return HybridLayerSnapshot(
            layer_type,
            tensors,
            {
                "num_arrays": state_count,
                "state_original_dtypes": original_dtypes,
                "metadata_arrays": metadata_arrays,
                "metadata_original_dtypes": metadata_original_dtypes,
                "meta_state": getattr(layer, "meta_state", ""),
            },
        )
    raise ValueError(f"unsupported persisted cache layer: {layer_type}")


def _iter_cache_arrays(layer: Any):
    if getattr(layer, "preserve_auxiliary_kv_state", False):
        state = getattr(layer, "state", None)
        if isinstance(state, (list, tuple)):
            yield from (value for value in state if value is not None)
        return
    if hasattr(layer, "state") and isinstance(layer.state, list):
        yield from (value for value in layer.state if value is not None)
        for attr in ("left_padding", "lengths"):
            value = getattr(layer, attr, None)
            if value is not None:
                yield value
        return
    for attr in ("keys", "values"):
        value = getattr(layer, attr, None)
        if value is not None:
            yield value


class MemoryAwarePrefixCache:
    """
    Prefix cache with memory-based eviction.

    This cache tracks memory usage per entry and evicts based on memory
    pressure rather than entry count. It uses LRU (Least Recently Used)
    ordering for eviction decisions.

    Key design decisions:
    - No deep copies on fetch: MLX arrays are immutable, so sharing is safe
    - Memory tracking per entry: Accurate accounting for eviction
    - Auto-detection of available RAM: Adapts to different systems
    - OrderedDict for O(1) LRU operations

    Thread Safety:
        This class is NOT thread-safe. Use external locking if needed.
    """

    def __init__(
        self,
        model: Any,
        config: MemoryCacheConfig | None = None,
        tokenizer: Any | None = None,
        model_identity: str | None = None,
        cache_runtime_identity: dict[str, Any] | None = None,
        template_renderer: Any | None = None,
        cache_owner_context: VerifiedCacheOwnerContext | None = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
            tokenizer: Tokenizer whose exact vocabulary and formatting define
                cache keys. Required for hybrid restart persistence.
            model_identity: Stable model artifact path or revision identity.
                Required for hybrid restart persistence.
            cache_runtime_identity: Cache-shaping runtime settings such as the
                rotating-cache limit. Included in the strict layout identity.
            template_renderer: Processor whose effective chat template renders
                cache-key token sequences. Included in tokenizer identity.
            cache_owner_context: Opaque context verified from the loaded model,
                tokenizer, artifact bytes, cache layout, and governed registry.
        """
        self._model_id = id(model)
        self._model = model
        self._config = config or MemoryCacheConfig()
        self._cache_runtime_identity = dict(cache_runtime_identity or {})
        self._model_fingerprint = _compute_model_fingerprint(model)
        try:
            if cache_owner_context is not None and not model_identity:
                raise ValueError(
                    "owner-bound cache requires the loader-resolved model identity"
                )
            self._persistence_identity = {
                "model": (
                    _compute_model_persistence_fingerprint(model, model_identity)
                    if model_identity
                    else ""
                ),
                "tokenizer": _compute_tokenizer_persistence_fingerprint(
                    tokenizer, template_renderer
                ),
                "cache_layout": _cache_topology_fingerprint(
                    _cache_topology(model),
                    self._config,
                    self._cache_runtime_identity,
                ),
            }
            if (
                cache_owner_context is not None
                and dict(cache_owner_context.persistence_identity)
                != self._persistence_identity
            ):
                raise ValueError(
                    "verified cache owner provenance does not match live cache"
                )
        except Exception as exc:
            if cache_owner_context is not None:
                raise
            logger.warning(
                "[cache_persist] strict identity unavailable; restart persistence "
                "disabled: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._persistence_identity = {
                "model": "",
                "tokenizer": "",
                "cache_layout": "",
            }
        self._owner_identity: CacheOwnerIdentity | None = None
        self._cache_owner_context = cache_owner_context
        self._owner_prepared_entries: dict[
            str, tuple[PreparedOwnerBoundCacheEntry, _CacheEntry]
        ] = {}

        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        self._entries: OrderedDict[tuple[int, ...], _CacheEntry] = OrderedDict()

        # Sorted index of token keys for efficient prefix/supersequence lookup.
        # Tuple lexicographic ordering means a prefix key P is always < any
        # extension of P, so bisect gives O(log N) range scans instead of O(N).
        self._sorted_keys: list[tuple[int, ...]] = []

        # Memory tracking
        self._max_memory = self._config.compute_memory_limit()
        configured_token_limit = self._cache_runtime_identity.get("max_kv_size")
        if not isinstance(configured_token_limit, int) or configured_token_limit < 1:
            cfg = None
            for cfg_attr in ("config", "args", "model_config"):
                cfg = getattr(model, cfg_attr, None)
                if cfg is not None:
                    break
            configured_token_limit = _identity_attr(
                cfg or model, "max_position_embeddings"
            )
        self._max_persisted_tokens = (
            configured_token_limit
            if isinstance(configured_token_limit, int) and configured_token_limit > 0
            else 1024 * 1024
        )
        self._current_memory = 0
        self._memory_lock = threading.RLock()
        # Serializes the entry-sized snapshot copy in store().  Separate
        # from _memory_lock so accounting stays responsive during the GPU
        # copy, while concurrent stores still materialize one at a time —
        # N simultaneous entry-sized copies would multiply peak memory by N.
        self._copy_lock = threading.Lock()

        # Statistics
        self._stats = CacheStats(max_memory_bytes=self._max_memory)

        # Track the match type from the last fetch() call
        self._last_match_type: str | None = None
        self._last_matched_key: tuple[int, ...] | None = None

        # Optional SSD cold tier (set via set_ssd_tier())
        self._ssd_tier = None

        logger.info(
            f"MemoryAwarePrefixCache initialized: "
            f"max_memory={self._max_memory / _BYTES_PER_MB:.1f}MB, "
            f"max_entries={self._config.max_entries}"
        )

    def fetch(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """Fetch through the legacy API only when no live owner is bound."""

        if self._owner_identity is not None:
            raise RuntimeError("owner-bound cache fetch requires a request lease")
        return self._fetch_unchecked(tokens)

    def _fetch_unchecked(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """
        Find cached KV state for the given tokens.

        This method searches for exact matches, prefix matches, supersequence
        matches, and longest-common-prefix (LCP) matches.  Uses a sorted key
        index for O(log N) lookup instead of scanning all entries.

        Returns the independently owned stored state directly. Callers must
        clone its cache wrappers before replay because some MLX cache buffers
        support in-place updates even though the stored backing is detached
        from the request that created it.

        Args:
            tokens: Input token sequence.

        Returns:
            Tuple of (cache, remaining_tokens):
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        self._last_matched_key = None
        if not tokens:
            self._stats.misses += 1
            self._last_match_type = "miss"
            return None, tokens
        if len(tokens) < self._config.min_prefix_tokens:
            self._stats.misses += 1
            self._last_match_type = "miss_short_prefix"
            return None, tokens

        tokens_key = tuple(tokens)

        # --- O(1) exact match ---
        if tokens_key in self._entries:
            entry = self._entries[tokens_key]
            self._entries.move_to_end(tokens_key)
            self._stats.hits += 1
            self._stats.tokens_saved += len(tokens)
            self._last_match_type = "exact"
            self._last_matched_key = tokens_key
            cache_out = (
                _dequantize_cache(entry.cache)
                if self._config.kv_quantize
                else entry.cache
            )
            return cache_out, []

        # --- O(log N) prefix & supersequence match via sorted index ---
        best_match: _CacheEntry | None = None
        best_length = 0
        best_super: _CacheEntry | None = None

        sorted_keys = self._sorted_keys
        if sorted_keys:
            # Find insertion point for tokens_key in the sorted list.
            # Keys that are prefixes of tokens_key or supersequences will be
            # clustered around this position due to lexicographic ordering.
            idx = bisect.bisect_left(sorted_keys, tokens_key)

            # Scan backwards from idx to find cached keys that are PREFIXES
            # of tokens_key (shorter cached sequences).  A prefix P of T
            # satisfies P <= T lexicographically, so P is at idx-1 or earlier.
            for i in range(idx - 1, -1, -1):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len >= len(tokens_key):
                    continue  # Not a prefix (same length or longer)
                # Check if cached_key is a prefix of tokens_key
                if tokens_key[:cached_len] == cached_key:
                    if cached_len > best_length:
                        best_match = self._entries[cached_key]
                        best_length = cached_len
                    # Found best prefix — shorter entries can't be longer
                    break
                # Once we go past the prefix range, stop
                if cached_key[0] != tokens_key[0]:
                    break

            # Scan forward from idx to find cached keys that are SUPERSEQUENCES
            # of tokens_key (longer cached sequences starting with tokens_key).
            for i in range(idx, len(sorted_keys)):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len < len(tokens_key):
                    continue
                # Check if tokens_key is a prefix of cached_key
                if cached_key[: len(tokens_key)] == tokens_key:
                    if best_super is None or cached_len > len(best_super.tokens):
                        best_super = self._entries[cached_key]
                else:
                    # Past the supersequence range
                    break

        # --- Supersequence match handling ---
        if best_super is not None:
            n_cached = len(best_super.tokens)
            n_requested = len(tokens)
            excess = n_cached - n_requested

            has_non_trimmable = any(
                not _is_cache_layer_trimmable(lc) for lc in best_super.cache
            )

            if excess > 0 and has_non_trimmable:
                logger.debug(
                    "[cache_fetch] supersequence match skipped: "
                    "non-trimmable cache layers (hybrid model)"
                )
            elif excess > 0:
                trimmed_cache = _trim_cache_offset(best_super.cache, excess)
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                self._last_matched_key = best_super.tokens
                trimmed_cache = (
                    _dequantize_cache(trimmed_cache)
                    if self._config.kv_quantize
                    else trimmed_cache
                )
                return trimmed_cache, []
            else:
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                self._last_matched_key = best_super.tokens
                cache_out = (
                    _dequantize_cache(best_super.cache)
                    if self._config.kv_quantize
                    else best_super.cache
                )
                return cache_out, []

        # --- Prefix match ---
        if best_match is not None:
            self._entries.move_to_end(best_match.tokens)
            self._stats.hits += 1
            self._stats.tokens_saved += best_length
            remaining = tokens[best_length:]
            self._last_match_type = "prefix"
            self._last_matched_key = best_match.tokens
            cache_out = (
                _dequantize_cache(best_match.cache)
                if self._config.kv_quantize
                else best_match.cache
            )
            return cache_out, remaining

        # --- LCP (Longest Common Prefix) for divergent sequences ---
        # This handles the agentic pattern: same system+context prefix
        # but different final user message.  Use the sorted index to find
        # the nearest neighbor which likely shares the longest prefix.
        best_lcp_entry: _CacheEntry | None = None
        best_lcp_length = 0

        if sorted_keys:
            idx = bisect.bisect_left(sorted_keys, tokens_key)
            # Check neighbors around insertion point (they share the most
            # common prefix due to lexicographic ordering).
            for i in (idx - 1, idx):
                if i < 0 or i >= len(sorted_keys):
                    continue
                cached_key = sorted_keys[i]
                if cached_key == tokens_key:
                    continue  # Skip exact (already handled)
                min_len = min(len(cached_key), len(tokens_key))
                if min_len <= best_lcp_length:
                    continue
                # Compute LCP length
                lcp = 0
                for j in range(min_len):
                    if cached_key[j] != tokens_key[j]:
                        break
                    lcp = j + 1
                if lcp > best_lcp_length:
                    best_lcp_entry = self._entries[cached_key]
                    best_lcp_length = lcp
                    logger.debug(
                        f"[cache_fetch] LCP scan: cached_len={len(cached_key)} "
                        f"req_len={len(tokens_key)} lcp={lcp}"
                    )

        if best_lcp_entry is not None and best_lcp_length > 0:
            if best_lcp_length < self._config.min_prefix_tokens:
                logger.debug(
                    "[cache_fetch] LCP skipped: shared=%s below min_prefix_tokens=%s",
                    best_lcp_length,
                    self._config.min_prefix_tokens,
                )
                self._stats.misses += 1
                self._last_match_type = "miss_short_lcp"
                return None, tokens
            excess = len(best_lcp_entry.tokens) - best_lcp_length

            has_non_trimmable = any(
                not _is_cache_layer_trimmable(lc) for lc in best_lcp_entry.cache
            )
            logger.debug(
                f"[cache_fetch] LCP candidate: lcp={best_lcp_length} "
                f"entry_len={len(best_lcp_entry.tokens)} excess={excess} "
                f"non_trimmable={has_non_trimmable} "
                f"cache_layers={len(best_lcp_entry.cache)} "
                f"layer_types={[type(lc).__name__ for lc in best_lcp_entry.cache[:3]]}"
            )

            if has_non_trimmable:
                # Hybrid model (SSM+Attention): SSM state can't be rewound.
                # Block LCP for hybrid models — use think-suffix stripping
                # in the engine layer to get clean PREFIX matches instead.
                logger.debug(
                    "[cache_fetch] LCP skipped: non-trimmable cache layers "
                    "(hybrid model, SSM state can't be rewound)"
                )
            else:
                trimmed_cache = _trim_cache_offset(best_lcp_entry.cache, excess)
                self._entries.move_to_end(best_lcp_entry.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += best_lcp_length
                remaining = tokens[best_lcp_length:]
                logger.debug(
                    f"[cache_fetch] LCP hit: shared={best_lcp_length} "
                    f"trimmed={excess} remaining={len(remaining)}"
                )
                self._last_match_type = "lcp"
                self._last_matched_key = best_lcp_entry.tokens
                trimmed_cache = (
                    _dequantize_cache(trimmed_cache)
                    if self._config.kv_quantize
                    else trimmed_cache
                )
                return trimmed_cache, remaining

        self._stats.misses += 1
        self._last_match_type = "miss"

        return None, tokens

    def bind_owner_context(
        self, context: VerifiedCacheOwnerContext
    ) -> ModelCacheOwnerBinding:
        """Bind this live owner to governed identity resolved by Runtime.

        The stable persistence identity is compatibility evidence.  The
        returned opaque handle is the separate process-local authority.
        """

        if any(not value for value in self._persistence_identity.values()):
            raise ValueError("complete model/tokenizer/cache provenance is required")
        if context is not self._cache_owner_context:
            raise ValueError("cache owner context does not belong to this cache")
        if dict(context.persistence_identity) != self._persistence_identity:
            raise ValueError("verified cache owner provenance does not match cache")
        if self._owner_identity is None:
            self._owner_identity = CacheOwnerIdentity(context=context)
        binding = self._owner_identity.mint_owner_binding()
        if not self._owner_identity.matches_verified_context(context):
            raise ValueError("verified context does not match bound owner")
        return binding

    def mint_owner_request(self, sequence_revision: int) -> ModelCacheRequestBinding:
        if self._owner_identity is None:
            raise RuntimeError("model/cache owner identity is not bound")
        return self._owner_identity.mint_request_binding(sequence_revision)

    def validate_owner_request(
        self, binding: ModelCacheRequestBinding | Any
    ) -> OwnerBindingDecision:
        if self._owner_identity is None:
            return OwnerBindingDecision(False, "cache_unsafe")
        return self._owner_identity.validate_request_binding(binding)

    def cancel_owner_request(
        self, binding: ModelCacheRequestBinding
    ) -> OwnerBindingDecision:
        if self._owner_identity is None:
            return OwnerBindingDecision(False, "cache_unsafe")
        return self._owner_identity.cancel_request(binding)

    def release_owner_request(
        self, binding: ModelCacheRequestBinding
    ) -> OwnerBindingDecision:
        if self._owner_identity is None:
            return OwnerBindingDecision(False, "cache_unsafe")
        decision = self._owner_identity.release_request(binding)
        with self._memory_lock:
            prepared_entries = getattr(self, "_owner_prepared_entries", {})
            stale = [
                handle
                for handle, (prepared, _entry) in prepared_entries.items()
                if prepared.request is binding
            ]
            for handle in stale:
                prepared_entries.pop(handle, None)
        return decision

    def invalidate_owner_identity(self, *, cache_namespace: str | None = None) -> None:
        if self._owner_identity is not None:
            self._owner_identity.invalidate(cache_namespace=cache_namespace)
        with self._memory_lock:
            getattr(self, "_owner_prepared_entries", {}).clear()

    def close_owner_identity(self) -> None:
        if self._owner_identity is not None:
            self._owner_identity.close()
        with self._memory_lock:
            getattr(self, "_owner_prepared_entries", {}).clear()

    def prepare_owner_bound_store(
        self,
        request_binding: ModelCacheRequestBinding,
        tokens: list[int],
        cache: list[Any],
        auxiliary: dict[str, Any] | None = None,
        *,
        persistence_eligible: bool = False,
    ) -> tuple[OwnerBindingDecision, PreparedOwnerBoundCacheEntry | None]:
        """Prepare private state only for a currently owned request."""

        decision = self.validate_owner_request(request_binding)
        if not decision.accepted:
            return decision, None
        entry = self._prepare_store_unchecked(
            tokens,
            cache,
            auxiliary,
            persistence_eligible=persistence_eligible,
        )
        if entry is None:
            return decision, None
        prepared = PreparedOwnerBoundCacheEntry(
            owner=request_binding.owner,
            request=request_binding,
            handle_id=uuid.uuid4().hex,
            tokens=tuple(entry.tokens),
            memory_bytes=int(entry.memory_bytes),
        )

        def register() -> bool:
            with self._memory_lock:
                self._owner_prepared_entries[prepared.handle_id] = (prepared, entry)
            return True

        owner = self._owner_identity
        if owner is None:
            return OwnerBindingDecision(False, "cache_unsafe"), None
        decision = owner._commit_request(request_binding, register)
        if not decision.accepted:
            with self._memory_lock:
                self._owner_prepared_entries.pop(prepared.handle_id, None)
            return decision, None
        return decision, prepared

    def _resolve_owner_prepared_entry(
        self, prepared: PreparedOwnerBoundCacheEntry
    ) -> _CacheEntry | None:
        if not isinstance(prepared, PreparedOwnerBoundCacheEntry):
            return None
        with self._memory_lock:
            record = self._owner_prepared_entries.get(prepared.handle_id)
        if record is None or record[0] is not prepared:
            return None
        return record[1]

    def clone_prepared_owner_bound_cache(
        self,
        prepared: PreparedOwnerBoundCacheEntry,
        cloner: Callable[[Any], Any],
    ) -> tuple[OwnerBindingDecision, Any | None]:
        """Clone prepared replay state only while its request lease is live."""

        if not isinstance(prepared, PreparedOwnerBoundCacheEntry):
            return OwnerBindingDecision(False, "runtime_error"), None
        decision = self.validate_owner_request(prepared.request)
        if not decision.accepted or prepared.owner is not prepared.request.owner:
            return decision, None
        entry = self._resolve_owner_prepared_entry(prepared)
        if entry is None:
            return OwnerBindingDecision(False, "runtime_error"), None
        try:
            cloned = cloner(entry.cache)
        except Exception:
            return OwnerBindingDecision(False, "runtime_error"), None
        decision = self.validate_owner_request(prepared.request)
        return (decision, cloned if decision.accepted else None)

    def fetch_owner_bound(
        self,
        request_binding: ModelCacheRequestBinding,
        tokens: list[int],
    ) -> tuple[OwnerBindingDecision, Any | None, list[int]]:
        """Fetch shared state only for a live request owned by this cache."""

        decision = self.validate_owner_request(request_binding)
        if not decision.accepted:
            return decision, None, tokens
        cache, remaining = self._fetch_unchecked(tokens)
        decision = self.validate_owner_request(request_binding)
        if not decision.accepted:
            return decision, None, tokens
        return decision, cache, remaining

    def clone_owner_bound_for_replay(
        self,
        request_binding: ModelCacheRequestBinding,
        cache: list[Any],
    ) -> tuple[OwnerBindingDecision, list[Any] | None]:
        """Clone fetched state only while its request lease remains live."""

        decision = self.validate_owner_request(request_binding)
        if not decision.accepted:
            return decision, None
        cloned = self._clone_for_replay_unchecked(cache)
        decision = self.validate_owner_request(request_binding)
        return (decision, cloned if decision.accepted else None)

    def clone_for_replay(self, cache: list[Any]) -> list[Any] | None:
        """Preserve legacy replay cloning only for non-owner-bound caches."""
        if self._cache_owner_context is not None:
            return None
        return self._clone_for_replay_unchecked(cache)

    def fetch_exact_auxiliary_owner_bound(
        self,
        request_binding: ModelCacheRequestBinding,
        tokens: list[int],
    ) -> tuple[OwnerBindingDecision, dict[str, Any] | None]:
        """Fetch exact-match auxiliary state for a live owned request."""

        decision = self.validate_owner_request(request_binding)
        if not decision.accepted:
            return decision, None
        auxiliary = self._fetch_exact_auxiliary_unchecked(tokens)
        decision = self.validate_owner_request(request_binding)
        return (decision, auxiliary if decision.accepted else None)

    def commit_owner_bound_store(
        self,
        prepared: PreparedOwnerBoundCacheEntry,
        *,
        evict_prefixes: bool = True,
        commit_lock: Any = None,
        commit_guard: Callable[[], bool] | None = None,
    ) -> OwnerBindingDecision:
        """Publish only while owner, namespace, epoch, and request remain live."""

        if (
            not isinstance(prepared, PreparedOwnerBoundCacheEntry)
            or prepared.version != PREPARED_STORE_VERSION
        ):
            return OwnerBindingDecision(False, "runtime_error")
        owner_decision = (
            self._owner_identity.validate_owner_binding(prepared.owner)
            if self._owner_identity is not None
            else OwnerBindingDecision(False, "cache_unsafe")
        )
        if not owner_decision.accepted or prepared.owner is not prepared.request.owner:
            return OwnerBindingDecision(False, "cache_unsafe")
        decision = self.validate_owner_request(prepared.request)
        if not decision.accepted:
            return decision
        entry = self._resolve_owner_prepared_entry(prepared)
        if entry is None:
            return OwnerBindingDecision(False, "runtime_error")

        def _owned_commit_allowed() -> bool:
            if commit_guard is not None and not bool(commit_guard()):
                return False
            return self.validate_owner_request(prepared.request).accepted

        if self._owner_identity is None:
            return OwnerBindingDecision(False, "cache_unsafe")
        commit_context = commit_lock if commit_lock is not None else nullcontext()
        with commit_context:
            result = self._owner_identity._commit_request(
                prepared.request,
                lambda: self._commit_prepared_unchecked(
                    entry,
                    evict_prefixes=evict_prefixes,
                    commit_guard=_owned_commit_allowed,
                ),
            )
        with self._memory_lock:
            getattr(self, "_owner_prepared_entries", {}).pop(prepared.handle_id, None)
        return result

    def prepare_store(
        self,
        tokens: list[int],
        cache: list[Any],
        auxiliary: dict[str, Any] | None = None,
        *,
        persistence_eligible: bool = False,
    ) -> _CacheEntry | None:
        """Prepare through the legacy API only when no owner is bound."""

        if self._owner_identity is not None:
            raise RuntimeError("owner-bound cache store requires a request lease")
        return self._prepare_store_unchecked(
            tokens,
            cache,
            auxiliary,
            persistence_eligible=persistence_eligible,
        )

    def _prepare_store_unchecked(
        self,
        tokens: list[int],
        cache: list[Any],
        auxiliary: dict[str, Any] | None = None,
        *,
        persistence_eligible: bool = False,
    ) -> _CacheEntry | None:
        """Detach and account an entry without publishing it.

        Hybrid recurrent caches cannot be rewound after generation starts,
        while ordinary ``KVCache`` buffers are updated in place.  Callers can
        therefore prepare an immutable entry at the exact prefill boundary,
        continue the request, and publish that already-detached entry only
        after the remaining prefill succeeds.
        """
        if not tokens or not cache:
            return None
        if len(tokens) < self._config.min_prefix_tokens:
            logger.debug(
                "[cache_store] skipped short prefix: tokens=%s min_prefix_tokens=%s",
                len(tokens),
                self._config.min_prefix_tokens,
            )
            return None

        tokens_key = tuple(tokens)
        with self._memory_lock:
            existing = self._entries.get(tokens_key)
            if existing is not None:
                return existing

        try:
            cache = _trim_to_offset(cache)
            if (
                self._config.kv_quantize
                and len(tokens) >= self._config.kv_min_quantize_tokens
            ):
                cache = _quantize_cache(
                    cache, self._config.kv_bits, self._config.kv_group_size
                )
            projected_bytes = estimate_kv_cache_memory(cache)
            if projected_bytes > self._max_memory:
                self._stats.store_rejections += 1
                logger.warning(
                    f"Cache entry too large: "
                    f"{projected_bytes / _BYTES_PER_MB:.1f}MB "
                    f"exceeds limit {self._max_memory / _BYTES_PER_MB:.1f}MB"
                )
                return None

            with self._copy_lock:
                cache = _detach_cache_for_storage(cache)
                detached_auxiliary = None
                if auxiliary:
                    import mlx.core as mx

                    detached_auxiliary = {
                        key: value + 0 if isinstance(value, mx.array) else value
                        for key, value in auxiliary.items()
                    }
                    mx.eval(
                        *(
                            value
                            for value in detached_auxiliary.values()
                            if isinstance(value, mx.array)
                        )
                    )
            entry = _CacheEntry.create(
                tokens,
                cache,
                detached_auxiliary,
                persistence_eligible=persistence_eligible,
            )
            if entry.memory_bytes > self._max_memory:
                self._stats.store_rejections += 1
                logger.warning(
                    "Cache entry too large after auxiliary accounting: "
                    "%.1fMB exceeds limit %.1fMB",
                    entry.memory_bytes / _BYTES_PER_MB,
                    self._max_memory / _BYTES_PER_MB,
                )
                return None
            return entry
        except UndetachableCacheError as e:
            self._stats.store_rejections += 1
            logger.warning("[cache_store] rejecting entry: %s", e)
            return None
        except Exception as e:
            self._stats.store_rejections += 1
            logger.warning("[cache_store] rejecting entry: %s: %s", type(e).__name__, e)
            return None

    def commit_prepared(
        self,
        entry: _CacheEntry,
        evict_prefixes: bool = True,
        *,
        commit_lock: Any = None,
        commit_guard: Callable[[], bool] | None = None,
    ) -> bool:
        """Commit through the legacy API only when no owner is bound."""

        if self._owner_identity is not None:
            raise RuntimeError("owner-bound cache commit requires a request lease")
        return self._commit_prepared_unchecked(
            entry,
            evict_prefixes=evict_prefixes,
            commit_lock=commit_lock,
            commit_guard=commit_guard,
        )

    def _commit_prepared_unchecked(
        self,
        entry: _CacheEntry,
        evict_prefixes: bool = True,
        *,
        commit_lock: Any = None,
        commit_guard: Callable[[], bool] | None = None,
    ) -> bool:
        """Atomically publish an entry returned by :meth:`prepare_store`."""
        tokens_key = entry.tokens
        commit_context = commit_lock if commit_lock is not None else nullcontext()
        try:
            with commit_context, self._memory_lock:
                if commit_guard is not None and not commit_guard():
                    return False
                if entry.memory_bytes > self._max_memory:
                    self._stats.store_rejections += 1
                    logger.warning(
                        "[cache_store] rejecting oversized prepared entry: "
                        "%.1fMB exceeds limit %.1fMB",
                        entry.memory_bytes / _BYTES_PER_MB,
                        self._max_memory / _BYTES_PER_MB,
                    )
                    return False
                if tokens_key in self._entries:
                    self._entries.move_to_end(tokens_key)
                    return True

                if evict_prefixes and self._sorted_keys:
                    to_remove = []
                    idx = bisect.bisect_left(self._sorted_keys, tokens_key)
                    for i in range(idx - 1, -1, -1):
                        key = self._sorted_keys[i]
                        klen = len(key)
                        if klen >= len(tokens_key):
                            continue
                        if tokens_key[:klen] == key:
                            to_remove.append(key)
                        elif key[0] != tokens_key[0]:
                            break
                    for key in to_remove:
                        old = self._entries.pop(key)
                        self._current_memory -= old.memory_bytes
                        self._stats.evictions += 1
                        self._remove_from_sorted(key)
                        logger.debug(
                            "[prefix_evict] removed %s tokens, freed %.2fMB, "
                            "new_entry=%s tokens",
                            len(key),
                            old.memory_bytes / _BYTES_PER_MB,
                            len(tokens_key),
                        )

                while (
                    self._current_memory + entry.memory_bytes > self._max_memory
                    or len(self._entries) >= self._config.max_entries
                ) and self._entries:
                    self._evict_lru()

                # Eviction may call an external SSD tier. Recheck immediately
                # before publication in case that callback revoked ownership.
                if commit_guard is not None and not commit_guard():
                    return False
                self._entries[tokens_key] = entry
                self._current_memory += entry.memory_bytes
                bisect.insort(self._sorted_keys, tokens_key)
                self._stats.entry_count = len(self._entries)
                self._stats.current_memory_bytes = self._current_memory
        except Exception as e:
            self._stats.store_rejections += 1
            logger.warning(
                "[cache_store] rejecting commit: %s: %s", type(e).__name__, e
            )
            return False

        logger.debug(
            f"Stored cache: {len(tokens_key)} tokens, "
            f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
            f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
        )
        return True

    def _clone_for_replay_unchecked(self, cache: list[Any]) -> list[Any] | None:
        """Return independently owned backing for a mutating model replay."""
        try:
            with self._copy_lock:
                return _detach_cache_for_storage(cache)
        except Exception as exc:
            logger.warning(
                "[cache_fetch] replay clone rejected: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    def fetch_exact_auxiliary(self, tokens: list[int]) -> dict[str, Any] | None:
        """Fetch auxiliary state only when no owner is bound."""

        if self._owner_identity is not None:
            raise RuntimeError("owner-bound auxiliary fetch requires a request lease")
        return self._fetch_exact_auxiliary_unchecked(tokens)

    def _fetch_exact_auxiliary_unchecked(
        self, tokens: list[int]
    ) -> dict[str, Any] | None:
        """Return auxiliary data only when an exact resident entry owns it."""
        with self._memory_lock:
            entry = self._entries.get(tuple(tokens))
            if entry is None or entry.auxiliary is None:
                return None
            return dict(entry.auxiliary)

    def store(
        self,
        tokens: list[int],
        cache: list[Any],
        evict_prefixes: bool = True,
    ) -> bool:
        """Detach, account, and atomically publish a reusable cache entry."""
        if self._owner_identity is not None:
            raise RuntimeError("owner-bound cache store requires a request lease")
        entry = self.prepare_store(tokens, cache)
        if entry is None:
            return False
        return self.commit_prepared(entry, evict_prefixes=evict_prefixes)

    def _remove_from_sorted(self, key: tuple[int, ...]) -> None:
        """Remove a key from the sorted index using bisect for O(log N)."""
        idx = bisect.bisect_left(self._sorted_keys, key)
        if idx < len(self._sorted_keys) and self._sorted_keys[idx] == key:
            self._sorted_keys.pop(idx)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry.

        If an SSD tier is attached, the entry is spilled to disk instead
        of being discarded.
        """
        with self._memory_lock:
            if not self._entries:
                return

            # popitem(last=False) removes oldest entry (FIFO order = LRU)
            tokens_key, entry = self._entries.popitem(last=False)
            self._current_memory -= entry.memory_bytes
            self._remove_from_sorted(tokens_key)
            self._stats.evictions += 1
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        # Spill to SSD tier if available
        if self._ssd_tier is not None:
            self._ssd_tier.enqueue_spill(tokens_key, entry.cache, entry.memory_bytes)

        logger.debug(
            f"[lru_evict] removed {len(tokens_key)} tokens, "
            f"freed {entry.memory_bytes / _BYTES_PER_MB:.2f}MB"
            f"{'  (spilled to SSD)' if self._ssd_tier is not None else ''}"
        )

    def remove(self, tokens: list[int], *, include_ssd: bool = False) -> bool:
        """
        Remove a specific cache entry.

        Args:
            tokens: Token sequence to remove.
            include_ssd: Also remove the same entry from the attached SSD tier.

        Returns:
            True if entry was found and removed.
        """
        tokens_key = tuple(tokens)
        removed = False
        with self._memory_lock:
            entry = self._entries.pop(tokens_key, None)
            if entry is not None:
                self._current_memory -= entry.memory_bytes
                self._remove_from_sorted(tokens_key)
                self._stats.entry_count = len(self._entries)
                self._stats.current_memory_bytes = self._current_memory
                removed = True

        if include_ssd and self._ssd_tier is not None:
            removed = self._ssd_tier.remove(tokens_key) or removed

        return removed

    def clear(self) -> None:
        """Clear all cached entries."""
        self.invalidate_owner_identity()
        with self._memory_lock:
            self._entries.clear()
            self._sorted_keys.clear()
            self._current_memory = 0
            self._stats = CacheStats(max_memory_bytes=self._max_memory)
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.to_dict()
        ssd_stats = self.get_ssd_stats()
        if ssd_stats is not None:
            stats["ssd"] = ssd_stats
        return stats

    def get_ssd_stats(self) -> dict[str, Any] | None:
        """Return attached SSD statistics without exposing tier internals."""
        tier = self._ssd_tier
        get_stats = getattr(tier, "get_stats", None) if tier is not None else None
        return get_stats() if callable(get_stats) else None

    def reset_stats(self) -> None:
        """Reset statistics while preserving cache contents."""
        with self._memory_lock:
            self._stats = CacheStats(
                max_memory_bytes=self._max_memory,
                current_memory_bytes=self._current_memory,
                entry_count=len(self._entries),
            )

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory / _BYTES_PER_MB

    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self._max_memory / _BYTES_PER_MB

    def try_reserve_memory(self, nbytes: int) -> bool:
        """Tentatively reserve cache memory for an upcoming promotion."""
        with self._memory_lock:
            if self._current_memory + nbytes > self._max_memory:
                return False
            self._current_memory += nbytes
            self._stats.current_memory_bytes = self._current_memory
            return True

    def release_reserved_memory(self, nbytes: int) -> None:
        """Release memory previously reserved by try_reserve_memory()."""
        with self._memory_lock:
            self._current_memory = max(0, self._current_memory - nbytes)
            self._stats.current_memory_bytes = self._current_memory

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def __contains__(self, tokens: list[int]) -> bool:
        """Check if tokens are cached."""
        return tuple(tokens) in self._entries

    def set_ssd_tier(self, ssd_tier) -> None:
        """Attach an SSD cache tier for eviction spilling.

        When set, evicted entries are spilled to SSD instead of discarded.

        Args:
            ssd_tier: An SSDCacheTier instance (or None to disable).
        """
        if ssd_tier is not None and self._cache_owner_context is not None:
            bind_identity = getattr(ssd_tier, "bind_persistence_identity", None)
            if not callable(bind_identity):
                raise RuntimeError("owner-bound SSD tier lacks identity binding")
            bind_identity(self._persistence_identity)
        self._ssd_tier = ssd_tier
        if ssd_tier is not None:
            logger.info("[memory_cache] SSD tier attached for eviction spilling")

    def get_ssd_tier(self):
        """Return the attached SSD tier through the cache's public seam."""
        return self._ssd_tier

    def check_ssd(self, tokens: list[int]) -> dict | None:
        """Check if tokens have an SSD cache hit (without reading data).

        Returns metadata dict with 'match_type' ('exact' or 'prefix') if
        found in SSD tier, None if not found. For prefix matches, the dict
        also includes 'matched_tokens' (the count of tokens the SSD entry
        covers).

        This is a fast synchronous call (SQLite lookup only).
        The actual data read happens via the scheduler handoff.
        """
        if self._ssd_tier is None:
            return None

        tokens_key = tuple(tokens)

        # If already in RAM, no SSD needed
        if tokens_key in self._entries:
            return None

        # Check SSD tier — exact match first, then prefix. The tier owns
        # lookup accounting so a miss is counted once after both probes.
        # Inspect the type so dynamic mocks/proxies cannot manufacture the
        # new API and bypass the compatibility path.
        lookup_candidate_impl = getattr(type(self._ssd_tier), "lookup_candidate", None)
        if callable(lookup_candidate_impl):
            lookup_candidate = self._ssd_tier.lookup_candidate
            candidate = lookup_candidate(tokens_key)
            validate_candidate = getattr(
                type(self._ssd_tier), "validate_candidate", None
            )
            if candidate is not None and callable(validate_candidate):
                if self._ssd_tier.validate_candidate(tokens_key, candidate) is None:
                    record_failure = getattr(
                        self._ssd_tier, "record_promotion_failure", None
                    )
                    if callable(record_failure):
                        record_failure()
                    return None
            return candidate

        # Compatibility fallback for lightweight legacy test doubles.
        candidate = self._ssd_tier.lookup_ssd(tokens_key)
        if candidate is not None:
            candidate["match_type"] = "exact"
            candidate["matched_tokens"] = len(tokens)
            candidate["matched_key"] = tokens_key
            return candidate

        prefix = self._ssd_tier.lookup_ssd_prefix(tokens_key)
        if prefix is not None:
            prefix["match_type"] = "prefix"
            prefix["matched_tokens"] = prefix["num_tokens"]
            prefix["matched_key"] = tokens_key[: prefix["num_tokens"]]
            return prefix

        return None

    # -----------------------------------------------------------------
    # Exact hybrid restart persistence
    # -----------------------------------------------------------------

    def prepare_hybrid_persistence_snapshot(self):
        """Materialize a stable numpy snapshot on the model-owning thread.

        This deliberately accepts only the Package 1 hybrid contract:
        ordered ``ArraysCache``/``KVCache`` layers plus exact-hit
        ``last_logits``. Unsupported or incomplete entries reject the whole
        snapshot rather than silently degrading a restored hit to prefill.
        """
        from .cache_persistence import (
            HybridCacheSnapshot,
            HybridEntrySnapshot,
            _entry_logical_nbytes,
        )

        if any(not value for value in self._persistence_identity.values()):
            logger.warning(
                "[cache_persist] hybrid snapshot rejected: model, tokenizer, "
                "and cache-layout identity are required"
            )
            return None
        with self._memory_lock:
            entries = [
                entry for entry in self._entries.values() if entry.persistence_eligible
            ]
        if not entries:
            return None

        expected_layout = self._persistence_identity["cache_layout"]
        snapshots = []
        try:
            for entry in entries:
                if set(entry.auxiliary or {}) != {"last_logits"}:
                    raise ValueError("hybrid entry is missing exact last_logits")
                persist_cache = (
                    _dequantize_cache(entry.cache)
                    if self._config.kv_quantize
                    else entry.cache
                )
                topology = tuple(
                    _cache_layer_topology(layer) for layer in persist_cache
                )
                if any(layer is None for layer in topology):
                    raise ValueError("cache layout contains an unsupported layer")
                if (
                    _cache_topology_fingerprint(
                        topology, self._config, self._cache_runtime_identity
                    )
                    != expected_layout
                ):
                    raise ValueError("cache layout differs from the loaded model")
                layers = tuple(_snapshot_hybrid_layer(layer) for layer in persist_cache)
                last_logits, logits_dtype = _mx_array_to_numpy(
                    entry.auxiliary["last_logits"]
                )
                auxiliary_snapshot = {"last_logits": last_logits}
                auxiliary_dtypes = {"last_logits": logits_dtype}
                snapshots.append(
                    HybridEntrySnapshot(
                        tokens=entry.tokens,
                        memory_bytes=_entry_logical_nbytes(
                            layers, auxiliary_snapshot, auxiliary_dtypes
                        ),
                        layers=layers,
                        auxiliary=auxiliary_snapshot,
                        auxiliary_original_dtypes=auxiliary_dtypes,
                    )
                )
            return HybridCacheSnapshot(
                identity=dict(self._persistence_identity),
                entries=tuple(snapshots),
            )
        except Exception as exc:
            logger.warning(
                "[cache_persist] hybrid snapshot rejected: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    def write_hybrid_persistence_snapshot(self, cache_dir: str, snapshot) -> bool:
        """Write a prepared numpy snapshot; safe to call on an I/O thread."""
        from .cache_persistence import write_hybrid_snapshot

        if snapshot is None:
            return False
        try:
            return write_hybrid_snapshot(cache_dir, snapshot)
        except Exception as exc:
            logger.warning(
                "[cache_persist] hybrid write rejected: %s: %s",
                type(exc).__name__,
                exc,
            )
            return False

    def read_hybrid_persistence_snapshot(self, cache_dir: str):
        """Read and validate numpy state; safe to call on an I/O thread."""
        from .cache_persistence import read_hybrid_snapshot

        try:
            return read_hybrid_snapshot(
                cache_dir,
                max_memory_bytes=self._max_memory,
                max_entries=self._config.max_entries,
                max_tokens=self._max_persisted_tokens,
            )
        except Exception as exc:
            logger.warning(
                "[cache_persist] hybrid load rejected: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    @staticmethod
    def _restore_hybrid_layer(layer_snapshot, expected_layer: Any):
        """Reconstruct one validated layer on the model-owning thread."""
        import mlx.core as mx

        layer_type = layer_snapshot.layer_type
        tensors = layer_snapshot.tensors
        metadata = layer_snapshot.metadata

        cls = type(expected_layer)
        if cls.__name__ != layer_type:
            raise ValueError("cache layer implementation mismatch")

        def restore_dtype(value, dtype_name):
            dtype = getattr(mx, dtype_name, None) if dtype_name else None
            return value.astype(dtype) if dtype is not None else value

        if layer_type in {"KVCache", "RotatingKVCache"}:
            allowed_metadata = {
                "offset",
                "keys_original_dtype",
                "values_original_dtype",
                "max_size",
                "keep",
                "step",
                "_idx",
            }
            if not set(metadata).issubset(allowed_metadata) or "offset" not in metadata:
                raise ValueError("KV layer metadata is invalid")
            if set(tensors) != {"keys", "values"}:
                raise ValueError("KV layer tensor names are invalid")
            keys_np = tensors["keys"]
            values_np = tensors["values"]
            if (
                keys_np.shape != values_np.shape
                or keys_np.ndim < 3
                or keys_np.dtype.kind != "f"
                or values_np.dtype.kind != "f"
            ):
                raise ValueError("KV layer shape or dtype is invalid")
            offset = metadata.get("offset")
            if not isinstance(offset, int) or offset < 1 or offset > keys_np.shape[-2]:
                raise ValueError("KV layer offset is invalid")
            layer = cls.__new__(cls)
            layer.keys = restore_dtype(
                mx.array(keys_np), metadata.get("keys_original_dtype")
            )
            layer.values = restore_dtype(
                mx.array(values_np), metadata.get("values_original_dtype")
            )
            layer.offset = offset
            for attr in ("max_size", "keep", "step", "_idx"):
                if attr in metadata:
                    setattr(layer, attr, metadata[attr])
            return layer

        if layer_type == "ArraysCache":
            if set(metadata) != {
                "num_arrays",
                "state_original_dtypes",
                "metadata_arrays",
                "metadata_original_dtypes",
                "meta_state",
            }:
                raise ValueError("ArraysCache metadata is invalid")
            num_arrays = metadata.get("num_arrays")
            if not isinstance(num_arrays, int) or num_arrays < 1:
                raise ValueError("ArraysCache arity is invalid")
            metadata_arrays = metadata.get("metadata_arrays")
            if (
                not isinstance(metadata_arrays, list)
                or len(set(metadata_arrays)) != len(metadata_arrays)
                or any(
                    name not in {"left_padding", "lengths"} for name in metadata_arrays
                )
            ):
                raise ValueError("ArraysCache metadata arrays are invalid")
            expected_names = {f"state_{index}" for index in range(num_arrays)} | set(
                metadata_arrays
            )
            if set(tensors) != expected_names:
                raise ValueError("ArraysCache tensor names are invalid")
            dtype_names = metadata.get("state_original_dtypes")
            if not isinstance(dtype_names, list) or len(dtype_names) != num_arrays:
                raise ValueError("ArraysCache dtype metadata is invalid")
            state = []
            for index in range(num_arrays):
                value = tensors[f"state_{index}"]
                allowed_kinds = {"f"}
                if num_arrays == 4 and index == 3:
                    allowed_kinds.add("i")
                if value.size == 0 or value.dtype.kind not in allowed_kinds:
                    raise ValueError("ArraysCache tensor is invalid")
                state.append(restore_dtype(mx.array(value), dtype_names[index]))
            layer = cls(num_arrays)
            layer.state = state
            metadata_dtype_names = metadata.get("metadata_original_dtypes")
            if not isinstance(metadata_dtype_names, dict) or set(
                metadata_dtype_names
            ) != set(metadata_arrays):
                raise ValueError("ArraysCache metadata dtype information is invalid")
            for attr in metadata_arrays:
                setattr(
                    layer,
                    attr,
                    restore_dtype(mx.array(tensors[attr]), metadata_dtype_names[attr]),
                )
            layer.meta_state = metadata["meta_state"]
            return layer
        if layer_type == "QSAKVCache":
            if (
                not getattr(expected_layer, "preserve_auxiliary_kv_state", False)
                or f"{cls.__module__}.{cls.__qualname__}"
                != "mlx_vlm.models.qwen4_exp.language.QSAKVCache"
                or set(metadata)
                != {"num_arrays", "state_original_dtypes", "state_container"}
                or metadata.get("num_arrays") != 4
                or metadata.get("state_container") != "tuple"
            ):
                raise ValueError("QSAKVCache restore contract is invalid")
            dtype_names = metadata.get("state_original_dtypes")
            if not isinstance(dtype_names, list) or len(dtype_names) != 4:
                raise ValueError("QSAKVCache dtype metadata is invalid")
            if set(tensors) != {f"state_{index}" for index in range(4)}:
                raise ValueError("QSAKVCache tensor names are invalid")
            state = tuple(
                restore_dtype(mx.array(tensors[f"state_{index}"]), dtype_names[index])
                for index in range(4)
            )
            layer = cls()
            layer.state = state
            return layer
        raise ValueError(f"unsupported cache layer: {layer_type}")

    def restore_hybrid_persistence_snapshot(self, loaded) -> int:
        """Reconstruct and atomically publish validated entries on owner thread."""
        self.invalidate_owner_identity()
        if loaded is None:
            return 0
        if loaded.identity != self._persistence_identity:
            logger.warning("[cache_persist] hybrid identity mismatch; discarding cache")
            return 0

        expected_qualified_topology = _cache_topology(self._model) or ()
        make_cache = getattr(self._model, "make_cache", None)
        if not callable(make_cache):
            return 0
        expected_topology = tuple(
            (qualified_name.rsplit(".", 1)[-1], arity)
            for qualified_name, arity in expected_qualified_topology
        )
        cfg = _model_text_config(self._model)
        vocab_size = _identity_attr(cfg or self._model, "vocab_size")
        candidates = []
        seen_tokens = set()
        try:
            for persisted in loaded.entries:
                if persisted.tokens in seen_tokens or any(
                    token < 0 for token in persisted.tokens
                ):
                    raise ValueError("duplicate or invalid token sequence")
                seen_tokens.add(persisted.tokens)
                topology = tuple(
                    (
                        layer.layer_type,
                        int(layer.metadata.get("num_arrays", 2)),
                    )
                    for layer in persisted.layers
                )
                if topology != expected_topology:
                    raise ValueError("persisted cache topology mismatch")
                expected_kv_heads = _identity_attr(
                    cfg or self._model, "num_key_value_heads"
                )
                expected_head_dim = _identity_attr(cfg or self._model, "head_dim")
                for layer in persisted.layers:
                    if layer.layer_type not in {"KVCache", "RotatingKVCache"}:
                        continue
                    shape = layer.tensors["keys"].shape
                    if layer.metadata.get("offset") != len(persisted.tokens):
                        raise ValueError("persisted KV/token length mismatch")
                    if shape[0] != 1:
                        raise ValueError("persisted KV batch geometry mismatch")
                    if (
                        isinstance(expected_kv_heads, int)
                        and shape[-3] != expected_kv_heads
                    ):
                        raise ValueError("persisted KV head geometry mismatch")
                    if (
                        isinstance(expected_head_dim, int)
                        and shape[-1] != expected_head_dim
                    ):
                        raise ValueError("persisted KV dimension mismatch")
                qsa_layers = [
                    layer
                    for layer in persisted.layers
                    if layer.layer_type == "QSAKVCache"
                ]
                for layer in qsa_layers:
                    keys = layer.tensors["state_0"]
                    index_keys = layer.tensors["state_2"]
                    position_ids = layer.tensors["state_3"]
                    if (
                        keys.shape[-2] != len(persisted.tokens)
                        or index_keys.shape[1] != len(persisted.tokens)
                        or position_ids.shape[-1] != len(persisted.tokens)
                    ):
                        raise ValueError("persisted QSA/token length mismatch")
                    if keys.shape[0] != 1:
                        raise ValueError("persisted QSA batch geometry mismatch")
                    if (
                        isinstance(expected_kv_heads, int)
                        and keys.shape[-3] != expected_kv_heads
                    ):
                        raise ValueError("persisted QSA head geometry mismatch")
                    if (
                        isinstance(expected_head_dim, int)
                        and keys.shape[-1] != expected_head_dim
                    ):
                        raise ValueError("persisted QSA dimension mismatch")
                arrays_layers = [
                    layer
                    for layer in persisted.layers
                    if layer.layer_type == "ArraysCache"
                ]
                if arrays_layers:
                    model_type = str(
                        _identity_attr(cfg or self._model, "model_type") or ""
                    ).replace(".", "_")
                    if not model_type.startswith(("qwen3_5", "qwen3_8", "qwen4_exp")):
                        raise ValueError("unsupported ArraysCache model geometry")
                    linear_k_heads = _identity_attr(
                        cfg or self._model, "linear_num_key_heads"
                    )
                    linear_v_heads = _identity_attr(
                        cfg or self._model, "linear_num_value_heads"
                    )
                    linear_k_dim = _identity_attr(
                        cfg or self._model, "linear_key_head_dim"
                    )
                    linear_v_dim = _identity_attr(
                        cfg or self._model, "linear_value_head_dim"
                    )
                    conv_kernel = _identity_attr(
                        cfg or self._model, "linear_conv_kernel_dim"
                    )
                    geometry = (
                        linear_k_heads,
                        linear_v_heads,
                        linear_k_dim,
                        linear_v_dim,
                        conv_kernel,
                    )
                    if not all(
                        isinstance(value, int) and value > 0 for value in geometry
                    ):
                        raise ValueError("missing ArraysCache model geometry")
                    conv_dim = (
                        2 * linear_k_heads * linear_k_dim
                        + linear_v_heads * linear_v_dim
                    )
                    expected_arrays_shapes = (
                        (1, conv_kernel - 1, conv_dim),
                        (1, linear_v_heads, linear_v_dim, linear_k_dim),
                    )
                    for layer in arrays_layers:
                        num_arrays = layer.metadata.get("num_arrays")
                        layer_shapes = expected_arrays_shapes
                        layer_kinds = ("f", "f")
                        if model_type.startswith("qwen4_exp") and num_arrays == 4:
                            ple_conv_kernel = _identity_attr(
                                cfg or self._model, "ple_conv_kernel_size"
                            )
                            ngram_size = _identity_attr(
                                cfg or self._model, "ngram_size"
                            )
                            hidden_size = _identity_attr(
                                cfg or self._model, "hidden_size"
                            )
                            hc_count = _identity_attr(cfg or self._model, "hc_count")
                            ple_geometry = (
                                ple_conv_kernel,
                                ngram_size,
                                hidden_size,
                                hc_count,
                            )
                            if not all(
                                isinstance(value, int) and value > 0
                                for value in ple_geometry
                            ):
                                raise ValueError(
                                    "missing PLE ArraysCache model geometry"
                                )
                            layer_shapes += (
                                (
                                    1,
                                    (ple_conv_kernel - 1) * ngram_size,
                                    hc_count * hidden_size,
                                ),
                                (1, ngram_size - 1),
                            )
                            layer_kinds += ("f", "i")
                        if num_arrays != len(layer_shapes):
                            raise ValueError("persisted ArraysCache arity mismatch")
                        values = tuple(
                            layer.tensors[f"state_{index}"]
                            for index in range(num_arrays)
                        )
                        if tuple(value.shape for value in values) != layer_shapes:
                            raise ValueError(
                                "persisted ArraysCache geometry mismatch: "
                                f"{tuple(value.shape for value in values)} != {layer_shapes}"
                            )
                        if tuple(value.dtype.kind for value in values) != layer_kinds:
                            raise ValueError("persisted ArraysCache dtype mismatch")
                logits_np = persisted.auxiliary.get("last_logits")
                if (
                    logits_np is None
                    or logits_np.size == 0
                    or logits_np.dtype.kind != "f"
                    or logits_np.ndim != 2
                    or logits_np.shape[0] != 1
                    or (
                        isinstance(vocab_size, int)
                        and logits_np.shape[-1] != vocab_size
                    )
                ):
                    raise ValueError("persisted last_logits shape or dtype is invalid")
                expected_cache = list(make_cache())
                if len(expected_cache) != len(persisted.layers):
                    raise ValueError("loaded cache topology changed during restore")
                cache = [
                    self._restore_hybrid_layer(layer, expected_cache[i])
                    for i, layer in enumerate(persisted.layers)
                ]
                if (
                    self._config.kv_quantize
                    and len(persisted.tokens) >= self._config.kv_min_quantize_tokens
                ):
                    cache = _quantize_cache(
                        cache, self._config.kv_bits, self._config.kv_group_size
                    )
                import mlx.core as mx

                logits = mx.array(logits_np)
                original_dtype = persisted.auxiliary_original_dtypes.get("last_logits")
                if original_dtype:
                    dtype = getattr(mx, original_dtype, None)
                    if dtype is None:
                        raise ValueError("persisted last_logits dtype is unavailable")
                    logits = logits.astype(dtype)
                auxiliary = {"last_logits": logits}
                mx.eval(
                    *(array for layer in cache for array in _iter_cache_arrays(layer)),
                    auxiliary["last_logits"],
                )
                candidates.append(
                    _CacheEntry.create(
                        list(persisted.tokens),
                        cache,
                        auxiliary,
                        persistence_eligible=True,
                    )
                )
        except Exception as exc:
            logger.warning(
                "[cache_persist] hybrid reconstruction rejected: %s: %s",
                type(exc).__name__,
                exc,
            )
            return 0

        loaded_count = 0
        with self._memory_lock:
            for entry in candidates:
                if entry.tokens in self._entries:
                    continue
                if (
                    self._current_memory + entry.memory_bytes > self._max_memory
                    or len(self._entries) >= self._config.max_entries
                ):
                    break
                self._entries[entry.tokens] = entry
                bisect.insort(self._sorted_keys, entry.tokens)
                self._current_memory += entry.memory_bytes
                loaded_count += 1
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
        return loaded_count

    # -----------------------------------------------------------------
    # Disk persistence — survives server restarts
    # -----------------------------------------------------------------

    def save_to_disk(self, cache_dir: str) -> bool:
        """Save all cache entries to disk using mlx_lm's safetensors format.

        Directory layout::

            cache_dir/
              index.json          # token keys + metadata per entry
              entry_0.safetensors # KV arrays for entry 0
              entry_1.safetensors
              ...

        Returns True if at least one entry was saved.
        """
        if self._cache_owner_context is not None:
            raise RuntimeError("owner-bound caches require strict hybrid persistence")
        import json
        import os
        import time as _time

        if not self._entries:
            logger.info("[cache_persist] nothing to save (0 entries)")
            return False

        t0 = _time.monotonic()
        os.makedirs(cache_dir, exist_ok=True)

        try:
            from mlx_lm.models.cache import save_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot save")
            return False

        index = {
            "version": _CACHE_PERSIST_VERSION,
            "model_fingerprint": self._model_fingerprint,
            "persistence_identity": (
                dict(self._persistence_identity)
                if all(self._persistence_identity.values())
                else None
            ),
            "num_entries": len(self._entries),
            "total_memory_bytes": self._current_memory,
            "entries": [],
        }

        saved = 0
        for i, (tokens_key, entry) in enumerate(self._entries.items()):
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            try:
                # Dequantize _QuantizedCacheWrapper layers before saving.
                # save_prompt_cache requires .state and .meta_state which
                # the wrapper does not provide; dequantizing restores the
                # original cache types that do.
                persist_cache = (
                    _dequantize_cache(entry.cache)
                    if any(isinstance(c, _QuantizedCacheWrapper) for c in entry.cache)
                    else entry.cache
                )
                save_prompt_cache(
                    entry_path,
                    persist_cache,
                    metadata={"num_tokens": str(len(tokens_key))},
                )
                # Save tokens separately (can be 100K+ ints → binary is smaller)
                tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")
                import array as _array

                arr = _array.array("i", tokens_key)  # 32-bit signed ints
                with open(tokens_path, "wb") as f:
                    arr.tofile(f)

                index["entries"].append(
                    {
                        "index": i,
                        "num_tokens": len(tokens_key),
                        "memory_bytes": entry.memory_bytes,
                    }
                )
                saved += 1
                logger.info(
                    f"[cache_persist] saved entry {i}: "
                    f"{len(tokens_key)} tokens, "
                    f"{entry.memory_bytes / _BYTES_PER_MB:.1f}MB KV, "
                    f"file={entry_path}"
                )
            except Exception as e:
                logger.warning(f"[cache_persist] failed to save entry {i}: {e}")

        index_path = os.path.join(cache_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] SAVED {saved}/{len(self._entries)} entries "
            f"to {cache_dir} in {dt:.1f}s "
            f"({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return saved > 0

    def load_from_disk(self, cache_dir: str) -> int:
        """Load cache entries from disk.

        Returns the number of entries successfully loaded.
        """
        if self._cache_owner_context is not None:
            raise RuntimeError("owner-bound caches require strict hybrid persistence")
        self.invalidate_owner_identity()
        import json
        import os
        import time as _time

        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            logger.info(f"[cache_persist] no index at {index_path}, nothing to load")
            return 0

        t0 = _time.monotonic()

        with open(index_path) as f:
            index = json.load(f)

        version = index.get("version", 1)
        if version != _CACHE_PERSIST_VERSION:
            logger.warning(
                f"[cache_persist] version mismatch: disk={version} "
                f"current={_CACHE_PERSIST_VERSION}, discarding stale cache"
            )
            return 0

        if self._cache_owner_context is not None:
            disk_identity = index.get("persistence_identity")
            if not isinstance(disk_identity, dict):
                logger.warning(
                    "[cache_persist] owner-bound cache rejects legacy weak identity"
                )
                return 0
            if disk_identity != self._persistence_identity:
                logger.warning(
                    "[cache_persist] strict persistence identity mismatch; "
                    "discarding incompatible cache"
                )
                return 0

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot load")
            return 0

        disk_fp = index.get("model_fingerprint", "")
        if disk_fp and disk_fp != self._model_fingerprint:
            logger.warning(
                f"[cache_persist] model fingerprint mismatch: "
                f"disk={disk_fp} current={self._model_fingerprint}, "
                f"discarding incompatible cache"
            )
            return 0
        loaded = 0
        for entry_meta in index.get("entries", []):
            i = entry_meta["index"]
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")

            if not os.path.exists(entry_path) or not os.path.exists(tokens_path):
                logger.warning(f"[cache_persist] missing files for entry {i}, skipping")
                continue

            try:
                # Load tokens from binary
                import array as _array

                arr = _array.array("i")
                with open(tokens_path, "rb") as f:
                    arr.fromfile(f, entry_meta["num_tokens"])
                tokens = list(arr)
                if len(tokens) < self._config.min_prefix_tokens:
                    logger.info(
                        "[cache_persist] skipping short entry %s: %s tokens < "
                        "min_prefix_tokens=%s",
                        i,
                        len(tokens),
                        self._config.min_prefix_tokens,
                    )
                    continue

                # Load KV cache
                cache = load_prompt_cache(entry_path)

                # Estimate memory
                memory = estimate_kv_cache_memory(cache)

                with self._memory_lock:
                    # Check if it fits
                    if self._current_memory + memory > self._max_memory:
                        logger.info(
                            f"[cache_persist] entry {i} would exceed memory limit "
                            f"({(self._current_memory + memory) / _BYTES_PER_MB:.0f}MB > "
                            f"{self._max_memory / _BYTES_PER_MB:.0f}MB), stopping load"
                        )
                        break

                    tokens_key = tuple(tokens)
                    entry = _CacheEntry(
                        tokens=tokens_key,
                        cache=cache,
                        memory_bytes=memory,
                    )
                    self._entries[tokens_key] = entry
                    self._current_memory += memory
                    bisect.insort(self._sorted_keys, tokens_key)
                    loaded += 1

                logger.info(
                    f"[cache_persist] loaded entry {i}: "
                    f"{len(tokens)} tokens, "
                    f"{memory / _BYTES_PER_MB:.1f}MB KV"
                )

            except Exception as e:
                logger.warning(f"[cache_persist] failed to load entry {i}: {e}")

        with self._memory_lock:
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] LOADED {loaded} entries from {cache_dir} "
            f"in {dt:.1f}s ({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return loaded
