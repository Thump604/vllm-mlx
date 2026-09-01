# SPDX-License-Identifier: Apache-2.0
"""
SSD KV cache tiering for vllm-mlx.

This module provides a cold-tier disk cache that sits behind
MemoryAwarePrefixCache. Evicted entries spill to NVMe instead of being
discarded, and cold-tier fetches reload from disk asynchronously with
RAM budget reservation before the read completes.

Key design:
- SQLite for atomic metadata index (no mutable JSON)
- Async writer thread for non-blocking spills
- Per-layer serializer interface for hybrid cache types
- Atomic temp-file + rename writes for crash consistency
- Metrics exposed from day one
"""

from __future__ import annotations

import array as _array
import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024
_PREFIX_FILTER_TOKENS = 16
_PERSISTENCE_IDENTITY_KEYS = ("model", "tokenizer", "cache_layout")


def _normalize_persistence_identity(
    identity: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Return a strict, JSON-safe cache identity or reject it.

    SSD entries are reusable only inside one exact model/tokenizer/cache-layout
    namespace.  ``None`` is retained for legacy ownerless tiers; an attached
    owner-bound cache binds a complete identity before its first spill.
    """
    if identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise ValueError("persistence identity must be a mapping")
    normalized = dict(identity)
    if set(normalized) != set(_PERSISTENCE_IDENTITY_KEYS):
        raise ValueError(
            "persistence identity requires model, tokenizer, and cache_layout"
        )
    if any(not isinstance(value, str) or not value for value in normalized.values()):
        raise ValueError("persistence identity values must be non-empty strings")
    return {key: normalized[key] for key in _PERSISTENCE_IDENTITY_KEYS}


def _safe_entry_path(relative_path: Any) -> bool:
    """Check that an index path names one direct SSD entry directory."""
    if not isinstance(relative_path, str) or not relative_path:
        return False
    normalized = os.path.normpath(relative_path)
    return not (
        os.path.isabs(relative_path)
        or normalized != relative_path
        or normalized in (".", "..")
        or normalized.startswith(f"..{os.sep}")
        or os.path.basename(relative_path) != relative_path
    )


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_persistence_identity(value: Any) -> dict[str, str] | None:
    """Decode an index identity; malformed/missing values remain unusable."""
    if value is None:
        return None
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        return _normalize_persistence_identity(decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class SSDCacheConfig:
    """Configuration for SSD cache tier.

    Attributes:
        cache_dir: Directory for SSD cache files. None = auto-detect
            (~/.cache/vllm-mlx/ssd_cache/{model}/).
        max_size_gb: Maximum total size of SSD cache in GB.
        max_entries: Maximum number of entries in SSD cache.
        file_permissions: Unix permission bits for cache data files.
        dir_permissions: Unix permission bits for cache directories.
        spill_queue_size: Max pending spill operations before dropping.
        retention_seconds: Optional max age for cache entries (None = no expiry).
    """

    cache_dir: str | None = None
    max_size_gb: float = 10.0
    max_entries: int = 10000
    file_permissions: int = 0o600
    dir_permissions: int = 0o700
    spill_queue_size: int = 64
    retention_seconds: int | None = None
    # Exact model/tokenizer/cache-layout identity for owner-bound tiers.
    # Legacy ownerless tiers may leave this unset.
    persistence_identity: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.max_size_gb <= 0:
            raise ValueError(f"max_size_gb must be > 0, got {self.max_size_gb}")
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.spill_queue_size < 1:
            raise ValueError(
                f"spill_queue_size must be >= 1, got {self.spill_queue_size}"
            )
        object.__setattr__(
            self,
            "persistence_identity",
            _normalize_persistence_identity(self.persistence_identity),
        )

    @property
    def max_size_bytes(self) -> int:
        """Maximum cache size in bytes."""
        return int(self.max_size_gb * _BYTES_PER_GB)


@dataclass
class SSDCacheStats:
    """Statistics for SSD cache tier — exposed from day one.

    Attributes:
        spill_count: Number of entries spilled to SSD.
        spill_bytes: Total bytes written to SSD.
        ssd_hits: Number of successful SSD cache lookups.
        ssd_misses: Number of SSD cache lookup misses.
        reload_latency_sum: Sum of reload latencies in seconds.
        reload_bytes: Total bytes read from SSD.
        promotion_failures: Number of failed promotions (RAM budget exhausted).
    """

    spill_count: int = 0
    spill_bytes: int = 0
    ssd_hits: int = 0
    ssd_misses: int = 0
    reload_latency_sum: float = 0.0
    reload_bytes: int = 0
    promotion_failures: int = 0

    def to_dict(self) -> dict:
        total_lookups = self.ssd_hits + self.ssd_misses
        hit_rate = self.ssd_hits / total_lookups if total_lookups > 0 else 0.0
        avg_latency_ms = (
            (self.reload_latency_sum / self.ssd_hits * 1000)
            if self.ssd_hits > 0
            else 0.0
        )
        return {
            "spill_count": self.spill_count,
            "spill_bytes": self.spill_bytes,
            "ssd_hits": self.ssd_hits,
            "ssd_misses": self.ssd_misses,
            "ssd_hit_rate": round(hit_rate, 4),
            "reload_latency_sum_s": round(self.reload_latency_sum, 4),
            "avg_reload_latency_ms": round(avg_latency_ms, 2),
            "reload_bytes": self.reload_bytes,
            "promotion_failures": self.promotion_failures,
        }


def _tokens_to_blob(tokens: tuple[int, ...]) -> bytes:
    """Serialize token tuple to a compact binary blob for SQLite storage.

    Uses the full token sequence as a binary blob for prefix matching.
    """
    arr = _array.array("i", tokens)
    return arr.tobytes()


def _blob_to_tokens(blob: bytes) -> tuple[int, ...]:
    """Deserialize binary blob back to token tuple."""
    arr = _array.array("i")
    arr.frombytes(blob)
    return tuple(arr)


def _tokens_hash(tokens: tuple[int, ...]) -> str:
    """Compute SHA-256 hex digest of a token sequence for use as primary key."""
    return hashlib.sha256(_tokens_to_blob(tokens)).hexdigest()


def _prefix_hash(tokens: tuple[int, ...]) -> str:
    """Hash the bounded token prefix used to prefilter prefix lookups."""
    return _tokens_hash(tokens[:_PREFIX_FILTER_TOKENS])


class SSDIndex:
    """SQLite-backed index for SSD cache entries.

    Uses SQLite for atomic metadata operations instead of mutable JSON.
    The token sequence is stored as a binary blob for prefix-searchable
    representation. The primary key is a SHA-256 hash of the token sequence.

    Thread safety: All operations are serialized through a threading.Lock.
    The SQLite connection uses WAL mode for concurrent read/write safety.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._db_lock = threading.Lock()
        db_path = os.path.join(cache_dir, "index.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        schema_sql = """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entries (
                token_hash   TEXT PRIMARY KEY,
                tokens_blob  BLOB NOT NULL,
                prefix_hash  TEXT,
                num_tokens   INTEGER NOT NULL,
                file_path    TEXT NOT NULL,
                memory_bytes INTEGER NOT NULL,
                persistence_identity TEXT,
                created_at   REAL NOT NULL,
                accessed_at  REAL NOT NULL
            );

            """
        self._conn.executescript(schema_sql)
        self._ensure_column("entries", "prefix_hash", "TEXT")
        self._ensure_column("entries", "persistence_identity", "TEXT")
        self._conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_entries_accessed
                ON entries(accessed_at);

            CREATE INDEX IF NOT EXISTS idx_entries_num_tokens
                ON entries(num_tokens);

            CREATE INDEX IF NOT EXISTS idx_entries_prefix_hash_num_tokens
                ON entries(prefix_hash, num_tokens);
            """)
        # Insert schema version if not present
        cur = self._conn.execute("SELECT COUNT(*) FROM schema_version")
        if cur.fetchone()[0] == 0:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (self._SCHEMA_VERSION,),
            )
        self._backfill_prefix_hashes()
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        if column not in {row["name"] for row in cur.fetchall()}:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _backfill_prefix_hashes(self) -> None:
        cur = self._conn.execute(
            "SELECT token_hash, tokens_blob FROM entries WHERE prefix_hash IS NULL"
        )
        rows = cur.fetchall()
        for row in rows:
            tokens = _blob_to_tokens(row["tokens_blob"])
            self._conn.execute(
                "UPDATE entries SET prefix_hash = ? WHERE token_hash = ?",
                (_prefix_hash(tokens), row["token_hash"]),
            )

    def insert_entry(
        self,
        tokens_key: tuple[int, ...],
        file_path: str,
        memory_bytes: int,
        num_tokens: int,
        persistence_identity: Mapping[str, str] | None = None,
    ) -> None:
        """Insert or replace a cache entry in the index."""
        normalized_identity = _normalize_persistence_identity(persistence_identity)
        identity_json = (
            json.dumps(normalized_identity, sort_keys=True, separators=(",", ":"))
            if normalized_identity is not None
            else None
        )
        now = time.time()
        token_hash = _tokens_hash(tokens_key)
        prefix_hash = _prefix_hash(tokens_key)
        tokens_blob = _tokens_to_blob(tokens_key)
        with self._db_lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO entries
                    (token_hash, tokens_blob, prefix_hash, num_tokens, file_path,
                     memory_bytes, persistence_identity, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token_hash,
                    tokens_blob,
                    prefix_hash,
                    num_tokens,
                    file_path,
                    memory_bytes,
                    identity_json,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def lookup_exact(self, tokens_key: tuple[int, ...]) -> dict | None:
        """Look up an exact token sequence. Returns dict or None."""
        token_hash = _tokens_hash(tokens_key)
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT file_path, memory_bytes, num_tokens, persistence_identity "
                "FROM entries WHERE token_hash = ?",
                (token_hash,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "file_path": row["file_path"],
            "memory_bytes": row["memory_bytes"],
            "num_tokens": row["num_tokens"],
            "persistence_identity": _decode_persistence_identity(
                row["persistence_identity"]
            ),
        }

    def lookup_prefix(self, query_tokens: tuple[int, ...]) -> list[dict]:
        """Find entries whose token sequence is a prefix of query_tokens.

        Uses a bounded token-prefix hash to avoid scanning all entries, then
        compares the full stored token blob against the corresponding prefix of
        query_tokens.

        Returns list of dicts sorted by num_tokens descending (longest prefix first).
        """
        query_len = len(query_tokens)
        query_blob = _tokens_to_blob(query_tokens)
        prefix_hashes = {
            _tokens_hash(query_tokens[:n])
            for n in range(1, min(query_len, _PREFIX_FILTER_TOKENS) + 1)
        }
        if not prefix_hashes:
            return []

        with self._db_lock:
            placeholders = ",".join("?" for _ in prefix_hashes)
            cur = self._conn.execute(
                "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes, "
                "persistence_identity "
                f"FROM entries WHERE num_tokens <= ? AND prefix_hash IN ({placeholders}) "
                "ORDER BY num_tokens DESC",
                (query_len, *prefix_hashes),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            stored_blob = row["tokens_blob"]
            n = row["num_tokens"]
            prefix_blob = query_blob[: n * 4]
            if stored_blob == prefix_blob:
                results.append(
                    {
                        "token_hash": row["token_hash"],
                        "file_path": row["file_path"],
                        "memory_bytes": row["memory_bytes"],
                        "num_tokens": n,
                        "persistence_identity": _decode_persistence_identity(
                            row["persistence_identity"]
                        ),
                    }
                )
        return results

    def delete_entry(self, tokens_key: tuple[int, ...]) -> None:
        """Delete an entry by token sequence."""
        token_hash = _tokens_hash(tokens_key)
        with self._db_lock:
            self._conn.execute(
                "DELETE FROM entries WHERE token_hash = ?", (token_hash,)
            )
            self._conn.commit()

    def get_lru(self, limit: int = 10) -> list[dict]:
        """Get the least recently used entries, ordered oldest first."""
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
                "FROM entries ORDER BY accessed_at ASC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "token_hash": row["token_hash"],
                    "tokens_blob": row["tokens_blob"],
                    "file_path": row["file_path"],
                    "memory_bytes": row["memory_bytes"],
                    "num_tokens": row["num_tokens"],
                }
            )
        return results

    def get_total_bytes(self) -> int:
        """Get total memory_bytes across all entries."""
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT COALESCE(SUM(memory_bytes), 0) FROM entries"
            )
            return cur.fetchone()[0]

    def get_entry_count(self) -> int:
        """Get number of entries in the index."""
        with self._db_lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM entries")
            return cur.fetchone()[0]

    def touch(self, tokens_key: tuple[int, ...]) -> None:
        """Update accessed_at timestamp for an entry (marks as recently used)."""
        token_hash = _tokens_hash(tokens_key)
        with self._db_lock:
            self._conn.execute(
                "UPDATE entries SET accessed_at = ? WHERE token_hash = ?",
                (time.time(), token_hash),
            )
            self._conn.commit()

    def all_entries(self) -> list[dict]:
        """Return all entries (for startup reconciliation)."""
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
                "FROM entries ORDER BY accessed_at DESC"
            )
            rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "token_hash": row["token_hash"],
                    "tokens_blob": row["tokens_blob"],
                    "file_path": row["file_path"],
                    "memory_bytes": row["memory_bytes"],
                    "num_tokens": row["num_tokens"],
                }
            )
        return results

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._db_lock:
            self._conn.close()


# Support matrix: maps cache type names to their serializer status
SERIALIZER_SUPPORT_MATRIX = {
    "KVCache": "supported",
    "RotatingKVCache": "supported",  # Serialized as KVCache (keys/values/offset)
    "ArraysCache": "supported",
    "MambaCache": "supported",  # Legacy name for ArraysCache
    "_QuantizedCacheWrapper": "supported_via_dequant_on_spill",
    "QuantizedKVCache": "supported_via_dequant_on_spill",
}


class LayerSerializer(ABC):
    """Interface for per-layer cache serialization.

    Spill is split across two threads: ``snapshot_layer`` runs on the
    producer (request handler) thread so the mx→numpy materialization
    happens where the per-request Stream(gpu, N) is registered;
    ``serialize_layer`` then runs on the SSD writer thread with numpy only.
    """

    @abstractmethod
    def snapshot_layer(self, layer: Any) -> dict[str, Any]:
        """Producer-thread CPU snapshot of an MLX-backed cache layer."""
        ...

    @abstractmethod
    def serialize_layer(
        self, snapshot: dict[str, Any], layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        """Writer-thread: persist a snapshot to safetensors at file_path.

        Returns metadata dict with at least 'layer_type'.
        """
        ...

    @abstractmethod
    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        """Read a layer back from disk. Returns layer-state dict."""
        ...


def _mx_to_numpy_safe(arr: Any) -> tuple[np.ndarray, str | None]:
    """mx.array → np.ndarray, upcasting numpy-unsupported dtypes (bf16) to fp32.

    Returns (numpy_array, original_dtype_name_or_None). The name is only set
    when an upcast happened, so the SSD-promote path can cast back.
    """
    try:
        return np.array(arr), None
    except RuntimeError as exc:
        # numpy ↔ mlx bf16 buffer-protocol mismatch on mlx ≥ 0.31. Re-raise
        # anything else — don't swallow unrelated errors.
        if "buffer format string" not in str(exc):
            raise
        import mlx.core as mx

        original_dtype = str(arr.dtype).rsplit(".", 1)[-1]
        upcast = arr.astype(mx.float32)
        mx.eval(upcast)  # astype is lazy; force materialization here
        return np.array(upcast), original_dtype


class KVCacheSerializer(LayerSerializer):
    """Serializer for KVCache and RotatingKVCache layers.

    Handles layers with .keys, .values, .offset attributes.
    RotatingKVCache also has .max_size, .keep, .step, ._idx.
    """

    # Extra attributes carried by RotatingKVCache (but not vanilla KVCache).
    # Stored in metadata so deserialize can faithfully reconstruct either type.
    _ROTATING_ATTRS = ("max_size", "keep", "step", "_idx")
    _KV_ATTRS = ("_max_size", "step")

    def snapshot_layer(self, layer: Any) -> dict[str, Any]:
        keys_np, keys_orig_dtype = _mx_to_numpy_safe(layer.keys)
        values_np, values_orig_dtype = _mx_to_numpy_safe(layer.values)

        # Quantized-spill cast-back sentinel: the enqueue_spill dequant path
        # may have cast bf16 → fp16 before snapshot to dodge numpy's PEP 3118
        # buffer-protocol mismatch. The cast loses the original-dtype signal
        # _mx_to_numpy_safe would otherwise capture (since fp16 IS numpy-
        # supported, _mx_to_numpy_safe returns dtype=None). Honor an explicit
        # sentinel on the layer when present so the reload path can restore
        # bf16 instead of leaving the model with fp16 KV.
        keys_orig_dtype = (
            getattr(layer, "_ssd_keys_original_dtype", None) or keys_orig_dtype
        )
        values_orig_dtype = (
            getattr(layer, "_ssd_values_original_dtype", None) or values_orig_dtype
        )

        snapshot: dict[str, Any] = {
            "keys_np": keys_np,
            "values_np": values_np,
            "offset": layer.offset,
            "layer_type": (
                "RotatingKVCache"
                if type(layer).__name__ == "RotatingKVCache"
                or all(hasattr(layer, attr) for attr in ("max_size", "keep", "_idx"))
                else "KVCache"
            ),
        }
        if keys_orig_dtype is not None:
            snapshot["keys_original_dtype"] = keys_orig_dtype
        if values_orig_dtype is not None:
            snapshot["values_original_dtype"] = values_orig_dtype

        # Cache-shaping and rotating extras (plain Python scalars, no MLX).
        attrs = (
            self._ROTATING_ATTRS
            if snapshot["layer_type"] == "RotatingKVCache"
            else self._KV_ATTRS
        )
        for attr in attrs:
            if hasattr(layer, attr):
                snapshot[attr] = getattr(layer, attr)
        return snapshot

    def serialize_layer(
        self, snapshot: dict[str, Any], layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        from safetensors.numpy import save_file

        tensors = {
            f"layer_{layer_idx}_keys": snapshot["keys_np"],
            f"layer_{layer_idx}_values": snapshot["values_np"],
        }
        save_file(tensors, file_path)

        metadata = {
            "layer_type": snapshot.get("layer_type", "KVCache"),
            "layer_idx": layer_idx,
            "offset": snapshot["offset"],
            "keys_shape": list(snapshot["keys_np"].shape),
            "values_shape": list(snapshot["values_np"].shape),
            "keys_dtype": str(snapshot["keys_np"].dtype),
            "values_dtype": str(snapshot["values_np"].dtype),
        }
        for k in ("keys_original_dtype", "values_original_dtype"):
            if k in snapshot:
                metadata[k] = snapshot[k]
        attrs = (
            self._ROTATING_ATTRS
            if metadata["layer_type"] == "RotatingKVCache"
            else self._KV_ATTRS
        )
        for attr in attrs:
            if attr in snapshot:
                metadata[attr] = snapshot[attr]

        return metadata

    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        from safetensors.numpy import load_file

        layer_idx = metadata["layer_idx"]
        tensors = load_file(file_path)

        result = {
            "keys": tensors[f"layer_{layer_idx}_keys"],
            "values": tensors[f"layer_{layer_idx}_values"],
            "offset": metadata["offset"],
            "layer_type": metadata["layer_type"],
            "keys_shape": metadata["keys_shape"],
            "values_shape": metadata["values_shape"],
            "keys_dtype": metadata["keys_dtype"],
            "values_dtype": metadata["values_dtype"],
        }
        # Dtype hints surfaced so _reconstruct_ssd_layers can cast back.
        for k in ("keys_original_dtype", "values_original_dtype"):
            if k in metadata:
                result[k] = metadata[k]
        attrs = (
            self._ROTATING_ATTRS
            if metadata["layer_type"] == "RotatingKVCache"
            else self._KV_ATTRS
        )
        for attr in attrs:
            if attr in metadata:
                result[attr] = metadata[attr]
        return result


class ArraysCacheSerializer(LayerSerializer):
    """Serializer for ArraysCache (Mamba/linear attention) layers.

    Handles layers with .state attribute containing a list of arrays.
    """

    def snapshot_layer(self, layer: Any) -> dict[str, Any]:
        state_np: list[np.ndarray] = []
        original_dtypes: list[str | None] = []
        for arr in layer.state:
            np_arr, orig = _mx_to_numpy_safe(arr)
            state_np.append(np_arr)
            original_dtypes.append(orig)

        snapshot: dict[str, Any] = {
            "state_np": state_np,
            "layer_type": (
                "MambaCache" if type(layer).__name__ == "MambaCache" else "ArraysCache"
            ),
        }
        # Skip the dtype list in the common (fp16/fp32) case.
        if any(d is not None for d in original_dtypes):
            snapshot["state_original_dtypes"] = original_dtypes
        metadata_np: dict[str, np.ndarray] = {}
        metadata_original_dtypes: dict[str, str | None] = {}
        for attr in ("left_padding", "lengths"):
            value = getattr(layer, attr, None)
            if value is None:
                continue
            np_value, original_dtype = _mx_to_numpy_safe(value)
            metadata_np[attr] = np_value
            metadata_original_dtypes[attr] = original_dtype
        if metadata_np:
            snapshot["metadata_np"] = metadata_np
            snapshot["metadata_original_dtypes"] = metadata_original_dtypes
        if hasattr(layer, "meta_state"):
            snapshot["meta_state"] = getattr(layer, "meta_state")
        return snapshot

    def serialize_layer(
        self, snapshot: dict[str, Any], layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        # Writer-thread side: pure numpy + disk.
        from safetensors.numpy import save_file

        state_np = snapshot["state_np"]
        tensors = {
            f"layer_{layer_idx}_state_{i}": arr for i, arr in enumerate(state_np)
        }
        metadata_np = snapshot.get("metadata_np", {})
        tensors.update(
            {f"layer_{layer_idx}_{name}": value for name, value in metadata_np.items()}
        )
        save_file(tensors, file_path)

        metadata = {
            "layer_type": snapshot.get("layer_type", "ArraysCache"),
            "layer_idx": layer_idx,
            "num_arrays": len(state_np),
            "state_shapes": [list(arr.shape) for arr in state_np],
            "state_dtypes": [str(arr.dtype) for arr in state_np],
            "metadata_arrays": sorted(metadata_np),
            "metadata_shapes": {
                name: list(value.shape) for name, value in metadata_np.items()
            },
            "metadata_dtypes": {
                name: str(value.dtype) for name, value in metadata_np.items()
            },
        }
        if "state_original_dtypes" in snapshot:
            metadata["state_original_dtypes"] = snapshot["state_original_dtypes"]
        if "metadata_original_dtypes" in snapshot:
            metadata["metadata_original_dtypes"] = snapshot["metadata_original_dtypes"]
        if "meta_state" in snapshot:
            metadata["meta_state"] = snapshot["meta_state"]
        return metadata

    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        from safetensors.numpy import load_file

        layer_idx = metadata["layer_idx"]
        num_arrays = metadata["num_arrays"]
        tensors = load_file(file_path)

        state = []
        for i in range(num_arrays):
            state.append(tensors[f"layer_{layer_idx}_state_{i}"])
        metadata_values = {}
        for name in metadata.get("metadata_arrays", []):
            metadata_values[name] = tensors[f"layer_{layer_idx}_{name}"]
        result = {
            "state": state,
            "layer_type": metadata["layer_type"],
            "state_shapes": metadata["state_shapes"],
            "state_dtypes": metadata["state_dtypes"],
            "metadata": metadata_values,
            "metadata_arrays": metadata.get("metadata_arrays", []),
            "metadata_shapes": metadata.get("metadata_shapes", {}),
            "metadata_dtypes": metadata.get("metadata_dtypes", {}),
        }
        if "state_original_dtypes" in metadata:
            result["state_original_dtypes"] = metadata["state_original_dtypes"]
        if "metadata_original_dtypes" in metadata:
            result["metadata_original_dtypes"] = metadata["metadata_original_dtypes"]
        if "meta_state" in metadata:
            result["meta_state"] = metadata["meta_state"]
        return result


def get_serializer_for_layer(layer: Any) -> LayerSerializer:
    """Return the appropriate serializer for a cache layer.

    Dispatches based on duck-typing:
    - If layer has .keys and .values and .offset -> KVCacheSerializer
    - If layer has .state and it's a list -> ArraysCacheSerializer

    Raises ValueError for unsupported layer types.
    """
    if hasattr(layer, "keys") and hasattr(layer, "values") and hasattr(layer, "offset"):
        return KVCacheSerializer()
    if hasattr(layer, "state") and isinstance(getattr(layer, "state", None), list):
        return ArraysCacheSerializer()
    raise ValueError(
        f"Unsupported cache layer type: {type(layer).__name__}. "
        f"Supported: {list(SERIALIZER_SUPPORT_MATRIX.keys())}"
    )


@dataclass(frozen=True)
class SSDReadResult:
    """Validated SSD bytes ready for owner-thread reconstruction."""

    tokens: tuple[int, ...]
    file_path: str
    memory_bytes: int
    layers: list[dict[str, Any]]
    read_bytes: int
    latency_seconds: float


def _strict_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("layer shape must be a non-empty sequence")
    shape = tuple(value)
    if any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape
    ):
        raise ValueError("layer shape dimensions must be positive integers")
    return shape


def _strict_dtype(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("layer dtype must be a non-empty string")
    return value


def _strict_original_dtype(value: Any) -> str:
    dtype = _strict_dtype(value)
    if dtype == "bfloat16":
        return dtype
    try:
        np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"unsupported original dtype {dtype!r}") from exc
    return dtype


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_layer_manifest(
    layer_meta: Any, layer_idx: int, layer_data: dict[str, Any]
) -> None:
    """Validate manifest metadata and deserialized arrays before publication."""
    if not isinstance(layer_meta, dict) or not isinstance(layer_data, dict):
        raise ValueError("layer metadata/data must be objects")
    if _strict_int(layer_meta.get("layer_idx"), "layer_idx") != layer_idx:
        raise ValueError("layer order is not contiguous")

    layer_type = layer_meta.get("layer_type")
    if layer_type in ("KVCache", "RotatingKVCache"):
        required = {
            "offset",
            "keys_shape",
            "values_shape",
            "keys_dtype",
            "values_dtype",
        }
        if not required.issubset(layer_meta):
            raise ValueError("KV layer metadata is incomplete")
        keys_shape = _strict_shape(layer_meta["keys_shape"])
        values_shape = _strict_shape(layer_meta["values_shape"])
        if len(keys_shape) < 3 or len(values_shape) < 3:
            raise ValueError("KV layer arrays must have at least three dimensions")
        if keys_shape[:3] != values_shape[:3]:
            raise ValueError("KV key/value batch geometry differs")
        if keys_shape[-2] != values_shape[-2]:
            raise ValueError("KV key/value sequence geometry differs")
        keys_dtype = _strict_dtype(layer_meta["keys_dtype"])
        values_dtype = _strict_dtype(layer_meta["values_dtype"])
        offset = _strict_int(layer_meta["offset"], "offset")
        if layer_type == "KVCache" and offset > keys_shape[-2]:
            raise ValueError("KV offset exceeds serialized sequence length")
        if not isinstance(layer_data.get("keys"), np.ndarray) or not isinstance(
            layer_data.get("values"), np.ndarray
        ):
            raise ValueError("KV layer arrays are not numpy arrays")
        keys = layer_data["keys"]
        values = layer_data["values"]
        if tuple(keys.shape) != keys_shape or tuple(values.shape) != values_shape:
            raise ValueError("KV layer shape does not match manifest")
        if str(keys.dtype) != keys_dtype or str(values.dtype) != values_dtype:
            raise ValueError("KV layer dtype does not match manifest")
        for name in ("keys_original_dtype", "values_original_dtype"):
            if name in layer_meta:
                _strict_original_dtype(layer_meta[name])

        if layer_type == "RotatingKVCache":
            for name in ("max_size", "keep", "step", "_idx"):
                if name not in layer_meta:
                    raise ValueError("RotatingKVCache metadata is incomplete")
            max_size = _strict_int(layer_meta["max_size"], "max_size", minimum=1)
            keep = _strict_int(layer_meta["keep"], "keep")
            step = _strict_int(layer_meta["step"], "step", minimum=1)
            idx = _strict_int(layer_meta["_idx"], "_idx")
            if keep >= max_size or idx > keys_shape[-2]:
                raise ValueError("invalid RotatingKVCache geometry")
            if _strict_int(layer_meta["offset"], "offset") < idx:
                raise ValueError("invalid RotatingKVCache offset/index state")
            if step < 1:
                raise ValueError("invalid RotatingKVCache step")
        else:
            if "_max_size" in layer_meta and layer_meta["_max_size"] is not None:
                _strict_int(layer_meta["_max_size"], "_max_size", minimum=1)
            if "step" in layer_meta:
                _strict_int(layer_meta["step"], "step", minimum=1)
        return

    if layer_type not in ("ArraysCache", "MambaCache"):
        raise ValueError(f"unsupported layer type {layer_type!r}")
    num_arrays = _strict_int(layer_meta.get("num_arrays"), "num_arrays", minimum=1)
    state_shapes = layer_meta.get("state_shapes")
    state_dtypes = layer_meta.get("state_dtypes")
    state = layer_data.get("state")
    if (
        not isinstance(state, list)
        or len(state) != num_arrays
        or not isinstance(state_shapes, list)
        or len(state_shapes) != num_arrays
        or not isinstance(state_dtypes, list)
        or len(state_dtypes) != num_arrays
    ):
        raise ValueError("ArraysCache metadata/state count mismatch")
    original_dtypes = layer_meta.get("state_original_dtypes")
    if original_dtypes is not None:
        if not isinstance(original_dtypes, list) or len(original_dtypes) != num_arrays:
            raise ValueError("ArraysCache original dtype count mismatch")
        for dtype in original_dtypes:
            if dtype is not None:
                _strict_original_dtype(dtype)
    for array, shape_meta, dtype_meta in zip(state, state_shapes, state_dtypes):
        shape = _strict_shape(shape_meta)
        dtype = _strict_dtype(dtype_meta)
        if not isinstance(array, np.ndarray):
            raise ValueError("ArraysCache state is not a numpy array")
        if tuple(array.shape) != shape or str(array.dtype) != dtype:
            raise ValueError("ArraysCache state does not match manifest")

    required_metadata = {"metadata_arrays", "metadata_shapes", "metadata_dtypes"}
    if not required_metadata.issubset(layer_meta):
        raise ValueError("ArraysCache metadata is incomplete")
    metadata_arrays = layer_meta["metadata_arrays"]
    if (
        not isinstance(metadata_arrays, list)
        or len(set(metadata_arrays)) != len(metadata_arrays)
        or any(name not in {"left_padding", "lengths"} for name in metadata_arrays)
    ):
        raise ValueError("ArraysCache metadata arrays are invalid")
    metadata_shapes = layer_meta["metadata_shapes"]
    metadata_dtypes = layer_meta["metadata_dtypes"]
    metadata_values = layer_data.get("metadata", {})
    if (
        not isinstance(metadata_shapes, dict)
        or not isinstance(metadata_dtypes, dict)
        or not isinstance(metadata_values, dict)
        or set(metadata_shapes) != set(metadata_arrays)
        or set(metadata_dtypes) != set(metadata_arrays)
        or set(metadata_values) != set(metadata_arrays)
    ):
        raise ValueError("ArraysCache metadata shape/count mismatch")
    metadata_original_dtypes = layer_meta.get("metadata_original_dtypes", {})
    if not isinstance(metadata_original_dtypes, dict) or not set(
        metadata_original_dtypes
    ).issubset(metadata_arrays):
        raise ValueError("ArraysCache metadata dtype information is invalid")
    for name in metadata_arrays:
        shape = _strict_shape(metadata_shapes[name])
        dtype = _strict_dtype(metadata_dtypes[name])
        value = metadata_values[name]
        if not isinstance(value, np.ndarray):
            raise ValueError("ArraysCache metadata is not a numpy array")
        if tuple(value.shape) != shape or str(value.dtype) != dtype:
            raise ValueError("ArraysCache metadata does not match manifest")
        original_dtype = metadata_original_dtypes.get(name)
        if original_dtype is not None:
            _strict_original_dtype(original_dtype)


def reconstruct_ssd_layers(layer_dicts: list[dict[str, Any]]) -> list[Any] | None:
    """Rebuild validated SSD layers on the caller's current MLX thread."""
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, KVCache

        try:
            from mlx_lm.models.cache import MambaCache
        except ImportError:
            MambaCache = None
        try:
            from mlx_lm.models.cache import RotatingKVCache
        except ImportError:
            RotatingKVCache = None

        if not isinstance(layer_dicts, list) or not layer_dicts:
            return None

        def restore_dtype(array, dtype_name):
            if dtype_name is None:
                return array
            dtype = getattr(mx, dtype_name, None)
            if dtype is None:
                return None
            return array.astype(dtype)

        result = []
        for layer_dict in layer_dicts:
            if not isinstance(layer_dict, dict):
                return None
            if "keys" in layer_dict and "values" in layer_dict:
                layer_type = layer_dict.get("layer_type", "KVCache")
                keys = mx.array(layer_dict["keys"])
                values = mx.array(layer_dict["values"])
                keys = restore_dtype(keys, layer_dict.get("keys_original_dtype"))
                values = restore_dtype(values, layer_dict.get("values_original_dtype"))
                if keys is None or values is None:
                    return None
                if layer_type == "RotatingKVCache":
                    if RotatingKVCache is None:
                        return None
                    cache = RotatingKVCache(
                        int(layer_dict["max_size"]), int(layer_dict["keep"])
                    )
                elif layer_type == "KVCache":
                    max_size = layer_dict.get("_max_size")
                    step = layer_dict.get("step")
                    try:
                        cache = KVCache(max_size=max_size, step=step)
                    except TypeError:
                        try:
                            cache = KVCache(max_size=max_size)
                        except TypeError:
                            # Older governed mlx-lm releases expose only the
                            # vanilla ``KVCache()`` constructor.  Instantiate
                            # that public class first, then restore optional
                            # shaping metadata on the object.
                            cache = KVCache()
                            if max_size is not None:
                                cache._max_size = int(max_size)
                            if step is not None:
                                cache.step = int(step)
                else:
                    return None
                cache.keys = keys
                cache.values = values
                cache.offset = int(layer_dict["offset"])
                if layer_type == "RotatingKVCache":
                    cache.max_size = int(layer_dict["max_size"])
                    cache.keep = int(layer_dict["keep"])
                    cache.step = int(layer_dict["step"])
                    cache._idx = int(layer_dict["_idx"])
                elif "_max_size" in layer_dict:
                    cache._max_size = layer_dict["_max_size"]
                if "step" in layer_dict:
                    cache.step = int(layer_dict["step"])
                result.append(cache)
            elif "state" in layer_dict:
                layer_type = layer_dict.get("layer_type", "ArraysCache")
                if layer_type == "ArraysCache":
                    cache = ArraysCache(len(layer_dict["state"]))
                elif layer_type == "MambaCache" and MambaCache is not None:
                    try:
                        cache = MambaCache(len(layer_dict["state"]))
                    except TypeError:
                        cache = MambaCache()
                else:
                    return None
                state = []
                original_dtypes = layer_dict.get("state_original_dtypes")
                for index, value in enumerate(layer_dict["state"]):
                    array = mx.array(value)
                    if original_dtypes is not None:
                        dtype_name = original_dtypes[index]
                        array = restore_dtype(array, dtype_name)
                        if array is None:
                            return None
                    state.append(array)
                cache.state = state
                metadata_values = layer_dict.get("metadata", {})
                metadata_original_dtypes = layer_dict.get(
                    "metadata_original_dtypes", {}
                )
                for name, value in metadata_values.items():
                    array = mx.array(value)
                    array = restore_dtype(array, metadata_original_dtypes.get(name))
                    if array is None:
                        return None
                    setattr(cache, name, array)
                if "meta_state" in layer_dict:
                    cache.meta_state = layer_dict["meta_state"]
                result.append(cache)
            else:
                return None
        return result
    except Exception as exc:
        logger.warning("[ssd_cache] reconstruction failed: %s", exc)
        return None


class SSDCacheTier:
    """Cold-tier disk cache for KV cache entries.

    Manages a SQLite-indexed on-disk cache directory. Evicted RAM entries
    are spilled here via an async writer thread. Cold-tier fetches reload
    from disk asynchronously with RAM budget reservation.

    Directory layout::

        cache_dir/
          index.db           # SQLite metadata index
          data/              # safetensors files per entry
            {hash}/          # one directory per entry
              layer_0.safetensors
              layer_1.safetensors
              manifest.json  # per-entry layer metadata
    """

    _WRITER_JOIN_TIMEOUT_S = 5.0

    def __init__(self, config: SSDCacheConfig) -> None:
        self._config = config
        self._closed = True
        self._writer_thread: threading.Thread | None = None

        if config.cache_dir is None:
            raise ValueError("SSDCacheConfig.cache_dir must be set")

        self._cache_dir = config.cache_dir
        self._data_dir = os.path.join(self._cache_dir, "data")

        # Create directory structure
        os.makedirs(self._cache_dir, mode=config.dir_permissions, exist_ok=True)
        os.makedirs(self._data_dir, mode=config.dir_permissions, exist_ok=True)

        try:
            # Open SQLite index
            self._index = SSDIndex(self._cache_dir)

            # Stats
            self._stats = SSDCacheStats()
            self._lock = threading.Lock()
            self._persistence_identity = config.persistence_identity

            # Lifecycle state is independent from the stats lock: close() may
            # wait for a writer that still needs the stats lock to finish its
            # current entry.
            self._lifecycle_lock = threading.Lock()
            self._accepting_spills = True
            self._writer_shutdown_requested = False

            # Spill queue and writer thread
            self._spill_queue: queue.Queue = queue.Queue(
                maxsize=config.spill_queue_size
            )
            self._closed = False
        except Exception:
            index = getattr(self, "_index", None)
            if index is not None:
                try:
                    index.close()
                except Exception:
                    logger.exception(
                        "ssd_cache: failed to close index during init cleanup"
                    )
            raise

    @staticmethod
    def _entry_hash(tokens: tuple[int, ...]) -> str:
        """Compute deterministic hash for a token sequence."""
        return _tokens_hash(tokens)

    def get_stats(self) -> dict:
        """Return current SSD cache statistics."""
        with self._lock:
            return self._stats.to_dict()

    @property
    def persistence_identity(self) -> dict[str, str] | None:
        """Return the exact identity required for owner-bound entries."""
        with self._lifecycle_lock:
            return (
                dict(self._persistence_identity)
                if self._persistence_identity is not None
                else None
            )

    def bind_persistence_identity(self, identity: Mapping[str, str]) -> None:
        """Bind a tier to one exact cache identity before it accepts spills."""
        normalized = _normalize_persistence_identity(identity)
        if normalized is None:
            raise ValueError("owner-bound SSD tiers require cache identity")
        with self._lifecycle_lock:
            current = self._persistence_identity
            if current is not None and current != normalized:
                raise ValueError("SSD tier cache identity cannot be changed")
            self._persistence_identity = normalized

    def record_lookup_miss(self) -> None:
        """Record one metadata lookup that found no SSD candidate."""
        with self._lock:
            self._stats.ssd_misses += 1

    def record_promotion_failure(self) -> None:
        """Record one candidate promotion that did not publish to RAM."""
        with self._lock:
            self._stats.promotion_failures += 1

    def start_writer(self) -> None:
        """Start the background spill writer thread."""
        with self._lifecycle_lock:
            if self._closed or self._writer_shutdown_requested:
                raise RuntimeError("cannot start a closed SSD cache tier")
            if self._writer_thread is not None:
                return
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="ssd-cache-writer"
            )
            self._writer_thread.start()
        logger.info("[ssd_cache] writer thread started")

    def _writer_loop(self) -> None:
        """Drain spill queue and persist entries. Numpy-only — no MLX here."""
        while True:
            item = self._spill_queue.get()
            if item is None:  # Poison pill for shutdown
                break

            tokens_key, layer_snapshots, memory_bytes = item
            try:
                self._write_entry(tokens_key, layer_snapshots, memory_bytes)
            except Exception:
                logger.exception(
                    f"[ssd_cache] failed to write entry " f"({len(tokens_key)} tokens)"
                )

    def enqueue_spill(
        self,
        tokens: tuple[int, ...],
        cache: list[Any],
        memory_bytes: int,
    ) -> bool:
        """Enqueue a cache entry for async spill to SSD.

        Must be called on the producer thread (the request handler that
        owns the layer's Stream(gpu, N)) — the snapshot below materializes
        MLX → numpy here so the writer thread never has to.

        Returns True if enqueued, False if queue is full (entry dropped).
        """
        with self._lifecycle_lock:
            if not self._accepting_spills:
                return False

        # Dequantize on the CALLER's thread, which owns the MLX GPU stream.
        # mx.dequantize is a GPU compute op; running it on the writer thread
        # aborts the process ("no Stream(gpu,N) in current thread"). Materialize
        # with mx.eval so the writer thread only does a host-side copy. The layer
        # serializers handle plain KVCache/ArraysCache only; any layer whose
        # .keys/.values are a tuple/list of arrays (packed, scales, biases)
        # — either our `_QuantizedCacheWrapper` or mlx-lm's native
        # `QuantizedKVCache` produced when --kv-cache-quantization is on — must
        # be reduced to a single dense array per attribute before queueing.
        import mlx.core as mx
        from .memory_cache import _QuantizedCacheWrapper, _dequantize_cache

        def _is_quantized_layer(layer):
            if isinstance(layer, _QuantizedCacheWrapper):
                return True
            keys = getattr(layer, "keys", None)
            return isinstance(keys, (tuple, list))

        if any(_is_quantized_layer(layer) for layer in cache):
            try:
                from mlx_lm.models.cache import QuantizedKVCache
            except ImportError:
                QuantizedKVCache = ()  # never matches isinstance below

            converted: list = []
            for layer in cache:
                if isinstance(layer, _QuantizedCacheWrapper):
                    # Existing path: wrapper → orig_type with dequantized keys.
                    converted.extend(_dequantize_cache([layer]))
                elif isinstance(layer, QuantizedKVCache) or (
                    hasattr(layer, "keys") and isinstance(layer.keys, (tuple, list))
                ):
                    # Native mlx-lm QuantizedKVCache: keys/values are
                    # (packed, scales, biases) tuples. Dequantize in place
                    # into a fresh KVCache-shaped duck-type for the serializer.
                    from mlx_lm.models.cache import KVCache as _KVCache

                    bits = getattr(layer, "bits", 8)
                    group_size = getattr(layer, "group_size", 64)
                    kv = _KVCache.__new__(_KVCache)
                    kv.keys = mx.dequantize(
                        *layer.keys, group_size=group_size, bits=bits
                    )
                    kv.values = mx.dequantize(
                        *layer.values, group_size=group_size, bits=bits
                    )
                    kv.offset = getattr(layer, "offset", kv.keys.shape[-2])
                    if (
                        kv.keys is not None
                        and hasattr(kv.keys, "shape")
                        and len(kv.keys.shape) >= 3
                        and kv.offset < kv.keys.shape[-2]
                    ):
                        kv.keys = kv.keys[..., : kv.offset, :]
                        kv.values = kv.values[..., : kv.offset, :]
                    converted.append(kv)
                else:
                    converted.append(layer)
            cache = converted
            # Numpy's PEP 3118 buffer protocol doesn't understand bfloat16 —
            # it sees the bf16 array as format "B" (uint8) but the buffer items
            # are 2 bytes, so np.array() raises a RuntimeError mismatch. Cast
            # to float16 here on the CALLER's stream: KV values sit well within
            # the fp16 ±65504 range, fp16 has MORE mantissa bits than bf16, and
            # the byte size is identical. Stash the pre-cast dtype as a sentinel
            # so KVCacheSerializer.snapshot_layer can record it and the reload
            # path (scheduler._reconstruct_ssd_layers) can cast back to bf16 —
            # otherwise the model receives fp16 KV where it computed bf16.
            # Then force eval so the writer thread sees materialized host buffers.
            for layer in cache:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if k is None or v is None or isinstance(k, (tuple, list)):
                    continue
                if str(getattr(k, "dtype", "")).endswith("bfloat16"):
                    layer._ssd_keys_original_dtype = "bfloat16"
                    layer._ssd_values_original_dtype = "bfloat16"
                    layer.keys = k.astype(mx.float16)
                    layer.values = v.astype(mx.float16)
                mx.eval(layer.keys, layer.values)
            logger.info(
                "[ssd_cache] dequantized %d layers before spill (%d tokens)",
                len(cache),
                len(tokens),
            )

        try:
            layer_snapshots: list[tuple[LayerSerializer, dict[str, Any]]] = []
            for layer in cache:
                serializer = get_serializer_for_layer(layer)
                snapshot = serializer.snapshot_layer(layer)
                layer_snapshots.append((serializer, snapshot))
        except Exception:
            logger.exception(
                "[ssd_cache] failed to snapshot layers for spill "
                f"({len(tokens)} tokens) — entry dropped"
            )
            return False

        with self._lifecycle_lock:
            if not self._accepting_spills:
                return False
            try:
                self._spill_queue.put_nowait((tokens, layer_snapshots, memory_bytes))
                return True
            except queue.Full:
                logger.warning(
                    f"[ssd_cache] spill queue full, dropping entry "
                    f"({len(tokens)} tokens, {memory_bytes} bytes)"
                )
                return False

    def _write_entry(
        self,
        tokens_key: tuple[int, ...],
        layer_snapshots: list[tuple[LayerSerializer, dict[str, Any]]],
        memory_bytes: int,
    ) -> None:
        """Atomically persist one entry (writer thread; numpy-only input)."""
        import shutil

        entry_hash = self._entry_hash(tokens_key)
        entry_dir = os.path.join(self._data_dir, entry_hash)
        tmp_dir = entry_dir + ".tmp"

        # Clean up any leftover tmp dir from a previous crash
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        os.makedirs(tmp_dir, mode=self._config.dir_permissions, exist_ok=True)

        layer_manifests = []
        total_file_bytes = 0

        for i, (serializer, snapshot) in enumerate(layer_snapshots):
            layer_path = os.path.join(tmp_dir, f"layer_{i}.safetensors")
            metadata = serializer.serialize_layer(snapshot, i, layer_path)
            metadata["file_sha256"] = _file_sha256(layer_path)
            layer_manifests.append(metadata)

            # Set file permissions
            os.chmod(layer_path, self._config.file_permissions)
            total_file_bytes += os.path.getsize(layer_path)

        # Write manifest
        manifest = {
            "num_layers": len(layer_snapshots),
            "layers": layer_manifests,
            "memory_bytes": memory_bytes,
            "num_tokens": len(tokens_key),
            "persistence_identity": self._persistence_identity,
        }
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        os.chmod(manifest_path, self._config.file_permissions)

        # Save tokens binary
        tokens_path = os.path.join(tmp_dir, "tokens.bin")
        arr = _array.array("i", tokens_key)
        with open(tokens_path, "wb") as f:
            arr.tofile(f)
        os.chmod(tokens_path, self._config.file_permissions)

        # Atomic rename: tmp_dir -> entry_dir
        if os.path.exists(entry_dir):
            shutil.rmtree(entry_dir)
        os.rename(tmp_dir, entry_dir)

        # Update index
        relative_path = entry_hash
        self._index.insert_entry(
            tokens_key=tokens_key,
            file_path=relative_path,
            memory_bytes=memory_bytes,
            num_tokens=len(tokens_key),
            persistence_identity=self._persistence_identity,
        )

        # Update stats
        with self._lock:
            self._stats.spill_count += 1
            self._stats.spill_bytes += total_file_bytes

        logger.debug(
            f"[ssd_cache] spilled entry: {len(tokens_key)} tokens, "
            f"{total_file_bytes} bytes on disk"
        )

        # Enforce capacity after write
        self._enforce_capacity()

    def lookup_ssd(self, tokens: tuple[int, ...]) -> dict | None:
        """Synchronous check whether tokens exist in SSD tier.

        This is fast (SQLite lookup only, no disk I/O for data).
        Called from synchronous fetch() to report an SSD candidate.

        Returns:
            Dict with entry metadata if found, None otherwise.
        """
        result = self._index.lookup_exact(tokens)
        if result is not None:
            return result
        return None

    def lookup_ssd_prefix(self, tokens: tuple[int, ...]) -> dict | None:
        """Find the longest prefix match in the SSD tier.

        Returns the longest-prefix entry metadata or None.
        """
        results = self._index.lookup_prefix(tokens)
        if results:
            return results[0]  # Already sorted by num_tokens DESC
        return None

    def lookup_candidate(self, tokens: tuple[int, ...]) -> dict | None:
        """Return one exact/prefix candidate and account for metadata misses."""
        tokens = tuple(tokens)
        candidate = self.lookup_ssd(tokens)
        if candidate is not None:
            candidate["match_type"] = "exact"
            candidate["matched_tokens"] = len(tokens)
            candidate["matched_key"] = tokens
            return candidate
        candidate = self.lookup_ssd_prefix(tokens)
        if candidate is not None:
            candidate["match_type"] = "prefix"
            candidate["matched_tokens"] = candidate["num_tokens"]
            candidate["matched_key"] = tokens[: candidate["num_tokens"]]
            return candidate
        self.record_lookup_miss()
        return None

    def validate_candidate(
        self, tokens: tuple[int, ...], candidate: dict
    ) -> tuple[int, ...] | None:
        """Validate candidate identity, token coverage, and entry path."""
        if not isinstance(candidate, dict):
            return None
        query = tuple(tokens)
        if not query:
            return None
        relative_path = candidate.get("file_path")
        if not _safe_entry_path(relative_path):
            return None

        memory_bytes = candidate.get("memory_bytes")
        if (
            not isinstance(memory_bytes, int)
            or isinstance(memory_bytes, bool)
            or memory_bytes <= 0
        ):
            return None
        matched_key = candidate.get("matched_key")
        if matched_key is None:
            matched_count = candidate.get("matched_tokens")
            if (
                not isinstance(matched_count, int)
                or isinstance(matched_count, bool)
                or matched_count <= 0
            ):
                return None
            matched_key = query[:matched_count]
        try:
            matched = tuple(matched_key)
        except TypeError:
            return None
        if not matched or len(matched) > len(query) or matched != query[: len(matched)]:
            return None
        num_tokens = candidate.get("num_tokens")
        if (
            not isinstance(num_tokens, int)
            or isinstance(num_tokens, bool)
            or num_tokens != len(matched)
        ):
            return None
        matched_count = candidate.get("matched_tokens")
        if matched_count is not None and matched_count != len(matched):
            return None
        match_type = candidate.get("match_type")
        if match_type is not None and match_type not in ("exact", "prefix"):
            return None
        if relative_path != self._entry_hash(matched):
            return None

        expected_identity = self.persistence_identity
        if expected_identity is not None:
            if candidate.get("persistence_identity") != expected_identity:
                return None
        return matched

    def read_validated_entry(
        self, tokens: tuple[int, ...], candidate: dict
    ) -> SSDReadResult | None:
        """Read one candidate after validating its exact identity and layout.

        The caller owns RAM reservation and any owner-thread reconstruction.
        This method performs only validated disk I/O and returns metadata
        needed for deterministic accounting.
        """
        matched = self.validate_candidate(tokens, candidate)
        if matched is None:
            self.record_promotion_failure()
            return None
        started = time.monotonic()
        layers = self._read_entry(
            matched,
            candidate["file_path"],
            expected_memory_bytes=candidate["memory_bytes"],
            expected_persistence_identity=candidate.get("persistence_identity"),
        )
        if layers is None:
            self.record_promotion_failure()
            return None
        return SSDReadResult(
            tokens=matched,
            file_path=candidate["file_path"],
            memory_bytes=candidate["memory_bytes"],
            layers=layers,
            read_bytes=self._entry_read_bytes(candidate["file_path"], len(layers)),
            latency_seconds=max(0.0, time.monotonic() - started),
        )

    def record_promotion_success(self, result: SSDReadResult) -> None:
        """Account and touch one result only after RAM publication succeeds."""
        if not isinstance(result, SSDReadResult):
            raise TypeError("promotion result must be SSDReadResult")
        with self._lock:
            self._stats.ssd_hits += 1
            self._stats.reload_latency_sum += result.latency_seconds
            self._stats.reload_bytes += result.read_bytes
        try:
            self._index.touch(result.tokens)
        except Exception:
            logger.debug("[ssd_cache] failed to touch promoted entry", exc_info=True)

    def _entry_read_bytes(self, relative_path: str, layer_count: int) -> int:
        total = 0
        for index in range(layer_count):
            try:
                total += os.path.getsize(
                    os.path.join(
                        self._data_dir,
                        relative_path,
                        f"layer_{index}.safetensors",
                    )
                )
            except OSError:
                pass
        return total

    def remove(self, tokens: tuple[int, ...]) -> bool:
        """Remove an entry from both the SSD index and its data directory."""
        import shutil

        meta = self._index.lookup_exact(tokens)
        if meta is None:
            return False

        # Remove the index first so a data-file cleanup failure cannot expose
        # the rejected entry to another promotion. Reconciliation removes any
        # orphaned directory left behind.
        self._index.delete_entry(tokens)
        entry_dir = os.path.join(self._data_dir, meta["file_path"])
        try:
            if os.path.exists(entry_dir):
                shutil.rmtree(entry_dir)
        except OSError:
            logger.warning(
                "[ssd_cache] failed to remove entry directory %s",
                meta["file_path"],
                exc_info=True,
            )
        return True

    async def async_promote(
        self,
        tokens: tuple[int, ...],
        reserve_budget_fn,
        release_budget_fn,
        *,
        record_success: bool = True,
        return_result: bool = False,
    ) -> list | SSDReadResult | None:
        """Promote an entry from SSD to RAM asynchronously.

        CRITICAL: Reserves RAM budget BEFORE the disk read, to avoid
        thrash when multiple promotions race.

        Args:
            tokens: Token sequence to promote.
            reserve_budget_fn: Callable(nbytes) -> bool. Must return True
                if budget is available and reserved, False otherwise.
            release_budget_fn: Callable(nbytes) -> None. Called to release
                budget on failure.

        Returns:
            List of deserialized cache layers, or None if promotion failed.
            When ``return_result`` is true, return the validated
            :class:`SSDReadResult` so a caller can publish it before recording
            a successful promotion.  Set ``record_success=False`` when that
            caller owns publication accounting.
        """
        import asyncio

        if not record_success and not return_result:
            raise ValueError(
                "return_result is required when the caller owns promotion accounting"
            )

        # Step 1: Look up and validate metadata (fast, SQLite)
        candidate = self.lookup_candidate(tokens)
        if candidate is None:
            return None
        matched = self.validate_candidate(tokens, candidate)
        if matched is None:
            self.record_promotion_failure()
            return None
        memory_bytes = candidate["memory_bytes"]

        # Step 2: Reserve RAM budget BEFORE disk read
        if not reserve_budget_fn(memory_bytes):
            self.record_promotion_failure()
            logger.warning(
                f"[ssd_cache] promotion denied: cannot reserve "
                f"{memory_bytes} bytes RAM budget"
            )
            return None

        # Step 3: Read from disk (in thread pool to avoid blocking event loop)
        # Use shield-and-await-on-cancel per CLAUDE.md Golden Rule #4:
        # budget must be released even if the calling task is cancelled.
        worker = asyncio.ensure_future(
            asyncio.to_thread(self.read_validated_entry, tokens, candidate)
        )
        try:
            read_result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Caller cancelled — still need to wait for the disk read
            # to finish, then release the budget
            try:
                await worker
            except Exception:
                pass
            release_budget_fn(memory_bytes)
            raise
        except Exception:
            # Release budget on read failure
            release_budget_fn(memory_bytes)
            self.record_promotion_failure()
            logger.exception(
                f"[ssd_cache] failed to read entry from disk "
                f"({candidate['num_tokens']} tokens)"
            )
            return None

        if read_result is None:
            # Corrupted entry — release budget; read_validated_entry recorded
            # one deterministic promotion failure and quarantined the entry.
            release_budget_fn(memory_bytes)
            return None

        if record_success:
            self.record_promotion_success(read_result)

        logger.info(
            f"[ssd_cache] promoted entry: {candidate['num_tokens']} tokens, "
            f"{read_result.read_bytes} bytes, "
            f"{read_result.latency_seconds * 1000:.1f}ms"
        )

        return read_result if return_result else read_result.layers

    def _read_entry(
        self,
        tokens: tuple[int, ...],
        relative_path: str,
        *,
        expected_memory_bytes: int | None = None,
        expected_persistence_identity: Mapping[str, str] | None = None,
    ) -> list | None:
        """Read and strictly validate one SSD entry before publication."""
        tokens = tuple(tokens)
        if not tokens or not _safe_entry_path(relative_path):
            return None
        if relative_path != self._entry_hash(tokens):
            logger.warning("[ssd_cache] entry path does not match token key")
            return None

        entry_dir = os.path.join(self._data_dir, relative_path)
        manifest_path = os.path.join(entry_dir, "manifest.json")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")

            expected_identity = self.persistence_identity
            if (
                expected_identity is not None
                and manifest.get("persistence_identity") != expected_identity
            ):
                if expected_persistence_identity == expected_identity:
                    # The index candidate was in this tier's namespace, so a
                    # missing/changed manifest identity is corruption. Quarantine
                    # before any layer can be published.
                    raise ValueError("manifest cache identity mismatch")
                # A shared tier may legitimately contain another model's
                # entry. Reject it without quarantining a valid foreign entry.
                logger.warning("[ssd_cache] entry cache identity mismatch")
                return None

            num_tokens = _strict_int(
                manifest.get("num_tokens"), "num_tokens", minimum=1
            )
            if num_tokens != len(tokens):
                raise ValueError("manifest token count differs from lookup key")
            memory_bytes = _strict_int(
                manifest.get("memory_bytes"), "memory_bytes", minimum=1
            )
            if (
                expected_memory_bytes is not None
                and memory_bytes != expected_memory_bytes
            ):
                raise ValueError("manifest memory accounting differs from index")
            num_layers = _strict_int(
                manifest.get("num_layers"), "num_layers", minimum=1
            )
            layer_manifests = manifest.get("layers")
            if (
                not isinstance(layer_manifests, list)
                or len(layer_manifests) != num_layers
            ):
                raise ValueError("manifest layer count is inconsistent")

            tokens_path = os.path.join(entry_dir, "tokens.bin")
            with open(tokens_path, "rb") as token_file:
                stored_tokens = _blob_to_tokens(token_file.read())
            if stored_tokens != tokens:
                raise ValueError("serialized token key differs from lookup key")

            cache_layers = []
            for layer_idx, layer_meta in enumerate(layer_manifests):
                if not isinstance(layer_meta, dict):
                    raise ValueError("layer metadata must be an object")
                layer_type = layer_meta.get("layer_type")
                if layer_type in ("KVCache", "RotatingKVCache"):
                    serializer = KVCacheSerializer()
                elif layer_type in ("ArraysCache", "MambaCache"):
                    serializer = ArraysCacheSerializer()
                else:
                    raise ValueError(f"unknown layer type {layer_type!r}")
                layer_path = os.path.join(entry_dir, f"layer_{layer_idx}.safetensors")
                file_digest = layer_meta.get("file_sha256")
                if (
                    not isinstance(file_digest, str)
                    or len(file_digest) != hashlib.sha256().digest_size * 2
                    or any(char not in "0123456789abcdef" for char in file_digest)
                    or _file_sha256(layer_path) != file_digest
                ):
                    raise ValueError("layer file digest does not match manifest")
                layer_data = serializer.deserialize_layer(layer_path, layer_meta)
                _validate_layer_manifest(layer_meta, layer_idx, layer_data)
                cache_layers.append(layer_data)
            return cache_layers
        except Exception as exc:
            logger.warning("[ssd_cache] corrupt entry %s: %s", relative_path, exc)
            self._quarantine_entry(tokens, relative_path)
            return None

    def _quarantine_entry(self, tokens: tuple[int, ...], relative_path: str) -> None:
        """Move a corrupt entry to quarantine and remove from index."""
        entry_dir = os.path.join(self._data_dir, relative_path)
        quarantine_dir = os.path.join(self._cache_dir, "quarantine", relative_path)

        try:
            if os.path.exists(entry_dir):
                os.makedirs(
                    os.path.dirname(quarantine_dir),
                    mode=self._config.dir_permissions,
                    exist_ok=True,
                )
                os.rename(entry_dir, quarantine_dir)
                logger.warning(
                    f"[ssd_cache] quarantined corrupt entry: {relative_path}"
                )
        except OSError as e:
            logger.warning(f"[ssd_cache] failed to quarantine {relative_path}: {e}")

        self._index.delete_entry(tokens)

    def _enforce_capacity(self) -> None:
        """Evict oldest SSD entries until within capacity limits.

        Called after each spill write. Removes entries by LRU order
        until both entry count and total bytes are within bounds.
        """
        import shutil

        while True:
            entry_count = self._index.get_entry_count()
            total_bytes = self._index.get_total_bytes()

            needs_evict = (
                entry_count > self._config.max_entries
                or total_bytes > self._config.max_size_bytes
            )
            if not needs_evict:
                break

            lru = self._index.get_lru(limit=1)
            if not lru:
                break

            victim = lru[0]
            victim_tokens = _blob_to_tokens(victim["tokens_blob"])
            victim_dir = os.path.join(self._data_dir, victim["file_path"])

            # Delete data files
            if os.path.exists(victim_dir):
                shutil.rmtree(victim_dir)

            # Delete from index
            self._index.delete_entry(victim_tokens)

            logger.debug(
                f"[ssd_cache] disk LRU evicted: {victim['num_tokens']} tokens, "
                f"{victim['memory_bytes']} bytes"
            )

    def reconcile(self) -> int:
        """Reconcile index with files on disk.

        Removes index entries whose data files are missing.
        Removes data directories not in the index.

        Returns number of entries cleaned up.
        """
        import shutil

        cleaned = 0

        # Phase 1: Remove index entries with missing data dirs
        all_entries = self._index.all_entries()
        for entry in all_entries:
            entry_dir = os.path.join(self._data_dir, entry["file_path"])
            manifest_path = os.path.join(entry_dir, "manifest.json")
            if not os.path.isdir(entry_dir) or not os.path.exists(manifest_path):
                tokens = _blob_to_tokens(entry["tokens_blob"])
                self._index.delete_entry(tokens)
                cleaned += 1
                logger.info(
                    f"[ssd_cache] reconcile: removed orphaned index entry "
                    f"({entry['num_tokens']} tokens, path={entry['file_path']})"
                )

        # Phase 2: Remove data directories not in the index
        if os.path.isdir(self._data_dir):
            indexed_hashes = {e["file_path"] for e in self._index.all_entries()}
            for entry_name in os.listdir(self._data_dir):
                entry_path = os.path.join(self._data_dir, entry_name)
                if (
                    os.path.isdir(entry_path)
                    and entry_name not in indexed_hashes
                    and not entry_name.endswith(".tmp")
                ):
                    shutil.rmtree(entry_path)
                    cleaned += 1
                    logger.info(
                        f"[ssd_cache] reconcile: removed orphaned data dir "
                        f"{entry_name}"
                    )

        if cleaned > 0:
            logger.info(f"[ssd_cache] reconciliation cleaned {cleaned} entries")

        return cleaned

    def close(self) -> None:
        """Drain pending spills, stop the writer, and release resources.

        If the writer does not terminate within the bounded join, retain both
        the thread reference and the SQLite index so a live writer can finish
        safely. A later close() call can retry the join.
        """
        with self._lifecycle_lock:
            if self._closed:
                return

            self._accepting_spills = False
            if self._writer_thread is not None:
                shutdown_deadline = time.monotonic() + self._WRITER_JOIN_TIMEOUT_S
                if not self._writer_shutdown_requested:
                    # FIFO ordering makes the poison pill a drain barrier: all
                    # spills accepted before shutdown are written first.
                    try:
                        self._spill_queue.put(None, timeout=self._WRITER_JOIN_TIMEOUT_S)
                    except queue.Full as exc:
                        raise TimeoutError(
                            "SSD cache shutdown sentinel could not be queued "
                            "before timeout"
                        ) from exc
                    self._writer_shutdown_requested = True

                join_timeout = max(0.0, shutdown_deadline - time.monotonic())
                self._writer_thread.join(timeout=join_timeout)
                if self._writer_thread.is_alive():
                    raise TimeoutError(
                        "SSD cache writer thread did not stop before timeout"
                    )
                self._writer_thread = None

            self._index.close()
            self._closed = True
        logger.info("[ssd_cache] SSDCacheTier closed")

    async def aclose(self) -> None:
        """Close without blocking the caller's asyncio event loop."""
        import asyncio

        await asyncio.to_thread(self.close)
