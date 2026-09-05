# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, numpy-only persistence for hybrid prefix-cache snapshots.

MLX-backed values are converted to :class:`HybridCacheSnapshot` by the model
owner before this module is called.  The functions below perform only numpy,
JSON, hashing, and filesystem work so they are safe to run on an I/O thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl
import numpy as np
from safetensors.numpy import load_file, save_file

HYBRID_CACHE_PERSIST_VERSION = 2
_READABLE_HYBRID_CACHE_VERSIONS = {1, HYBRID_CACHE_PERSIST_VERSION}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_V1_SUPPORTED_LAYERS = {"KVCache", "RotatingKVCache", "ArraysCache"}
_SUPPORTED_LAYERS_BY_VERSION = {
    1: _V1_SUPPORTED_LAYERS,
    2: _V1_SUPPORTED_LAYERS | {"QSAKVCache"},
}
_SUPPORTED_AUXILIARY = {"last_logits"}
_SUPPORTED_ORIGINAL_DTYPES = {None, "bfloat16", "float16", "float32"}
_ABSOLUTE_MAX_TOKENS = 16 * 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSON_DEPTH = 16
_MAX_JSON_NODES = 100_000
_CACHE_LOCK_NAME = ".hybrid-cache.lock"
logger = logging.getLogger(__name__)


class HybridCachePersistenceError(ValueError):
    """A hybrid cache snapshot is incomplete, incompatible, or corrupt."""


@dataclass(frozen=True)
class HybridLayerSnapshot:
    layer_type: str
    tensors: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HybridEntrySnapshot:
    tokens: tuple[int, ...]
    memory_bytes: int
    layers: tuple[HybridLayerSnapshot, ...]
    auxiliary: dict[str, np.ndarray]
    auxiliary_original_dtypes: dict[str, str | None]


@dataclass(frozen=True)
class HybridCacheSnapshot:
    identity: dict[str, str]
    entries: tuple[HybridEntrySnapshot, ...]


@dataclass(frozen=True)
class LoadedHybridEntry:
    tokens: tuple[int, ...]
    layers: tuple[HybridLayerSnapshot, ...]
    auxiliary: dict[str, np.ndarray]
    auxiliary_original_dtypes: dict[str, str | None]


@dataclass(frozen=True)
class LoadedHybridCache:
    identity: dict[str, str]
    entries: tuple[LoadedHybridEntry, ...]


@contextmanager
def _cache_file_lock(root: Path, *, exclusive: bool):
    """Serialize cooperating persistence callers for one cache root.

    The cache directory is private runtime state, normally created with mode
    0700. As with POSIX advisory locks generally, a same-UID process that
    deliberately replaces the lock inode is outside this coordination
    contract.
    """
    lock_path = root / _CACHE_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    locked = False
    try:
        try:
            lock_stat = os.lstat(lock_path)
        except FileNotFoundError:
            lock_stat = None
        except OSError as exc:
            raise HybridCachePersistenceError(f"invalid cache lock: {exc}") from exc
        if lock_stat is not None and stat.S_ISLNK(lock_stat.st_mode):
            raise HybridCachePersistenceError("cache lock must not be a symlink")
        if lock_stat is not None and not stat.S_ISREG(lock_stat.st_mode):
            raise HybridCachePersistenceError("cache lock must be a regular file")

        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise HybridCachePersistenceError(f"invalid cache lock: {exc}") from exc
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise HybridCachePersistenceError("cache lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except OSError as exc:
            raise HybridCachePersistenceError(
                f"unable to acquire cache lock: {exc}"
            ) from exc
        locked = True
        yield
    finally:
        if descriptor is not None:
            if locked:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    logger.warning("Unable to release cache lock %s", lock_path)
            try:
                os.close(descriptor)
            except OSError:
                logger.warning("Unable to close cache lock %s", lock_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_bounded_file(path: Path, *, max_bytes: int, context: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HybridCachePersistenceError(f"invalid {context}: {exc}") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_bytes:
            raise HybridCachePersistenceError(f"invalid {context} size")
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise HybridCachePersistenceError(f"invalid {context} size")
        return payload
    finally:
        os.close(descriptor)


def _validate_json_bounds(value: Any) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise HybridCachePersistenceError("JSON structure exceeds safety bounds")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise HybridCachePersistenceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise HybridCachePersistenceError(f"invalid JSON constant: {value}")

    try:
        payload = _read_bounded_file(
            path, max_bytes=_MAX_JSON_BYTES, context=f"JSON file {path.name}"
        ).decode("utf-8")
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HybridCachePersistenceError(
            f"invalid JSON at {path.name}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HybridCachePersistenceError(f"{path.name} must contain an object")
    _validate_json_bounds(value)
    return value


def _require_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise HybridCachePersistenceError(
            f"{context} keys mismatch: expected={sorted(expected)} "
            f"actual={sorted(value)}"
        )


def _safe_component(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or not _SAFE_NAME.fullmatch(value)
    ):
        raise HybridCachePersistenceError(f"invalid {context}: {value!r}")
    return value


def _validate_identity(identity: Any) -> dict[str, str]:
    if not isinstance(identity, dict):
        raise HybridCachePersistenceError("identity must be an object")
    expected = {"model", "tokenizer", "cache_layout"}
    _require_keys(identity, expected, "identity")
    if any(not isinstance(identity[key], str) or not identity[key] for key in expected):
        raise HybridCachePersistenceError("identity values must be nonempty strings")
    return {key: identity[key] for key in sorted(expected)}


def _validate_snapshot(
    snapshot: HybridCacheSnapshot, *, version: int = HYBRID_CACHE_PERSIST_VERSION
) -> None:
    if type(version) is not int:
        raise HybridCachePersistenceError(
            f"unsupported hybrid cache version: {version!r}"
        )
    supported_layers = _SUPPORTED_LAYERS_BY_VERSION.get(version)
    if supported_layers is None:
        raise HybridCachePersistenceError(
            f"unsupported hybrid cache version: {version!r}"
        )
    _validate_identity(snapshot.identity)
    if not snapshot.entries:
        raise HybridCachePersistenceError("snapshot has no entries")
    for entry in snapshot.entries:
        if not entry.tokens or any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or token < 0
            or token > np.iinfo(np.int32).max
            for token in entry.tokens
        ):
            raise HybridCachePersistenceError(
                "entry tokens must be nonnegative int32 values"
            )
        if not entry.layers:
            raise HybridCachePersistenceError("entry has no cache layers")
        if set(entry.auxiliary) != _SUPPORTED_AUXILIARY:
            raise HybridCachePersistenceError(
                "hybrid entry must contain exactly auxiliary['last_logits']"
            )
        for name, array in entry.auxiliary.items():
            _safe_component(name, "auxiliary name")
            if not isinstance(array, np.ndarray) or array.size == 0:
                raise HybridCachePersistenceError(f"invalid auxiliary tensor: {name}")
            if array.dtype.kind != "f" or not np.isfinite(array).all():
                raise HybridCachePersistenceError(
                    f"invalid non-finite auxiliary tensor: {name}"
                )
            if name == "last_logits" and (array.ndim != 2 or array.shape[0] != 1):
                raise HybridCachePersistenceError("invalid last_logits shape")
        if set(entry.auxiliary_original_dtypes) != _SUPPORTED_AUXILIARY:
            raise HybridCachePersistenceError("auxiliary dtype metadata is incomplete")
        if any(
            dtype not in _SUPPORTED_ORIGINAL_DTYPES
            for dtype in entry.auxiliary_original_dtypes.values()
        ):
            raise HybridCachePersistenceError("auxiliary dtype metadata is invalid")
        for layer in entry.layers:
            if layer.layer_type not in supported_layers:
                raise HybridCachePersistenceError(
                    f"unsupported cache layer: {layer.layer_type}"
                )
            if not layer.tensors or any(
                not isinstance(array, np.ndarray) or array.size == 0
                for array in layer.tensors.values()
            ):
                raise HybridCachePersistenceError(
                    "layer tensors must be nonempty arrays"
                )
            _validate_layer_payload(
                layer.layer_type,
                layer.tensors,
                layer.metadata,
                version=version,
            )
        if (
            _entry_logical_nbytes(
                entry.layers, entry.auxiliary, entry.auxiliary_original_dtypes
            )
            != entry.memory_bytes
        ):
            raise HybridCachePersistenceError("entry memory byte count mismatch")


def _validate_layer_payload(
    layer_type: str,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    version: int = HYBRID_CACHE_PERSIST_VERSION,
) -> None:
    """Validate one layer before it can cross the owner-thread boundary."""
    if layer_type not in _SUPPORTED_LAYERS_BY_VERSION.get(version, set()):
        raise HybridCachePersistenceError(
            f"unsupported cache layer for version {version}: {layer_type}"
        )
    if layer_type in {"KVCache", "RotatingKVCache"}:
        allowed = {
            "offset",
            "keys_original_dtype",
            "values_original_dtype",
            "max_size",
            "keep",
            "step",
            "_idx",
        }
        if not isinstance(metadata, dict) or not set(metadata).issubset(allowed):
            raise HybridCachePersistenceError("invalid KV layer metadata")
        if set(tensors) != {"keys", "values"}:
            raise HybridCachePersistenceError("invalid KV layer tensors")
        keys = tensors["keys"]
        values = tensors["values"]
        if (
            keys.shape != values.shape
            or keys.ndim != 4
            or keys.dtype.kind != "f"
            or values.dtype.kind != "f"
        ):
            raise HybridCachePersistenceError("invalid KV layer shape or dtype")
        offset = metadata.get("offset")
        if not isinstance(offset, int) or offset < 1 or offset > keys.shape[-2]:
            raise HybridCachePersistenceError("invalid KV layer offset")
        for name in ("keys_original_dtype", "values_original_dtype"):
            if metadata.get(name) not in _SUPPORTED_ORIGINAL_DTYPES:
                raise HybridCachePersistenceError("invalid KV original dtype")
        return

    if layer_type == "ArraysCache":
        expected_metadata = {
            "num_arrays",
            "state_original_dtypes",
            "metadata_arrays",
            "metadata_original_dtypes",
            "meta_state",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata:
            raise HybridCachePersistenceError("invalid ArraysCache metadata")
        num_arrays = metadata["num_arrays"]
        names = metadata["metadata_arrays"]
        state_dtypes = metadata["state_original_dtypes"]
        metadata_dtypes = metadata["metadata_original_dtypes"]
        meta_state = metadata["meta_state"]
        if not isinstance(num_arrays, int) or num_arrays < 1:
            raise HybridCachePersistenceError("invalid ArraysCache arity")
        if (
            not isinstance(names, list)
            or len(names) != len(set(names))
            or any(name not in {"left_padding", "lengths"} for name in names)
        ):
            raise HybridCachePersistenceError("invalid ArraysCache metadata arrays")
        if (
            not isinstance(state_dtypes, list)
            or len(state_dtypes) != num_arrays
            or any(dtype not in _SUPPORTED_ORIGINAL_DTYPES for dtype in state_dtypes)
        ):
            raise HybridCachePersistenceError("invalid ArraysCache state dtypes")
        if (
            not isinstance(metadata_dtypes, dict)
            or set(metadata_dtypes) != set(names)
            or any(
                dtype not in _SUPPORTED_ORIGINAL_DTYPES
                for dtype in metadata_dtypes.values()
            )
        ):
            raise HybridCachePersistenceError("invalid ArraysCache metadata dtypes")
        if meta_state != "":
            raise HybridCachePersistenceError("invalid ArraysCache meta_state")
        expected_tensors = {f"state_{index}" for index in range(num_arrays)} | set(
            names
        )
        if set(tensors) != expected_tensors:
            raise HybridCachePersistenceError("invalid ArraysCache tensor names")
        for index in range(num_arrays):
            value = tensors[f"state_{index}"]
            allowed_kinds = {"f"}
            if version == 2 and num_arrays == 4 and index == 3:
                allowed_kinds.add("i")
            if value.size == 0 or value.dtype.kind not in allowed_kinds:
                raise HybridCachePersistenceError("invalid ArraysCache state tensor")
        for name in names:
            value = tensors[name]
            if value.ndim != 1 or value.dtype.kind not in {"i", "u"}:
                raise HybridCachePersistenceError("invalid ArraysCache metadata tensor")
        return

    if layer_type == "QSAKVCache":
        expected_metadata = {
            "num_arrays",
            "state_original_dtypes",
            "state_container",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata:
            raise HybridCachePersistenceError("invalid QSAKVCache metadata")
        num_arrays = metadata["num_arrays"]
        state_dtypes = metadata["state_original_dtypes"]
        if num_arrays != 4 or metadata["state_container"] != "tuple":
            raise HybridCachePersistenceError("invalid QSAKVCache state contract")
        if (
            not isinstance(state_dtypes, list)
            or len(state_dtypes) != num_arrays
            or any(dtype not in _SUPPORTED_ORIGINAL_DTYPES for dtype in state_dtypes)
        ):
            raise HybridCachePersistenceError("invalid QSAKVCache state dtypes")
        if set(tensors) != {f"state_{index}" for index in range(num_arrays)}:
            raise HybridCachePersistenceError("invalid QSAKVCache tensor names")
        keys, values, index_keys, position_ids = (
            tensors[f"state_{index}"] for index in range(num_arrays)
        )
        if (
            keys.ndim != 4
            or values.shape != keys.shape
            or keys.dtype.kind != "f"
            or values.dtype.kind != "f"
            or index_keys.ndim < 2
            or index_keys.dtype.kind != "f"
            or position_ids.ndim not in {2, 3}
            or position_ids.dtype.kind not in {"i", "u"}
            or index_keys.shape[0] != keys.shape[0]
            or index_keys.shape[1] != keys.shape[-2]
            or position_ids.shape[-1] != keys.shape[-2]
        ):
            raise HybridCachePersistenceError("invalid QSAKVCache state shape or dtype")
        return

    raise HybridCachePersistenceError(f"unsupported cache layer: {layer_type}")


def _logical_nbytes(array: np.ndarray, original_dtype: str | None) -> int:
    if original_dtype in {"bfloat16", "float16"}:
        return int(array.size) * 2
    if original_dtype == "float32":
        return int(array.size) * 4
    return int(array.nbytes)


def _entry_logical_nbytes(
    layers: tuple[HybridLayerSnapshot, ...] | list[HybridLayerSnapshot],
    auxiliary: dict[str, np.ndarray],
    auxiliary_original_dtypes: dict[str, str | None],
) -> int:
    total = _logical_nbytes(
        auxiliary["last_logits"], auxiliary_original_dtypes.get("last_logits")
    )
    for layer in layers:
        if layer.layer_type in {"KVCache", "RotatingKVCache"}:
            total += _logical_nbytes(
                layer.tensors["keys"], layer.metadata.get("keys_original_dtype")
            )
            total += _logical_nbytes(
                layer.tensors["values"], layer.metadata.get("values_original_dtype")
            )
            continue
        for index, dtype in enumerate(layer.metadata["state_original_dtypes"]):
            total += _logical_nbytes(layer.tensors[f"state_{index}"], dtype)
        if layer.layer_type == "QSAKVCache":
            continue
        for name, dtype in layer.metadata["metadata_original_dtypes"].items():
            total += _logical_nbytes(layer.tensors[name], dtype)
    return total


def _prune_old_generations(root: Path, current: str) -> None:
    for path in root.glob("generation-*"):
        if path.name == current:
            continue
        try:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError:
            logger.warning("Unable to remove stale cache generation %s", path.name)


def write_hybrid_snapshot(cache_dir: str, snapshot: HybridCacheSnapshot) -> bool:
    """Atomically publish a complete immutable generation."""
    _validate_snapshot(snapshot, version=HYBRID_CACHE_PERSIST_VERSION)
    root = Path(cache_dir)
    if root.is_symlink():
        raise HybridCachePersistenceError("cache root must not be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise HybridCachePersistenceError("cache root must be a directory")
    with _cache_file_lock(root, exclusive=True):
        return _write_hybrid_snapshot_locked(root, snapshot)


def _write_hybrid_snapshot_locked(root: Path, snapshot: HybridCacheSnapshot) -> bool:
    generation = f"generation-{uuid.uuid4().hex}"
    temporary = root / f".{generation}.tmp"
    published = root / generation
    temporary.mkdir(mode=0o700)
    pointer_tmp = None

    try:
        entry_records = []
        for entry_index, entry in enumerate(snapshot.entries):
            entry_dir = temporary / f"entry-{entry_index}"
            entry_dir.mkdir(mode=0o700)
            tokens_path = entry_dir / "tokens.bin"
            token_array = np.asarray(entry.tokens, dtype="<i4")
            _write_bytes(tokens_path, token_array.tobytes(order="C"))

            layer_records = []
            for layer_index, layer in enumerate(entry.layers):
                layer_path = entry_dir / f"layer-{layer_index}.safetensors"
                save_file(layer.tensors, str(layer_path))
                os.chmod(layer_path, 0o600)
                _sync_file(layer_path)
                layer_records.append(
                    {
                        "file": layer_path.name,
                        "sha256": _sha256(layer_path),
                        "layer_type": layer.layer_type,
                        "metadata": layer.metadata,
                    }
                )

            auxiliary_path = entry_dir / "auxiliary.safetensors"
            save_file(entry.auxiliary, str(auxiliary_path))
            os.chmod(auxiliary_path, 0o600)
            _sync_file(auxiliary_path)
            _sync_directory(entry_dir)
            entry_records.append(
                {
                    "directory": entry_dir.name,
                    "num_tokens": len(entry.tokens),
                    "memory_bytes": entry.memory_bytes,
                    "tokens_sha256": _sha256(tokens_path),
                    "auxiliary_file": auxiliary_path.name,
                    "auxiliary_sha256": _sha256(auxiliary_path),
                    "auxiliary_original_dtypes": entry.auxiliary_original_dtypes,
                    "layers": layer_records,
                }
            )

        manifest = {
            "version": HYBRID_CACHE_PERSIST_VERSION,
            "identity": _validate_identity(snapshot.identity),
            "entries": entry_records,
        }
        manifest_path = temporary / "manifest.json"
        _write_bytes(manifest_path, _canonical_json(manifest))
        os.chmod(manifest_path, 0o600)
        _sync_directory(temporary)
        os.replace(temporary, published)
        _sync_directory(root)

        pointer = {
            "version": HYBRID_CACHE_PERSIST_VERSION,
            "generation": generation,
            "manifest_sha256": _sha256(published / "manifest.json"),
        }
        pointer_tmp = root / f".index-{uuid.uuid4().hex}.tmp"
        _write_bytes(pointer_tmp, _canonical_json(pointer))
        os.chmod(pointer_tmp, 0o600)
        os.replace(pointer_tmp, root / "index.json")
        _sync_directory(root)
        _prune_old_generations(root, generation)
        _sync_directory(root)
        return True
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if pointer_tmp is not None:
            pointer_tmp.unlink(missing_ok=True)
        raise


def _checked_file(parent: Path, name: Any, checksum: Any, context: str) -> Path:
    component = _safe_component(name, f"{context} file")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise HybridCachePersistenceError(f"invalid {context} checksum")
    path = parent / component
    if path.is_symlink() or not path.is_file():
        raise HybridCachePersistenceError(f"missing {context} file: {component}")
    if path.resolve().parent != parent.resolve():
        raise HybridCachePersistenceError(f"escaped {context} file: {component}")
    if _sha256(path) != checksum:
        raise HybridCachePersistenceError(f"{context} checksum mismatch")
    return path


def read_hybrid_snapshot(
    cache_dir: str,
    max_memory_bytes: int | None = None,
    max_entries: int | None = None,
    max_tokens: int | None = None,
) -> LoadedHybridCache | None:
    """Load and fully validate one committed generation without importing MLX."""
    root = Path(cache_dir)
    if root.is_symlink():
        raise HybridCachePersistenceError("cache root must not be a symlink")
    if not root.is_dir():
        return None
    # Preserve the historical empty-cache read contract without requiring
    # write access merely to report a miss. A cooperating writer creates the
    # lock before starting publication, so the presence of either the lock or
    # an existing index requires the normal shared-lock path below.
    lock_path = root / _CACHE_LOCK_NAME
    pointer_path = root / "index.json"
    if not os.path.lexists(lock_path) and not os.path.lexists(pointer_path):
        return None
    with _cache_file_lock(root, exclusive=False):
        return _read_hybrid_snapshot_locked(
            root,
            max_memory_bytes=max_memory_bytes,
            max_entries=max_entries,
            max_tokens=max_tokens,
        )


def _read_hybrid_snapshot_locked(
    root: Path,
    *,
    max_memory_bytes: int | None,
    max_entries: int | None,
    max_tokens: int | None,
) -> LoadedHybridCache | None:
    pointer_path = root / "index.json"
    if pointer_path.is_symlink():
        raise HybridCachePersistenceError("cache index must not be a symlink")
    if not pointer_path.is_file():
        return None
    pointer = _load_json(pointer_path)
    _require_keys(pointer, {"version", "generation", "manifest_sha256"}, "index")
    version = pointer["version"]
    if type(version) is not int or version not in _READABLE_HYBRID_CACHE_VERSIONS:
        raise HybridCachePersistenceError(
            f"unsupported hybrid cache version: {version!r}"
        )
    generation = _safe_component(pointer["generation"], "generation")
    generation_dir = root / generation
    if (
        generation_dir.is_symlink()
        or not generation_dir.is_dir()
        or generation_dir.resolve().parent != root.resolve()
    ):
        raise HybridCachePersistenceError("missing or invalid generation directory")
    manifest_path = _checked_file(
        generation_dir,
        "manifest.json",
        pointer["manifest_sha256"],
        "manifest",
    )
    manifest = _load_json(manifest_path)
    _require_keys(manifest, {"version", "identity", "entries"}, "manifest")
    if type(manifest["version"]) is not int or manifest["version"] != version:
        raise HybridCachePersistenceError("manifest version mismatch")
    identity = _validate_identity(manifest["identity"])
    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise HybridCachePersistenceError("manifest entries must be nonempty")
    if max_entries is not None and len(raw_entries) > max_entries:
        raise HybridCachePersistenceError("snapshot exceeds restore entry budget")
    token_limit = _ABSOLUTE_MAX_TOKENS
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise HybridCachePersistenceError("invalid restore token budget")
        token_limit = min(token_limit, max_tokens)
    if max_memory_bytes is not None:
        # A Python int plus its tuple slot costs about 36 bytes on CPython.
        # Use 40 so the materialized token tuple cannot escape the cache cap.
        token_limit = min(token_limit, max(1, max_memory_bytes // 40))
    declared_memory: list[int] = []
    declared_tokens: list[int] = []
    for record in raw_entries:
        value = record.get("memory_bytes") if isinstance(record, dict) else None
        if not isinstance(value, int) or value < 1:
            raise HybridCachePersistenceError("invalid declared memory byte count")
        declared_memory.append(value)
        num_tokens = record.get("num_tokens") if isinstance(record, dict) else None
        if not isinstance(num_tokens, int) or num_tokens < 1:
            raise HybridCachePersistenceError("invalid token count")
        declared_tokens.append(num_tokens)
    if max_memory_bytes is not None and sum(declared_memory) > max_memory_bytes:
        raise HybridCachePersistenceError("snapshot exceeds restore memory budget")
    if sum(declared_tokens) > token_limit:
        raise HybridCachePersistenceError("snapshot exceeds restore token budget")

    loaded_entries = []
    seen_entry_directories: set[str] = set()
    disk_tensor_bytes = 0
    # BF16 values are staged as FP32 because numpy/safetensors has no portable
    # BF16 array representation. Keep that conversion strictly bounded.
    disk_tensor_budget = (
        None if max_memory_bytes is None else (2 * max_memory_bytes) + 1024 * 1024
    )
    for entry_index, record in enumerate(raw_entries):
        if not isinstance(record, dict):
            raise HybridCachePersistenceError("entry record must be an object")
        _require_keys(
            record,
            {
                "directory",
                "num_tokens",
                "memory_bytes",
                "tokens_sha256",
                "auxiliary_file",
                "auxiliary_sha256",
                "auxiliary_original_dtypes",
                "layers",
            },
            f"entry {entry_index}",
        )
        entry_component = _safe_component(record["directory"], "entry directory")
        if entry_component in seen_entry_directories:
            raise HybridCachePersistenceError("duplicate entry directory")
        seen_entry_directories.add(entry_component)
        entry_dir = generation_dir / entry_component
        if (
            entry_dir.is_symlink()
            or not entry_dir.is_dir()
            or entry_dir.resolve().parent != generation_dir.resolve()
        ):
            raise HybridCachePersistenceError("missing or invalid entry directory")
        num_tokens = record["num_tokens"]
        if (
            not isinstance(num_tokens, int)
            or num_tokens < 1
            or num_tokens > token_limit
        ):
            raise HybridCachePersistenceError("invalid token count")
        if not isinstance(record["memory_bytes"], int) or record["memory_bytes"] < 1:
            raise HybridCachePersistenceError("invalid memory byte count")
        if max_memory_bytes is not None and record["memory_bytes"] > max_memory_bytes:
            raise HybridCachePersistenceError("entry exceeds restore memory budget")
        tokens_path = _checked_file(
            entry_dir, "tokens.bin", record["tokens_sha256"], "tokens"
        )
        disk_tensor_bytes += tokens_path.stat().st_size
        if disk_tensor_budget is not None and disk_tensor_bytes > disk_tensor_budget:
            raise HybridCachePersistenceError(
                "snapshot files exceed restore memory budget"
            )
        token_bytes = tokens_path.read_bytes()
        if len(token_bytes) != num_tokens * 4:
            raise HybridCachePersistenceError("token file length mismatch")
        tokens = tuple(int(token) for token in np.frombuffer(token_bytes, dtype="<i4"))
        if any(token < 0 for token in tokens):
            raise HybridCachePersistenceError("invalid negative token id")

        layers = []
        seen_layer_files: set[str] = set()
        raw_layers = record["layers"]
        if not isinstance(raw_layers, list) or not raw_layers:
            raise HybridCachePersistenceError("entry layers must be nonempty")
        for layer_index, layer_record in enumerate(raw_layers):
            if not isinstance(layer_record, dict):
                raise HybridCachePersistenceError("layer record must be an object")
            _require_keys(
                layer_record,
                {"file", "sha256", "layer_type", "metadata"},
                f"layer {layer_index}",
            )
            layer_type = layer_record["layer_type"]
            if layer_type not in _SUPPORTED_LAYERS_BY_VERSION[version]:
                raise HybridCachePersistenceError(
                    f"unsupported cache layer: {layer_type!r}"
                )
            layer_path = _checked_file(
                entry_dir,
                layer_record["file"],
                layer_record["sha256"],
                f"layer {layer_index}",
            )
            if layer_path.name in seen_layer_files:
                raise HybridCachePersistenceError("duplicate layer file")
            seen_layer_files.add(layer_path.name)
            disk_tensor_bytes += layer_path.stat().st_size
            if (
                disk_tensor_budget is not None
                and disk_tensor_bytes > disk_tensor_budget
            ):
                raise HybridCachePersistenceError(
                    "snapshot files exceed restore memory budget"
                )
            tensors = load_file(str(layer_path))
            if not tensors or any(array.size == 0 for array in tensors.values()):
                raise HybridCachePersistenceError("invalid empty layer tensor")
            metadata = layer_record["metadata"]
            _validate_layer_payload(
                layer_type,
                tensors,
                metadata,
                version=version,
            )
            layers.append(HybridLayerSnapshot(layer_type, tensors, metadata))

        auxiliary_path = _checked_file(
            entry_dir,
            record["auxiliary_file"],
            record["auxiliary_sha256"],
            "auxiliary",
        )
        if auxiliary_path.name in seen_layer_files:
            raise HybridCachePersistenceError("auxiliary aliases a layer file")
        disk_tensor_bytes += auxiliary_path.stat().st_size
        if disk_tensor_budget is not None and disk_tensor_bytes > disk_tensor_budget:
            raise HybridCachePersistenceError(
                "snapshot files exceed restore memory budget"
            )
        auxiliary = load_file(str(auxiliary_path))
        if set(auxiliary) != _SUPPORTED_AUXILIARY or auxiliary["last_logits"].size == 0:
            raise HybridCachePersistenceError(
                "hybrid entry must contain exactly auxiliary['last_logits']"
            )
        auxiliary_dtypes = record["auxiliary_original_dtypes"]
        if (
            not isinstance(auxiliary_dtypes, dict)
            or set(auxiliary_dtypes) != _SUPPORTED_AUXILIARY
            or any(
                value not in _SUPPORTED_ORIGINAL_DTYPES
                for value in auxiliary_dtypes.values()
            )
        ):
            raise HybridCachePersistenceError("invalid auxiliary dtype metadata")
        logits = auxiliary["last_logits"]
        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.dtype.kind != "f"
            or not np.isfinite(logits).all()
        ):
            raise HybridCachePersistenceError("invalid last_logits shape or dtype")

        logical_bytes = _entry_logical_nbytes(layers, auxiliary, dict(auxiliary_dtypes))
        if logical_bytes != record["memory_bytes"]:
            raise HybridCachePersistenceError("entry memory byte count mismatch")
        loaded_entries.append(
            LoadedHybridEntry(tokens, tuple(layers), auxiliary, dict(auxiliary_dtypes))
        )

    return LoadedHybridCache(identity, tuple(loaded_entries))
