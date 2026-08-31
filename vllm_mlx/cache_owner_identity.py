# SPDX-License-Identifier: Apache-2.0
"""Process-local authority for binding work to one live model/cache owner.

Persistent model/cache identity answers whether detached state is compatible
across restarts.  It is deliberately not authority for live work.  This
module adds the complementary process-local capability.  Capabilities are
never serialized; a compatible snapshot restored by a new owner must be
rebound by that new owner before it can authorize work.

This is a cooperative in-process ownership contract, not a security boundary
against malicious code executing in the same Python process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import threading
from types import MappingProxyType
import uuid
from typing import Any, Callable, Mapping

OWNER_API_VERSION = "vllm-mlx.model-cache-owner.v1"
IDENTITY_SCHEMA_VERSION = "specprefill.identity.v1"
IDENTITY_CANONICALIZATION = "rfc8785"
IDENTITY_DIGEST_ALGORITHM = "sha256"
IDENTITY_NUMBER_PROFILE = "rfc8785-ieee754-safe-integer-v1"
PREPARED_STORE_VERSION = "vllm-mlx.owner-bound-store.v1"

_HEX = frozenset("0123456789abcdef")


def _stable_provenance_digest(value: Mapping[str, str]) -> str:
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("owner provenance must contain only string keys and values")
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("owner provenance must be stable JSON data") from exc
    return hashlib.sha256(payload).hexdigest()


def _manifest_metadata(manifest: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    if not isinstance(manifest, MappingProxyType):
        raise ValueError("governed identity manifest must be validated and frozen")
    for key in (
        "model_cache_identity",
        "request_protocol_identity",
        "draft_compatibility",
    ):
        if not isinstance(manifest.get(key), MappingProxyType):
            raise ValueError("governed identity manifest must be validated and frozen")
    values = (
        manifest.get("schema_version"),
        manifest.get("canonicalization"),
        manifest.get("digest_algorithm"),
        manifest.get("number_profile"),
        manifest.get("digest"),
    )
    expected = (
        IDENTITY_SCHEMA_VERSION,
        IDENTITY_CANONICALIZATION,
        IDENTITY_DIGEST_ALGORITHM,
        IDENTITY_NUMBER_PROFILE,
    )
    if values[:4] != expected:
        raise ValueError("governed identity manifest metadata is unsupported")
    digest = values[4]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _HEX for character in digest)
    ):
        raise ValueError("governed identity manifest digest must be lowercase SHA-256")
    return values  # type: ignore[return-value]


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{field_name} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class OwnerBindingDecision:
    """Typed result used by owner validation and owner-bound publication."""

    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        allowed = {"none", "cache_unsafe", "runtime_error", "cancellation"}
        if self.reason not in allowed or self.accepted != (self.reason == "none"):
            raise ValueError("invalid owner binding decision")


_ACCEPTED = OwnerBindingDecision(True, "none")
_CACHE_UNSAFE = OwnerBindingDecision(False, "cache_unsafe")
_RUNTIME_ERROR = OwnerBindingDecision(False, "runtime_error")
_CANCELLED = OwnerBindingDecision(False, "cancellation")


@dataclass(frozen=True, slots=True, eq=False)
class ModelCacheOwnerBinding:
    """Opaque immutable handle for one owner namespace and lifecycle epoch."""

    schema_version: str
    canonicalization: str
    digest_algorithm: str
    number_profile: str
    manifest_digest: str
    model_cache_identity_digest: str
    owner_provenance_digest: str
    cache_namespace: str
    lifecycle_epoch: int
    runtime_composition_digest: str
    api_version: str = OWNER_API_VERSION
    _capability: object = field(repr=False, compare=False, hash=False, default=None)

    def __reduce__(self):
        raise TypeError("model/cache owner bindings are process-local")


@dataclass(frozen=True, slots=True, eq=False)
class ModelCacheRequestBinding:
    """Owner-minted request lease; the UID is never an external request ID."""

    owner: ModelCacheOwnerBinding
    stable_request_uid: str
    sequence_revision: int
    _capability: object = field(repr=False, compare=False, hash=False, default=None)

    def __reduce__(self):
        raise TypeError("model/cache request bindings are process-local")


@dataclass(frozen=True, slots=True, eq=False)
class PreparedOwnerBoundCacheEntry:
    """Public wrapper around the private cache entry publication boundary."""

    owner: ModelCacheOwnerBinding
    request: ModelCacheRequestBinding
    version: str = PREPARED_STORE_VERSION
    _entry: Any = field(repr=False, compare=False, hash=False, default=None)

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._entry.tokens)

    @property
    def memory_bytes(self) -> int:
        return int(self._entry.memory_bytes)

    def __reduce__(self):
        raise TypeError("prepared owner-bound entries are process-local")


class CacheOwnerIdentity:
    """Authority minted by the object that owns a loaded model and cache."""

    def __init__(
        self,
        *,
        cache_namespace: str,
        actual_provenance: Mapping[str, str],
        governed_manifest: Mapping[str, Any],
        actual_model_cache_identity_digest: str,
        governed_model_cache_identity_digest: str,
        runtime_composition_digest: str,
    ) -> None:
        if not isinstance(cache_namespace, str) or not cache_namespace:
            raise ValueError("cache namespace must be a non-empty string")
        if not isinstance(runtime_composition_digest, str):
            raise ValueError("runtime composition must be lowercase SHA-256")
        _require_sha256(runtime_composition_digest, field_name="runtime composition")
        for name, value in (
            ("actual model/cache identity", actual_model_cache_identity_digest),
            ("governed model/cache identity", governed_model_cache_identity_digest),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be lowercase SHA-256")
            _require_sha256(value, field_name=name)
        if actual_model_cache_identity_digest != governed_model_cache_identity_digest:
            raise ValueError("governed manifest does not match the loaded model/cache")
        metadata = _manifest_metadata(governed_manifest)
        if governed_manifest.get("role") != "target":
            raise ValueError("governed identity manifest must describe the target")
        self._manifest_metadata = metadata
        self._model_cache_identity_digest = actual_model_cache_identity_digest
        self._owner_provenance_digest = _stable_provenance_digest(actual_provenance)
        self._cache_namespace = cache_namespace
        self._runtime_composition_digest = runtime_composition_digest
        self._lock = threading.RLock()
        self._epoch = 0
        self._capability = object()
        self._binding: ModelCacheOwnerBinding | None = None
        self._request_states: dict[
            object, tuple[str, str, int, ModelCacheOwnerBinding]
        ] = {}
        self._closed = False

    def mint_owner_binding(self) -> ModelCacheOwnerBinding:
        with self._lock:
            if self._closed:
                raise RuntimeError("model/cache owner is closed")
            if self._binding is None:
                schema, canonicalization, algorithm, number_profile, digest = (
                    self._manifest_metadata
                )
                self._binding = ModelCacheOwnerBinding(
                    schema_version=schema,
                    canonicalization=canonicalization,
                    digest_algorithm=algorithm,
                    number_profile=number_profile,
                    manifest_digest=digest,
                    model_cache_identity_digest=self._model_cache_identity_digest,
                    owner_provenance_digest=self._owner_provenance_digest,
                    cache_namespace=self._cache_namespace,
                    lifecycle_epoch=self._epoch,
                    runtime_composition_digest=self._runtime_composition_digest,
                    _capability=self._capability,
                )
            return self._binding

    def validate_owner_binding(
        self, binding: ModelCacheOwnerBinding | Any
    ) -> OwnerBindingDecision:
        with self._lock:
            if self._closed or not isinstance(binding, ModelCacheOwnerBinding):
                return _CACHE_UNSAFE
            if (
                self._binding is None
                or binding._capability is not self._capability
                or binding is not self._binding
            ):
                return _CACHE_UNSAFE
            return _ACCEPTED

    def matches_governed_identity(
        self,
        manifest: Mapping[str, Any],
        *,
        cache_namespace: str,
        model_cache_identity_digest: str,
        runtime_composition_digest: str,
    ) -> bool:
        """Check repeat binding input without weakening the frozen contract."""

        try:
            metadata = _manifest_metadata(manifest)
        except ValueError:
            return False
        with self._lock:
            return bool(
                not self._closed
                and manifest.get("role") == "target"
                and metadata == self._manifest_metadata
                and cache_namespace == self._cache_namespace
                and model_cache_identity_digest == self._model_cache_identity_digest
                and runtime_composition_digest == self._runtime_composition_digest
            )

    def mint_request_binding(self, sequence_revision: int) -> ModelCacheRequestBinding:
        if (
            not isinstance(sequence_revision, int)
            or isinstance(sequence_revision, bool)
            or sequence_revision < 0
        ):
            raise ValueError("sequence revision must be a non-negative integer")
        with self._lock:
            owner = self.mint_owner_binding()
            capability = object()
            binding = ModelCacheRequestBinding(
                owner=owner,
                stable_request_uid=uuid.uuid4().hex,
                sequence_revision=sequence_revision,
                _capability=capability,
            )
            self._request_states[capability] = (
                "active",
                binding.stable_request_uid,
                binding.sequence_revision,
                binding.owner,
            )
            return binding

    def validate_request_binding(
        self, binding: ModelCacheRequestBinding | Any
    ) -> OwnerBindingDecision:
        with self._lock:
            if not isinstance(binding, ModelCacheRequestBinding):
                return _RUNTIME_ERROR
            owner_decision = self.validate_owner_binding(binding.owner)
            if not owner_decision.accepted:
                return owner_decision
            record = self._request_states.get(binding._capability)
            if record is None:
                return _CACHE_UNSAFE
            state, stable_uid, sequence_revision, owner = record
            if (
                stable_uid != binding.stable_request_uid
                or sequence_revision != binding.sequence_revision
                or owner is not binding.owner
            ):
                return _CACHE_UNSAFE
            if state == "active":
                return _ACCEPTED
            if state == "cancelled":
                return _CANCELLED
            return _CACHE_UNSAFE

    def cancel_request(self, binding: ModelCacheRequestBinding) -> OwnerBindingDecision:
        with self._lock:
            decision = self.validate_request_binding(binding)
            if decision.accepted:
                _, stable_uid, sequence_revision, owner = self._request_states[
                    binding._capability
                ]
                self._request_states[binding._capability] = (
                    "cancelled",
                    stable_uid,
                    sequence_revision,
                    owner,
                )
                return _CANCELLED
            return decision

    def _commit_request(
        self,
        binding: ModelCacheRequestBinding,
        publish: Callable[[], bool],
    ) -> OwnerBindingDecision:
        """Internal transaction for owner-aware cache publication."""

        with self._lock:
            decision = self.validate_request_binding(binding)
            if not decision.accepted:
                return decision
            try:
                stored = bool(publish())
            except Exception:
                return _RUNTIME_ERROR
            decision = self.validate_request_binding(binding)
            if not decision.accepted:
                return decision
            return _ACCEPTED if stored else _RUNTIME_ERROR

    def release_request(
        self, binding: ModelCacheRequestBinding
    ) -> OwnerBindingDecision:
        with self._lock:
            if not isinstance(binding, ModelCacheRequestBinding):
                return _RUNTIME_ERROR
            owner_decision = self.validate_owner_binding(binding.owner)
            if not owner_decision.accepted:
                return owner_decision
            record = self._request_states.get(binding._capability)
            if record is None:
                return _CACHE_UNSAFE
            _, stable_uid, sequence_revision, owner = record
            if (
                stable_uid != binding.stable_request_uid
                or sequence_revision != binding.sequence_revision
                or owner is not binding.owner
            ):
                return _CACHE_UNSAFE
            self._request_states.pop(binding._capability, None)
            return _ACCEPTED

    def invalidate(self, *, cache_namespace: str | None = None) -> None:
        """Advance the lifecycle and revoke every owner and request handle."""

        with self._lock:
            if cache_namespace is not None:
                if not isinstance(cache_namespace, str) or not cache_namespace:
                    raise ValueError("cache namespace must be a non-empty string")
                self._cache_namespace = cache_namespace
            self._epoch += 1
            self._capability = object()
            self._binding = None
            self._request_states.clear()

    def close(self) -> None:
        with self._lock:
            self.invalidate()
            self._closed = True


__all__ = [
    "CacheOwnerIdentity",
    "ModelCacheOwnerBinding",
    "ModelCacheRequestBinding",
    "OwnerBindingDecision",
    "PREPARED_STORE_VERSION",
    "PreparedOwnerBoundCacheEntry",
]
