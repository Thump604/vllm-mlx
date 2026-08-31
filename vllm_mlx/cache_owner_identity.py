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
import importlib.metadata
import json
from pathlib import Path
import threading
from types import MappingProxyType
import uuid
from typing import Any, Callable, Mapping
import weakref

OWNER_API_VERSION = "vllm-mlx.model-cache-owner.v1"
IDENTITY_SCHEMA_VERSION = "specprefill.identity.v1"
IDENTITY_CANONICALIZATION = "rfc8785"
IDENTITY_DIGEST_ALGORITHM = "sha256"
IDENTITY_NUMBER_PROFILE = "rfc8785-ieee754-safe-integer-v1"
PREPARED_STORE_VERSION = "vllm-mlx.owner-bound-store.v1"

_HEX = frozenset("0123456789abcdef")
_ISSUED_VERIFIED_CONTEXTS: weakref.WeakSet[VerifiedCacheOwnerContext]


def _stable_provenance_digest(value: Mapping[str, str]) -> str:
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("owner provenance must contain only string keys and values")
    try:
        payload = json.dumps(
            dict(value),
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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field_name} must be lowercase SHA-256")


def _freeze_contract_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_contract_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_contract_value(item) for item in value)
    return value


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
    """Opaque handle for a cache-owned prepared publication record."""

    owner: ModelCacheOwnerBinding
    request: ModelCacheRequestBinding
    handle_id: str
    tokens: tuple[int, ...]
    memory_bytes: int
    version: str = PREPARED_STORE_VERSION

    def __reduce__(self):
        raise TypeError("prepared owner-bound entries are process-local")


@dataclass(frozen=True, slots=True)
class CacheOwnerGovernanceTarget:
    """Complete independent registry evidence for one loaded cache owner."""

    manifest: Mapping[str, Any]
    expected_model_cache_identity_digest: str
    expected_loaded_owner_digest: str
    expected_identity_fields: tuple[tuple[str, Any], ...]
    expected_persistence_identity: Mapping[str, str]
    runtime_composition_digest: str
    cache_namespace: str
    registry_source: str
    registry_complete: bool

    def __post_init__(self) -> None:
        from .specprefill_contract import freeze_identity_manifest

        frozen = freeze_identity_manifest(self.manifest)
        object.__setattr__(self, "manifest", frozen)
        object.__setattr__(
            self,
            "expected_identity_fields",
            tuple(
                (path, _freeze_contract_value(value))
                for path, value in self.expected_identity_fields
            ),
        )
        if not isinstance(self.expected_persistence_identity, Mapping):
            raise ValueError("expected persistence identity must be an object")
        persistence = dict(self.expected_persistence_identity)
        if set(persistence) != {"model", "tokenizer", "cache_layout"}:
            raise ValueError("expected persistence identity is incomplete")
        for name, value in persistence.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"expected persistence {name} must be lowercase SHA-256"
                )
            _require_sha256(value, field_name=f"expected persistence {name}")
        object.__setattr__(
            self, "expected_persistence_identity", MappingProxyType(persistence)
        )
        _require_sha256(
            self.expected_model_cache_identity_digest,
            field_name="expected model/cache identity",
        )
        _require_sha256(
            self.expected_loaded_owner_digest,
            field_name="expected loaded owner",
        )
        _require_sha256(
            self.runtime_composition_digest,
            field_name="runtime composition",
        )
        if not isinstance(self.cache_namespace, str) or not self.cache_namespace:
            raise ValueError("cache namespace must be a non-empty string")
        if not isinstance(self.registry_source, str) or not self.registry_source:
            raise ValueError("cache owner registry source must be non-empty")
        if self.registry_complete is not True:
            raise ValueError("cache owner registry must be complete")


class VerifiedCacheOwnerContext:
    """Opaque result of comparing the loaded owner with complete governance."""

    manifest: Mapping[str, Any]
    model_cache_identity_digest: str
    persistence_identity: Mapping[str, str]
    runtime_composition_digest: str
    cache_namespace: str
    registry_source: str

    __slots__ = (
        "manifest",
        "model_cache_identity_digest",
        "persistence_identity",
        "runtime_composition_digest",
        "cache_namespace",
        "registry_source",
        "__weakref__",
    )

    def __new__(cls, *args, **kwargs):
        raise TypeError("verified cache owner contexts are issued by verification")

    def __setattr__(self, name, value):
        raise TypeError("verified cache owner contexts are immutable")

    def __reduce__(self):
        raise TypeError("verified cache owner contexts are process-local")


_ISSUED_VERIFIED_CONTEXTS = weakref.WeakSet()


def _issue_verified_cache_owner_context(
    *,
    manifest: Mapping[str, Any],
    model_cache_identity_digest: str,
    persistence_identity: Mapping[str, str],
    runtime_composition_digest: str,
    cache_namespace: str,
    registry_source: str,
) -> VerifiedCacheOwnerContext:
    context = object.__new__(VerifiedCacheOwnerContext)
    object.__setattr__(context, "manifest", manifest)
    object.__setattr__(
        context, "model_cache_identity_digest", model_cache_identity_digest
    )
    object.__setattr__(
        context,
        "persistence_identity",
        MappingProxyType(dict(persistence_identity)),
    )
    object.__setattr__(
        context, "runtime_composition_digest", runtime_composition_digest
    )
    object.__setattr__(context, "cache_namespace", cache_namespace)
    object.__setattr__(context, "registry_source", registry_source)
    _ISSUED_VERIFIED_CONTEXTS.add(context)
    return context


def _sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"identity file must not be a symlink: {path.name}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"identity file changed while hashing: {path.name}")
    return digest.hexdigest()


def build_loaded_runtime_composition_digest() -> str:
    """Hash the loaded cache-owner implementation and dependency versions."""

    import importlib

    module_names = (
        "vllm_mlx.cache_owner_identity",
        "vllm_mlx.memory_cache",
        "vllm_mlx.mllm_batch_generator",
        "vllm_mlx.mllm_scheduler",
        "vllm_mlx.scheduler",
        "vllm_mlx.engine.batched",
        "vllm_mlx.models.mllm",
    )
    modules = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = getattr(module, "__file__", None)
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"loaded runtime module source is unavailable: {module_name}"
            )
        source_path = Path(source)
        if source_path.suffix == ".pyc":
            source_path = source_path.with_suffix(".py")
        modules[module_name] = _sha256_file(source_path.resolve())

    dependencies = {}
    for distribution in ("mlx", "mlx-lm", "mlx-vlm", "vllm-mlx"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"loaded runtime dependency version is unavailable: {distribution}"
            ) from exc
    return _stable_provenance_digest(
        {
            **{f"module:{name}": digest for name, digest in modules.items()},
            **{f"dependency:{name}": version for name, version in dependencies.items()},
        }
    )


def _required_file_digest(root: Path, relative_path: str, expected: str) -> None:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("identity file path must be a non-empty string")
    if not isinstance(expected, str):
        raise ValueError("identity file digest must be lowercase SHA-256")
    _require_sha256(expected, field_name="identity file digest")
    unresolved = root / relative_path
    if unresolved.is_symlink():
        raise ValueError(f"identity file must not be a symlink: {relative_path}")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("identity file path escapes the model artifact") from exc
    if not path.is_file() or _sha256_file(path) != expected:
        raise ValueError(f"loaded model artifact mismatch: {relative_path}")


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def build_loaded_cache_owner_digest(
    *,
    persistence_identity: Mapping[str, str],
    artifact_source: Mapping[str, Any],
    loaded_identity: Mapping[str, Any],
    runtime_mode: Mapping[str, Any],
    tokenizer_implementation: str,
    tokenizer_implementation_version: str,
) -> str:
    """Hash the actual loaded owner facts used by cache compatibility."""

    payload = {
        "persistence_identity": dict(persistence_identity),
        "artifact_source": {
            "model_id": artifact_source.get("model_id"),
            "revision": artifact_source.get("revision"),
        },
        "loaded_identity": dict(loaded_identity),
        "runtime_mode": dict(runtime_mode),
        "tokenizer": {
            "implementation": tokenizer_implementation,
            "implementation_version": tokenizer_implementation_version,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_loaded_model_cache_owner_context(
    model: Any,
    processor: Any,
    model_source: str,
    target: CacheOwnerGovernanceTarget,
    *,
    cache_config: Any,
    cache_runtime_identity: Mapping[str, Any],
    runtime_mode: Mapping[str, Any],
) -> VerifiedCacheOwnerContext:
    """Validate the loaded model and source bytes against governed identity.

    The governed digest is independent configuration.  The matching actual
    digest is returned only after the loaded implementation, mode, and every
    artifact file named by the manifest have been verified.
    """

    from .specprefill_contract import governed_target_identity_reason

    unresolved_root = Path(model_source).expanduser()
    if unresolved_root.is_symlink():
        raise ValueError("cache owner model source must not be a symlink")
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise ValueError("cache owner model source must be a local directory")
    if not isinstance(target, CacheOwnerGovernanceTarget):
        raise TypeError("cache owner target must be CacheOwnerGovernanceTarget")
    frozen = target.manifest
    cache_identity = frozen["model_cache_identity"]
    reason = governed_target_identity_reason(
        frozen,
        expected_model_cache_digest=target.expected_model_cache_identity_digest,
        expected_identity_fields=target.expected_identity_fields,
        registry_complete=target.registry_complete,
    )
    if reason is not None:
        raise ValueError(f"cache owner target is not governed: {reason}")

    if cache_identity.get("artifact_id") != root.name:
        raise ValueError("loaded model artifact ID does not match its directory")

    required_files = {
        "artifact-source.json": cache_identity.get("artifact_digest"),
        "config.json": cache_identity.get("config_digest"),
        "model.safetensors.index.json": cache_identity.get("weight_index_digest"),
        "tokenizer_config.json": cache_identity.get("tokenizer", {}).get(
            "config_digest"
        ),
        "chat_template.jinja": cache_identity.get("chat_template", {}).get("sha256"),
    }
    for relative_path, expected in required_files.items():
        _required_file_digest(root, relative_path, expected)
    for record in cache_identity.get("tokenizer", {}).get("files", ()):
        _required_file_digest(root, record["path"], record["sha256"])
    for record in cache_identity.get("vision", {}).get("processor_files", ()):
        _required_file_digest(root, record["path"], record["sha256"])

    from .memory_cache import build_cache_owner_persistence_identity

    language_model = getattr(model, "language_model", model)
    tokenizer = getattr(processor, "tokenizer", processor)
    actual_provenance = dict(
        build_cache_owner_persistence_identity(
            language_model,
            tokenizer,
            str(root),
            cache_config,
            cache_runtime_identity,
            processor,
        )
    )
    if actual_provenance != dict(target.expected_persistence_identity):
        raise ValueError(
            "loaded model/tokenizer/cache provenance does not match governance"
        )

    from .specprefill_contract import parse_identity_json

    artifact_source = parse_identity_json(
        (root / "artifact-source.json").read_text(encoding="utf-8")
    )
    if artifact_source.get("model_id") != cache_identity.get(
        "model_id"
    ) or artifact_source.get("revision") != cache_identity.get("model_revision"):
        raise ValueError("loaded model source identity does not match governance")

    language_model = getattr(model, "language_model", None)
    model_config = getattr(model, "config", None)
    architectures = _config_value(model_config, "architectures")
    loaded_architecture = (
        architectures[0]
        if isinstance(architectures, (list, tuple)) and architectures
        else type(model).__name__
    )
    loaded_values = {
        "architecture": loaded_architecture,
        "model_module": getattr(type(model), "__module__", None),
        "language_module": (
            getattr(type(language_model), "__module__", None)
            if language_model is not None
            else None
        ),
        "model_type": _config_value(model_config, "model_type"),
    }
    for field_name, actual in loaded_values.items():
        if actual != cache_identity.get(field_name):
            raise ValueError(f"loaded model {field_name} does not match governance")

    governed_mode = cache_identity.get("mode")
    if not isinstance(runtime_mode, Mapping) or not isinstance(governed_mode, Mapping):
        raise ValueError("cache owner runtime mode must be an object")
    for field_name, governed_value in governed_mode.items():
        if field_name == "capability_modes":
            continue
        if runtime_mode.get(field_name) != governed_value:
            raise ValueError(f"loaded runtime mode mismatch: {field_name}")

    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer_identity = cache_identity.get("tokenizer", {})
    implementation = type(tokenizer).__module__.split(".", 1)[0]
    if implementation != tokenizer_identity.get("implementation"):
        raise ValueError("loaded tokenizer implementation does not match governance")
    try:
        implementation_version = importlib.metadata.version(implementation)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("loaded tokenizer package version is unavailable") from exc
    if implementation_version != tokenizer_identity.get("implementation_version"):
        raise ValueError("loaded tokenizer version does not match governance")

    for probe in tokenizer_identity.get("encode_probes", ()):
        try:
            token_ids = tokenizer.encode(probe["text"], add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(probe["text"])
        if [int(token) for token in token_ids] != list(probe["ids"]):
            raise ValueError("loaded tokenizer probe does not match governance")

    if cache_identity.get("cache_schema", {}).get("sha256") != actual_provenance.get(
        "cache_layout"
    ):
        raise ValueError("loaded cache schema does not match governance")

    actual_loaded_owner_digest = build_loaded_cache_owner_digest(
        persistence_identity=actual_provenance,
        artifact_source=artifact_source,
        loaded_identity=loaded_values,
        runtime_mode=runtime_mode,
        tokenizer_implementation=implementation,
        tokenizer_implementation_version=implementation_version,
    )
    if actual_loaded_owner_digest != target.expected_loaded_owner_digest:
        raise ValueError("loaded cache owner digest does not match governance")
    actual_runtime_composition_digest = build_loaded_runtime_composition_digest()
    if actual_runtime_composition_digest != target.runtime_composition_digest:
        raise ValueError("loaded runtime composition does not match governance")

    return _issue_verified_cache_owner_context(
        manifest=frozen,
        model_cache_identity_digest=actual_loaded_owner_digest,
        persistence_identity=actual_provenance,
        runtime_composition_digest=actual_runtime_composition_digest,
        cache_namespace=target.cache_namespace,
        registry_source=target.registry_source,
    )


class CacheOwnerIdentity:
    """Authority minted by the object that owns a loaded model and cache."""

    def __init__(
        self,
        *,
        context: VerifiedCacheOwnerContext,
    ) -> None:
        if (
            not isinstance(context, VerifiedCacheOwnerContext)
            or context not in _ISSUED_VERIFIED_CONTEXTS
        ):
            raise ValueError("cache owner requires a verified loaded-model context")
        metadata = _manifest_metadata(context.manifest)
        self._manifest_metadata = metadata
        self._model_cache_identity_digest = context.model_cache_identity_digest
        self._owner_provenance_digest = _stable_provenance_digest(
            context.persistence_identity
        )
        self._cache_namespace = context.cache_namespace
        self._runtime_composition_digest = context.runtime_composition_digest
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

    def matches_verified_context(self, context: VerifiedCacheOwnerContext) -> bool:
        """Check repeat binding without accepting raw caller authorization."""

        if (
            not isinstance(context, VerifiedCacheOwnerContext)
            or context not in _ISSUED_VERIFIED_CONTEXTS
        ):
            return False
        try:
            metadata = _manifest_metadata(context.manifest)
        except ValueError:
            return False
        with self._lock:
            return bool(
                not self._closed
                and metadata == self._manifest_metadata
                and context.cache_namespace == self._cache_namespace
                and context.model_cache_identity_digest
                == self._model_cache_identity_digest
                and context.runtime_composition_digest
                == self._runtime_composition_digest
                and _stable_provenance_digest(context.persistence_identity)
                == self._owner_provenance_digest
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
    "CacheOwnerGovernanceTarget",
    "CacheOwnerIdentity",
    "ModelCacheOwnerBinding",
    "ModelCacheRequestBinding",
    "OwnerBindingDecision",
    "PREPARED_STORE_VERSION",
    "PreparedOwnerBoundCacheEntry",
    "VerifiedCacheOwnerContext",
    "build_loaded_cache_owner_digest",
    "build_loaded_runtime_composition_digest",
    "verify_loaded_model_cache_owner_context",
]
