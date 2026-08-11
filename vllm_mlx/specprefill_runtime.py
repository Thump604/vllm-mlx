# SPDX-License-Identifier: Apache-2.0
"""Eager, fail-atomic Qwen continuous-batching SpecPrefill runtime builder."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

from .cooperative_specprefill import (
    CooperativeSpecPrefillConfig,
    CooperativeSpecPrefillSession,
)
from .engine.batched import PreparedMLLMSpecPrefillRuntime
from .mllm_batch_generator import (
    GemmaSparseBatchConfig,
    MLLMTargetForwardPhase,
    prepare_gemma_sparse_target,
)
from .mllm_scheduler import MLLMSpecPrefillAdmission, MLLMSpecPrefillCacheCapability
from .specprefill import SpecPrefillScorer
from .specprefill_cache import (
    SparseCacheExecutionConfig,
    SparseCacheState,
    SparsePolicyTuning,
)
from .specprefill_gemma_cache import (
    GEMMA4_ARTIFACTS,
    GemmaArtifactSpec,
    validate_aligned_scalar_cache,
)
from .specprefill_positions import (
    TargetPositionFamily,
    decode_plan,
    resolve_target_position_adapter,
)
from .specprefill_profiles import (
    SpecPrefillProfileKey,
    SpecPrefillProfileRegistry,
    SpecPrefillProfileTier,
    SpecPrefillTuning,
)
from .specprefill_scorer_session import SpecPrefillScorerSession
from .specprefill_selection import SPECPREFILL_SELECTOR_VERSION
from .specprefill_target_executor import SparseTargetPrefillSession
from .specprefill_target_hooks import TargetPositionHooks


class SpecPrefillRuntimePreparationError(RuntimeError):
    """The eager scorer/target runtime could not be prepared safely."""


@dataclass(frozen=True)
class LoadedSpecPrefillScorer:
    """One eagerly loaded scorer artifact and its release callback."""

    model: Any
    cleanup: Callable[[], None]

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError("loaded scorer model must not be None")
        if not callable(self.cleanup):
            raise TypeError("loaded scorer cleanup must be callable")


ScorerLoader = Callable[[str, str], LoadedSpecPrefillScorer]
TargetCacheFactory = Callable[[Any], Sequence[Any]]


@dataclass(frozen=True)
class TargetProcessorAttestation:
    """Trusted artifact identities resolved from the exact loaded objects."""

    target_model: Any
    processor: Any
    target_artifact_hash: str
    tokenizer_artifact_hash: str

    def __post_init__(self) -> None:
        for name in ("target_artifact_hash", "tokenizer_artifact_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from exc


TargetIdentityAttestor = Callable[[Any, Any], TargetProcessorAttestation]


@dataclass(frozen=True)
class _QwenCacheTopology:
    entries: tuple[tuple[type[Any], int], ...]


@dataclass(frozen=True)
class _GemmaCacheTopology:
    entries: tuple[tuple[type[Any], int | None, int | None], ...]


def build_qwen_cb_specprefill_prepare(
    *,
    scorer_artifact_path: str,
    scorer_artifact_hash: str,
    target_artifact_hash: str,
    tokenizer_artifact_hash: str,
    profile_registry: SpecPrefillProfileRegistry,
    profile_key: SpecPrefillProfileKey,
    calibrated_tuning: SpecPrefillTuning,
    estimated_residency_bytes: int,
    target_identity_attestor: TargetIdentityAttestor,
    diagnostic: bool = False,
    scorer_loader: ScorerLoader | None = None,
    target_cache_factory: TargetCacheFactory | None = None,
    scorer_prefill_step_size: int = 2048,
    target_prefill_step_size: int = 2048,
) -> Callable[[Any, Any], PreparedMLLMSpecPrefillRuntime]:
    """Return a once-per-startup eager runtime preparation callback.

    Artifact loading and hook installation happen inside the returned launch
    callback, never in the request session factory.
    """
    _validate_builder_inputs(
        scorer_artifact_path=scorer_artifact_path,
        scorer_artifact_hash=scorer_artifact_hash,
        target_artifact_hash=target_artifact_hash,
        tokenizer_artifact_hash=tokenizer_artifact_hash,
        profile_registry=profile_registry,
        profile_key=profile_key,
        calibrated_tuning=calibrated_tuning,
        estimated_residency_bytes=estimated_residency_bytes,
        diagnostic=diagnostic,
        scorer_prefill_step_size=scorer_prefill_step_size,
        target_prefill_step_size=target_prefill_step_size,
    )
    loader = scorer_loader or _default_scorer_loader
    if not callable(loader):
        raise TypeError("scorer_loader must be callable")
    if not callable(target_identity_attestor):
        raise TypeError("target_identity_attestor must be callable")
    cache_factory = target_cache_factory or _default_target_cache_factory
    if not callable(cache_factory):
        raise TypeError("target_cache_factory must be callable")
    sparse_tuning = SparsePolicyTuning(
        keep_pct=calibrated_tuning.keep_pct,
        backbone_pct=calibrated_tuning.backbone_pct,
        halo_chunks=calibrated_tuning.halo_chunks,
        anchor_chunks=calibrated_tuning.anchor_chunks,
        chunk_size=calibrated_tuning.chunk_size,
    )

    def prepare(target_model: Any, processor: Any) -> PreparedMLLMSpecPrefillRuntime:
        loaded: LoadedSpecPrefillScorer | None = None
        owner: _QwenCBRuntimeOwner | None = None
        try:
            actual_scorer_hash = sha256_artifact_path(scorer_artifact_path)
            if actual_scorer_hash != scorer_artifact_hash:
                raise ValueError("scorer artifact bytes do not match expected hash")
            attestation = target_identity_attestor(target_model, processor)
            if not isinstance(attestation, TargetProcessorAttestation):
                raise TypeError(
                    "target_identity_attestor must return TargetProcessorAttestation"
                )
            if (
                attestation.target_model is not target_model
                or attestation.processor is not processor
            ):
                raise ValueError("target attestation is not bound to the loaded objects")
            if (
                attestation.target_artifact_hash != target_artifact_hash
                or attestation.tokenizer_artifact_hash != tokenizer_artifact_hash
            ):
                raise ValueError("target/tokenizer attestation does not match profile")
            loaded = loader(scorer_artifact_path, scorer_artifact_hash)
            if not isinstance(loaded, LoadedSpecPrefillScorer):
                raise TypeError("scorer_loader must return LoadedSpecPrefillScorer")
            if sha256_artifact_path(scorer_artifact_path) != actual_scorer_hash:
                raise ValueError("scorer artifact changed while it was being loaded")
            scorer = SpecPrefillScorer.for_model(loaded.model)
            target_text_model = getattr(target_model, "language_model", target_model)
            adapter = resolve_target_position_adapter(target_model)
            if adapter.family not in {
                TargetPositionFamily.QWEN_DENSE,
                TargetPositionFamily.QWEN35_TEXT_HYBRID,
            }:
                raise SpecPrefillRuntimePreparationError(
                    "CB SpecPrefill runtime admits proven Qwen text targets only"
                )
            hooks = TargetPositionHooks.for_model(target_text_model, adapter)
            probe_cache = tuple(cache_factory(target_text_model))
            topology = _qwen_cache_topology(target_text_model, probe_cache)
            del probe_cache
            TargetPositionHooks.for_model(target_text_model, adapter)
            owner = _QwenCBRuntimeOwner(
                target_model=target_text_model,
                scorer=scorer,
                loaded_scorer=loaded,
                adapter=adapter,
                hooks=hooks,
                cache_factory=cache_factory,
                cache_topology=topology,
                sparse_tuning=sparse_tuning,
                target_id=(
                    f"{profile_key.target_artifact_id}@sha256:{target_artifact_hash}"
                ),
                tokenizer_id=f"tokenizer@sha256:{tokenizer_artifact_hash}",
                scorer_id=(
                    f"{profile_key.scorer_artifact_id}@sha256:{scorer_artifact_hash}"
                ),
                scorer_prefill_step_size=scorer_prefill_step_size,
                target_prefill_step_size=target_prefill_step_size,
            )
            return PreparedMLLMSpecPrefillRuntime(
                profile_registry=profile_registry,
                profile_key=profile_key,
                estimated_residency_bytes=estimated_residency_bytes,
                session_factory=owner.session_factory,
                cache_capability=MLLMSpecPrefillCacheCapability(
                    adapter_id=adapter.adapter_id,
                    layout="qwen3_5_nonrotating_hybrid",
                ),
                target_forward_context=owner.target_forward_context,
                target_model=target_model,
                processor=processor,
                target_artifact_hash=target_artifact_hash,
                tokenizer_artifact_hash=tokenizer_artifact_hash,
                scorer_artifact_hash=scorer_artifact_hash,
                cleanup=owner.close,
                diagnostic=diagnostic,
                advertisable=not diagnostic,
            )
        except BaseException as failure:
            try:
                if owner is not None:
                    owner.close()
                elif loaded is not None:
                    loaded.cleanup()
            except BaseException as cleanup_failure:
                failure.add_note(
                    "SpecPrefill preparation cleanup failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
                retained = owner.close if owner is not None else loaded.cleanup
                setattr(failure, "specprefill_retained_cleanup", retained)
            raise

    return prepare


def build_gemma_cb_specprefill_prepare(
    *,
    scorer_artifact_path: str,
    scorer_artifact_hash: str,
    target_artifact_path: str,
    target_artifact_hash: str,
    tokenizer_artifact_hash: str,
    gemma_artifact: GemmaArtifactSpec,
    profile_registry: SpecPrefillProfileRegistry,
    profile_key: SpecPrefillProfileKey,
    calibrated_tuning: SpecPrefillTuning,
    estimated_residency_bytes: int,
    target_identity_attestor: TargetIdentityAttestor,
    scorer_loader: ScorerLoader | None = None,
    target_cache_factory: TargetCacheFactory | None = None,
    scorer_prefill_step_size: int = 2048,
    target_prefill_step_size: int = 2048,
) -> Callable[[Any, Any], PreparedMLLMSpecPrefillRuntime]:
    """Prepare diagnostic-only Gemma CB SpecPrefill ownership eagerly.

    This is deliberately not a production or discovery capability. The exact
    target bytes, live object/topology, mlx-vlm scalar cache backend, scorer,
    hooks, and request cache factory are all proven before startup publishes a
    scheduler.
    """
    _validate_builder_inputs(
        scorer_artifact_path=scorer_artifact_path,
        scorer_artifact_hash=scorer_artifact_hash,
        target_artifact_hash=target_artifact_hash,
        tokenizer_artifact_hash=tokenizer_artifact_hash,
        profile_registry=profile_registry,
        profile_key=profile_key,
        calibrated_tuning=calibrated_tuning,
        estimated_residency_bytes=estimated_residency_bytes,
        diagnostic=True,
        scorer_prefill_step_size=scorer_prefill_step_size,
        target_prefill_step_size=target_prefill_step_size,
    )
    if GEMMA4_ARTIFACTS.get(gemma_artifact.artifact_id) != gemma_artifact:
        raise ValueError("gemma_artifact must be a certified Gemma artifact")
    if profile_key.target_artifact_id != gemma_artifact.artifact_id:
        raise ValueError("Gemma artifact must match the diagnostic profile key")
    loader = scorer_loader or _default_scorer_loader
    cache_factory = target_cache_factory or _default_target_cache_factory
    if not callable(loader):
        raise TypeError("scorer_loader must be callable")
    if not callable(target_identity_attestor):
        raise TypeError("target_identity_attestor must be callable")
    if not callable(cache_factory):
        raise TypeError("target_cache_factory must be callable")
    sparse_tuning = SparsePolicyTuning(
        keep_pct=calibrated_tuning.keep_pct,
        backbone_pct=calibrated_tuning.backbone_pct,
        halo_chunks=calibrated_tuning.halo_chunks,
        anchor_chunks=calibrated_tuning.anchor_chunks,
        chunk_size=calibrated_tuning.chunk_size,
    )

    def prepare(target_model: Any, processor: Any) -> PreparedMLLMSpecPrefillRuntime:
        loaded: LoadedSpecPrefillScorer | None = None
        owner: _GemmaCBRuntimeOwner | None = None
        try:
            actual_scorer_hash = sha256_artifact_path(scorer_artifact_path)
            if actual_scorer_hash != scorer_artifact_hash:
                raise ValueError("scorer artifact bytes do not match expected hash")
            identity = target_identity_attestor(target_model, processor)
            prepared_target = prepare_gemma_sparse_target(
                target_model=target_model,
                processor=processor,
                target_artifact_path=target_artifact_path,
                artifact=gemma_artifact,
                target_identity_attestation=identity,
            )
            if identity.tokenizer_artifact_hash != tokenizer_artifact_hash:
                raise ValueError("tokenizer attestation does not match profile")
            if prepared_target.target_artifact_hash != target_artifact_hash:
                raise ValueError("target attestation does not match profile")
            loaded = loader(scorer_artifact_path, scorer_artifact_hash)
            if not isinstance(loaded, LoadedSpecPrefillScorer):
                raise TypeError("scorer_loader must return LoadedSpecPrefillScorer")
            if sha256_artifact_path(scorer_artifact_path) != actual_scorer_hash:
                raise ValueError("scorer artifact changed while it was being loaded")
            scorer = SpecPrefillScorer.for_model(loaded.model)
            adapter = resolve_target_position_adapter(target_model)
            expected_family = (
                TargetPositionFamily.GEMMA4_A4B
                if gemma_artifact.artifact_id == "gemma4-26b-a4b"
                else TargetPositionFamily.GEMMA4_DENSE
            )
            if adapter.family is not expected_family:
                raise SpecPrefillRuntimePreparationError(
                    "Gemma adapter does not match the certified artifact"
                )
            text_model = prepared_target.text_model
            hooks = TargetPositionHooks.for_model(text_model, adapter)
            probe_cache = tuple(cache_factory(text_model))
            prepared_target.validate_cache(probe_cache)
            validate_aligned_scalar_cache(probe_cache, logical_position=0)
            topology = _gemma_cache_topology(probe_cache)
            del probe_cache
            owner = _GemmaCBRuntimeOwner(
                prepared_target=prepared_target,
                scorer=scorer,
                loaded_scorer=loaded,
                adapter=adapter,
                hooks=hooks,
                cache_factory=cache_factory,
                cache_topology=topology,
                sparse_tuning=sparse_tuning,
                target_id=prepared_target.canonical_target_id,
                tokenizer_id=f"tokenizer@sha256:{tokenizer_artifact_hash}",
                scorer_id=(
                    f"{profile_key.scorer_artifact_id}@sha256:{scorer_artifact_hash}"
                ),
                scorer_prefill_step_size=scorer_prefill_step_size,
                target_prefill_step_size=target_prefill_step_size,
            )
            execution_config = SparseCacheExecutionConfig(
                target_id=owner.target_id,
                tokenizer_id=owner.tokenizer_id,
                scorer_id=owner.scorer_id,
                selector_version=SPECPREFILL_SELECTOR_VERSION,
                tuning=sparse_tuning,
            )
            gemma_batch_config = GemmaSparseBatchConfig(
                prepared_target,
                execution_config,
            )
            return PreparedMLLMSpecPrefillRuntime(
                profile_registry=profile_registry,
                profile_key=profile_key,
                estimated_residency_bytes=estimated_residency_bytes,
                session_factory=owner.session_factory,
                cache_capability=MLLMSpecPrefillCacheCapability(
                    adapter_id=adapter.adapter_id,
                    layout=gemma_artifact.artifact_id,
                    backend="mlx_vlm",
                    rotating=True,
                    homogeneous_rows_only=True,
                ),
                target_forward_context=owner.target_forward_context,
                target_model=target_model,
                processor=processor,
                target_artifact_hash=target_artifact_hash,
                tokenizer_artifact_hash=tokenizer_artifact_hash,
                scorer_artifact_hash=scorer_artifact_hash,
                cleanup=owner.close,
                gemma_batch_config=gemma_batch_config,
                diagnostic=True,
                advertisable=False,
            )
        except BaseException as failure:
            try:
                if owner is not None:
                    owner.close()
                elif loaded is not None:
                    loaded.cleanup()
            except BaseException as cleanup_failure:
                failure.add_note(
                    "SpecPrefill preparation cleanup failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
                retained = owner.close if owner is not None else loaded.cleanup
                setattr(failure, "specprefill_retained_cleanup", retained)
            raise

    return prepare


class _QwenCBRuntimeOwner:
    """Mutable lifetime owner captured by immutable prepared-runtime callables."""

    def __init__(
        self,
        *,
        target_model: Any,
        scorer: SpecPrefillScorer,
        loaded_scorer: LoadedSpecPrefillScorer,
        adapter: Any,
        hooks: TargetPositionHooks,
        cache_factory: TargetCacheFactory,
        cache_topology: _QwenCacheTopology,
        sparse_tuning: SparsePolicyTuning,
        target_id: str,
        tokenizer_id: str,
        scorer_id: str,
        scorer_prefill_step_size: int,
        target_prefill_step_size: int,
    ) -> None:
        self.target_model = target_model
        self.scorer = scorer
        self.loaded_scorer = loaded_scorer
        self.adapter = adapter
        self.hooks = hooks
        self.cache_factory = cache_factory
        self.cache_topology = cache_topology
        self.sparse_tuning = sparse_tuning
        self.target_id = target_id
        self.tokenizer_id = tokenizer_id
        self.scorer_id = scorer_id
        self.scorer_prefill_step_size = scorer_prefill_step_size
        self.target_prefill_step_size = target_prefill_step_size
        self.closed = False

    def session_factory(
        self,
        request: Any,
        tokens: tuple[int, ...],
        config: CooperativeSpecPrefillConfig,
    ) -> MLLMSpecPrefillAdmission:
        self._require_open()
        if (
            config.tuning != self.sparse_tuning
            or config.target_id != self.target_id
            or config.tokenizer_id != self.tokenizer_id
            or config.scorer_id != self.scorer_id
        ):
            raise SpecPrefillRuntimePreparationError(
                "request artifacts/tuning do not match the prepared profile"
            )
        scorer_session = SpecPrefillScorerSession(
            self.scorer,
            tokens,
            prefill_step_size=self.scorer_prefill_step_size,
        )

        def target_factory(selected_tokens, sparse_state):
            self._require_open()
            cache = self._fresh_target_cache()
            return SparseTargetPrefillSession(
                self.target_model,
                selected_tokens,
                cache,
                sparse_state,
                self.adapter,
                step_size=self.target_prefill_step_size,
            )

        session = CooperativeSpecPrefillSession(
            request.request_id,
            tokens,
            scorer_session,
            target_factory,
            config,
        )
        return MLLMSpecPrefillAdmission(session)

    def target_forward_context(self, forward: Any):
        self._require_open()
        if forward.phase is not MLLMTargetForwardPhase.DECODE:
            raise SpecPrefillRuntimePreparationError(
                "prepared target context supports decode only"
            )
        rows = tuple(forward.sparse_row_states)
        if not rows or all(row is None for row in rows):
            return nullcontext()
        if any(row is None for row in rows):
            raise SpecPrefillRuntimePreparationError(
                "sparse and dense rows cannot share target forward context"
            )
        state = SparseCacheState(tuple(row.clone() for row in rows if row is not None))
        return self.hooks.session_for_plan(decode_plan(self.adapter, state))

    def _fresh_target_cache(self) -> tuple[Any, ...]:
        cache = tuple(self.cache_factory(self.target_model))
        topology = _qwen_cache_topology(self.target_model, cache)
        if topology != self.cache_topology:
            raise SpecPrefillRuntimePreparationError(
                "request cache topology differs from the prepared probe"
            )
        return cache

    def close(self) -> None:
        if self.closed:
            return
        self.loaded_scorer.cleanup()
        self.closed = True
        self.scorer = None
        self.loaded_scorer = None
        self.hooks = None
        self.target_model = None

    def _require_open(self) -> None:
        if self.closed:
            raise SpecPrefillRuntimePreparationError("prepared runtime is closed")


class _GemmaCBRuntimeOwner:
    """Lifetime owner for one exact diagnostic Gemma CB target."""

    def __init__(
        self,
        *,
        prepared_target: Any,
        scorer: SpecPrefillScorer,
        loaded_scorer: LoadedSpecPrefillScorer,
        adapter: Any,
        hooks: TargetPositionHooks,
        cache_factory: TargetCacheFactory,
        cache_topology: _GemmaCacheTopology,
        sparse_tuning: SparsePolicyTuning,
        target_id: str,
        tokenizer_id: str,
        scorer_id: str,
        scorer_prefill_step_size: int,
        target_prefill_step_size: int,
    ) -> None:
        self.prepared_target = prepared_target
        self.target_model = prepared_target.text_model
        self.scorer = scorer
        self.loaded_scorer = loaded_scorer
        self.adapter = adapter
        self.hooks = hooks
        self.cache_factory = cache_factory
        self.cache_topology = cache_topology
        self.sparse_tuning = sparse_tuning
        self.target_id = target_id
        self.tokenizer_id = tokenizer_id
        self.scorer_id = scorer_id
        self.scorer_prefill_step_size = scorer_prefill_step_size
        self.target_prefill_step_size = target_prefill_step_size
        self.closed = False

    def session_factory(
        self,
        request: Any,
        tokens: tuple[int, ...],
        config: CooperativeSpecPrefillConfig,
    ) -> MLLMSpecPrefillAdmission:
        self._require_open()
        expected_tail = self.prepared_target.artifact.sliding_window
        actual_tail = (
            None
            if config.rotating_tail_requirement is None
            else config.rotating_tail_requirement.window_tokens
        )
        if (
            config.tuning != self.sparse_tuning
            or config.target_id != self.target_id
            or config.tokenizer_id != self.tokenizer_id
            or config.scorer_id != self.scorer_id
            or actual_tail != expected_tail
        ):
            raise SpecPrefillRuntimePreparationError(
                "request artifacts/tuning/tail do not match prepared Gemma profile"
            )
        scorer_session = SpecPrefillScorerSession(
            self.scorer,
            tokens,
            prefill_step_size=self.scorer_prefill_step_size,
        )

        def target_factory(selected_tokens, sparse_state):
            self._require_open()
            return SparseTargetPrefillSession(
                self.target_model,
                selected_tokens,
                self._fresh_target_cache(),
                sparse_state,
                self.adapter,
                step_size=self.target_prefill_step_size,
            )

        return MLLMSpecPrefillAdmission(
            CooperativeSpecPrefillSession(
                request.request_id,
                tokens,
                scorer_session,
                target_factory,
                config,
            )
        )

    def target_forward_context(self, forward: Any):
        self._require_open()
        if forward.phase is not MLLMTargetForwardPhase.DECODE:
            raise SpecPrefillRuntimePreparationError(
                "prepared target context supports decode only"
            )
        rows = tuple(forward.sparse_row_states)
        if not rows or all(row is None for row in rows):
            return nullcontext()
        if any(row is None for row in rows):
            raise SpecPrefillRuntimePreparationError(
                "sparse and dense rows cannot share target forward context"
            )
        logical_positions = {
            row.next_logical_position for row in rows if row is not None
        }
        physical_lengths = {
            row.physical_valid_length for row in rows if row is not None
        }
        if len(logical_positions) != 1 or len(physical_lengths) != 1:
            raise SpecPrefillRuntimePreparationError(
                "Gemma target context requires homogeneous sparse rows"
            )
        state = SparseCacheState(tuple(row.clone() for row in rows if row is not None))
        return self.hooks.session_for_plan(decode_plan(self.adapter, state))

    def _fresh_target_cache(self) -> Sequence[Any]:
        cache = self.cache_factory(self.target_model)
        if not isinstance(cache, Sequence):
            raise SpecPrefillRuntimePreparationError(
                "Gemma cache factory must return an ordered cache sequence"
            )
        self.prepared_target.validate_cache(cache)
        validate_aligned_scalar_cache(cache, logical_position=0)
        if _gemma_cache_topology(cache) != self.cache_topology:
            raise SpecPrefillRuntimePreparationError(
                "request Gemma cache topology differs from prepared probe"
            )
        return cache

    def close(self) -> None:
        if self.closed:
            return
        self.loaded_scorer.cleanup()
        self.closed = True
        self.scorer = None
        self.loaded_scorer = None
        self.hooks = None
        self.target_model = None
        self.prepared_target = None

    def _require_open(self) -> None:
        if self.closed:
            raise SpecPrefillRuntimePreparationError("prepared runtime is closed")


def _validate_builder_inputs(**values: Any) -> None:
    path = values["scorer_artifact_path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("scorer_artifact_path must be non-empty")
    registry = values["profile_registry"]
    key = values["profile_key"]
    tuning = values["calibrated_tuning"]
    if not isinstance(registry, SpecPrefillProfileRegistry):
        raise TypeError("profile_registry must be SpecPrefillProfileRegistry")
    if not isinstance(key, SpecPrefillProfileKey):
        raise TypeError("profile_key must be SpecPrefillProfileKey")
    if not isinstance(tuning, SpecPrefillTuning):
        raise TypeError("calibrated_tuning must be SpecPrefillTuning")
    for input_name, key_value in (
        ("target_artifact_hash", key.target_artifact_hash),
        ("tokenizer_artifact_hash", key.tokenizer_artifact_hash),
        ("scorer_artifact_hash", key.scorer_artifact_hash),
    ):
        if values[input_name] != key_value:
            raise ValueError(f"{input_name} must match profile key")
    diagnostic = values["diagnostic"]
    tier = (
        SpecPrefillProfileTier.DIAGNOSTIC
        if diagnostic
        else SpecPrefillProfileTier.PRODUCTION
    )
    profiles = tuple(
        profile
        for profile in registry.profiles
        if profile.key == key and profile.tier is tier
    )
    if not profiles:
        raise ValueError("requested SpecPrefill profile is not registered")
    profile = profiles[0]
    if not diagnostic and not profile.production_certified:
        raise ValueError("production SpecPrefill requires certified evidence")
    if profile.calibration.tuning != tuning:
        raise ValueError("calibrated_tuning must match the registered profile")
    residency = values["estimated_residency_bytes"]
    if isinstance(residency, bool) or not isinstance(residency, int) or residency < 0:
        raise ValueError("estimated_residency_bytes must be non-negative")
    for name in ("scorer_prefill_step_size", "target_prefill_step_size"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be positive")


def _default_scorer_loader(path: str, _expected_hash: str) -> LoadedSpecPrefillScorer:
    from mlx_lm import load

    loaded = load(Path(path))
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    return LoadedSpecPrefillScorer(
        model=model,
        cleanup=lambda: None,
    )


def _default_target_cache_factory(target_model: Any) -> list[Any]:
    return list(make_prompt_cache(target_model))


def sha256_artifact_path(path: str | Path) -> str:
    """Digest actual artifact bytes using one canonical file/directory manifest."""
    artifact = Path(path)
    _reject_symlink_components(artifact)
    if artifact.is_file():
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not artifact.is_dir() or artifact.is_symlink():
        raise SpecPrefillRuntimePreparationError(
            "scorer artifact must be a regular file or directory"
        )
    descendants = tuple(sorted(artifact.rglob("*")))
    if any(candidate.is_symlink() for candidate in descendants):
        raise SpecPrefillRuntimePreparationError(
            "scorer artifact manifest cannot contain symlinks"
        )
    files = tuple(candidate for candidate in descendants if candidate.is_file())
    if not files:
        raise SpecPrefillRuntimePreparationError(
            "scorer artifact directory must contain only regular files"
        )
    digest = hashlib.sha256(b"vllm-mlx-specprefill-artifact-v1\0")
    for candidate in files:
        relative = candidate.relative_to(artifact).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "big"))
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise SpecPrefillRuntimePreparationError(
                "scorer artifact path cannot contain symlinked components"
            )


def _qwen_cache_topology(
    target_model: Any, cache: Sequence[Any]
) -> _QwenCacheTopology:
    layers = tuple(getattr(target_model, "layers", ()))
    if not layers or len(cache) != len(layers):
        raise SpecPrefillRuntimePreparationError(
            "target cache must have exactly one entry per Qwen text layer"
        )
    entries: list[tuple[type[Any], int]] = []
    physical_entries = 0
    for layer, entry in zip(layers, cache, strict=True):
        linear = bool(getattr(layer, "is_linear", False))
        if type(entry) is ArraysCache:
            if not linear or any(value is not None for value in entry.cache):
                raise SpecPrefillRuntimePreparationError(
                    "Qwen linear layers require fresh ArraysCache entries"
                )
            if entry.left_padding is not None or entry.lengths is not None:
                raise SpecPrefillRuntimePreparationError(
                    "Qwen probe ArraysCache must have no batch metadata"
                )
            entries.append((ArraysCache, len(entry.cache)))
        elif type(entry) is KVCache:
            if linear or entry.offset != 0 or entry.keys is not None or entry.values is not None:
                raise SpecPrefillRuntimePreparationError(
                    "Qwen attention layers require fresh nonrotating KVCache entries"
                )
            physical_entries += 1
            entries.append((KVCache, 0))
        else:
            raise SpecPrefillRuntimePreparationError(
                "Qwen CB cache contains an unsupported or rotating entry"
            )
    if physical_entries == 0:
        raise SpecPrefillRuntimePreparationError(
            "Qwen CB cache has no physical attention cache owners"
        )
    return _QwenCacheTopology(tuple(entries))


def _gemma_cache_topology(cache: Sequence[Any]) -> _GemmaCacheTopology:
    if not cache or len({id(entry) for entry in cache}) != len(cache):
        raise SpecPrefillRuntimePreparationError(
            "Gemma probe cache owners must be non-empty and unaliased"
        )
    return _GemmaCacheTopology(
        tuple(
            (
                type(entry),
                getattr(entry, "max_size", None),
                getattr(entry, "keep", None),
            )
            for entry in cache
        )
    )
