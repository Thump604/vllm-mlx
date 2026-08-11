# SPDX-License-Identifier: Apache-2.0
"""
Simple engine for maximum single-user throughput.

This engine wraps mlx-lm directly with zero overhead for optimal
performance when serving a single user at a time.
"""

import asyncio
import copy
import contextvars
import hashlib
import inspect
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

# Re-entrancy guard for SimpleEngine._track_request_stream so that
# internal fallback paths inside _stream_chat_impl (which call back into
# self.stream_generate) don't double-count a single external request.
# contextvars propagates per-asyncio-task, so concurrent requests still
# each get their own outermost tracking pass.
_in_tracker: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_simple_engine_in_tracker", default=False
)

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, has_media_content, is_mllm_model
from ..specprefill import (
    SPECPREFILL_SELECTOR_VERSION,
    SpecPrefillDecision,
    SpecPrefillPolicy,
    build_selection_plan,
    resolve_specprefill_decision,
)
from ..specprefill_cache import (
    SparseCacheIdentity,
    SparseCacheState,
    SparsePolicyTuning,
)
from ..specprefill_generation_context import SparseGenerationForwardContext
from ..specprefill_positions import (
    TargetPositionAdapter,
    TargetPositionError,
    TargetPositionFamily,
    resolve_target_position_adapter,
)
from ..specprefill_selection import RotatingTailRequirement
from ..specprefill_target_executor import (
    SparseTargetPrefillAuthorityError,
    execute_sparse_target_prefill,
)
from ..specprefill_target_hooks import TargetPositionHooks
from ..specprefill_profiles import (
    EMPTY_SPECPREFILL_PROFILE_REGISTRY,
    SpecPrefillCell,
    SpecPrefillEngine,
    SpecPrefillProfileKey,
    SpecPrefillProfileRegistry,
    SpecPrefillTuning,
)
from ..native_mtp_request import (
    NativeMTPRequestConfig,
    NativeMTPServerState,
    resolve_native_mtp_consumer,
)
from .base import (
    BaseEngine,
    EngineBusy,
    GenerationOutput,
    cleanup_startup_cancellation,
    run_blocking_startup_work,
)
from .chat_template_safety import normalize_messages_for_chat_template
from ..mlx_streams import bind_generation_streams

logger = logging.getLogger(__name__)


def _bind_worker_generation_streams() -> None:
    """Rebind mlx generation streams inside the current worker thread."""
    bind_generation_streams()


def _seed_logits_processors(
    seed_tokens: mx.array | None,
    processors: list[Any] | None,
) -> list[Any] | None:
    """Wrap logits processors so continuation decode sees the full prompt."""
    if not processors:
        return None
    if seed_tokens is None or seed_tokens.size == 0:
        return list(processors)

    def _wrap(processor):
        def _seeded(tokens, logits):
            merged = seed_tokens
            if tokens is not None:
                if not isinstance(tokens, mx.array):
                    tokens_arr = mx.array(tokens, dtype=mx.uint32)
                else:
                    tokens_arr = tokens
                if tokens_arr.size > 0:
                    merged = mx.concatenate([seed_tokens, tokens_arr])
            return processor(merged, logits)

        if getattr(processor, "native_mtp_replay_safe", False):
            _seeded.native_mtp_replay_safe = True
        return _seeded

    return [_wrap(processor) for processor in processors]


def _sample_with_processors(
    tokens: mx.array | None,
    logits: mx.array,
    sampler: Any,
    logits_processors: list[Any] | None,
) -> tuple[mx.array, mx.array]:
    """Sample a token while honoring any active logits processors."""
    if logits_processors:
        is_1d = logits.ndim == 1
        if is_1d:
            logits = logits[None]
        for processor in logits_processors:
            logits = processor(tokens, logits)
        if is_1d:
            logits = logits.squeeze(0)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok = sampler(logprobs)
    return tok, logprobs


def _new_sparse_detokenizer(tokenizer: Any) -> Any:
    """Create a request-local mlx-lm-compatible incremental detokenizer."""
    detokenizer = getattr(tokenizer, "detokenizer", None)
    for name in ("reset", "add_token", "finalize"):
        if not callable(getattr(detokenizer, name, None)):
            raise TargetPositionError(
                "tokenizer lacks the incremental detokenizer required by sparse decode"
            )
    detokenizer.reset()
    # Access once during admission so a malformed implementation fails before
    # scorer/RNG state advances.
    if not isinstance(getattr(detokenizer, "last_segment", None), str):
        raise TargetPositionError("incremental detokenizer last_segment must be text")
    return detokenizer


def _build_text_tokenizer(processor: Any) -> Any:
    """Build the mlx-lm tokenizer contract from an mlx-vlm processor.

    mlx-vlm installs the artifact-aware streaming detokenizer on the processor,
    while ``processor.tokenizer`` is the raw Hugging Face tokenizer.  Dense and
    sparse text-model generation must receive the same mlx-lm ``TokenizerWrapper``
    so each request gets an isolated detokenizer with identical token semantics.
    """
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    raw_tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(processor, "detokenizer", None)
    if raw_tokenizer is None or template is None:
        raise TargetPositionError(
            "MLLM text routing requires processor.tokenizer and processor.detokenizer"
        )
    for name in ("reset", "add_token", "finalize"):
        if not callable(getattr(template, name, None)):
            raise TargetPositionError(
                "MLLM processor detokenizer does not implement streaming semantics"
            )

    def _detokenizer_factory(_tokenizer):
        detokenizer = copy.copy(template)
        detokenizer.reset()
        return detokenizer

    eos_ids = getattr(processor, "eos_token_ids", None)
    if eos_ids is None:
        eos_ids = getattr(raw_tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        eos_ids = getattr(raw_tokenizer, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = (eos_ids,)
    return TokenizerWrapper(
        raw_tokenizer,
        detokenizer_class=_detokenizer_factory,
        eos_token_ids=eos_ids,
    )


def _detokenize_sparse_token(
    detokenizer: Any,
    token: int,
    eos_ids: frozenset[int],
    *,
    terminal: bool,
) -> tuple[str, bool]:
    """Detokenize one sampled token while suppressing EOS token text."""
    is_eos = token in eos_ids
    if not is_eos:
        detokenizer.add_token(token)
    if terminal or is_eos:
        detokenizer.finalize()
    segment = detokenizer.last_segment
    if not isinstance(segment, str):
        raise TargetPositionError("incremental detokenizer emitted non-text output")
    return segment, is_eos


def _processors_can_retire(processors: list[Any] | None) -> bool:
    """True when any processor advertises a retire-to-content transition."""
    if os.getenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME") != "1":
        return False
    return bool(processors) and any(
        isinstance(getattr(p, "is_retired", None), bool) for p in processors
    )


def _processors_retired(processors: list[Any] | None) -> bool:
    """True when any retire-capable processor has entered its retired state."""
    if os.getenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME") != "1":
        return False
    return bool(processors) and any(
        getattr(p, "is_retired", False) is True for p in processors
    )


def _request_can_compose_mtp(route_mtp: bool, processors: list[Any] | None) -> bool:
    """Whether this request reaches an MTP decode phase on the text route.

    A permanently active request-local processor disables MTP for the whole
    request. A retire-capable processor constrains only the thinking phase;
    its content continuation deliberately resumes native MTP and therefore
    requires the independently qualified combined profile cell.
    """
    return route_mtp and (not processors or _processors_can_retire(processors))


def _consume_native_mtp_request(
    kwargs: dict[str, Any], *, server_default: bool
) -> tuple[NativeMTPRequestConfig | None, bool, bool, str | None]:
    """Consume private server controls before backend kwargs are constructed."""

    config = kwargs.pop("_native_mtp_request_config", None)
    disabled = kwargs.pop("_native_mtp_disabled", False)
    bypass_reason = kwargs.pop("_native_mtp_bypass_reason", None)
    if config is not None and not isinstance(config, NativeMTPRequestConfig):
        raise ValueError("invalid native MTP request config")
    if not isinstance(disabled, bool):
        raise ValueError("invalid native MTP disable control")
    if config is not None and disabled:
        raise ValueError("native MTP request cannot be selected and disabled")
    if bypass_reason is not None and not isinstance(bypass_reason, str):
        raise ValueError("invalid native MTP bypass reason")
    if config is not None and bypass_reason is not None:
        raise ValueError("native MTP bypass cannot be selected")
    selected = config is not None or (server_default and not disabled)
    if bypass_reason is not None:
        selected = False
    return config, disabled, selected, bypass_reason


class _SpecPrefillCancelled(Exception):
    """Cooperative cancellation sentinel for blocking SpecPrefill workers."""


class _SpecPrefillAuthorityError(RuntimeError):
    """Receipt/cache authority could not prove that dense replay is safe."""


def _try_abandon_sparse_bootstrap(bootstrap: Any) -> bool:
    """Atomically abandon only authority that has not begun a claim."""
    abandon = getattr(bootstrap, "try_abandon_unclaimed", None)
    if not callable(abandon):
        raise _SpecPrefillAuthorityError(
            "sparse native MTP bootstrap lacks atomic abandonment"
        )
    try:
        abandoned = abandon()
    except BaseException as exc:
        raise _SpecPrefillAuthorityError(
            "sparse native MTP authority cleanup failed"
        ) from exc
    if type(abandoned) is not bool:
        raise _SpecPrefillAuthorityError(
            "sparse native MTP abandonment returned a non-boolean result"
        )
    return abandoned


async def _aclose_async_iterator(
    iterator: AsyncIterator[Any], primary_error: BaseException | None
) -> None:
    """Close a nested async iterator without masking its primary failure."""

    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except BaseException:
        if primary_error is None:
            raise
        logger.warning(
            "Failed to close nested generation stream after primary error",
            exc_info=True,
        )


@dataclass
class _SpecPrefillTelemetry:
    """Mutable, request-local SpecPrefill evidence shared by one stream.

    Selection and execution run in a blocking worker while responses are yielded
    on the event loop.  Keeping this state request-local means a sparse-prefill
    failure can atomically turn subsequent output into an explicit dense
    fallback without changing model-global state or MTP accounting.
    """

    decision: SpecPrefillDecision
    total_tokens: int | None
    selected_tokens: int | None
    scorer_ms: float | None = None
    target_prefill_ms: float | None = None
    profile_tuning: SpecPrefillTuning | None = None
    profile_selector_version: str | None = None

    def fallback(self, reason: str) -> None:
        self.decision = SpecPrefillDecision(
            requested_policy=self.decision.requested_policy,
            effective_policy=SpecPrefillPolicy.DENSE,
            coverage=self.decision.coverage,
            fallback_reason=reason,
        )
        # Dense fallback retains the complete prompt.  This is also the only
        # cache state that may be used after a sparse execution error.
        self.selected_tokens = self.total_tokens
        # Retain completed phase timings: they are evidence of the failed
        # sparse attempt and must not be mistaken for a zero-cost dense path.

    def as_output_kwargs(self) -> dict[str, Any]:
        decision = self.decision
        return {
            "specprefill_requested_policy": decision.requested_policy.value,
            "specprefill_effective_policy": decision.effective_policy.value,
            "specprefill_coverage": decision.coverage.value,
            "specprefill_engaged": decision.effective_policy
            is SpecPrefillPolicy.SPARSE,
            "specprefill_selector_version": (
                self.profile_selector_version or SPECPREFILL_SELECTOR_VERSION
            ),
            "specprefill_fallback_reason": decision.fallback_reason,
            "specprefill_total_tokens": self.total_tokens,
            "specprefill_selected_tokens": self.selected_tokens,
            "specprefill_scorer_ms": self.scorer_ms,
            "specprefill_target_prefill_ms": self.target_prefill_ms,
        }


class SimpleEngine(BaseEngine):
    """
    Simple engine for direct model calls.

    This engine provides maximum throughput for single-user scenarios
    by calling mlx-lm/mlx-vlm directly without batching overhead.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        enable_cache: bool = True,
        force_mllm: bool = False,
        mtp: bool = False,
        mtp_num_draft_tokens: int = 1,
        prefill_step_size: int = 2048,
        specprefill_enabled: bool = False,
        specprefill_threshold: int = 8192,
        specprefill_keep_pct: float = 0.3,
        specprefill_backbone_pct: float = 0.0,
        specprefill_draft_model: str | None = None,
        specprefill_diagnostic_mode: bool = False,
        specprefill_max_tokens: int | None = None,
        specprefill_profile_registry: SpecPrefillProfileRegistry | None = None,
        specprefill_sparse_profile_key: SpecPrefillProfileKey | None = None,
        specprefill_combined_profile_key: SpecPrefillProfileKey | None = None,
        specprefill_estimated_residency_bytes: int | None = None,
        max_kv_size: int = 0,
        mllm_draft_model: str | None = None,
        mllm_draft_kind: str | None = None,
        mllm_draft_block_size: int | None = None,
    ):
        """
        Initialize the simple engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable VLM cache for multimodal models
            force_mllm: Force loading as MLLM even if not auto-detected
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            mtp_num_draft_tokens: Draft tokens per speculative MTP step
            prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
            specprefill_enabled: Enable SpecPrefill (attention-based sparse prefill)
            specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill
            specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
            specprefill_backbone_pct: Fraction of chunks to reserve for evenly
                spaced coverage (default: 0.0)
            specprefill_draft_model: Path to small draft model for importance scoring
            specprefill_diagnostic_mode: Allow forced sparse requests and
                per-request selector overrides. Production routes keep this off.
            specprefill_max_tokens: Optional explicit scorer-admission cap.
                ``None`` leaves long-context admission to the profile/memory
                controller rather than imposing a hidden engine ceiling.
            specprefill_profile_registry: Exact calibrated profiles. ``None``
                uses an empty fail-closed registry.
            specprefill_sparse_profile_key: Exact SimpleEngine sparse-only
                artifact/adapter profile identity.
            specprefill_combined_profile_key: Exact SimpleEngine
                SpecPrefill+MTP profile identity.
            specprefill_estimated_residency_bytes: Estimated resident bytes for
                this exact request composition. This engine does not estimate
                or load artifacts.
            max_kv_size: Maximum KV cache size per sequence (0 = unbounded)
            mllm_draft_model: Optional MLLM speculative draft/assistant model path
            mllm_draft_kind: Optional mlx-vlm draft kind, for example "mtp"
            mllm_draft_block_size: Optional speculative block size for mlx-vlm
        """
        self._model_name = model_name
        self._created_at = time.time()
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._mtp = mtp
        self._mtp_num_draft_tokens = mtp_num_draft_tokens
        self._prefill_step_size = prefill_step_size

        # Request stats (parity with BatchedEngine for /v1/status monitoring).
        # Without these, monitoring sees zero traffic for SimpleEngine-backed
        # servers (e.g. Gemma 4 31B with --mllm-draft-model + MTP).
        self._total_requests_processed: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._num_running: int = 0
        # Rolling window of (completion_tokens, duration_s) for tps computation.
        self._recent_completions: deque = deque(maxlen=20)
        # Live per-request state, mirroring BatchedEngine's "requests" list
        # in /v1/status (request_id, phase, ttft_s, tokens_per_second, ...).
        self._active_requests: dict[str, dict[str, Any]] = {}

        # SpecPrefill config
        self._specprefill_enabled = specprefill_enabled
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct
        self._specprefill_backbone_pct = specprefill_backbone_pct
        self._specprefill_draft_model_path = specprefill_draft_model
        self._specprefill_diagnostic_mode = specprefill_diagnostic_mode
        if specprefill_max_tokens is not None and specprefill_max_tokens <= 0:
            raise ValueError("specprefill_max_tokens must be positive when set")
        self._specprefill_max_tokens = specprefill_max_tokens
        if specprefill_profile_registry is None:
            specprefill_profile_registry = EMPTY_SPECPREFILL_PROFILE_REGISTRY
        if not isinstance(specprefill_profile_registry, SpecPrefillProfileRegistry):
            raise ValueError(
                "specprefill_profile_registry must be SpecPrefillProfileRegistry"
            )
        for name, key, expected_cell in (
            (
                "specprefill_sparse_profile_key",
                specprefill_sparse_profile_key,
                SpecPrefillCell.SPARSE_ONLY,
            ),
            (
                "specprefill_combined_profile_key",
                specprefill_combined_profile_key,
                SpecPrefillCell.COMBINED_MTP,
            ),
        ):
            if key is not None and not isinstance(key, SpecPrefillProfileKey):
                raise ValueError(f"{name} must be SpecPrefillProfileKey")
            if key is not None and key.engine is not SpecPrefillEngine.SIMPLE:
                raise ValueError(f"{name} must target the SimpleEngine route")
            if key is not None and key.cell is not expected_cell:
                raise ValueError(f"{name} must target the {expected_cell.value} cell")
        if (
            specprefill_estimated_residency_bytes is not None
            and specprefill_estimated_residency_bytes < 0
        ):
            raise ValueError(
                "specprefill_estimated_residency_bytes must be non-negative"
            )
        self._specprefill_profile_registry = specprefill_profile_registry
        self._specprefill_sparse_profile_key = specprefill_sparse_profile_key
        self._specprefill_combined_profile_key = specprefill_combined_profile_key
        self._specprefill_estimated_residency_bytes = (
            specprefill_estimated_residency_bytes
        )
        self._mllm_draft_model_path = mllm_draft_model
        self._mllm_draft_kind = mllm_draft_kind
        self._mllm_draft_block_size = mllm_draft_block_size

        # KV cache size limit
        self._max_kv_size = max_kv_size

        self._model = None
        self._loaded = False

        # Per-request routing state (MLLM+MTP mode)
        self._text_model = None
        self._text_tokenizer = None

        # SpecPrefill draft model (loaded at start if enabled)
        self._draft_model = None

        # Lock to serialize MLX operations (prevents Metal command buffer conflicts)
        self._generation_lock = asyncio.Lock()
        self._generation_lock_admission = (
            os.environ.get("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "fail_fast")
            .strip()
            .lower()
        )
        if self._generation_lock_admission not in {"fail_fast", "wait"}:
            logger.warning(
                "Invalid VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=%r; using fail_fast",
                self._generation_lock_admission,
            )
            self._generation_lock_admission = "fail_fast"
        self._generation_waiters = 0
        self._generation_busy_rejections = 0

        # System prompt KV cache (reduces repeated prefill across requests).
        # OrderedDict acts as an LRU keyed by system-prefix hash so that the
        # main agent and any sub-agents with different toolsets can coexist
        # without thrashing a single snapshot slot.
        # Value is (snapshot_list, system_token_count).
        self._system_kv_capacity = max(
            1, int(os.environ.get("VLLM_MLX_SYSTEM_KV_SLOTS", "4"))
        )
        self._system_kv_cache: "OrderedDict[str, tuple[list, int]]" = OrderedDict()
        # Cache-effectiveness counters. Incremented only from inside the
        # serialized worker (single writer) so plain ``+=`` is safe; reads
        # from ``get_stats`` may be slightly stale, which is fine for
        # metrics.
        self._system_kv_cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }
        # True only when the model's prompt cache can be snapshotted and
        # restored for the manual system-prefix cache branch. Plain KV caches
        # and hybrid ``ArraysCache`` entries are safe when their state
        # containers are copied at snapshot/restore boundaries. Sliding-window
        # cache classes such as ``RotatingKVCache`` remain disabled because
        # their extra cursor metadata is not captured by ``.state`` alone.
        self._supports_system_kv_cache: bool = False

    def _resolve_specprefill_telemetry(
        self,
        *,
        legacy: bool | None,
        policy: str | None,
        coverage: str | None,
        has_media: bool,
        total_tokens: int | None,
        combined_mtp: bool = False,
    ) -> _SpecPrefillTelemetry:
        """Resolve request intent before execution, with a safe dense default.

        API validation rejects conflicts, but this engine may also be used
        directly.  Its default is deliberately conservative: an omitted
        coverage declaration is ``unknown`` and therefore never engages sparse
        prefill in a production engine.
        """
        explicit_policy = policy is not None
        if policy is None:
            policy = (
                "sparse" if legacy is True else "dense" if legacy is False else "auto"
            )
        if coverage is None:
            coverage = "unknown"

        requested_policy = SpecPrefillPolicy(policy)
        legacy_sparse_intent = legacy is True and not explicit_policy
        legacy_profile_managed_intent = (
            legacy_sparse_intent and not self._specprefill_diagnostic_mode
        )
        profile_managed_intent = (
            requested_policy is SpecPrefillPolicy.AUTO or legacy_profile_managed_intent
        )
        # ``sparse`` is a diagnostic forcing control. Its diagnostic-only
        # meaning includes bypassing calibrated profile lookup. Legacy boolean
        # enablement remains an intent, not a production forcing bypass.
        if (
            self._specprefill_diagnostic_mode
            and requested_policy is SpecPrefillPolicy.SPARSE
        ):
            threshold_met = True
        elif profile_managed_intent:
            # Production and diagnostic auto use the exact profile's calibrated
            # crossover. Do not impose the old engine-wide threshold here.
            threshold_met = True
        elif total_tokens is None:
            threshold_met = False
        else:
            threshold_met = total_tokens > self._specprefill_threshold
        # The engine owns no implicit maximum context. A deployment may install
        # an explicit scorer cap, while the normal long-context admission
        # decision belongs to the profile/memory controller.
        admission_allowed = (
            self._specprefill_max_tokens is None
            or total_tokens is None
            or total_tokens <= self._specprefill_max_tokens
        )
        # Legacy ``specprefill=true`` is production sparse intent, not a
        # production forcing bypass. In a diagnostic profile it retains the
        # legacy forcing behavior so existing sparse diagnostics do not become
        # unexpectedly coverage-gated by the new production policy.
        decision_policy = "auto" if legacy_profile_managed_intent else policy
        decision = resolve_specprefill_decision(
            decision_policy,
            coverage,
            production=not self._specprefill_diagnostic_mode,
            text_only=not has_media,
            threshold_met=threshold_met,
            admission_allowed=admission_allowed,
        )
        if legacy_sparse_intent:
            decision = SpecPrefillDecision(
                requested_policy=SpecPrefillPolicy.SPARSE,
                effective_policy=decision.effective_policy,
                coverage=decision.coverage,
                fallback_reason=decision.fallback_reason,
            )

        profile_tuning: SpecPrefillTuning | None = None
        profile_selector_version: str | None = None
        requires_profile = decision.effective_policy is SpecPrefillPolicy.SPARSE and (
            not self._specprefill_diagnostic_mode
            or requested_policy is SpecPrefillPolicy.AUTO
        )
        if requires_profile:
            profile_key = self._active_specprefill_profile_key(combined_mtp)
            if profile_key is None:
                decision = self._specprefill_dense_fallback(
                    decision, "profile_not_registered"
                )
            elif (
                total_tokens is None
                or self._specprefill_estimated_residency_bytes is None
            ):
                decision = self._specprefill_dense_fallback(
                    decision, "profile_residency_not_estimated"
                )
            else:
                profile_decision = self._specprefill_profile_registry.resolve(
                    profile_key,
                    prompt_tokens=total_tokens,
                    residency_bytes=self._specprefill_estimated_residency_bytes,
                    diagnostic=self._specprefill_diagnostic_mode,
                )
                if not profile_decision.eligible:
                    decision = self._specprefill_dense_fallback(
                        decision, profile_decision.fallback_reason
                    )
                else:
                    profile_tuning = profile_decision.tuning
                    profile_selector_version = profile_decision.selector_version
        if (
            decision.effective_policy is SpecPrefillPolicy.SPARSE
            and self._draft_model is None
        ):
            decision = SpecPrefillDecision(
                requested_policy=decision.requested_policy,
                effective_policy=SpecPrefillPolicy.DENSE,
                coverage=decision.coverage,
                fallback_reason="specprefill_unavailable",
            )
            profile_tuning = None
            profile_selector_version = None

        selected_tokens = (
            total_tokens
            if decision.effective_policy is SpecPrefillPolicy.DENSE
            else None
        )
        return _SpecPrefillTelemetry(
            decision,
            total_tokens,
            selected_tokens,
            profile_tuning=profile_tuning,
            profile_selector_version=profile_selector_version,
        )

    def _active_specprefill_profile_key(
        self, combined_mtp: bool
    ) -> SpecPrefillProfileKey | None:
        """Return the independently qualified sparse-only or combined cell."""
        return (
            self._specprefill_combined_profile_key
            if combined_mtp
            else self._specprefill_sparse_profile_key
        )

    @staticmethod
    def _callable_supports_forward_context(callable_obj: Any) -> bool:
        """Require an explicit mlx-lm continuation seam, never ``**kwargs``.

        Sparse target cache state has different logical and physical lengths.
        Forwarding it through an unverified compatibility ``**kwargs`` path
        risks silently dropping the request-local position context.
        """
        try:
            parameters = inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return False
        return "model_forward_context" in parameters

    def _supports_sparse_continuation(self, continuation: Any) -> bool:
        """Check every continuation boundary that must retain sparse state."""
        try:
            from mlx_lm import stream_generate as mlx_stream_generate
        except ImportError:
            return False
        return self._callable_supports_forward_context(
            continuation
        ) and self._callable_supports_forward_context(mlx_stream_generate)

    @staticmethod
    def _control_token_indices(tokenizer: Any, tokens: list[int]) -> tuple[int, ...]:
        """Return only tokenizer-declared control-token positions."""
        candidates: set[int] = set()
        special = getattr(tokenizer, "all_special_ids", ())
        if isinstance(special, (list, tuple, set)):
            candidates.update(
                value
                for value in special
                if isinstance(value, int) and not isinstance(value, bool)
            )
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                candidates.add(value)
        return tuple(index for index, token in enumerate(tokens) if token in candidates)

    @staticmethod
    def _eos_token_ids(tokenizer: Any) -> frozenset[int]:
        """Normalize single- and multi-EOS tokenizer contracts safely."""
        values = getattr(tokenizer, "eos_token_ids", None)
        if not isinstance(values, (list, tuple, set)):
            values = (getattr(tokenizer, "eos_token_id", None),)
        return frozenset(
            value
            for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        )

    @staticmethod
    def _rotating_tail_requirement(cache: list[Any]) -> RotatingTailRequirement | None:
        """Preserve the declared tail of every rotating target cache layer."""
        maxima: list[int] = []
        for entry in cache:
            if type(entry).__name__ != "RotatingKVCache":
                continue
            maximum = getattr(entry, "max_size", None)
            if (
                not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum <= 0
            ):
                raise TargetPositionError("rotating cache lacks a positive max_size")
            maxima.append(maximum)
        return RotatingTailRequirement(max(maxima)) if maxima else None

    def _admit_sparse_target(
        self, target_model: Any, *, combined_mtp: bool = False
    ) -> tuple[SpecPrefillProfileKey, TargetPositionAdapter]:
        """Resolve artifact/adapter/hook compatibility before scorer work."""
        if combined_mtp and self._max_kv_size:
            raise TargetPositionError(
                "sparse native MTP requires an exact nonrotating target cache"
            )
        profile_key = self._active_specprefill_profile_key(combined_mtp=combined_mtp)
        if profile_key is None:
            raise TargetPositionError("exact sparse cache identity is unavailable")
        adapter = resolve_target_position_adapter(target_model)
        if profile_key.adapter_id != adapter.adapter_id:
            raise TargetPositionError(
                "qualified profile adapter does not match the target position adapter"
            )
        if combined_mtp:
            if adapter.family is not TargetPositionFamily.QWEN35_TEXT_HYBRID:
                raise TargetPositionError(
                    "sparse native MTP composition supports exact Qwen text targets only"
                )
            capability = getattr(target_model, "mtp_capability", None)
            if capability is None or not capability.supported:
                reason = (
                    "native_mtp_model_capability_missing"
                    if capability is None
                    else capability.reason
                )
                raise TargetPositionError(reason)
            self._resolve_native_mtp_sparse_api()
        # Installation is topology-preserving and immutable.  Doing it here
        # guarantees executor admission before scorer/RNG state can advance.
        TargetPositionHooks.for_model(target_model, adapter)
        return profile_key, adapter

    @staticmethod
    def _resolve_native_mtp_sparse_api() -> tuple[Any, Any, Any, Any]:
        """Resolve the exact attested sparse-bootstrap API or fail closed."""
        try:
            from mlx_lm import stream_generate as mlx_stream_generate
            from mlx_lm.generate import (
                GenerationForwardPhase,
                NativeMTPSparseBootstrap,
                abandon_native_mtp_sparse_receipts,
                attested_target_forward,
            )
        except (ImportError, AttributeError) as exc:
            raise TargetPositionError(
                "mlx-lm does not expose attested native MTP sparse bootstrap"
            ) from exc
        try:
            parameters = inspect.signature(mlx_stream_generate).parameters
        except (TypeError, ValueError) as exc:
            raise TargetPositionError(
                "mlx-lm sparse native MTP public dispatch is unavailable"
            ) from exc
        required = {
            "mtp",
            "mtp_sampling_config",
            "model_forward_context",
            "sparse_bootstrap",
        }
        if not required.issubset(parameters):
            raise TargetPositionError(
                "mlx-lm sparse native MTP public dispatch is unavailable"
            )
        if not callable(
            getattr(NativeMTPSparseBootstrap, "try_abandon_unclaimed", None)
        ):
            raise TargetPositionError(
                "mlx-lm sparse native MTP atomic abandonment is unavailable"
            )
        return (
            GenerationForwardPhase,
            NativeMTPSparseBootstrap,
            attested_target_forward,
            abandon_native_mtp_sparse_receipts,
        )

    def _prepare_sparse_target_prefill(
        self,
        *,
        target_model: Any,
        tokenizer: Any,
        tokens: list[int],
        importance: Any,
        cache: list[Any],
        telemetry: _SpecPrefillTelemetry,
        keep_pct: float,
        backbone_pct: float,
        chunk_size: int,
        halo_chunks: int,
        anchor_chunks: int,
        profile_key: SpecPrefillProfileKey,
        adapter: TargetPositionAdapter,
        combined_mtp: bool = False,
        cancel_check: Any | None = None,
    ) -> tuple[Any, ...]:
        """Build exact sparse state and run a fresh target prefill atomically.

        This is intentionally the only SimpleEngine entry point to the new
        executor.  It rejects missing artifact identity and any non-fresh
        target cache before a sparse token can be emitted.
        """
        if not tokens:
            raise TargetPositionError("SpecPrefill requires a non-empty prompt")
        if profile_key.adapter_id != adapter.adapter_id:
            raise TargetPositionError("sparse profile/adapter admission changed")
        tuning = SparsePolicyTuning(
            keep_pct=keep_pct,
            backbone_pct=backbone_pct,
            halo_chunks=halo_chunks,
            anchor_chunks=anchor_chunks,
            chunk_size=chunk_size,
        )
        plan = build_selection_plan(
            importance,
            keep_pct=keep_pct,
            backbone_pct=backbone_pct,
            chunk_size=chunk_size,
            halo_chunks=halo_chunks,
            anchor_chunks=anchor_chunks,
            control_token_indices=tuple(
                sorted(
                    set(self._control_token_indices(tokenizer, tokens))
                    | {0, len(tokens) - 1}
                )
            ),
            rotating_tail_requirement=self._rotating_tail_requirement(cache),
        )
        if (
            plan.selected_indices[0] != 0
            or plan.selected_indices[-1] != len(tokens) - 1
        ):
            raise TargetPositionError(
                "selection plan must retain the first and final prompt tokens"
            )
        if (
            telemetry.profile_selector_version is not None
            and telemetry.profile_selector_version != plan.selector_version
        ):
            raise TargetPositionError(
                "calibrated selector version does not match the executed selection plan"
            )
        identity = SparseCacheIdentity.from_tokens(
            target_id=(
                f"{profile_key.target_artifact_id}@{profile_key.target_artifact_hash}"
                f"#adapter={adapter.adapter_id}"
            ),
            tokenizer_id=f"sha256:{profile_key.tokenizer_artifact_hash}",
            scorer_id=f"{profile_key.scorer_artifact_id}@{profile_key.scorer_artifact_hash}",
            selector_version=plan.selector_version,
            tuning=tuning,
            tokens=tokens,
            selection_fingerprint=plan.fingerprint,
        )
        sparse_state = SparseCacheState.from_selection(
            identity,
            (plan.selected_indices,),
            (len(tokens),),
        )
        target_forward = None
        receipt_abandon = None
        bootstrap_type = None
        if combined_mtp:
            (
                phase_type,
                bootstrap_type,
                attested_target_forward,
                receipt_abandon,
            ) = self._resolve_native_mtp_sparse_api()
            hooks = TargetPositionHooks.for_model(target_model, adapter)

            @contextmanager
            def _attested_forward_context(forward):
                if forward.model is not target_model or forward.cache is not cache:
                    raise TargetPositionError(
                        "attested target model/cache identity changed"
                    )
                if forward.phase is not phase_type.PREFILL:
                    raise TargetPositionError("attested sparse target phase changed")
                positions = tuple(forward.logical_positions)
                if positions != active_plan.logical_positions[0]:
                    raise TargetPositionError(
                        "attested sparse target logical positions changed"
                    )
                if tuple(forward.input_tokens.shape) != (1, len(positions)):
                    raise TargetPositionError("attested sparse target requires B=1")
                ack = getattr(forward, "logical_position_ack", None)
                if not callable(getattr(ack, "acknowledge", None)):
                    raise TargetPositionError(
                        "attested sparse target requires a position consumer"
                    )
                with hooks.session_for_plan(active_plan, logical_position_ack=ack):
                    yield

            active_plan: Any = None

            def _attested_chunk(token_rows, target_cache, chunk_plan):
                nonlocal active_plan
                if target_cache is not cache:
                    raise TargetPositionError("sparse target cache identity changed")
                positions = chunk_plan.logical_positions[0]
                token_ids = tuple(tokens[position] for position in positions)
                if tuple(token_rows.shape) != (1, len(token_ids)):
                    raise TargetPositionError("attested sparse target requires B=1")
                successors = tuple(
                    tokens[position + 1]
                    for position in positions
                    if position < len(tokens) - 1
                )
                expected_successors = len(positions)
                if positions[-1] == len(tokens) - 1:
                    expected_successors -= 1
                if len(successors) != expected_successors:
                    raise TargetPositionError(
                        "receipt chunk does not contain exact original successors"
                    )
                active_plan = chunk_plan
                try:
                    (logits, _hidden), receipt = attested_target_forward(
                        target_model,
                        token_ids,
                        cache,
                        phase=phase_type.PREFILL,
                        logical_positions=positions,
                        immediate_successor_token_ids=successors,
                        model_forward_context=_attested_forward_context,
                    )
                finally:
                    active_plan = None
                return logits, receipt

            target_forward = _attested_chunk

        result = execute_sparse_target_prefill(
            target_model,
            [tokens[index] for index in plan.selected_indices],
            cache,
            sparse_state,
            adapter,
            step_size=self._prefill_step_size,
            cancel_check=cancel_check,
            target_forward=target_forward,
            receipt_abandon=receipt_abandon,
        )
        telemetry.selected_tokens = result.telemetry.selected_tokens
        telemetry.target_prefill_ms = result.telemetry.target_prefill_ms
        sparse_bootstrap = None
        forward_context = None
        receipts = tuple(result.forward_receipts)
        try:
            forward_context = SparseGenerationForwardContext(
                target_model, cache, result.cache_state, adapter
            )
            if combined_mtp:
                selected_positions = tuple(plan.selected_indices)
                selected_token_ids = tuple(
                    tokens[position] for position in selected_positions
                )
                immediate_successors = tuple(
                    tokens[position + 1] for position in selected_positions[:-1]
                )
                sparse_bootstrap = bootstrap_type(
                    receipts=receipts,
                    selected_logical_positions=selected_positions,
                    selected_token_ids=selected_token_ids,
                    immediate_successor_token_ids=immediate_successors,
                    target_cache=cache,
                    next_logical_position=len(tokens),
                )
        except BaseException as preparation_error:
            if combined_mtp:
                if sparse_bootstrap is None:
                    try:
                        receipt_abandon(receipts)
                    except BaseException as close_error:
                        raise _SpecPrefillAuthorityError(
                            "sparse native MTP authority cleanup failed"
                        ) from close_error
                elif not _try_abandon_sparse_bootstrap(sparse_bootstrap):
                    raise _SpecPrefillAuthorityError(
                        "sparse native MTP bootstrap claim already began"
                    )
            if forward_context is not None:
                forward_context.finish()
            raise preparation_error
        prepared = (
            result,
            forward_context,
            plan,
        )
        return (*prepared, sparse_bootstrap) if combined_mtp else prepared

    @staticmethod
    def _specprefill_dense_fallback(
        decision: SpecPrefillDecision, reason: str | None
    ) -> SpecPrefillDecision:
        return SpecPrefillDecision(
            requested_policy=decision.requested_policy,
            effective_policy=SpecPrefillPolicy.DENSE,
            coverage=decision.coverage,
            fallback_reason=reason,
        )

    def _specprefill_controls(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove public prefill controls before forwarding to model APIs."""
        controls = {
            "legacy": kwargs.pop("specprefill", None),
            "policy": kwargs.pop("specprefill_policy", None),
            "coverage": kwargs.pop("specprefill_coverage", None),
            "has_media": bool(kwargs.pop("specprefill_has_media", False)),
            "keep_pct": kwargs.pop("specprefill_keep_pct", None),
            "backbone_pct": kwargs.pop("specprefill_backbone_pct", None),
        }
        if not self._specprefill_diagnostic_mode and (
            controls["keep_pct"] is not None or controls["backbone_pct"] is not None
        ):
            raise ValueError(
                "specprefill_keep_pct and specprefill_backbone_pct are "
                "diagnostic-only controls"
            )
        return controls

    def _has_eligible_sparse_chat_intent(self, kwargs: dict[str, Any]) -> bool:
        """Whether chat must use the stream seam to permit sparse prefill.

        The direct ``model.chat`` path has no sparse-prefill execution seam.
        This predicate deliberately does not treat a loaded scorer as intent;
        only a declared selective auto request or diagnostic sparse request may
        switch a chat request onto that seam. Prompt-length admission remains
        authoritative in the streaming implementation after template rendering.
        """
        policy = kwargs.get("specprefill_policy")
        legacy = kwargs.get("specprefill")
        coverage = kwargs.get("specprefill_coverage", "unknown")
        has_media = bool(kwargs.get("specprefill_has_media", False))
        if has_media or self._draft_model is None:
            return False
        if policy is None:
            policy = (
                "sparse" if legacy is True else "dense" if legacy is False else "auto"
            )
        if policy == "auto":
            return coverage == "selective"
        return policy == "sparse" and self._specprefill_diagnostic_mode

    @staticmethod
    def _prompt_add_special_tokens(tokenizer: Any, prompt: str) -> bool:
        """Tokenize without assuming a concrete tokenizer attribute type.

        Third-party wrappers and test doubles may expose ``bos_token`` as a
        sentinel object rather than a string. Such a value cannot be passed to
        ``str.startswith`` and should conservatively be treated as no BOS.
        """
        bos_token = getattr(tokenizer, "bos_token", None)
        return not isinstance(bos_token, str) or not prompt.startswith(bos_token)

    @classmethod
    def _encode_prompt_tokens(cls, tokenizer: Any, prompt: str) -> list[int]:
        return tokenizer.encode(
            prompt,
            add_special_tokens=cls._prompt_add_special_tokens(tokenizer, prompt),
        )

    @staticmethod
    def _clone_cache_state(value: Any) -> Any:
        """Copy cache state containers without duplicating immutable MLX arrays."""
        if isinstance(value, tuple):
            return tuple(SimpleEngine._clone_cache_state(v) for v in value)
        if isinstance(value, list):
            return [SimpleEngine._clone_cache_state(v) for v in value]
        return value

    @classmethod
    def _snapshot_prompt_cache(cls, prompt_cache: list[Any]) -> list[Any]:
        """Capture cache states without aliasing mutable state containers."""
        return [cls._clone_cache_state(c.state) for c in prompt_cache]

    @classmethod
    def _restore_prompt_cache(
        cls, prompt_cache: list[Any], snapshot: list[Any]
    ) -> None:
        """Restore cache states without letting decode mutate the saved snapshot."""
        for i, saved_state in enumerate(snapshot):
            prompt_cache[i].state = cls._clone_cache_state(saved_state)

    @staticmethod
    def _iter_cache_state_arrays(value: Any):
        if isinstance(value, (tuple, list)):
            for item in value:
                yield from SimpleEngine._iter_cache_state_arrays(item)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            yield value

    @classmethod
    def _eval_cache_snapshot(cls, snapshot: list[Any]) -> None:
        arrays = list(cls._iter_cache_state_arrays(snapshot))
        if arrays:
            mx.eval(arrays)

    @staticmethod
    def _cache_class_is_system_snapshot_safe(cache_entry: Any) -> bool:
        try:
            from mlx_lm.models.cache import ArraysCache, KVCache

            return isinstance(cache_entry, (KVCache, ArraysCache))
        except Exception:
            cache_type = type(cache_entry).__name__
            return cache_type in {"KVCache", "ArraysCache"}

    @classmethod
    def _probe_system_kv_cache_support(cls, model: Any, route: str) -> bool:
        try:
            from mlx_lm.models.cache import make_prompt_cache

            probe_cache = make_prompt_cache(model)
            supported = bool(probe_cache) and all(
                cls._cache_class_is_system_snapshot_safe(c) for c in probe_cache
            )
            if not supported:
                cache_types = sorted({type(c).__name__ for c in probe_cache})
                logger.info(
                    "System KV cache snapshot disabled (%s): model returned "
                    "unsupported cache entries (%s); requests will use the "
                    "uncached path",
                    route,
                    cache_types,
                )
            return supported
        except Exception as e:
            logger.debug(
                "System KV cache support probe failed (%s, %s); "
                "disabling snapshot path",
                route,
                e,
            )
            return False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        return self._is_mllm

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        if not self._loaded or self._model is None:
            return None
        if self._is_mllm:
            return getattr(self._model, "processor", None)
        return self._model.tokenizer

    def native_mtp_server_state(
        self, *, has_media: bool = False
    ) -> NativeMTPServerState:
        """Return native-MTP admission facts without mutating model state."""

        incompatibility = None
        if has_media:
            incompatibility = "native_mtp_media_unsupported"
        elif (self._max_kv_size or 0) > 0:
            incompatibility = "native_mtp_max_kv_unsupported"
        elif self._system_kv_cache:
            # Native MTP cannot consume the cache snapshot contract yet.  Do
            # not guess that a resident prefix belongs to another request.
            incompatibility = "native_mtp_prefix_cache_unsupported"

        if self._is_mllm:
            target = self._text_model
        else:
            target = getattr(self._model, "model", None)
        capability = getattr(target, "mtp_capability", None)
        capable = bool(capability is not None and capability.supported)
        if incompatibility is None and not capable:
            incompatibility = (
                "native_mtp_model_capability_missing"
                if capability is None
                else capability.reason
            )
        if incompatibility is None and resolve_native_mtp_consumer() is None:
            incompatibility = "native_mtp_consumer_contract_missing"
        return NativeMTPServerState(
            server_default=bool(self._mtp),
            capable=capable,
            num_draft_tokens=self._mtp_num_draft_tokens,
            supports_penalty_processors=not self._is_mllm,
            incompatibility=incompatibility,
        )

    def _generation_lock_holder_summary(self) -> str:
        if not self._active_requests:
            return "none"

        holders = []
        now = time.time()
        for request_id, info in self._active_requests.items():
            elapsed_s = info.get("elapsed_s")
            started_at = info.get("started_at")
            if started_at is not None:
                elapsed_s = round(now - started_at, 1)
            kind = info.get("kind", "unknown")
            status = info.get("status", "unknown")
            holders.append(
                f"{request_id}:{status}:{kind}:"
                f"prompt={info.get('prompt_tokens', 0)}:"
                f"completion={info.get('completion_tokens', 0)}:"
                f"elapsed_s={elapsed_s if elapsed_s is not None else 'unknown'}"
            )
        return ",".join(holders)

    @asynccontextmanager
    async def _acquire_generation_slot(self, request_id: str):
        """Admission control for SimpleEngine's serialized MLX route."""
        if (
            self._generation_lock_admission == "fail_fast"
            and self._generation_lock.locked()
        ):
            self._generation_busy_rejections += 1
            raise EngineBusy(
                "SimpleEngine serialized route is busy; "
                f"request_id={request_id}; "
                f"active={self._generation_lock_holder_summary()}; "
                f"waiters={self._generation_waiters}; "
                "retry later"
            )

        self._generation_waiters += 1
        acquired = False
        try:
            async with self._generation_lock:
                acquired = True
                self._generation_waiters -= 1
                yield
        finally:
            if not acquired and self._generation_waiters > 0:
                self._generation_waiters -= 1

    def prepare_for_start(self) -> None:
        """Load the backing model off the serving event loop."""
        if self._model is not None:
            return

        if self._is_mllm:
            from ..models.mllm import MLXMultimodalLM

            self._model = MLXMultimodalLM(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                enable_cache=self._enable_cache,
                max_kv_size=self._max_kv_size,
                draft_model=self._mllm_draft_model_path,
                draft_kind=self._mllm_draft_kind,
                draft_block_size=self._mllm_draft_block_size,
            )
        else:
            from ..models.llm import MLXLanguageModel

            self._model = MLXLanguageModel(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                mtp=self._mtp,
                mtp_num_draft_tokens=self._mtp_num_draft_tokens,
            )

        self._model.load()

    def _uses_default_prepare_for_start(self) -> bool:
        """Return True when prepare_for_start is the class implementation."""
        method = getattr(self.prepare_for_start, "__func__", None)
        return method is SimpleEngine.prepare_for_start

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return
        try:
            if self._model is None:
                if self._uses_default_prepare_for_start():
                    # MLX generation streams are thread-local. Keep model load on
                    # the event-loop thread so default LLM stream_generate() runs
                    # on the same thread that owns model-associated streams.
                    self.prepare_for_start()
                else:
                    # Test doubles and custom overrides may block; preserve the
                    # cancellation-safe threaded startup helper for those cases.
                    await run_blocking_startup_work(self.prepare_for_start)
            self._loaded = True

            if self._mtp and self._mtp_num_draft_tokens != 1:
                logger.warning(
                    "Native mlx_lm MTP currently ignores num_draft_tokens=%d; "
                    "effective speculative draft depth remains 1",
                    self._mtp_num_draft_tokens,
                )

            # Probe whether this model's prompt cache is snapshot-safe for the
            # stream_chat system-prefix cache branch. This is also refreshed
            # below for MLLM text routing after the parallel TextModel exists.
            if not self._is_mllm and self._model is not None:
                backing_model = getattr(self._model, "model", self._model)
                self._supports_system_kv_cache = self._probe_system_kv_cache_support(
                    backing_model,
                    "stream_chat",
                )

            # Build parallel mlx_lm TextModel for text-only routing.
            # Even when MTP is disabled, text-only requests should not be trapped
            # on the slower mlx_vlm multimodal path.
            if self._is_mllm and self._should_route_text_through_text_model():
                try:
                    from ..text_model_from_vlm import build_text_model

                    self._text_model = build_text_model(
                        self._model.model, self._model_name
                    )

                    if self._text_model is not None:
                        self._text_tokenizer = _build_text_tokenizer(
                            self._model.processor
                        )
                        self._supports_system_kv_cache = (
                            self._probe_system_kv_cache_support(
                                self._text_model,
                                "mllm_text",
                            )
                        )

                        # Apply Qwen3.5 eos_token fix (matches MLXLanguageModel.load)
                        if "qwen3" in self._model_name.lower():
                            self._text_tokenizer.eos_token = "<|im_end|>"
                            self._text_tokenizer.eos_token_id = (
                                self._text_tokenizer.convert_tokens_to_ids("<|im_end|>")
                            )
                            self._text_tokenizer.eos_token_ids = (
                                self._text_tokenizer.eos_token_id,
                            )

                        # Probe the derived TextModel's prompt cache for snapshot-safety
                        # (same gate stream_chat uses for the pure-LLM path).
                        # _stream_generate_text only enters the system-KV cache branch
                        # when this flag is True, so sliding-window text models won't
                        # desynchronize on restore.
                        #
                        # Probe args must match the runtime constructor in
                        # _stream_generate_text (max_kv_size=self._max_kv_size or None).
                        # Under bounded-KV serving (max_kv_size > 0) make_prompt_cache
                        # returns RotatingKVCache for models without a custom
                        # make_cache; probing with default args would mis-classify that
                        # path as snapshot-safe.
                        try:
                            from mlx_lm.models.cache import KVCache, make_prompt_cache

                            probe_cache = make_prompt_cache(
                                self._text_model, max_kv_size=self._max_kv_size or None
                            )
                            self._supports_system_kv_cache = bool(probe_cache) and all(
                                isinstance(c, KVCache) for c in probe_cache
                            )
                            if not self._supports_system_kv_cache:
                                cache_types = sorted(
                                    {type(c).__name__ for c in probe_cache}
                                )
                                logger.info(
                                    "System KV cache snapshot disabled for MLLM "
                                    "text routing: TextModel returned non-KVCache "
                                    "entries (%s); _stream_generate_text will use "
                                    "the uncached path",
                                    cache_types,
                                )
                        except Exception as e:
                            logger.debug(
                                "MLLM TextModel KV cache support probe failed "
                                "(%s); disabling snapshot path",
                                e,
                            )
                            self._supports_system_kv_cache = False

                        capability = getattr(self._text_model, "mtp_capability", None)
                        has_mtp = bool(capability is not None and capability.supported)
                        logger.info(
                            "MLLM text routing: text-only -> mlx_lm TextModel "
                            "(MTP=%s), media -> mlx_vlm",
                            has_mtp and self._mtp,
                        )
                    else:
                        self._text_model = None
                        self._text_tokenizer = None

                except Exception as e:
                    logger.error("MLLM text routing setup failed: %s", e)
                    self._text_model = None
                    self._text_tokenizer = None

            # Load SpecPrefill draft model (small model for importance scoring)
            if self._specprefill_enabled and self._specprefill_draft_model_path:
                try:
                    from mlx_lm import load as mlx_lm_load

                    self._draft_model, _ = mlx_lm_load(
                        self._specprefill_draft_model_path
                    )
                    logger.info(
                        "SpecPrefill: draft model loaded (%s), threshold=%d, keep=%.0f%%",
                        self._specprefill_draft_model_path,
                        self._specprefill_threshold,
                        self._specprefill_keep_pct * 100,
                    )
                except Exception as e:
                    logger.error("SpecPrefill: draft model load failed: %s", e)
                    self._draft_model = None

            # Warn if MTP is enabled without continuous-batching and text routing not available
            if self._mtp and (not self._is_mllm or self._text_model is None):
                logger.warning(
                    "[MTP] --enable-mtp without --continuous-batching: "
                    "speculative decoding via draft tokens will not be active. "
                    "For full MTP support, use: --enable-mtp --continuous-batching"
                )

            mtp_info = ""
            if self._mtp:
                mtp_info = (
                    f", MTP={self._mtp}(configured={self._mtp_num_draft_tokens}, "
                    "effective=1)"
                )
            routing = ", routing=per-request" if self._text_model is not None else ""
            specprefill_info = (
                ", SpecPrefill=active" if self._draft_model is not None else ""
            )
            logger.info(
                f"SimpleEngine loaded: {self._model_name} "
                f"(MLLM={self._is_mllm}{mtp_info}{routing}{specprefill_info})"
            )
        except asyncio.CancelledError:
            await cleanup_startup_cancellation(self.stop)
            raise

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._text_model = None
        self._text_tokenizer = None
        self._draft_model = None
        self._loaded = False
        self._system_kv_cache.clear()
        for k in self._system_kv_cache_stats:
            self._system_kv_cache_stats[k] = 0
        self._supports_system_kv_cache = False
        logger.info("SimpleEngine stopped")

    def _should_route_text_through_text_model(
        self, *, mllm_draft_requested: bool = False
    ) -> bool:
        """Return whether text-only MLLM requests may use mlx_lm TextModel."""
        return not (mllm_draft_requested and self._mllm_draft_model_path is not None)

    async def _run_blocking_serialized(
        self,
        func,
        /,
        *args,
        request_id: str | None = None,
        on_cancel=None,
        **kwargs,
    ):
        """Run a blocking MLX operation under the generation lock.

        Cancellation must not release the async lock before the worker thread
        finishes, or a follow-up request can enter MLX/Metal concurrently and
        corrupt the command-buffer state.
        """
        request_id = request_id or f"simple-{id(func):x}"
        async with self._acquire_generation_slot(request_id):
            started_at = time.time()
            self._active_requests[request_id] = {
                "request_id": request_id,
                "status": "running",
                "kind": "blocking_serialized",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_s": 0.0,
                "started_at": started_at,
            }

            def run_bound():
                _bind_worker_generation_streams()
                return func(*args, **kwargs)

            task = asyncio.create_task(asyncio.to_thread(run_bound))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if on_cancel is not None:
                    try:
                        on_cancel()
                    except Exception:
                        logger.debug(
                            "Blocking worker cancellation callback failed",
                            exc_info=True,
                        )
                try:
                    await task
                except BaseException:
                    pass
                raise
            finally:
                self._active_requests.pop(request_id, None)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Thin accumulator over stream_generate(). stream_generate() is the
        only code path that consumes per-request SpecPrefill overrides
        (`specprefill`, `specprefill_keep_pct`) and routes through
        _stream_generate_specprefill() when engaged. The prior direct
        self._model.generate() path silently dropped those overrides for
        non-streaming /v1/completions callers, so extra_body.specprefill
        was advertised by the server but had no effect on this route.

        By iterating stream_generate() and returning the last
        GenerationOutput, non-streaming clients get the same SpecPrefill
        engagement, accurate prompt_tokens reporting, and per-request
        override support as streaming clients.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional parameters forwarded to stream_generate,
                including per-request `specprefill` / `specprefill_keep_pct`

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        last_output: GenerationOutput | None = None
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        ):
            last_output = output

        if last_output is None:
            return GenerationOutput(text="", finish_reason="stop")

        text = clean_output_text(last_output.text)
        return GenerationOutput(
            text=text,
            tokens=list(last_output.tokens),
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            finish_reason=last_output.finish_reason,
            finished=True,
            mtp_drafts=last_output.mtp_drafts,
            mtp_accepted=last_output.mtp_accepted,
            mtp_bypass_reason=last_output.mtp_bypass_reason,
            logprobs=last_output.logprobs,
            specprefill_requested_policy=last_output.specprefill_requested_policy,
            specprefill_effective_policy=last_output.specprefill_effective_policy,
            specprefill_coverage=last_output.specprefill_coverage,
            specprefill_engaged=last_output.specprefill_engaged,
            specprefill_selector_version=last_output.specprefill_selector_version,
            specprefill_fallback_reason=last_output.specprefill_fallback_reason,
            specprefill_total_tokens=last_output.specprefill_total_tokens,
            specprefill_selected_tokens=last_output.specprefill_selected_tokens,
            specprefill_scorer_ms=last_output.specprefill_scorer_ms,
            specprefill_target_prefill_ms=last_output.specprefill_target_prefill_ms,
        )

    async def _track_request_stream(
        self,
        source_gen: AsyncIterator[GenerationOutput],
        *,
        max_tokens: int = 0,
    ) -> AsyncIterator[GenerationOutput]:
        """Yield-through wrapper that records per-request live state and
        final ``prompt_tokens``/``completion_tokens`` counters.

        Mirrors the fields BatchedEngine emits per running request
        (``request_id``, ``phase``, ``elapsed_s``, ``ttft_s``,
        ``tokens_per_second``, ``progress``, ...) so dashboards built
        against ``/v1/status`` show individual in-flight requests for
        SimpleEngine-backed services as well (Gemma 4 31B + MTP, etc.).

        Re-entrant calls (e.g. the cache-fallback path inside
        ``_stream_chat_impl`` that delegates to ``self.stream_generate``)
        are detected via the ``_in_tracker`` context variable and pass
        through without a second tracking entry, so each external
        request is counted exactly once.

        Note: we deliberately use ``set(True)``/``set(False)`` rather
        than ``set(token)``/``reset(token)``. FastAPI/uvicorn finalize
        streaming generators from a different async context than the
        one that created them; ``ContextVar.reset(token)`` raises
        ``ValueError`` in that case ("Token was created in a different
        Context"), which surfaces as a terminal-frame streaming error.
        ``set(False)`` works in any context and the contextvar is only
        consumed inside this method, so there is no value to preserve.
        """
        if _in_tracker.get():
            primary_error: BaseException | None = None
            try:
                async for output in source_gen:
                    yield output
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                await _aclose_async_iterator(source_gen, primary_error)
            return
        _in_tracker.set(True)
        request_id = str(uuid.uuid4())
        start = time.time()
        ttft_s: float | None = None
        last_p = 0
        last_c = 0
        entry: dict[str, Any] = {
            "request_id": request_id,
            "status": "running",
            "phase": "prefill",
            "elapsed_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_tokens": max_tokens,
            "progress": 0.0,
            "tokens_per_second": 0.0,
            "ttft_s": None,
            "cache_hit_type": None,
            "cached_tokens": 0,
        }
        self._active_requests[request_id] = entry
        self._num_running += 1
        primary_error: BaseException | None = None
        try:
            async for output in source_gen:
                now = time.time()
                if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                    last_p = output.prompt_tokens
                    entry["prompt_tokens"] = last_p
                if hasattr(output, "completion_tokens") and output.completion_tokens:
                    if ttft_s is None:
                        ttft_s = now - start
                        entry["ttft_s"] = round(ttft_s, 3)
                        entry["phase"] = "generation"
                    last_c = output.completion_tokens
                    entry["completion_tokens"] = last_c
                entry["elapsed_s"] = round(now - start, 2)
                if max_tokens > 0:
                    entry["progress"] = round(min(1.0, last_c / max_tokens), 3)
                if ttft_s is not None and last_c > 0:
                    gen_elapsed = max(1e-3, (now - start) - ttft_s)
                    entry["tokens_per_second"] = round(last_c / gen_elapsed, 1)
                yield output
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await _aclose_async_iterator(source_gen, primary_error)
            finally:
                self._active_requests.pop(request_id, None)
                self._num_running = max(0, self._num_running - 1)
                if last_c > 0:
                    duration = time.time() - start
                    self._total_requests_processed += 1
                    self._total_prompt_tokens += last_p
                    self._total_completion_tokens += last_c
                    self._recent_completions.append((last_c, duration))
                _in_tracker.set(False)

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Public stream-generate wrapper with request stats tracking."""
        tracked = self._track_request_stream(
            self._stream_generate_impl(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ),
            max_tokens=max_tokens,
        )
        primary_error: BaseException | None = None
        try:
            async for output in tracked:
                yield output
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            await _aclose_async_iterator(tracked, primary_error)

    async def _stream_generate_impl(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Resolve the complete public prefill contract before any model call.
        # These controls must never leak into mlx-lm/mlx-vlm kwargs.
        specprefill_controls = self._specprefill_controls(kwargs)
        native_mtp_config, native_mtp_disabled, effective_mtp, mtp_bypass_reason = (
            _consume_native_mtp_request(kwargs, server_default=self._mtp)
        )
        request_id = str(kwargs.pop("request_id", "") or f"simple-{id(prompt):x}")

        tokenizer = self._model.tokenizer
        tokens_list = self._encode_prompt_tokens(tokenizer, prompt)
        telemetry = self._resolve_specprefill_telemetry(
            legacy=specprefill_controls["legacy"],
            policy=specprefill_controls["policy"],
            coverage=specprefill_controls["coverage"],
            has_media=specprefill_controls["has_media"],
            total_tokens=len(tokens_list),
            # MLXLanguageModel injects native MTP into its decode call when
            # configured. Other routes pass their request-effective value
            # explicitly (for example, an MLLM assistant request).
            combined_mtp=effective_mtp,
        )

        # SpecPrefill is independent from MTP. The direct path is used only
        # for non-MLLM requests; the MLLM text route resolves the same contract
        # in _stream_generate_text.
        if (
            not self._is_mllm
            and telemetry.decision.effective_policy is SpecPrefillPolicy.SPARSE
            and max_tokens <= 0
        ):
            telemetry.fallback("sparse_no_completion_requested")
        if (
            not self._is_mllm
            and telemetry.decision.effective_policy is SpecPrefillPolicy.SPARSE
            and effective_mtp
            and native_mtp_config is None
        ):
            telemetry.fallback("native_mtp_request_contract_missing")

        if (
            not self._is_mllm
            and telemetry.decision.effective_policy is SpecPrefillPolicy.SPARSE
        ):
            sparse_stream = self._stream_generate_specprefill(
                prompt,
                tokens_list,
                max_tokens,
                temperature,
                top_p,
                stop=stop,
                telemetry=telemetry,
                native_mtp_request=native_mtp_config,
                mtp_bypass_reason=mtp_bypass_reason,
                specprefill_keep_pct=(
                    specprefill_controls["keep_pct"]
                    if self._specprefill_diagnostic_mode
                    else (
                        telemetry.profile_tuning.keep_pct
                        if telemetry.profile_tuning is not None
                        else None
                    )
                ),
                specprefill_backbone_pct=(
                    specprefill_controls["backbone_pct"]
                    if self._specprefill_diagnostic_mode
                    else (
                        telemetry.profile_tuning.backbone_pct
                        if telemetry.profile_tuning is not None
                        else None
                    )
                ),
                specprefill_chunk_size=(
                    telemetry.profile_tuning.chunk_size
                    if telemetry.profile_tuning is not None
                    else None
                ),
                specprefill_halo_chunks=(
                    telemetry.profile_tuning.halo_chunks
                    if telemetry.profile_tuning is not None
                    else None
                ),
                specprefill_anchor_chunks=(
                    telemetry.profile_tuning.anchor_chunks
                    if telemetry.profile_tuning is not None
                    else None
                ),
                **kwargs,
            )
            primary_error: BaseException | None = None
            try:
                async for output in sparse_stream:
                    yield output
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                # A public GeneratorExit can arrive while this wrapper is
                # suspended at ``yield``.  ``async for`` does not close the
                # delegated async iterator in that case, so explicitly drive
                # its cancellation/worker/request cleanup before the tracker
                # releases ownership.
                await _aclose_async_iterator(sparse_stream, primary_error)
            return

        async with self._acquire_generation_slot(request_id):
            started_at = time.time()
            self._active_requests[request_id] = {
                "request_id": request_id,
                "status": "running",
                "kind": "stream_generate",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_s": 0.0,
                "started_at": started_at,
            }
            # Non-stream chat runs in a worker thread and rebinds generation
            # streams there. Rebind again on the current thread before
            # stream_generate so nonstream->stream mode switches remain valid.
            _bind_worker_generation_streams()

            try:
                accumulated_text = ""
                prompt_tokens = 0
                completion_tokens = 0
                finished = False

                native_mtp_kwargs: dict[str, Any] = {}
                if native_mtp_config is not None:
                    native_mtp_kwargs["native_mtp_request"] = native_mtp_config
                elif native_mtp_disabled:
                    native_mtp_kwargs["native_mtp_disabled"] = True
                generation = self._model.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **native_mtp_kwargs,
                    **kwargs,
                )
                primary_error: BaseException | None = None
                try:
                    for chunk in generation:
                        prompt_tokens = (
                            chunk.prompt_tokens
                            if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens
                            else prompt_tokens
                        )
                        completion_tokens += 1
                        if request_id in self._active_requests:
                            self._active_requests[request_id].update(
                                {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "elapsed_s": round(time.time() - started_at, 1),
                                }
                            )
                        new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                        accumulated_text += new_text

                        finished = (
                            getattr(chunk, "finished", False)
                            or completion_tokens >= max_tokens
                        )
                        finish_reason = None
                        if finished:
                            finish_reason = getattr(chunk, "finish_reason", None)
                            if finish_reason is None:
                                finish_reason = (
                                    "length" if completion_tokens >= max_tokens else "stop"
                                )

                        yield GenerationOutput(
                            text=accumulated_text,
                            new_text=new_text,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            finished=finished,
                            finish_reason=finish_reason,
                            mtp_drafts=getattr(chunk, "mtp_drafts", 0),
                            mtp_accepted=getattr(chunk, "mtp_accepted", 0),
                            mtp_bypass_reason=(
                                mtp_bypass_reason
                                or getattr(chunk, "mtp_bypass_reason", None)
                            ),
                            logprobs=getattr(chunk, "logprobs", None),
                            **telemetry.as_output_kwargs(),
                        )

                        if finished:
                            break
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    close = getattr(generation, "close", None)
                    if close is not None:
                        try:
                            close()
                        except BaseException:
                            if primary_error is None:
                                raise
                            logger.warning(
                                "Failed to close model stream after generation error",
                                exc_info=True,
                            )

                if not finished:
                    if prompt_tokens == 0:
                        prompt_tokens = len(self._model.tokenizer.encode(prompt))
                    if request_id in self._active_requests:
                        self._active_requests[request_id].update(
                            {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "elapsed_s": round(time.time() - started_at, 1),
                            }
                        )
                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        finished=True,
                        finish_reason="stop",
                        mtp_bypass_reason=mtp_bypass_reason,
                        **telemetry.as_output_kwargs(),
                    )
            finally:
                self._active_requests.pop(request_id, None)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        if not self._loaded:
            await self.start()

        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})

        async def aggregate_stream_chat() -> GenerationOutput:
            final_output = GenerationOutput(text="")
            async for output in self.stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            ):
                final_output = output
            text = clean_output_text(final_output.text)
            return GenerationOutput(
                text=text,
                tokens=list(final_output.tokens),
                prompt_tokens=final_output.prompt_tokens,
                completion_tokens=final_output.completion_tokens,
                finish_reason=final_output.finish_reason,
                mtp_drafts=final_output.mtp_drafts,
                mtp_accepted=final_output.mtp_accepted,
                mtp_bypass_reason=final_output.mtp_bypass_reason,
                specprefill_requested_policy=final_output.specprefill_requested_policy,
                specprefill_effective_policy=final_output.specprefill_effective_policy,
                specprefill_coverage=final_output.specprefill_coverage,
                specprefill_engaged=final_output.specprefill_engaged,
                specprefill_selector_version=final_output.specprefill_selector_version,
                specprefill_fallback_reason=final_output.specprefill_fallback_reason,
                specprefill_total_tokens=final_output.specprefill_total_tokens,
                specprefill_selected_tokens=final_output.specprefill_selected_tokens,
                specprefill_scorer_ms=final_output.specprefill_scorer_ms,
                specprefill_target_prefill_ms=final_output.specprefill_target_prefill_ms,
            )

        # mlx-lm non-streaming chat with tools can stall indefinitely on some
        # local models, while the streaming path completes normally. Reuse the
        # streaming implementation and aggregate its final state so both chat
        # APIs share the same tool-capable execution path.
        if tools and not self._is_mllm:
            return await aggregate_stream_chat()

        # Keep the independently configurable prefill path identical for
        # streaming and non-streaming SimpleEngine chat. This is intentionally
        # not coupled to MTP.
        if not self._is_mllm and self._has_eligible_sparse_chat_intent(kwargs):
            return await aggregate_stream_chat()

        # Request-local logits processors (response_format / constrained JSON)
        # need token-boundary progress and cancellation.  The blocking
        # model.chat() call below only returns after the whole completion, so a
        # slow constrained decode can look like a no-progress non-stream wedge
        # and hold the serialized generation lock until max_tokens/timeout.
        if kwargs.get("logits_processors") and not self._is_mllm:
            return await aggregate_stream_chat()

        # Explicit native MTP is implemented by the streaming decode seam.
        # The legacy blocking chat call does not expose speculative controls.
        if kwargs.get("_native_mtp_request_config") is not None:
            return await aggregate_stream_chat()

        # Text-only requests on MLLM models should always aggregate the
        # streaming path for non-streaming chat. This keeps one execution seam
        # and avoids mlx_vlm non-stream thread/stream ownership mismatches.
        if self._is_mllm and not has_media_content(messages):
            return await aggregate_stream_chat()

        _, _, _, direct_mtp_bypass_reason = _consume_native_mtp_request(
            kwargs, server_default=self._mtp
        )

        # Convert tools for template if provided
        template_tools = convert_tools_for_template(tools) if tools else None

        if self._is_mllm:
            specprefill_controls = self._specprefill_controls(kwargs)
            telemetry = self._resolve_specprefill_telemetry(
                legacy=specprefill_controls["legacy"],
                policy=specprefill_controls["policy"],
                coverage=specprefill_controls["coverage"],
                has_media=(
                    has_media_content(messages) or specprefill_controls["has_media"]
                ),
                total_tokens=None,
            )
            if chat_template_kwargs:
                kwargs["chat_template_kwargs"] = chat_template_kwargs
            output = await self._run_blocking_serialized(
                self._model.chat,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=template_tools,
                **kwargs,
            )
            text = clean_output_text(output.text)
            return GenerationOutput(
                text=text,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finish_reason=output.finish_reason,
                mtp_drafts=getattr(output, "mtp_drafts", 0),
                mtp_accepted=getattr(output, "mtp_accepted", 0),
                mtp_bypass_reason=direct_mtp_bypass_reason,
                **telemetry.as_output_kwargs(),
            )
        else:
            # Direct dense chat cannot accept the public policy kwargs, and a
            # loaded scorer alone must not alter this route. Resolve after
            # prompt accounting so this path still publishes full telemetry.
            specprefill_controls = self._specprefill_controls(kwargs)
            output = await self._run_blocking_serialized(
                self._model.chat,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=template_tools,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            )
            text = clean_output_text(output.text)
            # Preserve upstream prompt accounting while routing the blocking
            # chat call through the cancellation-safe serialized runner.
            tokenizer = self._model.tokenizer
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if template_tools:
                template_kwargs["tools"] = template_tools
            prompt_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
            prompt_token_count = len(prompt_ids)
            telemetry = self._resolve_specprefill_telemetry(
                legacy=specprefill_controls["legacy"],
                policy=specprefill_controls["policy"],
                coverage=specprefill_controls["coverage"],
                has_media=specprefill_controls["has_media"],
                total_tokens=prompt_token_count,
            )
            return GenerationOutput(
                text=text,
                tokens=output.tokens,
                prompt_tokens=prompt_token_count,
                completion_tokens=len(output.tokens),
                finish_reason=output.finish_reason,
                mtp_bypass_reason=direct_mtp_bypass_reason,
                **telemetry.as_output_kwargs(),
            )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Public stream-chat wrapper with request stats tracking."""
        tracked = self._track_request_stream(
            self._stream_chat_impl(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            ),
            max_tokens=max_tokens,
        )
        primary_error: BaseException | None = None
        try:
            async for output in tracked:
                yield output
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            await _aclose_async_iterator(tracked, primary_error)

    async def _stream_chat_impl(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        (
            native_mtp_config,
            native_mtp_disabled,
            effective_native_mtp,
            mtp_bypass_reason,
        ) = _consume_native_mtp_request(kwargs, server_default=self._mtp)
        mllm_draft_requested = bool(kwargs.pop("mllm_draft", False))
        request_has_media = has_media_content(messages)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Per-request routing: text-only through mlx_lm TextModel
        routes_text_model = (
            self._is_mllm
            and self._text_model is not None
            and self._should_route_text_through_text_model(
                mllm_draft_requested=mllm_draft_requested
            )
            and not request_has_media
        )
        if routes_text_model:
            use_native_mtp = effective_native_mtp
            logger.info("Text-only request → LLM path (MTP=%s)", use_native_mtp)
            if chat_template_kwargs:
                kwargs["chat_template_kwargs"] = chat_template_kwargs
            if native_mtp_config is not None:
                kwargs["_native_mtp_request_config"] = native_mtp_config
            elif native_mtp_disabled:
                kwargs["_native_mtp_disabled"] = True
            elif mtp_bypass_reason is not None:
                kwargs["_native_mtp_bypass_reason"] = mtp_bypass_reason
            text_stream = self._stream_generate_text(
                messages,
                max_tokens,
                temperature,
                top_p,
                tools=template_tools,
                combined_mtp=use_native_mtp,
                **kwargs,
            )
            primary_error: BaseException | None = None
            try:
                async for chunk in text_stream:
                    yield chunk
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                await _aclose_async_iterator(text_stream, primary_error)
            return

        # Direct MLLM execution (including media) does not implement sparse
        # token selection. It still consumes and reports the public contract so
        # a media request is visibly dense rather than silently text-only.
        direct_mllm_telemetry: _SpecPrefillTelemetry | None = None
        if self._is_mllm:
            specprefill_controls = self._specprefill_controls(kwargs)
            direct_mllm_telemetry = self._resolve_specprefill_telemetry(
                legacy=specprefill_controls["legacy"],
                policy=specprefill_controls["policy"],
                coverage=specprefill_controls["coverage"],
                has_media=request_has_media or specprefill_controls["has_media"],
                total_tokens=None,
                combined_mtp=mllm_draft_requested,
            )

        def mllm_call_kwargs() -> dict:
            local_kwargs = dict(kwargs)
            if chat_template_kwargs:
                local_kwargs["chat_template_kwargs"] = chat_template_kwargs
            if mllm_draft_requested:
                local_kwargs["mllm_draft"] = True
            return local_kwargs

        # Build prompt using tokenizer
        if self._is_mllm:
            if self._text_model is not None:
                route_kind = "Media" if request_has_media else "Text-only"
                logger.info("%s request → MLLM path", route_kind)
            # For MLLM, use stream_chat which yields tokens incrementally.
            # Must hold the generation slot to prevent concurrent Metal access
            # (e.g. OpenCode sends title + main request simultaneously).
            accumulated_text = ""
            token_count = 0
            request_id = str(kwargs.pop("request_id", "") or f"simple-{id(messages):x}")
            native_video_request = bool(
                getattr(self._model, "_video_native", False) is True
                and self._model._collect_video_inputs(messages)
            )

            if not native_video_request:
                # Incremental mlx_vlm streams must stay on the model-owner
                # thread. Moving them through to_thread can raise a
                # Stream(gpu, N) ownership mismatch.
                local_kwargs = mllm_call_kwargs()

                async with self._acquire_generation_slot(request_id):
                    _bind_worker_generation_streams()
                    for chunk in self._model.stream_chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=template_tools,
                        **local_kwargs,
                    ):
                        token_count += 1
                        new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                        accumulated_text += new_text

                        finished = chunk.finish_reason is not None

                        yield GenerationOutput(
                            text=accumulated_text,
                            new_text=new_text,
                            prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                            completion_tokens=token_count,
                            finished=finished,
                            finish_reason=chunk.finish_reason if finished else None,
                            mtp_drafts=getattr(chunk, "mtp_drafts", 0),
                            mtp_accepted=getattr(chunk, "mtp_accepted", 0),
                            mtp_bypass_reason=mtp_bypass_reason,
                            **direct_mllm_telemetry.as_output_kwargs(),
                        )

                        if finished:
                            break
                return

            # mlx_vlm's native-video path is non-streaming and performs
            # blocking preprocessing and generation. Keep it off the event
            # loop while preserving serialized admission.
            def run_native_video():
                local_kwargs = mllm_call_kwargs()
                return list(
                    self._model.stream_chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=template_tools,
                        **local_kwargs,
                    )
                )

            chunks = await self._run_blocking_serialized(
                run_native_video,
                request_id=request_id,
            )
            for chunk in chunks:
                token_count += 1
                new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                accumulated_text += new_text
                finished = chunk.finish_reason is not None
                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                    completion_tokens=token_count,
                    finished=finished,
                    finish_reason=chunk.finish_reason if finished else None,
                    mtp_drafts=getattr(chunk, "mtp_drafts", 0),
                    mtp_accepted=getattr(chunk, "mtp_accepted", 0),
                    mtp_bypass_reason=mtp_bypass_reason,
                    **direct_mllm_telemetry.as_output_kwargs(),
                )
            return

        # For LLM, apply chat template and stream
        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            # Per-request enable_thinking override; default: True unless coder model.
            enable_thinking = kwargs.pop("enable_thinking", None)
            if enable_thinking is None:
                enable_thinking = "coder" not in self._model_name.lower()
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            if template_tools:
                template_kwargs["tools"] = template_tools
            safe_messages = normalize_messages_for_chat_template(messages)

            if getattr(self, "use_harmony_rendering", False):
                # GPT-OSS / harmony-format models: render via openai-harmony
                # instead of the Jinja chat_template. Bypasses the
                # ``extract_multimodal_content`` text-flattening upstream
                # (which drops structural ``tool_calls`` for non-native
                # parsers) and uses OpenAI's canonical renderer. See #568.
                from ..utils.harmony_render import (
                    render_messages as _harmony_render_messages,
                )

                _reasoning_effort = None
                if chat_template_kwargs:
                    _reasoning_effort = chat_template_kwargs.get("reasoning_effort")
                prompt = _harmony_render_messages(
                    safe_messages,
                    tools=template_tools,
                    reasoning_effort=_reasoning_effort,
                )
            else:
                try:
                    prompt = tokenizer.apply_chat_template(
                        safe_messages, **template_kwargs
                    )
                except TypeError:
                    # Some templates don't support all kwargs
                    for key in [
                        "tools",
                        "enable_thinking",
                        *chat_template_kwargs.keys(),
                    ]:
                        if key in template_kwargs:
                            del template_kwargs[key]
                    prompt = tokenizer.apply_chat_template(
                        safe_messages, **template_kwargs
                    )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        # --- System-prompt KV caching on the pure-LLM stream_chat path ---
        # Mirrors the cache in _stream_generate_text. Locates the system prefix
        # via probe-divergence (cf. prompt_warmup._build_strict_prefix_string):
        # render the template with two different user contents and take the
        # shared prefix. Works across Qwen/ChatML, Llama, Gemma, and any other
        # chat format -- no per-model marker list. Falls back to the original
        # uncached self.stream_generate() if the system prefix can't be
        # isolated or any step of the cache-aware path raises.
        cache_hit = False
        suffix_tokens = None
        system_tokens = None
        system_token_count = 0
        full_token_count = 0
        system_hash = None
        kv_cache_eligible = False
        # Snapshot reference captured at gate time so a concurrent MISS that
        # mutates ``self._system_kv_cache`` between the gate and the restore
        # (which runs later inside ``_run_blocking_serialized``) can't
        # desynchronize the restored KV from the hash that decided HIT.
        hit_snapshot: Any = None

        # Decode-control gate.
        # The cache branch below drives ``mlx_lm.stream_generate`` directly with only
        # ``prompt``, ``max_tokens``, ``sampler`` (built from temperature+top_p), and
        # ``prompt_cache``.
        # The uncached fallback threads ``**kwargs`` through ``self.stream_generate``,
        # which preserves ``stop``, request-local ``logits_processors`` (parser stop
        # tokens and JSON-constrained decoding attached by server.py per request), and
        # the ``top_k`` / ``min_p`` / ``presence_penalty`` / ``repetition_penalty``
        # sampling controls.
        # If the cache branch ran with any of those active, cache-eligible and uncached
        # requests would silently decode under different constraints.
        # Skip the cache branch in that case so both paths share identical decode
        # semantics.
        # server.py always supplies the no-op defaults (``top_k=0``, ``min_p=0.0``,
        # ``presence_penalty=0.0``, ``repetition_penalty=1.0``); compare against those
        # rather than ``key in kwargs`` so the common path still hits the cache.
        cache_blocking_controls: list[str] = []
        if kwargs.get("stop"):
            cache_blocking_controls.append("stop")
        if kwargs.get("logits_processors"):
            cache_blocking_controls.append("logits_processors")
        if (kwargs.get("top_k") or 0) > 0:
            cache_blocking_controls.append("top_k")
        if (kwargs.get("min_p") or 0.0) > 0.0:
            cache_blocking_controls.append("min_p")
        if (kwargs.get("presence_penalty") or 0.0) != 0.0:
            cache_blocking_controls.append("presence_penalty")
        if (kwargs.get("repetition_penalty") or 1.0) != 1.0:
            cache_blocking_controls.append("repetition_penalty")

        # Engine-feature gate.
        # The cache branch also bypasses engine-level features that
        # ``self.stream_generate`` (and the ``MLXLanguageModel.stream_generate``
        # wrapper underneath it) layer on top of ``mlx_lm.stream_generate``.
        # Same correctness reasoning as the decode-control gate: cache-eligible
        # and uncached requests must decode under identical engine semantics, so
        # skip the cache branch when any of these are active.
        # Specifically:
        #   - ``self._mtp`` injects ``mtp=True`` and ``num_draft_tokens`` into
        #     the mlx-lm call (see ``MLXLanguageModel.stream_generate``).
        #   - A loaded SpecPrefill draft model (``self._draft_model is not None``,
        #     set when ``specprefill_enabled`` + ``specprefill_draft_model`` are
        #     configured at engine init) routes large prompts through
        #     ``_stream_generate_specprefill`` instead of the plain stream path.
        #   - A per-request ``specprefill`` override from ``extra_body`` (popped
        #     by the wrapper from ``kwargs``) can force or suppress SpecPrefill
        #     for a single request.
        #     ``specprefill=False`` is a meaningful suppression signal — gate on
        #     ``is not None`` rather than truthiness so the wrapper sees it.
        #   - ``self._max_kv_size`` (when > 0) caps the prompt cache; the cache
        #     branch builds its cache with ``make_prompt_cache(model)`` and has
        #     no equivalent bound.
        if effective_native_mtp:
            cache_blocking_controls.append("mtp")
        if self._draft_model is not None:
            cache_blocking_controls.append("specprefill_loaded")
        if kwargs.get("specprefill") is not None:
            cache_blocking_controls.append("specprefill_request_override")
        if (self._max_kv_size or 0) > 0:
            cache_blocking_controls.append("max_kv_size")
        # Sliding-window models build their prompt cache from RotatingKVCache
        # entries whose ``.state`` aliases buffers that ``update_and_fetch``
        # mutates in place. Snapshot capture would corrupt the cached prefix
        # on the next decode. Probed once at start; ``False`` if the model
        # exposes any non-KVCache entries or the probe failed.
        if not self._supports_system_kv_cache:
            cache_blocking_controls.append("non_kv_cache_class")
        # The system-prefix probe (re-renders the conversation with two different
        # user contents and compares the rendered strings) goes through
        # ``tokenizer.apply_chat_template``. When the harmony rendering path is
        # active the actual prompt is built by ``openai-harmony`` instead, so the
        # probe and the prompt would diverge and the cache would never hit.
        # Falling back to the uncached path keeps correctness without splitting
        # the probe across both renderers.
        if getattr(self, "use_harmony_rendering", False):
            cache_blocking_controls.append("harmony_rendering")

        if cache_blocking_controls:
            logger.info(
                "System KV cache SKIP (stream_chat): request or engine has "
                "controls/features the cache branch cannot honor (%s); using "
                "uncached path",
                cache_blocking_controls,
            )

        # Normalize messages to plain dicts. The public stream_chat signature
        # types messages as list[dict], but internal callers (server.py,
        # tests) sometimes pass Pydantic Message objects directly; those
        # don't expose a dict-style .get() interface.
        def _to_msg_dict(m: Any) -> dict[str, Any]:
            if isinstance(m, dict):
                return m
            if hasattr(m, "model_dump"):
                return m.model_dump()
            if hasattr(m, "dict"):
                return m.dict()
            return {
                "role": getattr(m, "role", None),
                "content": getattr(m, "content", ""),
            }

        messages_for_cache = [_to_msg_dict(m) for m in messages]
        has_system = any(m.get("role") == "system" for m in messages_for_cache)
        if (
            has_system
            and not cache_blocking_controls
            and hasattr(tokenizer, "apply_chat_template")
        ):

            def _with_user(user_content: str) -> list[dict[str, Any]]:
                msgs = [dict(m) for m in messages_for_cache]
                if msgs and msgs[-1].get("role") == "user":
                    msgs[-1] = {**msgs[-1], "content": user_content}
                else:
                    msgs = [*msgs, {"role": "user", "content": user_content}]
                return msgs

            rendered_a: Any = None
            rendered_b: Any = None
            try:
                rendered_a = tokenizer.apply_chat_template(
                    _with_user("Alpha"), **template_kwargs
                )
                rendered_b = tokenizer.apply_chat_template(
                    _with_user("Bravo"), **template_kwargs
                )
            except Exception:
                pass

            if isinstance(rendered_a, str) and isinstance(rendered_b, str):
                boundary = 0
                diverged = False
                for i in range(min(len(rendered_a), len(rendered_b))):
                    if rendered_a[i] != rendered_b[i]:
                        diverged = True
                        break
                    boundary = i + 1

                if diverged and boundary >= 16:
                    system_prefix_text = rendered_a[:boundary]
                    system_hash = hashlib.sha256(
                        system_prefix_text.encode()
                    ).hexdigest()[:16]

                    add_special = self._prompt_add_special_tokens(tokenizer, prompt)
                    full_tokens_list = tokenizer.encode(
                        prompt, add_special_tokens=add_special
                    )
                    system_tokens_list = tokenizer.encode(
                        system_prefix_text, add_special_tokens=add_special
                    )
                    full_token_count = len(full_tokens_list)
                    system_token_count = len(system_tokens_list)

                    if (
                        len(full_tokens_list) > system_token_count
                        and full_tokens_list[:system_token_count] == system_tokens_list
                    ):
                        system_tokens = system_tokens_list
                        suffix_tokens = full_tokens_list[system_token_count:]
                        kv_cache_eligible = True
                        # Read the snapshot reference once. If we promote to
                        # HIT, ``hit_snapshot`` is the exact list the dict
                        # lookup just returned. A later concurrent MISS that
                        # mutates ``self._system_kv_cache`` before our
                        # serialized worker restores it cannot alias what we
                        # captured here — dict.get is atomic under the GIL
                        # and returns a reference to an immutable tuple.
                        candidate = self._system_kv_cache.get(system_hash)
                        if candidate is not None and system_token_count == candidate[1]:
                            cache_hit = True
                            hit_snapshot = candidate[0]
                            logger.info(
                                "System KV cache HIT (stream_chat): reusing %d "
                                "tokens, prefilling %d new (hash=%s)",
                                system_token_count,
                                len(suffix_tokens),
                                system_hash,
                            )
                        else:
                            logger.info(
                                "System KV cache MISS (stream_chat): will "
                                "prefill %d system + %d suffix tokens (hash=%s)",
                                system_token_count,
                                len(suffix_tokens),
                                system_hash,
                            )

        if kv_cache_eligible:
            # Cache-aware path: drive mlx-lm directly with a pre-populated cache.
            # Stream chunks back to the caller via an asyncio.Queue (mirrors
            # _stream_generate_text) so the client sees tokens as they arrive
            # rather than after the full generation finishes.
            loop = asyncio.get_running_loop()
            response_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            abort_event = threading.Event()

            def _emit_response(resp: Any) -> None:
                if abort_event.is_set():
                    return
                loop.call_soon_threadsafe(response_queue.put_nowait, ("resp", resp))

            def _emit_done() -> None:
                loop.call_soon_threadsafe(response_queue.put_nowait, ("done", None))

            def _emit_error(exc: BaseException) -> None:
                loop.call_soon_threadsafe(response_queue.put_nowait, ("error", exc))

            def _run_with_cache() -> None:
                from mlx_lm import stream_generate as mlx_stream_generate
                from mlx_lm.models.cache import make_prompt_cache
                from mlx_lm.sample_utils import make_sampler

                model = self._model.model
                sampler = make_sampler(temp=temperature, top_p=top_p)

                if cache_hit:
                    bc = make_prompt_cache(model)
                    # Restore from the closure-local reference captured at the
                    # gate, never from ``self._system_kv_cache`` directly:
                    # a concurrent MISS could have evicted the entry between
                    # the gate check and this point. Restore clones mutable
                    # state containers so decode cannot mutate the saved LRU
                    # snapshot by reference.
                    self._restore_prompt_cache(bc, hit_snapshot)
                    # Bump LRU position. Safe to mutate here because the
                    # worker is serialized under ``_generation_lock``.
                    if system_hash in self._system_kv_cache:
                        self._system_kv_cache.move_to_end(system_hash)
                    self._system_kv_cache_stats["hits"] += 1
                else:
                    bc = make_prompt_cache(model)
                    sys_arr = mx.array(system_tokens)
                    step = self._prefill_step_size
                    while sys_arr.size > step:
                        model(sys_arr[:step][None], cache=bc)
                        self._eval_cache_snapshot([c.state for c in bc])
                        sys_arr = sys_arr[step:]
                        mx.clear_cache()
                    if sys_arr.size > 0:
                        model(sys_arr[None], cache=bc)
                        self._eval_cache_snapshot([c.state for c in bc])

                    # Free intermediate prefill activations before snapshotting.
                    # Intentionally stricter than the MLLM path, which does not
                    # ``mx.clear_cache()`` between its last prefill chunk and
                    # the snapshot; here we want the snapshot to reflect only
                    # the KV state, not residual activations from prefill.
                    mx.clear_cache()

                    snapshot = self._snapshot_prompt_cache(bc)
                    self._eval_cache_snapshot(snapshot)
                    self._system_kv_cache[system_hash] = (snapshot, system_token_count)
                    self._system_kv_cache.move_to_end(system_hash)
                    evicted_count = 0
                    while len(self._system_kv_cache) > self._system_kv_capacity:
                        evicted_hash, _ = self._system_kv_cache.popitem(last=False)
                        self._system_kv_cache_stats["evictions"] += 1
                        evicted_count += 1
                        logger.info(
                            "System KV cache EVICTED (stream_chat): hash=%s "
                            "(capacity=%d)",
                            evicted_hash,
                            self._system_kv_capacity,
                        )
                    if evicted_count:
                        # Eviction dropped MLX array refs; reclaim Metal heap.
                        # Skip on the common non-eviction path to avoid
                        # flushing the Metal allocator's reuse pool.
                        mx.clear_cache()
                    self._system_kv_cache_stats["misses"] += 1
                    self._system_kv_cache_stats["stores"] += 1
                    try:
                        cache_mb = sum(c.nbytes for c in bc) / 1e6
                    except Exception:
                        cache_mb = -1
                    logger.info(
                        "System KV cache STORED (stream_chat): %d tokens " "(%.1f MB)",
                        system_token_count,
                        cache_mb,
                    )

                prompt_arr = mx.array(suffix_tokens)
                for resp in mlx_stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt_arr,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=bc,
                ):
                    if abort_event.is_set():
                        break
                    _emit_response(resp)

            async def _produce_responses() -> None:
                try:
                    await self._run_blocking_serialized(
                        _run_with_cache,
                        on_cancel=abort_event.set,
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    _emit_error(exc)
                else:
                    _emit_done()

            producer_task = asyncio.create_task(_produce_responses())

            accumulated_text = ""
            token_count = 0
            finished = False
            cache_path_failed_before_first_token = False
            try:
                while True:
                    kind, payload = await response_queue.get()
                    if kind == "done":
                        break
                    if kind == "error":
                        if token_count == 0:
                            logger.warning(
                                "Pure-LLM KV-cache path failed before first "
                                "token (%s); falling back to uncached "
                                "stream_generate",
                                payload,
                            )
                            cache_path_failed_before_first_token = True
                            break
                        # Already streamed partial output; can't cleanly
                        # restart on the uncached path, so surface the error.
                        raise payload
                    resp = payload
                    token_count += 1
                    new_text = resp.text if hasattr(resp, "text") else str(resp)
                    accumulated_text += new_text
                    finish_reason = getattr(resp, "finish_reason", None)
                    finished = finish_reason is not None or token_count >= max_tokens
                    if finish_reason is None and finished:
                        finish_reason = "stop"

                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=new_text,
                        prompt_tokens=full_token_count,
                        completion_tokens=token_count,
                        finished=finished,
                        finish_reason=finish_reason,
                        mtp_bypass_reason=mtp_bypass_reason,
                    )
                    if finished:
                        break
            finally:
                if not producer_task.done():
                    abort_event.set()
                    try:
                        await producer_task
                    except BaseException:
                        pass

            if cache_path_failed_before_first_token:
                # Internal fallback to the public stream_generate. The
                # ``_in_tracker`` context flag prevents double counting
                # in _track_request_stream.
                native_forward_kwargs: dict[str, Any] = {}
                if native_mtp_config is not None:
                    native_forward_kwargs["_native_mtp_request_config"] = (
                        native_mtp_config
                    )
                elif native_mtp_disabled:
                    native_forward_kwargs["_native_mtp_disabled"] = True
                elif mtp_bypass_reason is not None:
                    native_forward_kwargs["_native_mtp_bypass_reason"] = (
                        mtp_bypass_reason
                    )
                async for output in self.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **native_forward_kwargs,
                    **kwargs,
                ):
                    yield output
            return

        # Fallback: no system prefix detected -> original uncached path.
        # Re-entrancy guard in _track_request_stream keeps stats single-counted.
        native_forward_kwargs = {}
        if native_mtp_config is not None:
            native_forward_kwargs["_native_mtp_request_config"] = native_mtp_config
        elif native_mtp_disabled:
            native_forward_kwargs["_native_mtp_disabled"] = True
        elif mtp_bypass_reason is not None:
            native_forward_kwargs["_native_mtp_bypass_reason"] = mtp_bypass_reason
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **native_forward_kwargs,
            **kwargs,
        ):
            yield output

    async def _stream_generate_specprefill(
        self,
        prompt: str,
        tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        telemetry: _SpecPrefillTelemetry | None = None,
        native_mtp_request: NativeMTPRequestConfig | None = None,
        mtp_bypass_reason: str | None = None,
        specprefill_keep_pct: float | None = None,
        specprefill_backbone_pct: float | None = None,
        specprefill_chunk_size: int | None = None,
        specprefill_halo_chunks: int | None = None,
        specprefill_anchor_chunks: int | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Sparse-only path using request-local target position transport.

        Scores token importance with the draft model, sparse-prefills the target
        model, then generates autoregressively. Falls back to normal generation
        on any error.
        """
        from threading import Event

        model = self._model.model
        tokenizer = self._model.tokenizer
        n_tokens = len(tokens)
        if telemetry is None:
            telemetry = self._resolve_specprefill_telemetry(
                legacy=True,
                policy="sparse",
                coverage="selective",
                has_media=False,
                total_tokens=n_tokens,
            )
        cancel_requested = Event()

        def _request_cancel() -> None:
            cancel_requested.set()

        def _cancel_check() -> None:
            if cancel_requested.is_set():
                raise _SpecPrefillCancelled()

        loop = asyncio.get_running_loop()
        response_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        sparse_output_committed = False
        sparse_decode_transaction_started = False

        def _emit_response(resp: Any) -> None:
            if not cancel_requested.is_set():
                loop.call_soon_threadsafe(response_queue.put_nowait, ("resp", resp))

        def _emit_sparse_response(resp: Any) -> None:
            nonlocal sparse_output_committed
            # This boundary is deliberately before queuing the response: once a
            # sparse token exists, retrying dense would create a second answer.
            sparse_output_committed = True
            _emit_response(resp)

        def _emit_done() -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("done", None))

        def _emit_error(exc: BaseException) -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("error", exc))

        def _run_all():
            try:
                _run_specprefill()
            except _SpecPrefillCancelled:
                raise
            except Exception as e:
                if (
                    sparse_output_committed
                    or sparse_decode_transaction_started
                    or isinstance(e, _SpecPrefillAuthorityError)
                    or isinstance(e, SparseTargetPrefillAuthorityError)
                ):
                    raise
                logger.error("SpecPrefill failed, falling back to normal path: %s", e)
                telemetry.fallback("sparse_execution_failed")
                _run_normal()

        def _run_specprefill():
            """Score tokens, sparse prefill, generate autoregressively."""
            nonlocal sparse_decode_transaction_started
            import time
            from types import SimpleNamespace

            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

            from ..specprefill import score_tokens

            if not self._supports_sparse_continuation(self._model.stream_generate):
                raise TargetPositionError(
                    "mlx-lm does not expose model_forward_context continuation"
                )

            combined_mtp = native_mtp_request is not None
            if combined_mtp:
                profile_key, adapter = self._admit_sparse_target(
                    model, combined_mtp=True
                )
            else:
                profile_key, adapter = self._admit_sparse_target(model)
            detokenizer = _new_sparse_detokenizer(tokenizer)
            top_k = kwargs.get("top_k", 0)
            min_p = kwargs.get("min_p", 0.0)
            presence_penalty = kwargs.get("presence_penalty", 0.0)
            repetition_penalty = kwargs.get("repetition_penalty", 1.0)
            sampler = make_sampler(
                temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p
            )
            penalty_processors = make_logits_processors(
                repetition_penalty=(
                    repetition_penalty if repetition_penalty != 1.0 else None
                ),
                presence_penalty=(
                    presence_penalty if presence_penalty != 0.0 else None
                ),
            )
            all_processors = (kwargs.get("logits_processors") or []) + (
                penalty_processors or []
            )
            seeded_processors = _seed_logits_processors(
                mx.array(tokens, dtype=mx.uint32), all_processors
            )

            cache = (
                model.make_cache()
                if combined_mtp
                else make_prompt_cache(model, max_kv_size=self._max_kv_size or None)
            )
            if combined_mtp and (not isinstance(cache, list) or not cache):
                raise TargetPositionError(
                    "sparse native MTP target cache must be a fresh non-empty list"
                )

            # Phase 1: Score with draft model
            t0 = time.monotonic()
            importance = score_tokens(
                self._draft_model,
                tokens,
                prefill_step_size=self._prefill_step_size,
                cancel_check=_cancel_check,
            )
            t_score = time.monotonic() - t0
            telemetry.scorer_ms = t_score * 1000

            # Phase 2/3: semantic selection and request-local target prefill.
            _cancel_check()
            effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
            effective_backbone = (
                specprefill_backbone_pct
                if specprefill_backbone_pct is not None
                else self._specprefill_backbone_pct
            )
            effective_chunk_size = specprefill_chunk_size or 32
            effective_halo_chunks = (
                specprefill_halo_chunks if specprefill_halo_chunks is not None else 1
            )
            effective_anchor_chunks = (
                specprefill_anchor_chunks
                if specprefill_anchor_chunks is not None
                else 1
            )
            prepare_kwargs = dict(
                target_model=model,
                tokenizer=tokenizer,
                tokens=tokens,
                importance=importance,
                cache=cache,
                telemetry=telemetry,
                keep_pct=effective_keep,
                backbone_pct=effective_backbone,
                chunk_size=effective_chunk_size,
                halo_chunks=effective_halo_chunks,
                anchor_chunks=effective_anchor_chunks,
                profile_key=profile_key,
                adapter=adapter,
                cancel_check=_cancel_check,
            )
            if combined_mtp:
                prepare_kwargs["combined_mtp"] = True
            prepared = self._prepare_sparse_target_prefill(**prepare_kwargs)
            if combined_mtp:
                result, forward_context, plan, sparse_bootstrap = prepared
            else:
                result, forward_context, plan = prepared
                sparse_bootstrap = None
            logits = result.logits
            n_selected = len(plan.selected_indices)
            t_prefill = (telemetry.target_prefill_ms or 0.0) / 1000.0

            logger.info(
                "SpecPrefill: scored %d tokens in %.1fs, sparse prefill %d/%d "
                "(keep=%.0f%%) in %.1fs",
                n_tokens,
                t_score,
                n_selected,
                n_tokens,
                n_selected / n_tokens * 100,
                t_prefill,
            )

            if combined_mtp:
                generation = None
                authority_abandoned = False

                def _abandon_before_first_output() -> bool:
                    nonlocal authority_abandoned, sparse_decode_transaction_started
                    if authority_abandoned:
                        return True
                    try:
                        authority_abandoned = _try_abandon_sparse_bootstrap(
                            sparse_bootstrap
                        )
                    except BaseException:
                        sparse_decode_transaction_started = True
                        raise
                    if not authority_abandoned:
                        sparse_decode_transaction_started = True
                    return authority_abandoned

                try:
                    # Cancellation before the first resume owns no sample and
                    # must atomically abandon still-unclaimed authority.
                    _cancel_check()
                    generation = self._model.stream_generate(
                        prompt=None,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=None,
                        native_mtp_request=native_mtp_request,
                        sparse_bootstrap=sparse_bootstrap,
                        model_forward_context=forward_context,
                        # Upstream history starts with the final selected prompt
                        # token, so prefix all earlier original prompt tokens.
                        logits_processor_seed_tokens=mx.array(
                            tokens[:-1], dtype=mx.uint32
                        ),
                        **kwargs,
                    )
                    iterator = iter(generation)
                    while True:
                        try:
                            chunk = next(iterator)
                        except StopIteration:
                            break
                        except BaseException:
                            _abandon_before_first_output()
                            raise
                        # A yielded chunk proves sampling occurred and the
                        # bootstrap claim was consumed. Replay is now terminal.
                        sparse_decode_transaction_started = True
                        _cancel_check()
                        _emit_sparse_response(chunk)
                except BaseException:
                    if not sparse_decode_transaction_started:
                        _abandon_before_first_output()
                    raise
                finally:
                    try:
                        if generation is not None:
                            close = getattr(generation, "close", None)
                            if close is not None:
                                close()
                    finally:
                        try:
                            if not sparse_decode_transaction_started:
                                _abandon_before_first_output()
                        finally:
                            forward_context.finish()
                return

            # Sample the seed under the complete prompt history, then retain
            # request-local sparse positions for every continuation forward.
            _cancel_check()
            try:
                # Scoring, selection, and fresh-cache target prefill are an
                # isolated pre-output transaction: failures above this line
                # may discard the request-local cache and restart dense.  RNG
                # and logits-processor state become observable as soon as the
                # first sample begins, so replay is forbidden from here on.
                sparse_decode_transaction_started = True
                first_token, _ = _sample_with_processors(
                    None, logits[:, -1, :].squeeze(0), sampler, seeded_processors
                )
                mx.eval(first_token)
                first_token_id = first_token.item()
                eos_ids = self._eos_token_ids(tokenizer)
                first_text, first_is_eos = _detokenize_sparse_token(
                    detokenizer,
                    first_token_id,
                    eos_ids,
                    terminal=max_tokens == 1,
                )
            except BaseException:
                forward_context.finish()
                raise
            _emit_sparse_response(
                SimpleNamespace(
                    text=first_text,
                    finish_reason="stop" if first_is_eos else None,
                )
            )
            if not first_is_eos and max_tokens > 1:
                continuation_kwargs = dict(kwargs)
                continuation_kwargs["logits_processors"] = seeded_processors
                continuation_kwargs["presence_penalty"] = 0.0
                continuation_kwargs["repetition_penalty"] = 1.0
                continuation_kwargs["model_forward_context"] = forward_context
                try:
                    for chunk in self._model.stream_generate(
                        prompt=mx.array([first_token_id]),
                        max_tokens=max_tokens - 1,
                        temperature=temperature,
                        top_p=top_p,
                        stop=None,
                        prompt_cache=cache,
                        **continuation_kwargs,
                    ):
                        _cancel_check()
                        token_id = getattr(chunk, "token", None)
                        if not isinstance(token_id, int) or isinstance(token_id, bool):
                            raise TargetPositionError(
                                "sparse continuation must expose integer token IDs"
                            )
                        terminal = bool(getattr(chunk, "finished", False))
                        text, is_eos = _detokenize_sparse_token(
                            detokenizer,
                            token_id,
                            eos_ids,
                            terminal=terminal,
                        )
                        _emit_sparse_response(
                            SimpleNamespace(
                                text=text,
                                finish_reason=(
                                    "stop"
                                    if is_eos
                                    else getattr(chunk, "finish_reason", None)
                                ),
                            )
                        )
                finally:
                    forward_context.finish()
            else:
                forward_context.finish()

        def _run_normal():
            """Fallback: normal generation without specprefill."""
            normal_kwargs = dict(kwargs)
            if native_mtp_request is not None:
                normal_kwargs["native_mtp_request"] = native_mtp_request
            for chunk in self._model.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **normal_kwargs,
            ):
                _cancel_check()
                _emit_response(chunk)

        async def _produce_responses() -> None:
            try:
                await self._run_blocking_serialized(
                    _run_all,
                    on_cancel=_request_cancel,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                _emit_error(exc)
            else:
                _emit_done()

        producer_task = asyncio.create_task(_produce_responses())

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        producer_exhausted = False
        mtp_drafts = 0
        mtp_accepted = 0
        backend_mtp_bypass_reason = None
        last_logprobs = None
        try:
            while True:
                kind, payload = await response_queue.get()
                if kind == "done":
                    producer_exhausted = True
                    break
                if kind == "error":
                    if finished and isinstance(payload, _SpecPrefillCancelled):
                        break
                    raise payload
                if finished:
                    # The terminal token was already yielded. Drain until the
                    # producer's done/error sentinel so a late worker failure
                    # cannot be hidden by cancelling the producer on EOS.
                    continue
                resp = payload

                token_count += 1
                mtp_drafts = getattr(resp, "mtp_drafts", mtp_drafts)
                mtp_accepted = getattr(resp, "mtp_accepted", mtp_accepted)
                backend_mtp_bypass_reason = getattr(
                    resp, "mtp_bypass_reason", backend_mtp_bypass_reason
                )
                last_logprobs = getattr(resp, "logprobs", last_logprobs)
                new_text = resp.text if hasattr(resp, "text") else str(resp)
                accumulated_text += new_text

                stop_hit = bool(stop) and any(
                    stop_seq in accumulated_text for stop_seq in stop
                )
                finished = stop_hit or token_count >= max_tokens
                finish_reason = getattr(resp, "finish_reason", None)
                if stop_hit:
                    finish_reason = "stop"
                elif finish_reason is None and finished:
                    finish_reason = "length"
                elif finish_reason is not None:
                    finished = True
                if finished:
                    _request_cancel()

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=n_tokens,
                    completion_tokens=token_count,
                    finished=finished,
                    finish_reason=finish_reason,
                    mtp_drafts=mtp_drafts,
                    mtp_accepted=mtp_accepted,
                    mtp_bypass_reason=(mtp_bypass_reason or backend_mtp_bypass_reason),
                    logprobs=last_logprobs,
                    **telemetry.as_output_kwargs(),
                )

        finally:
            if not producer_task.done():
                _request_cancel()
            await producer_task

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=n_tokens,
                completion_tokens=token_count,
                finished=True,
                # A clean sparse producer end is an ordinary natural stop,
                # even when it did not emit a backend terminal reason.  Keep
                # this separate empty terminal frame so the already-observed
                # token frames and their telemetry remain unchanged.
                finish_reason="stop" if producer_exhausted else "length",
                mtp_drafts=mtp_drafts,
                mtp_accepted=mtp_accepted,
                mtp_bypass_reason=(mtp_bypass_reason or backend_mtp_bypass_reason),
                logprobs=last_logprobs,
                **telemetry.as_output_kwargs(),
            )

    async def _stream_generate_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list | None = None,
        combined_mtp: bool = False,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Text-only generation via mlx_lm TextModel.

        Used when text-only MLLM routing is active and the request has no media.
        Runs the full generation in a single thread to maintain Metal safety.

        System prompt KV caching: on the first request, prefills system tokens
        and snapshots backbone KV state. Subsequent requests with the same
        system prompt restore the snapshot and only prefill the suffix tokens.
        """
        import hashlib
        import os

        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.models import cache as cache_module
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        # The text route has its own prompt rendering, but shares exactly the
        # same policy resolver as direct SimpleEngine generation.
        specprefill_controls = self._specprefill_controls(kwargs)
        native_mtp_config, _, combined_mtp, mtp_bypass_reason = (
            _consume_native_mtp_request(kwargs, server_default=combined_mtp)
        )
        specprefill_keep_pct = (
            specprefill_controls["keep_pct"]
            if self._specprefill_diagnostic_mode
            else None
        )
        specprefill_backbone_pct = (
            specprefill_controls["backbone_pct"]
            if self._specprefill_diagnostic_mode
            else None
        )
        specprefill_chunk_size: int | None = None
        specprefill_halo_chunks: int | None = None
        specprefill_anchor_chunks: int | None = None
        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        presence_penalty = kwargs.pop("presence_penalty", 0.0)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        stop = kwargs.pop("stop", None)
        external_logits_processors = kwargs.pop("logits_processors", None)
        abort_event = threading.Event()

        if native_mtp_config is not None:
            sampling = native_mtp_config.sampling
            temperature = sampling.temperature
            top_p = sampling.top_p
            top_k = sampling.top_k
            min_p = sampling.min_p
            presence_penalty = sampling.presence_penalty
            repetition_penalty = sampling.repetition_penalty
        # Per-request enable_thinking override; fall back to env var / default True.
        enable_thinking = kwargs.pop("enable_thinking", None)
        if enable_thinking is None:
            enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
            enable_thinking = enable_thinking_env.lower() in ("true", "1", "yes")

        # Apply chat template for full prompt
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        template_kwargs.update(chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools
        safe_messages = normalize_messages_for_chat_template(messages)

        try:
            full_prompt = self._text_tokenizer.apply_chat_template(
                safe_messages, **template_kwargs
            )
        except TypeError:
            # Template doesn't accept tools= or enable_thinking=
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            full_prompt = self._text_tokenizer.apply_chat_template(
                safe_messages, **template_kwargs
            )

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        penalty_processors = make_logits_processors(
            repetition_penalty=(
                repetition_penalty if repetition_penalty != 1.0 else None
            ),
            presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
        )
        if native_mtp_config is not None:
            for processor in penalty_processors or ():
                try:
                    processor.native_mtp_replay_safe = True
                except (AttributeError, TypeError) as exc:
                    raise ValueError(
                        "native MTP penalty processor cannot be replayed safely"
                    ) from exc
        all_processors = (external_logits_processors or []) + (penalty_processors or [])
        custom_logits_active = bool(external_logits_processors)
        combined_mtp = (
            _request_can_compose_mtp(combined_mtp, external_logits_processors)
            and native_mtp_config is not None
        )
        sparse_no_completion_requested = max_tokens == 0
        max_tokens = max_tokens or 4096

        # --- System prompt KV caching ---
        backbone_cache = None  # Backbone-only cache (no MTP), used by both paths
        prompt_to_send = full_prompt  # Default: send full prompt text
        cache_hit = False
        system_token_count = 0
        full_token_count = 0
        system_hash = None
        system_tokens = None
        suffix_tokens = None
        full_tokens_list = None
        cache_blocking_controls = []
        if not self._supports_system_kv_cache:
            cache_blocking_controls.append("non_kv_cache_class")
        if cache_blocking_controls:
            logger.info(
                "System KV cache SKIP (text route): request or engine has "
                "controls/features the cache branch cannot honor (%s); using "
                "uncached path",
                cache_blocking_controls,
            )

        # Extract system messages for caching
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system and self._text_model is not None and not cache_blocking_controls:
            # Find system prefix boundary in full prompt text.
            # ChatML format: system section ends where first non-system message begins.
            # Works with tools (rendered inside system section by Qwen templates).
            system_prefix_end = -1
            for marker in ("<|im_start|>user\n", "<|im_start|>assistant\n"):
                idx = full_prompt.find(marker)
                if idx > 0:
                    system_prefix_end = idx
                    break

            if system_prefix_end > 0:
                system_prefix_text = full_prompt[:system_prefix_end]
                system_hash = hashlib.sha256(system_prefix_text.encode()).hexdigest()[
                    :16
                ]

                # Tokenize both (matching stream_generate's tokenization logic)
                tokenizer = self._text_tokenizer
                add_special = self._prompt_add_special_tokens(tokenizer, full_prompt)
                full_tokens_list = tokenizer.encode(
                    full_prompt, add_special_tokens=add_special
                )
                full_token_count = len(full_tokens_list)

                system_tokens_list = tokenizer.encode(
                    system_prefix_text, add_special_tokens=add_special
                )
                system_token_count = len(system_tokens_list)

                # Verify system tokens are a proper prefix of full tokens
                prefix_valid = (
                    len(full_tokens_list) > system_token_count
                    and full_tokens_list[:system_token_count] == system_tokens_list
                )

                if prefix_valid:
                    system_tokens = system_tokens_list
                    suffix_tokens = full_tokens_list[system_token_count:]

                    hit_candidate = self._system_kv_cache.get(system_hash)
                    if (
                        hit_candidate is not None
                        and system_token_count == hit_candidate[1]
                    ):
                        # Cache HIT — restore KV state into fresh backbone cache
                        def make_cache_with_snapshot(
                            text_model,
                            system_kv_snapshot,
                            _max_kv_size=self._max_kv_size,
                        ):
                            import mlx.core as mx
                            from mlx_lm.models.cache import make_prompt_cache

                            backbone_cache = make_prompt_cache(
                                text_model, max_kv_size=_max_kv_size or None
                            )
                            SimpleEngine._restore_prompt_cache(
                                backbone_cache,
                                system_kv_snapshot,
                            )

                            prompt_to_send = mx.array(suffix_tokens)
                            return backbone_cache, prompt_to_send

                        backbone_cache, prompt_to_send = (
                            await self._run_blocking_serialized(
                                make_cache_with_snapshot,
                                self._text_model,
                                hit_candidate[0],
                            )
                        )
                        # Bump LRU position now that we know we'll use it.
                        if system_hash in self._system_kv_cache:
                            self._system_kv_cache.move_to_end(system_hash)
                        self._system_kv_cache_stats["hits"] += 1
                        cache_hit = True

                        logger.info(
                            "System KV cache HIT: reusing %d cached tokens, "
                            "prefilling %d new tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                    else:
                        # Cache MISS — will prefill system tokens and snapshot
                        logger.info(
                            "System KV cache MISS: will prefill %d system tokens, "
                            "%d suffix tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                else:
                    logger.debug(
                        "System KV cache: prefix token validation failed, "
                        "using full prompt (%d tokens)",
                        len(full_tokens_list),
                    )
                    system_token_count = 0

        # Do not tokenize a normal dense text-model request merely to compute
        # optional feature telemetry: that breaks the no-extra-work contract of
        # the unsafe system-cache path. Only sparse-capable intent needs prompt
        # tokenization before the standard mlx-lm prefill.
        telemetry = self._resolve_specprefill_telemetry(
            legacy=specprefill_controls["legacy"],
            policy=specprefill_controls["policy"],
            coverage=specprefill_controls["coverage"],
            has_media=specprefill_controls["has_media"],
            total_tokens=(full_token_count if full_tokens_list is not None else None),
            combined_mtp=combined_mtp,
        )
        requested_policy = telemetry.decision.requested_policy
        should_measure_sparse_prompt = not specprefill_controls["has_media"] and (
            (
                requested_policy is SpecPrefillPolicy.AUTO
                and telemetry.decision.coverage.value == "selective"
            )
            or (
                requested_policy is SpecPrefillPolicy.SPARSE
                and self._specprefill_diagnostic_mode
            )
        )
        if should_measure_sparse_prompt and full_tokens_list is None:
            full_tokens_list = self._encode_prompt_tokens(
                self._text_tokenizer, full_prompt
            )
            full_token_count = len(full_tokens_list)
            telemetry = self._resolve_specprefill_telemetry(
                legacy=specprefill_controls["legacy"],
                policy=specprefill_controls["policy"],
                coverage=specprefill_controls["coverage"],
                has_media=False,
                total_tokens=full_token_count,
                combined_mtp=combined_mtp,
            )

        if (
            not self._specprefill_diagnostic_mode
            and telemetry.profile_tuning is not None
        ):
            specprefill_keep_pct = telemetry.profile_tuning.keep_pct
            specprefill_backbone_pct = telemetry.profile_tuning.backbone_pct
            specprefill_chunk_size = telemetry.profile_tuning.chunk_size
            specprefill_halo_chunks = telemetry.profile_tuning.halo_chunks
            specprefill_anchor_chunks = telemetry.profile_tuning.anchor_chunks

        # Tokens for specprefill: suffix (if system KV) or full prompt
        specprefill_tokens = (
            suffix_tokens if suffix_tokens is not None else full_tokens_list
        )
        specprefill_offset = system_token_count if suffix_tokens is not None else 0

        use_specprefill = (
            telemetry.decision.effective_policy is SpecPrefillPolicy.SPARSE
        )
        # Sparse target execution starts with an empty target cache.  A saved
        # system prefix has a different physical/logical topology, and MTP
        # cache pairing is intentionally qualified separately.
        if use_specprefill and sparse_no_completion_requested:
            telemetry.fallback("sparse_no_completion_requested")
            use_specprefill = False
        elif use_specprefill and (backbone_cache is not None or system_token_count):
            telemetry.fallback("sparse_prefix_cache_not_supported")
            use_specprefill = False
        elif use_specprefill and not self._supports_sparse_continuation(
            mlx_stream_generate
        ):
            telemetry.fallback("sparse_forward_context_unavailable")
            use_specprefill = False
        elif use_specprefill and _processors_can_retire(all_processors):
            telemetry.fallback("sparse_processor_retirement_unsupported")
            use_specprefill = False

        loop = asyncio.get_running_loop()
        response_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        sparse_output_committed = False
        sparse_decode_transaction_started = False

        def _emit_response(resp: Any) -> None:
            if abort_event.is_set():
                return
            loop.call_soon_threadsafe(response_queue.put_nowait, ("resp", resp))

        def _emit_sparse_response(resp: Any) -> None:
            nonlocal sparse_output_committed
            # A sparse token makes dense restart unsafe: callers would observe
            # two answers from one request. Commit before queueing to preserve
            # the boundary even if the consumer is cancelled immediately.
            sparse_output_committed = True
            _emit_response(resp)

        def _emit_done() -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("done", None))

        def _emit_error(exc: BaseException) -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("error", exc))

        def _seed_from_last_response(prompt_cache, last_resp):
            last_tok = getattr(last_resp, "token", None)
            if last_tok is not None:
                cache_module.trim_prompt_cache(prompt_cache, 1)
                return mx.array([last_tok], dtype=mx.uint32)
            return mx.array(
                self._text_tokenizer.encode(getattr(last_resp, "text", "")),
                dtype=mx.uint32,
            )

        def _resume_after_processor_retirement(
            model,
            prompt_cache,
            prompt,
            remaining_tokens: int,
            emit_response=_emit_response,
        ) -> None:
            resume_kwargs = dict(
                max_tokens=remaining_tokens,
                prefill_step_size=self._prefill_step_size,
                prompt_cache=prompt_cache,
            )
            if native_mtp_config is not None:
                # Resume speculative decode from the retained backbone cache with
                # a fresh MTP cache so stale speculative state cannot survive the
                # processor-to-content handoff.
                resume_kwargs.update(native_mtp_config.mlx_lm_call_kwargs())
            else:
                resume_kwargs["sampler"] = sampler
            for resp in mlx_stream_generate(
                model,
                self._text_tokenizer,
                prompt=prompt,
                **resume_kwargs,
            ):
                if abort_event.is_set():
                    logger.info("Text route: abort requested; stopping resume decode")
                    break
                emit_response(resp)

        # Run all Metal ops in a single serialized thread.
        def _run_all():
            nonlocal backbone_cache, prompt_to_send, use_specprefill

            model = self._text_model
            can_retire_processors = _processors_can_retire(all_processors)
            use_mtp = combined_mtp and not custom_logits_active
            if self._mtp and custom_logits_active:
                logger.info(
                    "Text route: disabling MTP for request-local logits processors"
                )
            # Cache MISS with valid prefix: prefill system tokens and snapshot
            if (
                not cache_hit
                and system_token_count > 0
                and system_tokens is not None
                and suffix_tokens is not None
            ):
                mc = make_prompt_cache(model, max_kv_size=self._max_kv_size or None)
                sys_arr = mx.array(system_tokens)

                # Prefill system tokens in chunks (matching generate_step)
                step = self._prefill_step_size
                while sys_arr.size > step:
                    model(sys_arr[:step][None], cache=mc)
                    self._eval_cache_snapshot([c.state for c in mc])
                    sys_arr = sys_arr[step:]
                    mx.clear_cache()
                if sys_arr.size > 0:
                    model(sys_arr[None], cache=mc)
                    self._eval_cache_snapshot([c.state for c in mc])

                # Snapshot backbone cache. Cache arrays are treated as immutable;
                # mutable state containers are copied so hybrid ArraysCache
                # entries cannot alias the saved system-prefix state.
                snapshot = self._snapshot_prompt_cache(mc)
                self._eval_cache_snapshot(snapshot)

                self._system_kv_cache[system_hash] = (snapshot, system_token_count)
                self._system_kv_cache.move_to_end(system_hash)
                evicted_count = 0
                while len(self._system_kv_cache) > self._system_kv_capacity:
                    evicted_hash, _ = self._system_kv_cache.popitem(last=False)
                    self._system_kv_cache_stats["evictions"] += 1
                    evicted_count += 1
                    logger.info(
                        "System KV cache EVICTED: hash=%s (capacity=%d)",
                        evicted_hash,
                        self._system_kv_capacity,
                    )
                if evicted_count:
                    # Eviction dropped MLX array refs; reclaim Metal heap.
                    # Skip on the common non-eviction path to avoid flushing
                    # the Metal allocator's reuse pool.
                    mx.clear_cache()
                self._system_kv_cache_stats["misses"] += 1
                self._system_kv_cache_stats["stores"] += 1

                backbone_cache = mc
                prompt_to_send = mx.array(suffix_tokens)
                logger.info(
                    "System KV cache: stored %d-token snapshot (%.1f MB), "
                    "prefilling %d remaining",
                    system_token_count,
                    sum(c.nbytes for c in mc) / 1e6,
                    len(suffix_tokens),
                )

            # --- SpecPrefill path (with fallback to normal on failure) ---
            if use_specprefill:
                try:
                    _run_specprefill(model, backbone_cache, use_mtp)
                    return
                except Exception as e:
                    if (
                        sparse_output_committed
                        or sparse_decode_transaction_started
                        or abort_event.is_set()
                        or isinstance(e, _SpecPrefillAuthorityError)
                        or isinstance(e, SparseTargetPrefillAuthorityError)
                    ):
                        raise
                    logger.error(
                        "SpecPrefill failed, falling back to normal MTP path: %s",
                        e,
                    )
                    telemetry.fallback("sparse_execution_failed")
                    # Discard potentially corrupted cache
                    backbone_cache = None
                    prompt_to_send = full_prompt

            # --- Normal path (mlx_lm stream_generate) ---
            prompt_cache = None
            if backbone_cache is not None:
                # Add MTP cache on top of backbone
                if use_mtp and hasattr(model, "make_mtp_cache"):
                    mtp_cache = model.make_mtp_cache()
                    prompt_cache = backbone_cache + mtp_cache
                else:
                    prompt_cache = backbone_cache

            gen_kwargs = dict(
                max_tokens=max_tokens,
                prefill_step_size=self._prefill_step_size,
            )
            if not use_mtp:
                # Native MTP samples from its immutable request config.  The
                # ordinary sampler is opaque to transactional replay and must
                # remain absent from the selected native call.
                gen_kwargs["sampler"] = sampler
            if all_processors:
                gen_kwargs["logits_processors"] = all_processors
            if use_mtp:
                if native_mtp_config is None:
                    raise RuntimeError(
                        "native MTP selection requires a request-local config"
                    )
                gen_kwargs.update(native_mtp_config.mlx_lm_call_kwargs())
            if prompt_cache is not None:
                gen_kwargs["prompt_cache"] = prompt_cache
            if can_retire_processors and not use_mtp:
                shared_cache = prompt_cache
                if shared_cache is None:
                    shared_cache = make_prompt_cache(
                        model, max_kv_size=self._max_kv_size or None
                    )
                gen_kwargs["prompt_cache"] = shared_cache

                token_count = 0
                last_resp = None
                retired = False
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=prompt_to_send,
                    **gen_kwargs,
                ):
                    if abort_event.is_set():
                        logger.info(
                            "Text route: abort requested; stopping decode after %d tokens",
                            token_count,
                        )
                        break
                    _emit_response(resp)
                    token_count += 1
                    last_resp = resp
                    retired = _processors_retired(all_processors)
                    if retired:
                        logger.info(
                            "Text route: request-local processor retired after %d tokens; "
                            "resuming content phase with MTP=%s",
                            token_count,
                            native_mtp_config is not None,
                        )
                        break

                if retired and token_count < max_tokens and last_resp is not None:
                    seed = _seed_from_last_response(shared_cache, last_resp)
                    _resume_after_processor_retirement(
                        model,
                        shared_cache,
                        seed,
                        max_tokens - token_count,
                    )
            else:
                generation = mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=prompt_to_send,
                    **gen_kwargs,
                )
                primary_error: BaseException | None = None
                try:
                    for resp in generation:
                        if abort_event.is_set():
                            logger.info("Text route: abort requested; stopping decode")
                            break
                        _emit_response(resp)
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    close = getattr(generation, "close", None)
                    if close is not None:
                        try:
                            close()
                        except BaseException:
                            if primary_error is None:
                                raise
                            logger.warning(
                                "Failed to close mlx-lm text stream after error",
                                exc_info=True,
                            )

        def _run_specprefill(model, bc, use_mtp):
            """Run sparse-only target prefill and retain its forward context."""
            nonlocal sparse_decode_transaction_started
            from types import SimpleNamespace

            from mlx_lm import stream_generate as mlx_stream_generate
            from mlx_lm.models.cache import make_prompt_cache

            from ..specprefill import score_tokens

            if bc is not None:
                raise TargetPositionError(
                    "sparse target prefill requires a fresh target cache"
                )
            if use_mtp and native_mtp_config is None:
                raise TargetPositionError(
                    "native MTP composition requires a request-local config"
                )
            if use_mtp:
                profile_key, adapter = self._admit_sparse_target(
                    model, combined_mtp=True
                )
            else:
                profile_key, adapter = self._admit_sparse_target(model)
            detokenizer = _new_sparse_detokenizer(self._text_tokenizer)
            seed_values = full_tokens_list
            if use_mtp and seed_values is not None:
                # Native sparse MTP supplies the final selected prompt token as
                # its initial history; prefix only the preceding original IDs.
                seed_values = seed_values[:-1]
            seed_tokens = (
                mx.array(seed_values, dtype=mx.uint32)
                if seed_values is not None
                else None
            )
            seeded_processors = _seed_logits_processors(seed_tokens, all_processors)
            cache = (
                model.make_cache()
                if use_mtp
                else make_prompt_cache(model, max_kv_size=self._max_kv_size or None)
            )
            if use_mtp and (not isinstance(cache, list) or not cache):
                raise TargetPositionError(
                    "sparse native MTP target cache must be a fresh non-empty list"
                )
            import time

            def _cancel_check() -> None:
                if abort_event.is_set():
                    raise _SpecPrefillCancelled()

            t0 = time.monotonic()
            importance = score_tokens(
                self._draft_model,
                specprefill_tokens,
                prefill_step_size=self._prefill_step_size,
                cancel_check=_cancel_check,
            )
            t_score = time.monotonic() - t0
            telemetry.scorer_ms = t_score * 1000
            effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
            effective_backbone = (
                specprefill_backbone_pct
                if specprefill_backbone_pct is not None
                else self._specprefill_backbone_pct
            )
            effective_chunk_size = specprefill_chunk_size or 32
            effective_halo_chunks = (
                specprefill_halo_chunks if specprefill_halo_chunks is not None else 1
            )
            effective_anchor_chunks = (
                specprefill_anchor_chunks
                if specprefill_anchor_chunks is not None
                else 1
            )
            prepare_kwargs = dict(
                target_model=model,
                tokenizer=self._text_tokenizer,
                tokens=specprefill_tokens,
                importance=importance,
                cache=cache,
                telemetry=telemetry,
                keep_pct=effective_keep,
                backbone_pct=effective_backbone,
                chunk_size=effective_chunk_size,
                halo_chunks=effective_halo_chunks,
                anchor_chunks=effective_anchor_chunks,
                profile_key=profile_key,
                adapter=adapter,
                cancel_check=_cancel_check,
            )
            if use_mtp:
                prepare_kwargs["combined_mtp"] = True
            prepared = self._prepare_sparse_target_prefill(**prepare_kwargs)
            if use_mtp:
                result, forward_context, plan, sparse_bootstrap = prepared
            else:
                result, forward_context, plan = prepared
                sparse_bootstrap = None
            logits = result.logits
            n_selected = len(plan.selected_indices)
            n_total = len(specprefill_tokens)
            t_prefill = (telemetry.target_prefill_ms or 0.0) / 1000.0

            logger.info(
                "SpecPrefill: scored %d tokens in %.1fs, sparse prefill %d/%d "
                "(keep=%.0f%%) in %.1fs",
                n_total,
                t_score,
                n_selected,
                n_total,
                n_selected / n_total * 100,
                t_prefill,
            )

            if use_mtp:
                generation = None
                authority_abandoned = False

                def _abandon_before_first_output() -> bool:
                    nonlocal authority_abandoned, sparse_decode_transaction_started
                    if authority_abandoned:
                        return True
                    try:
                        authority_abandoned = _try_abandon_sparse_bootstrap(
                            sparse_bootstrap
                        )
                    except BaseException:
                        sparse_decode_transaction_started = True
                        raise
                    if not authority_abandoned:
                        sparse_decode_transaction_started = True
                    return authority_abandoned

                try:
                    gen_kwargs = dict(
                        max_tokens=max_tokens,
                        prefill_step_size=self._prefill_step_size,
                        model_forward_context=forward_context,
                        sparse_bootstrap=sparse_bootstrap,
                    )
                    if seeded_processors:
                        gen_kwargs["logits_processors"] = seeded_processors
                    gen_kwargs.update(
                        native_mtp_config.mlx_lm_call_kwargs(
                            consumer=mlx_stream_generate
                        )
                    )
                    _cancel_check()
                    generation = mlx_stream_generate(
                        model,
                        self._text_tokenizer,
                        prompt=None,
                        **gen_kwargs,
                    )
                    iterator = iter(generation)
                    while True:
                        try:
                            resp = next(iterator)
                        except StopIteration:
                            break
                        except BaseException:
                            _abandon_before_first_output()
                            raise
                        sparse_decode_transaction_started = True
                        _cancel_check()
                        _emit_sparse_response(resp)
                except BaseException:
                    if not sparse_decode_transaction_started:
                        _abandon_before_first_output()
                    raise
                finally:
                    try:
                        if generation is not None:
                            close = getattr(generation, "close", None)
                            if close is not None:
                                close()
                    finally:
                        try:
                            if not sparse_decode_transaction_started:
                                _abandon_before_first_output()
                        finally:
                            forward_context.finish()
                return

            # Phase 4: Sample the first token from the prefilled logits, then
            # continue through mlx_lm with the request-local forward context.
            eos_ids = self._eos_token_ids(self._text_tokenizer)
            try:
                # Sampling is the exact no-replay boundary.  Everything above
                # uses request-local scorer/target state and can be discarded
                # for a dense retry; sampler/processor state cannot.
                sparse_decode_transaction_started = True
                y, _ = _sample_with_processors(
                    None, logits[:, -1, :].squeeze(0), sampler, seeded_processors
                )
                mx.eval(y)
                tok_id = y.item()
                new_text, is_eos = _detokenize_sparse_token(
                    detokenizer,
                    tok_id,
                    eos_ids,
                    terminal=max_tokens == 1,
                )
            except BaseException:
                forward_context.finish()
                raise
            _emit_sparse_response(
                SimpleNamespace(
                    text=new_text,
                    finish_reason=(
                        "stop" if is_eos else "length" if max_tokens <= 1 else None
                    ),
                )
            )

            if abort_event.is_set():
                logger.info("SpecPrefill text route: abort requested after seed token")
                forward_context.finish()
                return

            if is_eos or max_tokens <= 1:
                forward_context.finish()
                return

            continuation_prompt = mx.array([tok_id], dtype=mx.uint32)
            token_count = 1
            continuation_kwargs = dict(
                max_tokens=max_tokens - token_count,
                sampler=sampler,
                prefill_step_size=self._prefill_step_size,
                logits_processors=seeded_processors,
                prompt_cache=cache,
                model_forward_context=forward_context,
            )
            try:
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=continuation_prompt,
                    **continuation_kwargs,
                ):
                    if abort_event.is_set():
                        logger.info(
                            "SpecPrefill text route: abort requested; stopping decode"
                        )
                        break
                    response_token = getattr(resp, "token", None)
                    if not isinstance(response_token, int) or isinstance(
                        response_token, bool
                    ):
                        raise TargetPositionError(
                            "sparse continuation must expose integer token IDs"
                        )
                    terminal = getattr(resp, "finish_reason", None) is not None
                    text, response_is_eos = _detokenize_sparse_token(
                        detokenizer,
                        response_token,
                        eos_ids,
                        terminal=terminal,
                    )
                    _emit_sparse_response(
                        SimpleNamespace(
                            text=text,
                            finish_reason=(
                                "stop"
                                if response_is_eos
                                else getattr(resp, "finish_reason", None)
                            ),
                        )
                    )
            finally:
                forward_context.finish()

        async def _produce_responses() -> None:
            try:
                await self._run_blocking_serialized(
                    _run_all,
                    on_cancel=abort_event.set,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                _emit_error(exc)
            else:
                _emit_done()

        producer_task = asyncio.create_task(_produce_responses())

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        producer_exhausted = False
        mtp_drafts = 0
        mtp_accepted = 0
        backend_mtp_bypass_reason = None
        last_logprobs = None
        try:
            while True:
                kind, payload = await response_queue.get()
                if kind == "done":
                    producer_exhausted = True
                    break
                if kind == "error":
                    if finished and isinstance(payload, _SpecPrefillCancelled):
                        break
                    raise payload
                if finished:
                    # Drain producer completion after a terminal sparse token
                    # rather than cancelling it and masking a late error.
                    continue
                resp = payload

                token_count += 1
                # mlx-lm reports cumulative native counters on every response.
                # Assign rather than sum so Simple does not double-count them.
                mtp_drafts = getattr(resp, "mtp_drafts", mtp_drafts)
                mtp_accepted = getattr(resp, "mtp_accepted", mtp_accepted)
                backend_mtp_bypass_reason = getattr(
                    resp, "mtp_bypass_reason", backend_mtp_bypass_reason
                )
                last_logprobs = getattr(resp, "logprobs", None)
                new_text = resp.text if hasattr(resp, "text") else str(resp)
                accumulated_text += new_text

                stop_hit = False
                if stop:
                    stop_hit = any(stop_seq in accumulated_text for stop_seq in stop)
                finished = stop_hit or token_count >= max_tokens
                finish_reason = getattr(resp, "finish_reason", None)
                if stop_hit:
                    finish_reason = "stop"
                elif finish_reason is None and finished:
                    finish_reason = "length"
                elif finish_reason is not None:
                    finished = True
                if finished:
                    abort_event.set()

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=full_token_count or 0,
                    completion_tokens=token_count,
                    finished=finished,
                    finish_reason=finish_reason,
                    mtp_drafts=mtp_drafts,
                    mtp_accepted=mtp_accepted,
                    mtp_bypass_reason=(mtp_bypass_reason or backend_mtp_bypass_reason),
                    logprobs=last_logprobs,
                    **telemetry.as_output_kwargs(),
                )

        finally:
            if not producer_task.done():
                abort_event.set()
            await producer_task

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=full_token_count or 0,
                completion_tokens=token_count,
                finished=True,
                # A clean sparse producer end is an ordinary natural stop,
                # even when it did not emit a backend terminal reason.
                finish_reason="stop" if producer_exhausted else "length",
                mtp_drafts=mtp_drafts,
                mtp_accepted=mtp_accepted,
                mtp_bypass_reason=(mtp_bypass_reason or backend_mtp_bypass_reason),
                logprobs=last_logprobs,
                **telemetry.as_output_kwargs(),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        # Compute rolling generation_tps from recent completions.
        gen_tps = 0.0
        if self._recent_completions:
            total_tok = sum(c for c, _ in self._recent_completions)
            total_sec = sum(s for _, s in self._recent_completions)
            if total_sec > 0:
                gen_tps = total_tok / total_sec
        # Snapshot active requests with live elapsed_s refreshed at read time.
        now = time.time()
        requests_snapshot: list[dict[str, Any]] = []
        for entry in self._active_requests.values():
            snap = dict(entry)
            # entry stores last-known elapsed at last yield; refresh here so
            # the snapshot is meaningful even between yields.
            requests_snapshot.append(snap)
        stats = {
            "engine_type": "simple",
            "model_name": self._model_name,
            "uptime_seconds": now - self._created_at,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
            "running": self._loaded,
            "num_running": self._num_running,
            "num_waiting": self._generation_waiters,
            "num_requests_processed": self._total_requests_processed,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "batch_generator": {
                "generation_tps": gen_tps,
                "prompt_tps": 0.0,
            },
            "requests": requests_snapshot,
            "generation_lock": {
                "locked": self._generation_lock.locked(),
                "admission": self._generation_lock_admission,
                "busy_rejections": self._generation_busy_rejections,
            },
        }

        # MLLM prefix cache stats, remapped to the shape BatchedEngine emits
        # under "memory_aware_cache" so monitoring dashboards (which key off
        # current_memory_mb / max_memory_mb / memory_utilization /
        # entry_count) render cache utilization for SimpleEngine services.
        if self._is_mllm and self._model is not None:
            try:
                raw_cache = self._model.get_cache_stats()
            except Exception:
                raw_cache = None
            if raw_cache and raw_cache.get("enabled"):
                current_mb = float(raw_cache.get("memory_used_mb", 0) or 0)
                max_mb = float(raw_cache.get("max_memory_mb", 0) or 0)
                stats["memory_aware_cache"] = {
                    "hits": raw_cache.get("hits", 0),
                    "misses": raw_cache.get("misses", 0),
                    "hit_rate": raw_cache.get("hit_rate", 0.0),
                    "evictions": raw_cache.get("evictions", 0),
                    "tokens_saved": raw_cache.get("tokens_saved", 0),
                    "current_memory_mb": round(current_mb, 2),
                    "max_memory_mb": round(max_mb, 2),
                    "memory_utilization": (
                        round(current_mb / max_mb, 4) if max_mb > 0 else 0.0
                    ),
                    "entry_count": raw_cache.get(
                        "cache_entries", raw_cache.get("entries", 0)
                    ),
                }

        # SpecPrefill stats
        if self._draft_model is not None:
            stats["specprefill"] = {
                "enabled": True,
                "draft_model": self._specprefill_draft_model_path,
                "threshold": self._specprefill_threshold,
                "keep_pct": self._specprefill_keep_pct,
                "backbone_pct": self._specprefill_backbone_pct,
                "max_tokens": self._specprefill_max_tokens,
                "diagnostic_mode": self._specprefill_diagnostic_mode,
            }

        # System KV cache stats (LRU over multiple system prefixes)
        if self._system_kv_cache:
            slots = []
            total_bytes = 0
            for slot_hash, (snapshot, tokens) in self._system_kv_cache.items():
                slot_bytes = 0
                for entry in snapshot:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        slot_bytes += entry[0].nbytes + entry[1].nbytes
                    elif isinstance(entry, list):
                        slot_bytes += sum(a.nbytes for a in entry if a is not None)
                total_bytes += slot_bytes
                slots.append(
                    {
                        "hash": slot_hash,
                        "tokens": tokens,
                        "memory_mb": round(slot_bytes / 1e6, 1),
                    }
                )
            counters = dict(self._system_kv_cache_stats)
            denom = counters["hits"] + counters["misses"]
            counters["hit_ratio"] = (
                round(counters["hits"] / denom, 3) if denom > 0 else None
            )
            stats["system_kv_cache"] = {
                "capacity": self._system_kv_capacity,
                "in_use": len(self._system_kv_cache),
                "total_memory_mb": round(total_bytes / 1e6, 1),
                "slots": slots,
                "counters": counters,
            }

        # Include Metal memory stats
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics for the system-prompt KV LRU plus, when the
        model is multimodal, the MLLM's own cache stats.
        """
        result: dict[str, Any] = {}
        if self._supports_system_kv_cache:
            counters = dict(self._system_kv_cache_stats)
            denom = counters["hits"] + counters["misses"]
            counters["hit_ratio"] = (
                round(counters["hits"] / denom, 3) if denom > 0 else None
            )
            result["system_kv_cache"] = {
                "capacity": self._system_kv_capacity,
                "in_use": len(self._system_kv_cache),
                "counters": counters,
            }
        if self._is_mllm and self._model is not None:
            result["mllm_cache"] = self._model.get_cache_stats()
        return result or None

    def clear_runtime_caches(self) -> dict[str, Any] | None:
        """Clear engine-managed runtime caches.

        Includes the multi-slot system-prompt KV LRU — each retained snapshot
        is multi-GB on the Metal heap, so DELETE /v1/cache must drop them or
        the operator's reset is silently incomplete. Counters reset alongside
        so /v1/cache/stats reflects the cleared state immediately.

        OrderedDict ops are atomic under the GIL: a concurrent worker that has
        already captured a tuple reference from .get() finishes safely against
        its own copy; any new request after this call hits MISS and repopulates
        from scratch. No need to acquire _generation_lock for the clear itself.
        """
        result: dict[str, Any] = {}

        dropped = len(self._system_kv_cache)
        if dropped or any(self._system_kv_cache_stats.values()):
            self._system_kv_cache.clear()
            for k in self._system_kv_cache_stats:
                self._system_kv_cache_stats[k] = 0
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
            result["system_kv_cache"] = {"dropped_entries": dropped}

        if self._is_mllm and self._model is not None:
            self._model.clear_cache()
            result["model_cache"] = True

        return result or None
