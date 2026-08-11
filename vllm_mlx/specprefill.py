# SPDX-License-Identifier: Apache-2.0
"""SpecPrefill: Attention-based sparse prefill for MLX.

Full pipeline for reducing TTFT on long prompts:
  Step 1 (score_tokens): Use a small draft model to identify important tokens
  Step 2 (sparse_prefill): Prefill target model with only selected tokens,
         preserving original positional encoding via manual RoPE

Usage:
    from specprefill import score_tokens, select_chunks, sparse_prefill, cleanup_rope

    # 1. Score with draft model
    importance = score_tokens(draft_model, tokens)

    # 2. Select important token chunks
    selected = select_chunks(importance, keep_pct=0.3)

    # 3. Sparse prefill on target model
    target_cache = make_prompt_cache(target_model)
    logits = sparse_prefill(target_model, tokens, selected, target_cache)

    # 4. Generate normally using target_cache...

    # 5. Cleanup
    cleanup_rope(target_model)

Design notes:
    - RoPE is relative: Q_m @ K_p^T depends only on (m - p). Selected keys stored
      contiguously in the cache buffer with correct RoPE angles produce correct
      attention during decode.
    - After sparse prefill of N tokens from a total prompt of M, cache.offset = N
      but decode RoPE needs position M. The _OffsetAdjustedRoPE adds (M - N) to
      each RoPE offset call, so decode position = N + i + (M - N) = M + i.
    - GatedDeltaNet (linear attention) layers process sparse tokens through their
      conv/SSM state normally. This is lossy but acceptable per the SpecPrefill
      paper — attention layers are the primary long-range mechanism.

Reference: arxiv.org/abs/2502.02789 (SpecPrefill: Speculative Prefilling)
"""

import functools
import inspect
import math
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import mlx.core as mx

from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler

from vllm_mlx.specprefill_selection import (
    SPECPREFILL_SELECTOR_VERSION,  # noqa: F401 - re-exported public identity
    RotatingTailRequirement,
    SelectionPlan,
    SelectionPolicy,
    build_selection_plan_from_chunk_scores,
)


class SpecPrefillPolicy(str, Enum):
    """Requested or effective prefill policy for one text request."""

    AUTO = "auto"
    SPARSE = "sparse"
    DENSE = "dense"


class SpecPrefillCoverage(str, Enum):
    """Whether a workload can tolerate omitted prompt context."""

    SELECTIVE = "selective"
    EXHAUSTIVE = "exhaustive"
    UNKNOWN = "unknown"


class SpecPrefillScorerLaneBusy(RuntimeError):
    """Another bounded request currently owns the scorer lane."""


@dataclass(frozen=True)
class SpecPrefillDecision:
    """A request-local admission decision, independent from engine execution."""

    requested_policy: SpecPrefillPolicy
    effective_policy: SpecPrefillPolicy
    coverage: SpecPrefillCoverage
    fallback_reason: str | None = None
    selection_plan: SelectionPlan | None = None

    def __post_init__(self) -> None:
        if self.effective_policy is SpecPrefillPolicy.AUTO:
            raise ValueError("effective_policy must resolve auto to sparse or dense")
        if self.effective_policy is SpecPrefillPolicy.SPARSE and self.fallback_reason:
            raise ValueError("a sparse decision cannot have a fallback reason")
        if self.effective_policy is SpecPrefillPolicy.DENSE and self.selection_plan:
            raise ValueError("a dense decision cannot expose a sparse selection plan")


@dataclass(frozen=True)
class SpecPrefillArchitectureAdapter:
    """Explicit model-family contract for scoring and cache ownership."""

    name: str
    model_types: tuple[str, ...]
    query_extractor: Callable[..., mx.array]
    cache_map_builder: Callable[[Any], dict[int, int]]


class Gemma4Variant(str, Enum):
    """Gemma 4 text topology classes supported by the scorer."""

    DENSE = "dense"
    A4B = "a4b"


@dataclass(frozen=True)
class Gemma4Layout:
    """Known Gemma wrapper resolution, never an arbitrary attribute walk.

    ``execution_model`` is the object the caller forwards through.  The scorer
    installs captures on ``decoder_model`` and asks ``cache_model`` to make the
    compact owner cache.  These are distinct for the Gemma VLM outer wrapper.
    """

    execution_model: Any
    decoder_model: Any
    cache_model: Any
    previous_kvs: tuple[int, ...]
    variant: Gemma4Variant


def resolve_specprefill_decision(
    policy: SpecPrefillPolicy | str,
    coverage: SpecPrefillCoverage | str,
    *,
    production: bool,
    text_only: bool = True,
    threshold_met: bool = True,
    admission_allowed: bool = True,
) -> SpecPrefillDecision:
    """Resolve a conservative pre-execution policy decision.

    Production may only engage sparse prefill for declared selective text
    workloads that have passed both value and residency admission. Explicit
    sparse forcing remains available for diagnostic use only.
    """
    requested = SpecPrefillPolicy(policy)
    declared_coverage = SpecPrefillCoverage(coverage)
    if requested is SpecPrefillPolicy.DENSE:
        return SpecPrefillDecision(
            requested, SpecPrefillPolicy.DENSE, declared_coverage
        )
    if not text_only:
        return SpecPrefillDecision(
            requested, SpecPrefillPolicy.DENSE, declared_coverage, "media_request"
        )
    if (
        declared_coverage is not SpecPrefillCoverage.SELECTIVE
        and requested is not SpecPrefillPolicy.SPARSE
    ):
        return SpecPrefillDecision(
            requested,
            SpecPrefillPolicy.DENSE,
            declared_coverage,
            "coverage_not_selective",
        )
    if production and requested is SpecPrefillPolicy.SPARSE:
        return SpecPrefillDecision(
            requested,
            SpecPrefillPolicy.DENSE,
            declared_coverage,
            "sparse_forcing_diagnostic_only",
        )
    if not threshold_met:
        return SpecPrefillDecision(
            requested, SpecPrefillPolicy.DENSE, declared_coverage, "below_threshold"
        )
    if not admission_allowed:
        return SpecPrefillDecision(
            requested, SpecPrefillPolicy.DENSE, declared_coverage, "admission_denied"
        )
    return SpecPrefillDecision(requested, SpecPrefillPolicy.SPARSE, declared_coverage)


# ===========================================================================
# Step 1: Token importance scoring (draft model)
# ===========================================================================


class _AttentionCapture:
    """Standalone compatibility wrapper for direct capture tests.

    This helper is never installed on a scorer model. Replacing an MLX child
    module with a plain proxy removes its parameter subtree from
    :class:`mlx.nn.Module`. :class:`SpecPrefillScorer` instead installs a
    class-level dispatcher while retaining the exact attention instances.
    """

    def __init__(
        self,
        original,
        buf_idx,
        query_buffer=None,
        query_extractor=None,
    ):
        if isinstance(original, _AttentionCapture):
            raise RuntimeError("SpecPrefill attention wrappers cannot be nested")
        self._original = original
        self._buf_idx = buf_idx
        self._query_buffer = query_buffer
        self._query_extractor = query_extractor or _qwen35_extract_queries
        self._call_plan = _build_capture_call_plan(original)

    def __call__(self, *args, **kwargs):
        x, cache, capture_kwargs = _capture_call_arguments(
            self._call_plan, args, kwargs
        )
        queries = self._query_extractor(
            self._original, x, cache=cache, **capture_kwargs
        )
        self._query_buffer[self._buf_idx].append(queries)
        return self._original(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._original, name)


@dataclass(frozen=True)
class _CaptureCallPlan:
    input_name: str
    input_position: int | None
    cache_position: int | None
    positional_names: tuple[str, ...]


@dataclass(frozen=True)
class _InstalledAttentionCapture:
    """Immutable dispatch metadata for one original attention instance."""

    scorer_ref: weakref.ReferenceType["SpecPrefillScorer"]
    buffer_index: int
    call_plan: _CaptureCallPlan


@dataclass(frozen=True)
class _AttentionClassDispatch:
    """Original and stable dispatch callables installed for one Python class."""

    original_call: Callable[..., Any]
    dispatch_call: Callable[..., Any]
    call_plan: _CaptureCallPlan


_ATTENTION_DISPATCH_LOCK = threading.RLock()
_ATTENTION_DISPATCHED_CLASSES: dict[type, _AttentionClassDispatch] = {}
_ATTENTION_CAPTURES: dict[
    int, tuple[weakref.ReferenceType[Any], _InstalledAttentionCapture]
] = {}


def _attention_capture_for(instance: Any) -> _InstalledAttentionCapture | None:
    """Return identity-matched state without requiring a hashable nn.Module."""
    object_id = id(instance)
    entry = _ATTENTION_CAPTURES.get(object_id)
    if entry is None:
        return None
    instance_ref, installed = entry
    registered = instance_ref()
    if registered is instance:
        return installed
    if registered is None and _ATTENTION_CAPTURES.get(object_id) is entry:
        _ATTENTION_CAPTURES.pop(object_id, None)
        return None
    raise RuntimeError("SpecPrefill attention registry identity collision")


def _attention_registry_ref(instance: Any) -> weakref.ReferenceType[Any]:
    """Build a weak identity key whose callback cannot delete a reused ID."""
    object_id = id(instance)

    def _remove(instance_ref):
        with _ATTENTION_DISPATCH_LOCK:
            entry = _ATTENTION_CAPTURES.get(object_id)
            if entry is not None and entry[0] is instance_ref:
                _ATTENTION_CAPTURES.pop(object_id, None)

    try:
        return weakref.ref(instance, _remove)
    except TypeError as exc:
        raise RuntimeError(
            "SpecPrefill attention instances must support weak references"
        ) from exc


def _install_attention_capture(
    attention: Any, scorer: "SpecPrefillScorer", buffer_index: int
) -> None:
    """Register one attention instance without changing model topology.

    Python resolves ``instance(...)`` through ``type(instance).__call__``.
    Installing one stable dispatcher on that class lets us select capture
    behaviour through a weak per-instance registry while leaving the model's
    original child module, parameters, and state dictionary untouched.
    """
    attention_type = type(attention)
    with _ATTENTION_DISPATCH_LOCK:
        class_dispatch = _ATTENTION_DISPATCHED_CLASSES.get(attention_type)
        if class_dispatch is None:
            resolved_call = attention_type.__call__
            inherited_dispatch = next(
                (
                    installed_dispatch
                    for installed_dispatch in _ATTENTION_DISPATCHED_CLASSES.values()
                    if installed_dispatch.dispatch_call is resolved_call
                ),
                None,
            )
            if inherited_dispatch is None:
                original_call = resolved_call
                call_plan = _build_capture_call_plan(attention)
            else:
                # A subclass can inherit a base class dispatcher. Flatten it to
                # the original callable so capture is performed exactly once.
                original_call = inherited_dispatch.original_call
                call_plan = inherited_dispatch.call_plan

            @functools.wraps(original_call)
            def _dispatch(instance, *args, **kwargs):
                with _ATTENTION_DISPATCH_LOCK:
                    installed = _attention_capture_for(instance)
                if installed is None:
                    return original_call(instance, *args, **kwargs)
                owner = installed.scorer_ref()
                if owner is None:
                    with _ATTENTION_DISPATCH_LOCK:
                        entry = _ATTENTION_CAPTURES.get(id(instance))
                        if entry is not None and entry[0]() is instance:
                            _ATTENTION_CAPTURES.pop(id(instance), None)
                    return original_call(instance, *args, **kwargs)
                session = owner._capture_session_for_attention()
                if session is not None:
                    x, cache, capture_kwargs = _capture_call_arguments(
                        installed.call_plan, args, kwargs
                    )
                    queries = session.query_extractor(
                        instance, x, cache=cache, **capture_kwargs
                    )
                    session.query_buffer[installed.buffer_index].append(queries)
                return original_call(instance, *args, **kwargs)

            try:
                attention_type.__call__ = _dispatch
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "Cannot install a stable SpecPrefill dispatcher for "
                    f"attention class {attention_type}"
                ) from exc
            class_dispatch = _AttentionClassDispatch(
                original_call, _dispatch, call_plan
            )
            _ATTENTION_DISPATCHED_CLASSES[attention_type] = class_dispatch
        elif attention_type.__call__ is not class_dispatch.dispatch_call:
            raise RuntimeError("SpecPrefill attention class dispatcher was modified")
        else:
            call_plan = class_dispatch.call_plan

        existing = _attention_capture_for(attention)
        if existing is not None and existing.scorer_ref() is not scorer:
            raise RuntimeError("SpecPrefill attention instance is already registered")
        _ATTENTION_CAPTURES[id(attention)] = (
            _attention_registry_ref(attention),
            _InstalledAttentionCapture(weakref.ref(scorer), buffer_index, call_plan),
        )


def _uninstall_attention_capture(attention: Any, scorer: "SpecPrefillScorer") -> None:
    """Rollback a failed multi-layer registration without touching the model."""
    with _ATTENTION_DISPATCH_LOCK:
        installed = _attention_capture_for(attention)
        if installed is not None and installed.scorer_ref() is scorer:
            _ATTENTION_CAPTURES.pop(id(attention), None)


def _build_capture_call_plan(original):
    """Inspect an attention callable once when its stable wrapper is installed."""
    try:
        signature = inspect.signature(original)
        positional_names = tuple(
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except (TypeError, ValueError):
        positional_names = ()

    input_name = "x" if "x" in positional_names else None
    if input_name is None:
        input_name = positional_names[0] if positional_names else "x"
    input_position = (
        positional_names.index(input_name) if input_name in positional_names else 0
    )
    cache_position = (
        positional_names.index("cache") if "cache" in positional_names else 2
    )
    return _CaptureCallPlan(
        input_name=input_name,
        input_position=input_position,
        cache_position=cache_position,
        positional_names=positional_names,
    )


def _capture_call_arguments(plan, args, kwargs):
    """Extract scorer inputs without changing the delegated attention call."""
    if plan.input_name in kwargs:
        x = kwargs[plan.input_name]
    elif plan.input_position is not None and len(args) > plan.input_position:
        x = args[plan.input_position]
    else:
        raise RuntimeError("Cannot identify attention input for SpecPrefill")

    if "cache" in kwargs:
        cache = kwargs["cache"]
    elif plan.cache_position is not None and len(args) > plan.cache_position:
        cache = args[plan.cache_position]
    else:
        cache = None

    capture_kwargs = dict(kwargs)
    capture_kwargs.pop(plan.input_name, None)
    capture_kwargs.pop("mask", None)
    capture_kwargs.pop("cache", None)
    for position, name in enumerate(plan.positional_names):
        if position >= len(args):
            break
        if name not in (plan.input_name, "mask", "cache"):
            capture_kwargs.setdefault(name, args[position])
    return x, cache, capture_kwargs


def _qwen35_extract_queries(attn, x, cache=None, **_kwargs):
    """Extract post-RoPE queries from Qwen3.5 attention (gate split + q_norm).

    Qwen3.5 q_proj output is 2x wider: [queries, gate]. We split, normalize,
    then apply RoPE.
    """
    B, L, D = x.shape
    q_out = attn.q_proj(x)
    queries, _gate = mx.split(
        q_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _qwen_extract_queries(attn, x, cache=None, **_kwargs):
    """Extract normalized RoPE queries from dense Qwen attention.

    Qwen 3's projection is not gated, unlike the Qwen3.5 hybrid family, but
    it retains Q/K normalization. Treating it as a generic Llama block drops
    that normalization and changes the scoring distribution.
    """
    B, L, _ = x.shape
    n_heads = getattr(attn, "n_heads", getattr(attn, "num_attention_heads", None))
    if n_heads is None:
        raise ValueError("Cannot determine Qwen attention head count")
    queries = attn.q_proj(x).reshape(B, L, n_heads, -1)
    q_norm = getattr(attn, "q_norm", None)
    if q_norm is not None:
        queries = q_norm(queries)
    queries = queries.transpose(0, 2, 1, 3)
    if cache is not None:
        return attn.rope(queries, offset=cache.offset)
    return attn.rope(queries)


def _llama_extract_queries(attn, x, cache=None, **_kwargs):
    """Extract post-RoPE queries from standard transformer attention.

    Standard architecture: q_proj → reshape → RoPE. No gate, no q_norm.
    Works for Llama 3.x, Mistral, Gemma, GPT-OSS, and other GQA models.
    """
    B, L, D = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x)
    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _gemma4_extract_queries(attn, x, cache=None, offset=None, **_kwargs):
    """Extract Gemma 4 normalized partial-RoPE queries.

    KV-shared Gemma layers receive their logical offset explicitly from their
    owner layer. The capture wrapper must preserve that argument rather than
    substituting a fresh cache-local position.
    """
    B, L, _ = x.shape
    n_heads = getattr(attn, "n_heads", getattr(attn, "num_attention_heads", None))
    if n_heads is None:
        raise ValueError("Cannot determine Gemma 4 attention head count")
    queries = attn.q_proj(x).reshape(B, L, n_heads, -1)
    q_norm = getattr(attn, "q_norm", None)
    if q_norm is not None:
        queries = q_norm(queries)
    queries = queries.transpose(0, 2, 1, 3)
    if offset is None:
        offset = cache.offset if cache is not None else 0
    return attn.rope(queries, offset=offset)


def _nemotron_h_extract_queries(attn, x, cache=None, **_kwargs):
    """Extract queries from Nemotron-H attention (no RoPE, no gate, no q_norm).

    Nemotron-H attention layers have NO positional encoding — RoPE is absent.
    Positional modeling comes from Mamba2 layers. Attention is content-based only.
    """
    B, L, D = x.shape
    queries = attn.q_proj(x).reshape(B, L, attn.num_heads, -1).transpose(0, 2, 1, 3)
    # No RoPE to apply — queries are used as-is for content-based scoring
    return queries


@dataclass
class _ScorerCaptureSession:
    query_extractor: Callable[..., mx.array]
    query_buffer: list[list[mx.array]]
    owner_thread_id: int
    capture_enabled: bool = True


class SpecPrefillScorer:
    """Install-once, serialized scorer for one draft model.

    Attention instances remain the exact MLX child modules loaded with the
    model. Their Python classes receive one stable dispatcher, while immutable
    weak registry entries associate only this scorer's attention instances
    with request-local capture state. The initial implementation deliberately
    rejects overlapping work instead of allowing requests to share buffers.
    """

    def __init__(self, model):
        self._model_ref = weakref.ref(model)
        self.adapter = resolve_specprefill_adapter(model)
        self._decoder_model = _scorer_decoder_model(model, self.adapter)
        self._session_lock = threading.Lock()
        self._quantum_lock = threading.Lock()
        self._active_session = None
        self._session_context = threading.local()

        attention_layers = _find_attention_layers(self._decoder_model)
        if not attention_layers:
            raise ValueError("SpecPrefill scorer model has no attention layers")
        attentions = [
            (layer_idx, _get_attn_module(layer))
            for layer_idx, layer in attention_layers
        ]
        installed: list[Any] = []
        try:
            for buffer_index, (_layer_idx, attention) in enumerate(attentions):
                _install_attention_capture(attention, self, buffer_index)
                installed.append(attention)
        except Exception:
            for attention in reversed(installed):
                _uninstall_attention_capture(attention, self)
            raise
        self.attention_layer_indices = tuple(
            layer_idx for layer_idx, _attention in attentions
        )
        self._installed_attentions = tuple(attentions)

    @classmethod
    def for_model(cls, model):
        """Return the sole scorer for ``model``, installing dispatch once."""
        with _SCORER_REGISTRY_LOCK:
            scorer = _scorer_for_model(model)
            if scorer is None:
                scorer = cls(model)
                _register_scorer_model(model, scorer)
            scorer._verify_installed()
            return scorer

    @property
    def model(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("SpecPrefill scorer model is no longer available")
        return model

    @property
    def decoder_model(self):
        return self._decoder_model

    @property
    def cache_model(self):
        return _scorer_cache_model(self.model, self.adapter)

    @property
    def capture_active(self) -> bool:
        return self._active_session is not None

    def _verify_installed(self):
        model = self.decoder_model
        for buffer_index, (layer_idx, attention) in enumerate(
            self._installed_attentions
        ):
            if _get_attn_module(model.layers[layer_idx]) is not attention:
                raise RuntimeError("SpecPrefill scorer attention topology was modified")
            with _ATTENTION_DISPATCH_LOCK:
                installed = _attention_capture_for(attention)
                class_dispatch = _ATTENTION_DISPATCHED_CLASSES.get(type(attention))
            if (
                installed is None
                or installed.scorer_ref() is not self
                or installed.buffer_index != buffer_index
            ):
                raise RuntimeError(
                    "SpecPrefill scorer attention registration was modified"
                )
            if (
                class_dispatch is None
                or type(attention).__call__ is not class_dispatch.dispatch_call
            ):
                raise RuntimeError(
                    "SpecPrefill scorer attention class dispatcher was modified"
                )

    def _capture_session_for_attention(self):
        session = self._active_session
        if session is None:
            return None
        context_session = getattr(self._session_context, "capture", None)
        if (
            context_session is not session
            or session.owner_thread_id != threading.get_ident()
        ):
            raise RuntimeError(
                "SpecPrefill scorer invoked outside its active request session"
            )
        return session if session.capture_enabled else None

    @contextmanager
    def capture_session(self, query_extractor=None, *, capture_enabled=True):
        """Activate one fail-closed request-local capture session."""
        self._verify_installed()
        if not self._session_lock.acquire(blocking=False):
            raise RuntimeError("SpecPrefill scorer already has an active session")
        if self._active_session is not None:
            self._session_lock.release()
            raise RuntimeError("SpecPrefill scorer already has an active session")

        session = _ScorerCaptureSession(
            query_extractor=query_extractor or self.adapter.query_extractor,
            query_buffer=[[] for _ in self.attention_layer_indices],
            owner_thread_id=threading.get_ident(),
            capture_enabled=capture_enabled,
        )
        self._active_session = session
        self._session_context.capture = session
        try:
            yield session
        finally:
            self._active_session = None
            try:
                del self._session_context.capture
            except AttributeError:
                pass
            self._session_lock.release()

    @contextmanager
    def scoring_quantum(self):
        """Lease the scorer lane for one bounded request-local session step.

        Capture itself still takes the shorter ``capture_session`` lock.  This
        companion lock covers cache/query importance work too, then releases
        before another request's next quantum can run.
        """
        if not self._quantum_lock.acquire(blocking=False):
            raise SpecPrefillScorerLaneBusy("SpecPrefill scorer lane is busy")
        try:
            yield
        finally:
            self._quantum_lock.release()

    def score_tokens(
        self,
        tokens,
        n_lookahead=8,
        pool_kernel=13,
        temp=0.6,
        top_p=0.95,
        prefill_step_size=2048,
        query_extractor=None,
        cancel_check=None,
        sampling_seed=None,
    ):
        """Score through bounded request-local session quanta for compatibility."""
        # Local import keeps the session implementation free to reuse this
        # install-once scorer without a module-import cycle.
        from .specprefill_scorer_session import SpecPrefillScorerSession

        return SpecPrefillScorerSession(
            self,
            tokens,
            n_lookahead=n_lookahead,
            pool_kernel=pool_kernel,
            temp=temp,
            top_p=top_p,
            prefill_step_size=prefill_step_size,
            query_extractor=query_extractor,
            cancel_check=cancel_check,
            sampling_seed=sampling_seed,
        ).run_to_completion()


_SCORER_REGISTRY: dict[int, tuple[weakref.ReferenceType[Any], SpecPrefillScorer]] = {}
_SCORER_REGISTRY_LOCK = threading.RLock()


def _scorer_for_model(model: Any) -> SpecPrefillScorer | None:
    object_id = id(model)
    entry = _SCORER_REGISTRY.get(object_id)
    if entry is None:
        return None
    model_ref, scorer = entry
    registered = model_ref()
    if registered is model:
        return scorer
    if registered is None and _SCORER_REGISTRY.get(object_id) is entry:
        _SCORER_REGISTRY.pop(object_id, None)
        return None
    raise RuntimeError("SpecPrefill scorer registry identity collision")


def _register_scorer_model(model: Any, scorer: SpecPrefillScorer) -> None:
    object_id = id(model)

    def _remove(model_ref):
        with _SCORER_REGISTRY_LOCK:
            entry = _SCORER_REGISTRY.get(object_id)
            if entry is not None and entry[0] is model_ref:
                _SCORER_REGISTRY.pop(object_id, None)

    try:
        model_ref = weakref.ref(model, _remove)
    except TypeError as exc:
        raise RuntimeError(
            "SpecPrefill scorer models must support weak references"
        ) from exc
    existing = _scorer_for_model(model)
    if existing is not None and existing is not scorer:
        raise RuntimeError("SpecPrefill scorer model is already registered")
    _SCORER_REGISTRY[object_id] = (model_ref, scorer)


def _prefill_draft(model, tokens, cache, step_size=2048, cancel_check=None):
    """Prefill prompt tokens into cache. Returns logits from last token."""
    prompt = mx.array(tokens) if not isinstance(tokens, mx.array) else tokens
    n = len(tokens)
    processed = 0
    while n - processed > 1:
        if cancel_check is not None:
            cancel_check()
        chunk = min(step_size, n - processed - 1)
        model(prompt[processed : processed + chunk][None], cache=cache)
        mx.eval([c.state for c in cache])
        processed += chunk
        mx.clear_cache()
    if cancel_check is not None:
        cancel_check()
    logits = model(prompt[processed:][None], cache=cache)
    mx.eval(logits)
    return logits


def _lookahead_decode(
    model,
    first_logits,
    cache,
    n_steps,
    temp=0.6,
    top_p=0.95,
    cancel_check=None,
):
    """Run n_steps autoregressive decode, returning generated token ids.

    Query vectors are captured by the monkey-patched attention layers.
    """
    sampler = make_sampler(temp=temp, top_p=top_p)
    if cancel_check is not None:
        cancel_check()
    y = sampler(first_logits[:, -1, :])
    mx.eval(y)
    generated = [y.item()]
    for _ in range(n_steps):
        if cancel_check is not None:
            cancel_check()
        logits = model(y.reshape(1, -1), cache=cache)
        y = sampler(logits[:, -1, :])
        mx.eval(y)
        generated.append(y.item())
    return generated


def _avg_pool1d(x, kernel_size):
    """1D average pooling along last axis via prefix-sum.

    Args:
        x: (..., M) input
        kernel_size: window size (odd for centered)

    Returns:
        (..., M) pooled (same size, zero-padded at edges)
    """
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    zeros = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    prefix = mx.concatenate([zeros, mx.cumsum(padded, axis=-1)], axis=-1)
    return (prefix[..., kernel_size:] - prefix[..., :-kernel_size]) / kernel_size


def _compute_importance(
    query_buffer,
    attn_caches,
    n_prompt,
    n_attn_heads=None,
    n_kv_heads=None,
    pool_kernel=13,
):
    """Compute per-token importance from captured queries and cached keys.

    Aggregation (SpecPrefill paper):
      1. softmax(Q @ K^T / sqrt(d)) per head, per layer, per lookahead token
      2. avg_pool1d smoothing
      3. max across (layers × heads)
      4. mean across lookahead tokens

    Returns: (n_prompt,) importance scores.
    """
    # ``n_attn_heads`` and ``n_kv_heads`` remain accepted for compatibility
    # with older private callers, but heterogeneous models cannot use one
    # model-global pair. Derive the grouping from each realized query/cache
    # shape instead (Gemma 4 alternates sliding and global KV widths).
    del n_attn_heads, n_kv_heads
    all_scores = []

    for layer_i, captures in enumerate(query_buffer):
        if not captures:
            continue
        cache = attn_caches[layer_i]
        prompt_keys = cache.keys[..., :n_prompt, :]
        # Skip layers with windowed/rotating caches that don't span
        # the full prompt (e.g., GPT-OSS sliding_attention with 128-token window).
        # These lack global context and would produce mismatched score shapes.
        if prompt_keys.shape[-2] < n_prompt:
            continue
        head_dim = prompt_keys.shape[-1]
        q_stack = mx.concatenate(captures, axis=2)
        query_heads = int(q_stack.shape[1])
        key_heads = int(prompt_keys.shape[1])
        if query_heads <= 0 or key_heads <= 0 or query_heads % key_heads:
            raise RuntimeError(
                "SpecPrefill attention heads must have an integral per-layer "
                f"query/KV grouping (queries={query_heads}, keys={key_heads})"
            )
        if int(q_stack.shape[-1]) != int(head_dim):
            raise RuntimeError(
                "SpecPrefill captured query and cache key dimensions differ "
                f"({q_stack.shape[-1]} != {head_dim})"
            )
        heads_per_group = query_heads // key_heads
        if heads_per_group > 1:
            expanded_keys = mx.repeat(prompt_keys, heads_per_group, axis=1)
        else:
            expanded_keys = prompt_keys
        scale = head_dim**-0.5
        scores = (q_stack @ expanded_keys.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
        all_scores.append(weights.squeeze(0))

    if not all_scores:
        raise RuntimeError("No attention scores captured — check model/patching")

    combined = mx.concatenate(all_scores, axis=0)
    if pool_kernel and pool_kernel > 1:
        combined = _avg_pool1d(combined, pool_kernel)
    max_scores = mx.max(combined, axis=0)
    importance = mx.mean(max_scores, axis=0)
    return importance


def score_tokens(
    model,
    tokens,
    n_lookahead=8,
    pool_kernel=13,
    temp=0.6,
    top_p=0.95,
    prefill_step_size=2048,
    query_extractor=None,
    cancel_check=None,
    sampling_seed=None,
):
    """Score token importance using attention-based analysis on a draft model.

    Runs the full scoring pipeline:
      1. Prefill the draft model with all tokens
      2. N lookahead decode steps, capturing query vectors from attention layers
      3. Compute importance: Q_lookahead @ K_prompt^T, aggregated across heads/layers

    The draft model's cache is created internally and discarded after scoring.

    Args:
        model: Draft model (small, fast — e.g. 4B)
        tokens: list or mx.array of token IDs
        n_lookahead: decode steps for query capture (default 8)
        pool_kernel: smoothing kernel for avg_pool1d (default 13, 0=disable)
        temp: sampling temperature for lookahead (default 0.6)
        top_p: top-p for lookahead (default 0.95)
        prefill_step_size: chunk size for draft prefill (default 2048)
        query_extractor: function(attn, x, cache) → queries tensor.
            Default: _qwen35_extract_queries. Use _llama_extract_queries for
            standard Llama/Mistral/Gemma models.

    Returns:
        importance: (M,) mx.array of per-token importance scores
    """
    scorer = SpecPrefillScorer.for_model(model)
    return scorer.score_tokens(
        tokens,
        n_lookahead=n_lookahead,
        pool_kernel=pool_kernel,
        temp=temp,
        top_p=top_p,
        prefill_step_size=prefill_step_size,
        query_extractor=query_extractor,
        cancel_check=cancel_check,
        sampling_seed=sampling_seed,
    )


def _chunk_scores(importance, chunk_size: int) -> tuple[int, list[float]]:
    """Compute all chunk means in one MLX expression and validate them."""
    if len(importance.shape) != 1:
        raise ValueError("importance must be a one-dimensional tensor")
    prompt_length = int(importance.shape[0])
    if prompt_length <= 0:
        raise ValueError("importance must not be empty")
    n_chunks = math.ceil(prompt_length / chunk_size)
    padded_length = n_chunks * chunk_size
    padded = (
        mx.pad(importance, [(0, padded_length - prompt_length)])
        if padded_length != prompt_length
        else importance
    )
    sums = mx.sum(padded.reshape(n_chunks, chunk_size), axis=1)
    lengths = mx.minimum(
        mx.full((n_chunks,), chunk_size),
        prompt_length - mx.arange(n_chunks) * chunk_size,
    )
    scores = [float(score) for score in (sums / lengths).tolist()]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("importance must contain only finite values")
    return prompt_length, scores


def build_selection_plan(
    importance,
    keep_pct: float = 0.3,
    chunk_size: int = 32,
    backbone_pct: float = 0.0,
    *,
    halo_chunks: int = 1,
    anchor_chunks: int = 1,
    control_token_indices: tuple[int, ...] = (),
    rotating_tail_requirement: RotatingTailRequirement | None = None,
) -> SelectionPlan:
    """Build a deterministic, versioned hybrid sparse-prefill plan.

    Fixed anchors preserve prompt framing, a stratified backbone prevents
    score collapse into one region, and an importance-ranked halo keeps local
    context around a selected chunk. All chunk means are calculated on device
    before one O(chunks) host ranking operation.
    """
    prompt_length, scores = _chunk_scores(importance, chunk_size)
    policy = SelectionPolicy(
        keep_pct=keep_pct,
        backbone_pct=backbone_pct,
        halo_chunks=halo_chunks,
        anchor_chunks=anchor_chunks,
        chunk_size=chunk_size,
    )
    return build_selection_plan_from_chunk_scores(
        prompt_length=prompt_length,
        chunk_scores=scores,
        policy=policy,
        control_token_indices=control_token_indices,
        rotating_tail_requirement=rotating_tail_requirement,
    )


def select_chunks(
    importance,
    keep_pct=0.3,
    chunk_size=32,
    backbone_pct=0.0,
    *,
    halo_chunks=1,
    anchor_chunks=1,
    control_token_indices=(),
    rotating_tail_requirement=None,
):
    """Return executor-ready indices from :func:`build_selection_plan`."""
    plan = build_selection_plan(
        importance,
        keep_pct=keep_pct,
        chunk_size=chunk_size,
        backbone_pct=backbone_pct,
        halo_chunks=halo_chunks,
        anchor_chunks=anchor_chunks,
        control_token_indices=control_token_indices,
        rotating_tail_requirement=rotating_tail_requirement,
    )
    return mx.array(plan.selected_indices, dtype=mx.int32)


# ===========================================================================
# Step 2: Sparse prefill with non-contiguous position IDs (target model)
# ===========================================================================


# ---------------------------------------------------------------------------
# Manual RoPE at arbitrary positions
# ---------------------------------------------------------------------------


def manual_rope(x, positions, dims, base=10000.0, scale=1.0):
    """Apply RoPE at arbitrary (non-contiguous) positions.

    Uses non-traditional (interleaved) layout matching Qwen3.5:
    rotates first `dims` dimensions as pairs [0,half), [half,dims),
    passes through [dims:] unchanged.

    Args:
        x: (B, n_heads, L, head_dim) input tensor
        positions: (L,) position indices (can be non-contiguous)
        dims: number of dimensions to rotate (head_dim * partial_rotary_factor)
        base: RoPE base frequency (default 10000.0)
        scale: position scale divisor (default 1.0, higher = compressed positions)

    Returns:
        (B, n_heads, L, head_dim) with RoPE applied
    """
    half = dims // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    scaled_pos = positions.astype(mx.float32) / scale
    angles = scaled_pos[:, None] * inv_freq[None, :]  # (L, half)
    cos_a = mx.cos(angles)[None, None, :, :]  # (1, 1, L, half)
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


def manual_rope_with_freqs(x, positions, dims, freqs, pre_scale=1.0):
    """Apply RoPE at arbitrary positions using pre-computed frequencies.

    For custom RoPE variants (Llama3, Yarn, SuScaled) that store _freqs.
    """
    half = dims // 2
    inv_freq = (1.0 / freqs).astype(mx.float32)
    angles = positions[:, None].astype(mx.float32) * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    if pre_scale != 1.0:
        x_rot = pre_scale * x_rot
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


# ---------------------------------------------------------------------------
# RoPE wrappers
# ---------------------------------------------------------------------------


class _PositionMappedRoPE:
    """Wraps a RoPE module to apply rotation at non-contiguous positions.

    Used during sparse prefill. The `offset` parameter from the cache tells us
    which slice of the position array to use for the current chunk:
        positions = all_positions[(offset - cache_start) : (offset - cache_start) + L]

    When composing with a pre-populated cache (e.g., system KV cache), cache_start
    is the initial cache offset so indexing into the position array is correct.
    """

    def __init__(self, original_rope, all_positions, cache_start=0):
        self._original = original_rope
        self._all_positions = all_positions
        self._cache_start = cache_start
        self._has_custom_freqs = hasattr(original_rope, "_freqs")

        if self._has_custom_freqs:
            self._freqs = original_rope._freqs
            self._dims = _get_dims(original_rope)
            self._pre_scale = _get_pre_scale(original_rope)
        else:
            # Standard nn.RoPE: attributes are dims, base, scale (no underscore)
            self._dims = original_rope.dims
            self._base = original_rope.base
            self._scale = original_rope.scale

    def __call__(self, x, offset=0):
        L = x.shape[2]
        idx = offset - self._cache_start
        positions = self._all_positions[idx : idx + L]
        if self._has_custom_freqs:
            return manual_rope_with_freqs(
                x, positions, self._dims, self._freqs, pre_scale=self._pre_scale
            )
        return manual_rope(x, positions, self._dims, base=self._base, scale=self._scale)


class _OffsetAdjustedRoPE:
    """Wraps a RoPE module to add a constant offset for decode after sparse prefill.

    After sparse prefill of N tokens from a prompt of M total tokens:
      cache.offset = N + i  (i = decode step)
      desired RoPE position = M + i
      adjustment = M - N

    So: RoPE(x, offset = cache.offset + adjustment) = RoPE(x, M + i)
    """

    def __init__(self, original_rope, adjustment):
        self._original = original_rope
        self._adjustment = adjustment

    def __call__(self, x, offset=0):
        return self._original(x, offset=offset + self._adjustment)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dims(rope_module):
    """Extract rotary dimensions from any RoPE variant."""
    for attr in ("_dims", "dim", "dims"):
        if hasattr(rope_module, attr):
            return getattr(rope_module, attr)
    raise ValueError(f"Cannot determine dims from {type(rope_module)}")


def _get_pre_scale(rope_module):
    """Extract pre-scale factor from custom RoPE variants (SuScaled, Yarn)."""
    if hasattr(rope_module, "mscale"):
        return rope_module.mscale
    if hasattr(rope_module, "_scale") and hasattr(rope_module, "dim"):
        return rope_module._scale
    return 1.0


def _find_attention_layers(model):
    """Find all full-attention layers across architectures.

    Supports:
      - Qwen3.5 / Llama / GPT-OSS: layers with `self_attn` attribute
      - Nemotron-H: layers with `block_type == "*"` (attention blocks use `mixer`)

    Returns list of (layer_idx, layer) tuples.
    """
    results = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            results.append((idx, layer))
        elif getattr(layer, "block_type", None) == "*":
            results.append((idx, layer))
    return results


def _get_attn_module(layer):
    """Get the attention module from a layer (self_attn or mixer)."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if getattr(layer, "block_type", None) == "*":
        return layer.mixer
    return None


def _get_rope(attn):
    """Get the RoPE module from an attention layer, or None.

    mlx_lm models use ``self.rope``; mlx_vlm models use ``self.rotary_emb``.
    """
    return getattr(attn, "rope", None) or getattr(attn, "rotary_emb", None)


def _set_rope(attn, rope_module):
    """Set the RoPE module on an attention layer."""
    if hasattr(attn, "rope"):
        attn.rope = rope_module
    elif hasattr(attn, "rotary_emb"):
        attn.rotary_emb = rope_module


def _set_attn_module(layer, module):
    """Set the attention module on a layer (self_attn or mixer)."""
    if hasattr(layer, "self_attn"):
        layer.self_attn = module
    elif getattr(layer, "block_type", None) == "*":
        layer.mixer = module


def _get_model_type(model) -> str:
    """Return a model type from supported mlx-lm/mlx-vlm wrapper layouts."""
    if isinstance(model, str):
        return model
    candidates = (
        getattr(model, "config", None),
        getattr(model, "args", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "model", None), "args", None),
    )
    for candidate in candidates:
        model_type = getattr(candidate, "model_type", None)
        if model_type:
            return str(model_type)
    return ""


def _has_gemma4_decoder_contract(candidate: Any) -> bool:
    """Identify a Gemma text decoder without accepting arbitrary wrappers."""
    return (
        candidate is not None
        and hasattr(candidate, "layers")
        and getattr(candidate, "previous_kvs", None) is not None
    )


def _gemma4_variant(decoder_model: Any) -> Gemma4Variant:
    """Classify installed Gemma dense and A4B text contracts explicitly."""
    config = getattr(decoder_model, "config", None)
    if config is None or not hasattr(config, "num_experts"):
        config = getattr(getattr(decoder_model, "model", None), "config", None)
    experts = getattr(config, "num_experts", None)
    if isinstance(experts, bool) or not isinstance(experts, int) or experts <= 0:
        return Gemma4Variant.DENSE
    return Gemma4Variant.A4B


def resolve_gemma4_layout(model: Any) -> Gemma4Layout:
    """Resolve only observed mlx-lm/mlx-vlm Gemma text ownership layouts.

    Supported paths are the direct text decoder, the mlx-lm text wrapper's
    ``.model``, the extracted text wrapper's ``.model.previous_kvs``, and the
    mlx-vlm outer wrapper's ``.language_model.model``. Any other shape is
    rejected rather than guessed: cache ownership is a correctness contract,
    not a convenience traversal.
    """
    decoder_model = None
    cache_model = None
    previous_kvs = None
    if _has_gemma4_decoder_contract(model):
        decoder_model = model
        cache_model = model
        previous_kvs = model.previous_kvs
    elif _has_gemma4_decoder_contract(getattr(model, "model", None)):
        decoder_model = model.model
        cache_model = model
        previous_kvs = model.model.previous_kvs
    elif (
        hasattr(model, "layers")
        and getattr(getattr(model, "model", None), "previous_kvs", None) is not None
    ):
        # mlx-vlm's extracted text route can expose decoder layers on the
        # wrapper while retaining shared-KV ownership on ``.model``.
        decoder_model = model
        cache_model = model
        previous_kvs = model.model.previous_kvs
    else:
        language_model = getattr(model, "language_model", None)
        language_decoder = getattr(language_model, "model", None)
        if _has_gemma4_decoder_contract(language_decoder):
            decoder_model = language_decoder
            cache_model = language_model
            previous_kvs = language_decoder.previous_kvs
    if decoder_model is None or cache_model is None or previous_kvs is None:
        raise ValueError(
            "Unsupported Gemma 4 wrapper; expected direct decoder, .model, "
            "or .language_model.model cache ownership"
        )

    layers = tuple(decoder_model.layers)
    previous_kvs = tuple(previous_kvs)
    if not layers or len(previous_kvs) != len(layers):
        raise ValueError("Gemma 4 previous_kvs must cover every decoder layer")
    for layer_idx, owner_idx in enumerate(previous_kvs):
        if isinstance(owner_idx, bool) or not isinstance(owner_idx, int):
            raise ValueError("Gemma 4 previous_kvs entries must be integers")
        if not 0 <= owner_idx < len(layers):
            raise ValueError(
                f"Invalid Gemma 4 KV owner {owner_idx!r} for layer {layer_idx}"
            )
        if owner_idx > layer_idx:
            raise ValueError(
                f"Gemma 4 KV owner {owner_idx} must precede layer {layer_idx}"
            )
    return Gemma4Layout(
        execution_model=model,
        decoder_model=decoder_model,
        cache_model=cache_model,
        previous_kvs=previous_kvs,
        variant=_gemma4_variant(decoder_model),
    )


def _scorer_decoder_model(model: Any, adapter: SpecPrefillArchitectureAdapter) -> Any:
    """Return the child that owns decoder attention instances for capture."""
    if adapter is GEMMA4_ADAPTER:
        return resolve_gemma4_layout(model).decoder_model
    return model


def _scorer_cache_model(model: Any, adapter: SpecPrefillArchitectureAdapter) -> Any:
    """Return the model responsible for constructing the forward's KV cache."""
    if adapter is GEMMA4_ADAPTER:
        return resolve_gemma4_layout(model).cache_model
    return model


def _standard_layer_to_cache_map(model) -> dict[int, int]:
    """Map standard decoder layers to one cache entry each."""
    return {index: index for index in range(len(model.layers))}


def _gemma4_shared_kv_cache_map(model) -> dict[int, int]:
    """Map Gemma 4 layers to their physical KV-owning cache entries.

    Gemma's prompt-cache factory materializes one entry per unique KV owner,
    not one entry per decoder layer. ``previous_kvs`` contains decoder-layer
    owners, so map first-seen owners to their compact cache-list indices.
    """
    layout = resolve_gemma4_layout(model)
    previous_kvs = layout.previous_kvs
    owner_to_cache: dict[int, int] = {}
    mapping: dict[int, int] = {}
    for layer_idx, owner_idx in enumerate(previous_kvs):
        if owner_idx not in owner_to_cache:
            owner_to_cache[owner_idx] = len(owner_to_cache)
        mapping[layer_idx] = owner_to_cache[owner_idx]
    return mapping


def _hybrid_layer_to_cache_map(model) -> dict[int, int]:
    """Map Qwen3.5/3.6's mixed linear/attention stack to cache slots."""
    return _standard_layer_to_cache_map(model)


def validate_specprefill_cache_topology(
    adapter: SpecPrefillArchitectureAdapter,
    model: Any,
    cache: Sequence[Any],
    layer_to_cache: dict[int, int],
    *,
    attention_layer_indices: Sequence[int] = (),
) -> None:
    """Fail closed if a real prompt cache disagrees with its adapter topology."""
    if adapter is GEMMA4_ADAPTER:
        layout = resolve_gemma4_layout(model)
        layers = tuple(layout.decoder_model.layers)
        layer_count = len(layers)
        expected_cache_count = len(set(layout.previous_kvs))
    elif adapter is QWEN_HYBRID_ADAPTER:
        layer_count = len(model.layers)
        expected_cache_count = layer_count
    else:
        return

    if len(cache) != expected_cache_count:
        raise ValueError(
            f"{adapter.name} prompt cache has {len(cache)} entries; expected "
            f"{expected_cache_count} from its decoder topology"
        )
    expected_layers = (
        tuple(attention_layer_indices)
        if attention_layer_indices
        else tuple(range(layer_count))
    )
    for layer_idx in expected_layers:
        cache_idx = layer_to_cache.get(layer_idx)
        if cache_idx is None:
            raise ValueError(
                f"{adapter.name} cache map has no entry for decoder layer {layer_idx}"
            )
        if isinstance(cache_idx, bool) or not isinstance(cache_idx, int):
            raise ValueError(f"{adapter.name} cache index must be a host integer")
        if not 0 <= cache_idx < len(cache):
            raise ValueError(
                f"{adapter.name} cache index {cache_idx} is out of bounds for "
                f"decoder layer {layer_idx}"
            )

    if adapter is not GEMMA4_ADAPTER:
        return

    owner_to_cache: dict[int, int] = {}
    for layer_idx, owner_idx in enumerate(layout.previous_kvs):
        expected_cache_idx = owner_to_cache.setdefault(owner_idx, len(owner_to_cache))
        actual_cache_idx = layer_to_cache.get(layer_idx)
        if actual_cache_idx != expected_cache_idx:
            raise ValueError(
                "gemma4-shared-kv cache map disagrees with compact KV owner "
                f"{owner_idx} for decoder layer {layer_idx}"
            )
        layer_type = getattr(layers[layer_idx], "layer_type", None)
        owner_type = getattr(layers[owner_idx], "layer_type", None)
        if layer_type not in ("full_attention", "sliding_attention"):
            raise ValueError(
                f"Gemma 4 decoder layer {layer_idx} has unknown layer_type "
                f"{layer_type!r}"
            )
        if layer_type != owner_type:
            raise ValueError(
                "Gemma 4 shared-KV follower layer_type disagrees with its "
                f"owner: layer {layer_idx} is {layer_type!r}, owner {owner_idx} "
                f"is {owner_type!r}"
            )
        cache_entry = cache[expected_cache_idx]
        expected_type = KVCache if layer_type == "full_attention" else RotatingKVCache
        if type(cache_entry) is not expected_type:
            raise ValueError(
                f"Gemma 4 {layer_type} owner {owner_idx} requires "
                f"{expected_type.__name__} at compact cache index "
                f"{expected_cache_idx}, got {type(cache_entry).__name__}"
            )


def _nemotron_h_layer_to_cache_map(model) -> dict[int, int]:
    """Map only stateful Nemotron-H blocks into its compact cache list."""
    layer_to_cache: dict[int, int] = {}
    cache_idx = 0
    for layer_idx, layer in enumerate(model.layers):
        if getattr(layer, "block_type", None) in ("M", "*"):
            layer_to_cache[layer_idx] = cache_idx
            cache_idx += 1
    return layer_to_cache


STANDARD_QWEN_ADAPTER = SpecPrefillArchitectureAdapter(
    name="qwen-dense",
    model_types=("qwen2", "qwen2_moe", "qwen3", "qwen3_moe"),
    query_extractor=_qwen_extract_queries,
    cache_map_builder=_standard_layer_to_cache_map,
)
QWEN_HYBRID_ADAPTER = SpecPrefillArchitectureAdapter(
    name="qwen3.5-3.6-hybrid-moe",
    model_types=(
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_vl",
        "qwen3_vl_moe",
    ),
    query_extractor=_qwen35_extract_queries,
    cache_map_builder=_hybrid_layer_to_cache_map,
)
GEMMA4_ADAPTER = SpecPrefillArchitectureAdapter(
    name="gemma4-shared-kv",
    model_types=("gemma4", "gemma4_text"),
    query_extractor=_gemma4_extract_queries,
    cache_map_builder=_gemma4_shared_kv_cache_map,
)
NEMOTRON_H_ADAPTER = SpecPrefillArchitectureAdapter(
    name="nemotron-h",
    model_types=("nemotron_h",),
    query_extractor=_nemotron_h_extract_queries,
    cache_map_builder=_nemotron_h_layer_to_cache_map,
)
_ADAPTERS: tuple[SpecPrefillArchitectureAdapter, ...] = (
    STANDARD_QWEN_ADAPTER,
    QWEN_HYBRID_ADAPTER,
    GEMMA4_ADAPTER,
    NEMOTRON_H_ADAPTER,
)
ARCHITECTURE_ADAPTERS: dict[str, SpecPrefillArchitectureAdapter] = {
    model_type: adapter for adapter in _ADAPTERS for model_type in adapter.model_types
}


def resolve_specprefill_adapter(model: Any) -> SpecPrefillArchitectureAdapter:
    """Return the explicit scoring/cache contract for a supported model type."""
    model_type = _get_model_type(model)
    adapter = ARCHITECTURE_ADAPTERS.get(model_type)
    if adapter is None:
        supported = ", ".join(sorted(ARCHITECTURE_ADAPTERS))
        raise ValueError(
            f"Unsupported SpecPrefill model_type {model_type!r}; supported: {supported}"
        )
    return adapter


def _build_layer_to_cache_map(model):
    """Build mapping from model layer index to cache index.

    Standard models have one cache entry per layer. Gemma 4 shared-KV layers
    map to compact cache entries keyed by their physical owner. This
    compatibility helper intentionally keeps support for existing direct
    callers; new scoring uses the architecture adapter registry above.

    Nemotron-H: only M (Mamba2) and * (attention) layers have cache entries.
    MLP (-) and MoE (E) layers get no cache. The mapping is compacted.

    Returns dict {layer_idx: cache_idx}.
    """
    if (
        _get_model_type(model) in GEMMA4_ADAPTER.model_types
        or _has_gemma4_decoder_contract(model)
        or _has_gemma4_decoder_contract(getattr(model, "model", None))
        or (
            hasattr(model, "layers")
            and getattr(getattr(model, "model", None), "previous_kvs", None) is not None
        )
    ):
        return _gemma4_shared_kv_cache_map(model)

    has_block_type = any(hasattr(layer, "block_type") for layer in model.layers)
    if not has_block_type:
        return _standard_layer_to_cache_map(model)
    return _nemotron_h_layer_to_cache_map(model)


# ---------------------------------------------------------------------------
# Core API — sparse prefill
# ---------------------------------------------------------------------------


def sparse_prefill(
    model,
    tokens,
    selected_indices,
    cache,
    step_size=2048,
    position_offset=0,
    cancel_check=None,
):
    """Prefill the model cache with selected tokens at their original positions.

    Runs the model forward on only the selected tokens while preserving their
    original positional encoding via manual RoPE. After this call, the cache
    contains KV entries with correct RoPE positions, and attention layers have
    _OffsetAdjustedRoPE installed for correct decode positioning.

    Args:
        model: Language model with .layers property (TextModel or VLM Model)
        tokens: (M,) all prompt token IDs (mx.array or list)
        selected_indices: (N,) sorted indices into tokens to keep (mx.array or list)
        cache: list of KVCache/ArraysCache from make_prompt_cache()
        step_size: chunk size for processing (default 2048)
        position_offset: added to selected_indices for RoPE positions (default 0).
            Use when the cache already has tokens from a prior prefill (e.g.,
            system prompt KV cache with S tokens → position_offset=S).

    Returns:
        logits: (1, 1, vocab_size) from the last selected token

    Side effects:
        - Populates cache with KV for selected tokens
        - Installs _OffsetAdjustedRoPE on attention layers for decode
        - Call cleanup_rope(model) after generation to restore original RoPE
    """
    if not isinstance(tokens, mx.array):
        tokens = mx.array(tokens)
    if not isinstance(selected_indices, mx.array):
        selected_indices = mx.array(selected_indices)

    M = tokens.shape[0]

    # Detect RotatingKVCache and ensure tail tokens are included only when the
    # prompt actually exceeds the live cache window. If the full prompt still
    # fits inside ``max_size`` there is no eviction yet, so forcing the entire
    # tail back in would collapse sparse prefill into dense work.
    max_rotating_size = 0
    for c in cache:
        if type(c).__name__ == "RotatingKVCache":
            max_rotating_size = max(max_rotating_size, getattr(c, "max_size", 0))
    if max_rotating_size > 0 and M > max_rotating_size:
        tail_start = max(0, M - max_rotating_size)
        tail_indices = set(range(tail_start, M))
        existing = set(selected_indices.tolist())
        merged = sorted(existing | tail_indices)
        selected_indices = mx.array(merged)

    # RoPE positions: absolute positions accounting for any prefix
    selected_positions = selected_indices.astype(mx.int32) + position_offset
    selected_tokens = tokens[selected_indices]
    N = selected_tokens.shape[0]

    # Determine initial cache offset (non-zero when system KV cache is restored)
    attn_layers = _find_attention_layers(model)
    layer_to_cache = _build_layer_to_cache_map(model)
    first_attn_layer_idx = attn_layers[0][0]
    first_attn_cache_idx = layer_to_cache[first_attn_layer_idx]
    cache_start = (
        cache[first_attn_cache_idx].offset
        if hasattr(cache[first_attn_cache_idx], "offset")
        else 0
    )

    # Check if attention layers use RoPE (Nemotron-H has none)
    first_attn = _get_attn_module(attn_layers[0][1])
    has_rope = _get_rope(first_attn) is not None

    # Patch RoPE on attention layers for position-mapped prefill
    # (skipped for architectures without RoPE, e.g. Nemotron-H)
    original_ropes = {}
    if has_rope:
        for layer_idx, layer in attn_layers:
            attn = _get_attn_module(layer)
            rope = _get_rope(attn)
            original_ropes[layer_idx] = (attn, rope)
            _set_rope(
                attn,
                _PositionMappedRoPE(rope, selected_positions, cache_start=cache_start),
            )

    try:
        prompt = selected_tokens
        n = int(N)
        processed = 0

        while n - processed > 1:
            if cancel_check is not None:
                cancel_check()
            chunk = min(step_size, n - processed - 1)
            model(prompt[processed : processed + chunk][None], cache=cache)
            mx.eval([c.state for c in cache])
            processed += chunk
            mx.clear_cache()

        # Last token → logits
        if cancel_check is not None:
            cancel_check()
        logits = model(prompt[processed:][None], cache=cache)
        mx.eval(logits)

    finally:
        # Replace position-mapped RoPE with offset-adjusted RoPE for decode.
        # Skipped for architectures without RoPE (e.g. Nemotron-H).
        #
        # Total prompt length = position_offset + M (prefix + current tokens).
        # After prefill, cache offset = cache_start + N.
        # Decode needs RoPE position = total_len + i, cache gives offset = cache_start + N + i.
        # Adjustment = total_len - (cache_start + N) = position_offset + M - cache_start - N.
        # When cache_start == position_offset (normal case): adjustment = M - N.
        if has_rope:
            total_prompt_len = position_offset + M
            final_cache_offset = cache_start + N
            adjustment = int(total_prompt_len) - int(final_cache_offset)
            for layer_idx, layer in attn_layers:
                attn, original = original_ropes[layer_idx]
                if adjustment > 0:
                    _set_rope(attn, _OffsetAdjustedRoPE(original, adjustment))
                else:
                    _set_rope(attn, original)

    return logits


def cleanup_rope(model):
    """Restore original RoPE on all attention layers.

    Call this after generation is complete to remove _OffsetAdjustedRoPE
    wrappers installed by sparse_prefill(). No-op for architectures
    without RoPE (e.g. Nemotron-H).
    """
    for _, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if attn is None:
            continue
        rope = _get_rope(attn)
        if rope is None:
            continue
        if isinstance(rope, (_OffsetAdjustedRoPE, _PositionMappedRoPE)):
            _set_rope(attn, rope._original)
