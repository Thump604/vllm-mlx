# SPDX-License-Identifier: Apache-2.0
"""Foundation scheduler for batched text generation in MLLM mode.

This scheduler is intentionally conservative:
- GPU work is serialized through a shared asyncio.Lock
- cancellation waits for in-flight worker threads before releasing the lock
- the existing serial text path remains the production default until feature
  parity (SpecPrefill, prompt-prefix reuse) is restored

The implementation focuses on correctness, observability, and scalability for
continuous batching of text-only traffic. It includes prompt-prefix reuse for
shared system/history prefixes, while leaving cooperative SpecPrefill as the
remaining major parity gap.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, List, Optional

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, SequenceStateMachine
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

from .cooperative_specprefill import (
    CooperativeSpecPrefillResult,
    CooperativeSpecPrefillSession,
    PreseededSequenceStateMachine,
)
from .engine.base import GenerationOutput
from .engine_wrapper import EngineWrapper
from .scheduler import _install_cache_callbacks, _install_mtp
from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

logger = logging.getLogger(__name__)

KV_BYTES_PER_TOKEN = 24_576
FRAGMENTATION_OVERHEAD = 0.15
DEFAULT_CACHE_MEMORY_MB = 12_288
DEFAULT_MAX_ACTIVE_TOKENS = 768_000
DEFAULT_PREFILL_BATCH_SIZE = 8
DEFAULT_COMPLETION_BATCH_SIZE = 32
DEFAULT_PREFILL_STEP_SIZE = 2_048
DEFAULT_IDLE_TIMEOUT_S = 60.0
DEFAULT_MAX_WALL_CLOCK_S = 600.0
DEFAULT_QUEUE_SIZE = 256
DEFAULT_PROMPT_CACHE_ENTRIES = 64


@dataclass
class RequestState:
    """Per-request state for the text batch scheduler."""

    request_id: str
    token_ids: list[int]
    max_tokens: int
    queue: asyncio.Queue
    segments: list[list[int]] = field(default_factory=list)
    prefix_boundary: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    response_format: Any = None
    logits_processors: Optional[List[Callable]] = None
    uid: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    admitted: bool = False
    deferred: bool = False
    finished: bool = False
    is_detached: bool = False
    created_at: float = field(default_factory=time.monotonic)
    last_consumed_at: float = field(default_factory=time.monotonic)
    finish_reason: Optional[str] = None
    active_token_cost: int = 0
    resident_token_cost: int = 0
    cached_tokens: int = 0
    cache_hit_type: Optional[str] = None
    prefix_cache_saved: bool = False
    prepared_segments: list[list[int]] = field(default_factory=list)
    prepared_all_tokens: list[int] = field(default_factory=list)
    prepared_cache: Any = None
    cooperative_specprefill: bool = False
    cooperative_specprefill_tokens: list[int] = field(default_factory=list)
    cooperative_specprefill_position_offset: int = 0
    cooperative_specprefill_session: Any = None

    def __post_init__(self) -> None:
        self.prompt_tokens = len(self.token_ids)
        if not self.segments:
            self.segments = [list(self.token_ids)]
        if not self.prepared_segments:
            self.prepared_segments = [list(seg) for seg in self.segments]
        self.active_token_cost = self.prompt_tokens
        self.resident_token_cost = self.prompt_tokens


class TextBatchScheduler:
    """Cooperative scheduler for batched text generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        gpu_lock: asyncio.Lock,
        stop_tokens: set[int],
        *,
        draft_model: Any = None,
        enable_mtp: bool = False,
        cache_memory_mb: int = DEFAULT_CACHE_MEMORY_MB,
        max_active_tokens: int = DEFAULT_MAX_ACTIVE_TOKENS,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT_S,
        max_wall_clock: float = DEFAULT_MAX_WALL_CLOCK_S,
        prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
        max_queue_size: int = DEFAULT_QUEUE_SIZE,
        prompt_cache_entries: int = DEFAULT_PROMPT_CACHE_ENTRIES,
        specprefill_threshold: int | None = None,
        specprefill_keep_pct: float | None = None,
        model_path: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._model_path = model_path
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)
        self._gpu_lock = gpu_lock
        self._base_stop_tokens = set(stop_tokens or set())
        self._draft_model = draft_model
        self._enable_mtp = bool(
            enable_mtp
            and hasattr(model, "mtp")
            and getattr(model, "mtp", None) is not None
        )
        self._cache_memory_limit_bytes = int(cache_memory_mb * 1024 * 1024)
        self._max_active_tokens = int(max_active_tokens)
        self._idle_timeout = float(idle_timeout)
        self._max_wall_clock = float(max_wall_clock)
        self._prefill_batch_size = int(prefill_batch_size)
        self._completion_batch_size = int(completion_batch_size)
        self._prefill_step_size = int(prefill_step_size)
        self._max_queue_size = int(max_queue_size)
        self._prompt_cache_entries = int(prompt_cache_entries)
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct

        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._work_event = asyncio.Event()

        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._deferred: deque[str] = deque()
        self._cooperative_specprefill: deque[str] = deque()
        self.requests: dict[str, RequestState] = {}
        self.request_id_to_uid: dict[str, int] = {}
        self.uid_to_request_id: dict[int, str] = {}
        self._detokenizer_pool: dict[str, Any] = {}

        self._batch_generator: Optional[BatchGenerator] = None
        self._engine: Optional[EngineWrapper] = None

        prefix_cache_memory_mb = max(1, cache_memory_mb // 4)
        self._prefix_cache = MemoryAwarePrefixCache(
            model,
            MemoryCacheConfig(
                max_memory_mb=prefix_cache_memory_mb,
                max_entries=self._prompt_cache_entries,
            ),
        )

        self._current_cache_bytes = 0
        self._active_token_count = 0

        self._num_requests_processed = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._num_specprefill_requests = 0
        self._num_specprefill_fallbacks = 0
        self._latency_samples: deque[float] = deque(maxlen=1024)
        self._last_error: Optional[str] = None

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return

        self._ensure_engine()
        self._last_error = None
        self._running = True
        self._processing_task = asyncio.create_task(self._loop())
        logger.info(
            "TextBatchScheduler started: mtp=%s prefill_step_size=%d",
            self._enable_mtp,
            self._prefill_step_size,
        )

    async def stop(self) -> None:
        """Stop the scheduler and clean up all requests."""
        self._running = False
        if self._processing_task is not None:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("TextBatchScheduler stop observed loop failure: %s", exc)
            self._processing_task = None

        for request_id in list(self.requests.keys()):
            self._eject_request_by_id(request_id, reason="scheduler_stopped")

        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._batch_generator = None
        self._prefix_cache.clear()
        self._work_event.clear()
        logger.info("TextBatchScheduler stopped")

    async def submit(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Submit a text-only chat request and stream outputs."""
        if not self._running:
            await self.start()

        stop = kwargs.pop("stop", None)
        stop_list = [stop] if isinstance(stop, str) else list(stop or [])
        stop_token_ids = kwargs.pop("stop_token_ids", None) or []
        top_k = int(kwargs.pop("top_k", 0) or 0)
        min_p = float(kwargs.pop("min_p", 0.0) or 0.0)
        presence_penalty = float(kwargs.pop("presence_penalty", 0.0) or 0.0)
        repetition_penalty = float(kwargs.pop("repetition_penalty", 1.0) or 1.0)
        frequency_penalty = float(kwargs.pop("frequency_penalty", 0.0) or 0.0)
        response_format = kwargs.pop("response_format", None)
        logits_processors = kwargs.pop("logits_processors", None)

        chat_template_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        token_ids = self._encode_text(prompt)
        prefix_boundary = self._compute_prefix_boundary(messages, tools, token_ids)

        request_id = f"text-batch-{uuid.uuid4().hex[:12]}"
        state = RequestState(
            request_id=request_id,
            token_ids=token_ids,
            max_tokens=max_tokens or 256,
            queue=asyncio.Queue(maxsize=self._max_queue_size),
            segments=self._segment_prompt_tokens(token_ids, prefix_boundary),
            prefix_boundary=prefix_boundary,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop_list,
            stop_token_ids=list(stop_token_ids),
            response_format=response_format,
            logits_processors=list(logits_processors) if logits_processors else None,
        )
        self.requests[request_id] = state

        await self._pending.put(request_id)
        self._work_event.set()

        try:
            while True:
                item = await state.queue.get()
                if item is None:
                    break
                state.last_consumed_at = time.monotonic()
                yield item
                if item.finished:
                    break
        finally:
            current = self.requests.get(request_id)
            if current is not None and not current.finished and not current.is_detached:
                self._eject_request_by_id(request_id, reason="client_disconnected")

    async def _loop(self) -> None:
        while self._running:
            try:
                work_done = False

                while True:
                    try:
                        request_id = self._pending.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    state = self.requests.get(request_id)
                    if state is None or state.finished or state.is_detached:
                        continue

                    self._prepare_request(state)
                    if self._admit(state):
                        self._start_request(state)
                    else:
                        state.deferred = True
                        self._deferred.append(request_id)
                    work_done = True

                if self._retry_deferred():
                    work_done = True

                self._reap_dead_requests()

                if self._engine is not None and self._engine.has_work():
                    if await self._step_engine():
                        work_done = True

                if await self._step_cooperative_specprefill():
                    work_done = True

                if not work_done:
                    self._work_event.clear()
                    try:
                        await asyncio.wait_for(self._work_event.wait(), timeout=0.001)
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("TextBatchScheduler loop failed")
                self._fail_open_requests(f"scheduler_failed: {exc}")
                self._reset_engine_state()
                self._running = False
                self._work_event.clear()
                return

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return

        sampler = make_sampler(temp=0.0)
        bg_kwargs = dict(
            model=self.model,
            max_tokens=256,
            stop_tokens=self._base_stop_tokens,
            sampler=sampler,
            prefill_batch_size=self._prefill_batch_size,
            completion_batch_size=self._completion_batch_size,
            prefill_step_size=self._prefill_step_size,
        )
        if (
            "prompt_progress_callback"
            in inspect.signature(BatchGenerator.__init__).parameters
        ):
            bg_kwargs["prompt_progress_callback"] = self._log_prefill_progress

        batch_generator = BatchGenerator(**bg_kwargs)
        _install_cache_callbacks(
            batch_generator,
            prompt_progress_save=self._save_prefix_cache,
            uid_to_request_id=self.uid_to_request_id,
            requests=self.requests,
        )

        if self._enable_mtp:
            _install_mtp(
                batch_generator,
                model=self.model,
                num_draft_tokens=1,
                optimistic=False,
                stop_token_ids=self._base_stop_tokens,
            )

        self._batch_generator = batch_generator
        self._engine = EngineWrapper(batch_generator)

    def _log_prefill_progress(self, progress_list: list[tuple[int, int, int]]) -> None:
        for uid, processed, total in progress_list:
            request_id = self.uid_to_request_id.get(uid, "?")
            logger.debug(
                "[text_batch_prefill] request=%s tokens=%d/%d",
                request_id,
                processed,
                total,
            )

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
        enable_thinking = enable_thinking_env.lower() in {"true", "1", "yes"}

        messages = json.loads(json.dumps(messages, default=str))
        if tools:
            tools = json.loads(json.dumps(tools, default=str))

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if isinstance(chat_template_kwargs, dict):
            template_kwargs.update(chat_template_kwargs)
        template_kwargs.setdefault("enable_thinking", enable_thinking)
        if tools:
            template_kwargs["tools"] = tools

        try:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as exc:
            logger.debug("TextBatchScheduler chat template fallback: %s", exc)
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _encode_text(self, text: str) -> list[int]:
        try:
            return list(self._actual_tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self._actual_tokenizer.encode(text))

    def _compute_prefix_boundary(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
        token_ids: list[int],
    ) -> int:
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None or last_user_idx == 0:
            return 0

        try:
            dummy_messages = list(messages)
            dummy_messages[last_user_idx] = {
                **messages[last_user_idx],
                "content": "XXXXXXXXXX",
            }
            dummy_prompt = self._apply_chat_template(dummy_messages, tools=tools)
            dummy_tokens = self._encode_text(dummy_prompt)

            lcp = 0
            for idx in range(min(len(token_ids), len(dummy_tokens))):
                if token_ids[idx] != dummy_tokens[idx]:
                    break
                lcp = idx + 1

            return lcp
        except Exception as exc:
            logger.debug("TextBatchScheduler prefix-boundary fallback: %s", exc)
            return 0

    def _segment_prompt_tokens(
        self,
        token_ids: list[int],
        prefix_boundary: int,
    ) -> list[list[int]]:
        if 0 < prefix_boundary < len(token_ids):
            return [
                list(token_ids[:prefix_boundary]),
                list(token_ids[prefix_boundary:]),
            ]
        return [list(token_ids)]

    def _get_actual_tokenizer(self, tokenizer: Any) -> Any:
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            return tokenizer
        if hasattr(tokenizer, "tokenizer"):
            return tokenizer.tokenizer
        return tokenizer

    def _estimate_kv_bytes(self, num_tokens: int) -> int:
        return int(num_tokens * KV_BYTES_PER_TOKEN * (1.0 + FRAGMENTATION_OVERHEAD))

    def _check_memory_budget(self, new_request_tokens: int) -> bool:
        return (
            self._current_cache_bytes + self._estimate_kv_bytes(new_request_tokens)
            <= self._cache_memory_limit_bytes
        )

    def _check_compute_budget(self, new_request_tokens: int) -> bool:
        return (
            self._active_token_count + new_request_tokens
        ) <= self._max_active_tokens

    def _admit(self, state: RequestState) -> bool:
        return self._check_memory_budget(
            state.resident_token_cost
        ) and self._check_compute_budget(state.active_token_cost)

    def _should_use_cooperative_specprefill(
        self,
        state: RequestState,
        uncached_tokens: list[int],
    ) -> bool:
        if self._draft_model is None:
            return False
        if self._specprefill_threshold is None or self._specprefill_keep_pct is None:
            return False
        if len(uncached_tokens) <= int(self._specprefill_threshold):
            return False
        if state.prefix_boundary > 0 and state.cached_tokens < state.prefix_boundary:
            return False
        return True

    def _prepare_request(self, state: RequestState) -> None:
        cache_to_use = None
        cached_tokens = 0
        remaining_tokens = list(state.token_ids)
        state.cache_hit_type = "miss"

        if state.prefix_boundary > 0:
            cache_to_use, remaining_tokens = self._prefix_cache.fetch(state.token_ids)
            state.cache_hit_type = self._prefix_cache._last_match_type
            if cache_to_use is not None and remaining_tokens:
                cached_tokens = len(state.token_ids) - len(remaining_tokens)
            else:
                cache_to_use = None
                remaining_tokens = list(state.token_ids)

        if cache_to_use is not None and not self._cache_supports_batched_history(
            cache_to_use
        ):
            logger.warning(
                "TextBatchScheduler prefix cache for %s uses non-batchable history cache; "
                "falling back to uncached prompt admission",
                state.request_id,
            )
            cache_to_use = None
            cached_tokens = 0
            remaining_tokens = list(state.token_ids)
            state.cache_hit_type = "unsupported_history_cache"

        state.cached_tokens = cached_tokens
        state.prefix_cache_saved = cached_tokens >= state.prefix_boundary > 0
        state.prepared_cache = cache_to_use
        state.prepared_all_tokens = (
            list(state.token_ids[:cached_tokens]) if cached_tokens > 0 else []
        )
        state.prepared_segments = self._build_insert_segments(
            state.token_ids,
            state.prefix_boundary,
            cached_tokens,
        )
        state.cooperative_specprefill = self._should_use_cooperative_specprefill(
            state,
            remaining_tokens,
        )
        if state.cooperative_specprefill:
            state.cooperative_specprefill_tokens = list(remaining_tokens)
            state.cooperative_specprefill_position_offset = cached_tokens
            state.active_token_cost = 1
        else:
            state.cooperative_specprefill_tokens = []
            state.cooperative_specprefill_position_offset = 0
            state.active_token_cost = sum(
                len(segment) for segment in state.prepared_segments
            )
        state.resident_token_cost = state.prompt_tokens

    def _build_insert_segments(
        self,
        token_ids: list[int],
        prefix_boundary: int,
        cached_tokens: int,
    ) -> list[list[int]]:
        if cached_tokens <= 0:
            return self._segment_prompt_tokens(token_ids, prefix_boundary)

        segments: list[list[int]] = []
        if cached_tokens < prefix_boundary:
            segments.append(list(token_ids[cached_tokens:prefix_boundary]))
            if prefix_boundary < len(token_ids):
                segments.append(list(token_ids[prefix_boundary:]))
        elif cached_tokens < len(token_ids):
            segments.append(list(token_ids[cached_tokens:]))

        return [segment for segment in segments if segment]

    @staticmethod
    def _cache_supports_batched_history(cache_state: Any) -> bool:
        if not cache_state:
            return True
        return all(
            hasattr(layer, "merge") for layer in cache_state if layer is not None
        )

    def _build_sampler(self, state: RequestState):
        return make_sampler(
            temp=state.temperature,
            top_p=state.top_p,
            min_p=state.min_p,
            top_k=state.top_k,
        )

    def _build_logits_processors(self, state: RequestState) -> list[Any]:
        processors = make_logits_processors(
            repetition_penalty=(
                state.repetition_penalty if state.repetition_penalty != 1.0 else None
            ),
            presence_penalty=(
                state.presence_penalty if state.presence_penalty != 0.0 else None
            ),
            frequency_penalty=(
                state.frequency_penalty if state.frequency_penalty != 0.0 else None
            ),
        )
        if state.logits_processors:
            processors = (processors or []) + list(state.logits_processors)
        return processors

    def _build_state_machine(
        self, state: RequestState
    ) -> Optional[SequenceStateMachine]:
        if not state.stop:
            return None

        transitions: list[tuple[list[int], None]] = []
        for stop_text in state.stop:
            token_ids = self._encode_text(stop_text)
            if token_ids:
                transitions.append((token_ids, None))

        if not transitions:
            return None

        return SequenceStateMachine(
            transitions={"normal": transitions}, initial="normal"
        )

    def _begin_admitted_request(self, state: RequestState) -> None:
        if state.admitted:
            return
        state.admitted = True
        state.deferred = False
        self._active_token_count += state.active_token_cost
        self._current_cache_bytes += self._estimate_kv_bytes(state.resident_token_cost)
        self._total_prompt_tokens += state.prompt_tokens

    def _insert_prepared_request(
        self,
        state: RequestState,
        *,
        state_machine: Optional[SequenceStateMachine],
        already_admitted: bool = False,
        allow_cache_retry: bool = True,
    ) -> tuple[bool, Optional[str]]:
        self._ensure_engine()
        assert self._engine is not None

        sampler = self._build_sampler(state)
        logits_processors = self._build_logits_processors(state)
        stop_tokens = set(self._base_stop_tokens)
        stop_tokens.update(state.stop_token_ids)
        if stop_tokens != self._base_stop_tokens and self._batch_generator is not None:
            logger.debug(
                "TextBatchScheduler request %s adds stop_token_ids=%s; "
                "per-request token stop handling depends on underlying BatchGenerator state machine support",
                state.request_id,
                state.stop_token_ids,
            )

        try:
            uid = self._engine.insert_segments(
                [state.prepared_segments],
                max_tokens=[state.max_tokens],
                caches=(
                    [state.prepared_cache] if state.prepared_cache is not None else None
                ),
                all_tokens=[list(state.prepared_all_tokens)],
                samplers=[sampler],
                logits_processors=[logits_processors] if logits_processors else None,
                state_machines=[state_machine] if state_machine is not None else None,
            )[0]
        except Exception as exc:
            if allow_cache_retry and state.prepared_cache is not None:
                logger.warning(
                    "TextBatchScheduler cache insert failed for %s, retrying without cache: %s",
                    state.request_id,
                    exc,
                )
                state.prepared_cache = None
                state.cached_tokens = 0
                state.prefix_cache_saved = False
                state.prepared_all_tokens = []
                state.prepared_segments = self._segment_prompt_tokens(
                    state.token_ids,
                    state.prefix_boundary,
                )
                state.active_token_cost = sum(
                    len(segment) for segment in state.prepared_segments
                )
                state.resident_token_cost = state.prompt_tokens
                try:
                    uid = self._engine.insert_segments(
                        [state.prepared_segments],
                        max_tokens=[state.max_tokens],
                        all_tokens=[list(state.prepared_all_tokens)],
                        samplers=[sampler],
                        logits_processors=(
                            [logits_processors] if logits_processors else None
                        ),
                        state_machines=(
                            [state_machine] if state_machine is not None else None
                        ),
                    )[0]
                except Exception as retry_exc:
                    logger.exception(
                        "TextBatchScheduler failed to start request %s",
                        state.request_id,
                    )
                    return False, str(retry_exc)
            else:
                logger.exception(
                    "TextBatchScheduler failed to start request %s", state.request_id
                )
                return False, str(exc)

        if not already_admitted:
            self._begin_admitted_request(state)
        state.uid = uid
        state.deferred = False
        self.request_id_to_uid[state.request_id] = uid
        self.uid_to_request_id[uid] = state.request_id
        return True, None

    def _start_cooperative_specprefill(self, state: RequestState) -> None:
        try:
            state.cooperative_specprefill_session = CooperativeSpecPrefillSession(
                model=self.model,
                draft_model=self._draft_model,
                tokens=state.cooperative_specprefill_tokens,
                base_cache=state.prepared_cache,
                position_offset=state.cooperative_specprefill_position_offset,
                keep_pct=float(self._specprefill_keep_pct),
                chunk_size=self._prefill_step_size,
            )
        except Exception as exc:
            logger.warning(
                "TextBatchScheduler failed to initialize cooperative SpecPrefill for %s: %s",
                state.request_id,
                exc,
            )
            self._num_specprefill_fallbacks += 1
            state.cooperative_specprefill = False
            state.cooperative_specprefill_tokens = []
            state.cooperative_specprefill_position_offset = 0
            ok, error = self._insert_prepared_request(
                state,
                state_machine=self._build_state_machine(state),
            )
            if not ok:
                self._finish_state_with_error(state, f"insert_failed: {error}")
            return

        self._begin_admitted_request(state)
        self._cooperative_specprefill.append(state.request_id)
        self._num_specprefill_requests += 1
        self._work_event.set()

    def _start_request(self, state: RequestState) -> None:
        if state.cooperative_specprefill and state.cooperative_specprefill_tokens:
            self._start_cooperative_specprefill(state)
            return

        ok, error = self._insert_prepared_request(
            state,
            state_machine=self._build_state_machine(state),
        )
        if not ok:
            self._finish_state_with_error(state, f"insert_failed: {error}")

    def _retry_deferred(self) -> bool:
        if not self._deferred:
            return False

        work_done = False
        remaining: deque[str] = deque()
        while self._deferred:
            request_id = self._deferred.popleft()
            state = self.requests.get(request_id)
            if state is None or state.finished or state.is_detached:
                continue

            self._prepare_request(state)
            if self._admit(state):
                state.deferred = False
                self._start_request(state)
                work_done = True
            else:
                remaining.append(request_id)

        self._deferred = remaining
        return work_done

    async def _step_engine(self) -> bool:
        assert self._engine is not None

        async with self._gpu_lock:
            worker = asyncio.ensure_future(asyncio.to_thread(self._engine.step))
            try:
                prompt_responses, generation_responses = await asyncio.shield(worker)
            except asyncio.CancelledError:
                try:
                    await worker
                except Exception:
                    pass
                raise

        self._dispatch_responses(generation_responses)
        return bool(prompt_responses or generation_responses)

    def _remove_cooperative_specprefill(self, request_id: str) -> None:
        if not self._cooperative_specprefill:
            return
        self._cooperative_specprefill = deque(
            queued_id
            for queued_id in self._cooperative_specprefill
            if queued_id != request_id
        )

    def _sample_first_token(
        self,
        state: RequestState,
        result: CooperativeSpecPrefillResult,
    ) -> int:
        sample_logits = result.logits[:, -1, :]
        for processor in self._build_logits_processors(state):
            sample_logits = processor(state.token_ids, sample_logits)
        token = self._build_sampler(state)(sample_logits)
        mx.eval(token)
        return int(token.item())

    def _first_token_finish_reason(
        self,
        state: RequestState,
        token: int,
        state_machine: Optional[SequenceStateMachine],
    ) -> tuple[Optional[str], Optional[SequenceStateMachine]]:
        finish_reason = "length" if state.max_tokens <= 1 else None
        if token in self._base_stop_tokens or token in state.stop_token_ids:
            finish_reason = "stop"

        if state_machine is not None:
            _, matched_sequence, current_state = state_machine.match(
                state_machine.make_state(),
                token,
            )
            if matched_sequence is not None and current_state is None:
                finish_reason = "stop"
            state_machine = PreseededSequenceStateMachine(state_machine, [token])

        return finish_reason, state_machine

    def _fallback_from_cooperative_specprefill(
        self, state: RequestState, reason: str
    ) -> None:
        session = state.cooperative_specprefill_session
        if session is not None:
            try:
                session.cleanup()
            except Exception:
                logger.debug(
                    "Ignoring cooperative SpecPrefill cleanup failure for %s",
                    state.request_id,
                )

        self._remove_cooperative_specprefill(state.request_id)
        state.cooperative_specprefill_session = None
        state.cooperative_specprefill = False
        state.cooperative_specprefill_tokens = []
        state.cooperative_specprefill_position_offset = 0
        state.prepared_cache = None
        state.cached_tokens = 0
        state.prefix_cache_saved = False
        state.prepared_all_tokens = []
        state.prepared_segments = self._segment_prompt_tokens(
            state.token_ids,
            state.prefix_boundary,
        )
        state.active_token_cost = sum(
            len(segment) for segment in state.prepared_segments
        )
        state.resident_token_cost = state.prompt_tokens
        self._num_specprefill_fallbacks += 1
        logger.warning(
            "TextBatchScheduler falling back from cooperative SpecPrefill for %s: %s",
            state.request_id,
            reason,
        )

        if state.admitted and state.active_token_cost > 1:
            self._active_token_count += state.active_token_cost - 1

        ok, error = self._insert_prepared_request(
            state,
            state_machine=self._build_state_machine(state),
            already_admitted=state.admitted,
        )
        if not ok:
            self._finish_state_with_error(state, f"insert_failed: {error}")

    def _complete_cooperative_specprefill(
        self,
        state: RequestState,
        result: CooperativeSpecPrefillResult,
        first_token: int,
    ) -> None:
        session = state.cooperative_specprefill_session
        if session is not None:
            try:
                session.cleanup()
            except Exception:
                logger.debug(
                    "Ignoring cooperative SpecPrefill cleanup failure for %s",
                    state.request_id,
                )
        state.cooperative_specprefill_session = None
        state.cooperative_specprefill = False
        state.cooperative_specprefill_tokens = []
        state.cooperative_specprefill_position_offset = 0
        self._remove_cooperative_specprefill(state.request_id)

        state_machine = self._build_state_machine(state)
        finish_reason, seeded_state_machine = self._first_token_finish_reason(
            state,
            first_token,
            state_machine,
        )

        if finish_reason is not None:
            self._emit_token_output(state, first_token, finish_reason)
            return

        state.prepared_cache = result.cache
        state.prepared_all_tokens = list(state.token_ids)
        state.prepared_segments = [[first_token]]
        state.active_token_cost = 1

        ok, error = self._insert_prepared_request(
            state,
            state_machine=seeded_state_machine,
            already_admitted=True,
            allow_cache_retry=False,
        )
        if not ok:
            self._fallback_from_cooperative_specprefill(
                state,
                f"seeded_insert_failed: {error}",
            )
            return

        self._emit_token_output(state, first_token, None)

    async def _step_cooperative_specprefill(self) -> bool:
        while self._cooperative_specprefill:
            request_id = self._cooperative_specprefill.popleft()
            state = self.requests.get(request_id)
            if state is None or state.finished or state.is_detached:
                continue

            session = state.cooperative_specprefill_session
            if session is None:
                continue

            async with self._gpu_lock:
                worker = asyncio.ensure_future(asyncio.to_thread(session.step))
                try:
                    completed = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    try:
                        await worker
                    except Exception:
                        pass
                    raise
                except Exception as exc:
                    self._fallback_from_cooperative_specprefill(
                        state,
                        f"step_failed: {exc}",
                    )
                    return True

                result = None
                first_token = None
                if completed:
                    try:
                        result = session.finalize()
                        first_token = self._sample_first_token(state, result)
                    except Exception as exc:
                        self._fallback_from_cooperative_specprefill(
                            state,
                            f"finalize_failed: {exc}",
                        )
                        return True

            if completed and result is not None and first_token is not None:
                self._complete_cooperative_specprefill(state, result, first_token)
            else:
                self._cooperative_specprefill.append(request_id)

            return True

        return False

    def _emit_token_output(
        self,
        state: RequestState,
        token: int,
        finish_reason: Optional[str],
    ) -> None:
        request_id = state.request_id
        state.output_token_ids.append(token)
        state.completion_tokens = len(state.output_token_ids)

        detok = self._detokenizer_pool.get(request_id)
        if detok is None:
            detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
            detok.reset()
            self._detokenizer_pool[request_id] = detok

        if finish_reason == "stop":
            new_text = ""
        else:
            detok.add_token(token)
            new_text = detok.last_segment
            state.output_text += new_text

        output = GenerationOutput(
            text=state.output_text,
            tokens=list(state.output_token_ids),
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            finish_reason=None,
            new_text=new_text,
            finished=False,
        )

        if finish_reason is not None:
            output.finished = True
            output.finish_reason = finish_reason
            state.finish_reason = finish_reason
            state.finished = True

            final_detok = self._detokenizer_pool.pop(request_id, None)
            if final_detok is not None:
                final_detok.finalize()
                state.output_text = final_detok.text
            else:
                state.output_text = self._actual_tokenizer.decode(
                    state.output_token_ids
                )
            output.text = state.output_text

            self._latency_samples.append((time.monotonic() - state.created_at) * 1000.0)
            self._total_completion_tokens += state.completion_tokens
            self._num_requests_processed += 1

        try:
            state.queue.put_nowait(output)
        except asyncio.QueueFull:
            self._eject_request_by_id(request_id, reason="queue_full")
            return

        if state.finished:
            self._cleanup_finished_request(state)

    def _dispatch_responses(self, responses: list[Any]) -> None:
        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            state = self.requests.get(request_id)
            if state is None or state.finished or state.is_detached:
                continue

            self._emit_token_output(state, response.token, response.finish_reason)

    def _finish_state_with_error(self, state: RequestState, reason: str) -> None:
        output = GenerationOutput(
            text=state.output_text,
            tokens=list(state.output_token_ids),
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            finish_reason="error",
            new_text="",
            finished=True,
        )
        state.finished = True
        state.finish_reason = reason
        self._offer_final_output(state, output)
        self._cleanup_finished_request(state)

    def _fail_open_requests(self, reason: str) -> None:
        for state in list(self.requests.values()):
            if state.finished:
                continue
            self._finish_state_with_error(state, reason)

    def _reset_engine_state(self) -> None:
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception as exc:
                logger.debug(
                    "Ignoring EngineWrapper close failure after loop error: %s", exc
                )
        self._engine = None
        self._batch_generator = None
        self._cooperative_specprefill.clear()
        self._active_token_count = 0
        self._current_cache_bytes = 0

    def _offer_final_output(
        self, state: RequestState, output: GenerationOutput
    ) -> None:
        try:
            state.queue.put_nowait(output)
            return
        except asyncio.QueueFull:
            pass

        try:
            state.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        try:
            state.queue.put_nowait(output)
        except asyncio.QueueFull:
            pass

    def _cleanup_finished_request(self, state: RequestState) -> None:
        self._remove_cooperative_specprefill(state.request_id)
        if state.cooperative_specprefill_session is not None:
            try:
                state.cooperative_specprefill_session.cleanup()
            except Exception:
                logger.debug(
                    "Ignoring cooperative SpecPrefill cleanup failure for %s",
                    state.request_id,
                )
            state.cooperative_specprefill_session = None

        if state.admitted:
            self._active_token_count = max(
                0, self._active_token_count - state.active_token_cost
            )
            self._current_cache_bytes = max(
                0,
                self._current_cache_bytes
                - self._estimate_kv_bytes(state.resident_token_cost),
            )
            state.admitted = False

        if state.uid is not None:
            self.uid_to_request_id.pop(state.uid, None)
            self.request_id_to_uid.pop(state.request_id, None)
            state.uid = None

        state.cooperative_specprefill = False
        state.cooperative_specprefill_tokens = []
        state.cooperative_specprefill_position_offset = 0
        self._detokenizer_pool.pop(state.request_id, None)
        self.requests.pop(state.request_id, None)

    def _eject_request(self, uid: int, *, reason: str) -> None:
        request_id = self.uid_to_request_id.get(uid)
        if request_id is not None:
            self._eject_request_by_id(request_id, reason=reason)

    def _eject_request_by_id(self, request_id: str, *, reason: str) -> None:
        state = self.requests.get(request_id)
        if state is None or state.finished:
            return

        state.is_detached = True
        finish_reason = {
            "queue_full": "client_disconnected",
            "client_disconnected": "client_disconnected",
            "timeout": "timeout",
            "scheduler_stopped": "abort",
        }.get(reason, reason)

        if state.uid is not None and self._engine is not None:
            try:
                self._engine.remove([state.uid])
            except Exception:
                logger.debug(
                    "Ignoring remove failure for detached request %s", request_id
                )

        output = GenerationOutput(
            text=state.output_text,
            tokens=list(state.output_token_ids),
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            finish_reason=finish_reason,
            new_text="",
            finished=True,
        )
        self._offer_final_output(state, output)
        state.finished = True
        state.finish_reason = finish_reason
        self._cleanup_finished_request(state)

    def _reap_dead_requests(self) -> None:
        now = time.monotonic()
        for request_id, state in list(self.requests.items()):
            if state.finished:
                continue
            idle = now - state.last_consumed_at
            wall = now - state.created_at
            if idle > self._idle_timeout or wall > self._max_wall_clock:
                self._eject_request_by_id(request_id, reason="timeout")

    def _save_prompt_cache(self, uid: int, extracted_cache: Any) -> None:
        del uid, extracted_cache

    def _save_prefix_cache(
        self,
        uid: int,
        processed_tokens: int,
        extract_cache,
        end_of_segment: bool,
        end_of_prompt: bool,
    ) -> None:
        request_id = self.uid_to_request_id.get(uid)
        if request_id is None:
            return

        state = self.requests.get(request_id)
        if state is None or state.prefix_boundary <= 0 or state.prefix_cache_saved:
            return

        total_processed = state.cached_tokens + processed_tokens
        if not end_of_segment or total_processed != state.prefix_boundary:
            return
        if extract_cache is None:
            return

        try:
            extracted_cache = extract_cache()
        except Exception as exc:
            logger.debug("TextBatchScheduler prefix cache extract failed: %s", exc)
            return

        if not extracted_cache:
            return

        prefix_tokens = list(state.token_ids[: state.prefix_boundary])
        stored = self._prefix_cache.store(
            prefix_tokens,
            extracted_cache,
            evict_prefixes=False,
        )
        if stored:
            state.prefix_cache_saved = True
            logger.info(
                "[text_prefix_cache] request=%s tokens=%d",
                state.request_id,
                state.prefix_boundary,
            )

    @staticmethod
    def _safe_len(value: Any) -> int:
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            pass

        uids = getattr(value, "uids", None)
        if uids is not None:
            try:
                return len(uids)
            except TypeError:
                pass

        return 0

    def _batch_state(self) -> dict[str, int]:
        if self._batch_generator is None:
            return {
                "queued_sequences": 0,
                "currently_processing": 0,
                "prompt_batch_size": 0,
                "generation_batch_size": 0,
            }

        return {
            "queued_sequences": self._safe_len(
                getattr(self._batch_generator, "_unprocessed_sequences", None)
            ),
            "currently_processing": self._safe_len(
                getattr(self._batch_generator, "_currently_processing", None)
            ),
            "prompt_batch_size": self._safe_len(
                getattr(self._batch_generator, "_prompt_batch", None)
            ),
            "generation_batch_size": self._safe_len(
                getattr(self._batch_generator, "_generation_batch", None)
            ),
        }

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "running": self._running,
            "active_requests": sum(
                1
                for state in self.requests.values()
                if state.admitted and not state.finished
            ),
            "pending_requests": sum(
                1
                for state in self.requests.values()
                if not state.admitted and not state.deferred and not state.finished
            ),
            "deferred_requests": sum(
                1
                for state in self.requests.values()
                if state.deferred and not state.finished
            ),
            "active_token_count": self._active_token_count,
            "max_active_tokens": self._max_active_tokens,
            "cache_memory_mb": round(self._cache_memory_limit_bytes / (1024 * 1024), 1),
            "cache_bytes_in_use": self._current_cache_bytes,
            "enable_mtp": self._enable_mtp,
            "num_requests_processed": self._num_requests_processed,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "num_specprefill_requests": self._num_specprefill_requests,
            "num_specprefill_fallbacks": self._num_specprefill_fallbacks,
            "request_queue_depth": self._pending.qsize(),
            "deferred_queue_depth": len(self._deferred),
            "cooperative_specprefill_queue_depth": len(self._cooperative_specprefill),
            "stored_prompt_caches": len(self._prefix_cache),
            "prompt_prefix_reuse_foundation_ready": True,
            "system_prefix_reuse_foundation_ready": True,
            "tool_calling_foundation_ready": True,
            "specprefill_foundation_ready": True,
            "specprefill_cutover_ready": False,
            "batch_state": self._batch_state(),
        }
        if self._last_error:
            stats["last_error"] = self._last_error

        if self._engine is not None:
            try:
                stats["engine_has_work"] = self._engine.has_work()
            except Exception as exc:
                stats["engine_has_work_error"] = str(exc)

        p50, p95, p99 = self._latency_percentiles()
        stats["latency_p50_ms"] = p50
        stats["latency_p95_ms"] = p95
        stats["latency_p99_ms"] = p99
        stats["prefix_cache"] = self._prefix_cache.get_stats()

        return stats

    def _latency_percentiles(self) -> tuple[float, float, float]:
        if not self._latency_samples:
            return (0.0, 0.0, 0.0)

        values = sorted(self._latency_samples)

        def _percentile(p: float) -> float:
            if len(values) == 1:
                return float(values[0])
            idx = max(0, min(len(values) - 1, math.ceil((p / 100.0) * len(values)) - 1))
            return float(values[idx])

        return (_percentile(50), _percentile(95), _percentile(99))
