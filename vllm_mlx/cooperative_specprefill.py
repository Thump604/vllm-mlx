# SPDX-License-Identifier: Apache-2.0
"""Cooperative SpecPrefill primitives for TextBatchScheduler.

This module converts the serial SpecPrefill pipeline into bounded scheduler-side
steps so long prompts no longer monopolize the GPU lock.  It also provides a
per-request cache wrapper that keeps sparse-prefill decode RoPE adjustments on
the cache object instead of the shared model, making the resulting prompt cache
safe to batch with other requests.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .specprefill import (
    _PositionMappedRoPE,
    _build_layer_to_cache_map,
    _compute_importance,
    _find_attention_layers,
    _gemma4_extract_queries,
    _get_attn_module,
    _llama_extract_queries,
    _make_draft_cache,
    _nemotron_h_extract_queries,
    _patch_attention_for_capture,
    _qwen35_extract_queries,
    _unpatch_attention_capture,
    select_chunks,
)


class RopeAdjustedCache:
    """Per-request cache wrapper with separate RoPE and storage offsets.

    Attention layers read ``cache.offset`` when applying RoPE, but the cache's
    own update/mask logic must continue using the underlying contiguous storage
    offset.  This wrapper exposes an adjusted ``offset`` for RoPE while
    delegating storage/mask operations to the wrapped cache.
    """

    def __init__(self, cache: Any, adjustment: int | mx.array = 0):
        self._cache = cache._cache if isinstance(cache, RopeAdjustedCache) else cache
        self._adjustment = self._normalize_adjustment(adjustment)

    @staticmethod
    def _normalize_adjustment(value: Any) -> int | mx.array:
        if isinstance(value, mx.array):
            return value
        if isinstance(value, (list, tuple)):
            return mx.array(list(value), dtype=mx.int32)
        return int(value)

    @property
    def offset(self):
        return self._cache.offset + self._adjustment

    @property
    def adjustment(self) -> int | mx.array:
        return self._adjustment

    @property
    def state(self):
        return self._cache.state

    @state.setter
    def state(self, value):
        self._cache.state = value

    @property
    def meta_state(self):
        return getattr(self._cache, "meta_state", None)

    @meta_state.setter
    def meta_state(self, value):
        setattr(self._cache, "meta_state", value)

    def update_and_fetch(self, keys, values):
        return self._cache.update_and_fetch(keys, values)

    def make_mask(self, *args, **kwargs):
        return self._cache.make_mask(*args, **kwargs)

    def size(self):
        if hasattr(self._cache, "size"):
            return self._cache.size()
        actual_offset = getattr(self._cache, "offset", 0)
        return int(actual_offset.item() if hasattr(actual_offset, "item") else actual_offset)

    def empty(self):
        if hasattr(self._cache, "empty"):
            return self._cache.empty()
        return False

    def is_trimmable(self):
        return self._cache.is_trimmable()

    def trim(self, n):
        return self._cache.trim(n)

    def prepare(self, *args, **kwargs):
        if hasattr(self._cache, "prepare"):
            return self._cache.prepare(*args, **kwargs)
        return None

    def finalize(self):
        if hasattr(self._cache, "finalize"):
            return self._cache.finalize()
        return None

    def filter(self, batch_indices):
        if hasattr(self._cache, "filter"):
            self._cache.filter(batch_indices)
        if isinstance(self._adjustment, mx.array):
            self._adjustment = self._adjustment[batch_indices]

    def extend(self, other):
        other_cache = other._cache if isinstance(other, RopeAdjustedCache) else other
        self._cache.extend(other_cache)

        other_adjustment = (
            other._adjustment if isinstance(other, RopeAdjustedCache) else 0
        )
        if isinstance(self._adjustment, mx.array) or isinstance(other_adjustment, mx.array):
            left = (
                self._adjustment
                if isinstance(self._adjustment, mx.array)
                else mx.array([int(self._adjustment)], dtype=mx.int32)
            )
            right = (
                other_adjustment
                if isinstance(other_adjustment, mx.array)
                else mx.array([int(other_adjustment)], dtype=mx.int32)
            )
            self._adjustment = mx.concatenate([left, right], axis=0)
        else:
            self._adjustment = mx.array(
                [int(self._adjustment), int(other_adjustment)],
                dtype=mx.int32,
            )

    def extract(self, idx):
        extracted = self._cache.extract(idx)
        if isinstance(self._adjustment, mx.array):
            adjustment = int(self._adjustment[idx].item())
        else:
            adjustment = int(self._adjustment)
        return RopeAdjustedCache(extracted, adjustment)

    @classmethod
    def merge(cls, caches):
        base_caches = [
            cache._cache if isinstance(cache, RopeAdjustedCache) else cache
            for cache in caches
        ]
        merged = base_caches[0].merge(base_caches)
        adjustments = []
        for cache in caches:
            if isinstance(cache, RopeAdjustedCache):
                adj = cache.adjustment
                if isinstance(adj, mx.array):
                    adjustments.extend(int(x) for x in adj.tolist())
                else:
                    adjustments.append(int(adj))
            else:
                adjustments.append(0)
        return cls(merged, mx.array(adjustments, dtype=mx.int32))

    @property
    def nbytes(self):
        return getattr(self._cache, "nbytes", 0)

    def __bool__(self):
        return self._cache is not None

    def __len__(self):
        try:
            return len(self._cache)
        except TypeError:
            return 1 if self._cache is not None else 0

    def __iter__(self):
        try:
            return iter(self._cache)
        except TypeError:
            return iter((self._cache,))

    def __getitem__(self, index):
        return self._cache[index]

    def __setitem__(self, index, value):
        self._cache[index] = value

    def __copy__(self):
        new_obj = type(self).__new__(type(self))
        new_obj._cache = self._cache
        new_obj._adjustment = self._adjustment
        return new_obj

    def __deepcopy__(self, memo):
        new_obj = type(self).__new__(type(self))
        memo[id(self)] = new_obj
        new_obj._cache = copy.deepcopy(self._cache, memo)
        new_obj._adjustment = copy.deepcopy(self._adjustment, memo)
        return new_obj

    def __getstate__(self):
        return {
            "cache": self._cache,
            "adjustment": self._adjustment,
        }

    def __setstate__(self, state):
        self._cache = state["cache"]
        self._adjustment = self._normalize_adjustment(state["adjustment"])

    def __getattr__(self, name):
        cache = self.__dict__.get("_cache")
        if cache is None:
            raise AttributeError(name)
        return getattr(cache, name)


class PreseededSequenceStateMachine:
    """Advance a stop-state machine with synthetic preseeded tokens."""

    def __init__(self, base_machine: Any, seeded_tokens: list[int]):
        self._base_machine = base_machine
        self._seeded_tokens = list(seeded_tokens)

    def make_state(self):
        state = self._base_machine.make_state()
        for token in self._seeded_tokens:
            state, _, _ = self._base_machine.match(state, token)
        return state

    def match(self, state, token):
        return self._base_machine.match(state, token)


@dataclass
class CooperativeSpecPrefillResult:
    logits: Any
    cache: list[Any]
    cache_token_count: int
    selected_token_count: int


def _detect_query_extractor(model: Any):
    attn_layers = _find_attention_layers(model)
    if not attn_layers:
        raise RuntimeError("Model has no attention layers for SpecPrefill scoring")
    attn_obj = _get_attn_module(attn_layers[0][1])
    if hasattr(attn_obj, "v_norm"):
        return _gemma4_extract_queries
    if hasattr(attn_obj, "q_norm"):
        return _qwen35_extract_queries
    if not hasattr(attn_obj, "rope"):
        return _nemotron_h_extract_queries
    return _llama_extract_queries


class ChunkedDraftScorer:
    """Incremental draft-model scorer for cooperative SpecPrefill."""

    def __init__(
        self,
        draft_model: Any,
        tokens: list[int],
        *,
        chunk_size: int = 2048,
        n_lookahead: int = 8,
        pool_kernel: int = 13,
        temp: float = 0.6,
        top_p: float = 0.95,
        query_extractor=None,
    ):
        self._draft_model = draft_model
        self._tokens = list(tokens)
        self._chunk_size = max(1, int(chunk_size))
        self._n_lookahead = max(1, int(n_lookahead))
        self._pool_kernel = pool_kernel
        self._temp = temp
        self._top_p = top_p
        self._query_extractor = query_extractor

        self._cache = None
        self._logits = None
        self._sampler = make_sampler(temp=self._temp, top_p=self._top_p)
        self._query_buffer = None
        self._patches = None
        self._attn_indices = None
        self._layer_to_cache = None
        self._n_attn_heads = None
        self._n_kv_heads = None
        self._next_input = None
        self._prefill_processed = 0
        self._lookahead_steps = 0
        self._done = False

    @property
    def is_scoring(self) -> bool:
        return not self._done

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def tokens_processed(self) -> int:
        return self._prefill_processed

    @property
    def chunks_remaining(self) -> int:
        if self._done:
            return 0
        prefill_remaining = max(len(self._tokens) - self._prefill_processed, 0)
        prefill_chunks = math.ceil(prefill_remaining / self._chunk_size) if prefill_remaining else 0
        lookahead_remaining = max(self._n_lookahead - self._lookahead_steps, 0)
        return prefill_chunks + lookahead_remaining

    def _ensure_initialized(self) -> None:
        if self._cache is not None:
            return

        self._cache = _make_draft_cache(self._draft_model)
        attn_layers = _find_attention_layers(self._draft_model)
        attn_obj = _get_attn_module(attn_layers[0][1])
        self._n_attn_heads = getattr(
            attn_obj,
            "num_attention_heads",
            getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
        )
        self._n_kv_heads = getattr(
            attn_obj,
            "num_key_value_heads",
            getattr(attn_obj, "n_kv_heads", None),
        )
        self._layer_to_cache = _build_layer_to_cache_map(self._draft_model)
        if self._query_extractor is None:
            self._query_extractor = _detect_query_extractor(self._draft_model)

    def step(self) -> bool:
        if self._done:
            return True

        self._ensure_initialized()
        prompt = mx.array(self._tokens)
        remaining = len(self._tokens) - self._prefill_processed

        if self._logits is None:
            if remaining > self._chunk_size + 1:
                end = self._prefill_processed + self._chunk_size
                self._draft_model(
                    prompt[self._prefill_processed:end][None],
                    cache=self._cache,
                )
                mx.eval([c.state for c in self._cache])
                mx.clear_cache()
                self._prefill_processed = end
                return False

            self._logits = self._draft_model(
                prompt[self._prefill_processed:][None],
                cache=self._cache,
            )
            self._prefill_processed = len(self._tokens)
            mx.eval(self._logits)

            attn_layers = _find_attention_layers(self._draft_model)
            self._query_buffer = [[] for _ in range(len(attn_layers))]
            self._patches, self._attn_indices = _patch_attention_for_capture(
                self._draft_model,
                self._query_buffer,
                query_extractor=self._query_extractor,
            )
            self._next_input = self._sampler(self._logits[:, -1, :])
            mx.eval(self._next_input)
            return False

        logits = self._draft_model(self._next_input.reshape(1, -1), cache=self._cache)
        self._next_input = self._sampler(logits[:, -1, :])
        self._lookahead_steps += 1
        mx.eval(self._next_input)

        if self._lookahead_steps >= self._n_lookahead:
            if self._patches is not None:
                _unpatch_attention_capture(self._draft_model, self._patches)
                self._patches = None
            self._done = True
        return self._done

    def finalize(self):
        if not self._done:
            raise RuntimeError("Cannot finalize ChunkedDraftScorer before scoring completes")

        attn_caches = [
            self._cache[self._layer_to_cache[layer_idx]] for layer_idx in self._attn_indices
        ]
        importance = _compute_importance(
            self._query_buffer,
            attn_caches,
            len(self._tokens),
            self._n_attn_heads,
            self._n_kv_heads,
            pool_kernel=self._pool_kernel if self._pool_kernel > 0 else None,
        )
        mx.eval(importance)
        return importance

    def cleanup(self) -> None:
        if self._patches is not None:
            _unpatch_attention_capture(self._draft_model, self._patches)
            self._patches = None
        self._cache = None
        self._logits = None
        self._next_input = None
        self._query_buffer = None
        mx.clear_cache()


class ChunkedSparsePrefiller:
    """Incremental sparse-prefill runner for cooperative SpecPrefill."""

    def __init__(
        self,
        model: Any,
        tokens: list[int],
        selected_indices: Any,
        cache: list[Any] | None = None,
        *,
        step_size: int = 2048,
        position_offset: int = 0,
    ):
        self._model = model
        self._tokens = list(tokens)
        if isinstance(selected_indices, mx.array):
            selected_indices = selected_indices.tolist()
        self._selected_indices = list(selected_indices)
        self._step_size = max(1, int(step_size))
        self._position_offset = int(position_offset)

        self._cache = cache if cache is not None else make_prompt_cache(model)
        self._selected_tokens = None
        self._selected_positions = None
        self._processed = 0
        self._cache_start = 0
        self._adjustment = 0
        self._wrapped_cache = None
        self._logits = None
        self._done = False
        self._attn_layers = _find_attention_layers(model)
        self._original_ropes = {}
        self._has_rope = bool(self._attn_layers) and hasattr(
            _get_attn_module(self._attn_layers[0][1]), "rope"
        )

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def selected_token_count(self) -> int:
        return 0 if self._selected_positions is None else int(self._selected_positions.shape[0])

    @property
    def cache_token_count(self) -> int:
        if self._wrapped_cache is None:
            return 0
        first = self._wrapped_cache[0]
        if hasattr(first, "size"):
            return int(first.size())
        actual_offset = getattr(first, "_cache", first).offset
        return int(actual_offset.item() if hasattr(actual_offset, "item") else actual_offset)

    def _ensure_initialized(self) -> None:
        if self._selected_tokens is not None:
            return

        tokens = mx.array(self._tokens)
        selected_indices = mx.array(self._selected_indices)
        total_prompt_tokens = tokens.shape[0]

        max_rotating_size = 0
        for cache in self._cache:
            base_cache = cache._cache if isinstance(cache, RopeAdjustedCache) else cache
            if type(base_cache).__name__ == "RotatingKVCache":
                max_rotating_size = max(
                    max_rotating_size,
                    getattr(base_cache, "max_size", 0),
                )
        if max_rotating_size > 0:
            tail_start = max(0, total_prompt_tokens - max_rotating_size)
            tail_indices = set(range(tail_start, total_prompt_tokens))
            selected_indices = mx.array(sorted(set(selected_indices.tolist()) | tail_indices))

        self._selected_positions = selected_indices.astype(mx.int32) + self._position_offset
        self._selected_tokens = tokens[selected_indices]

        layer_to_cache = _build_layer_to_cache_map(self._model)
        first_attn_layer_idx = self._attn_layers[0][0]
        first_attn_cache_idx = layer_to_cache[first_attn_layer_idx]
        base_cache = (
            self._cache[first_attn_cache_idx]._cache
            if isinstance(self._cache[first_attn_cache_idx], RopeAdjustedCache)
            else self._cache[first_attn_cache_idx]
        )
        self._cache_start = (
            base_cache.offset if hasattr(base_cache, "offset") else 0
        )

        if self._has_rope:
            for layer_idx, layer in self._attn_layers:
                attn = _get_attn_module(layer)
                self._original_ropes[layer_idx] = attn.rope
                attn.rope = _PositionMappedRoPE(
                    attn.rope,
                    self._selected_positions,
                    cache_start=int(
                        self._cache_start.item()
                        if hasattr(self._cache_start, "item")
                        else self._cache_start
                    ),
                )

    def _restore_original_ropes(self) -> None:
        if not self._original_ropes:
            return
        for layer_idx, layer in self._attn_layers:
            attn = _get_attn_module(layer)
            if layer_idx in self._original_ropes:
                attn.rope = self._original_ropes[layer_idx]
        self._original_ropes = {}

    def step(self) -> bool:
        if self._done:
            return True

        self._ensure_initialized()
        remaining = int(self._selected_tokens.shape[0]) - self._processed

        if remaining > self._step_size + 1:
            end = self._processed + self._step_size
            self._model(self._selected_tokens[self._processed:end][None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
            mx.clear_cache()
            self._processed = end
            return False

        self._logits = self._model(self._selected_tokens[self._processed:][None], cache=self._cache)
        mx.eval(self._logits)
        self._processed = int(self._selected_tokens.shape[0])

        total_prompt_len = self._position_offset + len(self._tokens)
        final_cache_offset = int(
            self._cache_start.item() if hasattr(self._cache_start, "item") else self._cache_start
        ) + int(self._selected_tokens.shape[0])
        self._adjustment = int(total_prompt_len) - int(final_cache_offset)

        self._restore_original_ropes()
        self._wrapped_cache = [
            RopeAdjustedCache(cache, self._adjustment) for cache in self._cache
        ]
        self._done = True
        return True

    def finalize(self) -> tuple[Any, list[Any]]:
        if not self._done:
            raise RuntimeError("Cannot finalize ChunkedSparsePrefiller before prefill completes")
        return self._logits, self._wrapped_cache

    def cleanup(self) -> None:
        self._restore_original_ropes()
        self._logits = None
        self._wrapped_cache = None
        mx.clear_cache()


class CooperativeSpecPrefillSession:
    """End-to-end cooperative SpecPrefill session for one request."""

    def __init__(
        self,
        *,
        model: Any,
        draft_model: Any,
        tokens: list[int],
        base_cache: list[Any] | None,
        position_offset: int,
        keep_pct: float,
        chunk_size: int = 2048,
    ):
        self._model = model
        self._draft_model = draft_model
        self._tokens = list(tokens)
        self._base_cache = base_cache
        self._position_offset = int(position_offset)
        self._keep_pct = keep_pct
        self._chunk_size = chunk_size

        self._scorer = ChunkedDraftScorer(
            draft_model=self._draft_model,
            tokens=self._tokens,
            chunk_size=self._chunk_size,
        )
        self._prefiller = None
        self._result = None
        self._phase = "score"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def is_done(self) -> bool:
        return self._phase == "done"

    def step(self) -> bool:
        if self._phase == "done":
            return True

        if self._phase == "score":
            if self._scorer.step():
                importance = self._scorer.finalize()
                selected = select_chunks(importance, keep_pct=self._keep_pct)
                self._scorer.cleanup()
                self._prefiller = ChunkedSparsePrefiller(
                    self._model,
                    self._tokens,
                    selected,
                    self._base_cache,
                    step_size=self._chunk_size,
                    position_offset=self._position_offset,
                )
                self._phase = "prefill"
            return False

        if self._phase == "prefill":
            if self._prefiller.step():
                logits, wrapped_cache = self._prefiller.finalize()
                self._result = CooperativeSpecPrefillResult(
                    logits=logits,
                    cache=wrapped_cache,
                    cache_token_count=self._prefiller.cache_token_count,
                    selected_token_count=self._prefiller.selected_token_count,
                )
                self._phase = "done"
                return True
            return False

        raise RuntimeError(f"Unknown cooperative SpecPrefill phase: {self._phase}")

    def finalize(self) -> CooperativeSpecPrefillResult:
        if self._result is None:
            raise RuntimeError("Cannot finalize cooperative SpecPrefill before completion")
        return self._result

    def cleanup(self) -> None:
        self._scorer.cleanup()
        if self._prefiller is not None:
            self._prefiller.cleanup()
