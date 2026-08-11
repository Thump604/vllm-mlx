# SPDX-License-Identifier: Apache-2.0
"""Bounded, request-local SpecPrefill draft-scoring sessions.

The scorer model keeps its topology-preserving attention dispatch installed
once, but no request capture session survives a yielded quantum.  A scheduler
can therefore alternate sessions on one scorer model: each session owns its
draft cache, captured-query buffers, sampler/RNG closure, and importance
reducer while the scorer only owns the short active-forward capture context.

Importance is reduced one attention layer per quantum.  That avoids the old
all-layer ``concatenate`` and one giant realization.  A single layer still
forms ``lookahead × prompt`` attention scores and must be realized together;
the scheduler treats that bounded layer reduction as a scorer-lane quantum.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import apply_top_p

from .specprefill import SpecPrefillScorer, SpecPrefillScorerLaneBusy, _avg_pool1d

_MX_ARRAY_TYPE = type(mx.array([0]))


class SpecPrefillScorerSessionError(RuntimeError):
    """A bounded scorer session cannot safely advance."""


class ScorerSessionPhase(str, Enum):
    PREFILL = "prefill"
    LOOKAHEAD = "lookahead"
    IMPORTANCE = "importance"
    COMPLETE = "complete"
    CLOSED = "closed"


@dataclass(frozen=True)
class ScorerSessionProgress:
    """One completed scheduler quantum, suitable for CB scorer-lane telemetry."""

    phase: ScorerSessionPhase
    prefill_tokens: int
    lookahead_steps: int
    importance_layers: int
    complete: bool


class SpecPrefillScorerSession:
    """A request-owned, bounded scorer pipeline.

    ``step`` never retains an active scorer capture context when it returns.
    It raises ``SpecPrefillScorerSessionError`` when another request currently
    owns the scorer lane, leaving this session unchanged for scheduler retry.
    """

    def __init__(
        self,
        scorer: SpecPrefillScorer,
        tokens: Sequence[int] | mx.array,
        *,
        n_lookahead: int = 8,
        pool_kernel: int = 13,
        temp: float = 0.6,
        top_p: float = 0.95,
        prefill_step_size: int = 2048,
        query_extractor: Callable[..., mx.array] | None = None,
        cancel_check: Callable[[], None] | None = None,
        sampling_seed: int | None = None,
    ):
        if not isinstance(scorer, SpecPrefillScorer):
            raise TypeError("scorer must be a SpecPrefillScorer")
        if isinstance(tokens, _MX_ARRAY_TYPE):
            # Token IDs normally originate on the host tokenizer path.  An
            # array input requires this one explicit conversion at admission;
            # no per-quantum host/device token synchronization occurs.
            tokens = tokens.tolist()
        self._tokens = tuple(_validate_token(token) for token in tokens)
        if not self._tokens:
            raise ValueError("SpecPrefill scorer session needs at least one token")
        if (
            isinstance(n_lookahead, bool)
            or not isinstance(n_lookahead, int)
            or n_lookahead <= 0
        ):
            raise ValueError("n_lookahead must be a positive integer")
        if (
            isinstance(prefill_step_size, bool)
            or not isinstance(prefill_step_size, int)
            or prefill_step_size <= 0
        ):
            raise ValueError("prefill_step_size must be a positive integer")

        self.scorer = scorer
        self.cache = make_prompt_cache(scorer.model)
        self._layer_to_cache = scorer.adapter.cache_map_builder(scorer.model)
        try:
            for layer_index in scorer.attention_layer_indices:
                self._layer_to_cache[layer_index]
        except KeyError as exc:
            raise SpecPrefillScorerSessionError(
                "scorer cache map has no entry for an installed attention layer"
            ) from exc
        self._query_extractor = query_extractor or scorer.adapter.query_extractor
        self._sampler = _RequestLocalSampler(
            temp=temp,
            top_p=top_p,
            seed=_resolve_sampling_seed(self._tokens, sampling_seed),
        )
        self._cancel_check = cancel_check
        self._pool_kernel = pool_kernel if pool_kernel > 0 else None
        self._prefill_step_size = prefill_step_size
        self._n_lookahead = n_lookahead
        # Convert token IDs once at admission; every later prefill quantum is
        # a device slice rather than a host-to-device reconstruction.
        self._prompt_tokens = mx.array(self._tokens)
        self._query_buffer: list[list[mx.array]] = [
            [] for _ in scorer.attention_layer_indices
        ]
        self._prefill_offset = 0
        self._lookahead_done = 0
        self._next_token: mx.array | None = None
        self._last_logits: mx.array | None = None
        self._importance_layer = 0
        self._reduced_max: mx.array | None = None
        self._importance: mx.array | None = None
        self.phase = ScorerSessionPhase.PREFILL

    @property
    def complete(self) -> bool:
        return self.phase is ScorerSessionPhase.COMPLETE

    @property
    def closed(self) -> bool:
        return self.phase is ScorerSessionPhase.CLOSED

    @property
    def importance(self) -> mx.array:
        if self._importance is None:
            raise SpecPrefillScorerSessionError("scorer session is not complete")
        return self._importance

    @property
    def query_buffer(self) -> list[list[mx.array]]:
        """Request-owned captured queries; never stored in the scorer object."""
        return self._query_buffer

    def step(self) -> ScorerSessionProgress:
        """Run exactly one bounded scorer lane quantum and release it."""
        if self.closed:
            raise SpecPrefillScorerSessionError("scorer session is closed")
        if self.complete:
            return self._progress()
        try:
            with self.scorer.scoring_quantum():
                self._check_cancelled()
                if self.phase is ScorerSessionPhase.PREFILL:
                    self._step_prefill()
                elif self.phase is ScorerSessionPhase.LOOKAHEAD:
                    self._step_lookahead()
                elif self.phase is ScorerSessionPhase.IMPORTANCE:
                    self._step_importance()
                else:  # pragma: no cover - enum exhaustiveness guard
                    raise SpecPrefillScorerSessionError(
                        f"invalid scorer session phase {self.phase!r}"
                    )
                return self._progress()
        except SpecPrefillScorerLaneBusy:
            # A busy lane is a scheduler retry signal, not a request failure.
            # Preserve every cache/buffer/RNG field so this exact quantum can
            # resume after another request yields.
            raise
        except BaseException:
            self.cancel()
            raise

    def run_to_completion(self) -> mx.array:
        """Compatibility facade for the former monolithic ``score_tokens`` API."""
        while not self.complete:
            self.step()
        return self.importance

    def cancel(self) -> None:
        """Discard only this request's draft/cache/capture state."""
        if self.closed:
            return
        self._query_buffer.clear()
        self.cache = None
        self._last_logits = None
        self._next_token = None
        self._reduced_max = None
        self.phase = ScorerSessionPhase.CLOSED

    def _step_prefill(self) -> None:
        remaining = len(self._tokens) - self._prefill_offset
        if remaining <= 0:
            raise SpecPrefillScorerSessionError("scorer prefill cursor exceeded prompt")
        if remaining > 1:
            count = min(self._prefill_step_size, remaining - 1)
            prompt = self._prompt_tokens[
                self._prefill_offset : self._prefill_offset + count
            ]
            with self.scorer.capture_session(
                self._query_extractor, capture_enabled=False
            ) as capture:
                logits = self.scorer.model(prompt[None], cache=self.cache)
                _eval_quantum(logits, self.cache, capture.query_buffer)
            self._prefill_offset += count
            return

        prompt = self._prompt_tokens[self._prefill_offset :]
        with self.scorer.capture_session(
            self._query_extractor, capture_enabled=False
        ) as capture:
            self._last_logits = self.scorer.model(prompt[None], cache=self.cache)
            _eval_quantum(self._last_logits, self.cache, capture.query_buffer)
        self._prefill_offset = len(self._tokens)
        self.phase = (
            ScorerSessionPhase.LOOKAHEAD
            if self._n_lookahead
            else ScorerSessionPhase.IMPORTANCE
        )

    def _step_lookahead(self) -> None:
        if self._last_logits is None:
            raise SpecPrefillScorerSessionError("lookahead requires prefill logits")
        if self._next_token is None:
            self._next_token = self._sampler(self._last_logits[:, -1, :])
            mx.eval(self._next_token)
        with self.scorer.capture_session(self._query_extractor) as capture:
            # Keep the request's buffer through capture-session turnover; the
            # installed dispatcher references only this short-lived context.
            capture.query_buffer = self._query_buffer
            logits = self.scorer.model(
                self._next_token.reshape(1, -1), cache=self.cache
            )
            self._next_token = self._sampler(logits[:, -1, :])
            _eval_quantum(self._next_token, self.cache, self._query_buffer)
        self._lookahead_done += 1
        if self._lookahead_done >= self._n_lookahead:
            self.phase = ScorerSessionPhase.IMPORTANCE

    def _step_importance(self) -> None:
        if self._importance_layer >= len(self._query_buffer):
            if self._reduced_max is None:
                raise SpecPrefillScorerSessionError(
                    "no attention scores captured — check model/patching"
                )
            self._importance = mx.mean(self._reduced_max, axis=0)
            mx.eval(self._importance)
            self.phase = ScorerSessionPhase.COMPLETE
            return

        layer_index = self._importance_layer
        self._importance_layer += 1
        captures = self._query_buffer[layer_index]
        if not captures:
            return
        cache_index = self._layer_to_cache[
            self.scorer.attention_layer_indices[layer_index]
        ]
        layer_scores = _reduce_importance_layer(
            captures,
            self.cache[cache_index],
            len(self._tokens),
            pool_kernel=self._pool_kernel,
        )
        if layer_scores is None:
            return
        if self._reduced_max is None:
            self._reduced_max = layer_scores
        else:
            self._reduced_max = mx.maximum(self._reduced_max, layer_scores)
        # A layer × lookahead × prompt realization is the bounded unavoidable
        # reduction.  Do not concatenate every layer/head before yielding.
        mx.eval(self._reduced_max)

    def _check_cancelled(self) -> None:
        if self._cancel_check is not None:
            self._cancel_check()

    def _progress(self) -> ScorerSessionProgress:
        return ScorerSessionProgress(
            phase=self.phase,
            prefill_tokens=self._prefill_offset,
            lookahead_steps=self._lookahead_done,
            importance_layers=self._importance_layer,
            complete=self.complete,
        )


def _reduce_importance_layer(
    captures: list[mx.array], cache: Any, n_prompt: int, *, pool_kernel: int | None
) -> mx.array | None:
    """Exactly one layer's contribution to legacy importance aggregation."""
    prompt_keys = cache.keys[..., :n_prompt, :]
    if prompt_keys.shape[-2] < n_prompt:
        return None
    q_stack = mx.concatenate(captures, axis=2)
    query_heads = int(q_stack.shape[1])
    key_heads = int(prompt_keys.shape[1])
    if query_heads <= 0 or key_heads <= 0 or query_heads % key_heads:
        raise SpecPrefillScorerSessionError(
            "SpecPrefill attention heads must have an integral per-layer query/KV grouping"
        )
    if int(q_stack.shape[-1]) != int(prompt_keys.shape[-1]):
        raise SpecPrefillScorerSessionError(
            "SpecPrefill captured query and cache key dimensions differ"
        )
    expanded_keys = (
        mx.repeat(prompt_keys, query_heads // key_heads, axis=1)
        if query_heads > key_heads
        else prompt_keys
    )
    scores = (q_stack @ expanded_keys.transpose(0, 1, 3, 2)) * (
        prompt_keys.shape[-1] ** -0.5
    )
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).squeeze(0)
    if pool_kernel and pool_kernel > 1:
        weights = _avg_pool1d(weights, pool_kernel)
    return mx.max(weights, axis=0)


def _eval_quantum(
    logits: mx.array, cache: Any, query_buffer: list[list[mx.array]]
) -> None:
    """Realize all lazy work before the capture context and scorer lane yield."""
    mx.eval(logits, [entry.state for entry in cache], query_buffer)


def _validate_token(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("scorer tokens must be non-negative integer IDs")
    return value


class _RequestLocalSampler:
    """Top-p categorical sampler with an independent MLX PRNG key chain."""

    def __init__(self, *, temp: float, top_p: float, seed: int):
        if (
            isinstance(temp, bool)
            or not isinstance(temp, (int, float))
            or not math.isfinite(temp)
            or temp < 0
        ):
            raise ValueError(
                "scorer sampling temperature must be a finite non-negative number"
            )
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0 <= top_p <= 1
        ):
            raise ValueError("scorer top_p must be a finite number in [0, 1]")
        self._temp = temp
        self._top_p = top_p
        self._key = mx.random.key(seed)

    def __call__(self, logits: mx.array) -> mx.array:
        if self._temp == 0:
            return mx.argmax(logits, axis=-1)
        logprobs = _normalized_logprobs(logits)
        filtered = (
            apply_top_p(logprobs, self._top_p) if 0 < self._top_p < 1.0 else logprobs
        )
        key_pair = mx.random.split(self._key)
        self._key, sample_key = key_pair[0], key_pair[1]
        return mx.random.categorical(filtered * (1.0 / self._temp), key=sample_key)


def _normalized_logprobs(logits: mx.array) -> mx.array:
    """Normalize logits before top-p; mlx-lm's filter accepts log-probabilities."""
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _resolve_sampling_seed(tokens: tuple[int, ...], explicit_seed: int | None) -> int:
    """Derive deterministic request-local draft randomness without global RNG."""
    if explicit_seed is not None:
        if isinstance(explicit_seed, bool) or not isinstance(explicit_seed, int):
            raise ValueError("scorer sampling_seed must be an integer or None")
        return explicit_seed & 0xFFFFFFFF
    digest = hashlib.blake2s(digest_size=4, person=b"sprefill")
    for token in tokens:
        digest.update(str(token).encode("ascii"))
        digest.update(b",")
    return int.from_bytes(digest.digest(), "big")
