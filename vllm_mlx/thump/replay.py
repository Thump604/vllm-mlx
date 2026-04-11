"""Offline Gemma 4 Thump replay harness."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, RotatingKVCache

from vllm_mlx.specprefill import (
    cleanup_rope,
    score_tokens,
    select_chunks,
    sparse_prefill,
)
from vllm_mlx.utils.tokenizer import _load_strict_false

from .capture import (
    CaptureCollector,
    LayerCapture,
    capture_into,
    install_gemma4_capture_patch,
)
from .session import SessionSubstrate


def _sampler(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def _choose_pad_token(tokenizer: Any) -> int:
    for candidate in ("\n", " ", "\t"):
        encoded = tokenizer.encode(candidate)
        if len(encoded) == 1:
            return int(encoded[0])
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, list):
        eos = eos[0]
    if eos is None:
        raise RuntimeError("Unable to choose a single-token padding id")
    return int(eos)


def _clone_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(_clone_state(v) for v in state)
    if isinstance(state, list):
        return [_clone_state(v) for v in state]
    if isinstance(state, mx.array):
        return mx.array(state)
    return state


def clone_prompt_cache(cache: list[Any]) -> list[Any]:
    cloned = []
    for entry in cache:
        state = _clone_state(getattr(entry, "state", None))
        meta_state = getattr(entry, "meta_state", None)
        cls = type(entry)
        if isinstance(entry, KVCache):
            new_entry = KVCache(max_size=getattr(entry, "_max_size", None))
            new_entry.state = state
            cloned.append(new_entry)
            continue
        if isinstance(entry, RotatingKVCache):
            new_entry = RotatingKVCache(
                max_size=getattr(entry, "max_size"),
                keep=getattr(entry, "keep"),
            )
            new_entry.state = state
            new_entry.meta_state = meta_state
            cloned.append(new_entry)
            continue
        if cls.__name__ == "_OffsetCache":
            new_entry = cls()
            new_entry.offset = getattr(entry, "offset", 0)
            cloned.append(new_entry)
            continue
        if hasattr(cls, "from_state"):
            cloned.append(cls.from_state(state, meta_state))
        else:
            new_entry = cls.__new__(cls)
            new_entry.state = state
            if meta_state is not None:
                new_entry.meta_state = meta_state
            cloned.append(new_entry)
    return cloned


def _prefill_tokens(
    model: Any,
    tokens: list[int],
    cache: list[Any],
    *,
    prefill_step_size: int,
    collector: CaptureCollector | None = None,
) -> None:
    if not tokens:
        return
    prompt = mx.array(tokens, dtype=mx.int32)
    offset = 0
    while offset < len(tokens):
        chunk = min(prefill_step_size, len(tokens) - offset)
        if collector is None:
            model(prompt[offset : offset + chunk][None], cache=cache)
        else:
            with capture_into(collector):
                model(prompt[offset : offset + chunk][None], cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        offset += chunk
        mx.clear_cache()


def _merge_captures(
    left: dict[int, LayerCapture],
    right: dict[int, LayerCapture],
) -> dict[int, LayerCapture]:
    out: dict[int, LayerCapture] = {}
    layer_ids = sorted(set(left) | set(right))
    for layer_idx in layer_ids:
        left_cap = left.get(layer_idx)
        right_cap = right.get(layer_idx)
        if left_cap is None:
            out[layer_idx] = right_cap
            continue
        if right_cap is None:
            out[layer_idx] = left_cap
            continue
        out[layer_idx] = LayerCapture(
            keys=np.concatenate([left_cap.keys, right_cap.keys], axis=0),
            values=np.concatenate([left_cap.values, right_cap.values], axis=0),
        )
    return out


@dataclass
class ReplayTrace:
    name: str
    prefix_text: str
    insert_text: str
    suffix_text: str
    max_new_tokens: int = 48
    description: str = ""
    keep_pct: float = 0.5
    composition_threshold: int = 256
    prefill_step_size: int = 256
    capture_step_size: int | None = None
    expected_substrings: list[str] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayTrace":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class TraceTokens:
    prefix: list[int]
    insert: list[int]
    suffix: list[int]
    pad_token_id: int
    prefix_pad_tokens: int
    insert_pad_tokens: int

    @property
    def original_prompt(self) -> list[int]:
        return self.prefix + self.suffix

    @property
    def updated_prompt(self) -> list[int]:
        return self.prefix + self.insert + self.suffix

    @property
    def tail_without_last(self) -> list[int]:
        prompt = self.updated_prompt
        return prompt[len(self.prefix) : -1]

    @property
    def last_prompt_token(self) -> int:
        return self.updated_prompt[-1]

    @property
    def suffix_without_last(self) -> list[int]:
        return self.suffix[:-1]


@dataclass
class ReplayRunResult:
    variant: str
    continuation_latency_ms: float
    post_edit_total_ms: float
    session_total_ms: float
    output_text: str
    output_tokens: list[int]
    re_prefill_avoided_tokens: int
    fallback_count: int
    fallback_reason: str | None
    specprefill_selected_tokens: int = 0
    timings_ms: dict[str, float] | None = None


@dataclass
class ReplayComparison:
    model_path: str
    trace_name: str
    prefill_step_size: int
    capture_step_size: int
    isolation: dict[str, Any]
    composition: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayRunner:
    """Drive the first insert-only Thump replay slice on Gemma 4 text."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        thump_lib_path: str | Path | None = None,
        draft_model_path: str | Path | None = None,
        block_size_tokens: int = 1,
    ) -> None:
        install_gemma4_capture_patch()
        outer_model, tokenizer = _load_strict_false(
            str(model_path),
            {"trust_remote_code": True},
        )
        self.outer_model = outer_model
        self.model = outer_model.language_model
        self.tokenizer = tokenizer
        self.model_path = str(model_path)
        self.thump_lib_path = thump_lib_path
        # Token-sized blocks keep the offline replay slice honest: the
        # remaining parity gap is in suffix reuse after splice, not in
        # coarse block-level key quantization.
        self.block_size_tokens = block_size_tokens
        self.draft_model = None
        if draft_model_path is not None:
            draft_outer, _ = _load_strict_false(
                str(draft_model_path),
                {"trust_remote_code": True},
            )
            self.draft_model = draft_outer.language_model

    def build_trace_tokens(
        self, trace: ReplayTrace, *, block_size: int | None = None
    ) -> TraceTokens:
        block_size = block_size or self.block_size_tokens
        pad_token_id = _choose_pad_token(self.tokenizer)
        prefix = list(self.tokenizer.encode(trace.prefix_text))
        insert = list(self.tokenizer.encode(trace.insert_text))
        suffix = list(self.tokenizer.encode(trace.suffix_text))
        prefix_pad = (-len(prefix)) % block_size
        insert_pad = (-len(insert)) % block_size
        prefix.extend([pad_token_id] * prefix_pad)
        insert.extend([pad_token_id] * insert_pad)
        if not suffix:
            raise ValueError("suffix_text must produce at least one token")
        return TraceTokens(
            prefix=prefix,
            insert=insert,
            suffix=suffix,
            pad_token_id=pad_token_id,
            prefix_pad_tokens=prefix_pad,
            insert_pad_tokens=insert_pad,
        )

    def _decode_from_cache(
        self,
        cache: list[Any],
        last_prompt_token: int,
        *,
        max_new_tokens: int,
        start_time: float,
    ) -> tuple[float, str, list[int]]:
        output_tokens: list[int] = []
        continuation_ms = None
        generator = generate_step(
            mx.array([last_prompt_token], dtype=mx.int32),
            self.model,
            max_tokens=max_new_tokens,
            sampler=_sampler,
            prompt_cache=cache,
            prefill_step_size=1,
        )
        eos_ids = getattr(self.tokenizer, "eos_token_id", None)
        eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        for token, _logprobs in generator:
            token_id = int(token)
            output_tokens.append(token_id)
            if continuation_ms is None:
                continuation_ms = (time.perf_counter() - start_time) * 1000.0
            if token_id in eos_set:
                break
        output_text = self.tokenizer.decode(output_tokens)
        return continuation_ms or 0.0, output_text, output_tokens

    def run_baseline(
        self,
        trace: ReplayTrace,
        tokens: TraceTokens,
        *,
        use_specprefill: bool,
    ) -> ReplayRunResult:
        session_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        prefix_cache = self.model.make_cache()
        start = time.perf_counter()
        _prefill_tokens(
            self.model,
            tokens.prefix,
            prefix_cache,
            prefill_step_size=trace.prefill_step_size,
        )
        timings_ms["prefix_prefill_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        prefix_snapshot = clone_prompt_cache(prefix_cache)
        timings_ms["prefix_snapshot_clone_ms"] = (
            time.perf_counter() - start
        ) * 1000.0
        start = time.perf_counter()
        _prefill_tokens(
            self.model,
            tokens.suffix,
            prefix_cache,
            prefill_step_size=trace.prefill_step_size,
        )
        timings_ms["suffix_prefill_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        edit_cache = clone_prompt_cache(prefix_snapshot)
        timings_ms["edit_cache_clone_ms"] = (time.perf_counter() - start) * 1000.0
        edit_start = time.perf_counter()
        selected_count = 0
        if use_specprefill and self.draft_model is not None:
            start = time.perf_counter()
            importance = score_tokens(
                self.draft_model,
                tokens.tail_without_last,
                prefill_step_size=trace.prefill_step_size,
            )
            timings_ms["specprefill_score_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
            selected = select_chunks(importance, keep_pct=trace.keep_pct)
            selected_count = int(selected.shape[0])
            start = time.perf_counter()
            sparse_prefill(
                self.model,
                tokens.tail_without_last,
                selected,
                edit_cache,
                step_size=trace.prefill_step_size,
                position_offset=len(tokens.prefix),
            )
            timings_ms["specprefill_prefill_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
        else:
            start = time.perf_counter()
            _prefill_tokens(
                self.model,
                tokens.tail_without_last,
                edit_cache,
                prefill_step_size=trace.prefill_step_size,
            )
            timings_ms["delta_prefill_ms"] = (time.perf_counter() - start) * 1000.0
        decode_start = time.perf_counter()
        continuation_ms, output_text, output_tokens = self._decode_from_cache(
            edit_cache,
            tokens.last_prompt_token,
            max_new_tokens=trace.max_new_tokens,
            start_time=edit_start,
        )
        timings_ms["resumed_decode_ms"] = (time.perf_counter() - decode_start) * 1000.0
        post_edit_ms = (time.perf_counter() - edit_start) * 1000.0
        session_total_ms = (time.perf_counter() - session_start) * 1000.0
        post_edit_measured_ms = (
            timings_ms.get("delta_prefill_ms", 0.0)
            + timings_ms.get("specprefill_score_ms", 0.0)
            + timings_ms.get("specprefill_prefill_ms", 0.0)
            + timings_ms["resumed_decode_ms"]
        )
        timings_ms["post_edit_overhead_ms"] = max(
            0.0, post_edit_ms - post_edit_measured_ms
        )
        measured_session_ms = sum(timings_ms.values())
        timings_ms["session_overhead_ms"] = max(
            0.0, session_total_ms - measured_session_ms
        )
        cleanup_rope(self.model)
        return ReplayRunResult(
            variant="baseline",
            continuation_latency_ms=continuation_ms,
            post_edit_total_ms=post_edit_ms,
            session_total_ms=session_total_ms,
            output_text=output_text,
            output_tokens=output_tokens,
            re_prefill_avoided_tokens=0,
            fallback_count=0,
            fallback_reason=None,
            specprefill_selected_tokens=selected_count,
            timings_ms=timings_ms,
        )

    def run_thump(self, trace: ReplayTrace, tokens: TraceTokens) -> ReplayRunResult:
        session_start = time.perf_counter()
        try:
            capture_step_size = trace.capture_step_size or trace.prefill_step_size
            timings_ms: dict[str, float] = {}
            prefix_cache = self.model.make_cache()
            prefix_collector = CaptureCollector()
            start = time.perf_counter()
            _prefill_tokens(
                self.model,
                tokens.prefix,
                prefix_cache,
                prefill_step_size=capture_step_size,
                collector=prefix_collector,
            )
            timings_ms["prefix_capture_prefill_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
            start = time.perf_counter()
            prefix_snapshot = clone_prompt_cache(prefix_cache)
            timings_ms["prefix_snapshot_clone_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
            suffix_collector = CaptureCollector()
            start = time.perf_counter()
            _prefill_tokens(
                self.model,
                tokens.suffix,
                prefix_cache,
                prefill_step_size=capture_step_size,
                collector=suffix_collector,
            )
            timings_ms["suffix_capture_prefill_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
            start = time.perf_counter()
            full_capture = _merge_captures(
                prefix_collector.joined(),
                suffix_collector.joined(),
            )
            timings_ms["capture_merge_ms"] = (time.perf_counter() - start) * 1000.0
            total_blocks = (
                math.ceil(len(tokens.updated_prompt) / self.block_size_tokens) + 8
            )
            start = time.perf_counter()
            session = SessionSubstrate.from_gemma4_model(
                self.model,
                block_size_tokens=self.block_size_tokens,
                block_capacity=total_blocks,
                lib_path=self.thump_lib_path,
            )
            timings_ms["session_construct_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
            start = time.perf_counter()
            session.initialize_from_capture(
                full_capture,
                total_tokens=len(tokens.original_prompt),
            )
            timings_ms["capture_write_ms"] = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            delta_cache = clone_prompt_cache(prefix_snapshot)
            timings_ms["delta_cache_clone_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
            edit_start = time.perf_counter()
            insert_collector = CaptureCollector()
            start = time.perf_counter()
            _prefill_tokens(
                self.model,
                tokens.insert,
                delta_cache,
                prefill_step_size=capture_step_size,
                collector=insert_collector,
            )
            timings_ms["delta_prefill_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            session.splice_insert_from_capture(
                len(tokens.prefix),
                insert_collector.joined(),
                insert_token_count=len(tokens.insert),
            )
            timings_ms["splice_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            materialized_cache = session.materialize_prompt_cache(
                self.model,
                upto_tokens=len(tokens.updated_prompt) - 1,
            )
            timings_ms["materialize_ms"] = (time.perf_counter() - start) * 1000.0
            decode_start = time.perf_counter()
            continuation_ms, output_text, output_tokens = self._decode_from_cache(
                materialized_cache,
                tokens.last_prompt_token,
                max_new_tokens=trace.max_new_tokens,
                start_time=edit_start,
            )
            timings_ms["resumed_decode_ms"] = (time.perf_counter() - decode_start) * 1000.0
            post_edit_ms = (time.perf_counter() - edit_start) * 1000.0
            session_total_ms = (time.perf_counter() - session_start) * 1000.0
            post_edit_measured_ms = (
                timings_ms["delta_prefill_ms"]
                + timings_ms["splice_ms"]
                + timings_ms["materialize_ms"]
                + timings_ms["resumed_decode_ms"]
            )
            timings_ms["post_edit_overhead_ms"] = max(
                0.0, post_edit_ms - post_edit_measured_ms
            )
            measured_session_ms = sum(timings_ms.values())
            timings_ms["session_overhead_ms"] = max(
                0.0, session_total_ms - measured_session_ms
            )
            session.close()
            return ReplayRunResult(
                variant="thump",
                continuation_latency_ms=continuation_ms,
                post_edit_total_ms=post_edit_ms,
                session_total_ms=session_total_ms,
                output_text=output_text,
                output_tokens=output_tokens,
                re_prefill_avoided_tokens=max(0, len(tokens.suffix) - 1),
                fallback_count=0,
                fallback_reason=None,
                timings_ms=timings_ms,
            )
        except Exception as exc:
            fallback = self.run_baseline(trace, tokens, use_specprefill=False)
            fallback.variant = "thump_fallback"
            fallback.fallback_count = 1
            fallback.fallback_reason = str(exc)
            return fallback

    def run_comparison(self, trace: ReplayTrace) -> ReplayComparison:
        tokens = self.build_trace_tokens(trace)
        isolation_baseline = self.run_baseline(trace, tokens, use_specprefill=False)
        isolation_thump = self.run_thump(trace, tokens)

        composition = None
        if (
            self.draft_model is not None
            and len(tokens.tail_without_last) >= trace.composition_threshold
        ):
            composition = {
                "baseline": asdict(
                    self.run_baseline(trace, tokens, use_specprefill=True)
                ),
                "thump": asdict(self.run_thump(trace, tokens)),
                "specprefill_threshold": trace.composition_threshold,
                "keep_pct": trace.keep_pct,
            }

        isolation = {
            "baseline": asdict(isolation_baseline),
            "thump": asdict(isolation_thump),
            "output_exact_match": isolation_baseline.output_tokens
            == isolation_thump.output_tokens,
            "prefix_pad_tokens": tokens.prefix_pad_tokens,
            "insert_pad_tokens": tokens.insert_pad_tokens,
        }
        return ReplayComparison(
            model_path=self.model_path,
            trace_name=trace.name,
            prefill_step_size=trace.prefill_step_size,
            capture_step_size=trace.capture_step_size or trace.prefill_step_size,
            isolation=isolation,
            composition=composition,
        )
