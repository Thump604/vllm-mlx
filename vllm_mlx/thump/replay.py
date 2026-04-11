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


def _effective_selected_indices(
    selected_indices: mx.array,
    *,
    token_count: int,
    max_rotating_size: int,
) -> mx.array:
    """Mirror sparse_prefill's RotatingKVCache tail expansion for telemetry."""
    if max_rotating_size <= 0:
        return selected_indices
    tail_start = max(0, token_count - max_rotating_size)
    tail_indices = set(range(tail_start, token_count))
    merged = sorted(set(selected_indices.tolist()) | tail_indices)
    return mx.array(merged, dtype=selected_indices.dtype)


def _run_output_sequences(run: ReplayRunResult) -> list[list[int]]:
    if run.edit_output_tokens:
        return run.edit_output_tokens
    return [run.output_tokens]


def _output_exact_matches(
    lhs: ReplayRunResult,
    rhs: ReplayRunResult,
) -> list[bool]:
    left = _run_output_sequences(lhs)
    right = _run_output_sequences(rhs)
    count = min(len(left), len(right))
    return [left[index] == right[index] for index in range(count)]


@dataclass
class ReplayTrace:
    name: str
    prefix_text: str
    insert_text: str
    suffix_text: str
    insert_texts: list[str] | None = None
    max_new_tokens: int = 48
    description: str = ""
    keep_pct: float = 0.5
    composition_threshold: int = 256
    prefill_step_size: int = 256
    capture_step_size: int | None = None
    thump_refresh_tail_tokens: int = 0
    expected_substrings: list[str] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayTrace":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class TraceTokens:
    prefix: list[int]
    insert_variants: list[list[int]]
    suffix: list[int]
    pad_token_id: int
    prefix_pad_tokens: int
    insert_pad_tokens: int

    @property
    def insert(self) -> list[int]:
        return self.insert_variants[0]

    @property
    def edit_count(self) -> int:
        return len(self.insert_variants)

    @property
    def original_prompt(self) -> list[int]:
        return self.prefix + self.suffix

    @property
    def updated_prompt(self) -> list[int]:
        return self.prefix + self.insert + self.suffix

    def updated_prompt_for_edit(self, edit_index: int) -> list[int]:
        return self.prefix + self.insert_variants[edit_index] + self.suffix

    @property
    def tail_without_last(self) -> list[int]:
        prompt = self.updated_prompt
        return prompt[len(self.prefix) : -1]

    def tail_without_last_for_edit(self, edit_index: int) -> list[int]:
        prompt = self.updated_prompt_for_edit(edit_index)
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
    edit_count: int = 1
    edit_continuation_latencies_ms: list[float] | None = None
    edit_post_edit_totals_ms: list[float] | None = None
    edit_cumulative_session_totals_ms: list[float] | None = None
    edit_output_texts: list[str] | None = None
    edit_output_tokens: list[list[int]] | None = None
    specprefill_selected_tokens: int = 0
    specprefill_effective_selected_tokens: int = 0
    timings_ms: dict[str, float] | None = None


@dataclass
class ReplayComparison:
    model_path: str
    trace_name: str
    edit_count: int
    keep_pct: float
    composition_threshold: int
    prefill_step_size: int
    capture_step_size: int
    thump_refresh_tail_tokens: int
    tail_without_last_tokens: int
    max_rotating_size: int
    composition_control_viable: bool
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
        self.max_rotating_size = max(
            (
                int(getattr(c, "max_size", 0))
                for c in self.model.make_cache()
                if type(c).__name__ == "RotatingKVCache"
            ),
            default=0,
        )
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
        raw_inserts = trace.insert_texts or [trace.insert_text]
        insert_variants = [list(self.tokenizer.encode(text)) for text in raw_inserts]
        suffix = list(self.tokenizer.encode(trace.suffix_text))
        prefix_pad = (-len(prefix)) % block_size
        max_insert_len = max(len(tokens) for tokens in insert_variants)
        target_insert_len = math.ceil(max_insert_len / block_size) * block_size
        insert_pad = target_insert_len - len(insert_variants[0])
        prefix.extend([pad_token_id] * prefix_pad)
        padded_insert_variants: list[list[int]] = []
        for variant_tokens in insert_variants:
            pad = target_insert_len - len(variant_tokens)
            padded_insert_variants.append(variant_tokens + [pad_token_id] * pad)
        if not suffix:
            raise ValueError("suffix_text must produce at least one token")
        return TraceTokens(
            prefix=prefix,
            insert_variants=padded_insert_variants,
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
        timings_ms["prefix_snapshot_clone_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        _prefill_tokens(
            self.model,
            tokens.suffix,
            prefix_cache,
            prefill_step_size=trace.prefill_step_size,
        )
        timings_ms["suffix_prefill_ms"] = (time.perf_counter() - start) * 1000.0

        selected_count = 0
        effective_selected_count = 0
        edit_continuations_ms: list[float] = []
        edit_post_edit_totals_ms: list[float] = []
        edit_cumulative_session_totals_ms: list[float] = []
        edit_output_texts: list[str] = []
        edit_output_tokens: list[list[int]] = []
        for edit_index, insert_tokens in enumerate(tokens.insert_variants):
            start = time.perf_counter()
            edit_cache = clone_prompt_cache(prefix_snapshot)
            timings_ms["edit_cache_clone_ms"] = (
                timings_ms.get("edit_cache_clone_ms", 0.0)
                + (time.perf_counter() - start) * 1000.0
            )
            edit_start = time.perf_counter()
            tail_without_last = tokens.tail_without_last_for_edit(edit_index)
            edit_measured_ms = 0.0
            if use_specprefill and self.draft_model is not None:
                start = time.perf_counter()
                importance = score_tokens(
                    self.draft_model,
                    tail_without_last,
                    prefill_step_size=trace.prefill_step_size,
                )
                score_ms = (time.perf_counter() - start) * 1000.0
                timings_ms["specprefill_score_ms"] = (
                    timings_ms.get("specprefill_score_ms", 0.0) + score_ms
                )
                edit_measured_ms += score_ms
                selected = select_chunks(importance, keep_pct=trace.keep_pct)
                selected_count += int(selected.shape[0])
                effective_selected = _effective_selected_indices(
                    selected,
                    token_count=len(tail_without_last),
                    max_rotating_size=self.max_rotating_size,
                )
                effective_selected_count += int(effective_selected.shape[0])
                start = time.perf_counter()
                sparse_prefill(
                    self.model,
                    tail_without_last,
                    effective_selected,
                    edit_cache,
                    step_size=trace.prefill_step_size,
                    position_offset=len(tokens.prefix),
                )
                prefill_ms = (time.perf_counter() - start) * 1000.0
                timings_ms["specprefill_prefill_ms"] = (
                    timings_ms.get("specprefill_prefill_ms", 0.0) + prefill_ms
                )
                edit_measured_ms += prefill_ms
            else:
                start = time.perf_counter()
                _prefill_tokens(
                    self.model,
                    tail_without_last,
                    edit_cache,
                    prefill_step_size=trace.prefill_step_size,
                )
                prefill_ms = (time.perf_counter() - start) * 1000.0
                timings_ms["delta_prefill_ms"] = (
                    timings_ms.get("delta_prefill_ms", 0.0) + prefill_ms
                )
                edit_measured_ms += prefill_ms
            decode_start = time.perf_counter()
            continuation_ms, output_text, output_tokens = self._decode_from_cache(
                edit_cache,
                tokens.last_prompt_token,
                max_new_tokens=trace.max_new_tokens,
                start_time=edit_start,
            )
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            timings_ms["resumed_decode_ms"] = (
                timings_ms.get("resumed_decode_ms", 0.0) + decode_ms
            )
            edit_measured_ms += decode_ms
            post_edit_ms = (time.perf_counter() - edit_start) * 1000.0
            timings_ms["post_edit_overhead_ms"] = timings_ms.get(
                "post_edit_overhead_ms", 0.0
            ) + max(0.0, post_edit_ms - edit_measured_ms)
            edit_continuations_ms.append(continuation_ms)
            edit_post_edit_totals_ms.append(post_edit_ms)
            edit_cumulative_session_totals_ms.append(
                (time.perf_counter() - session_start) * 1000.0
            )
            edit_output_texts.append(output_text)
            edit_output_tokens.append(output_tokens)
        session_total_ms = (time.perf_counter() - session_start) * 1000.0
        measured_session_ms = sum(timings_ms.values())
        timings_ms["session_overhead_ms"] = max(
            0.0, session_total_ms - measured_session_ms
        )
        cleanup_rope(self.model)
        return ReplayRunResult(
            variant="baseline",
            continuation_latency_ms=edit_continuations_ms[0],
            post_edit_total_ms=sum(edit_post_edit_totals_ms),
            session_total_ms=session_total_ms,
            output_text=edit_output_texts[-1],
            output_tokens=edit_output_tokens[-1],
            re_prefill_avoided_tokens=0,
            fallback_count=0,
            fallback_reason=None,
            edit_count=tokens.edit_count,
            edit_continuation_latencies_ms=edit_continuations_ms,
            edit_post_edit_totals_ms=edit_post_edit_totals_ms,
            edit_cumulative_session_totals_ms=edit_cumulative_session_totals_ms,
            edit_output_texts=edit_output_texts,
            edit_output_tokens=edit_output_tokens,
            specprefill_selected_tokens=selected_count,
            specprefill_effective_selected_tokens=effective_selected_count,
            timings_ms=timings_ms,
        )

    def run_direct_full_prompt(
        self,
        trace: ReplayTrace,
        tokens: TraceTokens,
    ) -> ReplayRunResult:
        session_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        edit_continuations_ms: list[float] = []
        edit_post_edit_totals_ms: list[float] = []
        edit_cumulative_session_totals_ms: list[float] = []
        edit_output_texts: list[str] = []
        edit_output_tokens: list[list[int]] = []
        for edit_index in range(tokens.edit_count):
            cache = self.model.make_cache()
            prompt = tokens.updated_prompt_for_edit(edit_index)
            edit_start = time.perf_counter()
            start = time.perf_counter()
            _prefill_tokens(
                self.model,
                prompt[:-1],
                cache,
                prefill_step_size=trace.prefill_step_size,
            )
            prefill_ms = (time.perf_counter() - start) * 1000.0
            timings_ms["direct_full_prefill_ms"] = (
                timings_ms.get("direct_full_prefill_ms", 0.0) + prefill_ms
            )
            decode_start = time.perf_counter()
            continuation_ms, output_text, output_tokens = self._decode_from_cache(
                cache,
                prompt[-1],
                max_new_tokens=trace.max_new_tokens,
                start_time=edit_start,
            )
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            timings_ms["resumed_decode_ms"] = (
                timings_ms.get("resumed_decode_ms", 0.0) + decode_ms
            )
            post_edit_ms = (time.perf_counter() - edit_start) * 1000.0
            timings_ms["post_edit_overhead_ms"] = timings_ms.get(
                "post_edit_overhead_ms", 0.0
            ) + max(0.0, post_edit_ms - (prefill_ms + decode_ms))
            edit_continuations_ms.append(continuation_ms)
            edit_post_edit_totals_ms.append(post_edit_ms)
            edit_cumulative_session_totals_ms.append(
                (time.perf_counter() - session_start) * 1000.0
            )
            edit_output_texts.append(output_text)
            edit_output_tokens.append(output_tokens)
        session_total_ms = (time.perf_counter() - session_start) * 1000.0
        measured_session_ms = sum(timings_ms.values())
        timings_ms["session_overhead_ms"] = max(
            0.0, session_total_ms - measured_session_ms
        )
        return ReplayRunResult(
            variant="direct_full_prompt",
            continuation_latency_ms=edit_continuations_ms[0],
            post_edit_total_ms=sum(edit_post_edit_totals_ms),
            session_total_ms=session_total_ms,
            output_text=edit_output_texts[-1],
            output_tokens=edit_output_tokens[-1],
            re_prefill_avoided_tokens=0,
            fallback_count=0,
            fallback_reason=None,
            edit_count=tokens.edit_count,
            edit_continuation_latencies_ms=edit_continuations_ms,
            edit_post_edit_totals_ms=edit_post_edit_totals_ms,
            edit_cumulative_session_totals_ms=edit_cumulative_session_totals_ms,
            edit_output_texts=edit_output_texts,
            edit_output_tokens=edit_output_tokens,
            timings_ms=timings_ms,
        )

    def run_thump(self, trace: ReplayTrace, tokens: TraceTokens) -> ReplayRunResult:
        session_start = time.perf_counter()
        try:
            capture_step_size = trace.capture_step_size or trace.prefill_step_size
            refresh_tail_tokens = max(0, int(trace.thump_refresh_tail_tokens))
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
            replace_headroom_blocks = (
                math.ceil(len(tokens.insert) / self.block_size_tokens)
                if tokens.edit_count > 1
                else 0
            )
            total_blocks = (
                math.ceil(len(tokens.updated_prompt) / self.block_size_tokens)
                + replace_headroom_blocks
                + 8
            )
            start = time.perf_counter()
            session = SessionSubstrate.from_gemma4_model(
                self.model,
                block_size_tokens=self.block_size_tokens,
                block_capacity=total_blocks,
                lib_path=self.thump_lib_path,
            )
            timings_ms["session_construct_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            session.initialize_from_capture(
                full_capture,
                total_tokens=len(tokens.original_prompt),
            )
            timings_ms["capture_write_ms"] = (time.perf_counter() - start) * 1000.0
            edit_continuations_ms: list[float] = []
            edit_post_edit_totals_ms: list[float] = []
            edit_cumulative_session_totals_ms: list[float] = []
            edit_output_texts: list[str] = []
            edit_output_tokens: list[list[int]] = []
            insert_token_count = len(tokens.insert)
            for edit_index, insert_tokens in enumerate(tokens.insert_variants):
                start = time.perf_counter()
                delta_cache = clone_prompt_cache(prefix_snapshot)
                timings_ms["delta_cache_clone_ms"] = (
                    timings_ms.get("delta_cache_clone_ms", 0.0)
                    + (time.perf_counter() - start) * 1000.0
                )
                edit_start = time.perf_counter()
                insert_collector = CaptureCollector()
                start = time.perf_counter()
                _prefill_tokens(
                    self.model,
                    insert_tokens,
                    delta_cache,
                    prefill_step_size=capture_step_size,
                    collector=insert_collector,
                )
                delta_prefill_ms = (time.perf_counter() - start) * 1000.0
                timings_ms["delta_prefill_ms"] = (
                    timings_ms.get("delta_prefill_ms", 0.0) + delta_prefill_ms
                )
                start = time.perf_counter()
                if edit_index == 0:
                    session.splice_insert_from_capture(
                        len(tokens.prefix),
                        insert_collector.joined(),
                        insert_token_count=insert_token_count,
                    )
                else:
                    session.replace_equal_length_from_capture(
                        len(tokens.prefix),
                        insert_collector.joined(),
                        replace_token_count=insert_token_count,
                    )
                splice_ms = (time.perf_counter() - start) * 1000.0
                timings_ms["splice_ms"] = timings_ms.get("splice_ms", 0.0) + splice_ms
                start = time.perf_counter()
                prompt_without_last = tokens.updated_prompt_for_edit(edit_index)[:-1]
                refresh_count = min(refresh_tail_tokens, len(prompt_without_last))
                materialized_cache = session.materialize_prompt_cache(
                    self.model,
                    upto_tokens=len(prompt_without_last) - refresh_count,
                )
                materialize_ms = (time.perf_counter() - start) * 1000.0
                timings_ms["materialize_ms"] = (
                    timings_ms.get("materialize_ms", 0.0) + materialize_ms
                )
                refresh_ms = 0.0
                if refresh_count:
                    start = time.perf_counter()
                    _prefill_tokens(
                        self.model,
                        prompt_without_last[-refresh_count:],
                        materialized_cache,
                        prefill_step_size=trace.prefill_step_size,
                    )
                    refresh_ms = (time.perf_counter() - start) * 1000.0
                    timings_ms["refresh_prefill_ms"] = (
                        timings_ms.get("refresh_prefill_ms", 0.0) + refresh_ms
                    )
                decode_start = time.perf_counter()
                continuation_ms, output_text, output_tokens = self._decode_from_cache(
                    materialized_cache,
                    tokens.last_prompt_token,
                    max_new_tokens=trace.max_new_tokens,
                    start_time=edit_start,
                )
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                timings_ms["resumed_decode_ms"] = (
                    timings_ms.get("resumed_decode_ms", 0.0) + decode_ms
                )
                post_edit_ms = (time.perf_counter() - edit_start) * 1000.0
                timings_ms["post_edit_overhead_ms"] = timings_ms.get(
                    "post_edit_overhead_ms", 0.0
                ) + max(
                    0.0,
                    post_edit_ms
                    - (
                        delta_prefill_ms
                        + splice_ms
                        + materialize_ms
                        + refresh_ms
                        + decode_ms
                    ),
                )
                edit_continuations_ms.append(continuation_ms)
                edit_post_edit_totals_ms.append(post_edit_ms)
                edit_cumulative_session_totals_ms.append(
                    (time.perf_counter() - session_start) * 1000.0
                )
                edit_output_texts.append(output_text)
                edit_output_tokens.append(output_tokens)
            session_total_ms = (time.perf_counter() - session_start) * 1000.0
            measured_session_ms = sum(timings_ms.values())
            timings_ms["session_overhead_ms"] = max(
                0.0, session_total_ms - measured_session_ms
            )
            session.close()
            return ReplayRunResult(
                variant="thump",
                continuation_latency_ms=edit_continuations_ms[0],
                post_edit_total_ms=sum(edit_post_edit_totals_ms),
                session_total_ms=session_total_ms,
                output_text=edit_output_texts[-1],
                output_tokens=edit_output_tokens[-1],
                re_prefill_avoided_tokens=max(
                    0,
                    (len(tokens.suffix) - 1)
                    - min(refresh_tail_tokens, len(tokens.suffix) - 1),
                )
                * tokens.edit_count,
                fallback_count=0,
                fallback_reason=None,
                edit_count=tokens.edit_count,
                edit_continuation_latencies_ms=edit_continuations_ms,
                edit_post_edit_totals_ms=edit_post_edit_totals_ms,
                edit_cumulative_session_totals_ms=edit_cumulative_session_totals_ms,
                edit_output_texts=edit_output_texts,
                edit_output_tokens=edit_output_tokens,
                specprefill_effective_selected_tokens=0,
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
        isolation_direct = self.run_direct_full_prompt(trace, tokens)
        isolation_baseline = self.run_baseline(trace, tokens, use_specprefill=False)
        isolation_thump = self.run_thump(trace, tokens)

        composition = None
        if (
            self.draft_model is not None
            and len(tokens.tail_without_last) >= trace.composition_threshold
        ):
            composition_baseline = self.run_baseline(
                trace, tokens, use_specprefill=True
            )
            composition_thump = self.run_thump(trace, tokens)
            composition = {
                "baseline": asdict(composition_baseline),
                "thump": asdict(composition_thump),
                "specprefill_threshold": trace.composition_threshold,
                "keep_pct": trace.keep_pct,
                "edit_output_exact_matches": _output_exact_matches(
                    composition_baseline,
                    composition_thump,
                ),
            }
            composition["baseline_vs_isolation_baseline_exact_match"] = (
                composition["baseline"].get("edit_output_tokens")
                or [composition["baseline"]["output_tokens"]]
            ) == _run_output_sequences(isolation_baseline)

        isolation = {
            "direct_full_prompt": asdict(isolation_direct),
            "baseline": asdict(isolation_baseline),
            "thump": asdict(isolation_thump),
            "output_exact_match": _run_output_sequences(isolation_baseline)
            == _run_output_sequences(isolation_thump),
            "edit_output_exact_matches": _output_exact_matches(
                isolation_baseline,
                isolation_thump,
            ),
            "baseline_vs_direct_full_prompt_exact_match": _run_output_sequences(
                isolation_baseline
            )
            == _run_output_sequences(isolation_direct),
            "thump_vs_direct_full_prompt_exact_match": _run_output_sequences(
                isolation_thump
            )
            == _run_output_sequences(isolation_direct),
            "baseline_vs_direct_full_prompt_edit_exact_matches": _output_exact_matches(
                isolation_baseline,
                isolation_direct,
            ),
            "thump_vs_direct_full_prompt_edit_exact_matches": _output_exact_matches(
                isolation_thump,
                isolation_direct,
            ),
            "prefix_pad_tokens": tokens.prefix_pad_tokens,
            "insert_pad_tokens": tokens.insert_pad_tokens,
        }
        return ReplayComparison(
            model_path=self.model_path,
            trace_name=trace.name,
            edit_count=tokens.edit_count,
            keep_pct=trace.keep_pct,
            composition_threshold=trace.composition_threshold,
            prefill_step_size=trace.prefill_step_size,
            capture_step_size=trace.capture_step_size or trace.prefill_step_size,
            thump_refresh_tail_tokens=trace.thump_refresh_tail_tokens,
            tail_without_last_tokens=len(tokens.tail_without_last),
            max_rotating_size=self.max_rotating_size,
            composition_control_viable=len(tokens.tail_without_last)
            > self.max_rotating_size,
            isolation=isolation,
            composition=composition,
        )
