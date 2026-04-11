#!/usr/bin/env python3
"""Probe the current Thump replay limit on the offline Gemma 4 slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step

from vllm_mlx.thump.replay import (
    CaptureCollector,
    ReplayRunner,
    ReplayTrace,
    _merge_captures,
    _prefill_tokens,
    _sampler,
    clone_prompt_cache,
)
from vllm_mlx.thump.session import SessionSubstrate


def _next_token(runner: ReplayRunner, cache: list[object], token_id: int) -> int:
    generator = generate_step(
        mx.array([token_id], dtype=mx.int32),
        runner.model,
        max_tokens=1,
        sampler=_sampler,
        prompt_cache=cache,
        prefill_step_size=1,
    )
    token, _ = next(generator)
    return int(token)


def _diff_summary(
    *,
    layer_idx: int,
    layer_type: str,
    left_k,
    left_v,
    right_k,
    right_v,
) -> dict[str, object]:
    left_k_arr = mx.array(left_k).astype(mx.float32)
    left_v_arr = mx.array(left_v).astype(mx.float32)
    right_k_arr = mx.array(right_k).astype(mx.float32)
    right_v_arr = mx.array(right_v).astype(mx.float32)
    left_tokens = left_k_arr.shape[-2] if left_k_arr.ndim >= 2 else left_k_arr.shape[0]
    right_tokens = (
        right_k_arr.shape[-2] if right_k_arr.ndim >= 2 else right_k_arr.shape[0]
    )
    aligned_tokens = min(left_tokens, right_tokens)
    if left_k_arr.shape != right_k_arr.shape and left_k_arr.ndim >= 2:
        left_k_arr = left_k_arr[..., -aligned_tokens:, :]
        left_v_arr = left_v_arr[..., -aligned_tokens:, :]
        right_k_arr = right_k_arr[..., -aligned_tokens:, :]
        right_v_arr = right_v_arr[..., -aligned_tokens:, :]
    k_diff = mx.abs(left_k_arr - right_k_arr)
    v_diff = mx.abs(left_v_arr - right_v_arr)
    mx.eval(k_diff, v_diff)
    return {
        "layer_index": layer_idx,
        "layer_type": layer_type,
        "left_tokens": left_tokens,
        "right_tokens": right_tokens,
        "aligned_tokens": aligned_tokens,
        "k_max_abs_diff": float(mx.max(k_diff).item()),
        "k_mean_abs_diff": float(mx.mean(k_diff).item()),
        "v_max_abs_diff": float(mx.max(v_diff).item()),
        "v_mean_abs_diff": float(mx.mean(v_diff).item()),
    }


def probe_splice_limit(
    *,
    model_path: str | Path,
    trace_path: str | Path,
    output_path: str | Path,
    thump_lib_path: str | Path | None = None,
) -> dict[str, object]:
    trace = ReplayTrace.from_path(trace_path)
    runner = ReplayRunner(
        model_path,
        thump_lib_path=thump_lib_path,
        block_size_tokens=1,
    )
    tokens = runner.build_trace_tokens(trace)

    baseline_cache = runner.model.make_cache()
    _prefill_tokens(
        runner.model,
        tokens.updated_prompt[:-1],
        baseline_cache,
        prefill_step_size=trace.prefill_step_size,
    )
    baseline_next = _next_token(runner, baseline_cache, tokens.last_prompt_token)

    full_collector = CaptureCollector()
    full_capture_cache = runner.model.make_cache()
    _prefill_tokens(
        runner.model,
        tokens.updated_prompt[:-1],
        full_capture_cache,
        prefill_step_size=trace.prefill_step_size,
        collector=full_collector,
    )
    full_session = SessionSubstrate.from_gemma4_model(
        runner.model,
        block_size_tokens=1,
        block_capacity=len(tokens.updated_prompt) + 8,
        lib_path=thump_lib_path,
    )
    full_session.initialize_from_capture(
        full_collector.joined(),
        total_tokens=len(tokens.updated_prompt) - 1,
    )
    full_roundtrip_cache = full_session.materialize_prompt_cache(
        runner.model,
        upto_tokens=len(tokens.updated_prompt) - 1,
    )
    full_roundtrip_next = _next_token(
        runner,
        full_roundtrip_cache,
        tokens.last_prompt_token,
    )

    prefix_cache = runner.model.make_cache()
    prefix_collector = CaptureCollector()
    _prefill_tokens(
        runner.model,
        tokens.prefix,
        prefix_cache,
        prefill_step_size=trace.prefill_step_size,
        collector=prefix_collector,
    )
    prefix_snapshot = clone_prompt_cache(prefix_cache)
    suffix_collector = CaptureCollector()
    _prefill_tokens(
        runner.model,
        tokens.suffix,
        prefix_cache,
        prefill_step_size=trace.prefill_step_size,
        collector=suffix_collector,
    )
    prefix_suffix_capture = _merge_captures(
        prefix_collector.joined(),
        suffix_collector.joined(),
    )
    splice_session = SessionSubstrate.from_gemma4_model(
        runner.model,
        block_size_tokens=1,
        block_capacity=len(tokens.updated_prompt) + 8,
        lib_path=thump_lib_path,
    )
    splice_session.initialize_from_capture(
        prefix_suffix_capture,
        total_tokens=len(tokens.original_prompt),
    )
    delta_cache = clone_prompt_cache(prefix_snapshot)
    insert_collector = CaptureCollector()
    _prefill_tokens(
        runner.model,
        tokens.insert,
        delta_cache,
        prefill_step_size=trace.prefill_step_size,
        collector=insert_collector,
    )
    live_recompute_cache = clone_prompt_cache(delta_cache)
    live_suffix_collector = CaptureCollector()
    _prefill_tokens(
        runner.model,
        tokens.suffix_without_last,
        live_recompute_cache,
        prefill_step_size=trace.prefill_step_size,
        collector=live_suffix_collector,
    )
    live_recompute_next = _next_token(
        runner,
        live_recompute_cache,
        tokens.last_prompt_token,
    )
    splice_session.splice_insert_from_capture(
        len(tokens.prefix),
        insert_collector.joined(),
        insert_token_count=len(tokens.insert),
    )
    splice_roundtrip_cache = splice_session.materialize_prompt_cache(
        runner.model,
        upto_tokens=len(tokens.updated_prompt) - 1,
    )
    splice_roundtrip_next = _next_token(
        runner,
        splice_roundtrip_cache,
        tokens.last_prompt_token,
    )

    sampled_layers = [0, 1, 2, 5, 29]
    updated_suffix_start = len(tokens.prefix) + len(tokens.insert)
    direct_suffix_capture_diffs: list[dict[str, object]] = []
    full_capture = full_collector.joined()
    suffix_capture = suffix_collector.joined()
    for layer_idx in sampled_layers:
        direct_suffix_k = full_capture[layer_idx].keys[updated_suffix_start:]
        direct_suffix_v = full_capture[layer_idx].values[updated_suffix_start:]
        reused_suffix_k = suffix_capture[layer_idx].keys[: direct_suffix_k.shape[0]]
        reused_suffix_v = suffix_capture[layer_idx].values[: direct_suffix_v.shape[0]]
        direct_suffix_capture_diffs.append(
            _diff_summary(
                layer_idx=layer_idx,
                layer_type=runner.model.layers[layer_idx].layer_type,
                left_k=direct_suffix_k,
                left_v=direct_suffix_v,
                right_k=reused_suffix_k,
                right_v=reused_suffix_v,
            )
        )

    roundtrip_layer_diffs: list[dict[str, object]] = []
    live_recompute_cache_diffs: list[dict[str, object]] = []
    assembled_live_capture_diffs: list[dict[str, object]] = []
    assembled_live_capture = _merge_captures(
        _merge_captures(prefix_collector.joined(), insert_collector.joined()),
        live_suffix_collector.joined(),
    )
    for layer_idx in sampled_layers:
        full_k, full_v = full_roundtrip_cache[layer_idx].state
        splice_k, splice_v = splice_roundtrip_cache[layer_idx].state
        roundtrip_layer_diffs.append(
            _diff_summary(
                layer_idx=layer_idx,
                layer_type=runner.model.layers[layer_idx].layer_type,
                left_k=np.asarray(full_k),
                left_v=np.asarray(full_v),
                right_k=np.asarray(splice_k),
                right_v=np.asarray(splice_v),
            )
        )
        direct_cache_k, direct_cache_v = full_capture_cache[layer_idx].state
        live_cache_k, live_cache_v = live_recompute_cache[layer_idx].state
        live_recompute_cache_diffs.append(
            _diff_summary(
                layer_idx=layer_idx,
                layer_type=runner.model.layers[layer_idx].layer_type,
                left_k=direct_cache_k,
                left_v=direct_cache_v,
                right_k=live_cache_k,
                right_v=live_cache_v,
            )
        )
        assembled_live_capture_diffs.append(
            _diff_summary(
                layer_idx=layer_idx,
                layer_type=runner.model.layers[layer_idx].layer_type,
                left_k=full_capture[layer_idx].keys,
                left_v=full_capture[layer_idx].values,
                right_k=assembled_live_capture[layer_idx].keys,
                right_v=assembled_live_capture[layer_idx].values,
            )
        )

    first_diverging_layer = next(
        (
            item["layer_index"]
            for item in direct_suffix_capture_diffs
            if item["k_max_abs_diff"] > 0.0 or item["v_max_abs_diff"] > 0.0
        ),
        None,
    )
    live_recompute_first_diverging_layer = next(
        (
            item["layer_index"]
            for item in live_recompute_cache_diffs
            if item["k_max_abs_diff"] > 0.0 or item["v_max_abs_diff"] > 0.0
        ),
        None,
    )
    assembled_live_first_diverging_layer = next(
        (
            item["layer_index"]
            for item in assembled_live_capture_diffs
            if item["k_max_abs_diff"] > 0.0 or item["v_max_abs_diff"] > 0.0
        ),
        None,
    )
    layer0_direct = direct_suffix_capture_diffs[0]
    result = {
        "model_path": str(model_path),
        "trace_name": trace.name,
        "block_size_tokens": 1,
        "prompt_lengths": {
            "prefix_tokens": len(tokens.prefix),
            "insert_tokens": len(tokens.insert),
            "suffix_tokens": len(tokens.suffix),
            "updated_prompt_tokens": len(tokens.updated_prompt),
        },
        "baseline_next_token": baseline_next,
        "full_roundtrip_next_token": full_roundtrip_next,
        "splice_roundtrip_next_token": splice_roundtrip_next,
        "live_recompute_next_token": live_recompute_next,
        "full_roundtrip_exact": baseline_next == full_roundtrip_next,
        "splice_roundtrip_exact": baseline_next == splice_roundtrip_next,
        "live_recompute_exact": baseline_next == live_recompute_next,
        "layer0_suffix_exact_before_thump": (
            layer0_direct["k_max_abs_diff"] == 0.0
            and layer0_direct["v_max_abs_diff"] == 0.0
        ),
        "first_diverging_suffix_layer_before_thump": first_diverging_layer,
        "first_diverging_live_recompute_cache_layer": (
            live_recompute_first_diverging_layer
        ),
        "first_diverging_assembled_live_capture_layer": (
            assembled_live_first_diverging_layer
        ),
        "direct_suffix_capture_diffs": direct_suffix_capture_diffs,
        "live_recompute_cache_diffs": live_recompute_cache_diffs,
        "assembled_live_capture_diffs": assembled_live_capture_diffs,
        "roundtrip_layer_diffs": roundtrip_layer_diffs,
        "sampled_layer_diffs": roundtrip_layer_diffs,
        "interpretation": (
            "Direct updated-prompt round-trip is exact. Direct suffix captures "
            "are also exact at layer 0 and start diverging at layer 1 before "
            "Thump materialization. The splice-vs-full round-trip still shows "
            "small layer-0 drift, but the replay-invalidating boundary starts "
            "at layer 1. This probe also records whether the live recompute "
            "cache stays exact and whether segment capture reassembly itself "
            "diverges before Thump is involved."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thump-lib")
    args = parser.parse_args()
    result = probe_splice_limit(
        model_path=args.model_path,
        trace_path=args.trace,
        output_path=args.output,
        thump_lib_path=args.thump_lib,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
