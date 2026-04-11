#!/usr/bin/env python3
"""Probe the clean refresh boundary on the offline Gemma 4 Thump replay slice."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from vllm_mlx.thump.replay import (
    CaptureCollector,
    ReplayRunner,
    ReplayTrace,
    _merge_captures,
    _prefill_tokens,
    clone_prompt_cache,
)
from vllm_mlx.thump.session import SessionSubstrate

SAMPLED_LAYERS = (0, 1, 2, 5, 29)


def _cache_diff_summary(
    left_cache: list[object],
    right_cache: list[object],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for layer_idx in SAMPLED_LAYERS:
        left_k, left_v = left_cache[layer_idx].state
        right_k, right_v = right_cache[layer_idx].state
        left_k_arr = mx.array(left_k).astype(mx.float32)
        left_v_arr = mx.array(left_v).astype(mx.float32)
        right_k_arr = mx.array(right_k).astype(mx.float32)
        right_v_arr = mx.array(right_v).astype(mx.float32)
        k_diff = mx.abs(left_k_arr - right_k_arr)
        v_diff = mx.abs(left_v_arr - right_v_arr)
        mx.eval(k_diff, v_diff)
        rows.append(
            {
                "layer_index": layer_idx,
                "k_max_abs_diff": float(mx.max(k_diff).item()),
                "k_mean_abs_diff": float(mx.mean(k_diff).item()),
                "v_max_abs_diff": float(mx.max(v_diff).item()),
                "v_mean_abs_diff": float(mx.mean(v_diff).item()),
            }
        )
    return rows


def _build_splice_session(
    runner: ReplayRunner,
    trace: ReplayTrace,
):
    tokens = runner.build_trace_tokens(trace)

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
    merged_capture = _merge_captures(
        prefix_collector.joined(),
        suffix_collector.joined(),
    )

    session = SessionSubstrate.from_gemma4_model(
        runner.model,
        block_size_tokens=1,
        block_capacity=len(tokens.updated_prompt) + 8,
        lib_path=runner.thump_lib_path,
    )
    session.initialize_from_capture(
        merged_capture,
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
    session.splice_insert_from_capture(
        len(tokens.prefix),
        insert_collector.joined(),
        insert_token_count=len(tokens.insert),
    )
    return tokens, session


def probe_refresh_boundary(
    *,
    model_path: str | Path,
    trace_path: str | Path,
    output_path: str | Path,
    thump_lib_path: str | Path | None = None,
    refresh_values: list[int] | None = None,
) -> dict[str, object]:
    trace = ReplayTrace.from_path(trace_path)
    runner = ReplayRunner(
        model_path,
        thump_lib_path=thump_lib_path,
        block_size_tokens=1,
    )
    tokens = runner.build_trace_tokens(trace)
    prompt_without_last = tokens.updated_prompt_for_edit(0)[:-1]
    prefix_insert_tokens = len(tokens.prefix) + len(tokens.insert)
    first_suffix_token = tokens.suffix_without_last[0]
    refresh_values = refresh_values or [1101, 1102]

    direct_prefix_insert = runner.model.make_cache()
    _prefill_tokens(
        runner.model,
        prompt_without_last[:prefix_insert_tokens],
        direct_prefix_insert,
        prefill_step_size=trace.prefill_step_size,
    )
    direct_prefix_insert_plus_one = clone_prompt_cache(direct_prefix_insert)
    _prefill_tokens(
        runner.model,
        [first_suffix_token],
        direct_prefix_insert_plus_one,
        prefill_step_size=trace.prefill_step_size,
    )

    _tokens_for_session, splice_session = _build_splice_session(runner, trace)
    materialized_prefix_insert = splice_session.materialize_prompt_cache(
        runner.model,
        upto_tokens=prefix_insert_tokens,
    )
    materialized_prefix_insert_plus_one = splice_session.materialize_prompt_cache(
        runner.model,
        upto_tokens=prefix_insert_tokens + 1,
    )
    materialized_prefix_insert_then_first = clone_prompt_cache(
        materialized_prefix_insert
    )
    _prefill_tokens(
        runner.model,
        [first_suffix_token],
        materialized_prefix_insert_then_first,
        prefill_step_size=trace.prefill_step_size,
    )
    splice_session.close()

    comparisons = []
    for refresh_value in refresh_values:
        variant_trace = replace(trace, thump_refresh_tail_tokens=refresh_value)
        comparison = runner.run_comparison(variant_trace).to_dict()
        isolation = comparison["isolation"]
        comparisons.append(
            {
                "refresh_tokens": refresh_value,
                "upto_tokens": len(prompt_without_last)
                - min(refresh_value, len(prompt_without_last)),
                "baseline_vs_direct_full_prompt_exact_match": isolation[
                    "baseline_vs_direct_full_prompt_exact_match"
                ],
                "thump_vs_direct_full_prompt_exact_match": isolation[
                    "thump_vs_direct_full_prompt_exact_match"
                ],
                "output_exact_match": isolation["output_exact_match"],
                "baseline_session_total_ms": isolation["baseline"]["session_total_ms"],
                "thump_session_total_ms": isolation["thump"]["session_total_ms"],
                "baseline_continuation_ms": isolation["baseline"][
                    "continuation_latency_ms"
                ],
                "thump_continuation_ms": isolation["thump"]["continuation_latency_ms"],
                "thump_re_prefill_avoided_tokens": isolation["thump"][
                    "re_prefill_avoided_tokens"
                ],
                "thump_fallback_count": isolation["thump"]["fallback_count"],
            }
        )

    result = {
        "trace_name": trace.name,
        "prefix_tokens": len(tokens.prefix),
        "insert_tokens": len(tokens.insert),
        "suffix_without_last_tokens": len(tokens.suffix_without_last),
        "prompt_without_last_tokens": len(prompt_without_last),
        "first_suffix_token": first_suffix_token,
        "sampled_layers": list(SAMPLED_LAYERS),
        "boundary_materialization": {
            "prefix_insert_upto_tokens": prefix_insert_tokens,
            "prefix_insert_plus_one_upto_tokens": prefix_insert_tokens + 1,
            "direct_vs_materialized_prefix_insert": _cache_diff_summary(
                direct_prefix_insert,
                materialized_prefix_insert,
            ),
            "direct_vs_materialized_prefix_insert_plus_one": _cache_diff_summary(
                direct_prefix_insert_plus_one,
                materialized_prefix_insert_plus_one,
            ),
            "direct_vs_materialized_prefix_insert_then_first_suffix_prefill": _cache_diff_summary(
                direct_prefix_insert_plus_one,
                materialized_prefix_insert_then_first,
            ),
        },
        "refresh_comparisons": comparisons,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thump-lib")
    parser.add_argument(
        "--refresh-values",
        default="1101,1102",
        help="Comma-separated refresh values to compare",
    )
    args = parser.parse_args()

    refresh_values = [
        int(part.strip()) for part in args.refresh_values.split(",") if part.strip()
    ]
    result = probe_refresh_boundary(
        model_path=args.model_path,
        trace_path=args.trace,
        output_path=args.output,
        thump_lib_path=args.thump_lib,
        refresh_values=refresh_values,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
