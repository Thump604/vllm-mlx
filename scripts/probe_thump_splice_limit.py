#!/usr/bin/env python3
"""Probe the current Thump replay limit on the offline Gemma 4 slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
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

    sampled_layers = [0, 5, 29]
    layer_diffs: list[dict[str, object]] = []
    for layer_idx in sampled_layers:
        full_k, full_v = full_roundtrip_cache[layer_idx].state
        splice_k, splice_v = splice_roundtrip_cache[layer_idx].state
        k_diff = mx.abs(full_k.astype(mx.float32) - splice_k.astype(mx.float32))
        v_diff = mx.abs(full_v.astype(mx.float32) - splice_v.astype(mx.float32))
        mx.eval(k_diff, v_diff)
        layer_diffs.append(
            {
                "layer_index": layer_idx,
                "layer_type": runner.model.layers[layer_idx].layer_type,
                "k_max_abs_diff": float(mx.max(k_diff).item()),
                "k_mean_abs_diff": float(mx.mean(k_diff).item()),
                "v_max_abs_diff": float(mx.max(v_diff).item()),
                "v_mean_abs_diff": float(mx.mean(v_diff).item()),
            }
        )

    result = {
        "model_path": str(model_path),
        "trace_name": trace.name,
        "block_size_tokens": 1,
        "baseline_next_token": baseline_next,
        "full_roundtrip_next_token": full_roundtrip_next,
        "splice_roundtrip_next_token": splice_roundtrip_next,
        "full_roundtrip_exact": baseline_next == full_roundtrip_next,
        "splice_roundtrip_exact": baseline_next == splice_roundtrip_next,
        "interpretation": (
            "Direct updated-prompt round-trip is exact, but prefix+suffix seed "
            "plus insert splice is still not parity-safe."
        ),
        "sampled_layer_diffs": layer_diffs,
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
