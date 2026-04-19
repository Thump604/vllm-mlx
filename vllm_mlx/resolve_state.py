# SPDX-License-Identifier: Apache-2.0
"""Resolve registry.yaml + runtime-state.json into launcher shell variables.

Phase 2 bridge: reads the Phase 1 contract schemas and emits the same
shell variable assignments that the mode.json inline reader in
bin/start-vllm-mlx produces. This lets the launcher switch between the
two config paths without changing any downstream flag logic.

Usage from bash (called by bin/start-vllm-mlx):
    "$PYTHON" -m vllm_mlx.resolve_state "$REGISTRY_FILE" "$STATE_FILE"

Emits shell-safe variable assignments to stdout, one per line.
Exit 0 on success, 1 on validation failure (printed to stderr).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .runtime_config import (
    load_registry_document,
    load_runtime_state,
    resolve_runtime_state,
)


def _sq(value: str) -> str:
    """Shell-safe single-quoted string."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def emit_shell_vars(registry_path: str, state_path: str) -> None:
    """Load, resolve, and emit shell variable assignments."""
    registry = load_registry_document(registry_path)
    state = load_runtime_state(state_path)
    resolved = resolve_runtime_state(registry, state)

    # If no active model preset, emit NONE so the launcher can handle it
    if resolved.model is None:
        print("MODE_MODEL=NONE")
        print("SERVED_MODEL_NAME=''")
        print("THINK=true")
        print("TOOL_PARSER=qwen3_coder")
        print("REASONING_PARSER=qwen3")
        print("USE_MLLM=true")
        print("USE_CONT_BATCH=false")
        print("USE_PREFIX_CACHE=false")
        print("USE_MTP=false")
        print("CACHE_MEMORY_MB=''")
        print("MAX_KV_SIZE=''")
        print("PREFILL_STEP_SIZE=''")
        print("SPECPREFILL_ENABLED=false")
        print("SPECPREFILL_THRESHOLD=''")
        print("SPECPREFILL_KEEP_PCT=''")
        print("SPECPREFILL_DRAFT_MODEL=''")
        print("MODELS_CONFIG=''")
        print("MODE_TEMP=''")
        print("MODE_TOP_P=''")
        print("MODE_TOP_K=''")
        print("MODE_MIN_P=''")
        print("MODE_PRESENCE_PENALTY=''")
        print("MODE_REPETITION_PENALTY=''")
        print("KV_QUANTIZE=false")
        print("KV_QUANTIZE_BITS='8'")
        print("TEXT_BATCH_SCHEDULER_CANARY=false")
        print("SCHED_POLICY='fifo'")
        print("SCHED_HEADROOM_GB='8'")
        print("SPECPREFILL_CHUNK_SIZE=''")
        return

    model = resolved.model
    preset = resolved.model_preset
    sp = model.serving_profile

    # Preserve the full model source path so the launcher can resolve it
    # directly.  The old mode.json reader uses basename under MODELS_ROOT,
    # but the registry contract allows arbitrary absolute paths.
    mode_model = model.source

    # Served model name is the registry ID
    served_name = model.id

    # Thinking default from serving profile, overridable by preset
    think = True
    if sp.enable_thinking_default is not None:
        think = sp.enable_thinking_default
    if preset and preset.sampling_profile.enable_thinking is not None:
        think = preset.sampling_profile.enable_thinking

    # Parsers from serving profile
    tool_parser = sp.tool_call_parser or "qwen3_coder"
    reasoning_parser = sp.reasoning_parser or "qwen3"

    # Engine flags from serving profile
    use_mllm = sp.force_mllm if sp.force_mllm is not None else model.multimodal
    use_cb = sp.continuous_batching or False
    use_prefix_cache = use_cb  # prefix cache requires continuous batching
    use_mtp = model.supports_mtp

    # Prefill step size
    prefill_ss = sp.prefill_step_size

    # SpecPrefill
    specprefill = sp.specprefill
    sp_enabled = specprefill.enabled if specprefill else False
    sp_threshold = specprefill.threshold if specprefill else None
    sp_keep_pct = specprefill.keep_pct if specprefill else None
    sp_draft = None
    if model.draft_model is not None:
        sp_draft = model.draft_model.source

    # Sampling from preset (if set)
    mode_temp = None
    mode_top_p = None
    mode_top_k = None
    mode_min_p = None
    mode_presence_penalty = None
    mode_repetition_penalty = None
    if preset:
        mode_temp = preset.sampling_profile.temperature
        mode_top_p = preset.sampling_profile.top_p
        mode_top_k = preset.sampling_profile.top_k
        mode_min_p = preset.sampling_profile.min_p
        mode_presence_penalty = preset.sampling_profile.presence_penalty
        mode_repetition_penalty = preset.sampling_profile.repetition_penalty

    # KV quantization from model capabilities
    kv_quant = model.supports_kv_quant

    def _val(v):
        if v is None:
            return "''"
        return _sq(str(v))

    print(f"MODE_MODEL={_sq(mode_model)}")
    print(f"SERVED_MODEL_NAME={_sq(served_name)}")
    print(f"THINK={_sq(str(think).lower())}")
    print(f"TOOL_PARSER={_sq(tool_parser)}")
    print(f"REASONING_PARSER={_sq(reasoning_parser)}")
    print(f"USE_MLLM={_sq(str(use_mllm).lower())}")
    print(f"USE_CONT_BATCH={_sq(str(use_cb).lower())}")
    print(f"USE_PREFIX_CACHE={_sq(str(use_prefix_cache).lower())}")
    print(f"USE_MTP={_sq(str(use_mtp).lower())}")
    print(f"CACHE_MEMORY_MB={_val(None)}")
    print(f"MAX_KV_SIZE={_val(None)}")
    print(f"PREFILL_STEP_SIZE={_val(prefill_ss)}")
    print(f"SPECPREFILL_ENABLED={_sq(str(sp_enabled).lower())}")
    print(f"SPECPREFILL_THRESHOLD={_val(sp_threshold)}")
    print(f"SPECPREFILL_KEEP_PCT={_val(sp_keep_pct)}")
    print(f"SPECPREFILL_DRAFT_MODEL={_val(sp_draft)}")
    print(f"MODELS_CONFIG={_val(None)}")
    print(f"MODE_TEMP={_val(mode_temp)}")
    print(f"MODE_TOP_P={_val(mode_top_p)}")
    print(f"MODE_TOP_K={_val(mode_top_k)}")
    print(f"MODE_MIN_P={_val(mode_min_p)}")
    print(f"MODE_PRESENCE_PENALTY={_val(mode_presence_penalty)}")
    print(f"MODE_REPETITION_PENALTY={_val(mode_repetition_penalty)}")
    print(f"KV_QUANTIZE={_sq(str(kv_quant).lower())}")
    print(f"KV_QUANTIZE_BITS={_sq('8')}")
    print(f"TEXT_BATCH_SCHEDULER_CANARY=false")
    print(f"SCHED_POLICY={_sq('fifo')}")
    print(f"SCHED_HEADROOM_GB={_sq('8')}")
    print(f"SPECPREFILL_CHUNK_SIZE={_val(None)}")


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python -m vllm_mlx.resolve_state <registry.yaml> <runtime-state.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    registry_path = sys.argv[1]
    state_path = sys.argv[2]

    if not Path(registry_path).is_file():
        print(f"Registry file not found: {registry_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(state_path).is_file():
        print(f"State file not found: {state_path}", file=sys.stderr)
        sys.exit(1)

    try:
        emit_shell_vars(registry_path, state_path)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"Registry/state resolution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
