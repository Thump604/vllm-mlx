# Qwen 35B DFlash Code-Path Comparison

This note tracks the first Qwen 3.6 35B-A3B DFlash integration path for
`waybarrios/vllm-mlx#502`.

## Draft Contract

- Draft model: `z-lab/Qwen3.6-35B-A3B-DFlash`
- Draft revision: `42d3b34d588423cdae7ba8f53a8cf7789346a719`
- Source of truth: draft `config.json`
- Required `dflash_config.target_layer_ids`: `[1, 10, 19, 28, 37]`
- Required `num_target_layers`: `40`
- Required `mask_token_id`: `248070`
- Required `block_size`: `16`
- Required `sliding_window`: `null`
- Required `use_sliding_window`: `false`

The adapter rejects mismatches. It does not infer layer ids, substitute draft
families, or silently enable sliding-window drafts.

## Reference Surface

`mlx-vlm` DFlash runs through the speculative round loop in
`mlx_vlm/speculative/dflash.py`. The reference path expects the target model to
return configured hidden states through `capture_layer_ids`, then verifies the
draft block against the target and rolls target cache state back after rejected
tokens.

## vllm-mlx Surface Before This Patch

Current `vllm-mlx` SimpleEngine text routing already has:

- TextModel-direct generation through `SimpleEngine._stream_generate_text`
- serialized MLX execution through `_run_blocking_serialized`
- native MTP as a separate speculative path
- SpecPrefill as a separate draft/prefill path
- prompt-cache trim support through `mlx_lm.models.cache`

Current `vllm-mlx` does not expose `capture_layer_ids` on the TextModel route,
so this patch uses request-scoped layer wrappers in `vllm_mlx/dflash.py` for
the first SimpleEngine/TextModel milestone. The wrappers are installed only
inside the DFlash request context and restored in `finally`/context-manager
closeout.

## Added Local Code Paths

- `vllm_mlx/dflash.py`
  - DFlash draft config parsing and Qwen 35B contract validation
  - request-local target layer capture
  - DFlash draft model definition and weight loading
  - draft/verify/rollback loop with metadata counters
  - rollback through prompt-cache trim when available, and through
    snapshot/restore plus committed-token replay for hybrid Qwen caches that
    expose non-trimmable `ArraysCache` state
- `vllm_mlx/engine/simple.py`
  - default-off DFlash backend loading
  - TextModel route dispatch when `speculative_method="dflash"`
  - request fallback to normal generation when unsupported per-request controls
    are active
  - stats metadata for draft/accepted/rejected tokens
- `vllm_mlx/server.py`, `vllm_mlx/cli.py`, `vllm_mlx/lifecycle.py`,
  `vllm_mlx/model_registry.py`
  - explicit config plumbing for `speculative_method`,
    `dflash_draft_model`, `dflash_block_size`, and
    `dflash_draft_sliding_window_size`

## Explicit Non-Claims

This patch does not add:

- CB/BatchedEngine DFlash routing
- media/VLM DFlash routing
- Gemma DFlash
- Qwen 27B DFlash
- Jobs/Ops mode exposure
- resident/default mode changes
- runtime qualification evidence

Qwen 27B remains blocked until gated draft access is available and its
sliding-window attention requirements are inspected.

## Promotion Gate For CB/Batched

SimpleEngine/TextModel must first produce artifact-backed evidence with:

- final content present
- accepted drafts greater than zero
- clean parser/finalization behavior
- no repetition or corruption
- target cache rollback/cleanup intact
- cancellation/admission safety intact
- baseline comparison for wall time, throughput, and peak memory

Only after that evidence should DFlash be wired into CB/Batched routes.
