# SPDX-License-Identifier: Apache-2.0
"""Probe LiteRT drafter contract compatibility against a local Gemma 4 MLX model."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.utils import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _supports_parameter(callable_obj: Any, name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _resolve_text_module(model: Any) -> Any:
    candidates = (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "language_model", None),
        getattr(model, "text_model", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _normalize_dtype(dtype: str) -> str:
    normalized = str(dtype).upper()
    normalized = normalized.removeprefix("MLX.CORE.")
    return normalized


def _infer_layout(shape: tuple[int, ...], head_dim: int) -> str:
    if len(shape) != 4:
        return "unknown"
    if shape[-1] == head_dim:
        return "BHTD"
    if shape[-2] == head_dim:
        return "BHDT"
    return "unknown"


def _cache_probe_entry(
    runtime_cache: Any,
    contract_spec: dict[str, Any],
) -> dict[str, Any]:
    key_state, value_state = runtime_cache.state
    key_shape = tuple(int(dim) for dim in key_state.shape)
    value_shape = tuple(int(dim) for dim in value_state.shape)
    head_dim = int(contract_spec["head_dim"])
    key_layout = _infer_layout(key_shape, head_dim)
    value_layout = _infer_layout(value_shape, head_dim)
    key_dtype = _normalize_dtype(str(key_state.dtype))
    value_dtype = _normalize_dtype(str(value_state.dtype))
    contract_key_dtype = _normalize_dtype(contract_spec["key"]["dtype"])
    contract_value_dtype = _normalize_dtype(contract_spec["value"]["dtype"])

    return {
        "layer_index": int(contract_spec["layer_index"]),
        "runtime_source_layer_index": int(contract_spec["runtime_source_layer_index"]),
        "source_layer_type": contract_spec["source_layer_type"],
        "runtime_cache_type": type(runtime_cache).__name__,
        "runtime_offset": int(getattr(runtime_cache, "offset", 0)),
        "runtime_meta_state": getattr(runtime_cache, "meta_state", None),
        "runtime_key_shape": list(key_shape),
        "runtime_value_shape": list(value_shape),
        "runtime_key_dtype": key_dtype,
        "runtime_value_dtype": value_dtype,
        "runtime_key_layout": key_layout,
        "runtime_value_layout": value_layout,
        "contract_key_shape": contract_spec["key"]["shape"],
        "contract_value_shape": contract_spec["value"]["shape"],
        "contract_key_dtype": contract_key_dtype,
        "contract_value_dtype": contract_value_dtype,
        "contract_key_layout": contract_spec["key_layout"],
        "contract_value_layout": contract_spec["value_layout"],
        "contract_time_capacity": int(contract_spec["time_capacity"]),
        "key_dtype_match": key_dtype == contract_key_dtype,
        "value_dtype_match": value_dtype == contract_value_dtype,
        "key_layout_match": key_layout == contract_spec["key_layout"],
        "value_layout_match": value_layout == contract_spec["value_layout"],
        "fixed_capacity_match": key_shape[2] == int(contract_spec["time_capacity"]),
    }


def main() -> None:
    args = parse_args()
    contract = _load_json(args.contract_json)
    model, _ = load(str(args.model_path))

    call_signature = str(inspect.signature(model.__call__))
    model_impl = _resolve_text_module(model)

    cache = model.make_cache()
    tokens = mx.array([[1, 2, 3]], dtype=mx.uint32)
    model(tokens, cache=cache)

    cache_probe = [
        _cache_probe_entry(cache[int(spec["runtime_source_layer_index"])], spec)
        for spec in contract["kv_cache_specs"]
    ]

    payload = {
        "model_path": str(args.model_path),
        "contract_json": str(args.contract_json),
        "model_surface": {
            "class_name": type(model).__name__,
            "call_signature": call_signature,
            "supports_return_hidden": _supports_parameter(
                model.__call__, "return_hidden"
            ),
            "has_mtp_forward": callable(getattr(model, "mtp_forward", None)),
            "has_make_mtp_cache": callable(getattr(model, "make_mtp_cache", None)),
            "has_embed_tokens": hasattr(model_impl, "embed_tokens"),
            "has_get_per_layer_inputs": callable(
                getattr(model_impl, "get_per_layer_inputs", None)
            ),
            "has_project_per_layer_inputs": callable(
                getattr(model_impl, "project_per_layer_inputs", None)
            ),
            "hidden_size_per_layer_input": int(
                getattr(model_impl, "hidden_size_per_layer_input", 0)
            ),
        },
        "litert_contract_surface": {
            "activations_formula": contract["activations_formula"],
            "next_token_ids_required": bool(contract["next_token_ids_required"]),
            "requires_draft_token_embedding_lookup": bool(
                contract["requires_draft_token_embedding_lookup"]
            ),
            "requires_projected_activations_feedback": bool(
                contract["requires_projected_activations_feedback"]
            ),
            "requires_explicit_kv_cache_adapter": bool(
                contract["requires_explicit_kv_cache_adapter"]
            ),
            "requires_external_verifier_loop": bool(
                contract["requires_external_verifier_loop"]
            ),
            "projected_state_cache_field": contract["projected_state_cache_field"],
            "runtime_adapter_strategy": contract["runtime_adapter_strategy"],
        },
        "cache_probe": cache_probe,
    }

    native_mtp_surface_compatible = (
        payload["model_surface"]["supports_return_hidden"]
        and payload["model_surface"]["has_mtp_forward"]
        and payload["model_surface"]["has_make_mtp_cache"]
    )
    native_cache_surface_compatible = all(
        entry["key_dtype_match"]
        and entry["value_dtype_match"]
        and entry["key_layout_match"]
        and entry["value_layout_match"]
        and entry["fixed_capacity_match"]
        for entry in cache_probe
    )
    payload["summary"] = {
        "native_mtp_surface_compatible": native_mtp_surface_compatible,
        "native_cache_surface_compatible": native_cache_surface_compatible,
        "drop_in_native_adapter_viable": (
            native_mtp_surface_compatible and native_cache_surface_compatible
        ),
        "blocker": (
            "Current Gemma MLX model surface is not a drop-in target for the "
            "published LiteRT drafter contract: it lacks the native MTP hooks "
            "and exposes dynamic BF16 caches instead of the contract's "
            "single-buffer INT8 cache surface."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
