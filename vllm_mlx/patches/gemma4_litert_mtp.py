# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 LiteRT MTP contract builder.

This module turns the extracted LiteRT/TFLite drafter interface into a
runtime-facing adapter contract that matches the existing MLX MTP entrypoint:

    mtp_forward(hidden_states, next_token_ids, mtp_cache) -> logits

The Gemma LiteRT drafter does not expose that exact signature. Public
LiteRT-LM source shows the real runtime loop concatenating the next-token
embedding with carried projected activations outside the graph, feeding
explicit KV cache tensors plus a small param tensor into the graph, and
running a separate draft+verify choreography around it. The functions here
formalize that adapter contract without pretending the graph is a drop-in
native MLX MTP head.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LiteRTTensorSpec:
    short_name: str
    tensor_name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def width(self) -> int:
        if not self.shape:
            raise ValueError(f"{self.short_name} has no shape")
        return int(self.shape[-1])


@dataclass(frozen=True)
class Gemma4LiteRTKVCacheSpec:
    layer_index: int
    runtime_source_layer_index: int
    source_layer_type: str
    key: LiteRTTensorSpec
    value: LiteRTTensorSpec
    time_capacity: int
    head_dim: int
    key_layout: str
    value_layout: str


@dataclass(frozen=True)
class Gemma4LiteRTMTPContract:
    signature_key: str
    model_type: str
    model_hidden_size: int
    model_vocab_size: int
    hidden_size_per_layer_input: int
    activations_input_size: int
    projected_activations_size: int
    logits_vocab_size: int
    input_pos_dtype: str
    mask_shape: tuple[int, ...]
    param_tensor_shape: tuple[int, ...]
    next_token_ids_required: bool
    requires_draft_token_embedding_lookup: bool
    requires_projected_activations_feedback: bool
    requires_explicit_kv_cache_adapter: bool
    requires_external_verifier_loop: bool
    activations_formula: str
    runtime_adapter_strategy: str
    projected_state_cache_field: str
    kv_cache_specs: tuple[Gemma4LiteRTKVCacheSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _resolve_text_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return model_config.get("text_config", model_config)


def _resolve_layer_types(text_config: dict[str, Any]) -> list[str]:
    layer_types = text_config.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        return [str(item) for item in layer_types]

    num_hidden_layers = int(text_config.get("num_hidden_layers", 0))
    sliding_window_pattern = int(text_config.get("sliding_window_pattern", 5))
    if num_hidden_layers <= 0:
        raise ValueError("Gemma text config missing num_hidden_layers")
    pattern = ["sliding_attention"] * (sliding_window_pattern - 1) + ["full_attention"]
    repeated = pattern * (num_hidden_layers // len(pattern) + 1)
    return repeated[:num_hidden_layers]


def _tensor_spec(entry: dict[str, Any], short_name: str) -> LiteRTTensorSpec:
    return LiteRTTensorSpec(
        short_name=short_name,
        tensor_name=str(entry["name"]),
        shape=tuple(int(dim) for dim in entry["shape"]),
        dtype=str(entry["type"]),
    )


def _find_signature(
    payload: dict[str, Any], key: str = "mtp_drafter"
) -> dict[str, Any]:
    for signature in payload.get("signature_defs", []):
        if signature.get("key") == key:
            return signature
    raise ValueError(f"LiteRT interface missing signature key={key!r}")


def _main_subgraph(payload: dict[str, Any]) -> dict[str, Any]:
    main = payload.get("main_subgraph")
    if not isinstance(main, dict):
        raise ValueError("LiteRT interface missing main_subgraph")
    return main


def _tensor_map(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = str(entry["name"])
        short = name
        if short.startswith("mtp_drafter_"):
            short = short.removeprefix("mtp_drafter_")
        if short.endswith(":0"):
            short = short[:-2]
        result[short] = entry
    return result


def _cache_specs(
    inputs_by_name: dict[str, dict[str, Any]],
    *,
    text_config: dict[str, Any],
) -> tuple[Gemma4LiteRTKVCacheSpec, ...]:
    layer_types = _resolve_layer_types(text_config)
    cache_layers = sorted(
        {
            int(name.split("_")[-1])
            for name in inputs_by_name
            if name.startswith("kv_cache_k_")
        }
    )
    specs: list[Gemma4LiteRTKVCacheSpec] = []
    for layer_index in cache_layers:
        key_name = f"kv_cache_k_{layer_index}"
        value_name = f"kv_cache_v_{layer_index}"
        if key_name not in inputs_by_name or value_name not in inputs_by_name:
            raise ValueError(
                f"LiteRT interface missing cache pair for layer {layer_index}"
            )
        key = _tensor_spec(inputs_by_name[key_name], key_name)
        value = _tensor_spec(inputs_by_name[value_name], value_name)
        if len(key.shape) != 4 or len(value.shape) != 4:
            raise ValueError(f"Unexpected cache rank for layer {layer_index}")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError(
                f"Cache batch/head prefix mismatch for layer {layer_index}"
            )
        if key.shape[2] != value.shape[3]:
            raise ValueError(f"Cache time dimension mismatch for layer {layer_index}")
        if key.shape[3] != value.shape[2]:
            raise ValueError(f"Cache head-dim mismatch for layer {layer_index}")
        specs.append(
            Gemma4LiteRTKVCacheSpec(
                layer_index=layer_index,
                runtime_source_layer_index=layer_index,
                source_layer_type=str(layer_types[layer_index]),
                key=key,
                value=value,
                time_capacity=int(key.shape[2]),
                head_dim=int(key.shape[3]),
                key_layout="BHTD",
                value_layout="BHDT",
            )
        )
    return tuple(specs)


def build_gemma4_litert_mtp_contract(
    interface_payload: dict[str, Any],
    model_config: dict[str, Any],
) -> Gemma4LiteRTMTPContract:
    text_config = _resolve_text_config(model_config)
    signature = _find_signature(interface_payload)
    main = _main_subgraph(interface_payload)
    inputs_by_name = _tensor_map(main.get("inputs", []))
    outputs_by_name = _tensor_map(main.get("outputs", []))

    required_inputs = ("activations", "input_pos", "mask", "param_tensor")
    required_outputs = ("StatefulPartitionedCall", "StatefulPartitionedCall:1")
    for name in required_inputs:
        if name not in inputs_by_name:
            raise ValueError(f"LiteRT interface missing required input {name!r}")

    activations = _tensor_spec(inputs_by_name["activations"], "activations")
    input_pos = _tensor_spec(inputs_by_name["input_pos"], "input_pos")
    mask = _tensor_spec(inputs_by_name["mask"], "mask")
    param_tensor = _tensor_spec(inputs_by_name["param_tensor"], "param_tensor")

    # Output names in the extracted interface remain StatefulPartitionedCall:*
    # while signature_defs give them semantic names. Prefer the tensor payload.
    logits_entry = next(
        (
            entry
            for entry in main.get("outputs", [])
            if entry.get("name") == "StatefulPartitionedCall:0"
        ),
        None,
    )
    projected_entry = next(
        (
            entry
            for entry in main.get("outputs", [])
            if entry.get("name") == "StatefulPartitionedCall:1"
        ),
        None,
    )
    if logits_entry is None or projected_entry is None:
        raise ValueError("LiteRT interface missing logits/projected outputs")
    logits = _tensor_spec(logits_entry, "logits")
    projected = _tensor_spec(projected_entry, "projected_activations")

    model_hidden_size = int(text_config["hidden_size"])
    model_vocab_size = int(text_config["vocab_size"])
    hidden_size_per_layer_input = int(text_config.get("hidden_size_per_layer_input", 0))

    if logits.width != model_vocab_size:
        raise ValueError(
            f"LiteRT logits width {logits.width} does not match model vocab {model_vocab_size}"
        )
    if projected.width != model_hidden_size:
        raise ValueError(
            "LiteRT projected activations width "
            f"{projected.width} does not match model hidden size {model_hidden_size}"
        )

    if activations.width == projected.width:
        activations_formula = "projected_activations"
        next_token_ids_required = False
    elif activations.width == model_hidden_size + projected.width:
        activations_formula = "concat(next_token_embedding, projected_activations)"
        next_token_ids_required = True
    else:
        raise ValueError(
            "LiteRT activations width does not line up with Gemma hidden sizes: "
            f"{activations.width} vs hidden={model_hidden_size}, projected={projected.width}"
        )
    projected_state_cache_field = "projected_activations"

    return Gemma4LiteRTMTPContract(
        signature_key=str(signature["key"]),
        model_type=str(
            text_config.get("model_type", model_config.get("model_type", ""))
        ),
        model_hidden_size=model_hidden_size,
        model_vocab_size=model_vocab_size,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        activations_input_size=activations.width,
        projected_activations_size=projected.width,
        logits_vocab_size=logits.width,
        input_pos_dtype=input_pos.dtype,
        mask_shape=mask.shape,
        param_tensor_shape=param_tensor.shape,
        next_token_ids_required=next_token_ids_required,
        requires_draft_token_embedding_lookup=next_token_ids_required,
        requires_projected_activations_feedback=True,
        requires_explicit_kv_cache_adapter=True,
        requires_external_verifier_loop=True,
        activations_formula=activations_formula,
        runtime_adapter_strategy=(
            "do not treat the LiteRT graph as a drop-in native MLX MTP head; "
            "look up the next-token embedding outside the graph, persist "
            "projected_activations between draft steps, maintain dedicated "
            "single-buffer INT8 LiteRT KV caches, and run the external "
            "draft+verify choreography around the signature"
        ),
        projected_state_cache_field=projected_state_cache_field,
        kv_cache_specs=_cache_specs(inputs_by_name, text_config=text_config),
    )


def build_contract_from_paths(
    interface_path: str | Path,
    model_config_path: str | Path,
) -> Gemma4LiteRTMTPContract:
    return build_gemma4_litert_mtp_contract(
        _load_json(interface_path),
        _load_json(model_config_path),
    )
