# SPDX-License-Identifier: Apache-2.0
"""Convert extracted Gemma 4 LiteRT MTP tensors into an MLX adapter payload."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class _MappingRule:
    pattern: str
    output_template: str
    quantize: bool = True

    def match(self, source_name: str) -> str | None:
        matched = re.match(self.pattern, source_name)
        if matched is None:
            return None
        output = self.output_template
        for index, value in enumerate(matched.groups(), start=1):
            output = output.replace(f"{{{index}}}", value)
        return output


_MAPPING_RULES = (
    _MappingRule(
        r"^MtpDrafterModel\.mtp_pre_project/.*$",
        "mtp_drafter.mtp_pre_project.weight",
    ),
    _MappingRule(
        r"^MtpDrafterModel\.mtp_post_project/.*$",
        "mtp_drafter.mtp_post_project.weight",
    ),
    _MappingRule(
        r"^MtpDrafterModel\.decode_softmax/.*/embedder\.decode/composite$",
        "mtp_drafter.decode_softmax.weight",
    ),
    _MappingRule(
        r"^MtpDrafterModel\.decode_softmax/.*/div1$",
        "mtp_drafter.decode_softmax.div1",
        quantize=False,
    ),
    _MappingRule(
        r"^MtpDrafterModel\.decode_softmax/.*/div$",
        "mtp_drafter.decode_softmax.div",
        quantize=False,
    ),
    _MappingRule(
        r"^layer_(\d+)/.*q_einsum.*$",
        "mtp_drafter.layers.{1}.attn.q_proj.weight",
    ),
    _MappingRule(
        r"^layer_(\d+)/.*attn_vec_einsum.*$",
        "mtp_drafter.layers.{1}.attn.o_proj.weight",
    ),
    _MappingRule(
        r"^layer_(\d+)/.*gating_einsum1.*$",
        "mtp_drafter.layers.{1}.mlp.gating_einsum1.weight",
    ),
    _MappingRule(
        r"^layer_(\d+)/.*gating_einsum2.*$",
        "mtp_drafter.layers.{1}.mlp.gating_einsum2.weight",
    ),
    _MappingRule(
        r"^layer_(\d+)/.*mlp/linear.*$",
        "mtp_drafter.layers.{1}.mlp.linear.weight",
    ),
    _MappingRule(
        r"^layer_(\d+)/.*maybe_rope/div;?$",
        "mtp_drafter.layers.{1}.attn.rope.div",
        quantize=False,
    ),
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _map_output_key(source_name: str) -> tuple[str, bool] | None:
    for rule in _MAPPING_RULES:
        output_key = rule.match(source_name)
        if output_key is not None:
            return output_key, rule.quantize
    return None


def _load_numpy_array(entry: dict[str, Any]) -> np.ndarray:
    return np.load(entry["array_path"], allow_pickle=False)


def _load_quantized_array(entry: dict[str, Any]) -> np.ndarray:
    array = _load_numpy_array(entry)
    quant = entry.get("quantization")
    if quant is None:
        return array.astype(np.float32, copy=False)

    scales_path = Path(entry["array_path"]).with_suffix(".scales.npy")
    zero_points_path = Path(entry["array_path"]).with_suffix(".zero_points.npy")
    scales = np.load(scales_path, allow_pickle=False).astype(np.float32, copy=False)
    zero_points = np.load(zero_points_path, allow_pickle=False).astype(
        np.float32, copy=False
    )
    quantized_dimension = int(quant["quantized_dimension"])
    shape = [1] * array.ndim
    shape[quantized_dimension] = scales.shape[0]
    return (array.astype(np.float32) - zero_points.reshape(shape)) * scales.reshape(
        shape
    )


def _mx_array_from_entry(entry: dict[str, Any]) -> mx.array:
    if entry.get("quantization") is not None:
        array = _load_quantized_array(entry)
        return mx.array(array.astype(np.float32, copy=False))

    numpy_array = _load_numpy_array(entry)
    tensor_type = str(entry["type"])
    if tensor_type == "FLOAT32":
        return mx.array(numpy_array.astype(np.float32, copy=False))
    if tensor_type == "INT32":
        return mx.array(numpy_array.astype(np.int32, copy=False))
    return mx.array(numpy_array)


def convert_extracted_tensors(
    extract_summary_path: str | Path,
    *,
    contract_path: str | Path,
    output_dir: str | Path,
    target_bits: int,
    group_size: int,
) -> dict[str, Any]:
    mx.set_default_device(mx.cpu)

    extract_summary_path = Path(extract_summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(extract_summary_path)
    contract = _load_json(contract_path)
    converted_weights: dict[str, mx.array] = {}
    converted_tensors: list[dict[str, Any]] = []
    ignored_tensors: list[dict[str, Any]] = []

    for entry in summary.get("tensors", []):
        mapped = _map_output_key(str(entry["name"]))
        if mapped is None:
            ignored_tensors.append(
                {
                    "name": entry["name"],
                    "shape": entry["shape"],
                    "type": entry["type"],
                }
            )
            continue

        output_key, should_quantize = mapped
        array = _mx_array_from_entry(entry)
        weight_keys: list[str]
        if should_quantize and array.ndim >= 2 and array.shape[-1] >= group_size:
            q_weight, q_scales, q_biases = mx.quantize(
                array, group_size=group_size, bits=target_bits
            )
            converted_weights[output_key] = q_weight
            converted_weights[output_key.replace(".weight", ".scales")] = q_scales
            converted_weights[output_key.replace(".weight", ".biases")] = q_biases
            weight_keys = [
                output_key,
                output_key.replace(".weight", ".scales"),
                output_key.replace(".weight", ".biases"),
            ]
        else:
            scalar_or_constant = (
                array.astype(mx.bfloat16) if array.dtype == mx.float32 else array
            )
            converted_weights[output_key] = scalar_or_constant
            weight_keys = [output_key]

        converted_tensors.append(
            {
                "source_name": entry["name"],
                "source_shape": entry["shape"],
                "source_type": entry["type"],
                "output_key": output_key,
                "saved_keys": weight_keys,
                "quantized": bool(
                    should_quantize
                    and array.ndim >= 2
                    and array.shape[-1] >= group_size
                ),
            }
        )

    shard_path = output_dir / "model-mtp-litert.safetensors"
    mx.save_safetensors(str(shard_path), converted_weights)

    payload = {
        "adapter_payload_format": "gemma4_litert_mtp_v1",
        "extract_summary_path": extract_summary_path,
        "contract_path": Path(contract_path),
        "output_shard": shard_path,
        "model_type": contract["model_type"],
        "target_bits": int(target_bits),
        "group_size": int(group_size),
        "converted_tensor_count": len(converted_tensors),
        "ignored_tensor_count": len(ignored_tensors),
        "saved_key_count": len(converted_weights),
        "converted_tensors": converted_tensors,
        "ignored_tensors": ignored_tensors,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(payload), indent=2) + "\n"
    )
    return payload
