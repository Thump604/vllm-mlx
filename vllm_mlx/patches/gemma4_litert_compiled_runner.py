# SPDX-License-Identifier: Apache-2.0
"""Lower-level LiteRT compiled-model runner for Gemma 4 MTP probing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class Gemma4LiteRTCompiledProbeResult:
    logits: np.ndarray
    projected_activations: np.ndarray
    cache_input_mutation: dict[str, dict[str, int]]
    bool_write_supported: bool


class Gemma4LiteRTCompiledRunner:
    """Probe the lower-level CompiledModel/TensorBuffer API."""

    def __init__(self, *, compiled_model: Any, contract: dict[str, Any]):
        self._model = compiled_model
        self.contract = contract
        self.signature_key = str(contract["signature_key"])
        self.input_details = compiled_model.get_input_tensor_details(self.signature_key)
        self.output_names = tuple(
            compiled_model.get_signature_list()[self.signature_key]["outputs"]
        )
        self.token_embedding_size = int(contract["model_hidden_size"])
        self.projected_size = int(contract["projected_activations_size"])

    @classmethod
    def from_file(
        cls,
        *,
        model_path: str | Path,
        contract_path: str | Path,
    ) -> "Gemma4LiteRTCompiledRunner":
        try:
            from ai_edge_litert.compiled_model import (
                CompiledModel,
                HardwareAccelerator,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Gemma4LiteRTCompiledRunner requires ai-edge-litert in the active Python environment."
            ) from exc

        compiled_model = CompiledModel.from_file(
            str(model_path), hardware_accel=HardwareAccelerator.CPU
        )
        return cls(compiled_model=compiled_model, contract=_load_json(contract_path))

    def _create_named_input_buffers(self) -> dict[str, Any]:
        return {
            name: self._model.create_input_buffer_by_name(self.signature_key, name)
            for name in self.input_details
        }

    def _create_named_output_buffers(self) -> dict[str, Any]:
        return {
            name: self._model.create_output_buffer_by_name(self.signature_key, name)
            for name in self.output_names
        }

    def run_once(
        self,
        *,
        input_pos: np.ndarray,
        token_embeddings: np.ndarray,
        param_tensor: np.ndarray,
        kv_cache_k_13: np.ndarray,
        kv_cache_k_14: np.ndarray,
        kv_cache_v_13: np.ndarray,
        kv_cache_v_14: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> Gemma4LiteRTCompiledProbeResult:
        inputs = self._create_named_input_buffers()
        outputs = self._create_named_output_buffers()

        arrays: dict[str, np.ndarray] = {
            "input_pos": np.asarray(input_pos, dtype=np.int32),
            "activations": np.asarray(token_embeddings, dtype=np.float32),
            "param_tensor": np.asarray(param_tensor, dtype=np.int32),
            "kv_cache_k_13": np.asarray(kv_cache_k_13, dtype=np.int8),
            "kv_cache_k_14": np.asarray(kv_cache_k_14, dtype=np.int8),
            "kv_cache_v_13": np.asarray(kv_cache_v_13, dtype=np.int8),
            "kv_cache_v_14": np.asarray(kv_cache_v_14, dtype=np.int8),
        }

        bool_write_supported = True
        if mask is None:
            mask = np.zeros(tuple(self.input_details["mask"]["shape"]), dtype=np.bool_)
        try:
            inputs["mask"].write(np.asarray(mask, dtype=np.bool_))
        except ValueError:
            bool_write_supported = False

        for name, array in arrays.items():
            inputs[name].write(array)

        before_caches = {
            name: array.copy()
            for name, array in arrays.items()
            if name.startswith("kv_cache_")
        }

        self._model.run_by_name(self.signature_key, inputs, outputs)

        cache_input_mutation: dict[str, dict[str, int]] = {}
        for name, before in before_caches.items():
            after = (
                inputs[name].read(before.size, before.dtype.type).reshape(before.shape)
            )
            diff = np.abs(after.astype(np.int32) - before.astype(np.int32))
            cache_input_mutation[name] = {
                "max_abs": int(diff.max()),
                "sum_abs": int(diff.sum()),
            }

        output_specs = {
            "logits": ((1, 1, int(self.contract["logits_vocab_size"])), np.float32),
            "projected_activations": ((1, 1, self.projected_size), np.float32),
        }
        logits = (
            outputs["logits"]
            .read(int(np.prod(output_specs["logits"][0])), output_specs["logits"][1])
            .reshape(output_specs["logits"][0])
        )
        projected = (
            outputs["projected_activations"]
            .read(
                int(np.prod(output_specs["projected_activations"][0])),
                output_specs["projected_activations"][1],
            )
            .reshape(output_specs["projected_activations"][0])
        )

        return Gemma4LiteRTCompiledProbeResult(
            logits=logits,
            projected_activations=projected,
            cache_input_mutation=cache_input_mutation,
            bool_write_supported=bool_write_supported,
        )
