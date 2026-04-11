# SPDX-License-Identifier: Apache-2.0
"""Direct LiteRT runner wrapper for the Gemma 4 MTP drafter."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _to_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        array = np.asarray(value)
    if dtype is not None and array.dtype != dtype:
        return array.astype(dtype, copy=False)
    return array


@dataclass
class Gemma4LiteRTMTPState:
    projected_activations: np.ndarray
    kv_cache_k_13: np.ndarray
    kv_cache_k_14: np.ndarray
    kv_cache_v_13: np.ndarray
    kv_cache_v_14: np.ndarray

    def copy(self) -> "Gemma4LiteRTMTPState":
        return Gemma4LiteRTMTPState(
            projected_activations=self.projected_activations.copy(),
            kv_cache_k_13=self.kv_cache_k_13.copy(),
            kv_cache_k_14=self.kv_cache_k_14.copy(),
            kv_cache_v_13=self.kv_cache_v_13.copy(),
            kv_cache_v_14=self.kv_cache_v_14.copy(),
        )


class Gemma4LiteRTMTPRunner:
    """Thin wrapper around the LiteRT `mtp_drafter` signature.

    This is intentionally offline and adapter-facing. It does not patch the
    runtime scheduler yet; it only formalizes the direct execution seam now
    that the extracted drafter can be invoked locally.
    """

    def __init__(
        self,
        *,
        signature_runner: Any,
        contract: dict[str, Any],
        input_details: dict[str, dict[str, Any]],
    ):
        self._runner = signature_runner
        self.contract = contract
        self.input_details = input_details
        self.activations_formula = str(contract["activations_formula"])
        self.hidden_size = int(contract["model_hidden_size"])
        self.projected_size = int(contract["projected_activations_size"])
        self.param_shape = tuple(int(dim) for dim in contract["param_tensor_shape"])

    @classmethod
    def from_tflite(
        cls,
        *,
        model_path: str | Path,
        contract_path: str | Path,
    ) -> "Gemma4LiteRTMTPRunner":
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Gemma4LiteRTMTPRunner requires ai-edge-litert in the active Python environment."
            ) from exc

        interpreter = Interpreter(model_path=str(model_path))
        signature_runner = interpreter.get_signature_runner("mtp_drafter")
        return cls(
            signature_runner=signature_runner,
            contract=_load_json(contract_path),
            input_details=signature_runner.get_input_details(),
        )

    def make_cache(self) -> Gemma4LiteRTMTPState:
        projected = np.zeros((1, 1, self.projected_size), dtype=np.float32)
        return Gemma4LiteRTMTPState(
            projected_activations=projected,
            kv_cache_k_13=np.zeros(
                tuple(self.input_details["kv_cache_k_13"]["shape"]), dtype=np.int8
            ),
            kv_cache_k_14=np.zeros(
                tuple(self.input_details["kv_cache_k_14"]["shape"]), dtype=np.int8
            ),
            kv_cache_v_13=np.zeros(
                tuple(self.input_details["kv_cache_v_13"]["shape"]), dtype=np.int8
            ),
            kv_cache_v_14=np.zeros(
                tuple(self.input_details["kv_cache_v_14"]["shape"]), dtype=np.int8
            ),
        )

    def build_activations(
        self,
        hidden_states: np.ndarray,
        state: Gemma4LiteRTMTPState,
    ) -> np.ndarray:
        hidden = _to_numpy(hidden_states, dtype=np.float32)
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden.shape[-1]}"
            )
        if self.activations_formula == "concat(hidden_states, projected_activations)":
            return np.concatenate([hidden, state.projected_activations], axis=-1)
        if self.activations_formula == "hidden_states":
            return hidden
        raise ValueError(
            f"Unsupported activations formula {self.activations_formula!r}"
        )

    def run_step(
        self,
        hidden_states: np.ndarray,
        *,
        input_pos: int,
        state: Gemma4LiteRTMTPState,
        mask: np.ndarray | None = None,
        param_tensor: np.ndarray | None = None,
    ) -> dict[str, Any]:
        activations = self.build_activations(hidden_states, state)
        if mask is None:
            mask = np.zeros(tuple(self.input_details["mask"]["shape"]), dtype=np.bool_)
        if param_tensor is None:
            param_tensor = np.zeros(self.param_shape, dtype=np.int32)

        start = time.perf_counter()
        outputs = self._runner(
            activations=activations,
            input_pos=np.array([input_pos], dtype=np.int32),
            kv_cache_k_13=state.kv_cache_k_13,
            kv_cache_k_14=state.kv_cache_k_14,
            kv_cache_v_13=state.kv_cache_v_13,
            kv_cache_v_14=state.kv_cache_v_14,
            mask=_to_numpy(mask, dtype=np.bool_),
            param_tensor=_to_numpy(param_tensor, dtype=np.int32),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        state.projected_activations = _to_numpy(
            outputs["projected_activations"], dtype=np.float32
        )
        return {
            "logits": _to_numpy(outputs["logits"], dtype=np.float32),
            "projected_activations": state.projected_activations,
            "elapsed_ms": elapsed_ms,
        }
