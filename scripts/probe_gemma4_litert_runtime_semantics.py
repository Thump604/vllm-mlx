# SPDX-License-Identifier: Apache-2.0
"""Probe semantic liveness of explicit Gemma 4 LiteRT drafter inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllm_mlx.patches.gemma4_litert_runner import (
    Gemma4LiteRTMTPRunner,
    Gemma4LiteRTMTPState,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--param-value", type=int, default=128)
    parser.add_argument("--mask-tokens", type=int, default=128)
    parser.add_argument("--mask-tokens-alt", type=int, default=256)
    parser.add_argument("--input-pos", type=int, default=17)
    return parser.parse_args()


def _stats(array: np.ndarray) -> dict[str, float]:
    flat = array.reshape(-1)
    return {
        "sum": float(flat.sum()),
        "max": float(flat.max()),
        "min": float(flat.min()),
    }


def _diff_stats(base: np.ndarray, other: np.ndarray) -> dict[str, float]:
    diff = np.abs(base - other)
    return {
        "max_abs": float(diff.max()),
        "sum_abs": float(diff.sum()),
    }


def _make_dense_state(
    runner: Gemma4LiteRTMTPRunner, rng: np.random.Generator
) -> Gemma4LiteRTMTPState:
    state = runner.make_cache()
    state.projected_activations[...] = rng.standard_normal(
        state.projected_activations.shape, dtype=np.float32
    )
    for name in ("kv_cache_k_13", "kv_cache_k_14", "kv_cache_v_13", "kv_cache_v_14"):
        array = getattr(state, name)
        array[...] = rng.integers(-3, 4, size=array.shape, dtype=np.int8)
    return state


def _make_mask(shape: tuple[int, ...], tokens: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.bool_)
    mask[..., :tokens] = True
    return mask


def _run(
    runner: Gemma4LiteRTMTPRunner,
    hidden_states: np.ndarray,
    state: Gemma4LiteRTMTPState,
    *,
    input_pos: int,
    mask: np.ndarray,
    param_tensor: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    return runner.run_step(
        hidden_states,
        input_pos=input_pos,
        state=state.copy(),
        mask=mask,
        param_tensor=param_tensor,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    runner = Gemma4LiteRTMTPRunner.from_tflite(
        model_path=args.model,
        contract_path=args.contract_json,
    )
    base_state = _make_dense_state(runner, rng)
    hidden_states = rng.standard_normal((1, 1, runner.hidden_size), dtype=np.float32)
    base_mask = _make_mask(
        tuple(runner.input_details["mask"]["shape"]), args.mask_tokens
    )
    alt_mask = _make_mask(
        tuple(runner.input_details["mask"]["shape"]), args.mask_tokens_alt
    )

    baseline = _run(
        runner,
        hidden_states,
        base_state,
        input_pos=args.input_pos,
        mask=base_mask,
    )

    payload: dict[str, object] = {
        "model": str(args.model),
        "contract_json": str(args.contract_json),
        "seed": args.seed,
        "baseline": {
            "input_pos": args.input_pos,
            "mask_tokens": args.mask_tokens,
            "logits": _stats(baseline["logits"]),
            "projected_activations": _stats(baseline["projected_activations"]),
            "elapsed_ms": round(float(baseline["elapsed_ms"]), 3),
        },
        "input_pos_probe": {},
        "mask_probe": {},
        "param_slot_probes": [],
        "param_slot_capacity_probes": [],
        "slot0_threshold_probe": [],
        "mask_probe_with_slot0_capacity": {},
        "cache_zero_probes": [],
        "cache_zero_probes_by_slot0": [],
    }

    input_pos_alt = _run(
        runner,
        hidden_states,
        base_state,
        input_pos=args.input_pos + 1,
        mask=base_mask,
    )
    payload["input_pos_probe"] = {
        "alt_input_pos": args.input_pos + 1,
        "logits_diff": _diff_stats(baseline["logits"], input_pos_alt["logits"]),
        "projected_diff": _diff_stats(
            baseline["projected_activations"],
            input_pos_alt["projected_activations"],
        ),
    }

    mask_alt = _run(
        runner,
        hidden_states,
        base_state,
        input_pos=args.input_pos,
        mask=alt_mask,
    )
    payload["mask_probe"] = {
        "alt_mask_tokens": args.mask_tokens_alt,
        "logits_diff": _diff_stats(baseline["logits"], mask_alt["logits"]),
        "projected_diff": _diff_stats(
            baseline["projected_activations"],
            mask_alt["projected_activations"],
        ),
    }

    for slot in range(int(np.prod(runner.param_shape))):
        param_tensor = np.zeros(runner.param_shape, dtype=np.int32)
        param_tensor.reshape(-1)[slot] = args.param_value
        probe = _run(
            runner,
            hidden_states,
            base_state,
            input_pos=args.input_pos,
            mask=base_mask,
            param_tensor=param_tensor,
        )
        payload["param_slot_probes"].append(
            {
                "slot": slot,
                "value": args.param_value,
                "logits_diff": _diff_stats(baseline["logits"], probe["logits"]),
                "projected_diff": _diff_stats(
                    baseline["projected_activations"],
                    probe["projected_activations"],
                ),
            }
        )

    time_capacity = max(
        int(runner.input_details["kv_cache_k_13"]["shape"][2]),
        int(runner.input_details["kv_cache_k_14"]["shape"][2]),
    )
    for slot in range(int(np.prod(runner.param_shape))):
        param_tensor = np.zeros(runner.param_shape, dtype=np.int32)
        param_tensor.reshape(-1)[slot] = time_capacity
        probe = _run(
            runner,
            hidden_states,
            base_state,
            input_pos=args.input_pos,
            mask=base_mask,
            param_tensor=param_tensor,
        )
        payload["param_slot_capacity_probes"].append(
            {
                "slot": slot,
                "value": time_capacity,
                "logits_diff": _diff_stats(baseline["logits"], probe["logits"]),
                "projected_diff": _diff_stats(
                    baseline["projected_activations"],
                    probe["projected_activations"],
                ),
            }
        )

    slot0_thresholds = (
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        time_capacity - 4,
        time_capacity - 3,
        time_capacity - 2,
        time_capacity - 1,
        time_capacity,
    )
    for value in slot0_thresholds:
        param_tensor = np.zeros(runner.param_shape, dtype=np.int32)
        param_tensor.reshape(-1)[0] = value
        probe = _run(
            runner,
            hidden_states,
            base_state,
            input_pos=args.input_pos,
            mask=base_mask,
            param_tensor=param_tensor,
        )
        payload["slot0_threshold_probe"].append(
            {
                "value": value,
                "logits_diff": _diff_stats(baseline["logits"], probe["logits"]),
                "projected_diff": _diff_stats(
                    baseline["projected_activations"],
                    probe["projected_activations"],
                ),
            }
        )

    slot0_capacity = np.zeros(runner.param_shape, dtype=np.int32)
    slot0_capacity.reshape(-1)[0] = time_capacity
    mask_with_slot0_capacity = _run(
        runner,
        hidden_states,
        base_state,
        input_pos=args.input_pos,
        mask=alt_mask,
        param_tensor=slot0_capacity,
    )
    slot0_capacity_base = _run(
        runner,
        hidden_states,
        base_state,
        input_pos=args.input_pos,
        mask=base_mask,
        param_tensor=slot0_capacity,
    )
    payload["mask_probe_with_slot0_capacity"] = {
        "slot0_value": time_capacity,
        "alt_mask_tokens": args.mask_tokens_alt,
        "logits_diff": _diff_stats(
            slot0_capacity_base["logits"], mask_with_slot0_capacity["logits"]
        ),
        "projected_diff": _diff_stats(
            slot0_capacity_base["projected_activations"],
            mask_with_slot0_capacity["projected_activations"],
        ),
    }

    for name in ("kv_cache_k_13", "kv_cache_k_14", "kv_cache_v_13", "kv_cache_v_14"):
        cache_state = base_state.copy()
        getattr(cache_state, name)[...] = 0
        probe = _run(
            runner,
            hidden_states,
            cache_state,
            input_pos=args.input_pos,
            mask=base_mask,
        )
        payload["cache_zero_probes"].append(
            {
                "cache_name": name,
                "logits_diff": _diff_stats(baseline["logits"], probe["logits"]),
                "projected_diff": _diff_stats(
                    baseline["projected_activations"],
                    probe["projected_activations"],
                ),
            }
        )

    for slot0_value in (512, 1024, time_capacity - 1, time_capacity):
        slot0_param = np.zeros(runner.param_shape, dtype=np.int32)
        slot0_param.reshape(-1)[0] = slot0_value
        slot0_base = _run(
            runner,
            hidden_states,
            base_state,
            input_pos=args.input_pos,
            mask=base_mask,
            param_tensor=slot0_param,
        )
        slot_payload = {
            "slot0_value": slot0_value,
            "cache_probes": [],
        }
        for name in (
            "kv_cache_k_13",
            "kv_cache_k_14",
            "kv_cache_v_13",
            "kv_cache_v_14",
        ):
            cache_state = base_state.copy()
            getattr(cache_state, name)[...] = 0
            probe = _run(
                runner,
                hidden_states,
                cache_state,
                input_pos=args.input_pos,
                mask=base_mask,
                param_tensor=slot0_param,
            )
            slot_payload["cache_probes"].append(
                {
                    "cache_name": name,
                    "logits_diff": _diff_stats(slot0_base["logits"], probe["logits"]),
                    "projected_diff": _diff_stats(
                        slot0_base["projected_activations"],
                        probe["projected_activations"],
                    ),
                }
            )
        payload["cache_zero_probes_by_slot0"].append(slot_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
