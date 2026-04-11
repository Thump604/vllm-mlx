# SPDX-License-Identifier: Apache-2.0
"""Probe whether LiteRT Python compiled-model buffers preserve KV mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllm_mlx.patches.gemma4_litert_compiled_runner import (
    Gemma4LiteRTCompiledRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input-pos", type=int, default=17)
    return parser.parse_args()


def _stats(array: np.ndarray) -> dict[str, float]:
    flat = array.reshape(-1)
    return {
        "sum": float(flat.sum()),
        "max": float(flat.max()),
        "min": float(flat.min()),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    runner = Gemma4LiteRTCompiledRunner.from_file(
        model_path=args.model,
        contract_path=args.contract_json,
    )
    result = runner.run_once(
        input_pos=np.array([args.input_pos], dtype=np.int32),
        token_embeddings=rng.standard_normal(
            (1, 1, runner.token_embedding_size), dtype=np.float32
        ),
        param_tensor=np.array(
            [[[[args.input_pos, args.input_pos + 1, args.input_pos + 1, 0, 0, 0, 0]]]],
            dtype=np.int32,
        ),
        kv_cache_k_13=rng.integers(-3, 4, size=(1, 1, 32003, 256), dtype=np.int8),
        kv_cache_k_14=rng.integers(-3, 4, size=(1, 1, 32003, 512), dtype=np.int8),
        kv_cache_v_13=rng.integers(-3, 4, size=(1, 1, 256, 32003), dtype=np.int8),
        kv_cache_v_14=rng.integers(-3, 4, size=(1, 1, 512, 32003), dtype=np.int8),
    )

    payload = {
        "model": str(args.model),
        "contract_json": str(args.contract_json),
        "seed": args.seed,
        "input_pos": args.input_pos,
        "bool_write_supported": result.bool_write_supported,
        "cache_input_mutation": result.cache_input_mutation,
        "cache_inputs_mutated": any(
            probe["sum_abs"] > 0 for probe in result.cache_input_mutation.values()
        ),
        "logits": _stats(result.logits),
        "projected_activations": _stats(result.projected_activations),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
