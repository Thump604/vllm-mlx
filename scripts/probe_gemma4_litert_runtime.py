# SPDX-License-Identifier: Apache-2.0
"""Probe direct LiteRT execution for the extracted Gemma 4 MTP drafter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllm_mlx.patches.gemma4_litert_runner import Gemma4LiteRTMTPRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
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

    runner = Gemma4LiteRTMTPRunner.from_tflite(
        model_path=args.model,
        contract_path=args.contract_json,
    )

    zero_state = runner.make_cache()
    zero_token_embeddings = np.zeros(
        (1, 1, runner.token_embedding_size), dtype=np.float32
    )
    zero_result = runner.run_step(zero_token_embeddings, input_pos=0, state=zero_state)

    random_state = runner.make_cache()
    random_token_embeddings = rng.standard_normal(
        (1, 1, runner.token_embedding_size), dtype=np.float32
    )
    random_result = runner.run_step(
        random_token_embeddings, input_pos=17, state=random_state
    )

    payload = {
        "model": str(args.model),
        "contract_json": str(args.contract_json),
        "input_shapes": {
            name: np.asarray(detail["shape"]).tolist()
            for name, detail in runner.input_details.items()
        },
        "zero_step": {
            "elapsed_ms": round(zero_result["elapsed_ms"], 3),
            "logits": _stats(zero_result["logits"]),
            "projected_activations": _stats(zero_result["projected_activations"]),
        },
        "random_step": {
            "elapsed_ms": round(random_result["elapsed_ms"], 3),
            "logits": _stats(random_result["logits"]),
            "projected_activations": _stats(random_result["projected_activations"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
