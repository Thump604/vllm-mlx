#!/usr/bin/env python3
"""Build the Gemma 4 LiteRT MTP adapter contract from extracted artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_mlx.patches.gemma4_litert_mtp import build_contract_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interface-json",
        required=True,
        type=Path,
        help="Path to the extracted drafter interface JSON",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        type=Path,
        help="Path to Gemma config.json (or top-level model config)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional output path for the derived contract JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = build_contract_from_paths(args.interface_json, args.model_config)
    payload = contract.to_dict()
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
