# SPDX-License-Identifier: Apache-2.0
"""Convert extracted Gemma 4 LiteRT MTP tensors into an MLX adapter payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_mlx.patches.gemma4_litert_mtp_convert import convert_extracted_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-summary", required=True, type=Path)
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-bits", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = convert_extracted_tensors(
        args.extract_summary,
        contract_path=args.contract_json,
        output_dir=args.output_dir,
        target_bits=args.target_bits,
        group_size=args.group_size,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
