#!/usr/bin/env python3
"""Run the offline Gemma 4 Thump replay benchmark and write a JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_mlx.thump.replay import ReplayRunner, ReplayTrace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace", required=True, help="Path to replay trace JSON")
    parser.add_argument(
        "--output", required=True, help="Where to write the JSON result"
    )
    parser.add_argument("--draft-model-path")
    parser.add_argument("--thump-lib")
    args = parser.parse_args()

    trace = ReplayTrace.from_path(args.trace)
    runner = ReplayRunner(
        args.model_path,
        thump_lib_path=args.thump_lib,
        draft_model_path=args.draft_model_path,
    )
    comparison = runner.run_comparison(trace)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison.to_dict(), indent=2))


if __name__ == "__main__":
    main()
