#!/usr/bin/env python3
"""Run the offline Gemma 4 Thump session restart-recovery benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from vllm_mlx.thump.recovery import (
    FEATURE_FLAG_ENV,
    CheckpointArtifact,
    RecoveryRunResult,
    SessionRecoveryRunner,
    SessionRecoveryTrace,
    build_recovery_comparison,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _run_subprocess(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _checkpoint_mode(args: argparse.Namespace) -> None:
    runner = SessionRecoveryRunner(
        args.model_path,
        thump_lib_path=args.thump_lib or None,
        block_size_tokens=args.block_size_tokens,
    )
    trace = SessionRecoveryTrace.from_path(args.trace)
    artifact = runner.create_checkpoint(trace, bundle_dir=args.bundle_dir)
    _write_json(Path(args.output), artifact.__dict__)


def _restore_mode(args: argparse.Namespace) -> None:
    runner = SessionRecoveryRunner(
        args.model_path,
        thump_lib_path=args.thump_lib or None,
        block_size_tokens=args.block_size_tokens,
    )
    trace = SessionRecoveryTrace.from_path(args.trace)
    artifact = CheckpointArtifact.from_path(args.checkpoint_json)
    result = runner.restore_and_continue(trace, artifact)
    _write_json(Path(args.output), result.__dict__)


def _cold_mode(args: argparse.Namespace) -> None:
    runner = SessionRecoveryRunner(
        args.model_path,
        thump_lib_path=args.thump_lib or None,
        block_size_tokens=args.block_size_tokens,
    )
    trace = SessionRecoveryTrace.from_path(args.trace)
    artifact = CheckpointArtifact.from_path(args.checkpoint_json)
    result = runner.cold_rebuild_and_continue(trace, artifact)
    _write_json(Path(args.output), result.__dict__)


def _benchmark_mode(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve()
    trace_path = Path(args.trace).resolve()
    output_path = Path(args.output).resolve()
    work_dir = output_path.parent / f"{output_path.stem}-work"
    bundle_dir = work_dir / "bundle"
    checkpoint_json = work_dir / "checkpoint.json"
    restore_json = work_dir / "restore.json"
    cold_json = work_dir / "cold.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        str(script_path),
        "--model-path",
        args.model_path,
        "--trace",
        str(trace_path),
        "--block-size-tokens",
        str(args.block_size_tokens),
    ]
    if args.thump_lib:
        base_cmd.extend(["--thump-lib", args.thump_lib])

    _run_subprocess(
        base_cmd
        + [
            "--mode",
            "checkpoint",
            "--bundle-dir",
            str(bundle_dir),
            "--output",
            str(checkpoint_json),
        ]
    )

    restore_env = os.environ.copy()
    restore_env[FEATURE_FLAG_ENV] = "1"
    _run_subprocess(
        base_cmd
        + [
            "--mode",
            "restore",
            "--checkpoint-json",
            str(checkpoint_json),
            "--output",
            str(restore_json),
        ],
        env=restore_env,
    )
    _run_subprocess(
        base_cmd
        + [
            "--mode",
            "cold",
            "--checkpoint-json",
            str(checkpoint_json),
            "--output",
            str(cold_json),
        ]
    )

    trace = SessionRecoveryTrace.from_path(trace_path)
    artifact = CheckpointArtifact.from_path(checkpoint_json)
    restore = RecoveryRunResult(**json.loads(restore_json.read_text()))
    cold = RecoveryRunResult(**json.loads(cold_json.read_text()))
    comparison = build_recovery_comparison(
        trace,
        artifact,
        restore=restore,
        cold_rebuild=cold,
    )
    _write_json(output_path, comparison.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("benchmark", "checkpoint", "restore", "cold"),
        default="benchmark",
    )
    parser.add_argument("--bundle-dir")
    parser.add_argument("--checkpoint-json")
    parser.add_argument("--thump-lib")
    parser.add_argument("--block-size-tokens", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "checkpoint":
        if not args.bundle_dir:
            raise SystemExit("--bundle-dir is required for checkpoint mode")
        _checkpoint_mode(args)
        return
    if args.mode == "restore":
        if not args.checkpoint_json:
            raise SystemExit("--checkpoint-json is required for restore mode")
        _restore_mode(args)
        return
    if args.mode == "cold":
        if not args.checkpoint_json:
            raise SystemExit("--checkpoint-json is required for cold mode")
        _cold_mode(args)
        return
    _benchmark_mode(args)


if __name__ == "__main__":
    main()
