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


def _resolved_thump_lib(args: argparse.Namespace) -> str | None:
    if args.thump_lib:
        return args.thump_lib
    if args.thump_prefix:
        return str(Path(args.thump_prefix) / "lib" / "libthump_runtime.dylib")
    return None


def _run_subprocess(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _with_thump_runtime_env(
    env: dict[str, str] | None,
    *,
    thump_prefix: str | None = None,
) -> dict[str, str]:
    merged = dict(os.environ if env is None else env)
    if thump_prefix:
        lib_dir = str(Path(thump_prefix) / "lib")
        existing = merged.get("DYLD_LIBRARY_PATH")
        merged["DYLD_LIBRARY_PATH"] = (
            f"{lib_dir}:{existing}" if existing else lib_dir
        )
        merged.setdefault("VLLM_MLX_THUMP_PREFIX", thump_prefix)
    return merged


def _run_thumpctl_json(
    *,
    thumpctl_path: str,
    command: str,
    manifest_path: str,
    thump_prefix: str | None,
) -> dict:
    proc = subprocess.run(
        [thumpctl_path, command, manifest_path, "--json"],
        check=False,
        text=True,
        capture_output=True,
        env=_with_thump_runtime_env(None, thump_prefix=thump_prefix),
    )
    payload = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    try:
        payload["json"] = json.loads(proc.stdout) if proc.stdout else None
    except json.JSONDecodeError:
        payload["json"] = None
    return payload


def _checkpoint_mode(args: argparse.Namespace) -> None:
    runner = SessionRecoveryRunner(
        args.model_path,
        thump_lib_path=_resolved_thump_lib(args),
        block_size_tokens=args.block_size_tokens,
    )
    trace = SessionRecoveryTrace.from_path(args.trace)
    artifact = runner.create_checkpoint(trace, bundle_dir=args.bundle_dir)
    _write_json(Path(args.output), artifact.__dict__)


def _restore_mode(args: argparse.Namespace) -> None:
    runner = SessionRecoveryRunner(
        args.model_path,
        thump_lib_path=_resolved_thump_lib(args),
        block_size_tokens=args.block_size_tokens,
    )
    trace = SessionRecoveryTrace.from_path(args.trace)
    artifact = CheckpointArtifact.from_path(args.checkpoint_json)
    result = runner.restore_and_continue(trace, artifact)
    _write_json(Path(args.output), result.__dict__)


def _cold_mode(args: argparse.Namespace) -> None:
    runner = SessionRecoveryRunner(
        args.model_path,
        thump_lib_path=_resolved_thump_lib(args),
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
    if args.thump_prefix:
        base_cmd.extend(["--thump-prefix", args.thump_prefix])
    elif args.thump_lib:
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

    operator_validation = None
    if args.thumpctl:
        manifest_path = json.loads(checkpoint_json.read_text())["manifest_path"]
        operator_validation = {
            "thump_prefix": args.thump_prefix,
            "thumpctl_path": args.thumpctl,
            "inspect": _run_thumpctl_json(
                thumpctl_path=args.thumpctl,
                command="inspect",
                manifest_path=manifest_path,
                thump_prefix=args.thump_prefix,
            ),
            "validate_session": _run_thumpctl_json(
                thumpctl_path=args.thumpctl,
                command="validate-session",
                manifest_path=manifest_path,
                thump_prefix=args.thump_prefix,
            ),
            "scale": _run_thumpctl_json(
                thumpctl_path=args.thumpctl,
                command="scale",
                manifest_path=manifest_path,
                thump_prefix=args.thump_prefix,
            ),
        }

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
    payload = comparison.to_dict()
    telemetry = payload.setdefault("telemetry", {})
    telemetry["thump_package_prefix"] = args.thump_prefix
    telemetry["thumpctl_path"] = args.thumpctl
    if operator_validation is not None:
        inspect_json = operator_validation.get("inspect", {}).get("json") or {}
        if inspect_json:
            artifact_metrics = {
                key: inspect_json.get(key)
                for key in (
                    "artifact_kind",
                    "bank_count",
                    "changed_bank_count",
                    "manifest_bytes",
                    "data_bytes",
                    "total_bytes",
                    "layout_kind",
                    "live_block_count",
                    "live_block_capacity",
                    "estimated_full_snapshot_bytes",
                    "estimated_compact_snapshot_bytes",
                )
            }
            payload["checkpoint"]["artifact_metrics"] = artifact_metrics
            telemetry["artifact_metrics"] = artifact_metrics
        validate_json = operator_validation.get("validate_session", {}).get("json") or {}
        telemetry["operator_validation"] = {
            "inspect_ok": operator_validation.get("inspect", {}).get("returncode") == 0,
            "validate_session_ok": validate_json.get("ok"),
            "validate_session_reason": (validate_json.get("diagnostic") or {}).get(
                "reason"
            ),
            "scale_ok": operator_validation.get("scale", {}).get("returncode") == 0,
        }
        payload["operator_validation"] = operator_validation
    _write_json(output_path, payload)


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
    parser.add_argument("--thump-prefix")
    parser.add_argument("--thumpctl")
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
