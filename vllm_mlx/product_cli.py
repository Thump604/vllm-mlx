# SPDX-License-Identifier: Apache-2.0
"""Product-oriented command shell over the control and inference APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, cast

from .control_client import ControlClient


def add_product_parser(subparsers: argparse._SubParsersAction) -> None:
    product = subparsers.add_parser("product", help="Use the managed model product")
    product.add_argument(
        "--control-url", default="http://127.0.0.1:8080", help="Runtime base URL"
    )
    product.add_argument("--api-key", default=None, help="Runtime API key")
    commands = product.add_subparsers(dest="product_command", required=True)

    catalog = commands.add_parser("catalog", help="List curated model profiles")
    catalog.add_argument("profile_id", nargs="?", help="Show one complete profile")

    for name in ("install", "activate"):
        command = commands.add_parser(name, help=f"{name.title()} an exact profile")
        _add_profile_reference_arguments(command)
        command.add_argument("--idempotency-key", required=True)
        if name == "activate":
            command.add_argument("--override", action="append", default=[])

    stop = commands.add_parser("stop", help="Stop the active model")
    stop.add_argument("--idempotency-key", required=True)

    commands.add_parser("status", help="Show managed runtime status")
    commands.add_parser("diagnostics", help="Show sanitized runtime diagnostics")

    operation = commands.add_parser("operation", help="Inspect or cancel an operation")
    operation.add_argument("operation_id")
    operation.add_argument("--cancel", action="store_true")
    operation.add_argument("--idempotency-key")

    chat = commands.add_parser("chat", help="Send a chat request without activation")
    chat.add_argument("--model", required=True)
    chat.add_argument("--message", required=True)
    chat.add_argument("--stream", action="store_true")
    chat.add_argument("--max-tokens", type=int)

    coding = commands.add_parser("coding-setup", help="Emit coding client settings")
    coding.add_argument("--client", choices=["openai", "anthropic"], required=True)
    coding.add_argument("--model", required=True)
    coding.add_argument("--endpoint", default="http://127.0.0.1:8080")
    coding.add_argument("--output")


def product_command(
    args: argparse.Namespace, *, client: ControlClient | None = None
) -> None:
    client = client or ControlClient(args.control_url, api_key=args.api_key)
    command = args.product_command
    payload: Any
    if command == "catalog":
        payload = (
            client.profile(args.profile_id) if args.profile_id else client.catalog()
        )
    elif command == "install":
        payload = client.install(_profile_reference(args), args.idempotency_key)
    elif command == "activate":
        payload = client.activate(
            _profile_reference(args),
            args.idempotency_key,
            overrides=_parse_overrides(args.override),
        )
    elif command == "stop":
        payload = client.stop(args.idempotency_key)
    elif command == "status":
        payload = client.status()
    elif command == "diagnostics":
        payload = client.diagnostics()
    elif command == "operation":
        if args.cancel:
            if not args.idempotency_key:
                raise ValueError("--idempotency-key is required with --cancel")
            payload = client.cancel_operation(args.operation_id, args.idempotency_key)
        else:
            payload = client.operation(args.operation_id)
    elif command == "chat":
        payload = client.chat(
            model=args.model,
            message=args.message,
            stream=args.stream,
            max_tokens=args.max_tokens,
        )
        if args.stream:
            for chunk in cast(Iterator[str], payload):
                print(chunk, end="", flush=True)
            print(flush=True)
            return
    elif command == "coding-setup":
        payload = build_coding_setup(
            args.client,
            args.model,
            args.endpoint,
            runtime_api_key_configured=bool(args.api_key),
        )
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            payload = {"written": str(output.resolve()), "configuration": payload}
    else:
        raise ValueError(f"unsupported product command: {command}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _add_profile_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("profile_id")
    parser.add_argument("--profile-revision", type=int, required=True)
    parser.add_argument("--subject-digest", required=True)


def _profile_reference(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile_id": args.profile_id,
        "profile_revision": args.profile_revision,
        "subject_digest": args.subject_digest,
    }


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"activation override must use NAME=VALUE: {value}")
        name, raw = value.split("=", 1)
        if name in overrides:
            raise ValueError(f"duplicate activation override: {name}")
        if name.startswith("features.") and raw in {"true", "false"}:
            overrides[name] = raw == "true"
        elif name.startswith("limits."):
            try:
                parsed = int(raw)
            except ValueError as exc:
                raise ValueError(f"limit override must be an integer: {name}") from exc
            if parsed < 1:
                raise ValueError(f"limit override must be positive: {name}")
            overrides[name] = parsed
        else:
            raise ValueError(f"unsupported activation override: {name}")
    return overrides


def build_coding_setup(
    client: str,
    model: str,
    endpoint: str,
    *,
    runtime_api_key_configured: bool,
) -> dict[str, Any]:
    endpoint = endpoint.rstrip("/")
    if client == "openai":
        return {
            "client": client,
            "model": model,
            "environment": {"OPENAI_BASE_URL": f"{endpoint}/v1"},
            "authentication": {
                "client_environment_variable": "OPENAI_API_KEY",
                "source_environment_variable": "VLLM_MLX_API_KEY",
                "required": runtime_api_key_configured,
            },
        }
    return {
        "client": client,
        "model": model,
        "environment": {"ANTHROPIC_BASE_URL": endpoint},
        "authentication": {
            "client_environment_variable": "ANTHROPIC_API_KEY",
            "source_environment_variable": "VLLM_MLX_API_KEY",
            "required": runtime_api_key_configured,
        },
    }
