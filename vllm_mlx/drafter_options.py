# SPDX-License-Identifier: Apache-2.0
"""Shared CLI options for MLLM speculative drafters."""

from argparse import ArgumentParser

from .cli_arg_types import make_positive_int_arg_parser

MLLM_DRAFT_KINDS = ("dflash", "eagle3", "mtp")


def add_mllm_draft_arguments(
    parser: ArgumentParser, *, include_default: bool = False
) -> None:
    """Register the common MLLM drafter options on a serve parser."""
    parser.add_argument(
        "--mllm-draft-model",
        type=str,
        default=None,
        help="Path to an mlx-vlm MLLM draft/assistant model.",
    )
    parser.add_argument(
        "--mllm-draft-kind",
        type=str,
        default=None,
        choices=MLLM_DRAFT_KINDS,
        help="mlx-vlm draft kind for --mllm-draft-model.",
    )
    parser.add_argument(
        "--mllm-draft-block-size",
        type=make_positive_int_arg_parser("--mllm-draft-block-size"),
        default=None,
        help="Draft block size passed to mlx-vlm for --mllm-draft-model.",
    )
    if include_default:
        parser.add_argument(
            "--default-mllm-draft",
            action="store_true",
            help=(
                "Use the configured MLLM draft model when a request omits "
                "mllm_draft. Clients may still disable it with mllm_draft=false."
            ),
        )
