from vllm_mlx.cli import create_parser
from vllm_mlx.server import create_parser as create_server_parser


def test_cli_and_standalone_server_share_drafter_kinds():
    cli_args = create_parser().parse_args(
        [
            "serve",
            "model",
            "--mllm-draft-model",
            "draft",
            "--mllm-draft-kind",
            "dflash",
            "--mllm-draft-block-size",
            "8",
            "--default-mllm-draft",
        ]
    )
    server_args = create_server_parser().parse_args(
        [
            "--model",
            "model",
            "--mllm-draft-model",
            "draft",
            "--mllm-draft-kind",
            "eagle3",
            "--mllm-draft-block-size",
            "4",
        ]
    )

    assert (cli_args.mllm_draft_model, cli_args.mllm_draft_kind) == (
        "draft",
        "dflash",
    )
    assert cli_args.default_mllm_draft is True
    assert (server_args.mllm_draft_model, server_args.mllm_draft_kind) == (
        "draft",
        "eagle3",
    )
