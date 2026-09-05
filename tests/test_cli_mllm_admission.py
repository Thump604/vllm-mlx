import pytest
from types import SimpleNamespace


@pytest.mark.parametrize(
    ("option", "attribute"),
    [
        ("--mllm-max-inflight-requests", "mllm_max_inflight_requests"),
        ("--mllm-max-inflight-prompt-tokens", "mllm_max_inflight_prompt_tokens"),
    ],
)
def test_serve_parser_accepts_positive_admission_limits(option, attribute):
    from vllm_mlx.cli import create_parser

    args = create_parser().parse_args(["serve", "model", option, "7"])
    assert getattr(args, attribute) == 7


def test_serve_parser_defaults_admission_limits_to_none():
    from vllm_mlx.cli import create_parser

    args = create_parser().parse_args(["serve", "model"])
    assert args.mllm_max_inflight_requests is None
    assert args.mllm_max_inflight_prompt_tokens is None


@pytest.mark.parametrize(
    "option", ["--mllm-max-inflight-requests", "--mllm-max-inflight-prompt-tokens"]
)
@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_serve_parser_rejects_invalid_admission_limits(option, value, capsys):
    from vllm_mlx.cli import create_parser

    with pytest.raises(SystemExit) as exc:
        create_parser().parse_args(["serve", "model", option, value])
    assert exc.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize(
    "option", ["mllm_max_inflight_requests", "mllm_max_inflight_prompt_tokens"]
)
def test_admission_limit_validation_precedes_server_import(monkeypatch, option):
    from vllm_mlx import cli

    imported = []
    real_import = __import__

    def recording_import(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", recording_import)
    args = SimpleNamespace(mllm=False, continuous_batching=False, **{option: 1})
    with pytest.raises(SystemExit):
        cli.serve_command(args)
    assert "uvicorn" not in imported
    assert "vllm_mlx.server" not in imported


def test_serve_command_wires_admission_limits(monkeypatch):
    from vllm_mlx import cli, server
    from vllm_mlx.utils import download

    loaded = {}
    monkeypatch.setattr(
        download, "ensure_model_downloaded", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        server, "load_model", lambda *args, **kwargs: loaded.update(kwargs)
    )
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    args = cli.create_parser().parse_args(
        [
            "serve",
            "model",
            "--mllm",
            "--continuous-batching",
            "--mllm-max-inflight-requests",
            "3",
            "--mllm-max-inflight-prompt-tokens",
            "4096",
        ]
    )
    cli.serve_command(args)
    config = loaded["scheduler_config"]
    assert config.mllm_max_inflight_requests == 3
    assert config.mllm_max_inflight_prompt_tokens == 4096
