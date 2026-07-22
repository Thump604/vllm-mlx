# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vllm_mlx.cli import create_parser
from vllm_mlx.product_cli import _parse_overrides, product_command


class FakeClient:
    def __init__(self):
        self.calls = []

    def activate(self, profile, key, overrides=None):
        self.calls.append(("activate", profile, key, overrides))
        return {"operation_id": "op-1"}

    def status(self):
        self.calls.append(("status",))
        return {"state": "unloaded"}


def test_product_parser_captures_exact_activation_identity():
    args = create_parser().parse_args(
        [
            "product",
            "activate",
            "laguna-s-2.1",
            "--profile-revision",
            "3",
            "--subject-digest",
            "a" * 64,
            "--idempotency-key",
            "activate-laguna",
            "--override",
            "limits.serving_context=32768",
        ]
    )
    client = FakeClient()
    product_command(args, client=client)
    assert client.calls == [
        (
            "activate",
            {
                "profile_id": "laguna-s-2.1",
                "profile_revision": 3,
                "subject_digest": "a" * 64,
            },
            "activate-laguna",
            {"limits.serving_context": 32768},
        )
    ]


def test_product_status_prints_deterministic_json(capsys):
    args = create_parser().parse_args(["product", "status"])
    product_command(args, client=FakeClient())
    assert json.loads(capsys.readouterr().out) == {"state": "unloaded"}


def test_product_streaming_chat_flushes_each_chunk(monkeypatch):
    class StreamingClient:
        def chat(self, **kwargs):
            assert kwargs["stream"] is True
            return iter(["first", "second"])

    calls = []

    def record_print(*values, **kwargs):
        calls.append((values, kwargs))

    monkeypatch.setattr("builtins.print", record_print)
    args = create_parser().parse_args(
        [
            "product",
            "chat",
            "--model",
            "laguna",
            "--message",
            "review this",
            "--stream",
        ]
    )
    product_command(args, client=StreamingClient())

    assert calls == [
        (("first",), {"end": "", "flush": True}),
        (("second",), {"end": "", "flush": True}),
        ((), {"flush": True}),
    ]


def test_coding_setup_writes_configuration_without_client_mutation(tmp_path, capsys):
    output = tmp_path / "coding.json"
    args = create_parser().parse_args(
        [
            "product",
            "--api-key",
            "secret-not-persisted",
            "coding-setup",
            "--client",
            "openai",
            "--model",
            "laguna",
            "--output",
            str(output),
        ]
    )
    product_command(args, client=FakeClient())
    assert json.loads(output.read_text())["environment"] == {
        "OPENAI_BASE_URL": "http://127.0.0.1:8080/v1"
    }
    assert json.loads(output.read_text())["authentication"] == {
        "client_environment_variable": "OPENAI_API_KEY",
        "source_environment_variable": "VLLM_MLX_API_KEY",
        "required": True,
    }
    assert "secret-not-persisted" not in output.read_text()
    assert json.loads(capsys.readouterr().out)["written"] == str(output.resolve())


def test_activation_override_parser_is_strict():
    assert _parse_overrides(["features.mtp=true", "limits.serving_context=32768"]) == {
        "features.mtp": True,
        "limits.serving_context": 32768,
    }
    with pytest.raises(ValueError, match="duplicate"):
        _parse_overrides(["features.mtp=true", "features.mtp=false"])
    with pytest.raises(ValueError, match="unsupported"):
        _parse_overrides(["sampling.temperature=0.7"])
    with pytest.raises(ValueError, match="positive"):
        _parse_overrides(["limits.serving_context=0"])
