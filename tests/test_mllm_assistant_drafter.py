# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM assistant-drafter speculative wiring."""

import sys
from types import SimpleNamespace

import pytest


def test_mllm_loads_dflash_through_drafter_loader(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    draft = SimpleNamespace(config=SimpleNamespace(model_type="laguna"))
    captured = {}

    def load_drafter(path, kind=None):
        captured["load"] = (path, kind)
        return draft, "dflash"

    def validate(target, loaded_draft, kind):
        captured["validate"] = (target, loaded_draft, kind)

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.speculative.drafters",
        SimpleNamespace(
            load_drafter=load_drafter,
            validate_drafter_compatibility=validate,
        ),
    )

    target = object()
    model = MLXMultimodalLM(
        "target", draft_model="draft", draft_kind="dflash", draft_block_size=16
    )
    model.model = target

    assert model._load_draft_model() is draft
    assert captured["load"] == ("draft", "dflash")
    assert captured["validate"] == (target, draft, "dflash")
    assert model.draft_kind == "dflash"


def test_mllm_load_forwards_trust_remote_code(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}

    def load(path, **kwargs):
        captured["load"] = (path, kwargs)
        return SimpleNamespace(config=SimpleNamespace()), object()

    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(load=load))
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.utils",
        SimpleNamespace(load_config=lambda path: {"model_type": "laguna"}),
    )

    model = MLXMultimodalLM("local-laguna", trust_remote_code=True)
    model.load()

    assert captured["load"] == (
        "local-laguna",
        {"trust_remote_code": True},
    )


def test_mllm_uses_resolved_drafter_kind(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.speculative.drafters",
        SimpleNamespace(
            load_drafter=lambda *args, **kwargs: (object(), "dflash"),
            validate_drafter_compatibility=lambda *args: None,
        ),
    )

    model = MLXMultimodalLM("target", draft_model="draft", draft_kind=None)
    model.model = object()
    model._load_draft_model()

    assert model.draft_kind == "dflash"


def test_server_cli_accepts_dflash_drafter_kind():
    from vllm_mlx import server

    args = server.create_parser().parse_args(
        [
            "--model",
            "target",
            "--mllm-draft-model",
            "draft",
            "--mllm-draft-kind",
            "dflash",
        ]
    )

    assert args.mllm_draft_kind == "dflash"


def test_mllm_chat_forwards_configured_assistant_drafter(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}
    draft_model = SimpleNamespace(accept_lens=[99])

    def fake_generate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["accept_lens_at_call"] = list(draft_model.accept_lens)
        draft_model.accept_lens = [1, 2]
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=2)

    fake_cache = SimpleNamespace(make_prompt_cache=lambda *args, **kwargs: ["cache"])
    fake_models = SimpleNamespace(cache=fake_cache)
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(generate=fake_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        SimpleNamespace(get_chat_template=lambda *args, **kwargs: "rendered prompt"),
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", fake_models)

    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    processor = SimpleNamespace(tokenizer=tokenizer)
    target = SimpleNamespace(language_model=object())

    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._loaded = True
    model.model = target
    model.processor = processor
    model.config = {}
    model._draft_model = draft_model
    model._cache_manager = None

    output = model.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        mllm_draft=True,
    )

    assert output.text == "ok"
    assert output.mtp_drafts == 6
    assert output.mtp_accepted == 3
    assert captured["accept_lens_at_call"] == []
    assert captured["kwargs"]["draft_model"] is draft_model
    assert captured["kwargs"]["draft_kind"] == "mtp"
    assert captured["kwargs"]["draft_block_size"] == 4


def test_mllm_draft_metrics_use_recorded_draft_counts():
    from vllm_mlx.models.mllm import MLXMultimodalLM

    draft_model = SimpleNamespace(
        accept_lens=[1, 0],
        _vllm_mlx_draft_counts=[2, 1],
        config=SimpleNamespace(block_size=4),
    )
    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._draft_model = draft_model

    assert model._draft_metrics_since(0) == {
        "mtp_drafts": 3,
        "mtp_accepted": 1,
    }


def test_mllm_chat_uses_configured_drafter_over_call_kwargs(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}
    configured_draft = SimpleNamespace(accept_lens=[], _vllm_mlx_draft_counts=[])
    caller_draft = SimpleNamespace()

    def fake_generate(*args, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=1)

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(generate=fake_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        SimpleNamespace(get_chat_template=lambda *args, **kwargs: "rendered prompt"),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models",
        SimpleNamespace(cache=SimpleNamespace(make_prompt_cache=lambda *a, **k: None)),
    )

    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._loaded = True
    model.model = SimpleNamespace(language_model=object())
    model.processor = SimpleNamespace(tokenizer=tokenizer)
    model.config = {}
    model._draft_model = configured_draft
    model._cache_manager = None

    output = model.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        mllm_draft=True,
        draft_model=caller_draft,
        draft_kind="other",
        draft_block_size=99,
    )

    assert output.text == "ok"
    assert captured["kwargs"]["draft_model"] is configured_draft
    assert captured["kwargs"]["draft_kind"] == "mtp"
    assert captured["kwargs"]["draft_block_size"] == 4


def test_mllm_chat_requires_request_draft_opt_in(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}
    configured_draft = SimpleNamespace(accept_lens=[], _vllm_mlx_draft_counts=[])

    def fake_generate(*args, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=1)

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(generate=fake_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        SimpleNamespace(get_chat_template=lambda *args, **kwargs: "rendered prompt"),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models",
        SimpleNamespace(cache=SimpleNamespace(make_prompt_cache=lambda *a, **k: None)),
    )

    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._loaded = True
    model.model = SimpleNamespace(language_model=object())
    model.processor = SimpleNamespace(tokenizer=tokenizer)
    model.config = {}
    model._draft_model = configured_draft
    model._cache_manager = None

    output = model.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        draft_model=object(),
        draft_kind="other",
        draft_block_size=99,
    )

    assert output.text == "ok"
    assert "mllm_draft" not in captured["kwargs"]
    assert "draft_model" not in captured["kwargs"]
    assert "draft_kind" not in captured["kwargs"]
    assert "draft_block_size" not in captured["kwargs"]
    assert output.mtp_drafts == 0
    assert output.mtp_accepted == 0


def test_simple_engine_text_route_stays_default_when_mllm_drafter_configured():
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine(
        "gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=4,
    )
    engine._loaded = True
    engine._text_model = object()

    assert engine._should_route_text_through_text_model() is True
    assert (
        engine._should_route_text_through_text_model(mllm_draft_requested=True) is False
    )


def test_chat_request_passes_mllm_draft_opt_in():
    from vllm_mlx.server import (
        ChatCompletionRequest,
        Message,
        _prepare_chat_completion_invocation,
    )

    class Engine:
        is_mllm = True
        preserve_native_tool_format = False

    request = ChatCompletionRequest(
        model="gemma4",
        messages=[Message(role="user", content="hello")],
        mllm_draft=True,
    )

    prepared = _prepare_chat_completion_invocation(Engine(), request, 16)

    assert prepared.chat_kwargs["mllm_draft"] is True


def test_chat_request_uses_server_draft_default_and_allows_opt_out(monkeypatch):
    from vllm_mlx import server
    from vllm_mlx.server import (
        ChatCompletionRequest,
        Message,
        _prepare_chat_completion_invocation,
    )

    class Engine:
        is_mllm = True
        preserve_native_tool_format = False

    monkeypatch.setattr(server, "_default_mllm_draft", True)
    implicit = ChatCompletionRequest(
        model="laguna", messages=[Message(role="user", content="hello")]
    )
    disabled = ChatCompletionRequest(
        model="laguna",
        messages=[Message(role="user", content="hello")],
        mllm_draft=False,
    )

    assert _prepare_chat_completion_invocation(Engine(), implicit, 16).chat_kwargs[
        "mllm_draft"
    ] is True
    assert _prepare_chat_completion_invocation(Engine(), disabled, 16).chat_kwargs[
        "mllm_draft"
    ] is False


@pytest.mark.anyio
async def test_simple_engine_forwards_mllm_draft_opt_in_to_mllm_path():
    from vllm_mlx.engine.simple import SimpleEngine

    captured = {}

    class FakeMLLM:
        def stream_chat(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            yield SimpleNamespace(
                text="ok",
                finish_reason="stop",
                prompt_tokens=3,
                mtp_drafts=2,
                mtp_accepted=1,
            )

    engine = SimpleEngine(
        "gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=4,
    )
    engine._loaded = True
    engine._is_mllm = True
    engine._text_model = object()
    engine._model = FakeMLLM()

    outputs = [
        output
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=8,
            temperature=0.0,
            mllm_draft=True,
        )
    ]

    assert captured["kwargs"]["mllm_draft"] is True
    assert outputs[-1].mtp_drafts == 2
    assert outputs[-1].mtp_accepted == 1
