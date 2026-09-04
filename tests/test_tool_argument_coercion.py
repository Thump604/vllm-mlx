import json

import pytest

from vllm_mlx.server import (
    _coerce_tool_arguments,
    _finalize_streaming_tool_calls,
    _merge_streaming_tool_call_fragments,
    _requires_buffered_tool_argument_coercion,
)


def _tool_schema(properties):
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {"type": "object", "properties": properties},
            },
        }
    ]


def test_coerces_json_encoded_array_and_integer_to_declared_types():
    arguments = json.dumps(
        {
            "argv": '["/usr/bin/printf", "transport-ok\\n"]',
            "timeout_seconds": "1",
        }
    )
    tools = _tool_schema(
        {
            "argv": {"type": "array", "items": {"type": "string"}},
            "timeout_seconds": {"type": "integer"},
        }
    )

    assert json.loads(_coerce_tool_arguments(arguments, "terminal", tools)) == {
        "argv": ["/usr/bin/printf", "transport-ok\n"],
        "timeout_seconds": 1,
    }


def test_preserves_invalid_value_when_conversion_is_not_lossless():
    arguments = json.dumps({"argv": "not-json", "timeout_seconds": "1.5"})
    tools = _tool_schema(
        {
            "argv": {"type": "array"},
            "timeout_seconds": {"type": "integer"},
        }
    )

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_value_that_already_matches_any_union_type():
    arguments = json.dumps({"timeout_seconds": "1"})
    tools = _tool_schema({"timeout_seconds": {"type": ["integer", "string"]}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"answer": 42, "labels": ["café", "ok"]},
            '{"answer":42,"labels":["café","ok"]}',
        ),
        ([{"answer": 42}, "café"], '[{"answer":42},"café"]'),
    ],
)
def test_object_and_array_to_string_normalization_uses_compact_json(value, expected):
    arguments = json.dumps({"content": value})
    tools = _tool_schema({"content": {"type": "string"}})

    normalized = json.loads(_coerce_tool_arguments(arguments, "terminal", tools))
    assert normalized["content"] == expected


def test_preserves_existing_string_bytes():
    content = '{\n  "answer": 42,\n  "label": "café"\n}'
    arguments = json.dumps({"content": content})
    tools = _tool_schema({"content": {"type": "string"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_coerces_json_boolean_without_treating_it_as_integer():
    arguments = json.dumps({"enabled": "true", "count": True})
    tools = _tool_schema({"enabled": {"type": "boolean"}, "count": {"type": "integer"}})

    assert json.loads(_coerce_tool_arguments(arguments, "terminal", tools)) == {
        "enabled": True,
        "count": True,
    }


@pytest.mark.parametrize("encoded", ["1e999", "-1e999", "NaN", "Infinity"])
def test_preserves_nonfinite_numeric_strings_byte_for_byte(encoded):
    arguments = json.dumps({"value": encoded})
    tools = _tool_schema({"value": {"type": "number"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


@pytest.mark.parametrize(
    ("encoded", "declared_type", "expected"),
    [
        (
            '[[1, 2], {"nested": [3, 4]}]',
            "array",
            [[1, 2], {"nested": [3, 4]}],
        ),
        (
            '{"nested": {"values": [1, 2]}, "ok": true}',
            "object",
            {"nested": {"values": [1, 2]}, "ok": True},
        ),
    ],
)
def test_recovers_finite_nested_arrays_and_objects_without_bare_nonfinite(
    encoded, declared_type, expected
):
    arguments = json.dumps({"value": encoded}, ensure_ascii=False)
    tools = _tool_schema({"value": {"type": declared_type}})

    normalized = _coerce_tool_arguments(arguments, "terminal", tools)

    def reject_nonfinite(token):
        raise AssertionError(f"unexpected nonfinite JSON constant: {token}")

    assert json.loads(normalized, parse_constant=reject_nonfinite) == {
        "value": expected
    }


@pytest.mark.parametrize(
    "encoded",
    ['[1, {"nested": 1e999}]', '{"nested": [Infinity]}'],
)
def test_preserves_nested_nonfinite_json_strings_byte_for_byte(encoded):
    arguments = json.dumps({"value": encoded})
    declared_type = "object" if encoded.startswith("{") else "array"
    tools = _tool_schema({"value": {"type": declared_type}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_invalid_object_instead_of_stringifying_nonfinite_values():
    arguments = '{ "value": {"nested": [NaN]} }'
    tools = _tool_schema({"value": {"type": "string"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_invalid_outer_payload_when_another_field_is_recoverable():
    arguments = '{ "count": "1", "unrelated": Infinity }'
    tools = _tool_schema({"count": {"type": "integer"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_changed_serialization_never_emits_bare_nonfinite():
    arguments = json.dumps(
        {"argv": '["printf"]', "value": "1e999", "options": '{"ok": true}'},
        ensure_ascii=False,
    )
    tools = _tool_schema(
        {
            "argv": {"type": "array"},
            "value": {"type": "number"},
            "options": {"type": "object"},
        }
    )

    normalized = _coerce_tool_arguments(arguments, "terminal", tools)

    def reject_nonfinite(token):
        raise AssertionError(f"unexpected nonfinite JSON constant: {token}")

    assert json.loads(normalized, parse_constant=reject_nonfinite) == {
        "argv": ["printf"],
        "value": "1e999",
        "options": {"ok": True},
    }


def test_buffers_typed_stream_fragments_until_complete_json_is_available():
    tools = _tool_schema(
        {"argv": {"type": "array"}, "timeout_seconds": {"type": "integer"}}
    )
    calls = {}
    fragments = [
        {"index": 0, "id": "call_1", "function": {"name": "terminal"}},
        {"index": 0, "function": {"arguments": '{"argv": "[\\"printf\\"]", '}},
        {"index": 0, "function": {"arguments": '"timeout_seconds": "1"}'}},
    ]

    assert _requires_buffered_tool_argument_coercion(tools) is True
    for fragment in fragments:
        _merge_streaming_tool_call_fragments(calls, [fragment])

    finalized = _finalize_streaming_tool_calls(calls, tools)
    assert finalized == [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": json.dumps(
                    {"argv": ["printf"], "timeout_seconds": 1},
                    ensure_ascii=False,
                ),
            },
        }
    ]


def test_string_only_schema_preserves_incremental_streaming():
    assert (
        _requires_buffered_tool_argument_coercion(
            _tool_schema({"content": {"type": "string"}})
        )
        is False
    )


@pytest.mark.anyio
async def test_stream_buffers_fragments_before_schema_coercion(monkeypatch):
    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.server import stream_chat_completion
    import vllm_mlx.server as server

    fragments = [
        {"index": 0, "id": "call_1", "function": {"name": "terminal"}},
        {"index": 0, "function": {"arguments": '{"argv": '}},
        {"index": 0, "function": {"arguments": '"[\\"printf\\"]"'}},
        {"index": 0, "function": {"arguments": ', "timeout_seconds": "1"}'}},
    ]

    class FakeEngine:
        model_name = "fake-engine"

        async def stream_chat(self, messages, **kwargs):
            for index in range(len(fragments) + 1):
                finished = index == len(fragments)
                yield GenerationOutput(
                    text="",
                    new_text=f"fragment-{index}",
                    finished=finished,
                    finish_reason="stop" if finished else None,
                    prompt_tokens=3 if finished else 0,
                    completion_tokens=4 if finished else 0,
                )

    class FakeParser:
        def __init__(self):
            self.index = 0

        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, request=None
        ):
            if self.index == len(fragments):
                return None
            fragment = fragments[self.index]
            self.index += 1
            return {"tool_calls": [fragment]}

    monkeypatch.setattr(server, "_model_name", "served-model")
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_get_streaming_tool_parser", lambda *_: FakeParser())
    monkeypatch.setattr(
        server, "_streaming_tool_markup_possible_after_delta", lambda *_: True
    )

    request = ChatCompletionRequest(
        model="served-model",
        messages=[Message(role="user", content="run it")],
        tools=_tool_schema(
            {"argv": {"type": "array"}, "timeout_seconds": {"type": "integer"}}
        ),
        stream=True,
    )
    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            FakeEngine(), request.messages, request
        )
    ]
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk != "data: [DONE]\n\n"
    ]
    tool_payloads = [
        payload
        for payload in payloads
        if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
    ]

    assert len(tool_payloads) == 1
    call = tool_payloads[0]["choices"][0]["delta"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {
        "argv": ["printf"],
        "timeout_seconds": 1,
    }
    assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
@pytest.mark.parametrize("use_reasoning_parser", [False, True])
async def test_stream_emits_sibling_content_and_reasoning_before_terminal_tool_call(
    monkeypatch, use_reasoning_parser
):
    from types import SimpleNamespace

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.server import stream_chat_completion
    import vllm_mlx.server as server

    fragments = [
        {"index": 0, "id": "call_1", "function": {"name": "terminal"}},
        {"index": 0, "function": {"arguments": '{"argv": '}},
        {"index": 0, "function": {"arguments": '"[\\"printf\\"]"}'}},
    ]
    parser_results = [
        {"tool_calls": [fragments[0]], "content": "sibling content"},
        {"tool_calls": [fragments[1]]},
        {"tool_calls": [fragments[2]]},
        None,
    ]

    class FakeEngine:
        model_name = "fake-engine"

        async def stream_chat(self, messages, **kwargs):
            for index in range(len(parser_results)):
                yield GenerationOutput(
                    text="",
                    new_text="<tool_call>" if index == 0 else "delta",
                    finished=index == len(parser_results) - 1,
                    finish_reason="stop" if index == len(parser_results) - 1 else None,
                    prompt_tokens=3 if index == len(parser_results) - 1 else 0,
                    completion_tokens=4 if index == len(parser_results) - 1 else 0,
                )

    class FakeParser:
        def __init__(self):
            self.index = 0

        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, request=None
        ):
            result = parser_results[self.index]
            self.index += 1
            return result

    class FakeReasoningParser:
        def __init__(self, tokenizer=None):
            self.index = 0

        def reset_state(self):
            self.index = 0

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            if self.index == 0:
                self.index += 1
                return SimpleNamespace(
                    reasoning="sibling reasoning", content=delta_text
                )
            self.index += 1
            return SimpleNamespace(reasoning=None, content=delta_text)

    monkeypatch.setattr(server, "_model_name", "served-model")
    monkeypatch.setattr(
        server,
        "_reasoning_parser",
        FakeReasoningParser() if use_reasoning_parser else None,
    )
    monkeypatch.setattr(server, "_reasoning_parser_name", None)
    monkeypatch.setattr(server, "_get_streaming_tool_parser", lambda *_: FakeParser())
    monkeypatch.setattr(
        server, "_streaming_tool_markup_possible_after_delta", lambda *_: True
    )

    request = ChatCompletionRequest(
        model="served-model",
        messages=[Message(role="user", content="run it")],
        tools=_tool_schema({"argv": {"type": "array"}}),
        stream=True,
    )
    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            FakeEngine(), request.messages, request
        )
    ]
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk != "data: [DONE]\n\n"
    ]
    choices = [payload["choices"][0] for payload in payloads if payload["choices"]]
    content_deltas = [
        choice["delta"].get("content")
        for choice in choices
        if choice["delta"].get("content")
    ]
    reasoning_deltas = [
        choice["delta"].get("reasoning_content")
        for choice in choices
        if choice["delta"].get("reasoning_content")
    ]
    tool_payloads = [
        payload
        for payload in payloads
        if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
    ]

    assert content_deltas == ["sibling content"]
    assert reasoning_deltas == (["sibling reasoning"] if use_reasoning_parser else [])
    assert len(tool_payloads) == 1
    tool_index = payloads.index(tool_payloads[0])
    sibling_index = next(
        index
        for index, payload in enumerate(payloads)
        if payload["choices"]
        and payload["choices"][0]["delta"].get("content") == "sibling content"
    )
    assert sibling_index < tool_index
    call = tool_payloads[0]["choices"][0]["delta"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"argv": ["printf"]}
    assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"
