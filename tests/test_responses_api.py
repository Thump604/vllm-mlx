# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible Responses API."""

import platform
import sys
from types import SimpleNamespace
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


@pytest.fixture()
def client():
    from vllm_mlx.server import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def server_state():
    import vllm_mlx.server as srv

    original_engine = srv._engine
    original_model_name = srv._model_name
    original_store = srv._responses_store
    original_api_key = srv._api_key

    srv._engine = None
    srv._model_name = "test-model"
    srv._responses_store = OrderedDict()
    srv._api_key = None

    try:
        yield
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name
        srv._responses_store = original_store
        srv._api_key = original_api_key


def _mock_engine(*outputs):
    engine = MagicMock()
    engine.model_name = "test-model"
    engine.preserve_native_tool_format = False
    engine.chat = AsyncMock(side_effect=list(outputs))
    stream_calls = []

    async def _stream_chat(**kwargs):
        stream_calls.append(kwargs)
        for output in getattr(engine, "_stream_outputs", []):
            yield output

    engine._stream_calls = stream_calls
    engine._stream_outputs = []
    engine.stream_chat = _stream_chat
    return engine


def _output(text: str, prompt_tokens: int = 7, completion_tokens: int = 3):
    return SimpleNamespace(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
    )


def _stream_output(
    new_text: str,
    prompt_tokens: int = 7,
    completion_tokens: int = 1,
    finish_reason: str | None = None,
):
    return SimpleNamespace(
        new_text=new_text,
        text=new_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        finished=finish_reason is not None,
    )


class TestResponsesEndpoint:
    def test_basic_response(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello there"))

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Say hello"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "response"
        assert body["output_text"] == "Hello there"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["type"] == "output_text"
        assert body["usage"]["input_tokens"] == 7
        assert body["usage"]["output_tokens"] == 3

    def test_previous_response_id_reuses_prior_context(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "user"
        assert second_messages[0]["content"] == "First prompt"
        assert second_messages[1]["role"] == "assistant"
        assert second_messages[1]["content"] == "First answer"
        assert second_messages[2]["role"] == "user"
        assert second_messages[2]["content"] == "Follow-up prompt"

    def test_previous_response_id_chains_across_multiple_follow_ups(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(
            _output("First answer"),
            _output("Second answer"),
            _output("Third answer"),
        )
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Second prompt",
            },
        )
        second_id = second.json()["id"]

        third = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": second_id,
                "input": "Third prompt",
            },
        )

        assert third.status_code == 200
        third_messages = engine.chat.call_args_list[2].kwargs["messages"]
        assert third_messages[0]["role"] == "user"
        assert third_messages[0]["content"] == "First prompt"
        assert third_messages[1]["role"] == "assistant"
        assert third_messages[1]["content"] == "First answer"
        assert third_messages[2]["role"] == "user"
        assert third_messages[2]["content"] == "Second prompt"
        assert third_messages[3]["role"] == "assistant"
        assert third_messages[3]["content"] == "Second answer"
        assert third_messages[4]["role"] == "user"
        assert third_messages[4]["content"] == "Third prompt"

    def test_previous_response_id_does_not_carry_prior_instructions(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "First system instruction",
                "input": "First prompt",
            },
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "Second system instruction",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "system"
        assert second_messages[0]["content"] == "Second system instruction"
        assert "First system instruction" not in second_messages[0]["content"]
        assert second_messages[1]["role"] == "user"
        assert second_messages[1]["content"] == "First prompt"
        assert second_messages[2]["role"] == "assistant"
        assert second_messages[3]["role"] == "user"

    def test_previous_response_id_preserves_prior_system_message_items(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "system", "content": "Persist me"},
                    {"type": "message", "role": "user", "content": "First prompt"},
                ],
            },
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "system"
        assert second_messages[0]["content"] == "Persist me"
        assert second_messages[1]["role"] == "user"
        assert second_messages[1]["content"] == "First prompt"
        assert second_messages[2]["role"] == "assistant"
        assert second_messages[3]["role"] == "user"

    def test_developer_role_is_normalized_to_system(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Hi"},
                    {"type": "message", "role": "developer", "content": "Be terse"},
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be terse"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"

    def test_instructions_and_developer_message_are_merged(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "System instructions",
                "input": [
                    {"type": "message", "role": "developer", "content": "Developer note"},
                    {"type": "message", "role": "user", "content": "Hi"},
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert len([m for m in messages if m["role"] == "system"]) == 1
        assert messages[0]["role"] == "system"
        assert "System instructions" in messages[0]["content"]
        assert "Developer note" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_function_call_output_input_is_mapped_cleanly(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Done"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Run it"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "shell",
                        "arguments": "{\"cmd\":\"pwd\"}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "/tmp/work",
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[1]["role"] == "assistant"
        assert "[Calling tool: shell(" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "[Tool Result (call_1)]" in messages[2]["content"]
        assert "/tmp/work" in messages[2]["content"]

    def test_unsupported_tools_and_items_do_not_fail(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Fallback answer"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Answer directly"},
                    {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {"type": "search", "query": "ignored"},
                    },
                ],
                "tools": [
                    {"type": "web_search_preview"},
                    {"type": "file_search", "vector_store_ids": ["vs_123"]},
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "not available on this backend" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert engine.chat.call_args.kwargs["tools"][0]["type"] == "function"

    def test_function_call_response_item(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(
            _output('<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>')
        )

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Use a tool",
                "tools": [
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["output"][0]["type"] == "function_call"
        assert body["output"][0]["name"] == "shell"
        assert body["output_text"] == ""

    def test_streaming_response_returns_sse_events(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(side_effect=AssertionError("stream path should not call chat"))
        engine._stream_outputs = [
            _stream_output("Hello ", completion_tokens=1),
            _stream_output("stream", completion_tokens=2, finish_reason="stop"),
        ]
        srv._engine = engine

        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "stream": True},
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        assert "event: response.created" in body
        assert "event: response.output_text.delta" in body
        assert "Hello stream" in body
        assert "event: response.completed" in body
        assert len(engine._stream_calls) == 1
        engine.chat.assert_not_awaited()

    def test_json_object_response_format_is_rejected(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello"))

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "text": {"format": {"type": "json_object"}},
            },
        )

        assert resp.status_code == 400
        assert "json_object" in resp.json()["detail"]

    def test_reasoning_configuration_is_rejected(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello"))

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "reasoning": {"effort": "xhigh"},
            },
        )

        assert resp.status_code == 400
        assert "reasoning configuration" in resp.json()["detail"]

    def test_reasoning_input_item_is_rejected(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello"))

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Hello"},
                    {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "x"}]},
                ],
            },
        )

        assert resp.status_code == 400
        assert "reasoning input items are not supported" in resp.json()["detail"]

    def test_length_finish_reason_marks_response_incomplete(self, client):
        import vllm_mlx.server as srv

        output = _output("Cut off", completion_tokens=5)
        output.finish_reason = "length"
        srv._engine = _mock_engine(output)

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "max_output_tokens": 5},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": "max_output_tokens"}
