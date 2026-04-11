# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Gemma 4 tool-call OpenAI response formatting."""

import json

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    Usage,
)
from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


def _build_response_from_parser(parser_output, model_name="gemma-4-27b-it"):
    """Mirror the server response wrapper for parsed tool-call output."""
    if parser_output.tools_called:
        tool_calls = [
            ToolCall(
                id=tool_call.get("id", "call_test"),
                type="function",
                function=FunctionCall(
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                ),
            )
            for tool_call in parser_output.tool_calls
        ]
        content = parser_output.content if parser_output.content else None
        finish_reason = "tool_calls"
    else:
        tool_calls = None
        content = parser_output.content
        finish_reason = "stop"

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestGemma4OpenAIFormat:
    """Verify the response matches OpenAI-compatible client expectations."""

    def setup_method(self):
        self.parser = Gemma4ToolParser()

    def test_tool_call_response_has_correct_structure(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/test.py<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)

        data = json.loads(response.model_dump_json(exclude_none=True))
        choice = data["choices"][0]
        message = choice["message"]

        assert choice["finish_reason"] == "tool_calls"
        assert message.get("content") is None
        assert isinstance(message["tool_calls"], list)
        assert len(message["tool_calls"]) == 1

        tool_call = message["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["id"]
        assert isinstance(tool_call["id"], str)
        assert tool_call["function"]["name"] == "read_file"
        assert isinstance(tool_call["function"]["arguments"], str)
        assert json.loads(tool_call["function"]["arguments"]) == {
            "path": "/tmp/test.py"
        }

    def test_multiple_tool_calls_response(self):
        output = (
            "<|tool_call>"
            'call:read_file{path:<|"|>/a.py<|"|>}'
            'call:read_file{path:<|"|>/b.py<|"|>}'
            "<tool_call|>"
        )
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "read_file"
        assert tool_calls[1]["function"]["name"] == "read_file"
        assert tool_calls[0]["id"] != tool_calls[1]["id"]

    def test_content_before_tool_call_preserved(self):
        output = 'Let me check that.\n<|tool_call>call:read_file{path:<|"|>/tmp/x<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        message = data["choices"][0]["message"]
        assert message["content"] == "Let me check that."
        assert len(message["tool_calls"]) == 1

    def test_no_tool_call_response(self):
        output = "The answer is 42."
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        message = data["choices"][0]["message"]
        assert message["content"] == "The answer is 42."
        assert "tool_calls" not in message
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_complex_arguments_serialize_correctly(self):
        output = '<|tool_call>call:configure{settings:{enabled:true,tags:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        assert json.loads(tool_call["function"]["arguments"]) == {
            "settings": {"enabled": True, "tags": ["a", "b"]}
        }
