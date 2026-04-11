# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma 4 tool call parser."""

import json

from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


class TestGemma4ToolParserExtract:
    """Test extract_tool_calls on complete model output."""

    def setup_method(self):
        self.parser = Gemma4ToolParser()

    def test_single_tool_call_string_arg(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/foo.py<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call["name"] == "read_file"
        assert json.loads(tool_call["arguments"]) == {"path": "/tmp/foo.py"}
        assert result.content is None

    def test_single_tool_call_numeric_arg(self):
        output = "<|tool_call>call:search{limit:10,verbose:false}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "limit": 10,
            "verbose": False,
        }

    def test_mixed_types(self):
        output = '<|tool_call>call:search{query:<|"|>hello world<|"|>,limit:10,verbose:false}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "query": "hello world",
            "limit": 10,
            "verbose": False,
        }

    def test_nested_object(self):
        output = '<|tool_call>call:configure{settings:{enabled:true,name:<|"|>test<|"|>}}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "settings": {"enabled": True, "name": "test"}
        }

    def test_array_argument(self):
        output = '<|tool_call>call:tag{items:[<|"|>foo<|"|>,<|"|>bar<|"|>]}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "items": ["foo", "bar"]
        }

    def test_multiple_tool_calls_in_one_block(self):
        output = (
            "<|tool_call>"
            'call:glob{pattern:<|"|>README*.md<|"|>}'
            'call:glob{pattern:<|"|>CONTRIBUTING.md<|"|>}'
            "<tool_call|>"
        )
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "pattern": "README*.md"
        }
        assert json.loads(result.tool_calls[1]["arguments"]) == {
            "pattern": "CONTRIBUTING.md"
        }

    def test_content_before_tool_call(self):
        output = 'Let me read that file for you.\n<|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert result.content == "Let me read that file for you."
        assert len(result.tool_calls) == 1

    def test_no_tool_calls(self):
        output = "Hello, how can I help you today?"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == output

    def test_empty_tool_call_block(self):
        output = "<|tool_call><tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is False
        assert result.tool_calls == []

    def test_tool_call_id_generated(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/a<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tool_calls[0]["id"].startswith("call_")

    def test_string_with_special_chars(self):
        output = '<|tool_call>call:write{content:<|"|>line1\\nline2<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert (
            json.loads(result.tool_calls[0]["arguments"])["content"] == "line1\\nline2"
        )

    def test_deeply_nested_objects(self):
        output = "<|tool_call>call:update{a:{b:{c:1,d:true}}}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "a": {"b": {"c": 1, "d": True}}
        }

    def test_null_value(self):
        output = "<|tool_call>call:clear{target:null}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {"target": None}

    def test_unicode_emoji_in_args(self):
        output = (
            '<|tool_call>call:search{query:<|"|>hello world 🌍 你好<|"|>}<tool_call|>'
        )
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "query": "hello world 🌍 你好"
        }

    def test_braces_inside_string_value(self):
        output = '<|tool_call>call:run{code:<|"|>if (x) { return y; }<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "code": "if (x) { return y; }"
        }

    def test_quoted_keys(self):
        output = '<|tool_call>call:read{<|"|>path<|"|>:<|"|>/tmp/foo<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {"path": "/tmp/foo"}

    def test_think_tags_stripped(self):
        output = '<think>Let me think about this...</think><|tool_call>call:search{query:<|"|>test<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1

    def test_missing_end_delimiter(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0]["arguments"]) == {"path": "/tmp/foo"}

    def test_string_with_colon(self):
        output = '<|tool_call>call:connect{url:<|"|>host:8080<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {"url": "host:8080"}

    def test_string_with_newline_and_quote(self):
        output = '<|tool_call>call:write{text:<|"|>line1\nline2 said "hello"<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "text": 'line1\nline2 said "hello"'
        }


class TestGemma4ToolParserStreaming:
    """Test streaming tool call extraction."""

    def setup_method(self):
        self.parser = Gemma4ToolParser()
        self.parser.reset()

    def test_streaming_no_tool_call(self):
        result = self.parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Hello",
            delta_text="Hello",
        )
        assert result == {"content": "Hello"}

    def test_streaming_suppresses_during_tool_call(self):
        result = self.parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Sure. ",
            delta_text="Sure. ",
        )
        assert result == {"content": "Sure. "}

        result = self.parser.extract_tool_calls_streaming(
            previous_text="Sure. ",
            current_text="Sure. <|tool_call>call:read",
            delta_text="<|tool_call>call:read",
        )
        assert result is None

        result = self.parser.extract_tool_calls_streaming(
            previous_text="Sure. <|tool_call>call:read",
            current_text='Sure. <|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}',
            delta_text='_file{path:<|"|>/tmp/foo<|"|>}',
        )
        assert result is None

    def test_streaming_emits_on_close(self):
        full_text = (
            'Sure. <|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}<tool_call|>'
        )
        result = self.parser.extract_tool_calls_streaming(
            previous_text='Sure. <|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}',
            current_text=full_text,
            delta_text="<tool_call|>",
        )
        assert result is not None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        tool_call = result["tool_calls"][0]
        assert tool_call["function"]["name"] == "read_file"
        assert tool_call["type"] == "function"
        assert tool_call["index"] == 0


class TestGemma4Registration:
    """Test parser registration and flags."""

    def test_registered_in_manager(self):
        from vllm_mlx.tool_parsers import ToolParserManager

        parser_cls = ToolParserManager.get_tool_parser("gemma4")
        assert parser_cls is Gemma4ToolParser

    def test_native_format_false(self):
        assert Gemma4ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is False
        assert Gemma4ToolParser.supports_native_format() is False

    def test_auto_parser_detects_gemma4_format(self):
        from vllm_mlx.tool_parsers import AutoToolParser

        parser = AutoToolParser()
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/foo.py<|"|>}<tool_call|>'

        result = parser.extract_tool_calls(output)

        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "read_file"
