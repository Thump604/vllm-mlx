# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's native tool call format:
  <|tool_call>call:func_name{<|"|>key<|"|>: <|"|>value<|"|>, num: 42}<tool_call|>

Gemma 4 uses special tokens instead of JSON:
- <|tool_call> / <tool_call|> delimit tool call blocks
- <|"|> replaces " for string values
- Keys are unquoted bare identifiers
- Multiple call:name{...} can appear in a single block
"""

import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")
_STRING_DELIM_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_CALL_PREFIX = re.compile(r"call:(\w+)\s*\{")
_BARE_KEY = re.compile(r"(?<=[{,])\s*(\w+)\s*:")
_MAX_ARG_BLOCK_LEN = 1_048_576


def _find_balanced_brace(text: str, start: int) -> int:
    """Find the index of the closing } that balances the { at `start`."""
    if len(text) - start > _MAX_ARG_BLOCK_LEN:
        return -1

    depth = 0
    i = start
    in_string = False
    while i < len(text):
        if text.startswith('<|"|>', i):
            in_string = not in_string
            i += 5
            continue
        if not in_string:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _gemma4_args_to_json(text: str) -> str:
    """Convert Gemma 4 tool call args into valid JSON."""
    strings: list[str] = []

    def _capture(match: re.Match) -> str:
        strings.append(match.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = _STRING_DELIM_RE.sub(_capture, text)
    text = _BARE_KEY.sub(r'"\1":', text)

    def _restore(match: re.Match) -> str:
        idx = int(match.group(1))
        return json.dumps(strings[idx]) if idx < len(strings) else match.group(0)

    return _PLACEHOLDER_RE.sub(_restore, text)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module("gemma4")
class Gemma4ToolParser(ToolParser):
    """Tool call parser for Gemma 4 models."""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Gemma 4 model response."""
        cleaned = self.strip_think_tags(model_output)

        start_idx = cleaned.find(TOOL_CALL_START)
        if start_idx == -1:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_before = cleaned[:start_idx].strip() or None

        block_start = start_idx + len(TOOL_CALL_START)
        end_idx = cleaned.find(TOOL_CALL_END, block_start)
        if end_idx == -1:
            block = cleaned[block_start:]
        else:
            block = cleaned[block_start:end_idx]

        tool_calls: list[dict[str, Any]] = []
        pos = 0
        while pos < len(block):
            match = _CALL_PREFIX.search(block, pos)
            if not match:
                break

            func_name = match.group(1)
            brace_start = match.end() - 1
            brace_end = _find_balanced_brace(block, brace_start)
            if brace_end == -1:
                pos = match.end()
                continue

            args_raw = block[brace_start : brace_end + 1]
            try:
                args_json = _gemma4_args_to_json(args_raw)
                json.loads(args_json)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": args_json,
                    }
                )
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Gemma 4 tool parser: failed to parse args for call:%s: %s",
                    func_name,
                    exc,
                )

            pos = brace_end + 1

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_before,
            )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract tool calls from streaming Gemma 4 model output."""
        if TOOL_CALL_START not in current_text:
            return {"content": delta_text}

        if TOOL_CALL_END in delta_text:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                            },
                        }
                        for i, tool_call in enumerate(result.tool_calls)
                    ]
                }

        return None
