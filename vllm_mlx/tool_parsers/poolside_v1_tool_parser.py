# SPDX-License-Identifier: Apache-2.0
"""Tool parser for the Poolside v1 Laguna chat-template format."""

import json
import re
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParserManager,
)
from .glm47_tool_parser import Glm47ToolParser, generate_tool_id


@ToolParserManager.register_module("poolside_v1")
class PoolsideV1ToolParser(Glm47ToolParser):
    """Parse Laguna tool calls and stream schema-declared strings incrementally."""

    _UNCLOSED_TOOL_CALL = re.compile(r"<tool_call>.*$", re.DOTALL)
    _START = "<tool_call>"
    _END = "</tool_call>"
    _KEY_START = "<arg_key>"
    _KEY_END = "</arg_key>"
    _VALUE_START = "<arg_value>"
    _VALUE_END = "</arg_value>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._buffer = ""
        self._in_tool_call = False
        self._current_tool_name: str | None = None
        self._pending_key: str | None = None
        self._streaming_string_value = False
        self._reject_current = False
        self._tool_ids: list[str] = []
        self._args_started: list[bool] = []
        self._args_closed: list[bool] = []
        self._seen_keys: list[set[str]] = []

    @staticmethod
    def _string_argument_names(
        request: dict[str, Any] | None, tool_name: str
    ) -> set[str]:
        if request is None:
            return set()
        for tool in request.get("tools", []):
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict) or function.get("name") != tool_name:
                continue
            parameters = function.get("parameters")
            properties = (
                parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            )
            return {
                name
                for name, schema in properties.items()
                if isinstance(schema, dict) and schema.get("type") == "string"
            }
        return set()

    @staticmethod
    def _escape_string_content(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)[1:-1]

    @staticmethod
    def _hold_partial_suffix(buffer: str, marker: str) -> tuple[str, str]:
        for size in range(min(len(marker) - 1, len(buffer)), 0, -1):
            if buffer.endswith(marker[:size]):
                return buffer[:-size], buffer[-size:]
        return buffer, ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        cleaned_text = self.strip_think_tags(model_output)
        valid_names = self._get_tool_names(request)
        tool_calls: list[dict[str, Any]] = []

        for match in self.FUNC_DETAIL_PATTERN.finditer(cleaned_text):
            tool_name = match.group(1).strip()
            if not tool_name or (valid_names and tool_name not in valid_names):
                continue
            string_arguments = self._string_argument_names(request, tool_name)
            arguments: dict[str, Any] = {}
            for raw_key, raw_value in self.ARG_PATTERN.findall(match.group(2) or ""):
                key = raw_key.strip()
                if not key or key in arguments:
                    continue
                arguments[key] = (
                    raw_value
                    if key in string_arguments
                    else self._deserialize(raw_value.strip())
                )
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        marker = cleaned_text.find(self._START)
        content = cleaned_text[:marker] if marker >= 0 else cleaned_text
        content = content.strip() or None
        if tool_calls:
            return ExtractedToolCallInformation(True, tool_calls, content)

        if marker >= 0:
            content = self._UNCLOSED_TOOL_CALL.sub("", cleaned_text).strip() or None
        return ExtractedToolCallInformation(False, [], content)

    def _begin_tool_call(self) -> None:
        self.current_tool_id += 1
        self._tool_ids.append(generate_tool_id())
        self._args_started.append(False)
        self._args_closed.append(False)
        self._seen_keys.append(set())
        self._in_tool_call = True
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False
        self._reject_current = False

    def _finish_tool_call(self) -> None:
        self._in_tool_call = False
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False
        self._reject_current = False

    def _delta(
        self,
        pending: dict[int, dict[str, Any]],
        *,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        if self._reject_current:
            return
        delta = pending.setdefault(
            self.current_tool_id,
            {
                "index": self.current_tool_id,
                "id": self._tool_ids[self.current_tool_id],
                "type": "function",
                "function": {"arguments": ""},
            },
        )
        if name is not None:
            delta["function"]["name"] = name
        delta["function"]["arguments"] += arguments

    def _argument_prefix(self, key: str) -> str | None:
        seen = self._seen_keys[self.current_tool_id]
        if not key or key in seen:
            return None
        seen.add(key)
        separator = "{" if not self._args_started[self.current_tool_id] else ", "
        self._args_started[self.current_tool_id] = True
        return separator + json.dumps(key, ensure_ascii=False) + ": "

    def _close_arguments(self) -> str:
        if self._args_closed[self.current_tool_id]:
            return ""
        self._args_closed[self.current_tool_id] = True
        return "}" if self._args_started[self.current_tool_id] else "{}"

    def _discard_through_tool_end(self) -> bool:
        end = self._buffer.find(self._END)
        if end < 0:
            return False
        self._buffer = self._buffer[end + len(self._END) :]
        self._finish_tool_call()
        return True

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
        del previous_text, current_text, previous_token_ids, current_token_ids
        del delta_token_ids
        self._buffer += delta_text
        pending: dict[int, dict[str, Any]] = {}
        content = ""
        valid_names = self._get_tool_names(request)

        while True:
            if not self._in_tool_call:
                start = self._buffer.find(self._START)
                if start < 0:
                    emitted, self._buffer = self._hold_partial_suffix(
                        self._buffer, self._START
                    )
                    content += emitted
                    break
                content += self._buffer[:start]
                self._buffer = self._buffer[start + len(self._START) :]
                self._begin_tool_call()
                continue

            if self._current_tool_name is None:
                positions = [
                    position
                    for position in (
                        self._buffer.find("\n"),
                        self._buffer.find(self._KEY_START),
                        self._buffer.find(self._END),
                    )
                    if position >= 0
                ]
                if not positions:
                    break
                cut = min(positions)
                tool_name = self._buffer[:cut].strip()
                if self._buffer.startswith("\n", cut):
                    self._buffer = self._buffer[cut + 1 :]
                else:
                    self._buffer = self._buffer[cut:]
                if not tool_name:
                    self._reject_current = True
                elif valid_names and tool_name not in valid_names:
                    self._reject_current = True
                else:
                    self._current_tool_name = tool_name
                    self._delta(pending, name=tool_name)
                if self._reject_current:
                    if not self._discard_through_tool_end():
                        break
                continue

            if self._streaming_string_value:
                value_end = self._buffer.find(self._VALUE_END)
                if value_end >= 0:
                    fragment = self._escape_string_content(self._buffer[:value_end])
                    self._buffer = self._buffer[value_end + len(self._VALUE_END) :]
                    self._delta(pending, arguments=fragment + '"')
                    self._streaming_string_value = False
                    self._pending_key = None
                    continue
                if self._END in self._buffer:
                    self._reject_current = True
                    self._discard_through_tool_end()
                    continue
                emitted, self._buffer = self._hold_partial_suffix(
                    self._buffer, self._VALUE_END
                )
                if emitted:
                    self._delta(pending, arguments=self._escape_string_content(emitted))
                break

            if self._pending_key is not None:
                value_start = self._buffer.find(self._VALUE_START)
                if value_start < 0:
                    if self._END in self._buffer:
                        self._reject_current = True
                        self._discard_through_tool_end()
                        continue
                    break
                self._buffer = self._buffer[value_start + len(self._VALUE_START) :]
                key = self._pending_key.strip()
                prefix = self._argument_prefix(key)
                if prefix is None:
                    self._pending_key = None
                    continue
                string_names = self._string_argument_names(
                    request, self._current_tool_name
                )
                if key in string_names:
                    self._delta(pending, arguments=prefix + '"')
                    self._streaming_string_value = True
                    continue
                value_end = self._buffer.find(self._VALUE_END)
                if value_end < 0:
                    self._buffer = self._VALUE_START + self._buffer
                    break
                raw_value = self._buffer[:value_end].strip()
                self._buffer = self._buffer[value_end + len(self._VALUE_END) :]
                self._pending_key = None
                value = json.dumps(self._deserialize(raw_value), ensure_ascii=False)
                self._delta(pending, arguments=prefix + value)
                continue

            tool_end = self._buffer.find(self._END)
            key_start = self._buffer.find(self._KEY_START)
            if tool_end >= 0 and (key_start < 0 or tool_end < key_start):
                self._buffer = self._buffer[tool_end + len(self._END) :]
                self._delta(pending, arguments=self._close_arguments())
                self._finish_tool_call()
                continue
            if key_start < 0:
                break
            self._buffer = self._buffer[key_start + len(self._KEY_START) :]
            key_end = self._buffer.find(self._KEY_END)
            if key_end < 0:
                self._buffer = self._KEY_START + self._buffer
                break
            self._pending_key = self._buffer[:key_end]
            self._buffer = self._buffer[key_end + len(self._KEY_END) :]

        payload: dict[str, Any] = {}
        if content:
            payload["content"] = content
        if pending:
            payload["tool_calls"] = list(pending.values())
        return payload or None
