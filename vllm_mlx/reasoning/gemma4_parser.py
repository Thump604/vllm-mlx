# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 reasoning parser for vllm-mlx.

Handles Gemma 4's channel-based thinking format:
    <|channel>thought
    ...reasoning content...
    <channel|>
    Final answer content

The thinking channel is activated when <|think|> appears in the system prompt.
"""

import re

from .base import DeltaMessage, ReasoningParser

START_TOKEN = "<|channel>thought\n"
END_TOKEN = "<channel|>"
TURN_START_TOKEN = "<|turn>"
TURN_END_TOKEN = "<turn|>"
TOOL_CALL_START_TOKEN = "<|tool_call>"
TOOL_CALL_END_TOKEN = "<tool_call|>"
_START_TOKEN_VARIANTS = (
    START_TOKEN,
    "<|channel>thought",
    "<|channel>-thought\n",
    "<|channel>-thought",
)

_CONTROL_TOKENS = (
    START_TOKEN,
    "<|channel>thought",
    "<|channel>-thought\n",
    "<|channel>-thought",
    END_TOKEN,
    TURN_START_TOKEN,
    TURN_END_TOKEN,
    TOOL_CALL_START_TOKEN,
    TOOL_CALL_END_TOKEN,
    "<|think|>",
    "<|tool>",
    "<tool|>",
    "<|tool_response>",
    "<tool_response|>",
)
_CONTROL_TOKEN_RE = re.compile(
    "|".join(re.escape(token) for token in sorted(_CONTROL_TOKENS, key=len, reverse=True))
)
_CONTROL_LINE_RE = re.compile(r"(?m)^[ \t]*(?://)?thought[ \t]*$")
_INLINE_THOUGHT_SUFFIX_RE = re.compile(r"(?m)//thought[ \t]*$")
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_START_PREFIX_RE = re.compile(r"^<\|channel>\s*-?\s*thought(?:\n)?", re.IGNORECASE)


class Gemma4ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Gemma 4 models.

    Extracts reasoning from <|channel>thought ... <channel|> blocks.
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._in_reasoning = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        text = self._trim_partial_control_suffix(model_output or "")
        if not text:
            return None, None

        start_match = _START_PREFIX_RE.match(text)
        if start_match:
            return self._extract_reasoning_prefixed(text, start_match.end())

        if self._looks_like_partial_reasoning_prefix(text):
            return None, None

        # Gemma reasoning is only valid when the completion begins with the
        # thought channel. Later thought/channel fragments are control leakage.
        return None, self._extract_visible_content(text)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        del delta_text

        prev_reasoning, prev_content = self.extract_reasoning(previous_text)
        curr_reasoning, curr_content = self.extract_reasoning(current_text)

        reasoning_delta = self._suffix_delta(prev_reasoning, curr_reasoning)
        content_delta = self._suffix_delta(prev_content, curr_content)

        if reasoning_delta is None and content_delta is None:
            return None

        return DeltaMessage(
            reasoning=reasoning_delta,
            content=content_delta,
        )

    def reset_state(self):
        self._in_reasoning = False

    def _extract_reasoning_prefixed(
        self,
        text: str,
        start_len: int,
    ) -> tuple[str | None, str | None]:
        after_start = text[start_len:]
        if END_TOKEN not in after_start:
            reasoning = self._clean_reasoning_text(after_start)
            if reasoning:
                return reasoning, None
            return None, None

        reasoning_text, _, remaining = after_start.partition(END_TOKEN)
        reasoning = self._clean_reasoning_text(reasoning_text)
        content = self._extract_visible_content(remaining)
        return reasoning or None, content

    def _extract_visible_content(self, text: str) -> str | None:
        visible_text = self._truncate_at_first_marker(
            text,
            (
                TOOL_CALL_START_TOKEN,
                TOOL_CALL_END_TOKEN,
                TURN_START_TOKEN,
                TURN_END_TOKEN,
                START_TOKEN,
                END_TOKEN,
            ),
        )
        cleaned = self._clean_content_text(visible_text)
        if cleaned:
            return cleaned

        # Fall back to aggressive cleanup when the model leaks control markers
        # without a clean turn terminator, but still avoid surfacing the markers.
        cleaned = self._clean_content_text(text)
        return cleaned or None

    @staticmethod
    def _truncate_at_first_marker(text: str, markers: tuple[str, ...]) -> str:
        cut_positions = [idx for idx in (text.find(marker) for marker in markers) if idx >= 0]
        if not cut_positions:
            return text
        return text[: min(cut_positions)]

    @staticmethod
    def _clean_reasoning_text(text: str) -> str:
        text = _CONTROL_TOKEN_RE.sub("", text)
        text = _BLANK_LINE_RE.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def _clean_content_text(text: str) -> str:
        text = _INLINE_THOUGHT_SUFFIX_RE.sub("", text)
        text = _CONTROL_TOKEN_RE.sub("", text)
        text = _CONTROL_LINE_RE.sub("", text)
        text = _BLANK_LINE_RE.sub("\n\n", text)
        deduped_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line:
                if deduped_lines and deduped_lines[-1]:
                    deduped_lines.append("")
                continue
            if deduped_lines and deduped_lines[-1] == line:
                continue
            deduped_lines.append(line)
        return "\n".join(deduped_lines).strip()

    @staticmethod
    def _trim_partial_control_suffix(text: str) -> str:
        if not text:
            return text
        for token in _CONTROL_TOKENS:
            max_suffix = min(len(token) - 1, len(text))
            for suffix_len in range(max_suffix, 0, -1):
                suffix = text[-suffix_len:]
                if not suffix.startswith("<"):
                    continue
                if token.startswith(suffix) and suffix != token:
                    return text[:-suffix_len]
        return text

    @staticmethod
    def _looks_like_partial_reasoning_prefix(text: str) -> bool:
        if not text or not text.startswith("<"):
            return False
        return any(token.startswith(text) for token in _START_TOKEN_VARIANTS)

    @staticmethod
    def _suffix_delta(previous: str | None, current: str | None) -> str | None:
        if not current:
            return None
        if not previous:
            return current
        if current == previous:
            return None
        if current.startswith(previous):
            delta = current[len(previous):]
            return delta or None
        return None
