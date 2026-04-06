# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Qwen3 models.

Qwen3 uses <think>...</think> tags for reasoning content and supports
a strict switch via 'enable_thinking=False' in chat template kwargs.

Supports implicit reasoning mode where <think> is injected in the prompt
by AI agents (e.g., OpenCode) and only </think> appears in the output.
"""

import re

from .think_parser import BaseThinkingReasoningParser

_PLAINTEXT_REASONING_RE = re.compile(
    r"^\s*(?:thinking process|reasoning|thought process)\s*:\s*",
    re.IGNORECASE,
)
_DRAFT_MARKER_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*Draft:\*\*|\*\*Draft\*\*\s*:|Draft:|Final Answer:|Answer:|Response:)\s*(.+)$"
)
_DRAFT_SECTION_RE = re.compile(
    r"(?ims)^\*Draft(?:\s+\d+)?:\*\s*(.+?)(?=^\*(?:Word Count Check|Constraints Check):\*|^\d+\.\s+\*\*|^\*\*[^*]+:\*\*|\Z)"
)
_LISTY_PARAGRAPH_RE = re.compile(r"^\s*(?:\d+\.\s|[-*]\s|\*\*[^*]+:\*\*)")
_RECOVERY_HINTS = (
    "the user wants me to",
    "actually, looking at the conversation history",
    "[moderator]",
    "*draft",
    "*word count check:*",
    "*constraints check:*",
    "<think",
    "</think>",
    "let's stop.",
)


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Qwen3 models.

    Qwen3 uses <think>...</think> tokens to denote reasoning text.

    Supports three scenarios:
    1. Both tags in output: <think>reasoning</think>content
    2. Only closing tag (think in prompt): reasoning</think>content
    3. No tags: pure content

    Example (normal):
        Input: "<think>Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."

    Example (think in prompt):
        Input: "Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Qwen3 output.

        Handles both explicit <think>...</think> tags and implicit mode
        where <think> was in the prompt (only </think> in output).

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        # If no end token at all, treat as pure content
        if self.end_token not in model_output:
            reasoning, content = self._extract_plaintext_reasoning(model_output, request=request)
            return self._recover_final_content(reasoning, content)

        # Use base class implementation (handles both explicit and implicit)
        reasoning, content = super().extract_reasoning(model_output)
        return self._recover_final_content(reasoning, content)

    def _extract_plaintext_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        """Split plaintext Qwen reasoning scaffolding from the visible answer."""
        match = _PLAINTEXT_REASONING_RE.match(model_output)
        if not match:
            return None, model_output

        body = model_output[match.end() :].strip()
        if not body:
            return None, None

        draft_match = _DRAFT_MARKER_RE.search(body)
        if draft_match:
            content = draft_match.group(1).strip() or None
            reasoning = body[: draft_match.start()].strip() or None
            return reasoning, content

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        content = None
        reasoning = body

        for idx in range(len(paragraphs) - 1, -1, -1):
            paragraph = paragraphs[idx]
            if _LISTY_PARAGRAPH_RE.match(paragraph):
                continue
            content = paragraph.strip() or None
            reasoning = "\n\n".join(paragraphs[:idx]).strip() or None
            break

        if content is None:
            kwargs = getattr(request, "chat_template_kwargs", None) or {}
            if kwargs.get("force_nonempty_content") is True:
                return None, body
            return body, None

        return reasoning, content

    def _recover_final_content(
        self,
        reasoning: str | None,
        content: str | None,
    ) -> tuple[str | None, str | None]:
        """Promote the final draft from reasoning when visible content is junk.

        If content was originally None or empty, the model intentionally
        emitted reasoning with no visible answer (e.g. ``<think>X</think>``).
        Respect that and do not promote reasoning to content.
        """
        if not content:
            return reasoning, content
        if not self._content_needs_recovery(content):
            return reasoning, content

        for text in (reasoning, content):
            recovered = self._extract_draft_content(text)
            if recovered:
                return reasoning, recovered

        return reasoning, content

    def _content_needs_recovery(self, content: str) -> bool:
        text = content.strip().lower()
        if not text:
            return True
        return any(hint in text for hint in _RECOVERY_HINTS)

    def _extract_draft_content(self, text: str | None) -> str | None:
        if not text:
            return None

        section_match = _DRAFT_SECTION_RE.search(text)
        if section_match:
            return self._normalize_candidate(section_match.group(1))

        marker_match = _DRAFT_MARKER_RE.search(text)
        if marker_match:
            return self._normalize_candidate(marker_match.group(1))

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for paragraph in reversed(paragraphs):
            candidate = self._normalize_candidate(paragraph)
            if candidate:
                return candidate
        return None

    def _normalize_candidate(self, text: str | None) -> str | None:
        if not text:
            return None
        lines = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if _LISTY_PARAGRAPH_RE.match(stripped):
                continue
            if lower.startswith("*word count check:*") or lower.startswith("*constraints check:*"):
                continue
            lines.append(stripped)

        if not lines:
            return None

        candidate = " ".join(lines).strip()
        if self._content_needs_recovery(candidate):
            return None
        return candidate
