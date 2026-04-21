# SPDX-License-Identifier: Apache-2.0
"""
Base parser for models using <think>...</think> tags for reasoning.

This module provides BaseThinkingReasoningParser, a concrete implementation
for extracting reasoning content from models that use thinking tags.

Supports three scenarios:
1. Both tags in output: <think>reasoning</think>content
2. Only closing tag (think injected in prompt): reasoning</think>content
3. No tags: pure content
"""

from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base parser for models using <think>...</think> style tags.

    This parser handles the common pattern where reasoning content is wrapped
    in special tags. Subclasses define the specific start and end tokens.

    Supports "implicit reasoning mode" where <think> is injected in the prompt
    and only </think> appears in the model output. This is common with AI agents
    like OpenCode that force models to reason by injecting thinking tags.

    The parser tracks state during streaming to correctly separate reasoning
    from content as tokens arrive incrementally.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token/tag that starts reasoning content (e.g., '<think>')."""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token/tag that ends reasoning content (e.g., '</think>')."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Streaming state.
        self._seen_start = False
        self._seen_end = False
        # Buffer for partial tag matches at chunk boundaries.
        self._boundary_buffer = ""

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Handles three cases:
        1. Both tags present: <think>reasoning</think>content
        2. Only closing tag: reasoning</think>content (think in prompt)
        3. No tags: pure content

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        text = model_output

        # Case 1: Both tags present (normal case)
        if self.start_token in text and self.end_token in text:
            # Get everything after start token
            _, _, after_start = text.partition(self.start_token)
            # Split on end token
            reasoning, _, content = after_start.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only closing tag (think was injected in prompt)
        # Everything before </think> is reasoning
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 3: Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4: No tags at all - pure content
        return None, model_output

    def reset_state(self):
        """Reset internal state for a new streaming request."""
        self._seen_start = False
        self._seen_end = False
        self._boundary_buffer = ""

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta using state-machine tracking.

        Uses internal state (_seen_start, _seen_end) to avoid O(n) searches
        on the full accumulated text every token. Only searches the small
        delta_text plus a boundary buffer for tag transitions.

        Args:
            previous_text: Text accumulated before this delta (used for
                           first-call bootstrap only).
            current_text: Text including this delta (unused in fast path).
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # Bootstrap: on first call, seed state from previous_text (covers
        # cases where the parser is attached mid-stream or tags were in
        # earlier chunks processed by a different code path).
        if not self._seen_start and not self._seen_end and previous_text:
            if self.end_token in previous_text:
                self._seen_end = True
            elif self.start_token in previous_text:
                self._seen_start = True

        self._boundary_buffer += delta_text
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        while self._boundary_buffer:
            if self._seen_end:
                content_parts.append(self._boundary_buffer)
                self._boundary_buffer = ""
                break

            if self._seen_start:
                end_idx = self._boundary_buffer.find(self.end_token)
                if end_idx != -1:
                    if end_idx:
                        reasoning_parts.append(self._boundary_buffer[:end_idx])
                    self._boundary_buffer = self._boundary_buffer[
                        end_idx + len(self.end_token) :
                    ]
                    self._seen_end = True
                    continue

                hold = self._partial_tag_suffix_len(
                    self._boundary_buffer, (self.end_token,)
                )
                emit = self._boundary_buffer[:-hold] if hold else self._boundary_buffer
                if emit:
                    reasoning_parts.append(emit)
                self._boundary_buffer = self._boundary_buffer[-hold:] if hold else ""
                break

            start_idx = self._boundary_buffer.find(self.start_token)
            end_idx = self._boundary_buffer.find(self.end_token)
            candidates = [idx for idx in (start_idx, end_idx) if idx != -1]

            if candidates:
                next_idx = min(candidates)
                if start_idx != -1 and start_idx == next_idx:
                    prefix = self._boundary_buffer[:start_idx]
                    if prefix:
                        reasoning_parts.append(prefix)
                    self._boundary_buffer = self._boundary_buffer[
                        start_idx + len(self.start_token) :
                    ]
                    self._seen_start = True
                    continue

                prefix = self._boundary_buffer[:end_idx]
                if prefix:
                    reasoning_parts.append(prefix)
                self._boundary_buffer = self._boundary_buffer[
                    end_idx + len(self.end_token) :
                ]
                self._seen_end = True
                continue

            hold = self._partial_tag_suffix_len(
                self._boundary_buffer, (self.start_token, self.end_token)
            )
            emit = self._boundary_buffer[:-hold] if hold else self._boundary_buffer
            if emit:
                reasoning_parts.append(emit)
            self._boundary_buffer = self._boundary_buffer[-hold:] if hold else ""
            break

        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)

    def finalize_stream(self) -> DeltaMessage | None:
        """Flush any buffered text that did not complete a tag."""
        if not self._boundary_buffer:
            return None

        pending = self._boundary_buffer
        self._boundary_buffer = ""
        if self._seen_end:
            return DeltaMessage(content=pending)
        return DeltaMessage(reasoning=pending)

    @staticmethod
    def _partial_tag_suffix_len(text: str, tags: tuple[str, ...]) -> int:
        """Return the longest suffix of text that could continue as a tag."""
        for length in range(len(text), 0, -1):
            suffix = text[-length:]
            if any(tag.startswith(suffix) for tag in tags):
                return length
        return 0
