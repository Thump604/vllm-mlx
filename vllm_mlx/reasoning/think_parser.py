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
        # Streaming state (avoids O(n^2) full-text search on every token)
        self._seen_start = False
        self._seen_end = False
        # Buffer for partial tag matches at chunk boundaries
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
            if self.start_token in previous_text:
                self._seen_start = True
            if self.end_token in previous_text:
                self._seen_end = True

        # Skip if delta is just the special tokens themselves
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            self._seen_start = True
            return None
        if stripped_delta == self.end_token:
            self._seen_end = True
            return None

        # Check for tags in delta (O(len(delta)) which is small)
        # Also check boundary: tag might span previous chunk + this delta
        boundary_check = self._boundary_buffer + delta_text
        start_in_delta = self.start_token in boundary_check
        end_in_delta = self.end_token in boundary_check

        # Update boundary buffer (keep last N chars where N = max tag length - 1)
        max_tag_len = max(len(self.start_token), len(self.end_token))
        self._boundary_buffer = delta_text[-(max_tag_len - 1):] if len(delta_text) >= max_tag_len - 1 else (self._boundary_buffer + delta_text)[-(max_tag_len - 1):]

        # Detect transitions from delta
        if start_in_delta:
            self._seen_start = True
        if end_in_delta:
            self._seen_end = True

        # Case 1: Explicit <think> found - standard behavior
        if self._seen_start:
            return self._handle_explicit_think(
                previous_text, delta_text,
                start_in_prev=(self._seen_start and not start_in_delta),
                end_in_prev=(self._seen_end and not end_in_delta),
                end_in_delta=end_in_delta,
            )

        # Case 2: No <think> but </think> found - implicit reasoning mode
        if self._seen_end:
            return self._handle_implicit_think(
                delta_text,
                end_in_prev=(self._seen_end and not end_in_delta),
                end_in_delta=end_in_delta,
            )

        # Case 3: No think tags seen yet - treat as reasoning
        return DeltaMessage(reasoning=delta_text)

    def _handle_explicit_think(
        self,
        previous_text: str,
        delta_text: str,
        start_in_prev: bool,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> tag is explicitly in the output."""
        start_in_delta = self.start_token in delta_text

        if start_in_prev:
            # We're after the start token
            if end_in_delta:
                # Transition: end token in this delta
                idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            elif end_in_prev:
                # Already past reasoning phase - pure content
                return DeltaMessage(content=delta_text)
            else:
                # Still in reasoning phase
                return DeltaMessage(reasoning=delta_text)

        elif start_in_delta:
            # Start token is in this delta
            start_idx = delta_text.find(self.start_token)

            if end_in_delta:
                # Both tokens in this delta
                end_idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[start_idx + len(self.start_token) : end_idx]
                content_part = delta_text[end_idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            else:
                # Only start token - beginning of reasoning
                reasoning_part = delta_text[start_idx + len(self.start_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None
                )

        # Fallback - treat as content
        return DeltaMessage(content=delta_text)

    def _handle_implicit_think(
        self,
        delta_text: str,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> was in prompt (only </think> in output)."""
        if end_in_delta:
            # Transition: end token in this delta
            idx = delta_text.find(self.end_token)
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len(self.end_token) :]
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )
        elif end_in_prev:
            # Already past reasoning phase - pure content
            return DeltaMessage(content=delta_text)
        else:
            # Still in implicit reasoning phase
            return DeltaMessage(reasoning=delta_text)
