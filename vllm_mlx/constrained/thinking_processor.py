"""Thinking-aware logits processor for reasoning models.

Manages the full thinking lifecycle: budget enforcement, phase transitions,
and content-phase constrained decoding delegation.
"""

from __future__ import annotations

from collections import deque


class BoundedSuffixMatcher:
    """Detect a target token sequence in a stream using a rolling suffix buffer.

    Unlike a naive sequential matcher that resets to position 0 on mismatch,
    this uses a bounded buffer that catches overlapping prefixes.
    """

    __slots__ = ("target", "_buf", "_max_len")

    def __init__(self, target_ids: list[int]) -> None:
        if not target_ids:
            raise ValueError("target_ids must be non-empty")
        self.target = tuple(target_ids)
        self._max_len = len(target_ids)
        self._buf: deque[int] = deque(maxlen=self._max_len)

    def feed(self, token_id: int) -> bool:
        """Feed one token. Returns True when the buffer suffix equals the target."""
        self._buf.append(token_id)
        return len(self._buf) == self._max_len and tuple(self._buf) == self.target

    def reset(self) -> None:
        """Clear the buffer."""
        self._buf.clear()
