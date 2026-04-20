"""Tests for ThinkingAwareLogitsProcessor and supporting utilities."""

from typing import Callable

import mlx.core as mx
import pytest

from vllm_mlx.constrained.thinking_processor import (
    BoundedSuffixMatcher,
    Phase,
    ThinkingAwareLogitsProcessor,
)


class TestBoundedSuffixMatcher:
    def test_single_token_match(self):
        m = BoundedSuffixMatcher([42])
        assert m.feed(42) is True

    def test_single_token_no_match(self):
        m = BoundedSuffixMatcher([42])
        assert m.feed(99) is False

    def test_multi_token_sequence(self):
        m = BoundedSuffixMatcher([10, 20, 30])
        assert m.feed(10) is False
        assert m.feed(20) is False
        assert m.feed(30) is True

    def test_partial_match_then_mismatch_resets(self):
        m = BoundedSuffixMatcher([10, 20, 30])
        assert m.feed(10) is False
        assert m.feed(20) is False
        assert m.feed(99) is False  # mismatch
        # Must restart; next 10,20,30 should still match
        assert m.feed(10) is False
        assert m.feed(20) is False
        assert m.feed(30) is True

    def test_overlapping_prefix_not_missed(self):
        # Target: [1, 1, 2]. Stream: 1, 1, 1, 2.
        # A naive reset-to-0 matcher would miss this.
        m = BoundedSuffixMatcher([1, 1, 2])
        assert m.feed(1) is False
        assert m.feed(1) is False  # partial: [1, 1]
        assert m.feed(1) is False  # buf=[1, 1, 1] != [1, 1, 2]
        assert m.feed(2) is True  # buf=[1, 1, 2] == target

    def test_back_to_back_matches(self):
        m = BoundedSuffixMatcher([5, 6])
        assert m.feed(5) is False
        assert m.feed(6) is True
        # Second match immediately
        assert m.feed(5) is False
        assert m.feed(6) is True

    def test_empty_target_raises(self):
        with pytest.raises(ValueError):
            BoundedSuffixMatcher([])


# --- ThinkingAwareLogitsProcessor tests ---

# Use small token IDs for test markers.
# <think> = [10, 11], </think> = [20, 21]
_START_IDS = [10, 11]
_END_IDS = [20, 21]
_VOCAB_SIZE = 100


def _make_processor(
    budget: int = 1000,
    inner: Callable | None = None,
    start_ids: list[int] = _START_IDS,
    end_ids: list[int] = _END_IDS,
) -> ThinkingAwareLogitsProcessor:
    return ThinkingAwareLogitsProcessor(
        start_token_ids=start_ids,
        end_token_ids=end_ids,
        thinking_token_budget=budget,
        inner=inner,
        vocab_size=_VOCAB_SIZE,
    )


def _uniform_logits() -> mx.array:
    return mx.zeros((_VOCAB_SIZE,))


def _feed_sequence(
    proc: ThinkingAwareLogitsProcessor,
    token_ids: list[int],
) -> list[mx.array]:
    """Feed a sequence of token IDs one at a time, returning logits after each."""
    results = []
    tokens_so_far = []
    for tid in token_ids:
        tokens_so_far.append(tid)
        logits = proc(mx.array(tokens_so_far), _uniform_logits())
        results.append(logits)
    return results


class TestPhaseTransitions:
    def test_starts_in_idle(self):
        proc = _make_processor()
        assert proc.state == Phase.IDLE

    def test_idle_to_thinking_on_start_sequence(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11])  # <think>
        assert proc.state == Phase.THINKING

    def test_thinking_to_content_on_natural_end(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11, 50, 51, 20, 21])  # <think> tokens </think>
        assert proc.state == Phase.CONTENT

    def test_content_is_terminal_no_reentry(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11, 50, 20, 21])  # reach CONTENT
        assert proc.state == Phase.CONTENT
        _feed_sequence(proc, [10, 11])  # emit start markers again
        assert proc.state == Phase.CONTENT  # no re-entry

    def test_idle_passes_logits_through(self):
        proc = _make_processor()
        logits = _uniform_logits()
        tokens = mx.array([99])
        result = proc(tokens, logits)
        # In IDLE, logits are unchanged
        assert mx.array_equal(result, logits)

    def test_thinking_passes_logits_through(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11])  # enter THINKING
        logits = _uniform_logits()
        result = proc(mx.array([10, 11, 50]), logits)
        assert mx.array_equal(result, logits)

    def test_thinking_tokens_counted(self):
        proc = _make_processor(budget=100)
        _feed_sequence(proc, [10, 11, 50, 51, 52])
        # 3 thinking tokens (50, 51, 52), start sequence not counted
        assert proc.thinking_tokens == 3
