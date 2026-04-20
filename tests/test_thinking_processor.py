"""Tests for ThinkingAwareLogitsProcessor and supporting utilities."""

import pytest

from vllm_mlx.constrained.thinking_processor import BoundedSuffixMatcher


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
