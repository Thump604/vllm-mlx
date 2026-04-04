# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from vllm_mlx.cooperative_specprefill import (
    ChunkedDraftScorer,
    PreseededSequenceStateMachine,
    RopeAdjustedCache,
)


def test_chunked_scorer_initial_state():
    """Scorer starts with correct initial state."""
    scorer = ChunkedDraftScorer(
        draft_model=MagicMock(),
        tokens=list(range(10000)),
        chunk_size=4096,
    )
    assert scorer.is_scoring
    assert not scorer.is_done
    assert scorer.chunks_remaining > 0
    assert scorer.tokens_processed == 0


def test_chunked_scorer_chunks_remaining():
    """Chunks remaining computed correctly."""
    scorer = ChunkedDraftScorer(
        draft_model=MagicMock(),
        tokens=list(range(10000)),
        chunk_size=4096,
    )
    # 10000 tokens, chunk_size=4096: (9999 processable) / 4096 = 3 chunks
    assert scorer.chunks_remaining >= 2


def test_chunked_scorer_small_prompt():
    """Prompt smaller than chunk_size: still works."""
    scorer = ChunkedDraftScorer(
        draft_model=MagicMock(),
        tokens=list(range(100)),
        chunk_size=4096,
    )
    assert scorer.chunks_remaining >= 1


def test_chunked_scorer_cleanup_frees_resources():
    """cleanup() frees cache and intermediates."""
    scorer = ChunkedDraftScorer(
        draft_model=MagicMock(),
        tokens=list(range(100)),
        chunk_size=50,
    )
    # Simulate partial state
    scorer._cache = MagicMock()
    scorer._logits = MagicMock()
    scorer.cleanup()
    assert scorer._cache is None
    assert scorer._logits is None


def test_chunked_scorer_finalize_before_done_raises():
    """finalize() raises if scoring not complete."""
    scorer = ChunkedDraftScorer(
        draft_model=MagicMock(),
        tokens=list(range(10000)),
        chunk_size=4096,
    )
    import pytest

    with pytest.raises(RuntimeError, match="Cannot finalize"):
        scorer.finalize()


def test_preseeded_state_machine_replays_seed_tokens():
    base = MagicMock()
    base.make_state.return_value = "start"
    base.match.side_effect = [
        ("seed-1", None, "normal"),
        ("seed-2", None, "normal"),
        ("after", [3], None),
    ]

    machine = PreseededSequenceStateMachine(base, [11, 22])

    state = machine.make_state()
    next_state, matched_sequence, current_state = machine.match(state, 33)

    assert state == "seed-2"
    assert next_state == "after"
    assert matched_sequence == [3]
    assert current_state is None


def test_rope_adjusted_cache_reports_adjusted_offset():
    base = MagicMock()
    base.offset = 7
    cache = RopeAdjustedCache(base, adjustment=5)

    assert cache.offset == 12
