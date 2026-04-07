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


def test_session_routes_target_chunk_size_to_sparse_prefiller():
    """``CooperativeSpecPrefillSession`` must support a separate
    ``target_chunk_size`` so callers can keep the draft scoring at a
    large chunk (the small draft model handles big forwards easily)
    while constraining the sparse-prefill chunk on the target model
    to keep individual Metal command buffers under the macOS GPU
    watchdog at large cumulative cache sizes (Session 84 Fix 2)."""
    from vllm_mlx.cooperative_specprefill import CooperativeSpecPrefillSession
    import vllm_mlx.cooperative_specprefill as csp

    # Stub the underlying primitives so we can read what step_size
    # the prefiller was constructed with without running real models.
    captured = {}

    class _FakeScorer:
        def __init__(self, **kwargs):
            captured["scorer_kwargs"] = kwargs
            self._steps = 0

        @property
        def is_done(self):
            return self._steps >= 1

        def step(self):
            self._steps += 1
            return self._steps >= 1

        def finalize(self):
            import mlx.core as mx

            return mx.zeros((4,), dtype=mx.float32)

        def cleanup(self):
            pass

    class _FakePrefiller:
        def __init__(self, model, tokens, selected_indices, cache, *, step_size, position_offset):
            captured["prefiller_step_size"] = step_size
            captured["prefiller_position_offset"] = position_offset
            self._done = False

        @property
        def is_done(self):
            return self._done

        @property
        def selected_token_count(self):
            return 0

        @property
        def cache_token_count(self):
            return 0

        def step(self):
            self._done = True
            return True

        def finalize(self):
            return None, []

        def cleanup(self):
            pass

    import unittest.mock

    with (
        unittest.mock.patch.object(csp, "ChunkedDraftScorer", _FakeScorer),
        unittest.mock.patch.object(csp, "ChunkedSparsePrefiller", _FakePrefiller),
        unittest.mock.patch.object(
            csp, "select_chunks", lambda importance, keep_pct: [0, 1, 2, 3]
        ),
    ):
        session = CooperativeSpecPrefillSession(
            model=MagicMock(),
            draft_model=MagicMock(),
            tokens=list(range(16)),
            base_cache=None,
            position_offset=0,
            keep_pct=0.5,
            chunk_size=2048,
            target_chunk_size=512,
        )
        # Drive to completion so the prefiller is constructed.
        for _ in range(8):
            if session.step():
                break

    assert captured["scorer_kwargs"]["chunk_size"] == 2048
    assert captured["prefiller_step_size"] == 512


def test_session_target_chunk_size_defaults_to_chunk_size():
    """If ``target_chunk_size`` is not supplied, the existing
    ``chunk_size`` value must be used for both phases. This pins the
    backwards-compatible default for the existing call sites
    (TextBatchScheduler) that have not been updated."""
    from vllm_mlx.cooperative_specprefill import CooperativeSpecPrefillSession

    session = CooperativeSpecPrefillSession(
        model=MagicMock(),
        draft_model=MagicMock(),
        tokens=list(range(16)),
        base_cache=None,
        position_offset=0,
        keep_pct=0.5,
        chunk_size=1024,
    )
    assert session._target_chunk_size == 1024
