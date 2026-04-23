# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RotatingKVCache handling in sparse_prefill."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class _FakeAttention:
    def __init__(self):
        self.num_heads = 1
        self.q_proj = lambda x: x


class _FakeLayer:
    def __init__(self):
        self.block_type = "*"
        self.mixer = _FakeAttention()


class _FakeModel:
    def __init__(self):
        self.layers = [_FakeLayer()]
        self.calls: list[list[int]] = []

    def __call__(self, x, cache=None):
        self.calls.append(x.tolist())
        logits = mx.zeros((1, x.shape[1], 8), dtype=mx.float32)
        return logits


class RotatingKVCache:
    def __init__(self, max_size: int, keep: int = 0):
        self.max_size = max_size
        self.keep = keep
        self.offset = 0
        self.state = mx.array([0], dtype=mx.float32)


def _run_sparse_prefill(total_tokens: int, selected_indices: list[int], max_size: int):
    from vllm_mlx.specprefill import sparse_prefill

    model = _FakeModel()
    tokens = list(range(total_tokens))
    cache = [RotatingKVCache(max_size=max_size, keep=0)]
    sparse_prefill(
        model,
        tokens,
        selected_indices,
        cache,
        step_size=64,
    )
    return model.calls


def test_sparse_prefill_does_not_expand_tail_when_prompt_fits_window():
    calls = _run_sparse_prefill(
        total_tokens=6,
        selected_indices=[0, 2, 4],
        max_size=8,
    )

    flattened = [token for chunk in calls for row in chunk for token in row]
    assert flattened == [0, 2, 4]


def test_sparse_prefill_expands_tail_when_prompt_exceeds_window():
    calls = _run_sparse_prefill(
        total_tokens=10,
        selected_indices=[0, 2],
        max_size=8,
    )

    flattened = [token for chunk in calls for row in chunk for token in row]
    assert flattened == [0, 2, 3, 4, 5, 6, 7, 8, 9]


def test_score_tokens_uses_original_cache_factory_when_available(monkeypatch):
    from vllm_mlx import specprefill

    model = object()
    original_factory = MagicMock(return_value=["original-cache"])
    patched_factory = MagicMock(
        side_effect=AssertionError("quantized cache factory should be bypassed")
    )

    monkeypatch.setattr(specprefill, "make_prompt_cache", patched_factory)
    monkeypatch.setattr(
        specprefill.cache_module,
        "_original_make_prompt_cache",
        original_factory,
        raising=False,
    )

    cache = specprefill._make_draft_prompt_cache(model)

    assert cache == ["original-cache"]
    original_factory.assert_called_once_with(model)


def test_score_tokens_falls_back_to_current_cache_factory(monkeypatch):
    from vllm_mlx import specprefill

    model = object()
    monkeypatch.delattr(
        specprefill.cache_module, "_original_make_prompt_cache", raising=False
    )
    patched_factory = MagicMock(return_value=["patched-cache"])
    monkeypatch.setattr(specprefill, "make_prompt_cache", patched_factory)

    cache = specprefill._make_draft_prompt_cache(model)

    assert cache == ["patched-cache"]
    patched_factory.assert_called_once_with(model)


def test_select_chunks_reserves_evenly_spaced_backbone_within_budget():
    from vllm_mlx.specprefill import select_chunks

    importance = mx.array(
        [10.0] * 32 + [0.1] * 32 + [0.1] * 32 + [9.0] * 32 + [0.1] * 32 + [8.0] * 32,
        dtype=mx.float32,
    )

    selected = select_chunks(
        importance,
        keep_pct=0.5,  # 3 of 6 chunks
        chunk_size=32,
        backbone_pct=1 / 6,  # reserve 1 chunk for global coverage
    ).tolist()

    selected_chunks = sorted({idx // 32 for idx in selected})
    assert len(selected_chunks) == 3
    assert 0 in selected_chunks
    assert 3 in selected_chunks
    assert 5 in selected_chunks


def test_select_chunks_defaults_to_top_scores_when_backbone_disabled():
    from vllm_mlx.specprefill import select_chunks

    importance = mx.array(
        [10.0] * 32 + [0.1] * 32 + [0.1] * 32 + [9.0] * 32 + [0.1] * 32 + [8.0] * 32,
        dtype=mx.float32,
    )

    selected = select_chunks(
        importance,
        keep_pct=0.5,
        chunk_size=32,
        backbone_pct=0.0,
    ).tolist()

    selected_chunks = sorted({idx // 32 for idx in selected})
    assert selected_chunks == [0, 3, 5]
