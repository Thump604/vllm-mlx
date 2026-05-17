# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the default-off DFlash backend boundary."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

QWEN35_DFLASH_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "block_size": 16,
    "dflash_config": {
        "mask_token_id": 248070,
        "target_layer_ids": [1, 10, 19, 28, 37],
    },
    "hidden_size": 2048,
    "num_hidden_layers": 8,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000000,
    "max_position_embeddings": 262144,
    "num_target_layers": 40,
    "layer_types": ["full_attention"] * 8,
    "sliding_window": None,
    "use_sliding_window": False,
}


class FakeTokenizer:
    vocab_size = 248320

    def __len__(self):
        return self.vocab_size


class FakeQwenAddedTokenTokenizer:
    vocab_size = 248044

    def __len__(self):
        return 248077


class FakeLayer:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, x):
        return x + self.value


class FakeTarget:
    def __init__(self, layer_count: int = 40):
        self.model = SimpleNamespace(
            layers=[FakeLayer(float(i)) for i in range(layer_count)]
        )


def test_qwen35_dflash_contract_accepts_published_config():
    """The 35B path is driven by the draft config, not inferred from hooks."""
    from vllm_mlx.dflash import DFlashDraftConfig

    cfg = DFlashDraftConfig.from_dict(QWEN35_DFLASH_CONFIG)

    cfg.validate_qwen35_a3b_contract(
        target_model=FakeTarget(),
        target_tokenizer=FakeTokenizer(),
    )

    assert cfg.target_layer_ids == (1, 10, 19, 28, 37)
    assert cfg.num_target_layers == 40
    assert cfg.mask_token_id == 248070
    assert cfg.block_size == 16
    assert cfg.sliding_window is None
    assert cfg.use_sliding_window is False


def test_qwen35_dflash_contract_uses_effective_tokenizer_length():
    """Qwen added special-token ids can be above base tokenizer.vocab_size."""
    from vllm_mlx.dflash import DFlashDraftConfig

    cfg = DFlashDraftConfig.from_dict(QWEN35_DFLASH_CONFIG)

    cfg.validate_qwen35_a3b_contract(
        target_model=FakeTarget(),
        target_tokenizer=FakeQwenAddedTokenTokenizer(),
    )


def test_qwen35_dflash_contract_rejects_layer_mismatch():
    """A target/draft layer mismatch must fail closed with no substitution."""
    from vllm_mlx.dflash import DFlashCompatibilityError, DFlashDraftConfig

    cfg = DFlashDraftConfig.from_dict({**QWEN35_DFLASH_CONFIG, "num_target_layers": 39})

    with pytest.raises(DFlashCompatibilityError, match="num_target_layers"):
        cfg.validate_qwen35_a3b_contract(
            target_model=FakeTarget(),
            target_tokenizer=FakeTokenizer(),
        )


def test_qwen35_dflash_contract_rejects_swa_before_adapter_wiring():
    """The first Qwen 35B milestone must reject SWA drafts before routing."""
    from vllm_mlx.dflash import DFlashCompatibilityError, DFlashDraftConfig

    cfg = DFlashDraftConfig.from_dict(
        {
            **QWEN35_DFLASH_CONFIG,
            "sliding_window": 4096,
            "use_sliding_window": True,
        }
    )

    with pytest.raises(DFlashCompatibilityError, match="sliding_window"):
        cfg.validate_qwen35_a3b_contract(
            target_model=FakeTarget(),
            target_tokenizer=FakeTokenizer(),
        )


def test_layer_capture_restores_target_layers_after_exception():
    """Request-local capture must always restore the target model."""
    from vllm_mlx.dflash import DFlashLayerCapture

    target = FakeTarget()
    original_layers = list(target.model.layers)

    with pytest.raises(RuntimeError, match="boom"):
        with DFlashLayerCapture(target, (1, 10)):
            assert target.model.layers[1] is not original_layers[1]
            raise RuntimeError("boom")

    assert target.model.layers == original_layers


def test_layer_capture_concatenates_configured_hidden_states_and_clears():
    """Captured hidden states are per-forward and explicitly clearable."""
    from vllm_mlx.dflash import DFlashLayerCapture

    target = FakeTarget()
    x = mx.zeros((1, 2, 3), dtype=mx.float32)

    with DFlashLayerCapture(target, (1, 10)) as capture:
        target.model.layers[1](x)
        target.model.layers[10](x)
        hidden = capture.concat_hidden_states()

        assert hidden.shape == (1, 2, 6)
        assert hidden[:, :, :3].tolist() == mx.full((1, 2, 3), 1.0).tolist()
        assert hidden[:, :, 3:].tolist() == mx.full((1, 2, 3), 10.0).tolist()

        capture.clear()
        with pytest.raises(RuntimeError, match="missing captured hidden states"):
            capture.concat_hidden_states()


def test_dflash_can_snapshot_non_trimmable_array_cache():
    """Hybrid Qwen caches can rollback by restore/replay when trim is unavailable."""
    from mlx_lm.models.cache import ArraysCache

    from vllm_mlx.dflash import (
        _can_trim_prompt_cache,
        _restore_cache_state,
        _snapshot_cache_state,
    )

    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 2, 3), dtype=mx.float32)

    assert _can_trim_prompt_cache([cache]) is False

    snapshot = _snapshot_cache_state([cache])
    cache[0] = mx.zeros((1, 2, 3), dtype=mx.float32)

    _restore_cache_state([cache], snapshot)

    assert cache[0].tolist() == mx.ones((1, 2, 3), dtype=mx.float32).tolist()


def test_trim_recent_cache_rolls_back_rejected_tokens():
    """Verification rollback trims only the rejected recent cache entries."""
    from vllm_mlx.dflash import _trim_recent_cache

    class TrimmableCache:
        def __init__(self):
            self.offset = 9
            self.trimmed = []

        def trim(self, n):
            self.trimmed.append(n)
            self.offset -= n

    cache = TrimmableCache()

    _trim_recent_cache([cache], 3)

    assert cache.trimmed == [3]
    assert cache.offset == 6


def test_dflash_stats_include_block_acceptance_telemetry():
    """DFlash telemetry must expose whether acceptance is useful by block."""
    from vllm_mlx.dflash import DFlashSpeculativeDecoder

    draft = SimpleNamespace(config=SimpleNamespace(block_size=16))
    decoder = DFlashSpeculativeDecoder(draft, draft_model_name="draft-path")

    decoder._record_block(draft_count=15, accepted_count=3)
    decoder._record_block(draft_count=15, accepted_count=0)

    stats = decoder.snapshot_stats()

    assert stats["blocks"] == 2
    assert stats["avg_accepted_per_block"] == 1.5
    assert stats["acceptance_by_block"] == [
        {"draft_tokens": 15, "accepted_tokens": 3, "acceptance_rate": 0.2},
        {"draft_tokens": 15, "accepted_tokens": 0, "acceptance_rate": 0.0},
    ]
