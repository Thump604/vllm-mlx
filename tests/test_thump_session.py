from __future__ import annotations

import numpy as np

from mlx_lm.models.cache import KVCache

from vllm_mlx.thump.adapter import BlockGeometry, RopeConfig
from vllm_mlx.thump.capture import LayerCapture
from vllm_mlx.thump.session import (
    LayerSpec,
    SessionSubstrate,
    _deinterleave_split_rotary_pairs,
    _interleave_split_rotary_pairs,
)


class _DummyModel:
    def make_cache(self):
        return [KVCache()]


def test_session_substrate_splice_and_materialize_round_trip(tmp_path):
    geometry = BlockGeometry(
        block_size_tokens=16,
        num_kv_heads=2,
        head_dim=8,
        group_size=1,
        rope=RopeConfig(variant=0, theta=1.0, partial_rotary_factor=0.0),
    )
    spec = LayerSpec(layer_index=0, layer_type="full_attention", geometry=geometry)
    session = SessionSubstrate(
        [spec],
        block_capacity=8,
        root_dir=tmp_path,
    )

    initial = LayerCapture(
        keys=np.full((20, 2, 8), 1.0, dtype=np.float16),
        values=np.full((20, 2, 8), 10.0, dtype=np.float16),
    )
    session.initialize_from_capture({0: initial}, total_tokens=20)
    assert session.total_tokens == 20

    inserted = LayerCapture(
        keys=np.full((16, 2, 8), 2.0, dtype=np.float16),
        values=np.full((16, 2, 8), 20.0, dtype=np.float16),
    )
    session.splice_insert_from_capture(
        16,
        {0: inserted},
        insert_token_count=16,
    )
    assert session.total_tokens == 36

    caches = session.materialize_prompt_cache(_DummyModel(), upto_tokens=35)
    keys, values = caches[0].state
    values_np = np.asarray(values)
    assert values_np.shape == (1, 2, 35, 8)
    tokens_first = np.transpose(values_np[0], (1, 0, 2))
    assert np.allclose(tokens_first[:16], 10.0, atol=0.1)
    assert np.allclose(tokens_first[16:32], 20.0, atol=0.1)


def test_session_substrate_replace_equal_length_round_trip(tmp_path):
    geometry = BlockGeometry(
        block_size_tokens=16,
        num_kv_heads=2,
        head_dim=8,
        group_size=1,
        rope=RopeConfig(variant=0, theta=1.0, partial_rotary_factor=0.0),
    )
    spec = LayerSpec(layer_index=0, layer_type="full_attention", geometry=geometry)
    session = SessionSubstrate(
        [spec],
        block_capacity=12,
        root_dir=tmp_path,
    )

    initial = LayerCapture(
        keys=np.full((48, 2, 8), 1.0, dtype=np.float16),
        values=np.full((48, 2, 8), 10.0, dtype=np.float16),
    )
    session.initialize_from_capture({0: initial}, total_tokens=48)

    replacement = LayerCapture(
        keys=np.full((16, 2, 8), 3.0, dtype=np.float16),
        values=np.full((16, 2, 8), 30.0, dtype=np.float16),
    )
    session.replace_equal_length_from_capture(
        16,
        {0: replacement},
        replace_token_count=16,
    )

    caches = session.materialize_prompt_cache(_DummyModel(), upto_tokens=48)
    _keys, values = caches[0].state
    values_np = np.asarray(values)
    tokens_first = np.transpose(values_np[0], (1, 0, 2))
    assert np.allclose(tokens_first[:16], 10.0, atol=0.1)
    assert np.allclose(tokens_first[16:32], 30.0, atol=0.1)
    assert np.allclose(tokens_first[32:48], 10.0, atol=0.1)


def test_nontraditional_rope_layout_helpers_round_trip():
    keys = np.arange(16, dtype=np.float16).reshape(2, 1, 8)
    interleaved = _interleave_split_rotary_pairs(keys, 4)
    restored = _deinterleave_split_rotary_pairs(interleaved, 4)
    assert np.array_equal(restored, keys)
    assert np.array_equal(
        interleaved[0, 0],
        np.array([0, 2, 1, 3, 4, 5, 6, 7], dtype=np.float16),
    )
