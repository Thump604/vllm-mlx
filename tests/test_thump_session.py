from __future__ import annotations

import numpy as np

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from vllm_mlx.thump.adapter import BlockGeometry, RopeConfig
from vllm_mlx.thump.capture import LayerCapture
from vllm_mlx.thump.session import (
    LayerSpec,
    SessionSubstrate,
    _deinterleave_split_rotary_pairs,
    _interleave_split_rotary_pairs,
    _restore_rotating_cache_layout,
    _rotating_cache_next_index,
)
from vllm_mlx.thump.recovery import (
    CheckpointArtifact,
    RecoveryRunResult,
    SessionRecoveryTrace,
    _build_sampler,
    _decode_tokens,
    _encode_prompt_tokens,
    build_recovery_comparison,
)


class _DummyModel:
    def make_cache(self):
        return [KVCache()]


class _SlidingDummyModel:
    def make_cache(self):
        return [RotatingKVCache(max_size=4, keep=0)]


class _TokenizerWithBos:
    bos_token = "<bos>"

    def __init__(self):
        self.calls = []

    def encode(self, prompt_text, add_special_tokens=True):
        self.calls.append((prompt_text, add_special_tokens))
        return [101, 102]


class _TokenizerWithoutSpecialArg:
    bos_token = "<bos>"

    def encode(self, prompt_text):
        return [201, 202]


class _DecodeTokenizer:
    eos_token_id = None

    def decode(self, tokens):
        return ",".join(str(token) for token in tokens)


class _DecodeModel:
    def __call__(self, prompt, cache=None):
        return mx.zeros((1, 1, 4), dtype=mx.float32)


def test_encode_prompt_tokens_matches_engine_special_token_policy():
    tokenizer = _TokenizerWithBos()
    tokens = _encode_prompt_tokens(tokenizer, "hello")
    assert tokens == [101, 102]
    assert tokenizer.calls == [("hello", True)]

    tokenizer = _TokenizerWithBos()
    tokens = _encode_prompt_tokens(tokenizer, "<bos>hello")
    assert tokens == [101, 102]
    assert tokenizer.calls == [("<bos>hello", False)]


def test_encode_prompt_tokens_falls_back_when_tokenizer_lacks_special_arg():
    tokenizer = _TokenizerWithoutSpecialArg()
    assert _encode_prompt_tokens(tokenizer, "hello") == [201, 202]


def test_build_sampler_uses_argmax_at_zero_temperature():
    sampler = _build_sampler(temperature=0.0, top_p=0.95, top_k=64, min_p=0.0)
    logits = mx.array([[1.0, 3.0, 2.0]], dtype=mx.float32)
    token = sampler(logits)
    assert int(token.item()) == 1


def test_decode_tokens_uses_supplied_sampler_and_sampling_seed(monkeypatch):
    seeds = []

    def fake_seed(value):
        seeds.append(value)

    monkeypatch.setattr(mx.random, "seed", fake_seed)

    def fake_sampler(_logits):
        return mx.array([2], dtype=mx.int32)

    continuation_ms, output_text, output_tokens = _decode_tokens(
        _DecodeModel(),
        _DecodeTokenizer(),
        [],
        1,
        max_new_tokens=2,
        sampler=fake_sampler,
        sampling_seed=123,
    )

    assert continuation_ms >= 0.0
    assert seeds == [123]
    assert output_tokens == [2, 2]
    assert output_text == "2,2"


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


def test_session_substrate_insert_then_replace_round_trip(tmp_path):
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
        keys=np.full((32, 2, 8), 1.0, dtype=np.float16),
        values=np.full((32, 2, 8), 10.0, dtype=np.float16),
    )
    session.initialize_from_capture({0: initial}, total_tokens=32)

    inserted = LayerCapture(
        keys=np.full((16, 2, 8), 2.0, dtype=np.float16),
        values=np.full((16, 2, 8), 20.0, dtype=np.float16),
    )
    session.splice_insert_from_capture(
        16,
        {0: inserted},
        insert_token_count=16,
    )

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


def test_session_substrate_checkpoint_and_attach_round_trip(tmp_path):
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
        root_dir=tmp_path / "live",
    )

    initial = LayerCapture(
        keys=np.full((32, 2, 8), 1.0, dtype=np.float16),
        values=np.full((32, 2, 8), 10.0, dtype=np.float16),
    )
    session.initialize_from_capture({0: initial}, total_tokens=32)
    checkpoint = session.checkpoint(
        tmp_path / "live" / "session.tsmf",
        model_id_hash=0x1234,
        session_id=0x5678,
        sequence_id=0x9ABC,
        prompt_tokens=24,
        generated_tokens=9,
    )
    session.close()

    restored, restored_checkpoint = SessionSubstrate.attach_from_manifest(
        [spec],
        checkpoint.manifest_path,
        expected_model_id_hash=0x1234,
    )
    assert restored_checkpoint.sequence_id == 0x9ABC
    assert restored.total_tokens == 32

    caches = restored.materialize_prompt_cache(_DummyModel(), upto_tokens=32)
    _keys, values = caches[0].state
    values_np = np.asarray(values)
    tokens_first = np.transpose(values_np[0], (1, 0, 2))
    assert np.allclose(tokens_first[:32], 10.0, atol=0.1)
    restored.close()


def test_session_substrate_initialize_from_live_kv_cache_round_trip(tmp_path):
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
        exact_hot_restart=True,
    )

    cache = KVCache()
    values = np.arange(20, dtype=np.float16).reshape(20, 1, 1)
    values = np.broadcast_to(values, (20, 2, 8)).copy()
    keys = values + np.float16(100)
    cache.keys = mx.array(np.transpose(keys[None, ...], (0, 2, 1, 3)), dtype=mx.float16)
    cache.values = mx.array(
        np.transpose(values[None, ...], (0, 2, 1, 3)), dtype=mx.float16
    )
    cache.offset = 20

    session.initialize_from_live_cache([cache], total_tokens=20)
    restored = session.materialize_prompt_cache(_DummyModel(), upto_tokens=20)[0]
    restored_keys, restored_values = restored.state
    restored_value_bits = np.transpose(
        np.array(restored_values.view(mx.uint16))[0], (1, 0, 2)
    )
    restored_key_bits = np.transpose(
        np.array(restored_keys.view(mx.uint16))[0], (1, 0, 2)
    )
    expected_value_bits = np.transpose(np.array(cache.values.view(mx.uint16))[0], (1, 0, 2))
    expected_key_bits = np.transpose(np.array(cache.keys.view(mx.uint16))[0], (1, 0, 2))
    assert np.array_equal(restored_value_bits, expected_value_bits)
    assert np.array_equal(restored_key_bits, expected_key_bits)


def test_session_substrate_initialize_from_live_bf16_kv_cache_round_trip(tmp_path):
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
        exact_hot_restart=True,
    )

    cache = KVCache()
    values = np.arange(20, dtype=np.float32).reshape(20, 1, 1)
    values = np.broadcast_to(values, (20, 2, 8)).copy()
    keys = values + np.float32(100)
    cache.keys = mx.array(
        np.transpose(keys[None, ...], (0, 2, 1, 3)), dtype=mx.bfloat16
    )
    cache.values = mx.array(
        np.transpose(values[None, ...], (0, 2, 1, 3)), dtype=mx.bfloat16
    )
    cache.offset = 20

    session.initialize_from_live_cache([cache], total_tokens=20)
    restored = session.materialize_prompt_cache(_DummyModel(), upto_tokens=20)[0]
    restored_keys, restored_values = restored.state
    restored_value_bits = np.transpose(
        np.array(restored_values.view(mx.uint16))[0], (1, 0, 2)
    )
    restored_key_bits = np.transpose(
        np.array(restored_keys.view(mx.uint16))[0], (1, 0, 2)
    )
    expected_value_bits = np.transpose(
        np.array(cache.values.view(mx.uint16))[0], (1, 0, 2)
    )
    expected_key_bits = np.transpose(
        np.array(cache.keys.view(mx.uint16))[0], (1, 0, 2)
    )
    assert np.array_equal(restored_value_bits, expected_value_bits)
    assert np.array_equal(restored_key_bits, expected_key_bits)


def test_session_substrate_initialize_from_live_rotating_cache_round_trip(tmp_path):
    geometry = BlockGeometry(
        block_size_tokens=16,
        num_kv_heads=2,
        head_dim=8,
        group_size=1,
        rope=RopeConfig(variant=0, theta=1.0, partial_rotary_factor=0.0),
    )
    spec = LayerSpec(
        layer_index=0,
        layer_type="sliding_attention",
        geometry=geometry,
        window_size=4,
    )
    session = SessionSubstrate(
        [spec],
        block_capacity=8,
        root_dir=tmp_path,
        exact_hot_restart=True,
    )

    cache = RotatingKVCache(max_size=4, keep=0)
    for token in range(20):
        value = mx.full((1, 2, 1, 8), float(token), dtype=mx.float16)
        key = value + np.float16(100)
        cache.update_and_fetch(key, value)

    session.initialize_from_live_cache([cache], total_tokens=20)
    restored = session.materialize_prompt_cache(_SlidingDummyModel(), upto_tokens=20)[0]
    restored_values = np.array(restored._temporal_order(restored.values).astype(mx.float32))
    restored_keys = np.array(restored._temporal_order(restored.keys).astype(mx.float32))
    restored_values = np.transpose(restored_values[0], (1, 0, 2))[..., 0]
    restored_keys = np.transpose(restored_keys[0], (1, 0, 2))[..., 0]
    expected_values = np.array(
        [[16.0, 16.0], [17.0, 17.0], [18.0, 18.0], [19.0, 19.0]],
        dtype=np.float16,
    )
    expected_keys = expected_values + np.float16(100)
    assert np.allclose(restored_values, expected_values, atol=0.1)
    assert np.allclose(restored_keys, expected_keys, atol=0.1)


def test_session_substrate_initialize_from_live_bf16_rotating_cache_round_trip(tmp_path):
    geometry = BlockGeometry(
        block_size_tokens=16,
        num_kv_heads=2,
        head_dim=8,
        group_size=1,
        rope=RopeConfig(variant=0, theta=1.0, partial_rotary_factor=0.0),
    )
    spec = LayerSpec(
        layer_index=0,
        layer_type="sliding_attention",
        geometry=geometry,
        window_size=4,
    )
    session = SessionSubstrate(
        [spec],
        block_capacity=8,
        root_dir=tmp_path,
        exact_hot_restart=True,
    )

    cache = RotatingKVCache(max_size=4, keep=0)
    for token in range(20):
        value = mx.full((1, 2, 1, 8), float(token), dtype=mx.bfloat16)
        key = value + np.float32(100)
        cache.update_and_fetch(key, value)

    session.initialize_from_live_cache([cache], total_tokens=20)
    restored = session.materialize_prompt_cache(_SlidingDummyModel(), upto_tokens=20)[0]
    restored_values = np.array(restored._temporal_order(restored.values).astype(mx.float32))
    restored_keys = np.array(restored._temporal_order(restored.keys).astype(mx.float32))
    restored_values = np.transpose(restored_values[0], (1, 0, 2))[..., 0]
    restored_keys = np.transpose(restored_keys[0], (1, 0, 2))[..., 0]
    expected_values = np.array(
        [[16.0, 16.0], [17.0, 17.0], [18.0, 18.0], [19.0, 19.0]],
        dtype=np.float32,
    )
    expected_keys = expected_values + np.float32(100)
    assert np.allclose(restored_values, expected_values, atol=0.1)
    assert np.allclose(restored_keys, expected_keys, atol=0.1)


def test_nontraditional_rope_layout_helpers_round_trip():
    keys = np.arange(16, dtype=np.float16).reshape(2, 1, 8)
    interleaved = _interleave_split_rotary_pairs(keys, 4)
    restored = _deinterleave_split_rotary_pairs(interleaved, 4)
    assert np.array_equal(restored, keys)
    assert np.array_equal(
        interleaved[0, 0],
        np.array([0, 2, 1, 3, 4, 5, 6, 7], dtype=np.float16),
    )


def test_restore_rotating_cache_layout_round_trip():
    temporal = np.arange(4, dtype=np.float16).reshape(1, 1, 4, 1)
    restored, idx = _restore_rotating_cache_layout(
        temporal,
        offset=6,
        max_size=4,
        keep=0,
    )
    assert idx == _rotating_cache_next_index(6, 4, 0) == 2
    cache = RotatingKVCache(max_size=4, keep=0)
    cache.keys = mx.array(restored, dtype=mx.float16)
    cache.values = mx.array(restored, dtype=mx.float16)
    cache.offset = 6
    cache._idx = idx
    reordered = np.asarray(cache._temporal_order(cache.values))
    assert np.array_equal(reordered, temporal)

    next_token = mx.array([[[[9.0]]]], dtype=mx.float16)
    cache.update_and_fetch(next_token, next_token)
    after = np.asarray(cache._temporal_order(cache.values))
    expected = np.array([[[[1.0], [2.0], [3.0], [9.0]]]], dtype=np.float16)
    assert np.array_equal(after, expected)


def test_session_substrate_restores_rotating_cache_internal_state(tmp_path):
    geometry = BlockGeometry(
        block_size_tokens=16,
        num_kv_heads=2,
        head_dim=8,
        group_size=1,
        rope=RopeConfig(variant=0, theta=1.0, partial_rotary_factor=0.0),
    )
    spec = LayerSpec(
        layer_index=0,
        layer_type="sliding_attention",
        geometry=geometry,
        window_size=4,
    )
    session = SessionSubstrate([spec], block_capacity=8, root_dir=tmp_path)
    values = np.arange(20, dtype=np.float16).reshape(20, 1, 1)
    values = np.broadcast_to(values, (20, 2, 8)).copy()
    capture = LayerCapture(keys=values.copy(), values=values.copy())
    session.initialize_from_capture({0: capture}, total_tokens=20)

    cache = session.materialize_prompt_cache(_SlidingDummyModel(), upto_tokens=20)[0]
    assert cache.offset == 20
    assert cache._idx == 4
    restored = np.asarray(cache._temporal_order(cache.values))
    restored_tokens = np.transpose(restored[0], (1, 0, 2))[..., 0]
    expected = np.array([[16.0, 16.0], [17.0, 17.0], [18.0, 18.0], [19.0, 19.0]], dtype=np.float16)
    assert np.allclose(restored_tokens, expected, atol=0.75)


def test_build_recovery_comparison_emits_first_class_telemetry():
    trace = SessionRecoveryTrace(name="demo", prompt_text="hello")
    artifact = CheckpointArtifact(
        trace_name="demo",
        model_path="/tmp/model",
        manifest_path="/tmp/session.tsmf",
        root_dir="/tmp",
        prompt_tokens=[1, 2, 3],
        seed_tokens=[4, 5],
        prompt_token_count=3,
        generated_tokens=2,
        context_tokens=4,
        artifact_size_bytes=1234,
        model_id_hash=1,
        session_id=2,
        sequence_id=3,
        block_size_tokens=1,
        prefill_step_size=128,
        capture_step_size=128,
        checkpoint_latency_ms=12.5,
        checkpoint_rss_telemetry={
            "rss_before_prefill_bytes": 100,
            "rss_after_capture_merge_bytes": 250,
            "checkpoint_peak_rss_bytes": 250,
        },
    )
    restore = RecoveryRunResult(
        variant="restore",
        validate_latency_ms=4.0,
        validate_materialize_latency_ms=9.0,
        cold_rebuild_latency_ms=0.0,
        continuation_latency_ms=2.0,
        output_text="ok",
        output_tokens=[7, 8],
        fallback_count=0,
        fallback_reason=None,
        session_total_ms=20.0,
    )
    cold = RecoveryRunResult(
        variant="cold_rebuild",
        validate_latency_ms=0.0,
        validate_materialize_latency_ms=0.0,
        cold_rebuild_latency_ms=30.0,
        continuation_latency_ms=2.5,
        output_text="ok",
        output_tokens=[7, 8],
        fallback_count=0,
        fallback_reason=None,
        session_total_ms=40.0,
    )

    comparison = build_recovery_comparison(trace, artifact, restore=restore, cold_rebuild=cold)

    assert comparison.go_no_go == "GO"
    assert comparison.telemetry["checkpoint_latency_ms"] == 12.5
    assert comparison.telemetry["checkpoint_rss_telemetry"]["rss_before_prefill_bytes"] == 100
    assert comparison.telemetry["checkpoint_peak_rss_bytes"] == 250
    assert comparison.telemetry["restore_mode"] == "restore"
    assert comparison.telemetry["artifact_size_bytes"] == 1234
    assert comparison.telemetry["exact_fidelity"] is True
    assert comparison.telemetry["fallback_count"] == 0
