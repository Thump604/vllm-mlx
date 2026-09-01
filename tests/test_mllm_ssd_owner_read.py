# SPDX-License-Identifier: Apache-2.0
"""Owner-bound MLLM cold-tier read regressions."""

from pathlib import Path
import asyncio
import json
from types import SimpleNamespace
import time
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_mlx.cache_owner_identity import OwnerBindingDecision
from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, PrefillAbortedError
from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.ssd_cache import (
    SSDCacheConfig,
    SSDCacheTier,
    SSDReadResult,
    reconstruct_ssd_layers,
)

_TOKENS = [11, 22, 33]
_MATCHED = (11, 22)
_IDENTITY = {
    "model": "model-a",
    "tokenizer": "tokenizer-a",
    "cache_layout": "layout-a",
}


class _DType:
    def __init__(self, size):
        self.size = size


class _Array:
    """Numpy-backed MLX-shaped test array with dtype-size metadata."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = _DType(self._data.dtype.itemsize)

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


class KVCache:
    def __init__(self, *, offset=3):
        self.keys = _Array(np.zeros((1, 2, offset, 4), dtype=np.float16))
        self.values = _Array(np.ones((1, 2, offset, 4), dtype=np.float16))
        self.offset = offset


class RotatingKVCache(KVCache):
    def __init__(self, *, max_size=8, keep=2, offset=3):
        super().__init__(offset=offset)
        self.max_size = max_size
        self.keep = keep
        self.step = 2
        self._idx = offset


class ArraysCache:
    def __init__(self):
        self.state = [_Array(np.arange(6, dtype=np.float32).reshape(2, 3))]


def _generator():
    """Build the production fetch seam without constructing an MLX model."""
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    binding = object()
    generator.prefix_cache = MagicMock()
    generator.prefix_cache._persistence_identity = dict(_IDENTITY)
    generator.prefix_cache._ssd_tier = None
    generator.prefix_cache.fetch_owner_bound.return_value = (
        OwnerBindingDecision(True, "none"),
        None,
        list(_TOKENS),
    )
    generator.prefix_cache.validate_owner_request.return_value = OwnerBindingDecision(
        True, "none"
    )
    generator._cache_owner_required = True
    generator._cache_owner_requests = {"request-1": binding}
    generator._prefix_checkpoint_lock = threading.RLock()
    generator._aborted_request_ids = set()
    generator._owner_thread_id = threading.get_ident()

    tier = MagicMock()
    tier._stats = SimpleNamespace(
        ssd_hits=0,
        ssd_misses=0,
        reload_latency_sum=0.0,
        reload_bytes=0,
        promotion_failures=0,
    )
    tier._data_dir = "/tmp/nonexistent-mllm-ssd-test"
    generator._ssd_tier = tier
    tier.record_promotion_failure.side_effect = lambda: setattr(
        tier._stats, "promotion_failures", tier._stats.promotion_failures + 1
    )
    tier.record_promotion_success.side_effect = lambda _result: setattr(
        tier._stats, "ssd_hits", tier._stats.ssd_hits + 1
    )
    tier.validate_candidate.side_effect = lambda _tokens, candidate: (
        tuple(_MATCHED)
        if candidate.get("persistence_identity", _IDENTITY) == _IDENTITY
        and tuple(candidate.get("matched_key", _MATCHED)) == _MATCHED
        else None
    )
    generator._reconstruct_ssd_layers = MagicMock(return_value=["restored-cache"])
    return generator, binding, tier


def _candidate(**overrides):
    candidate = {
        "file_path": "entry-a",
        "memory_bytes": 32,
        "num_tokens": len(_MATCHED),
        "matched_tokens": len(_MATCHED),
        "matched_key": _MATCHED,
        "match_type": "prefix",
        "persistence_identity": dict(_IDENTITY),
    }
    candidate.update(overrides)
    return candidate


def _configure_hit(generator, tier, *, candidate=None, layers=None):
    generator.prefix_cache.check_ssd.return_value = candidate or _candidate()
    generator.prefix_cache.try_reserve_memory.return_value = True
    generator.prefix_cache.prepare_owner_bound_store.return_value = (
        OwnerBindingDecision(True, "none"),
        SimpleNamespace(tokens=_MATCHED),
    )
    generator.prefix_cache.commit_owner_bound_store.return_value = OwnerBindingDecision(
        True, "none"
    )
    tier.read_validated_entry.return_value = SSDReadResult(
        tokens=_MATCHED,
        file_path="entry-a",
        memory_bytes=32,
        layers=(
            [{"keys": "keys", "values": "values", "offset": 2}]
            if layers is None
            else layers
        ),
        read_bytes=5,
        latency_seconds=0.001,
    )


def _wait_for_spill(tier, expected=1):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if tier.get_stats()["spill_count"] >= expected:
            return
        time.sleep(0.01)
    raise AssertionError("SSD spill did not complete")


def _real_cache_with_ssd(tmp_path, *, identity=None):
    identity = identity or dict(_IDENTITY)
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(tmp_path / "ssd"),
            persistence_identity=identity,
        )
    )
    tier.start_writer()
    cache = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(max_memory_mb=1, max_entries=1, min_prefix_tokens=1),
    )
    cache.set_ssd_tier(tier)
    return cache, tier


def test_owner_bound_production_fetch_promotes_ssd_after_ram_miss(tmp_path: Path):
    generator, binding, tier = _generator()
    _configure_hit(generator, tier)
    entry_dir = tmp_path / "entry-a"
    entry_dir.mkdir()
    (entry_dir / "layer_0.safetensors").write_bytes(b"bytes")
    tier._data_dir = str(tmp_path)

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache == ["restored-cache"]
    assert remaining == [33]
    generator.prefix_cache.fetch_owner_bound.assert_called_once_with(binding, _TOKENS)
    generator.prefix_cache.check_ssd.assert_called_once_with(_TOKENS)
    tier.read_validated_entry.assert_called_once()
    generator.prefix_cache.prepare_owner_bound_store.assert_called_once_with(
        binding,
        list(_MATCHED),
        ["restored-cache"],
        auxiliary=None,
        persistence_eligible=False,
    )
    generator.prefix_cache.commit_owner_bound_store.assert_called_once()
    generator.prefix_cache.release_reserved_memory.assert_called_once_with(32)
    assert tier._stats.ssd_hits == 1
    tier.record_promotion_success.assert_called_once()


def test_owner_bound_fetch_prefers_longer_ssd_match_over_short_ram_prefix(
    tmp_path: Path,
):
    generator, binding, tier = _generator()
    generator.prefix_cache.fetch_owner_bound.return_value = (
        OwnerBindingDecision(True, "none"),
        ["short-ram-cache"],
        list(_TOKENS[1:]),
    )
    _configure_hit(generator, tier)
    entry_dir = tmp_path / "entry-a"
    entry_dir.mkdir()
    (entry_dir / "layer_0.safetensors").write_bytes(b"bytes")
    tier._data_dir = str(tmp_path)

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache == ["restored-cache"]
    assert remaining == [33]
    generator.prefix_cache.fetch_owner_bound.assert_called_once_with(binding, _TOKENS)
    generator.prefix_cache.check_ssd.assert_called_once_with(_TOKENS)
    tier.read_validated_entry.assert_called_once()
    tier.record_promotion_success.assert_called_once()


def test_owner_bound_fetch_keeps_longer_ram_prefix_without_reading_ssd():
    generator, binding, tier = _generator()
    generator.prefix_cache.fetch_owner_bound.return_value = (
        OwnerBindingDecision(True, "none"),
        ["longer-ram-cache"],
        [],
    )
    generator.prefix_cache.check_ssd.return_value = _candidate(matched_tokens=2)

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache == ["longer-ram-cache"]
    assert remaining == []
    generator.prefix_cache.fetch_owner_bound.assert_called_once_with(binding, _TOKENS)
    generator.prefix_cache.check_ssd.assert_called_once_with(_TOKENS)
    tier.read_validated_entry.assert_not_called()


def test_owner_bound_corrupt_ssd_candidate_is_a_miss_and_releases_reservation():
    generator, _binding, tier = _generator()
    _configure_hit(generator, tier, layers=None)

    def corrupt_read(*_args, **_kwargs):
        tier.record_promotion_failure()
        return None

    tier.read_validated_entry.side_effect = corrupt_read

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache is None
    assert remaining == _TOKENS
    generator.prefix_cache.prepare_owner_bound_store.assert_not_called()
    generator.prefix_cache.commit_owner_bound_store.assert_not_called()
    generator.prefix_cache.release_reserved_memory.assert_called_once_with(32)
    assert tier._stats.ssd_hits == 0
    assert tier._stats.promotion_failures == 1


def test_owner_bound_identity_mismatch_candidate_is_a_miss_without_read():
    generator, _binding, tier = _generator()
    _configure_hit(
        generator,
        tier,
        candidate=_candidate(
            persistence_identity={
                "model": "different-model",
                "tokenizer": "tokenizer-a",
                "cache_layout": "layout-a",
            }
        ),
    )

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache is None
    assert remaining == _TOKENS
    tier.read_validated_entry.assert_not_called()
    generator.prefix_cache.try_reserve_memory.assert_not_called()
    generator.prefix_cache.prepare_owner_bound_store.assert_not_called()
    assert tier._stats.promotion_failures == 1


def test_owner_bound_malformed_identity_candidate_is_a_miss_without_read():
    generator, _binding, tier = _generator()
    _configure_hit(generator, tier, candidate=_candidate(persistence_identity=None))

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache is None
    assert remaining == _TOKENS
    tier.read_validated_entry.assert_not_called()
    generator.prefix_cache.try_reserve_memory.assert_not_called()
    generator.prefix_cache.prepare_owner_bound_store.assert_not_called()
    assert tier._stats.promotion_failures == 1


def test_owner_bound_ssd_budget_denial_is_a_miss_without_read_or_leak():
    generator, _binding, tier = _generator()
    _configure_hit(generator, tier)
    generator.prefix_cache.try_reserve_memory.return_value = False

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache is None
    assert remaining == _TOKENS
    tier.read_validated_entry.assert_not_called()
    generator.prefix_cache.release_reserved_memory.assert_not_called()
    generator.prefix_cache.prepare_owner_bound_store.assert_not_called()
    assert tier._stats.promotion_failures == 1


def test_owner_bound_token_identity_mismatch_is_a_miss_without_read():
    generator, _binding, tier = _generator()
    _configure_hit(generator, tier, candidate=_candidate(matched_key=(11, 99)))

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=True
    )

    assert cache is None
    assert remaining == _TOKENS
    tier.read_validated_entry.assert_not_called()
    generator.prefix_cache.try_reserve_memory.assert_not_called()
    assert tier._stats.promotion_failures == 1


def test_owner_bound_media_lookup_does_not_promote_ssd():
    generator, _binding, tier = _generator()
    _configure_hit(generator, tier)

    cache, remaining = generator._fetch_prefix_cache(
        "request-1", _TOKENS, allow_ssd_promotion=False
    )

    assert cache is None
    assert remaining == _TOKENS
    generator.prefix_cache.check_ssd.assert_not_called()
    tier.read_validated_entry.assert_not_called()


def test_owner_bound_ssd_reconstruction_is_owner_thread_only():
    generator, _binding, _tier = _generator()
    generator._owner_thread_id = threading.get_ident() + 1

    with pytest.raises(RuntimeError, match="owner thread"):
        generator._fetch_prefix_cache("request-1", _TOKENS, allow_ssd_promotion=True)


def test_owner_bound_ssd_promotion_requires_owner_thread_marker():
    generator, _binding, _tier = _generator()
    del generator._owner_thread_id

    with pytest.raises(RuntimeError, match="owner thread"):
        generator._fetch_prefix_cache("request-1", _TOKENS, allow_ssd_promotion=True)


@pytest.mark.parametrize("decision_reason", ["cancellation", "cache_unsafe"])
def test_owner_bound_ssd_promotion_rechecks_request_lease_before_publish(
    decision_reason,
):
    generator, binding, tier = _generator()
    _configure_hit(generator, tier)
    generator.prefix_cache.validate_owner_request.side_effect = [
        OwnerBindingDecision(True, "none"),
        OwnerBindingDecision(False, decision_reason),
    ]

    if decision_reason == "cancellation":
        with pytest.raises(PrefillAbortedError):
            generator._fetch_prefix_cache(
                "request-1", _TOKENS, allow_ssd_promotion=True
            )
        cache, remaining = None, _TOKENS
    else:
        cache, remaining = generator._fetch_prefix_cache(
            "request-1", _TOKENS, allow_ssd_promotion=True
        )

    assert cache is None
    assert remaining == _TOKENS
    tier.read_validated_entry.assert_called_once()
    generator.prefix_cache.prepare_owner_bound_store.assert_not_called()
    generator.prefix_cache.commit_owner_bound_store.assert_not_called()
    generator.prefix_cache.release_reserved_memory.assert_called_once_with(32)
    assert tier._stats.ssd_hits == 0


def test_owner_bound_ssd_promotion_cancellation_before_io_is_fail_closed():
    generator, _binding, tier = _generator()
    generator.prefix_cache.validate_owner_request.return_value = OwnerBindingDecision(
        False, "cancellation"
    )

    with pytest.raises(PrefillAbortedError):
        generator._fetch_prefix_cache("request-1", _TOKENS, allow_ssd_promotion=True)

    tier.validate_candidate.assert_not_called()
    generator.prefix_cache.try_reserve_memory.assert_not_called()
    generator.prefix_cache.release_reserved_memory.assert_not_called()
    assert tier._stats.promotion_failures == 1


def test_ssd_default_promotion_gate_is_fail_closed_for_media():
    generator, _binding, tier = _generator()

    cache, remaining = generator._fetch_prefix_cache("request-1", _TOKENS)

    assert cache is None
    assert remaining == _TOKENS
    generator.prefix_cache.check_ssd.assert_not_called()
    tier.read_validated_entry.assert_not_called()


def test_real_memory_cache_ssd_round_trip_persists_identity_and_stats(tmp_path):
    tokens = (1, 2, 3)
    cache, tier = _real_cache_with_ssd(tmp_path)
    try:
        assert cache.store(list(tokens), [KVCache()])
        assert cache.store([4, 5, 6], [KVCache()])
        _wait_for_spill(tier)

        assert cache.fetch(list(tokens))[0] is None
        candidate = cache.check_ssd(list(tokens))
        assert candidate is not None
        assert candidate["persistence_identity"] == _IDENTITY
        result = tier.read_validated_entry(tokens, candidate)
        assert result is not None
        assert result.layers[0]["layer_type"] == "KVCache"
        tier.record_promotion_success(result)

        stats = tier.get_stats()
        assert stats["spill_count"] == 1
        assert stats["ssd_hits"] == 1
        assert stats["reload_bytes"] > 0
        assert stats["reload_latency_sum_s"] >= 0
        assert cache.get_stats()["ssd"] == stats
    finally:
        tier.close()


def test_ssd_real_round_trip_preserves_rotating_and_hybrid_layer_types(tmp_path):
    tokens = (7, 8, 9)
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(tmp_path / "typed"),
            persistence_identity=dict(_IDENTITY),
        )
    )
    tier.start_writer()
    try:
        hybrid = ArraysCache()
        hybrid.left_padding = _Array([0])
        hybrid.lengths = _Array([3])
        vanilla = KVCache()
        assert tier.enqueue_spill(
            tokens,
            [RotatingKVCache(), vanilla, hybrid],
            memory_bytes=256,
        )
        _wait_for_spill(tier)
        candidate = tier.lookup_candidate(tokens)
        result = tier.read_validated_entry(tokens, candidate)
        assert result is not None
        assert [layer["layer_type"] for layer in result.layers] == [
            "RotatingKVCache",
            "KVCache",
            "ArraysCache",
        ]
        assert result.layers[0]["max_size"] == 8
        assert result.layers[0]["keep"] == 2
        assert result.layers[0]["step"] == 2
        assert result.layers[0]["_idx"] == 3
        assert result.layers[2]["metadata_arrays"] == ["left_padding", "lengths"]

        # Reconstruction remains on the caller's (owner) thread and retains
        # the rotating implementation and hybrid state container.
        try:
            from mlx_lm.models.cache import (
                ArraysCache as _MLXArraysCache,
                RotatingKVCache as _MLXRotatingKVCache,
            )
        except ImportError:
            pytest.skip("installed mlx-lm is unavailable in this CPU test lane")
        assert _MLXArraysCache is not None
        assert _MLXRotatingKVCache is not None
        reconstructed = reconstruct_ssd_layers(result.layers)
        assert reconstructed is not None
        assert type(reconstructed[0]).__name__ == "RotatingKVCache"
        assert type(reconstructed[1]).__name__ == "KVCache"
        assert type(reconstructed[2]).__name__ == "ArraysCache"
        assert reconstructed[0].offset == 3
        assert reconstructed[0].max_size == 8
        np.testing.assert_array_equal(
            np.asarray(reconstructed[1].keys), np.asarray(vanilla.keys)
        )
        np.testing.assert_array_equal(
            np.asarray(reconstructed[1].values), np.asarray(vanilla.values)
        )
        assert reconstructed[1].offset == vanilla.offset
        np.testing.assert_array_equal(np.asarray(reconstructed[2].left_padding), [0])
        np.testing.assert_array_equal(np.asarray(reconstructed[2].lengths), [3])
    finally:
        tier.close()


def test_reconstruct_vanilla_kv_uses_governed_no_kwargs_constructor(monkeypatch):
    try:
        from mlx_lm.models import cache as cache_module
    except ImportError:
        pytest.skip("installed mlx-lm is unavailable in this CPU test lane")

    class GovernedVanillaKVCache:
        instances = 0

        def __init__(self):
            type(self).instances += 1
            self.keys = None
            self.values = None
            self.offset = 0

    monkeypatch.setattr(cache_module, "KVCache", GovernedVanillaKVCache)
    keys = np.zeros((1, 2, 3, 4), dtype=np.float16)
    values = np.ones((1, 2, 3, 4), dtype=np.float16)
    reconstructed = reconstruct_ssd_layers(
        [
            {
                "keys": keys,
                "values": values,
                "offset": 3,
                "layer_type": "KVCache",
                "keys_shape": list(keys.shape),
                "values_shape": list(values.shape),
                "keys_dtype": "float16",
                "values_dtype": "float16",
            }
        ]
    )

    assert reconstructed is not None
    assert len(reconstructed) == 1
    assert isinstance(reconstructed[0], GovernedVanillaKVCache)
    assert GovernedVanillaKVCache.instances == 1
    np.testing.assert_array_equal(np.asarray(reconstructed[0].keys), keys)
    np.testing.assert_array_equal(np.asarray(reconstructed[0].values), values)
    assert reconstructed[0].offset == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest_truncated",
        "layer_truncated",
        "layer_count",
        "reordered",
        "shape",
        "dtype",
        "offset",
    ],
)
def test_ssd_malformed_entry_is_quarantined_before_publication(tmp_path, mutation):
    tokens = (10, 11, 12)
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(tmp_path / mutation),
            persistence_identity=dict(_IDENTITY),
        )
    )
    tier.start_writer()
    try:
        layers = [KVCache(), KVCache()] if mutation == "reordered" else [KVCache()]
        assert tier.enqueue_spill(tokens, layers, memory_bytes=256)
        _wait_for_spill(tier)
        candidate = tier.lookup_candidate(tokens)
        assert candidate is not None
        entry_dir = Path(tier._data_dir) / candidate["file_path"]
        manifest_path = entry_dir / "manifest.json"
        if mutation == "manifest_truncated":
            manifest_path.write_text("{\n")
        elif mutation == "layer_truncated":
            (entry_dir / "layer_0.safetensors").write_bytes(b"broken")
        else:
            manifest = json.loads(manifest_path.read_text())
            if mutation == "layer_count":
                manifest["num_layers"] = 1
                manifest["layers"] = []
            elif mutation == "reordered":
                manifest["layers"].reverse()
            elif mutation == "shape":
                manifest["layers"][0]["keys_shape"][0] += 1
            elif mutation == "dtype":
                manifest["layers"][0]["keys_dtype"] = "float32"
            elif mutation == "offset":
                manifest["layers"][0]["offset"] = 999
            manifest_path.write_text(json.dumps(manifest))

        assert tier.read_validated_entry(tokens, candidate) is None
        assert tier._index.lookup_exact(tokens) is None
        assert (Path(tier._cache_dir) / "quarantine" / candidate["file_path"]).exists()
        assert tier.get_stats()["ssd_hits"] == 0
    finally:
        tier.close()


@pytest.mark.parametrize(
    "identity", [None, {"model": "wrong", "tokenizer": "t", "cache_layout": "l"}]
)
def test_ssd_missing_or_mismatched_identity_is_rejected_without_quarantine(
    tmp_path, identity
):
    tokens = (20, 21, 22)
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(tmp_path / ("missing" if identity is None else "wrong")),
            persistence_identity=dict(_IDENTITY),
        )
    )
    tier.start_writer()
    try:
        assert tier.enqueue_spill(tokens, [KVCache()], memory_bytes=256)
        _wait_for_spill(tier)
        original = tier.lookup_candidate(tokens)
        assert original is not None
        tier._index.insert_entry(
            tokens,
            original["file_path"],
            original["memory_bytes"],
            len(tokens),
            persistence_identity=identity,
        )
        candidate = tier.lookup_candidate(tokens)
        assert candidate is not None
        assert tier.validate_candidate(tokens, candidate) is None
        assert tier.read_validated_entry(tokens, candidate) is None
        assert (Path(tier._data_dir) / candidate["file_path"]).exists()
        assert tier._index.lookup_exact(tokens) is not None
        assert tier.get_stats()["ssd_hits"] == 0
        assert tier.get_stats()["promotion_failures"] == 1
    finally:
        tier.close()


@pytest.mark.parametrize(
    "manifest_identity",
    [None, {"model": "wrong", "tokenizer": "tokenizer-a", "cache_layout": "layout-a"}],
)
def test_ssd_manifest_identity_corruption_is_quarantined(tmp_path, manifest_identity):
    tokens = (23, 24, 25)
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(
                tmp_path / ("missing" if manifest_identity is None else "wrong")
            ),
            persistence_identity=dict(_IDENTITY),
        )
    )
    tier.start_writer()
    try:
        assert tier.enqueue_spill(tokens, [KVCache()], memory_bytes=256)
        _wait_for_spill(tier)
        candidate = tier.lookup_candidate(tokens)
        assert candidate is not None
        manifest_path = Path(tier._data_dir) / candidate["file_path"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest_identity is None:
            manifest.pop("persistence_identity")
        else:
            manifest["persistence_identity"] = manifest_identity
        manifest_path.write_text(json.dumps(manifest))

        assert tier.read_validated_entry(tokens, candidate) is None
        assert tier._index.lookup_exact(tokens) is None
        assert (Path(tier._cache_dir) / "quarantine" / candidate["file_path"]).exists()
        assert tier.get_stats()["ssd_hits"] == 0
        assert tier.get_stats()["promotion_failures"] == 1
    finally:
        tier.close()


def test_ssd_path_traversal_candidate_is_rejected_without_read(tmp_path):
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(tmp_path / "path"),
            persistence_identity=dict(_IDENTITY),
        )
    )
    try:
        candidate = {
            "file_path": "../escape",
            "memory_bytes": 1,
            "num_tokens": 1,
            "matched_tokens": 1,
            "matched_key": (1,),
        }
        assert tier.validate_candidate((1,), candidate) is None
        assert tier.read_validated_entry((1,), candidate) is None
        assert not (Path(tier._data_dir).parent / "escape").exists()
        assert tier.get_stats()["promotion_failures"] == 1
    finally:
        tier.close()


def test_ssd_async_promotion_cancellation_before_io_balances_reservation(
    tmp_path, monkeypatch
):
    tokens = (30, 31, 32)
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(tmp_path / "cancel-before"),
            persistence_identity=dict(_IDENTITY),
        )
    )
    tier.start_writer()
    try:
        assert tier.enqueue_spill(tokens, [KVCache()], memory_bytes=256)
        _wait_for_spill(tier)
        candidate = tier.lookup_candidate(tokens)
        assert candidate is not None

        reserved = []
        released = []
        to_thread_entered = asyncio.Event()
        allow_read = asyncio.Event()
        original_to_thread = asyncio.to_thread

        async def delay_io(function, *args, **kwargs):
            to_thread_entered.set()
            await allow_read.wait()
            return await original_to_thread(function, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", delay_io)

        async def cancel_before_worker_io():
            task = asyncio.create_task(
                tier.async_promote(
                    tokens,
                    lambda nbytes: reserved.append(nbytes) or True,
                    released.append,
                )
            )
            await asyncio.wait_for(to_thread_entered.wait(), timeout=2)
            task.cancel()
            allow_read.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_before_worker_io())
        assert reserved == [candidate["memory_bytes"]]
        assert released == [candidate["memory_bytes"]]
        assert tier.get_stats()["ssd_hits"] == 0
    finally:
        tier.close()


def test_ssd_async_promotion_cancellation_after_io_balances_reservation(tmp_path):
    tokens = (33, 34, 35)
    tier = SSDCacheTier(
        SSDCacheConfig(
            cache_dir=str(tmp_path / "cancel-after"),
            persistence_identity=dict(_IDENTITY),
        )
    )
    tier.start_writer()
    try:
        assert tier.enqueue_spill(tokens, [KVCache()], memory_bytes=256)
        _wait_for_spill(tier)
        candidate = tier.lookup_candidate(tokens)
        assert candidate is not None
        original_read = tier.read_validated_entry
        read_started = threading.Event()
        allow_read = threading.Event()

        def block_read(read_tokens, read_candidate):
            read_started.set()
            assert allow_read.wait(timeout=2)
            return original_read(read_tokens, read_candidate)

        tier.read_validated_entry = block_read
        reserved = []
        released = []

        async def cancel_after_worker_io():
            task = asyncio.create_task(
                tier.async_promote(
                    tokens,
                    lambda nbytes: reserved.append(nbytes) or True,
                    released.append,
                )
            )
            for _ in range(200):
                if read_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert read_started.is_set()
            task.cancel()
            allow_read.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_after_worker_io())
        assert reserved == [candidate["memory_bytes"]]
        assert released == [candidate["memory_bytes"]]
        assert tier.get_stats()["ssd_hits"] == 0
    finally:
        tier.close()


def test_standard_async_promotion_store_failure_is_not_an_ssd_hit():
    try:
        from vllm_mlx.scheduler import Scheduler
    except ImportError:
        pytest.skip("installed mlx-lm is unavailable in this CPU test lane")

    class FakeTier:
        def __init__(self):
            self.read_calls = []
            self.failure_count = 0
            self.success_count = 0

        async def async_promote(self, *args, **kwargs):
            self.read_calls.append((args, kwargs))
            assert args[1](32)
            return SSDReadResult(
                tokens=(1, 2),
                file_path="entry",
                memory_bytes=32,
                layers=[{"serialized": True}],
                read_bytes=8,
                latency_seconds=0.001,
            )

        def record_promotion_failure(self):
            self.failure_count += 1

        def record_promotion_success(self, _result):
            self.success_count += 1

    class RejectingMemoryCache:
        def __init__(self):
            self.reserved = []
            self.released = []
            self.store_calls = []

        def try_reserve_memory(self, nbytes):
            self.reserved.append(nbytes)
            return True

        def release_reserved_memory(self, nbytes):
            self.released.append(nbytes)

        def store(self, *args, **kwargs):
            self.store_calls.append((args, kwargs))
            return False

    tier = FakeTier()
    memory_cache = RejectingMemoryCache()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._ssd_tier = tier
    scheduler.memory_aware_cache = memory_cache
    scheduler._reconstruct_ssd_layers = lambda _layers: ["restored"]
    request = SimpleNamespace(
        request_id="standard-async-store-failure",
        prompt_token_ids=[1, 2, 3],
        _ssd_candidate={
            "matched_key": (1, 2),
            "matched_tokens": 2,
            "memory_bytes": 32,
            "match_type": "prefix",
        },
    )

    assert asyncio.run(scheduler.promote_from_ssd(request)) is False
    assert tier.read_calls[0][1] == {
        "record_success": False,
        "return_result": True,
    }
    assert memory_cache.reserved == [32]
    assert memory_cache.released == [32]
    assert len(memory_cache.store_calls) == 1
    assert tier.failure_count == 1
    assert tier.success_count == 0
    assert request.cache_hit_type == "miss"
