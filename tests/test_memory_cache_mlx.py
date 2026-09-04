# SPDX-License-Identifier: Apache-2.0
"""MLX-dependent regression tests for the LCP trim contamination fix (#384).

These tests use real ``mlx_lm.models.cache.KVCache`` / ``RotatingKVCache``
objects backed by ``mlx.core`` arrays.  They run only on the apple-silicon
CI matrix (and locally on M-series hardware); the Linux ``test-matrix``
job excludes this file because MLX has no Linux distribution.
"""

from unittest.mock import MagicMock

import pytest


class TestTrimCacheOffset:
    """Tests for ``_trim_cache_offset``, focused on the LCP contamination fix.

    Regression: when the LCP fetch path trimmed a cache entry by shrinking
    the offset while still sharing the underlying (oversized) key/value
    arrays, downstream attention layers that read ``cache.state`` directly
    (e.g. Gemma 4 KV-shared layers) could see stale tokens from the previous
    owner of the entry.  See issue #384.  The fix slices the arrays down to
    new_offset so no memory beyond the new boundary remains accessible.
    """

    def test_plain_kv_cache_array_sliced_to_new_offset(self):
        """Plain-KVCache-like layer: after trim, keys.shape[-2] == new_offset."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        # Pretend a previous request wrote 500 tokens worth of data
        layer.keys = mx.arange(1 * 4 * 500 * 8, dtype=mx.float32).reshape(1, 4, 500, 8)
        layer.values = mx.arange(1 * 4 * 500 * 8, dtype=mx.float32).reshape(
            1, 4, 500, 8
        )
        layer.offset = 500

        # New request shares only the first 60 tokens as prefix
        trim_by = 500 - 60
        trimmed = _trim_cache_offset([layer], trim_by)
        tc = trimmed[0]

        assert tc.offset == 60
        # The underlying array MUST be shrunk, not just the offset pointer.
        # Otherwise Gemma 4's cache.state-reading layers would see positions
        # 60..500 filled with the previous request's tokens.
        assert tc.keys.shape[-2] == 60
        assert tc.values.shape[-2] == 60

    def test_plain_kv_cache_no_stale_tokens_visible_via_state(self):
        """A layer that reads the full cache.state must not see tokens past new_offset."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        # Positions 0..60: shared prefix (same for everyone).  Positions 60..500:
        # private content from a previous request that must NOT leak.
        shared = mx.ones((1, 4, 60, 8), dtype=mx.float32)
        private = mx.full((1, 4, 440, 8), 7.0, dtype=mx.float32)
        layer.keys = mx.concatenate([shared, private], axis=2)
        layer.values = layer.keys
        layer.offset = 500

        tc = _trim_cache_offset([layer], 500 - 60)[0]

        # cache.state is what KV-shared layers read directly.
        keys_view, _ = tc.state
        assert keys_view.shape[-2] == 60
        # No "7.0" tokens anywhere — private content was excluded.
        assert float(mx.max(keys_view).item()) == 1.0

    def test_plain_kv_cache_no_trim_preserves_array(self):
        """If trim_by == 0 or offset already equals shape, array is untouched."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 100, 8), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 100, 8), dtype=mx.float32)
        layer.offset = 100

        tc = _trim_cache_offset([layer], 0)[0]

        assert tc.offset == 100
        assert tc.keys.shape[-2] == 100

    def test_plain_kv_cache_trim_by_exceeds_offset_clamps_to_zero(self):
        """trim_by larger than offset yields an empty-but-valid trimmed cache."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 80, 8), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 80, 8), dtype=mx.float32)
        layer.offset = 80

        tc = _trim_cache_offset([layer], 1000)[0]

        assert tc.offset == 0
        assert tc.keys.shape[-2] == 0
        assert tc.values.shape[-2] == 0

    def test_plain_kv_cache_stored_entry_unaffected_after_trim(self):
        """Calling _trim_cache_offset must not mutate the source layer in place.

        The stored prefix-cache entry is the source here; a later lookup for
        a different request should get the same pristine data.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        full = mx.arange(1 * 2 * 200 * 4, dtype=mx.float32).reshape(1, 2, 200, 4)
        layer.keys = full
        layer.values = full
        layer.offset = 200
        original_shape = layer.keys.shape

        _trim_cache_offset([layer], 150)

        # Source entry keeps its full shape and offset.
        assert layer.keys.shape == original_shape
        assert layer.values.shape == original_shape
        assert layer.offset == 200

    def test_plain_kv_cache_in_place_write_does_not_corrupt_source(self):
        """After trim, writing through the returned cache must not leak into
        the stored entry.  This is the direct semantics of the fix: the stored
        prefix-cache entry has to survive concurrent use by other requests.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        # Source stored entry: positions 0..300 holding 5.0.
        layer = KVCache()
        layer.keys = mx.full((1, 2, 300, 4), 5.0, dtype=mx.float32)
        layer.values = mx.full((1, 2, 300, 4), 5.0, dtype=mx.float32)
        layer.offset = 300

        # New request shares first 50 tokens.
        tc = _trim_cache_offset([layer], 300 - 50)[0]

        # The trimmed cache now only has 50 tokens.  Writing new tokens via
        # update_and_fetch allocates a new array (because prev + N > current
        # shape) and does not touch the source.
        new_keys = mx.zeros((1, 2, 10, 4), dtype=mx.float32)
        new_values = mx.zeros((1, 2, 10, 4), dtype=mx.float32)
        tc.update_and_fetch(new_keys, new_values)

        # Source remains untouched (all 5.0 values preserved across full range).
        assert layer.keys.shape[-2] == 300
        assert float(mx.min(layer.keys).item()) == 5.0
        assert float(mx.max(layer.keys).item()) == 5.0

    def test_plain_kv_cache_multiple_layers_all_sliced(self):
        """Caches with several KVCache layers: every layer gets sliced."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layers = []
        for _ in range(5):
            layer = KVCache()
            layer.keys = mx.ones((1, 4, 200, 8), dtype=mx.float32)
            layer.values = mx.ones((1, 4, 200, 8), dtype=mx.float32)
            layer.offset = 200
            layers.append(layer)

        trimmed = _trim_cache_offset(layers, 150)

        assert len(trimmed) == 5
        for tc in trimmed:
            assert tc.offset == 50
            assert tc.keys.shape[-2] == 50
            assert tc.values.shape[-2] == 50

    def test_plain_kv_cache_slice_works_for_float16_and_bfloat16(self):
        """Fix must be dtype-agnostic so quantized / mixed-precision KV caches
        receive the same treatment as fp32.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        for dtype in (mx.float16, mx.bfloat16):
            layer = KVCache()
            layer.keys = mx.ones((1, 2, 120, 4), dtype=dtype)
            layer.values = mx.ones((1, 2, 120, 4), dtype=dtype)
            layer.offset = 120

            tc = _trim_cache_offset([layer], 80)[0]

            assert tc.offset == 40, f"dtype={dtype}"
            assert tc.keys.shape[-2] == 40, f"dtype={dtype}"
            assert tc.keys.dtype == dtype, f"dtype={dtype}"

    def test_plain_kv_cache_rotating_layers_unchanged_behavior(self):
        """RotatingKVCache was already trimming correctly before this fix.
        The plain-KVCache branch is the only one that changed; the rotating
        branch is exercised here to catch regressions.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = RotatingKVCache(max_size=128, keep=0)
        # Layer already rotated once: offset=200, buffer holds max_size entries.
        layer.keys = mx.ones((1, 4, 128, 8), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 128, 8), dtype=mx.float32)
        layer.offset = 200
        layer._idx = 128

        tc = _trim_cache_offset([layer], 100)[0]

        # Offset dropped by trim_by, clamped at >= 0.
        assert tc.offset == 100
        # Rotating path materialises a buffer whose shape matches new_offset
        # (padding with zeros if needed).  It must not come back as None.
        assert tc.keys is not None
        assert tc.values is not None
        # Dtype preserved through trim.
        assert tc.keys.dtype == mx.float32
        # Type-specific attrs preserved.
        assert hasattr(tc, "max_size")
        assert tc.max_size == 128

    def test_fetch_returns_sliced_cache_on_lcp_match(self):
        """End-to-end: MemoryAwarePrefixCache.fetch on a request that shares
        only a prefix with a longer stored entry must return a cache whose
        arrays are already sliced down.  This is the full regression of the
        #384 scenario above the unit level.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        model = MagicMock()
        cache = MemoryAwarePrefixCache(
            model,
            MemoryCacheConfig(max_memory_mb=64, max_entries=10, min_prefix_tokens=1),
        )

        # Stored: tokens [1..120] with 120 positions of KV data, the first 60
        # tokens being the shared prefix (all 1.0), the last 60 private (7.0).
        stored_layer = KVCache()
        shared = mx.ones((1, 2, 60, 4), dtype=mx.float32)
        private = mx.full((1, 2, 60, 4), 7.0, dtype=mx.float32)
        stored_layer.keys = mx.concatenate([shared, private], axis=2)
        stored_layer.values = stored_layer.keys
        stored_layer.offset = 120
        cache.store(list(range(1, 121)), [stored_layer])

        # New request: tokens [1..59] + [999, 1000, 1001] — first 59 tokens
        # match, then diverge.  LCP is 59.
        new_tokens = list(range(1, 60)) + [999, 1000, 1001]
        fetched, remaining = cache.fetch(new_tokens)

        assert fetched is not None
        tc = fetched[0]
        # LCP of 59 (the divergent tokens are stripped).
        assert tc.offset == 59
        assert tc.keys.shape[-2] == 59
        # Critical: the "7.0" private content from the stored entry must NOT
        # be visible anywhere in the returned cache (this is what caused the
        # cross-request contamination in #384).
        assert float(mx.max(tc.keys).item()) == 1.0
        assert remaining == [999, 1000, 1001]


class TestRotatingCachePartialReuse:
    """Regression coverage for saturated rotating caches in prefix reuse (#678)."""

    @staticmethod
    def _full_cache(length):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        layer = KVCache()
        layer.keys = mx.ones((1, 1, length, 4), dtype=mx.float32)
        layer.values = layer.keys
        layer.offset = length
        return layer

    @staticmethod
    def _saturated_rotating_cache(offset=12, max_size=8):
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        layer = RotatingKVCache(max_size=max_size, keep=0)
        layer.keys = mx.ones((1, 1, max_size, 4), dtype=mx.float32)
        layer.values = layer.keys
        layer.offset = offset
        layer._idx = max_size
        return layer

    @staticmethod
    def _prefix_cache():
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        return MemoryAwarePrefixCache(
            MagicMock(),
            MemoryCacheConfig(max_memory_mb=64, max_entries=10, min_prefix_tokens=1),
        )

    def test_supersequence_skips_saturated_rotating_cache(self):
        cache = self._prefix_cache()
        rotating = self._saturated_rotating_cache()
        cache.store(
            [1, 2, 3, 4, 5, 6],
            [self._full_cache(6), rotating],
        )

        fetched, remaining = cache.fetch([1, 2, 3, 4])

        assert fetched is None
        assert remaining == [1, 2, 3, 4]
        assert rotating.offset == 12

    def test_lcp_skips_saturated_rotating_cache(self):
        cache = self._prefix_cache()
        rotating = self._saturated_rotating_cache()
        cache.store(
            [1, 2, 3, 4, 5, 6],
            [self._full_cache(6), rotating],
        )

        fetched, remaining = cache.fetch([1, 2, 9])

        assert fetched is None
        assert remaining == [1, 2, 9]
        assert rotating.offset == 12

    def test_exact_and_prefix_hits_do_not_require_rewind(self):
        cache = self._prefix_cache()
        cache.store(
            [1, 2, 3, 4],
            [self._full_cache(4), self._saturated_rotating_cache()],
        )

        exact, exact_remaining = cache.fetch([1, 2, 3, 4])
        prefix, prefix_remaining = cache.fetch([1, 2, 3, 4, 5])

        assert exact is not None
        assert exact_remaining == []
        assert prefix is not None
        assert prefix_remaining == [5]

    def test_cache_list_is_rejected_even_when_children_are_trimmable(self):
        from mlx_lm.models.cache import CacheList

        from vllm_mlx.memory_cache import _is_cache_layer_trimmable

        layer = CacheList(self._full_cache(8), self._full_cache(8))

        assert not _is_cache_layer_trimmable(layer)

    def test_supersequence_skips_cache_list_container(self):
        from mlx_lm.models.cache import CacheList

        cache = self._prefix_cache()
        container = CacheList(self._full_cache(6), self._full_cache(6))
        cache.store([1, 2, 3, 4, 5, 6], [container])

        fetched, remaining = cache.fetch([1, 2, 3, 4])

        assert fetched is None
        assert remaining == [1, 2, 3, 4]
        assert all(child.offset == 6 for child in container.caches)

    def test_quantized_wrapper_rejects_rotating_metadata(self):
        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _is_cache_layer_trimmable,
        )

        wrapper = _QuantizedCacheWrapper.__new__(_QuantizedCacheWrapper)
        wrapper.keys = object()
        wrapper.values = object()
        wrapper.bits = 8
        wrapper.group_size = 64
        wrapper.orig_type = object
        wrapper.orig_attrs = {"max_size": 8}

        for offset in (4, 8):
            wrapper.offset = offset
            assert not _is_cache_layer_trimmable(wrapper)


class TestDequantizeCacheSlice:
    """Tests for _dequantize_cache slicing after dequantization.

    When KV cache quantization is enabled (--kv-cache-quantization), the
    prefix cache stores _QuantizedCacheWrapper layers.  After LCP trim
    reduces the offset, _dequantize_cache must slice the dequantized arrays
    down to offset to prevent readers that bypass offset (e.g. Gemma 4's
    KV-shared layers reading cache.state) from seeing stale tokens.

    This is the quantized-cache counterpart of the plain-KVCache fix
    tested in TestTrimCacheOffset above.
    """

    def test_dequantize_slices_to_offset(self):
        """After trim + dequantize, keys/values shape[-2] == offset."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
            _trim_cache_offset,
        )

        # Build a KVCache with 500 tokens, quantize it, then trim to 60.
        layer = KVCache()
        layer.keys = mx.ones((1, 4, 512, 64), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 512, 64), dtype=mx.float32)
        layer.offset = 512
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        trimmed = _trim_cache_offset([qw], 512 - 60)
        result = _dequantize_cache(trimmed)

        tc = result[0]
        assert tc.offset == 60
        assert tc.keys.shape[-2] == 60
        assert tc.values.shape[-2] == 60

    def test_dequantize_no_stale_tokens_via_state(self):
        """Stale tokens from a previous request must not be visible via cache.state."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
            _trim_cache_offset,
        )

        layer = KVCache()
        # First 64 positions: shared prefix (1.0), next 448: private (7.0)
        shared = mx.ones((1, 4, 64, 64), dtype=mx.float32)
        private = mx.full((1, 4, 448, 64), 7.0, dtype=mx.float32)
        layer.keys = mx.concatenate([shared, private], axis=2)
        layer.values = mx.concatenate([shared, private], axis=2)
        layer.offset = 512
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        trimmed = _trim_cache_offset([qw], 512 - 64)
        result = _dequantize_cache(trimmed)

        tc = result[0]
        keys_view, _ = tc.state
        assert keys_view.shape[-2] == 64
        # Dequantized values are approximate (quantization error), but should
        # be close to 1.0 (the shared prefix), never near 7.0 (the private data).
        assert float(mx.max(keys_view).item()) < 2.0

    def test_dequantize_no_trim_preserves_full_array(self):
        """When offset == shape[-2], no slicing occurs."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
        )

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        layer.offset = 128
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        result = _dequantize_cache([qw])

        tc = result[0]
        assert tc.offset == 128
        assert tc.keys.shape[-2] == 128
        assert tc.values.shape[-2] == 128

    def test_dequantize_source_unaffected(self):
        """Dequantizing must not mutate the stored quantized wrapper."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
            _trim_cache_offset,
        )

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 256, 64), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 256, 64), dtype=mx.float32)
        layer.offset = 256
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        original_offset = qw.offset
        original_keys_shape = qw.keys[0].shape  # quantized data tuple

        trimmed = _trim_cache_offset([qw], 192)
        _dequantize_cache(trimmed)

        # Source wrapper unchanged
        assert qw.offset == original_offset
        assert qw.keys[0].shape == original_keys_shape

    def test_dequantize_end_to_end_fetch_with_quantization(self):
        """End-to-end: store with kv_quantize=True, fetch with LCP, verify no stale data."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            MemoryAwarePrefixCache,
            MemoryCacheConfig,
        )

        model = MagicMock()
        pc = MemoryAwarePrefixCache(
            model,
            MemoryCacheConfig(
                max_memory_mb=64,
                max_entries=10,
                kv_quantize=True,
                kv_bits=8,
                kv_group_size=64,
                kv_min_quantize_tokens=0,
                min_prefix_tokens=1,
            ),
        )

        # Store a KVCache with 128 tokens — store() quantizes automatically.
        layer = KVCache()
        shared = mx.ones((1, 2, 64, 64), dtype=mx.float32)
        private = mx.full((1, 2, 64, 64), 7.0, dtype=mx.float32)
        layer.keys = mx.concatenate([shared, private], axis=2)
        layer.values = mx.concatenate([shared, private], axis=2)
        layer.offset = 128
        mx.eval(layer.keys, layer.values)

        pc.store(list(range(1, 129)), [layer])

        # Fetch with partial match (first 60 tokens match, then diverge).
        # fetch() dequantizes automatically when kv_quantize=True.
        new_tokens = list(range(1, 61)) + [999, 1000]
        fetched, remaining = pc.fetch(new_tokens)

        assert fetched is not None
        tc = fetched[0]
        assert tc.offset == 60
        assert tc.keys.shape[-2] == 60
        # No private data (7.0) visible — only shared prefix (~1.0 with quantization noise)
        assert float(mx.max(tc.keys).item()) < 2.0
        assert remaining == [999, 1000]


class TestDetachCacheForStorage:
    """Tests for ``_detach_cache_for_storage``: stored entries must not alias
    live batch state or retain lazy computation graphs.

    Regression: ``MemoryAwarePrefixCache.store()`` stored per-request cache
    layers by reference.  For hybrid models (e.g. Qwen3.5/3.6, whose
    ``ArraysCache`` layers expose their mutable ``cache`` list via
    ``.state``), the stored entry aliased the live state container (same
    class of bug as the SimpleEngine snapshot aliasing, #575) and — because
    the extracted per-request arrays are lazy slices of batch-wide arrays —
    retained the entire upstream computation graph.  Under sustained traffic
    each stored entry pinned Metal buffer handles roughly proportional to
    generated tokens, until the process hit the device resource limit
    (``[metal::malloc] Resource limit (N) exceeded``) and aborted.
    """

    def test_arrays_cache_container_not_aliased(self):
        """The stored ArraysCache snapshot must not follow later mutation."""
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        parent = mx.arange(32, dtype=mx.float32).reshape(4, 8)
        mx.eval(parent)
        lazy = parent[1:2]
        for _ in range(5):
            lazy = lazy + 1
        expected = [[float(v) + 5 for v in range(8, 16)]]

        layer = ArraysCache(size=1)
        layer[0] = lazy

        detached = _detach_cache_for_storage([layer])

        assert detached[0] is not layer
        assert detached[0].cache is not layer.cache
        assert detached[0][0].tolist() == expected

        # Simulate the batch generator advancing the live state after the
        # snapshot was stored — the stored copy must not change.
        layer[0] = mx.zeros((1, 8))
        assert detached[0][0].tolist() == expected

    def test_kv_cache_layer_snapshotted_with_equal_arrays(self):
        """KVCache layers are snapshotted; array contents are preserved."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        layer = KVCache()
        layer.keys = mx.arange(1 * 2 * 4 * 3, dtype=mx.float32).reshape(1, 2, 4, 3)
        layer.values = mx.ones((1, 2, 4, 3))
        layer.offset = 4

        detached = _detach_cache_for_storage([layer])

        assert detached[0] is not layer
        assert detached[0].offset == 4
        assert detached[0].keys.tolist() == layer.keys.tolist()
        assert detached[0].values.tolist() == layer.values.tolist()

    def test_lazy_graph_is_cut(self):
        """Detaching must not retain the upstream lazy graph / parent buffers.

        Builds a long lazy op chain rooted at a large parent array, detaches,
        then drops every reference except the detached copy.  If the graph
        were retained, active memory would still include the parent (8MB+);
        the detached copy itself is only 16KB.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        mx.eval(mx.zeros((1,)))
        base = mx.get_active_memory()

        big_parent = mx.random.normal((512, 4096))  # ~8MB
        chain = big_parent[0:1]
        for _ in range(200):
            chain = chain * 1.0001
        layer = ArraysCache(size=1)
        layer[0] = chain

        detached = _detach_cache_for_storage([layer])

        del big_parent, chain, layer
        mx.eval(mx.zeros((1,)))

        retained = mx.get_active_memory() - base
        assert retained < 1_000_000, (
            f"lazy graph retained: {retained / 1e6:.1f}MB still active "
            f"after dropping originals (expected < 1MB)"
        )
        assert detached[0][0].shape == (1, 4096)

    def test_cache_list_children_snapshotted_without_write_through(self):
        """CacheList's ``state`` setter writes through to child caches in
        place; the detach path must snapshot children recursively instead."""
        import mlx.core as mx
        from mlx_lm.models.cache import CacheList, KVCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        inner = KVCache()
        inner.keys = mx.arange(1 * 2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)
        inner.values = mx.ones((1, 2, 3, 4))
        inner.offset = 3
        original_keys = inner.keys

        cl = CacheList(inner)

        detached = _detach_cache_for_storage([cl])

        assert detached[0] is not cl
        assert detached[0].caches is not cl.caches
        assert detached[0].caches[0] is not inner
        assert detached[0].caches[0].keys.tolist() == original_keys.tolist()
        # The live child must be untouched (no setter write-through)
        assert cl.caches[0] is inner
        assert inner.keys is original_keys

    def test_slots_class_snapshotted(self):
        """Layers using ``__slots__`` (no ``__dict__``), like
        ``_QuantizedCacheWrapper``, must snapshot without crashing."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _detach_cache_for_storage,
            _QuantizedCacheWrapper,
        )

        kv = KVCache()
        kv.keys = mx.ones((1, 2, 4, 64))
        kv.values = mx.ones((1, 2, 4, 64))
        kv.offset = 4
        mx.eval(kv.keys, kv.values)
        wrapper = _QuantizedCacheWrapper(kv, bits=8, group_size=32)

        detached = _detach_cache_for_storage([wrapper])

        assert detached[0] is not wrapper
        assert isinstance(detached[0], _QuantizedCacheWrapper)
        assert detached[0].offset == wrapper.offset
        # mx.quantize returns a (data, scales, biases) container; the
        # snapshot must preserve its type and contents.
        assert type(detached[0].keys) is type(wrapper.keys)
        assert len(detached[0].keys) == len(wrapper.keys)
        for got, orig in zip(detached[0].keys, wrapper.keys):
            assert got.tolist() == orig.tolist()

    def test_snapshot_failure_raises_undetachable(self):
        """A recognized layer whose snapshot fails raises
        UndetachableCacheError (fail-closed) instead of silently aliasing."""
        import pytest

        from vllm_mlx.memory_cache import (
            UndetachableCacheError,
            _detach_cache_for_storage,
        )

        class Undetachable:
            def __init__(self):
                self.values = None
                self.offset = 0

            @property
            def keys(self):  # read-only: snapshot assignment must fail
                return None

        with pytest.raises(UndetachableCacheError):
            _detach_cache_for_storage([Undetachable()])

    def test_unknown_array_bearing_layer_raises_undetachable(self):
        """An unrecognized layer type that carries arrays must fail closed —
        storing it by reference would reintroduce the retention leak."""
        import mlx.core as mx
        import pytest

        from vllm_mlx.memory_cache import (
            UndetachableCacheError,
            _detach_cache_for_storage,
        )

        class ExoticCache:  # no keys/caches/state — matches no branch
            def __init__(self):
                self.buffers = [mx.ones((1, 4)), mx.zeros((1, 4))]

        with pytest.raises(UndetachableCacheError):
            _detach_cache_for_storage([ExoticCache()])

    def test_unknown_array_free_layer_passes_through(self):
        """An unrecognized layer with no arrays has nothing to pin and is
        passed through unchanged."""
        from vllm_mlx.memory_cache import _detach_cache_for_storage

        class Metadata:
            def __init__(self):
                self.n_tokens = 7
                self.tag = "prefill"

        layer = Metadata()
        detached = _detach_cache_for_storage([layer])
        assert detached[0] is layer


class TestStoreFailClosedAndEviction:
    """store()-level tests requested in review: fail-closed rejection,
    small-cap eviction with detached entries, a real extracted batch slice,
    and nested container / dict state."""

    def _make_cache(self, max_memory_mb=2):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        class _Model:
            pass

        return MemoryAwarePrefixCache(
            _Model(),
            MemoryCacheConfig(max_memory_mb=max_memory_mb, min_prefix_tokens=1),
        )

    @staticmethod
    def _kv_layer(n_tokens):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        kv = KVCache()
        kv.keys = mx.zeros((1, 8, n_tokens, 64))
        kv.values = mx.zeros((1, 8, n_tokens, 64))
        kv.offset = n_tokens
        mx.eval(kv.keys, kv.values)
        return kv

    def test_store_rejects_undetachable_entry(self):
        """store() returns False and stores nothing when any layer of the
        entry cannot be detached."""
        import mlx.core as mx

        class ExoticCache:
            def __init__(self):
                self.buffers = [mx.ones((1, 4))]

        cache = self._make_cache()
        ok = cache.store([1, 2, 3], [self._kv_layer(4), ExoticCache()])

        assert ok is False
        assert len(cache) == 0
        got, remaining = cache.fetch([1, 2, 3])
        assert got is None
        assert remaining == [1, 2, 3]

    def test_small_cap_eviction_with_detached_entries(self):
        """Detached entries are priced by the byte accounting and evicted
        LRU under a small cap."""
        # Each entry: keys+values = 2 * (8*224*64*4) bytes ~ 0.875 MB.
        # Cap 2 MB -> third store must evict the least recently used.
        cache = self._make_cache(max_memory_mb=2)

        assert cache.store([1], [self._kv_layer(224)])
        assert cache.store([2], [self._kv_layer(224)])
        assert cache.store([3], [self._kv_layer(224)])

        stats = cache.get_stats()
        assert stats["evictions"] >= 1
        assert stats["current_memory_mb"] <= 2.0

        got, _ = cache.fetch([1])
        assert got is None  # LRU-evicted
        got, remaining = cache.fetch([3])
        assert got is not None
        assert remaining == []

    def test_store_real_extracted_batch_slice_cuts_batch_retention(self):
        """Drive store() with what the MLLM path actually produces —
        BatchKVCache.extract() / ArraysCache.extract() slices, which are
        LAZY expressions over the batch-wide buffers — and assert the leak
        dimension: after storing and dropping the batch, resident memory is
        entry-sized, not batch-sized.  (Extract's output objects are always
        fresh, so aliasing checks alone cannot fail; retention is what the
        old store-by-reference code got wrong.)"""
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, BatchKVCache

        mx.eval(mx.zeros((1,)))
        base = mx.get_active_memory()

        # 8 rows x 8 heads x 2048 tokens x 128 dims fp32 ~ 67 MB per array.
        batch_kv = BatchKVCache(left_padding=[0] * 8)
        keys = mx.random.normal((8, 8, 2048, 128))
        values = mx.random.normal((8, 8, 2048, 128))
        mx.eval(keys, values)
        batch_kv.update_and_fetch(keys, values)

        batch_state = ArraysCache(size=1)
        state = mx.random.normal((8, 4096))
        mx.eval(state)
        batch_state[0] = state

        extracted = [batch_kv.extract(1), batch_state.extract(1)]
        expected_keys = keys[1:2].tolist()
        expected_state = state[1:2].tolist()
        entry_bytes = 2 * (8 * 2048 * 128 * 4) + 4096 * 4  # one row of k+v

        cache = self._make_cache(max_memory_mb=64)
        assert cache.store([5, 6, 7], extracted)

        # The batch finishes: every batch-wide buffer should now be
        # releasable.  Store-by-reference kept the stored entry's lazy
        # extract() expressions rooted in them.
        del batch_kv, batch_state, keys, values, state, extracted
        mx.eval(mx.zeros((1,)))
        retained = mx.get_active_memory() - base
        assert retained < entry_bytes * 2, (
            f"stored entry retains {retained / 1e6:.1f}MB — batch-wide "
            f"buffers pinned (entry itself is {entry_bytes / 1e6:.1f}MB)"
        )

        got, remaining = cache.fetch([5, 6, 7])
        assert remaining == []
        assert got[0].keys.tolist() == expected_keys
        assert got[0].offset == 2048
        assert got[1][0].tolist() == expected_state

    def test_store_nested_container_and_dict_state(self):
        """Nested CacheList (child KVCache + ArraysCache) and extracted
        dict-state layers round-trip through store()/fetch() as snapshots."""
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

        inner_kv = KVCache()
        inner_kv.keys = mx.arange(1 * 2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)
        inner_kv.values = mx.ones((1, 2, 3, 4))
        inner_kv.offset = 3
        inner_state = ArraysCache(size=1)
        inner_state[0] = mx.full((1, 8), 2.0)
        cl = CacheList(inner_kv, inner_state)

        dict_layer = {
            "class": "KVCache",
            "state": (mx.ones((1, 2, 3, 4)), mx.zeros((1, 2, 3, 4))),
            "meta_state": (),
        }

        expected_keys = inner_kv.keys.tolist()

        cache = self._make_cache(max_memory_mb=8)
        assert cache.store([9, 10], [cl, dict_layer])

        # Mutate the live containers after storing.
        inner_kv.keys = mx.zeros((1, 2, 3, 4))
        inner_state[0] = mx.zeros((1, 8))

        got, remaining = cache.fetch([9, 10])
        assert remaining == []
        assert got[0].caches[0].keys.tolist() == expected_keys
        assert got[0].caches[1][0].tolist() == [[2.0] * 8]
        assert got[1]["class"] == "KVCache"
        assert got[1]["state"][0].tolist() == mx.ones((1, 2, 3, 4)).tolist()


class TestOwnerReviewRegressions:
    """Regressions requested in the #642 owner review: guaranteed-allocation
    detachment, pre-materialization size preflight, quantize-before-detach
    graph cut, and nested-mapping state handling."""

    @staticmethod
    def _make_cache(max_memory_mb=8, **cfg):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        class _Model:
            pass

        return MemoryAwarePrefixCache(
            _Model(),
            MemoryCacheConfig(max_memory_mb=max_memory_mb, min_prefix_tokens=1, **cfg),
        )

    def test_rotating_snapshot_immune_to_source_updates(self):
        """Finding 1: an already-contiguous, evaluated RotatingKVCache must
        be stored as a freshly allocated copy — continuing update_and_fetch()
        on the source must not change the fetched entry."""
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        live = RotatingKVCache(max_size=8)
        k = mx.random.normal((1, 2, 8, 16))
        v = mx.random.normal((1, 2, 8, 16))
        mx.eval(k, v)
        live.update_and_fetch(k, v)
        mx.eval(live.keys, live.values)  # source contiguous and evaluated

        cache = self._make_cache()
        assert cache.store([1, 2, 3], [live])
        got, _ = cache.fetch([1, 2, 3])
        before = got[0].keys.tolist()

        # Ring buffer writes in place once wrapped.
        for _ in range(6):
            live.update_and_fetch(
                mx.random.normal((1, 2, 1, 16)), mx.random.normal((1, 2, 1, 16))
            )
        mx.eval(live.keys, live.values)

        got, remaining = cache.fetch([1, 2, 3])
        assert remaining == []
        assert got[0].keys.tolist() == before

    def test_oversized_lazy_entry_rejected_without_materialization(self):
        """Finding 2: an over-limit lazy entry must be rejected by the
        shape-based preflight BEFORE anything materializes it.

        The entry uses ``offset < keys.shape[2]`` deliberately — the normal
        step-padded shape mlx-lm produces — so the trim path is exercised
        too: a trim that evaluates its slices would materialize the entry
        before the preflight can refuse it.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        mx.eval(mx.zeros((1,)))
        base = mx.get_active_memory()

        kv = KVCache()
        # ~64 MB promised, never evaluated, with step-chunk padding past
        # offset (offset 3900 < 4096 slots) as update_and_fetch produces.
        kv.keys = mx.zeros((1, 8, 4096, 512))
        kv.values = mx.zeros((1, 8, 4096, 512))
        kv.offset = 3900

        cache = self._make_cache(max_memory_mb=2)
        ok = cache.store([1, 2, 3, 4], [kv])

        mx.eval(mx.zeros((1,)))
        grown = mx.get_active_memory() - base
        assert ok is False
        assert len(cache) == 0
        assert (
            grown < 5_000_000
        ), f"rejected entry still materialized {grown / 1e6:.1f}MB"

    def test_padded_rotating_cache_accounted_at_stored_size(self):
        """Sliding-window layers cannot be sliced to offset, so detachment
        copies their padded buffers — the estimator must price those same
        buffers, or the byte cap is silently breached (measured 20-30%
        resident-vs-accounted drift before the fix)."""
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        from vllm_mlx.memory_cache import estimate_kv_cache_memory

        mx.eval(mx.zeros((1,)))
        base = mx.get_active_memory()

        live = RotatingKVCache(max_size=1024)
        # Grow past a step boundary so keys.shape[2] > offset (padding):
        # the initial update sizes exactly; subsequent single-token updates
        # grow the buffer in step-sized chunks.
        live.update_and_fetch(
            mx.random.normal((1, 8, 500, 128)), mx.random.normal((1, 8, 500, 128))
        )
        for _ in range(5):
            live.update_and_fetch(
                mx.random.normal((1, 8, 1, 128)), mx.random.normal((1, 8, 1, 128))
            )
        mx.eval(live.keys, live.values)
        assert live.keys.shape[2] > live.offset  # padded, untrimmable

        projected = estimate_kv_cache_memory([live])
        raw_bytes = live.keys.size * 4 + live.values.size * 4
        assert (
            projected == raw_bytes
        ), f"estimator prices {projected} but detach will copy {raw_bytes}"

        cache = self._make_cache(max_memory_mb=64)
        assert cache.store([1, 2], [live])
        # Drop the live source: what remains resident is the stored copy.
        del live
        mx.eval(mx.zeros((1,)))
        resident = mx.get_active_memory() - base

        stats = cache.get_stats()
        accounted = stats["current_memory_mb"] * 1024 * 1024
        assert abs(resident - accounted) < 2_000_000, (
            f"resident {resident / 1e6:.1f}MB vs accounted "
            f"{accounted / 1e6:.1f}MB — byte cap not trustworthy"
        )

    def test_dict_layer_with_arrays_outside_state_fails_closed(self):
        """A dict layer smuggling arrays outside its 'state' field must be
        rejected, not stored with those arrays aliased."""
        import mlx.core as mx

        layer = {
            "state": (mx.ones((1, 4)),),
            "aux": mx.zeros((1024, 1024)),  # aliased under the old code
        }
        cache = self._make_cache()
        assert cache.store([5, 6], [layer]) is False
        assert len(cache) == 0

    def test_detach_preserves_bool_dtype(self):
        """The guaranteed-allocation copy must be dtype-faithful — bare
        ``+ 0`` promotes bool to int32."""
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        layer = ArraysCache(size=1)
        layer[0] = mx.array([True, False, True])
        detached = _detach_cache_for_storage([layer])
        assert detached[0][0].dtype == mx.bool_
        assert detached[0][0].tolist() == [True, False, True]

    def test_batch_kv_left_padding_detached(self):
        """Batch cache variants carry per-row metadata arrays the batch
        generator rebinds; the snapshot must not alias them."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        batch = BatchKVCache(left_padding=[0, 2])
        batch.update_and_fetch(
            mx.random.normal((2, 2, 4, 8)), mx.random.normal((2, 2, 4, 8))
        )
        detached = _detach_cache_for_storage([batch])

        assert detached[0].left_padding is not batch.left_padding
        assert detached[0].left_padding.tolist() == batch.left_padding.tolist()
        if hasattr(batch.offset, "shape"):
            assert detached[0].offset is not batch.offset

    def test_quantized_store_does_not_retain_fp_graph(self):
        """Finding 3: with kv_quantize enabled the stored representation is
        the quantized one — evaluated and graph-detached — so resident memory
        tracks the quantized size, not the full-precision source."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        mx.eval(mx.zeros((1,)))
        base = mx.get_active_memory()

        parent = mx.random.normal((1, 8, 4096, 64))  # 8.4 MB fp32
        mx.eval(parent)
        kv = KVCache()
        kv.keys = parent[:, :, :2048, :]  # lazy slices, as extract() produces
        kv.values = parent[:, :, 2048:, :]
        kv.offset = 2048
        expected = kv.keys.tolist()
        fp_bytes = 2 * (8 * 2048 * 64 * 4)  # 8.4 MB

        cache = self._make_cache(
            max_memory_mb=16,
            kv_quantize=True,
            kv_min_quantize_tokens=1,
            kv_bits=8,
            kv_group_size=64,
        )
        assert cache.store([7, 8, 9], [kv])

        del parent, kv
        mx.eval(mx.zeros((1,)))
        retained = mx.get_active_memory() - base
        assert retained < fp_bytes / 2, (
            f"stored quantized entry retains {retained / 1e6:.1f}MB — "
            f"full-precision graph not cut (fp size {fp_bytes / 1e6:.1f}MB)"
        )

        got, remaining = cache.fetch([7, 8, 9])
        assert remaining == []
        err = float(mx.max(mx.abs(got[0].keys - mx.array(expected))).item())
        assert err < 0.1, f"quantization roundtrip error too large: {err}"
        assert got[0].offset == 2048

    def test_nested_mapping_state_detached_and_accounted(self):
        """Finding 4: {"state": {"ssm": array}} layers must be deep-detached
        (mapping rebuilt, arrays copied), priced by the recursive estimator,
        and evictable under the byte cap."""
        import mlx.core as mx

        from vllm_mlx.memory_cache import estimate_kv_cache_memory

        def _mapping_layer(fill):
            parent = mx.full((256, 1024), fill)  # 1 MB fp32
            mx.eval(parent)
            return parent, {
                "class": "SSMCache",
                "state": {"ssm": parent[0:256] * 1.0, "step": None},
                "meta_state": (),
            }

        parent, layer = _mapping_layer(3.0)

        # Accounting: recursive walk prices the nested mapping.
        assert estimate_kv_cache_memory([layer]) == 256 * 1024 * 4

        cache = self._make_cache(max_memory_mb=8)
        assert cache.store([1, 2], [layer])

        # Mutation isolation: the fetched mapping is a rebuilt container
        # holding copied arrays, not the caller's dict.
        layer["state"]["ssm"] = mx.zeros((256, 1024))
        got, remaining = cache.fetch([1, 2])
        assert remaining == []
        assert got[0]["state"] is not layer["state"]
        assert got[0]["state"]["step"] is None
        assert float(got[0]["state"]["ssm"][0, 0].item()) == 3.0

        # Eviction: byte-cap LRU fires on mapping-state entries (~1 MB each,
        # 2 MB cap -> third store evicts the least recently used).
        small = self._make_cache(max_memory_mb=2)
        for i, fill in enumerate([1.0, 2.0, 4.0]):
            _, lyr = _mapping_layer(fill)
            assert small.store([10 + i], [lyr])
        stats = small.get_stats()
        assert stats["evictions"] >= 1
        gone, _ = small.fetch([10])
        assert gone is None
        kept, remaining = small.fetch([12])
        assert kept is not None and remaining == []


class TestFailClosedPostcondition:
    """Regressions from the third independent review round: the fail-closed
    guarantee must hold for ``_BaseCache`` subclasses (which all inherit a
    default ``state`` property returning ``[]``, making the state branch
    match any of them), and the estimator must price the metadata arrays
    detachment copies on state-carrying layers."""

    @staticmethod
    def _make_cache(max_memory_mb=8):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        class _Model:
            pass

        return MemoryAwarePrefixCache(
            _Model(),
            MemoryCacheConfig(max_memory_mb=max_memory_mb, min_prefix_tokens=1),
        )

    def test_base_cache_subclass_with_sidecar_array_fails_closed(self):
        """A ``_BaseCache`` subclass holding an array in an attribute that
        ``.state`` does not expose must be rejected, not silently stored as
        an alias of live memory priced at zero bytes.  The inherited default
        ``state`` property (returns ``[]``) previously routed such layers
        through the state branch, past the ``_bears_arrays`` fallback."""
        import mlx.core as mx
        import pytest
        from mlx_lm.models.cache import _BaseCache

        from vllm_mlx.memory_cache import (
            UndetachableCacheError,
            _detach_cache_for_storage,
        )

        class SidecarCache(_BaseCache):
            def __init__(self, buf):
                self.buf = buf

        live = mx.zeros((8, 8), dtype=mx.float32)
        mx.eval(live)

        with pytest.raises(UndetachableCacheError):
            _detach_cache_for_storage([SidecarCache(live)])

        cache = self._make_cache()
        ok = cache.store([1, 2, 3], [SidecarCache(live)])
        assert ok is False
        assert len(cache) == 0
        assert cache.get_stats()["store_rejections"] == 1
        got, remaining = cache.fetch([1, 2, 3])
        assert got is None
        assert remaining == [1, 2, 3]

    def test_arrays_cache_metadata_arrays_priced(self):
        """State metadata arrays must be included in resident accounting."""
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache

        from vllm_mlx.memory_cache import estimate_kv_cache_memory

        ac = ArraysCache(2, left_padding=[0, 1])
        ac[0] = mx.random.normal((2, 4))
        ac[1] = mx.random.normal((2, 4))
        ac.lengths = mx.array([3, 4])
        mx.eval(ac[0], ac[1], ac.lengths, ac.left_padding)

        resident = ac[0].nbytes + ac[1].nbytes
        for attr in ("left_padding", "lengths"):
            extra = getattr(ac, attr, None)
            if extra is not None and hasattr(extra, "nbytes"):
                resident += extra.nbytes

        assert estimate_kv_cache_memory([ac]) == resident


class TestHybridRestartPersistence:
    class _Tokenizer:
        name_or_path = "qualified-tokenizer"
        vocab_size = 16
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        chat_template = "{{ messages }}"

        def __init__(self, suffix=""):
            self.suffix = suffix

        def get_vocab(self):
            return {f"token-{index}{self.suffix}": index for index in range(16)}

    class _Renderer:
        def __init__(self, suffix=""):
            self.chat_template = f"{{{{ messages }}}}{suffix}"

    class _Args:
        model_type = "qwen3_5"
        num_hidden_layers = 2
        hidden_size = 8
        intermediate_size = 16
        vocab_size = 16
        num_attention_heads = 2
        num_key_value_heads = 1
        head_dim = 4
        linear_num_value_heads = 2
        linear_num_key_heads = 1
        linear_key_head_dim = 4
        linear_value_head_dim = 4
        linear_conv_kernel_dim = 3
        layer_types = ["linear_attention", "full_attention"]
        tie_word_embeddings = True

    class _Model:
        def __init__(self):
            self.args = TestHybridRestartPersistence._Args()

        def make_cache(self):
            from mlx_lm.models.cache import ArraysCache, KVCache

            return [ArraysCache(size=2), KVCache()]

    class _VLMArgs:
        model_type = "qwen3_5"

        def __init__(self, *, as_mapping=False):
            text_config = TestHybridRestartPersistence._Args()
            if as_mapping:
                keys = (
                    "model_type",
                    "num_hidden_layers",
                    "vocab_size",
                    "num_key_value_heads",
                    "head_dim",
                    "linear_num_value_heads",
                    "linear_num_key_heads",
                    "linear_key_head_dim",
                    "linear_value_head_dim",
                    "linear_conv_kernel_dim",
                )
                text_config = {key: getattr(text_config, key) for key in keys}
            self.text_config = text_config

    class _VLMModel:
        def __init__(self, *, text_config_as_mapping=False):
            self.args = TestHybridRestartPersistence._VLMArgs(
                as_mapping=text_config_as_mapping
            )

        def make_cache(self):
            from mlx_vlm.models.cache import ArraysCache, KVCache

            return [ArraysCache(size=2), KVCache()]

    class _QSAModel:
        def __init__(self):
            self.args = TestHybridRestartPersistence._Args()
            self.args.model_type = "qwen4_exp"
            self.args.num_hidden_layers = 1
            self.args.head_dim = 64
            self.args.layer_types = ["full_attention"]

        def make_cache(self):
            from mlx_vlm.models.qwen4_exp.language import QSAKVCache

            return [QSAKVCache()]

    class _Qwen4PLEModel:
        def __init__(self):
            self.args = TestHybridRestartPersistence._Args()
            self.args.model_type = "qwen4_exp"
            self.args.num_hidden_layers = 3
            self.args.layer_types = [
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            self.args.hidden_size = 8
            self.args.hc_count = 4
            self.args.ple_conv_kernel_size = 4
            self.args.ngram_size = 3

        def make_cache(self):
            from mlx_vlm.models.cache import ArraysCache
            from mlx_vlm.models.qwen4_exp.language import QSAKVCache

            return [ArraysCache(size=2), ArraysCache(size=4), QSAKVCache()]

    @staticmethod
    def _qwen4_ple_state():
        import mlx.core as mx
        from mlx_vlm.models.cache import ArraysCache
        from mlx_vlm.models.qwen4_exp.language import QSAKVCache

        linear = ArraysCache(size=2)
        linear.state = [
            mx.zeros((1, 2, 16), dtype=mx.float32),
            mx.zeros((1, 2, 4, 4), dtype=mx.float32),
        ]
        ple = ArraysCache(size=4)
        ple.state = [
            mx.ones((1, 2, 16), dtype=mx.float32),
            mx.ones((1, 2, 4, 4), dtype=mx.float32),
            mx.ones((1, 9, 32), dtype=mx.float32),
            mx.array([[7, 9]], dtype=mx.int64),
        ]
        qsa = QSAKVCache()
        qsa.state = (
            mx.ones((1, 1, 4, 4), dtype=mx.float32),
            mx.ones((1, 1, 4, 4), dtype=mx.float32),
            mx.ones((1, 4, 4), dtype=mx.float32),
            mx.arange(4, dtype=mx.int32).reshape(1, 4),
        )
        logits = mx.arange(16, dtype=mx.float32).reshape(1, 16)
        mx.eval(*linear.state, *ple.state, *qsa.state, logits)
        return [linear, ple, qsa], logits

    @staticmethod
    def _state(*, vlm=False):
        import mlx.core as mx

        if vlm:
            from mlx_vlm.models.cache import ArraysCache, KVCache
        else:
            from mlx_lm.models.cache import ArraysCache, KVCache

        arrays = ArraysCache(size=2)
        arrays.state = [
            mx.arange(32, dtype=mx.float32).reshape(1, 2, 16),
            mx.ones((1, 2, 4, 4), dtype=mx.float32),
        ]
        arrays.left_padding = mx.array([0], dtype=mx.int32)
        arrays.lengths = mx.array([4], dtype=mx.int32)
        kv = KVCache()
        kv.keys = mx.ones((1, 1, 4, 4), dtype=mx.float32)
        kv.values = mx.full((1, 1, 4, 4), 2, dtype=mx.float32)
        kv.offset = 4
        logits = mx.arange(16, dtype=mx.float32).reshape(1, 16)
        mx.eval(*arrays.state, kv.keys, kv.values, logits)
        return [arrays, kv], logits

    @staticmethod
    def _qsa_state():
        import mlx.core as mx
        from mlx_vlm.models.qwen4_exp.language import QSAKVCache

        qsa = QSAKVCache()
        qsa.state = (
            mx.arange(256, dtype=mx.float32).reshape(1, 1, 4, 64),
            mx.full((1, 1, 4, 64), 2, dtype=mx.float32),
            mx.arange(256, dtype=mx.float32).reshape(1, 4, 64),
            mx.arange(4, dtype=mx.int32).reshape(1, 4),
        )
        logits = mx.arange(16, dtype=mx.float32).reshape(1, 16)
        mx.eval(*qsa.state, logits)
        return [qsa], logits

    def _cache(
        self,
        tokenizer=None,
        renderer=None,
        model_identity=None,
        model=None,
        *,
        kv_quantize=False,
    ):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        return MemoryAwarePrefixCache(
            model or self._Model(),
            MemoryCacheConfig(
                max_memory_mb=8,
                min_prefix_tokens=1,
                kv_quantize=kv_quantize,
                kv_bits=8,
                kv_group_size=64,
                kv_min_quantize_tokens=1,
            ),
            tokenizer=tokenizer or self._Tokenizer(),
            model_identity=model_identity or "qualified-model@revision",
            template_renderer=renderer or self._Renderer(),
        )

    def test_unpinned_remote_model_identity_disables_restart(self):
        from vllm_mlx.memory_cache import _compute_model_persistence_fingerprint

        assert (
            _compute_model_persistence_fingerprint(self._Model(), "owner/model") == ""
        )
        assert _compute_model_persistence_fingerprint(
            self._Model(), "owner/model@0123456789abcdef"
        )

    def test_round_trip_preserves_hybrid_state_and_exact_logits(self, tmp_path):
        import mlx.core as mx

        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)

        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert snapshot is not None
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        restored = self._cache()
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))
        assert restored.restore_hybrid_persistence_snapshot(loaded) == 1
        fetched, remaining = restored.fetch([1, 2, 3, 4])
        auxiliary = restored.fetch_exact_auxiliary([1, 2, 3, 4])

        assert remaining == []
        assert [type(layer).__name__ for layer in fetched] == [
            "ArraysCache",
            "KVCache",
        ]
        assert auxiliary is not None
        assert mx.array_equal(auxiliary["last_logits"], logits).item()
        for actual, expected in zip(fetched[0].state, state[0].state, strict=True):
            assert mx.array_equal(actual, expected).item()
        assert mx.array_equal(fetched[0].left_padding, state[0].left_padding).item()
        assert mx.array_equal(fetched[0].lengths, state[0].lengths).item()
        assert fetched[0].meta_state == state[0].meta_state
        assert fetched[1].offset == state[1].offset
        assert mx.array_equal(fetched[1].keys, state[1].keys).item()
        assert mx.array_equal(fetched[1].values, state[1].values).item()

    def test_real_qsa_tuple_state_round_trips_with_owned_accounting(self, tmp_path):
        import mlx.core as mx
        from mlx_vlm.models.qwen4_exp.language import QSAKVCache

        model = self._QSAModel()
        source = self._cache(model=model)
        state, logits = self._qsa_state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)

        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert snapshot is not None
        assert snapshot.entries[0].layers[0].layer_type == "QSAKVCache"
        assert snapshot.entries[0].layers[0].metadata["num_arrays"] == 4
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        restored = self._cache(model=model)
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))
        assert restored.restore_hybrid_persistence_snapshot(loaded) == 1
        fetched, remaining = restored.fetch([1, 2, 3, 4])

        assert remaining == []
        assert isinstance(fetched[0], QSAKVCache)
        for actual, expected in zip(fetched[0].state, state[0].state, strict=True):
            assert mx.array_equal(actual, expected).item()
        assert restored._current_memory == source._current_memory

    def test_qwen4_qsa_prefix_storage_quantizes_and_reconstructs_exact_state(self):
        import mlx.core as mx
        from mlx_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache
        from vllm_mlx.memory_cache import _dequantize_cache, _quantize_cache

        state, _ = self._qsa_state()
        quantized = _quantize_cache(state, bits=8, group_size=64)
        assert isinstance(quantized[0], QSAQuantizedKVCache)
        assert quantized[0].bits == 8
        assert quantized[0].group_size == 64

        restored = _dequantize_cache(quantized)
        assert type(restored[0]) is QSAKVCache
        assert restored[0].offset == state[0].offset
        for actual, expected in zip(restored[0].state, state[0].state, strict=True):
            assert actual.dtype == expected.dtype
            assert mx.allclose(actual, expected, rtol=0.02, atol=0.1).item()

    def test_qwen4_qsa_quantized_prefix_rejects_malformed_contract(self):
        from vllm_mlx.memory_cache import _dequantize_cache, _quantize_cache

        state, _ = self._qsa_state()
        assert _quantize_cache(state, bits=4, group_size=64)[0] is state[0]

        quantized = _quantize_cache(state, bits=8, group_size=64)
        quantized[0].bits = 4
        with pytest.raises(ValueError, match="unsupported Qwen4 QSA quantization"):
            _dequantize_cache(quantized)

        quantized = _quantize_cache(state, bits=8, group_size=64)
        quantized[0].index_keys = None
        with pytest.raises(ValueError, match="Qwen4 QSA indexer state is incomplete"):
            _dequantize_cache(quantized)

    def test_qwen4_qsa_quantized_ram_uses_float_v2_persistence_contract(self, tmp_path):
        import mlx.core as mx
        from mlx_vlm.models.qwen4_exp.language import QSAQuantizedKVCache

        from vllm_mlx.memory_cache import estimate_kv_cache_memory

        source = self._cache(model=self._QSAModel(), kv_quantize=True)
        state, logits = self._qsa_state()
        full_precision_bytes = estimate_kv_cache_memory(state)
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        stored = source._entries[(1, 2, 3, 4)]
        assert type(stored.cache[0]) is QSAQuantizedKVCache
        assert stored.memory_bytes < full_precision_bytes
        assert stored.cache[0].index_keys is not state[0].index_keys
        assert stored.cache[0].index_position_ids is not state[0].index_position_ids

        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert snapshot is not None
        layer = snapshot.entries[0].layers[0]
        assert layer.layer_type == "QSAKVCache"
        assert set(layer.tensors) == {"state_0", "state_1", "state_2", "state_3"}
        assert snapshot.entries[0].memory_bytes > stored.memory_bytes
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        restored = self._cache(model=self._QSAModel(), kv_quantize=True)
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))
        assert restored.restore_hybrid_persistence_snapshot(loaded) == 1
        restored_entry = restored._entries[(1, 2, 3, 4)]
        restored_qsa = restored_entry.cache[0]
        assert type(restored_qsa) is QSAQuantizedKVCache
        assert restored_entry.memory_bytes < snapshot.entries[0].memory_bytes
        assert restored._current_memory == restored_entry.memory_bytes
        assert mx.array_equal(restored_qsa.index_keys, state[0].index_keys).item()
        assert mx.array_equal(
            restored_qsa.index_position_ids, state[0].index_position_ids
        ).item()

        fetched, remaining = restored.fetch([1, 2, 3, 4])
        assert remaining == []
        assert mx.array_equal(fetched[0].index_keys, state[0].index_keys).item()
        assert mx.array_equal(
            fetched[0].index_position_ids, state[0].index_position_ids
        ).item()

        next_snapshot = restored.prepare_hybrid_persistence_snapshot()
        assert next_snapshot is not None
        next_layer = next_snapshot.entries[0].layers[0]
        assert next_layer.layer_type == "QSAKVCache"
        assert next_layer.metadata["state_container"] == "tuple"
        assert next_layer.metadata["num_arrays"] == 4
        assert next_snapshot.entries[0].memory_bytes == snapshot.entries[0].memory_bytes
        assert restored.write_hybrid_persistence_snapshot(
            str(tmp_path / "next"), next_snapshot
        )

    def test_real_qwen4_ple_and_qsa_round_trip_through_disk(self, tmp_path):
        import mlx.core as mx

        model = self._Qwen4PLEModel()
        source = self._cache(model=model)
        state, logits = self._qwen4_ple_state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert snapshot is not None
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)
        restored = self._cache(model=model)
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))
        assert restored.restore_hybrid_persistence_snapshot(loaded) == 1
        fetched, remaining = restored.fetch([1, 2, 3, 4])
        assert remaining == []
        assert [type(layer).__name__ for layer in fetched] == [
            "ArraysCache",
            "ArraysCache",
            "QSAKVCache",
        ]
        for actual_layer, expected_layer in zip(fetched, state, strict=True):
            for actual, expected in zip(
                actual_layer.state, expected_layer.state, strict=True
            ):
                assert actual.dtype == expected.dtype
                assert mx.array_equal(actual, expected).item()
        assert fetched[1].state[3].dtype == mx.int64
        assert restored._current_memory == source._current_memory
        assert restored._current_memory <= restored._max_memory

    @pytest.mark.parametrize("text_config_as_mapping", [False, True])
    def test_round_trip_uses_nested_vlm_text_config_geometry(
        self, tmp_path, text_config_as_mapping
    ):
        def make_cache():
            return self._cache(
                model=self._VLMModel(text_config_as_mapping=text_config_as_mapping),
                model_identity="qualified-vlm@revision",
            )

        source = make_cache()
        state, logits = self._state(vlm=True)
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        restored = make_cache()
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))

        assert restored.restore_hybrid_persistence_snapshot(loaded) == 1
        assert len(restored) == 1
        fetched, remaining = restored.fetch([1, 2, 3, 4])
        assert remaining == []
        assert all(
            type(layer).__module__ == "mlx_vlm.models.cache" for layer in fetched
        )
        assert restored.prepare_hybrid_persistence_snapshot() is not None

    def test_nested_vlm_missing_text_geometry_publishes_nothing(self, tmp_path):
        source = self._cache(
            model=self._VLMModel(),
            model_identity="qualified-vlm@revision",
        )
        state, logits = self._state(vlm=True)
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        invalid_model = self._VLMModel(text_config_as_mapping=True)
        invalid_model.args.text_config["linear_conv_kernel_dim"] = None
        restored = self._cache(
            model=invalid_model,
            model_identity="qualified-vlm@revision",
        )
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))

        assert restored.restore_hybrid_persistence_snapshot(loaded) == 0
        assert len(restored) == 0

    def test_ineligible_multimodal_entry_is_not_persisted(self):
        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=False,
        )
        assert prepared is not None and source.commit_prepared(prepared)

        assert source.prepare_hybrid_persistence_snapshot() is None

    def test_cache_runtime_identity_mismatch_publishes_nothing(self, tmp_path):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        source = MemoryAwarePrefixCache(
            self._Model(),
            MemoryCacheConfig(max_memory_mb=8, min_prefix_tokens=1),
            tokenizer=self._Tokenizer(),
            model_identity="qualified-model@revision",
            cache_runtime_identity={"max_kv_size": 4096},
        )
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        incompatible = MemoryAwarePrefixCache(
            self._Model(),
            MemoryCacheConfig(max_memory_mb=8, min_prefix_tokens=1),
            tokenizer=self._Tokenizer(),
            model_identity="qualified-model@revision",
            cache_runtime_identity={"max_kv_size": 8192},
        )
        loaded = incompatible.read_hybrid_persistence_snapshot(str(tmp_path))

        assert incompatible.restore_hybrid_persistence_snapshot(loaded) == 0
        assert len(incompatible) == 0

    def test_tokenizer_identity_mismatch_publishes_nothing(self, tmp_path):
        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        incompatible = self._cache(self._Tokenizer(suffix="-changed"))
        loaded = incompatible.read_hybrid_persistence_snapshot(str(tmp_path))

        assert incompatible.restore_hybrid_persistence_snapshot(loaded) == 0
        assert len(incompatible) == 0

    def test_effective_processor_template_mismatch_publishes_nothing(self, tmp_path):
        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        incompatible = self._cache(renderer=self._Renderer("-changed"))
        loaded = incompatible.read_hybrid_persistence_snapshot(str(tmp_path))

        assert incompatible.restore_hybrid_persistence_snapshot(loaded) == 0
        assert len(incompatible) == 0

    def test_mutated_local_model_artifact_publishes_nothing(self, tmp_path):
        import os

        model_dir = tmp_path / "model"
        cache_dir = tmp_path / "cache"
        model_dir.mkdir()
        marker = model_dir / "config.json"
        marker.write_text("first")
        original_times = marker.stat()
        source = self._cache(model_identity=str(model_dir))
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(cache_dir), snapshot)

        marker.write_text("other")
        os.utime(
            marker,
            ns=(original_times.st_atime_ns, original_times.st_mtime_ns),
        )
        incompatible = self._cache(model_identity=str(model_dir))
        loaded = incompatible.read_hybrid_persistence_snapshot(str(cache_dir))

        assert incompatible.restore_hybrid_persistence_snapshot(loaded) == 0
        assert len(incompatible) == 0

    def test_in_range_kv_offset_mismatch_publishes_nothing(self, tmp_path):
        from dataclasses import replace

        from vllm_mlx.cache_persistence import LoadedHybridCache

        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)
        loaded = source.read_hybrid_persistence_snapshot(str(tmp_path))
        bad_kv = replace(
            loaded.entries[0].layers[1],
            metadata={**loaded.entries[0].layers[1].metadata, "offset": 3},
        )
        bad_entry = replace(
            loaded.entries[0],
            layers=(loaded.entries[0].layers[0], bad_kv),
        )
        target = self._cache()

        assert (
            target.restore_hybrid_persistence_snapshot(
                LoadedHybridCache(loaded.identity, (bad_entry,))
            )
            == 0
        )
        assert len(target) == 0

    def test_bfloat16_last_logits_restore_exact_dtype(self, tmp_path):
        import mlx.core as mx

        source = self._cache()
        state, _ = self._state()
        logits = mx.arange(16, dtype=mx.bfloat16).reshape(1, 16)
        mx.eval(logits)
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        restored = self._cache()
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))
        assert restored.restore_hybrid_persistence_snapshot(loaded) == 1
        auxiliary = restored.fetch_exact_auxiliary([1, 2, 3, 4])

        assert auxiliary["last_logits"].dtype == mx.bfloat16
        assert mx.array_equal(auxiliary["last_logits"], logits).item()

    def test_each_required_identity_field_fails_closed(self, tmp_path):
        from vllm_mlx.cache_persistence import LoadedHybridCache

        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)
        loaded = source.read_hybrid_persistence_snapshot(str(tmp_path))

        for field in ("model", "tokenizer", "cache_layout"):
            changed_identity = dict(loaded.identity)
            changed_identity[field] = f"changed-{field}"
            incompatible = LoadedHybridCache(changed_identity, loaded.entries)
            target = self._cache()
            assert target.restore_hybrid_persistence_snapshot(incompatible) == 0
            assert len(target) == 0

    def test_invalid_second_entry_publishes_no_partial_restore(self, tmp_path):
        from dataclasses import replace

        import numpy as np

        from vllm_mlx.cache_persistence import LoadedHybridCache

        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)
        loaded = source.read_hybrid_persistence_snapshot(str(tmp_path))
        invalid_second = replace(
            loaded.entries[0],
            tokens=(5, 6, 7, 8),
            auxiliary={"last_logits": np.arange(16, dtype=np.float32)},
        )
        multi_entry = LoadedHybridCache(
            loaded.identity, (loaded.entries[0], invalid_second)
        )
        target = self._cache()

        assert target.restore_hybrid_persistence_snapshot(multi_entry) == 0
        assert len(target) == 0

    def test_restored_exact_hit_uses_logits_without_rewind_or_prefill(self, tmp_path):
        import threading
        from types import SimpleNamespace

        import mlx.core as mx

        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        source = self._cache()
        state, logits = self._state()
        prepared = source.prepare_store(
            [1, 2, 3, 4],
            state,
            auxiliary={"last_logits": logits},
            persistence_eligible=True,
        )
        assert prepared is not None and source.commit_prepared(prepared)
        snapshot = source.prepare_hybrid_persistence_snapshot()
        assert source.write_hybrid_persistence_snapshot(str(tmp_path), snapshot)

        restored = self._cache()
        loaded = restored.read_hybrid_persistence_snapshot(str(tmp_path))
        assert restored.restore_hybrid_persistence_snapshot(loaded) == 1

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.max_kv_size = 0
        generator._stats = MLLMBatchStats()
        generator._pending_error_responses = []
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        generator.prefill_step_size = 512
        generator._think_suffix_len = 0
        generator.prefix_cache = restored
        generator.model = SimpleNamespace(
            config=SimpleNamespace(image_token_index=None)
        )
        generator.language_model = MagicMock(
            side_effect=AssertionError("exact restart hit must not run prefill")
        )
        generator.sampler = lambda scores: mx.argmax(scores, axis=-1)
        generator._preprocess_request = lambda _request: None
        generator._derive_request_rope_deltas = lambda _request: None
        generator._prepare_rotating_caches = lambda _cache: True
        generator._run_chunked_text_prefill = MagicMock(
            side_effect=AssertionError("exact restart hit must not run prefill")
        )

        request = MLLMBatchRequest(
            uid=1,
            request_id="restart-exact-hit",
            prompt="prompt",
            temperature=0.0,
        )
        request.input_ids = mx.array([[1, 2, 3, 4]])
        request.is_text_only = True

        batch = generator._process_prompts([request])

        assert batch.y.tolist() == [15]
        generator.language_model.assert_not_called()
        generator._run_chunked_text_prefill.assert_not_called()
