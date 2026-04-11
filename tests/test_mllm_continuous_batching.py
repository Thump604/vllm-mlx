# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM (Multimodal Language Model) continuous batching.

These tests verify that the MLLM batch generator and scheduler work correctly
for batching multiple multimodal requests together.

Test Cases:
- Single MLLM request works correctly
- 2, 4, 8 concurrent requests with batching
- Vision cache hits/misses
- Streaming with batching
- Mixed text-only and multimodal requests
"""

import base64
import os
import tempfile
from unittest.mock import MagicMock

import pytest

# Skip all tests if MLX is not available
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# Test image (small PNG)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def create_test_image(path: str, size: tuple = (32, 32)) -> str:
    """Create a test image file."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        img.save(path)
        return path
    except ImportError:
        # Fallback: write a minimal valid PNG
        png_data = base64.b64decode(TEST_IMAGE_B64)
        with open(path, "wb") as f:
            f.write(png_data)
        return path


class TestMLLMBatchRequest:
    """Tests for MLLMBatchRequest dataclass."""

    def test_create_request(self):
        """Test creating a basic request."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=0,
            request_id="test-1",
            prompt="What's in this image?",
            images=["test.jpg"],
            max_tokens=100,
        )

        assert req.uid == 0
        assert req.request_id == "test-1"
        assert req.prompt == "What's in this image?"
        assert req.images == ["test.jpg"]
        assert req.max_tokens == 100
        assert req.num_tokens == 0
        assert req.vision_encoded is False

    def test_request_defaults(self):
        """Test default values."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=1,
            request_id="test-2",
            prompt="Hello",
        )

        assert req.images is None
        assert req.videos is None
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 0
        assert req.min_p == 0.0
        assert req.presence_penalty == 0.0
        assert req.repetition_penalty == 1.0
        assert req.stop_token_ids == []
        assert req.output_tokens == []


class TestMLLMBatchResponse:
    """Tests for MLLMBatchResponse dataclass."""

    def test_create_response(self):
        """Test creating a response."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        logprobs = mx.array([0.1, 0.2, 0.3])

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=42,
            logprobs=logprobs,
            finish_reason=None,
        )

        assert resp.uid == 0
        assert resp.request_id == "test-1"
        assert resp.token == 42
        assert resp.finish_reason is None

    def test_finished_response(self):
        """Test response with finish reason."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=2,  # EOS
            logprobs=mx.array([0.1]),
            finish_reason="stop",
        )

        assert resp.finish_reason == "stop"


class TestMLLMBatch:
    """Tests for MLLMBatch class."""

    def test_batch_length(self):
        """Test batch length calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(3)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2],
            request_ids=["req-0", "req-1", "req-2"],
            y=mx.array([100, 200, 300]),
            logprobs=[mx.array([0.1]), mx.array([0.2]), mx.array([0.3])],
            max_tokens=[100, 100, 100],
            num_tokens=[0, 0, 0],
            cache=[],
            requests=requests,
        )

        assert len(batch) == 3

    def test_batch_filter(self):
        """Test filtering a batch."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(4)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2, 3],
            request_ids=["req-0", "req-1", "req-2", "req-3"],
            y=mx.array([100, 200, 300, 400]),
            logprobs=[
                mx.array([0.1]),
                mx.array([0.2]),
                mx.array([0.3]),
                mx.array([0.4]),
            ],
            max_tokens=[100, 100, 100, 100],
            num_tokens=[0, 0, 0, 0],
            cache=[],
            requests=requests,
        )

        # Keep only indices 1 and 3
        batch.filter([1, 3])

        assert len(batch) == 2
        assert batch.uids == [1, 3]
        assert batch.request_ids == ["req-1", "req-3"]

    def test_extract_cache_clamps_negative_left_padding_for_rotating_cache(self):
        """BatchRotatingKVCache extraction must not slice from the tail."""
        from mlx_lm.generate import BatchRotatingKVCache
        from mlx_lm.models.cache import RotatingKVCache
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        batch_cache = BatchRotatingKVCache(4, [0])
        batch_cache.keys = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
        batch_cache.values = (mx.arange(4, dtype=mx.float32) + 1).reshape(1, 1, 4, 1)
        batch_cache.left_padding = mx.array([-2], dtype=mx.int32)
        batch_cache.offset = mx.array([6], dtype=mx.int32)
        batch_cache._idx = 4
        batch_cache.rotated = False

        batch = MLLMBatch(
            uids=[0],
            request_ids=["req-1"],
            y=mx.array([1], dtype=mx.int32),
            logprobs=[mx.array([0.0], dtype=mx.float32)],
            max_tokens=[8],
            num_tokens=[0],
            cache=[batch_cache],
            requests=[MLLMBatchRequest(uid=0, request_id="req-1", prompt="hi")],
        )

        extracted = batch.extract_cache(0)

        assert isinstance(extracted[0], RotatingKVCache)
        assert extracted[0].offset == 6
        assert extracted[0].keys.shape == (1, 1, 4, 1)


class TestMLLMBatchStats:
    """Tests for MLLMBatchStats."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tokens == 0
        assert stats.generation_tokens == 0
        assert stats.prompt_time == 0
        assert stats.generation_time == 0
        assert stats.num_images_processed == 0

    def test_tps_calculation(self):
        """Test tokens per second calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()
        stats.prompt_tokens = 100
        stats.prompt_time = 2.0
        stats.generation_tokens = 50
        stats.generation_time = 1.0

        assert stats.prompt_tps == 50.0
        assert stats.generation_tps == 50.0

    def test_tps_zero_time(self):
        """Test TPS with zero time."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tps == 0
        assert stats.generation_tps == 0

    def test_normalize_rotating_cache_for_merge(self):
        """Trim oversized rotating-cache tensors before batch merge."""
        from mlx_lm.models.cache import RotatingKVCache
        from vllm_mlx.mllm_batch_generator import _normalize_cache_for_merge

        cache = RotatingKVCache(max_size=1024, keep=0)
        cache.keys = mx.arange(1452, dtype=mx.float32).reshape(1, 1, 1452, 1)
        cache.values = (mx.arange(1452, dtype=mx.float32) + 1).reshape(1, 1, 1452, 1)
        cache.offset = 1452
        cache._idx = 1452

        normalized = _normalize_cache_for_merge(cache)

        assert normalized.offset == 1452
        assert normalized.size() == 1024
        assert normalized.keys.shape == (1, 1, 1024, 1)
        assert normalized.values.shape == (1, 1, 1024, 1)

        merged = normalized.merge([normalized])
        assert merged.keys.shape == (1, 1, 1024, 1)
        assert merged.values.shape == (1, 1, 1024, 1)
        assert merged.offset.tolist() == [1452]


class TestMLLMBatchGeneratorSampling:
    """Tests for request-scoped sampling on the MLLM path."""

    def test_preprocess_failure_yields_error_response_and_drains_queue(self):
        """A request whose preprocessing raises must not be retried forever.

        Regression test for the MLLM scheduler retry loop: before the fix,
        when `_preprocess_request` raised (e.g. PIL rejecting a malformed
        image), `_process_prompts` propagated the exception, `_next`
        never trimmed `unprocessed_requests`, and the outer scheduler
        process loop retried the same failing input indefinitely.

        Expected behavior after the fix:
        - `_process_prompts` catches the per-request exception and returns
          `(None, [failed_req])`.
        - `_next` always trims `unprocessed_requests`.
        - `_next` emits exactly one synthetic `MLLMBatchResponse` with
          `finish_reason="error"` so the scheduler can drain the failed
          request via its existing error-handling branch.
        """
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

        generator = MLLMBatchGenerator(
            MagicMock(), MagicMock(), enable_vision_cache=False
        )

        # Force preprocessing to fail with the same error shape PIL raises
        # on malformed 1x1 greyscale PNGs.
        def boom(request):
            raise ValueError(
                "Failed to process inputs with error: "
                "Cannot handle this data type: (1, 1, 1), |u1"
            )

        generator._preprocess_request = boom  # type: ignore[method-assign]

        req = MLLMBatchRequest(
            uid=0,
            request_id="req-bad-image",
            prompt="describe",
            images=["<malformed-png-b64>"],
        )

        # _process_prompts should isolate the failure.
        batch, failed = generator._process_prompts([req])
        assert batch is None
        assert failed == [req]

        # Now exercise the _next() path, which owns queue ownership and
        # error-response emission.
        generator.unprocessed_requests = [req]
        responses = generator._next()

        assert len(responses) == 1
        assert responses[0].uid == 0
        assert responses[0].request_id == "req-bad-image"
        assert responses[0].finish_reason == "error"

        # Queue must be drained; without this the scheduler would retry
        # the same bad input forever.
        assert generator.unprocessed_requests == []
        assert generator.active_batch is None

        generator.close()

    def test_step_uses_request_specific_samplers_and_processors(self):
        """Each active request should apply its own processors and sampler."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest

        generator = MLLMBatchGenerator(
            MagicMock(), MagicMock(), enable_vision_cache=False
        )
        generator.language_model = MagicMock(
            return_value=mx.array(
                [
                    [[0.2, 0.1, 0.0]],
                    [[0.0, 0.1, 0.2]],
                ],
                dtype=mx.float32,
            )
        )

        seen_contexts = []

        def processor_a(tokens, logits):
            seen_contexts.append(("a", list(tokens)))
            return mx.array([[0.0, 10.0, 0.0]], dtype=logits.dtype)

        def processor_b(tokens, logits):
            seen_contexts.append(("b", list(tokens)))
            return mx.array([[10.0, 0.0, 0.0]], dtype=logits.dtype)

        req_a = MLLMBatchRequest(
            uid=0,
            request_id="req-a",
            prompt="hello",
            input_ids=mx.array([11, 12]),
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            logits_processors=[processor_a],
        )
        req_a.output_tokens = [21]

        req_b = MLLMBatchRequest(
            uid=1,
            request_id="req-b",
            prompt="world",
            input_ids=mx.array([31]),
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            logits_processors=[processor_b],
        )

        sampled, logprobs = generator._step(
            mx.array([5, 6], dtype=mx.int32),
            cache=[],
            requests=[req_a, req_b],
        )

        assert sampled.tolist() == [1, 0]
        assert seen_contexts == [("a", [11, 12, 21]), ("b", [31])]
        assert logprobs[0].shape[-1] == 3
        assert logprobs[1].shape[-1] == 3

        generator.close()


class TestMLLMBatchGeneratorChunkedPrefill:
    """Tests for chunked prefill in ``MLLMBatchGenerator._run_vision_encoding``.

    Regression cover for the Session 84 Gemma 4 long-context fix: a single
    256K-token forward pass on the MLLM path was hitting Metal OOM and
    600s client timeouts at 128K. The fix splits text-only prefill into
    ``prefill_step_size``-sized chunks while preserving the per-request
    cache identity, and forces evaluation of cache state between chunks
    so the activation graph cannot grow unbounded across the loop.

    Vision (``pixel_values is not None``) requests must NOT be chunked
    because image-token placeholders need to share the same forward pass
    as their pixels for ``masked_scatter`` inside ``get_input_embeddings``.
    """

    def _build_generator(self, prefill_step_size=1024):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator(
            MagicMock(),
            MagicMock(),
            enable_vision_cache=False,
            prefill_step_size=prefill_step_size,
        )
        return generator

    def _make_fake_cache(self, n_layers=2):
        """Build a list of fake cache layers exposing a ``state`` property
        so the chunked prefill loop can call ``mx.eval([c.state for ...])``.
        """

        class _FakeCacheLayer:
            def __init__(self):
                self._state = mx.array([0], dtype=mx.int32)

            @property
            def state(self):
                return self._state

        return [_FakeCacheLayer() for _ in range(n_layers)]

    def test_short_text_only_prefill_uses_single_call(self):
        """Inputs <= prefill_step_size go through a single ``self.model`` call."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(prefill_step_size=1024)
        last_logits = mx.zeros((1, 512, 8), dtype=mx.float32)
        model_mock = MagicMock(return_value=last_logits)
        gen.model = model_mock

        req = MLLMBatchRequest(
            uid=0,
            request_id="r-short",
            prompt="hi",
            input_ids=mx.zeros(512, dtype=mx.int32),
        )

        cache = self._make_fake_cache()
        out = gen._run_vision_encoding(req, cache=cache)

        assert model_mock.call_count == 1
        called_input = model_mock.call_args_list[0].args[0]
        assert called_input.shape == (1, 512)
        assert model_mock.call_args_list[0].kwargs.get("cache") is cache
        assert req.vision_encoded is True
        assert mx.array_equal(out, last_logits).item()

        gen.close()

    def test_long_text_only_prefill_chunks_with_shared_cache(self):
        """Text-only inputs > prefill_step_size are split into chunks; the
        cache identity is preserved across calls and only the first chunk
        receives any vision-related kwargs. Returned logits == last chunk."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(prefill_step_size=1024)

        chunk_outputs = [
            mx.zeros((1, 1024, 8), dtype=mx.float32),
            mx.ones((1, 476, 8), dtype=mx.float32),
        ]
        call_log = []

        def model_call(input_ids, **kwargs):
            call_log.append(
                {
                    "input_ids_shape": input_ids.shape,
                    "cache_arg": kwargs.get("cache"),
                    "kwargs_no_cache": {
                        k: v for k, v in kwargs.items() if k != "cache"
                    },
                }
            )
            return chunk_outputs[len(call_log) - 1]

        model_mock = MagicMock(side_effect=model_call)
        gen.model = model_mock

        cache = self._make_fake_cache()
        req = MLLMBatchRequest(
            uid=0,
            request_id="r-long",
            prompt="needle",
            input_ids=mx.zeros(1500, dtype=mx.int32),
        )

        out = gen._run_vision_encoding(req, cache=cache)

        assert len(call_log) == 2
        # Chunk 0 covers tokens [0:1024], chunk 1 covers tokens [1024:1500].
        assert call_log[0]["input_ids_shape"] == (1, 1024)
        assert call_log[1]["input_ids_shape"] == (1, 476)
        # Both calls share the SAME cache list object so KV state accumulates.
        assert call_log[0]["cache_arg"] is cache
        assert call_log[1]["cache_arg"] is cache
        # Subsequent chunks must NOT carry vision kwargs (otherwise the
        # vision tower would re-run on every chunk and the chunked path
        # would be no faster than the original single-shot call).
        for forbidden in ("pixel_values", "attention_mask", "image_grid_thw"):
            assert forbidden not in call_log[1]["kwargs_no_cache"]
        # The returned logits are the LAST chunk's output (only those matter
        # for first-token sampling).
        assert mx.array_equal(out, chunk_outputs[1]).item()
        assert req.vision_encoded is True

        gen.close()

    def test_long_text_only_prefill_handles_3_chunks(self):
        """Verifies the chunk loop handles inputs that require 3+ chunks."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(prefill_step_size=1024)

        chunk_outputs = [
            mx.zeros((1, 1024, 4), dtype=mx.float32),
            mx.ones((1, 1024, 4), dtype=mx.float32),
            mx.full((1, 200, 4), 7.0, dtype=mx.float32),
        ]
        call_log = []

        def model_call(input_ids, **kwargs):
            call_log.append(input_ids.shape)
            return chunk_outputs[len(call_log) - 1]

        gen.model = MagicMock(side_effect=model_call)
        cache = self._make_fake_cache()

        req = MLLMBatchRequest(
            uid=0,
            request_id="r-3chunks",
            prompt="long needle",
            input_ids=mx.zeros(2248, dtype=mx.int32),  # 1024 + 1024 + 200
        )

        out = gen._run_vision_encoding(req, cache=cache)

        assert call_log == [(1, 1024), (1, 1024), (1, 200)]
        assert mx.array_equal(out, chunk_outputs[2]).item()

        gen.close()

    def test_image_request_skips_chunking(self):
        """Requests with pixel_values must NOT be chunked; image-token
        placeholders need to stay in the same forward pass as their pixels
        so masked_scatter inside get_input_embeddings can do its job."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(prefill_step_size=1024)
        model_mock = MagicMock(return_value=mx.zeros((1, 1500, 8), dtype=mx.float32))
        gen.model = model_mock

        req = MLLMBatchRequest(
            uid=0,
            request_id="r-img",
            prompt="describe",
            input_ids=mx.zeros(1500, dtype=mx.int32),
            pixel_values=mx.zeros((1, 3, 32, 32), dtype=mx.float32),
        )

        gen._run_vision_encoding(req, cache=None)

        # Single forward call carrying pixel_values.
        assert model_mock.call_count == 1
        kwargs = model_mock.call_args_list[0].kwargs
        assert "pixel_values" in kwargs

        gen.close()

    def test_chunk_boundary_exactly_on_step_size(self):
        """Inputs exactly equal to prefill_step_size still take a single call."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(prefill_step_size=1024)
        model_mock = MagicMock(return_value=mx.zeros((1, 1024, 4), dtype=mx.float32))
        gen.model = model_mock

        req = MLLMBatchRequest(
            uid=0,
            request_id="r-exact",
            prompt="exact",
            input_ids=mx.zeros(1024, dtype=mx.int32),
        )

        gen._run_vision_encoding(req, cache=self._make_fake_cache())

        assert model_mock.call_count == 1

        gen.close()


class TestMLLMBatchGeneratorSpecPrefill:
    """Tests for SpecPrefill integration on the MLLM batch generator path.

    Session 84 Fix 2: text-only long-context requests on the MLLM
    scheduler path can route through cooperative SpecPrefill (draft
    scoring + sparse prefill) to cut the bulk of the prefill work,
    matching what the text scheduler already does for non-MLLM models.

    Eligibility rules:
    - draft_model must be plumbed through BatchedEngine -> MLLMScheduler
      -> MLLMBatchGenerator (None disables the path)
    - specprefill_threshold + keep_pct must be set
    - request must be text-only (pixel_values is None) so vision-token
      placeholders are not dropped from the keep mask
    - input_ids length must exceed specprefill_threshold

    The result must be processed as a SOLO batch because the wrapped
    cache produced by sparse prefill carries a per-request RoPE
    adjustment that does not safely co-merge with dense-path caches.
    """

    def _build_generator(self, *, draft_model=None, threshold=None, keep_pct=None):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        return MLLMBatchGenerator(
            MagicMock(),
            MagicMock(),
            enable_vision_cache=False,
            prefill_step_size=1024,
            draft_model=draft_model,
            specprefill_threshold=threshold,
            specprefill_keep_pct=keep_pct,
        )

    def test_eligibility_requires_draft_model(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(threshold=512, keep_pct=0.3)
        req = MLLMBatchRequest(
            uid=0,
            request_id="r-no-draft",
            prompt="x",
            input_ids=mx.zeros(2048, dtype=mx.int32),
        )
        assert gen._qualifies_for_specprefill(req) is False
        gen.close()

    def test_eligibility_requires_text_only(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(
            draft_model=MagicMock(), threshold=512, keep_pct=0.3
        )
        req = MLLMBatchRequest(
            uid=0,
            request_id="r-img",
            prompt="x",
            input_ids=mx.zeros(2048, dtype=mx.int32),
            pixel_values=mx.zeros((1, 3, 8, 8), dtype=mx.float32),
        )
        assert gen._qualifies_for_specprefill(req) is False
        gen.close()

    def test_eligibility_requires_above_threshold(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(
            draft_model=MagicMock(), threshold=4096, keep_pct=0.3
        )
        req = MLLMBatchRequest(
            uid=0,
            request_id="r-short",
            prompt="x",
            input_ids=mx.zeros(1024, dtype=mx.int32),
        )
        assert gen._qualifies_for_specprefill(req) is False
        gen.close()

    def test_eligibility_passes_for_long_text_only_with_draft(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(
            draft_model=MagicMock(), threshold=512, keep_pct=0.3
        )
        req = MLLMBatchRequest(
            uid=0,
            request_id="r-long",
            prompt="x",
            input_ids=mx.zeros(2048, dtype=mx.int32),
        )
        assert gen._qualifies_for_specprefill(req) is True
        gen.close()

    def test_eligibility_capped_at_specprefill_max_input(self):
        """Inputs above ``specprefill_max_input`` must fall through to
        the dense path. The cap is the macOS GPU watchdog workaround
        for the manual_rope_proportional path on the target VLM at
        very long prompts (Session 84 Fix 2 — 256K crashes the sparse
        path on Gemma 4 26B regardless of target_chunk_size)."""
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
        )

        gen = MLLMBatchGenerator(
            MagicMock(),
            MagicMock(),
            enable_vision_cache=False,
            prefill_step_size=1024,
            draft_model=MagicMock(),
            specprefill_threshold=512,
            specprefill_keep_pct=0.5,
            specprefill_max_input=4096,
        )

        below_cap = MLLMBatchRequest(
            uid=0,
            request_id="r-below",
            prompt="x",
            input_ids=mx.zeros(2048, dtype=mx.int32),
        )
        at_cap = MLLMBatchRequest(
            uid=1,
            request_id="r-at",
            prompt="x",
            input_ids=mx.zeros(4096, dtype=mx.int32),
        )
        above_cap = MLLMBatchRequest(
            uid=2,
            request_id="r-above",
            prompt="x",
            input_ids=mx.zeros(8192, dtype=mx.int32),
        )

        assert gen._qualifies_for_specprefill(below_cap) is True
        # Equal to cap is allowed (we use ``> max``, not ``>=``).
        assert gen._qualifies_for_specprefill(at_cap) is True
        assert gen._qualifies_for_specprefill(above_cap) is False

        gen.close()

    def test_eligibility_no_cap_when_max_input_none(self):
        """Setting ``specprefill_max_input=None`` disables the cap.
        Used for tests/benchmarks that want to exercise the sparse
        path at any size."""
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
        )

        gen = MLLMBatchGenerator(
            MagicMock(),
            MagicMock(),
            enable_vision_cache=False,
            draft_model=MagicMock(),
            specprefill_threshold=512,
            specprefill_keep_pct=0.5,
            specprefill_max_input=None,
        )
        huge = MLLMBatchRequest(
            uid=0,
            request_id="r-huge",
            prompt="x",
            input_ids=mx.zeros(1_000_000, dtype=mx.int32),
        )
        assert gen._qualifies_for_specprefill(huge) is True
        gen.close()

    def test_run_specprefill_drives_session_to_completion(self, monkeypatch):
        """``_run_specprefill_for_request`` should construct a
        cooperative session against the runtime's draft + target model,
        drive ``step()`` until done, and return the session's logits +
        wrapped cache."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest
        import vllm_mlx.mllm_batch_generator as mbg

        constructed = []
        steps_called = [0]
        cleanup_called = [0]
        fake_logits = mx.zeros((1, 1, 16), dtype=mx.float32)
        fake_cache = [object(), object()]

        class _FakeSession:
            def __init__(self, **kwargs):
                constructed.append(kwargs)
                self._done = False

            @property
            def is_done(self):
                return self._done

            def step(self):
                steps_called[0] += 1
                if steps_called[0] >= 2:
                    self._done = True
                return self._done

            def finalize(self):
                from vllm_mlx.cooperative_specprefill import (
                    CooperativeSpecPrefillResult,
                )

                return CooperativeSpecPrefillResult(
                    logits=fake_logits,
                    cache=fake_cache,
                    cache_token_count=512,
                    selected_token_count=512,
                )

            def cleanup(self):
                cleanup_called[0] += 1

        monkeypatch.setattr(mbg, "CooperativeSpecPrefillSession", _FakeSession)

        draft_mock = MagicMock(name="draft_model")
        gen = self._build_generator(draft_model=draft_mock, threshold=512, keep_pct=0.3)

        req = MLLMBatchRequest(
            uid=0,
            request_id="r-long",
            prompt="needle",
            input_ids=mx.zeros(2048, dtype=mx.int32),
        )

        logits, wrapped_cache = gen._run_specprefill_for_request(req)

        assert len(constructed) == 1
        kwargs = constructed[0]
        assert kwargs["draft_model"] is draft_mock
        assert kwargs["model"] is gen.model
        assert kwargs["keep_pct"] == 0.3
        # Tokens should be a Python list, not an mx.array, per session API.
        assert isinstance(kwargs["tokens"], list)
        assert len(kwargs["tokens"]) == 2048

        assert steps_called[0] >= 2
        assert cleanup_called[0] == 1
        assert mx.array_equal(logits, fake_logits).item()
        assert wrapped_cache is fake_cache

        gen.close()

    def test_run_specprefill_unwraps_language_model_output(self, monkeypatch):
        """Gemma 4 / Qwen 3.5 VLM language models return a wrapper
        object whose ``.logits`` attribute is the actual mx.array.
        ``_run_specprefill_for_request`` must unwrap before returning
        so the caller can subscript ``logits[:, -1, :]`` for first
        token sampling."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest
        import vllm_mlx.mllm_batch_generator as mbg

        inner_logits = mx.zeros((1, 1, 16), dtype=mx.float32)

        class _LanguageModelOutput:
            def __init__(self, logits):
                self.logits = logits

        wrapped_logits = _LanguageModelOutput(inner_logits)
        fake_cache = [object()]

        class _FakeSession:
            def __init__(self, **kwargs):
                self._done = False

            @property
            def is_done(self):
                return self._done

            def step(self):
                self._done = True
                return True

            def finalize(self):
                from vllm_mlx.cooperative_specprefill import (
                    CooperativeSpecPrefillResult,
                )

                return CooperativeSpecPrefillResult(
                    logits=wrapped_logits,
                    cache=fake_cache,
                    cache_token_count=512,
                    selected_token_count=512,
                )

            def cleanup(self):
                pass

        monkeypatch.setattr(mbg, "CooperativeSpecPrefillSession", _FakeSession)

        gen = self._build_generator(
            draft_model=MagicMock(name="draft"),
            threshold=512,
            keep_pct=0.3,
        )
        req = MLLMBatchRequest(
            uid=0,
            request_id="r-wrapped",
            prompt="needle",
            input_ids=mx.zeros(2048, dtype=mx.int32),
        )

        logits, cache = gen._run_specprefill_for_request(req)

        # The wrapper must be unwrapped: ``logits`` is the inner
        # mx.array, not the LanguageModelOutput. This is the bit that
        # must hold so the caller can do ``logits[:, -1, :]``.
        assert isinstance(logits, mx.array)
        assert logits is inner_logits
        assert cache is fake_cache
        # Sanity: the unwrapped logits actually subscript.
        last_token_slice = logits[:, -1, :]
        assert last_token_slice.shape == (1, 16)

        gen.close()

    def test_process_prompts_routes_eligible_request_through_specprefill(
        self, monkeypatch
    ):
        """``_process_prompts`` with one SpecPrefill-eligible request must
        produce a single-row MLLMBatch whose cache comes from the session,
        bypassing the regular vision encoding + merge path."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        gen = self._build_generator(
            draft_model=MagicMock(name="draft"),
            threshold=512,
            keep_pct=0.3,
        )

        # Skip preprocessing — just install the input_ids directly so the
        # request looks pre-tokenized to the eligibility check.
        gen._preprocess_request = lambda req: None  # type: ignore[method-assign]

        fake_logits = mx.zeros((1, 1, 8), dtype=mx.float32)
        fake_wrapped_cache = [object(), object()]

        run_calls = []

        def fake_run_specprefill(req):
            run_calls.append(req)
            return fake_logits, fake_wrapped_cache

        gen._run_specprefill_for_request = fake_run_specprefill  # type: ignore[method-assign]

        # Spy: dense path must NOT be entered.
        gen._run_vision_encoding = MagicMock(
            side_effect=AssertionError("dense path should not run"),
        )

        # Sample helper produces a deterministic first token.
        def fake_sample(req, logits):
            return mx.array([99]), mx.array([[0.0] * 8])

        gen._sample_request = fake_sample  # type: ignore[method-assign]

        req = MLLMBatchRequest(
            uid=42,
            request_id="r-eligible",
            prompt="needle",
            input_ids=mx.zeros(2048, dtype=mx.int32),
        )

        batch, failed = gen._process_prompts([req])

        assert failed == []
        assert isinstance(batch, MLLMBatch)
        assert len(batch) == 1
        assert batch.uids == [42]
        assert batch.cache is fake_wrapped_cache
        assert batch.y.tolist() == [99]
        assert len(run_calls) == 1
        assert run_calls[0] is req

        gen.close()

    def test_process_prompts_isolates_eligible_and_defers_other_requests(
        self, monkeypatch
    ):
        """When a slice of requests includes a SpecPrefill-eligible one
        alongside short ones, the eligible request is processed solo
        and the other requests are pushed BACK to the unprocessed queue
        head so a future ``_next`` call can batch them normally."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        gen = self._build_generator(
            draft_model=MagicMock(name="draft"),
            threshold=512,
            keep_pct=0.3,
        )
        gen._preprocess_request = lambda req: None  # type: ignore[method-assign]

        fake_logits = mx.zeros((1, 1, 8), dtype=mx.float32)
        fake_wrapped_cache = [object()]
        gen._run_specprefill_for_request = (  # type: ignore[method-assign]
            lambda req: (fake_logits, fake_wrapped_cache)
        )
        gen._run_vision_encoding = MagicMock(
            side_effect=AssertionError("dense path should not run"),
        )
        gen._sample_request = lambda req, logits: (  # type: ignore[method-assign]
            mx.array([7]),
            mx.array([[0.0] * 8]),
        )

        short_a = MLLMBatchRequest(
            uid=1,
            request_id="r-short-a",
            prompt="hi",
            input_ids=mx.zeros(64, dtype=mx.int32),
        )
        long_b = MLLMBatchRequest(
            uid=2,
            request_id="r-long-b",
            prompt="needle",
            input_ids=mx.zeros(2048, dtype=mx.int32),
        )
        short_c = MLLMBatchRequest(
            uid=3,
            request_id="r-short-c",
            prompt="hello",
            input_ids=mx.zeros(64, dtype=mx.int32),
        )

        batch, failed = gen._process_prompts([short_a, long_b, short_c])

        # Solo batch: only the eligible request makes it through.
        assert failed == []
        assert batch is not None
        assert batch.uids == [2]
        # Short requests must be back on the queue (head order preserved
        # so they batch together on the next _next call).
        assert [r.request_id for r in gen.unprocessed_requests] == [
            "r-short-a",
            "r-short-c",
        ]

        gen.close()


class TestMLLMSchedulerSpecPrefillConfig:
    """Tests for the SpecPrefill plumbing across MLLMScheduler ->
    MLLMBatchGenerator. Verifies the config flows from constructor
    args through to the batch generator that actually decides
    eligibility."""

    def test_scheduler_forwards_specprefill_config_to_generator(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        draft_mock = MagicMock(name="draft")
        scheduler = MLLMScheduler(
            MagicMock(),
            MagicMock(),
            config=MLLMSchedulerConfig(),
            draft_model=draft_mock,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
        )
        scheduler._ensure_batch_generator()

        gen = scheduler.batch_generator
        assert gen is not None
        assert gen._draft_model is draft_mock
        assert gen._specprefill_threshold == 8192
        assert gen._specprefill_keep_pct == 0.3

    def test_scheduler_defaults_specprefill_to_disabled(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler(MagicMock(), MagicMock())
        scheduler._ensure_batch_generator()

        gen = scheduler.batch_generator
        assert gen is not None
        assert gen._draft_model is None
        assert gen._specprefill_threshold is None
        assert gen._specprefill_keep_pct is None


class TestMLLMSchedulerConfig:
    """Tests for MLLMSchedulerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig()

        assert config.max_num_seqs == 16
        # prefill_batch_size set equal to max_num_seqs to avoid batch extend issues
        assert config.prefill_batch_size == 16
        assert config.completion_batch_size == 16
        assert config.enable_vision_cache is True
        assert config.vision_cache_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig(
            max_num_seqs=8,
            prefill_batch_size=2,
            completion_batch_size=8,
            enable_vision_cache=False,
        )

        assert config.max_num_seqs == 8
        assert config.prefill_batch_size == 2
        assert config.completion_batch_size == 8
        assert config.enable_vision_cache is False


class TestMLLMRequest:
    """Tests for MLLMRequest dataclass."""

    def test_create_request(self):
        """Test creating an MLLM request."""
        from vllm_mlx.mllm_scheduler import MLLMRequest
        from vllm_mlx.request import RequestStatus

        req = MLLMRequest(
            request_id="req-1",
            prompt="Describe this image",
            images=["image.jpg"],
        )

        assert req.request_id == "req-1"
        assert req.prompt == "Describe this image"
        assert req.images == ["image.jpg"]
        assert req.status == RequestStatus.WAITING
        assert req.output_text == ""

    def test_add_request_preserves_full_sampling_params(self):
        """Scheduler should retain the full sampling config for MLLM requests."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler(MagicMock(), MagicMock())
        request_id = scheduler.add_request(
            prompt="Describe this image",
            max_tokens=64,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.05,
            presence_penalty=1.5,
            repetition_penalty=1.1,
            stop=["DONE"],
            stop_token_ids=[42],
            request_id="req-full",
        )

        req = scheduler.requests[request_id]
        assert req.sampling_params.max_tokens == 64
        assert req.sampling_params.temperature == 1.0
        assert req.sampling_params.top_p == 0.95
        assert req.sampling_params.top_k == 20
        assert req.sampling_params.min_p == 0.05
        assert req.sampling_params.presence_penalty == 1.5
        assert req.sampling_params.repetition_penalty == 1.1
        assert req.sampling_params.stop == ["DONE"]
        assert req.sampling_params.stop_token_ids == [42]

    def test_schedule_waiting_builds_request_scoped_sampling_components(self):
        """Scheduled batch requests should carry their own sampler/processors."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler(MagicMock(), MagicMock())
        captured_batch_reqs = []
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.insert.side_effect = (
            lambda reqs: captured_batch_reqs.extend(reqs) or [99]
        )

        scheduler.add_request(
            prompt="Describe this image",
            top_k=20,
            min_p=0.05,
            presence_penalty=1.5,
            repetition_penalty=1.2,
            stop_token_ids=[7, 9],
            request_id="req-scheduled",
        )

        scheduled = scheduler._schedule_waiting()

        assert [req.request_id for req in scheduled] == ["req-scheduled"]
        assert len(captured_batch_reqs) == 1
        batch_req = captured_batch_reqs[0]
        assert batch_req.top_k == 20
        assert batch_req.min_p == 0.05
        assert batch_req.presence_penalty == 1.5
        assert batch_req.repetition_penalty == 1.2
        assert batch_req.stop_token_ids == [7, 9]
        assert callable(batch_req.sampler)
        assert batch_req.logits_processors is not None


class TestMLLMSchedulerOutput:
    """Tests for MLLMSchedulerOutput."""

    def test_empty_output(self):
        """Test empty scheduler output."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerOutput

        output = MLLMSchedulerOutput()

        assert output.scheduled_request_ids == []
        assert output.num_scheduled_tokens == 0
        assert output.finished_request_ids == set()
        assert output.outputs == []
        assert output.has_work is False


class TestMultimodalProcessorBatch:
    """Tests for MultimodalProcessor batch methods."""

    def test_batch_pixel_values_empty(self):
        """Test batching empty pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        # Create mock processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        result = processor.batch_pixel_values([None, None])
        assert result is None

    def test_batch_pixel_values_single(self):
        """Test batching single pixel value."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        result = processor.batch_pixel_values([pixels])

        assert result is not None
        assert result.shape == (1, 3, 32, 32)

    def test_batch_pixel_values_multiple(self):
        """Test batching multiple pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels1 = mx.ones((1, 3, 32, 32))
        pixels2 = mx.ones((1, 3, 32, 32)) * 2

        result = processor.batch_pixel_values([pixels1, pixels2])

        assert result is not None
        assert result.shape == (2, 3, 32, 32)

    def test_batch_image_grid_thw(self):
        """Test batching image grid thw."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        grid1 = mx.array([[1, 4, 4]])
        grid2 = mx.array([[1, 8, 8]])

        result = processor.batch_image_grid_thw([grid1, grid2])

        assert result is not None
        assert result.shape[0] == 2

    def test_prepare_for_batch(self):
        """Test prepare_for_batch method."""
        from vllm_mlx.multimodal_processor import (
            MultimodalProcessor,
            ProcessedMultimodalInput,
        )

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        # Create processed inputs
        inputs = [
            ProcessedMultimodalInput(
                input_ids=mx.array([1, 2, 3]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=3,
            ),
            ProcessedMultimodalInput(
                input_ids=mx.array([4, 5, 6, 7, 8]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=5,
            ),
        ]

        input_ids, batch_kwargs, padding = processor.prepare_for_batch(inputs)

        # Check left-padding
        assert input_ids.shape == (2, 5)  # max length is 5
        assert padding == [2, 0]  # first input needs 2 padding

    def test_compute_vision_hash(self):
        """Test vision hash computation."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        hash1 = processor.compute_vision_hash(pixels)
        hash2 = processor.compute_vision_hash(pixels)

        # Same input should give same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars


class TestVisionCache:
    """Tests for VLM cache functionality."""

    def test_cache_creation(self):
        """Test VLM cache creation."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=10)

        assert len(cache) == 0
        assert cache.max_size == 10

    def test_cache_miss(self):
        """Test cache miss."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is None
        assert hit is False
        assert cache.stats.misses == 1

    def test_cache_store_and_fetch(self):
        """Test storing and fetching from cache."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        # Store cache
        test_cache = [{"key": "value"}]
        cache.store_cache(["image.jpg"], "prompt", test_cache, num_tokens=100)

        # Fetch cache
        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is not None
        assert hit is True
        assert cache.stats.hits == 1
        assert cache.stats.tokens_saved == 100

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=2)

        # Fill cache
        cache.store_cache(["img1.jpg"], "prompt1", [1], num_tokens=10)
        cache.store_cache(["img2.jpg"], "prompt2", [2], num_tokens=20)

        assert len(cache) == 2

        # Add one more (should evict oldest)
        cache.store_cache(["img3.jpg"], "prompt3", [3], num_tokens=30)

        assert len(cache) == 2
        assert cache.stats.evictions == 1

        # img1 should be evicted
        _, hit = cache.fetch_cache(["img1.jpg"], "prompt1")
        assert hit is False


# Integration tests (require model loading)
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="Slow tests disabled")
class TestMLLMSchedulerIntegration:
    """Integration tests for MLLMScheduler with real models."""

    @pytest.fixture
    def test_image_path(self):
        """Create a test image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = create_test_image(f.name)
            yield path
            os.unlink(path)

    async def test_single_request(self, test_image_path):
        """Test single MLLM request."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        # Load a small model
        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = scheduler.add_request(
                prompt="What's in this image?",
                images=[test_image_path],
                max_tokens=50,
            )

            # Run until complete
            while scheduler.has_requests():
                output = scheduler.step()
                if request_id in output.finished_request_ids:
                    break

            # Check result
            request = scheduler.get_request(request_id)
            assert request is not None
            assert len(request.output_tokens) > 0

        finally:
            await scheduler.stop()

    async def test_concurrent_requests(self, test_image_path):
        """Test multiple concurrent MLLM requests."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            # Add multiple requests
            request_ids = []
            for i in range(4):
                req_id = scheduler.add_request(
                    prompt=f"Describe image {i}",
                    images=[test_image_path],
                    max_tokens=30,
                )
                request_ids.append(req_id)

            # Run until all complete
            finished = set()
            while len(finished) < len(request_ids):
                output = scheduler.step()
                finished.update(output.finished_request_ids)

            # Check all completed
            assert len(finished) == 4

            # Check stats show batching
            stats = scheduler.get_stats()
            assert stats["num_requests_processed"] == 4

        finally:
            await scheduler.stop()

    async def test_streaming(self, test_image_path):
        """Test streaming MLLM generation."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig()
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = await scheduler.add_request_async(
                prompt="Describe this image briefly",
                images=[test_image_path],
                max_tokens=30,
            )

            tokens_received = 0
            async for output in scheduler.stream_outputs(request_id):
                tokens_received += len(output.new_token_ids)
                if output.finished:
                    break

            assert tokens_received > 0

        finally:
            await scheduler.stop()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
