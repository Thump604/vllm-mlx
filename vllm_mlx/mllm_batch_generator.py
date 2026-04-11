# SPDX-License-Identifier: Apache-2.0
"""
MLLM Batch Generator for multimodal continuous batching.

This module implements continuous batching for Multimodal Language Models (MLLMs)
like Qwen3-VL, following the same architecture as LLM continuous batching but
adapted for vision models.

Key insight: VLM models have a `model.language_model` which is a standard LLM.
After the initial forward pass with vision encoding, text generation uses only
the language model - which CAN be batched using the same BatchKVCache pattern.

Architecture:
1. Vision inputs are processed per-request (not batched)
2. Initial VLM forward pass extracts cross-attention states / encoder outputs
3. Language model generation is batched using BatchKVCache (like LLM batching)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .cooperative_specprefill import CooperativeSpecPrefillSession
from .multimodal_processor import MultimodalProcessor
from .vision_embedding_cache import VisionEmbeddingCache

logger = logging.getLogger(__name__)


def _validate_caches_mergeable(per_request_caches: List[List[Any]]) -> None:
    """Validate that all cache layers support merge() for batch creation."""
    for layer_idx, layer_cache in enumerate(per_request_caches[0]):
        if not hasattr(layer_cache, "merge"):
            raise ValueError(
                f"MLLM continuous batching requires mergeable cache types "
                f"but layer {layer_idx} has {type(layer_cache).__name__} "
                f"which lacks a merge() method."
            )


def _normalize_cache_for_merge(layer_cache: Any) -> Any:
    """Trim rotating caches to their retained temporal window before merge()."""
    try:
        from mlx_lm.models.cache import RotatingKVCache
    except ImportError:
        return layer_cache

    if not isinstance(layer_cache, RotatingKVCache):
        return layer_cache
    if layer_cache.keys is None or layer_cache.values is None:
        return layer_cache

    retained = layer_cache.size()
    current = layer_cache.keys.shape[2]
    if retained <= 0 or current <= retained:
        return layer_cache

    ordered_keys = layer_cache._temporal_order(layer_cache.keys)
    ordered_values = layer_cache._temporal_order(layer_cache.values)

    normalized = RotatingKVCache(max_size=layer_cache.max_size, keep=layer_cache.keep)
    normalized.keys = mx.contiguous(ordered_keys[..., -retained:, :])
    normalized.values = mx.contiguous(ordered_values[..., -retained:, :])
    normalized.offset = layer_cache.offset
    normalized._idx = retained

    logger.info(
        "Normalized %s for merge: offset=%s retained=%s tensor_len=%s",
        type(layer_cache).__name__,
        layer_cache.offset,
        retained,
        current,
    )

    return normalized


def _flatten_token_ids(input_ids: Optional[mx.array]) -> List[int]:
    """Convert request input IDs into a flat Python token list."""
    if input_ids is None:
        return []

    tokens = input_ids.tolist()
    if not isinstance(tokens, list):
        return [int(tokens)]

    flattened: List[int] = []
    stack: List[Any] = list(tokens)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = list(item) + stack
        else:
            flattened.append(int(item))
    return flattened


@dataclass
class MLLMBatchRequest:
    """
    Request data for MLLM batch processing.

    Contains all information needed to process a multimodal request
    within the batch generator.
    """

    uid: int  # Unique identifier within the batch generator
    request_id: str  # External request ID
    prompt: str  # Text prompt
    images: Optional[List[str]] = None  # Image paths/URLs/base64
    videos: Optional[List[str]] = None  # Video inputs
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)
    sampler: Optional[Callable[[mx.array], mx.array]] = None
    logits_processors: Optional[List[Callable[[List[int], mx.array], mx.array]]] = None

    # Processed inputs (set after vision preprocessing)
    input_ids: Optional[mx.array] = None
    prompt_token_ids: List[int] = field(default_factory=list)
    pixel_values: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None
    image_grid_thw: Optional[mx.array] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Generation state
    num_tokens: int = 0  # Tokens generated so far
    output_tokens: List[int] = field(default_factory=list)

    # Vision state (populated after initial VLM forward pass)
    vision_encoded: bool = False
    cross_attention_states: Optional[Any] = None  # For models that use cross-attention
    encoder_outputs: Optional[Any] = None  # For encoder-decoder models


@dataclass
class MLLMBatchResponse:
    """
    Response from a batch generation step.

    Contains the generated token and metadata for a single request.
    """

    uid: int  # Batch generator UID
    request_id: str  # External request ID
    token: int  # Generated token
    logprobs: mx.array  # Log probabilities
    finish_reason: Optional[str] = None  # "stop", "length", or None
    prompt_cache: Optional[Callable[[], List[Any]]] = None  # Cache extraction function


@dataclass
class MLLMBatch:
    """
    Represents an active batch of MLLM requests.

    Manages the batch state including tokens, caches, and metadata
    for all requests being processed together.
    """

    uids: List[int]
    request_ids: List[str]
    y: mx.array  # Current token(s) for each request [batch_size]
    logprobs: List[mx.array]  # Log probs for each request
    max_tokens: List[int]  # Max tokens per request
    num_tokens: List[int]  # Tokens generated per request
    cache: List[Any]  # BatchKVCache for language model
    requests: List[MLLMBatchRequest]  # Full request data

    def __len__(self) -> int:
        return len(self.uids)

    def filter(self, keep_idx: List[int]) -> None:
        """
        Filter batch to keep only requests at specified indices.

        Args:
            keep_idx: Indices of requests to keep
        """
        self.uids = [self.uids[k] for k in keep_idx]
        self.request_ids = [self.request_ids[k] for k in keep_idx]
        self.logprobs = [self.logprobs[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        self.requests = [self.requests[k] for k in keep_idx]

        keep_idx_array = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx_array]

        # Filter cache entries
        for c in self.cache:
            if hasattr(c, "filter"):
                c.filter(keep_idx_array)

    def extend(self, other: "MLLMBatch") -> None:
        """
        Extend this batch with another batch.

        Args:
            other: Batch to merge into this one
        """
        self.uids.extend(other.uids)
        self.request_ids.extend(other.request_ids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs.extend(other.logprobs)
        self.num_tokens.extend(other.num_tokens)
        self.max_tokens.extend(other.max_tokens)
        self.requests.extend(other.requests)

        # Extend cache - handle None and incompatible caches
        for c, o in zip(self.cache, other.cache):
            if c is not None and o is not None and hasattr(c, "extend"):
                try:
                    # Only extend if both caches have valid keys
                    if (
                        hasattr(c, "keys")
                        and c.keys is not None
                        and hasattr(o, "keys")
                        and o.keys is not None
                    ):
                        c.extend(o)
                except Exception as e:
                    logger.warning(f"Failed to extend cache: {e}")

    def extract_cache(self, idx: int) -> List[Any]:
        """
        Extract cache for a single request (for prefix caching).

        Handles BatchRotatingKVCache negative left_padding bug: during
        generation with rotation, left_padding can become negative, which
        would make the default extract() path slice from the tail.
        """
        from mlx_lm.generate import BatchRotatingKVCache
        from mlx_lm.models.cache import RotatingKVCache

        result = []
        for c in self.cache:
            if not hasattr(c, "extract"):
                result.append(None)
            elif isinstance(c, BatchRotatingKVCache):
                cache = RotatingKVCache(c.max_size)
                padding = max(0, c.left_padding[idx].item())
                offset = c.offset[idx].item()
                cache.keys = c.keys[idx : idx + 1]
                cache.values = c.values[idx : idx + 1]
                cache._idx = c._idx
                if c.rotated:
                    cache.keys = mx.roll(cache.keys, -c._idx, axis=2)
                    cache.values = mx.roll(cache.values, -c._idx, axis=2)
                    cache._idx = c.max_size
                cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
                cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
                cache.offset = offset
                cache._idx = cache.keys.shape[2]
                cache.step = getattr(c, "step", c.max_size)
                cache.keep = getattr(c, "keep", 0)
                result.append(cache)
            else:
                result.append(c.extract(idx))
        return result


class MLLMBatchStats:
    """Statistics for MLLM batch generation."""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.prompt_time: float = 0
        self.generation_tokens: int = 0
        self.generation_time: float = 0
        self.vision_encoding_time: float = 0
        self.num_images_processed: int = 0
        self.peak_memory: float = 0

    @property
    def prompt_tps(self) -> float:
        if self.prompt_time == 0:
            return 0
        return self.prompt_tokens / self.prompt_time

    @property
    def generation_tps(self) -> float:
        if self.generation_time == 0:
            return 0
        return self.generation_tokens / self.generation_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_time": self.prompt_time,
            "prompt_tps": self.prompt_tps,
            "generation_tokens": self.generation_tokens,
            "generation_time": self.generation_time,
            "generation_tps": self.generation_tps,
            "vision_encoding_time": self.vision_encoding_time,
            "num_images_processed": self.num_images_processed,
            "peak_memory": self.peak_memory,
        }


def _make_batch_cache(model: nn.Module, left_padding: List[int]) -> List[Any]:
    """
    Create batch-aware KV cache for the language model.

    Args:
        model: The language model (model.language_model from VLM)
        left_padding: Padding amounts for left-padded prompts

    Returns:
        List of batch-aware cache objects for each layer
    """
    from mlx_lm.models.cache import (
        ArraysCache,
        BatchKVCache,
        BatchRotatingKVCache,
        CacheList,
        KVCache,
        RotatingKVCache,
    )

    def to_batch_cache(c):
        if type(c) is KVCache:
            return BatchKVCache(left_padding)
        elif isinstance(c, ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError(
                    "RotatingKVCache with keep tokens is not supported "
                    "in MLLM continuous batching."
                )
            return BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, CacheList):
            return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(
                f"MLLM continuous batching does not support {type(c).__name__}. "
                f"Supported: KVCache, ArraysCache, RotatingKVCache, CacheList."
            )

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        return [BatchKVCache(left_padding) for _ in model.layers]


def _left_pad_prompts(
    prompts: List[List[int]], max_length: Optional[int] = None
) -> mx.array:
    """
    Left-pad prompts to uniform length.

    Args:
        prompts: List of token lists
        max_length: Target length (computed if not provided)

    Returns:
        Padded prompts as mx.array [batch_size, seq_len]
    """
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([[0] * (max_length - len(p)) + list(p) for p in prompts])


class MLLMBatchGenerator:
    """
    Batch generator for Vision Language Models.

    This class manages continuous batching for MLLM requests:

    1. Vision Encoding Phase:
       - Process images/videos through vision encoder (per-request)
       - Extract vision features and merge with text embeddings
       - Store cross-attention states for language model

    2. Language Generation Phase:
       - Use language model with BatchKVCache for batched generation
       - Generate tokens for all requests simultaneously
       - Same pattern as LLM BatchGenerator

    Example:
        >>> generator = MLLMBatchGenerator(model, processor)
        >>> uids = generator.insert([request1, request2])
        >>> while responses := generator.next():
        ...     for resp in responses:
        ...         print(f"Request {resp.request_id}: token={resp.token}")
    """

    # Generation stream for async eval
    _stream = None

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        mm_processor: Optional[MultimodalProcessor] = None,
        max_tokens: int = 256,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        prefill_batch_size: int = 4,  # Smaller for MLLM due to vision overhead
        completion_batch_size: int = 16,  # Can be larger for text generation
        prefill_step_size: int = 1024,
        enable_vision_cache: bool = True,
        vision_cache_size: int = 100,
        draft_model: Optional[nn.Module] = None,
        specprefill_threshold: Optional[int] = None,
        specprefill_keep_pct: Optional[float] = None,
        specprefill_max_input: Optional[int] = 131072,
    ):
        """
        Initialize MLLM batch generator.

        Args:
            model: The VLM model (must have model.language_model)
            processor: The VLM processor for tokenization and image processing
            mm_processor: Optional MultimodalProcessor for input preparation
            max_tokens: Default max tokens per request
            stop_tokens: Set of stop token IDs
            sampler: Sampling function (default: argmax)
            prefill_batch_size: Max requests to prefill together
            completion_batch_size: Max requests for completion batching
            prefill_step_size: Tokens to process per prefill step
            enable_vision_cache: Enable vision embedding caching
            vision_cache_size: Max entries in vision cache
        """
        self.model = model
        self.processor = processor
        self.mm_processor = mm_processor

        # Get language model for text generation
        self.language_model = getattr(model, "language_model", model)

        # Check if this is actually a VLM with separate language model
        self.is_vlm = hasattr(model, "language_model")
        if self.is_vlm:
            logger.info(
                "MLLMBatchGenerator: Using VLM's language_model for batched generation"
            )
        else:
            logger.warning(
                "MLLMBatchGenerator: Model does not have language_model, using model directly"
            )

        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.prefill_step_size = prefill_step_size

        # SpecPrefill configuration. Session 84 Fix 2: text-only long
        # prompts on the MLLM scheduler path can route through
        # cooperative SpecPrefill (draft scoring + sparse prefill) to
        # cut prefill work, matching the text scheduler path. Disabled
        # by leaving draft_model None or threshold/keep_pct None.
        # ``specprefill_max_input`` caps eligibility above which the
        # request falls through to the dense chunked path; this is
        # the macOS GPU watchdog workaround for the manual_rope path
        # at very long prompts.
        self._draft_model = draft_model
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct
        self._specprefill_max_input = specprefill_max_input

        # Request management
        self.unprocessed_requests: List[MLLMBatchRequest] = []
        self.active_batch: Optional[MLLMBatch] = None
        self.uid_counter = 0

        # Statistics
        self._stats = MLLMBatchStats()

        # Vision embedding cache for repeated images
        self.vision_cache = VisionEmbeddingCache(
            max_pixel_entries=vision_cache_size,
            max_encoding_entries=vision_cache_size // 2,
            enabled=enable_vision_cache,
        )
        if enable_vision_cache:
            logger.info(
                f"MLLMBatchGenerator: Vision cache enabled (size={vision_cache_size})"
            )

        # Generation stream
        if MLLMBatchGenerator._stream is None:
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())

        # Memory management
        self._old_wired_limit = None
        if mx.metal.is_available():
            self._old_wired_limit = mx.set_wired_limit(
                mx.device_info()["max_recommended_working_set_size"]
            )

    @staticmethod
    def _request_context_tokens(request: MLLMBatchRequest) -> List[int]:
        """Return prompt+generated tokens for request-scoped penalties."""
        if not request.prompt_token_ids and request.input_ids is not None:
            request.prompt_token_ids = _flatten_token_ids(request.input_ids)
        tokens = list(request.prompt_token_ids)
        if request.output_tokens:
            tokens.extend(int(token) for token in request.output_tokens)
        return tokens

    def _sample_request(
        self, request: MLLMBatchRequest, logits: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Apply request-specific processors and sampler to one row of logits."""
        if request.logits_processors:
            context_tokens = self._request_context_tokens(request)
            for processor in request.logits_processors:
                logits = processor(context_tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampler = request.sampler or self.sampler
        sampled = sampler(logprobs)
        return sampled, logprobs

    def close(self) -> None:
        """Release resources and reset wired limit."""
        if self._old_wired_limit is not None:
            mx.synchronize(MLLMBatchGenerator._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def insert(
        self,
        requests: List[MLLMBatchRequest],
    ) -> List[int]:
        """
        Insert requests for batch processing.

        Args:
            requests: List of MLLMBatchRequest to process

        Returns:
            List of UIDs assigned to requests
        """
        uids = []
        for req in requests:
            req.uid = self.uid_counter
            self.uid_counter += 1
            self.unprocessed_requests.append(req)
            uids.append(req.uid)

        # Sort by estimated complexity (no images = simpler)
        self.unprocessed_requests = sorted(
            self.unprocessed_requests,
            key=lambda x: (
                0 if not x.images and not x.videos else 1,
                len(x.images or []) + len(x.videos or []),
            ),
        )

        logger.debug(f"Inserted {len(requests)} requests, UIDs: {uids}")
        return uids

    def remove(self, uids: List[int]) -> None:
        """
        Remove requests from processing.

        Args:
            uids: List of UIDs to remove
        """
        uid_set = set(uids)

        # Remove from active batch
        if self.active_batch is not None:
            keep_idx = [
                i for i, uid in enumerate(self.active_batch.uids) if uid not in uid_set
            ]
            if keep_idx:
                self.active_batch.filter(keep_idx)
            else:
                self.active_batch = None

        # Remove from unprocessed
        self.unprocessed_requests = [
            r for r in self.unprocessed_requests if r.uid not in uid_set
        ]

    def _preprocess_request(self, request: MLLMBatchRequest) -> None:
        """
        Preprocess a single MLLM request (vision encoding).

        This prepares the inputs by:
        1. Processing images/videos through the processor
        2. Tokenizing the prompt with image tokens
        3. Running vision encoder to get features

        Uses vision cache to skip processing for repeated images.

        Args:
            request: Request to preprocess
        """
        from mlx_vlm.utils import prepare_inputs

        tic = time.perf_counter()

        # Collect all images (including video frames)
        all_images = []

        if request.images:
            from .models.mllm import process_image_input

            for img in request.images:
                try:
                    path = process_image_input(img)
                    all_images.append(path)
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")

        if request.videos:
            from .models.mllm import (
                process_video_input,
                extract_video_frames_smart,
                save_frames_to_temp,
                DEFAULT_FPS,
                MAX_FRAMES,
            )

            for video in request.videos:
                try:
                    video_path = process_video_input(video)
                    frames = extract_video_frames_smart(
                        video_path,
                        fps=DEFAULT_FPS,
                        max_frames=MAX_FRAMES,
                    )
                    frame_paths = save_frames_to_temp(frames)
                    all_images.extend(frame_paths)
                except Exception as e:
                    logger.warning(f"Failed to process video: {e}")

        # Check pixel cache first
        cached_pixels = self.vision_cache.get_pixel_cache(all_images, request.prompt)
        if cached_pixels is not None:
            # Cache hit - use cached pixel values
            request.input_ids = cached_pixels.input_ids
            request.pixel_values = cached_pixels.pixel_values
            request.attention_mask = cached_pixels.attention_mask
            request.image_grid_thw = cached_pixels.image_grid_thw
            request.extra_kwargs = dict(cached_pixels.extra_kwargs)

            logger.debug(
                f"Pixel cache HIT for request {request.request_id}: "
                f"saved {cached_pixels.processing_time:.2f}s"
            )
            return

        # Cache miss - process images
        # Get model config
        model_config = getattr(self.model, "config", None)
        image_token_index = (
            getattr(model_config, "image_token_index", None) if model_config else None
        )

        # Prepare inputs using mlx_vlm
        inputs = prepare_inputs(
            self.processor,
            images=all_images if all_images else None,
            prompts=request.prompt,
            image_token_index=image_token_index,
        )

        request.input_ids = inputs.get("input_ids")
        request.prompt_token_ids = _flatten_token_ids(request.input_ids)
        request.pixel_values = inputs.get("pixel_values")
        request.attention_mask = inputs.get("attention_mask")

        # Extract extra kwargs
        request.extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        request.image_grid_thw = request.extra_kwargs.pop("image_grid_thw", None)

        processing_time = time.perf_counter() - tic

        # Store in pixel cache for future reuse
        if all_images and request.pixel_values is not None:
            self.vision_cache.set_pixel_cache(
                images=all_images,
                prompt=request.prompt,
                pixel_values=request.pixel_values,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                image_grid_thw=request.image_grid_thw,
                extra_kwargs=request.extra_kwargs,
                processing_time=processing_time,
            )

        self._stats.num_images_processed += len(all_images)
        self._stats.vision_encoding_time += processing_time

        logger.debug(
            f"Preprocessed request {request.request_id}: "
            f"{len(all_images)} images, {request.input_ids.size if request.input_ids is not None else 0} tokens "
            f"({processing_time:.2f}s)"
        )

    def _qualifies_for_specprefill(self, request: MLLMBatchRequest) -> bool:
        """Decide whether a preprocessed request can route through
        cooperative SpecPrefill on the MLLM path.

        SpecPrefill drops "unimportant" prompt tokens before the dense
        forward, so it must NOT run for requests carrying images: image
        token placeholders need to stay aligned with their pixel
        features inside ``masked_scatter`` in the VLM's
        ``get_input_embeddings``. Above-threshold text-only requests
        are eligible when the runtime supplies a draft model and the
        threshold/keep configuration.

        Inputs above ``self._specprefill_max_input`` (when set) are
        ineligible: the sparse-prefill path on the target VLM goes
        through ``manual_rope_proportional`` (a sequence of ~20 mlx
        ops per attention layer) instead of the fused
        ``mx.fast.rope`` kernel that the dense path uses, and at
        very long prompts the cumulative GPU dispatch pressure from
        500+ chunks crosses the macOS GPU watchdog threshold and
        fires ``kIOGPUCommandBufferCallbackErrorImpactingInteractivity``.
        Capping eligibility lets such requests fall through to the
        dense chunked path, which is slower but reliable at 256K.
        """
        if self._draft_model is None:
            return False
        if self._specprefill_threshold is None or self._specprefill_keep_pct is None:
            return False
        if request.pixel_values is not None or request.image_grid_thw is not None:
            return False
        if request.input_ids is None:
            return False
        iids = request.input_ids
        if iids.ndim == 1:
            seq_len = int(iids.size)
        else:
            seq_len = int(iids.shape[-1])
        if seq_len <= int(self._specprefill_threshold):
            return False
        if self._specprefill_max_input is not None and seq_len > int(
            self._specprefill_max_input
        ):
            return False
        return True

    def _run_specprefill_for_request(
        self, request: MLLMBatchRequest
    ) -> Tuple[mx.array, List[Any]]:
        """Run cooperative SpecPrefill end-to-end on a single text-only
        request.

        Pre-allocates the base cache from ``self.language_model`` so
        the cache shape matches what the VLM's underlying language
        model expects (e.g. Gemma 4's hybrid sliding-window + global
        layout). Drives the session synchronously since the MLLM
        scheduler does not currently expose cooperative step
        boundaries; the dense path inside ``_process_prompts`` is
        already serialized at the request boundary today.

        Returns ``(logits, wrapped_cache)`` where ``wrapped_cache`` is
        a list of ``RopeAdjustedCache`` layers ready for decode. The
        adjustment carried by each wrapper compensates for the gap
        between the (smaller) sparse-prefilled cache size and the
        original prompt length so RoPE positions during decode line up
        with the original token positions.
        """
        from mlx_lm.models.cache import make_prompt_cache

        iids = request.input_ids
        if iids.ndim == 2:
            iids = iids[0]
        tokens = iids.tolist()

        base_cache = make_prompt_cache(self.language_model)

        # SpecPrefill chunk sizes are split between two phases:
        #
        # - ``chunk_size=2048`` for the draft scoring phase. The draft
        #   model is small (~1.6B) and handles large forward chunks
        #   easily; smaller chunks just add per-chunk loop overhead.
        # - ``target_chunk_size=256`` for the sparse prefill phase on
        #   the target model. Matches the dense-path
        #   ``prefill_step_size=256`` lower bound. The sparse path
        #   goes through ``manual_rope_proportional`` (a sequence of
        #   pure mlx ops, ~20 per attention layer) instead of the
        #   fused ``mx.fast.rope`` kernel that the dense path uses,
        #   so per-chunk cumulative GPU time is higher than the raw
        #   attention pair count would suggest. 512 was empirically
        #   too large for 256K input + keep_pct=0.5: the watchdog
        #   ``kIOGPUCommandBufferCallbackErrorImpactingInteractivity``
        #   fires on the late chunks. 256 stays under the limit at
        #   256K + keep_pct=0.5, matching the dense lower bound.
        session = CooperativeSpecPrefillSession(
            model=self.model,
            draft_model=self._draft_model,
            tokens=tokens,
            base_cache=base_cache,
            position_offset=0,
            keep_pct=float(self._specprefill_keep_pct),
            chunk_size=2048,
            target_chunk_size=256,
        )

        try:
            while not session.is_done:
                session.step()
            result = session.finalize()
        finally:
            session.cleanup()

        request.vision_encoded = True

        # Unwrap LanguageModelOutput (Gemma 4 / Qwen 3.5 VLM language
        # models return a wrapper object whose ``logits`` attribute is
        # the actual mx.array). The cooperative session passes the raw
        # call result through without unwrapping, so we have to do it
        # here before _process_prompts subscripts ``logits[:, -1, :]``
        # for first-token sampling.
        logits = result.logits
        if hasattr(logits, "logits"):
            logits = logits.logits
        return logits, result.cache

    def _run_vision_encoding(
        self, request: MLLMBatchRequest, cache: Optional[List[Any]] = None
    ) -> mx.array:
        """
        Run the initial VLM forward pass to encode vision and get first logits.

        This runs the full VLM model (vision + language) on the prompt,
        which encodes the images and fills the provided KV cache.

        Long text-only prompts are split into ``prefill_step_size``-sized
        chunks so the activation memory needed for any single forward pass
        stays bounded. Cache state is materialized between chunks via
        ``mx.eval`` so the lazy graph cannot grow unbounded across the
        chunk loop. Vision (``pixel_values is not None``) requests stay
        on the single-shot path because image-token placeholders need to
        share the same forward pass as their pixel features for
        ``masked_scatter`` inside the VLM's ``get_input_embeddings`` to
        align them correctly.

        Args:
            request: Preprocessed request with input_ids and pixel_values
            cache: KV cache list for the language model. If provided, the
                   language model writes its KV state directly into this cache
                   during the forward pass.

        Returns:
            Logits from the (last) forward pass
        """
        # Build model call kwargs
        kwargs = dict(request.extra_kwargs)

        if request.pixel_values is not None:
            kwargs["pixel_values"] = request.pixel_values
        if request.attention_mask is not None:
            kwargs["attention_mask"] = request.attention_mask
        if request.image_grid_thw is not None:
            kwargs["image_grid_thw"] = request.image_grid_thw

        # Run full VLM forward pass with cache.
        # The VLM passes cache= through to self.language_model(),
        # so the language model writes KV state directly into our cache.
        input_ids = request.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        seq_len = input_ids.shape[1]
        chunk_size = max(1, int(self.prefill_step_size or seq_len))
        has_vision = request.pixel_values is not None

        if has_vision or seq_len <= chunk_size:
            # Single-shot path: short input, or vision-bearing input where
            # image-token placeholders must stay aligned with their pixels
            # inside one forward.
            output = self.model(input_ids, cache=cache, **kwargs)
        else:
            # Chunked text-only path. Subsequent chunks pass only input_ids
            # and cache; the VLM's get_input_embeddings sees pixel_values=None
            # and skips the vision tower entirely. Cache state is forced
            # between chunks so the lazy MLX graph cannot grow unbounded.
            output = None
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = input_ids[:, start:end]
                if start == 0:
                    output = self.model(chunk, cache=cache, **kwargs)
                else:
                    output = self.model(chunk, cache=cache)
                if cache is not None and end < seq_len:
                    # Materialize cache writes so activation graph stays bounded.
                    try:
                        mx.eval([c.state for c in cache])
                    except Exception:
                        # Defensive: if a cache layer doesn't expose .state
                        # we still want chunked prefill to keep working;
                        # MLX will lazily evaluate on the next chunk's read.
                        pass
                    if hasattr(mx, "clear_cache"):
                        mx.clear_cache()

        request.vision_encoded = True

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            return output.logits
        return output

    def _process_prompts(
        self, requests: List[MLLMBatchRequest]
    ) -> Tuple[Optional[MLLMBatch], List[MLLMBatchRequest]]:
        """
        Process a batch of requests through vision encoding and initial prefill.

        For MLLM, this is more complex than LLM:
        1. Preprocess each request (tokenize, process images)
        2. Run vision encoding per-request with individual KVCache objects
        3. Merge individual caches into a BatchKVCache for generation

        Requests whose preprocessing raises (e.g. malformed image data
        rejected by PIL or the model's image processor) are isolated in
        ``failed_requests`` so the caller can drain them via synthetic error
        responses instead of the whole batch raising and the scheduler
        process loop retrying the same bad input indefinitely.

        Args:
            requests: Requests to process

        Returns:
            Tuple of (MLLMBatch of successfully preprocessed requests or
            None if every request in this slice failed, list of requests
            whose preprocessing raised).
        """
        from mlx_lm.models.cache import make_prompt_cache

        tic = time.perf_counter()

        # Preprocess all requests. Per-request try/except isolates bad
        # image data so one malformed input does not poison the whole
        # batch or cause the outer scheduler process loop to retry the
        # same failing input forever.
        valid_requests: List[MLLMBatchRequest] = []
        failed_requests: List[MLLMBatchRequest] = []
        for req in requests:
            try:
                self._preprocess_request(req)
            except Exception as e:
                logger.warning(
                    f"Preprocessing failed for request {req.request_id}: "
                    f"{type(e).__name__}: {e}"
                )
                failed_requests.append(req)
                continue
            valid_requests.append(req)

        if not valid_requests:
            # Every request in this slice failed preprocessing; the caller
            # drains them via synthetic error responses.
            self._stats.prompt_time += time.perf_counter() - tic
            return None, failed_requests

        # Session 84 Fix 2: detect the first SpecPrefill-eligible
        # request in the slice. If found, isolate it as a SOLO batch
        # because the wrapped cache produced by sparse prefill carries
        # a per-request RoPE adjustment that does not safely co-merge
        # with dense-path caches in the existing
        # _normalize_cache_for_merge + .merge() path. Other valid
        # requests in this slice are pushed BACK to the head of
        # ``unprocessed_requests`` so the next ``_next`` call can batch
        # them normally on the dense path.
        specprefill_req: Optional[MLLMBatchRequest] = None
        for i, candidate in enumerate(valid_requests):
            if self._qualifies_for_specprefill(candidate):
                specprefill_req = candidate
                deferred = valid_requests[:i] + valid_requests[i + 1 :]
                if deferred:
                    self.unprocessed_requests = list(deferred) + list(
                        self.unprocessed_requests
                    )
                valid_requests = [candidate]
                break

        total_prompt_tokens = sum(
            req.input_ids.size if req.input_ids is not None else 1
            for req in valid_requests
        )
        self._stats.prompt_tokens += total_prompt_tokens

        # Log large prompts for monitoring instead of hard-failing here.
        max_batch_tokens = self.prefill_step_size * len(valid_requests)
        if total_prompt_tokens > max_batch_tokens:
            logger.warning(
                f"Large batch prefill: {total_prompt_tokens} tokens "
                f"(step_size={self.prefill_step_size}, requests={len(valid_requests)}). "
                f"Processing may be slow."
            )

        # Run vision encoding for each request with its own KVCache.
        # Vision encoding cannot be batched because each request may have
        # different images/pixel values. We pass a per-request KVCache to
        # the VLM so the language model writes its KV state directly into it.
        first_tokens = []
        all_logprobs = []
        per_request_caches = []

        if specprefill_req is not None:
            # Solo SpecPrefill path: bypass the dense vision encoding
            # loop and the per-layer cache merge entirely. The session
            # produces a fully-populated, RoPE-adjusted wrapped cache
            # that decode can read directly without going through
            # MLLMBatch's merge step.
            with mx.stream(MLLMBatchGenerator._stream):
                logits, wrapped_cache = self._run_specprefill_for_request(
                    specprefill_req
                )
                last_logits = logits[:, -1, :]
                sampled, logprobs = self._sample_request(specprefill_req, last_logits)
                mx.eval(sampled, logprobs)
                first_tokens.append(sampled.item())
                all_logprobs.append(logprobs.squeeze(0))

            self._stats.prompt_time += time.perf_counter() - tic
            return (
                MLLMBatch(
                    uids=[specprefill_req.uid],
                    request_ids=[specprefill_req.request_id],
                    y=mx.array(first_tokens),
                    logprobs=all_logprobs,
                    max_tokens=[specprefill_req.max_tokens],
                    num_tokens=[0],
                    cache=wrapped_cache,
                    requests=[specprefill_req],
                ),
                failed_requests,
            )

        for req in valid_requests:
            # Create a fresh KVCache for this request's language model prefill
            request_cache = make_prompt_cache(self.language_model)

            with mx.stream(MLLMBatchGenerator._stream):
                # Run VLM forward pass — cache= flows through to language_model
                logits = self._run_vision_encoding(req, cache=request_cache)

                # Extract last token logits and sample
                last_logits = logits[:, -1, :]
                sampled, logprobs = self._sample_request(req, last_logits)

                mx.eval(sampled, logprobs)

                first_tokens.append(sampled.item())
                all_logprobs.append(logprobs.squeeze(0))

            per_request_caches.append(request_cache)

        _validate_caches_mergeable(per_request_caches)
        normalized_caches = [
            [_normalize_cache_for_merge(layer_cache) for layer_cache in request_cache]
            for request_cache in per_request_caches
        ]

        try:
            batch_cache = [
                normalized_caches[0][layer_idx].merge(
                    [c[layer_idx] for c in normalized_caches]
                )
                for layer_idx in range(len(normalized_caches[0]))
            ]
        except Exception as e:
            logger.error(
                f"Failed to merge per-request KV caches: {type(e).__name__}: {e}"
            )
            raise

        # Create initial y (first generated tokens)
        y = mx.array(first_tokens)

        self._stats.prompt_time += time.perf_counter() - tic

        return (
            MLLMBatch(
                uids=[req.uid for req in valid_requests],
                request_ids=[req.request_id for req in valid_requests],
                y=y,
                logprobs=all_logprobs,
                max_tokens=[req.max_tokens for req in valid_requests],
                num_tokens=[0] * len(valid_requests),
                cache=batch_cache,
                requests=valid_requests,
            ),
            failed_requests,
        )

    def _step(
        self,
        input_tokens: mx.array,
        cache: List[Any],
        requests: List[MLLMBatchRequest],
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Run one generation step through the language model.

        Args:
            input_tokens: Input tokens [batch_size, 1] or [batch_size]
            cache: BatchKVCache for the language model
            requests: Active requests aligned with the batch rows

        Returns:
            Tuple of (sampled tokens, logprobs list)
        """
        # Ensure correct shape
        if input_tokens.ndim == 1:
            input_tokens = input_tokens[:, None]

        # Run language model only (not full VLM)
        output = self.language_model(input_tokens, cache=cache)

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        logits = logits[:, -1, :]

        sampled_rows: List[mx.array] = []
        logprobs: List[mx.array] = []
        for i, request in enumerate(requests):
            sampled, row_logprobs = self._sample_request(request, logits[i : i + 1, :])
            sampled_rows.append(sampled)
            logprobs.append(row_logprobs.squeeze(0))

        sampled_tokens = (
            sampled_rows[0]
            if len(sampled_rows) == 1
            else mx.concatenate(sampled_rows, axis=0)
        )
        return sampled_tokens, logprobs

    def _next(self) -> List[MLLMBatchResponse]:
        """
        Internal next() implementation.

        Returns:
            List of MLLMBatchResponse for this step
        """
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        error_responses: List[MLLMBatchResponse] = []

        # Only start a new batch when there is no active batch generating.
        # Per-request KV caches are created during vision encoding and then
        # merged into a single BatchKVCache. Merging into an active batch
        # mid-generation would cause shape mismatches in attention layers,
        # so queued requests wait until the current batch finishes.
        if num_active == 0:
            requests = self.unprocessed_requests[: self.completion_batch_size]

            if len(requests) == 0:
                self.active_batch = None
                return []

            # Always take ownership of this slice, whether preprocessing
            # succeeds or fails. Leaving failed requests on the queue
            # causes the scheduler process loop to retry the same failing
            # input indefinitely.
            self.unprocessed_requests = self.unprocessed_requests[len(requests) :]

            new_batch, failed_requests = self._process_prompts(requests)

            # Synthesize error responses for requests whose preprocessing
            # raised so the scheduler can drain them through its existing
            # ``finish_reason == "error"`` branch in
            # MLLMScheduler._process_batch_responses.
            for req in failed_requests:
                error_responses.append(
                    MLLMBatchResponse(
                        uid=req.uid,
                        request_id=req.request_id,
                        token=0,
                        logprobs=mx.array([]),
                        finish_reason="error",
                        prompt_cache=None,
                    )
                )

            self.active_batch = new_batch
            prompt_processing = new_batch is not None

        # Generate next token for active batch
        batch = self.active_batch
        if batch is None:
            # No valid batch this step — surface any synthetic error
            # responses we built so the scheduler cleans up failed requests.
            return error_responses

        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache, batch.requests)
        mx.async_eval(batch.y, *batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()

        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic

        # Build responses and track finished. Error responses for failed
        # preprocessing are emitted first so the scheduler cleans them up
        # alongside any in-flight generation tokens in this step.
        keep_idx = []
        end_idx = []
        responses: List[MLLMBatchResponse] = list(error_responses)

        for i, (token, uid, request_id, num_tok, max_tok, req) in enumerate(
            zip(
                y,
                batch.uids,
                batch.request_ids,
                batch.num_tokens,
                batch.max_tokens,
                batch.requests,
            )
        ):
            num_tok += 1
            batch.num_tokens[i] = num_tok
            req.num_tokens = num_tok
            req.output_tokens.append(token)

            finish_reason = None
            cache_fn = None

            if token in self.stop_tokens or token in req.stop_token_ids:
                finish_reason = "stop"
                end_idx.append(i)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(i)
            else:
                keep_idx.append(i)

            if finish_reason is not None:
                # Extract cache for this request
                cache_fn = lambda idx=i: batch.extract_cache(idx)

            responses.append(
                MLLMBatchResponse(
                    uid=uid,
                    request_id=request_id,
                    token=token,
                    logprobs=logprobs[i],
                    finish_reason=finish_reason,
                    prompt_cache=cache_fn,
                )
            )

        # Remove finished requests from batch
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        # Count only real token responses, not the synthetic error ones.
        self._stats.generation_tokens += len(responses) - len(error_responses)
        return responses

    def next(self) -> List[MLLMBatchResponse]:
        """
        Generate next token for all requests in the batch.

        Returns:
            List of MLLMBatchResponse, one per active request
        """
        with mx.stream(MLLMBatchGenerator._stream):
            return self._next()

    def stats(self) -> MLLMBatchStats:
        """
        Get generation statistics.

        Returns:
            MLLMBatchStats with timing and token counts
        """
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def get_vision_cache_stats(self) -> Dict[str, Any]:
        """Get vision cache statistics."""
        return self.vision_cache.get_stats()

    def has_pending(self) -> bool:
        """Check if there are pending or active requests."""
        return bool(self.unprocessed_requests or self.active_batch)
