# SPDX-License-Identifier: Apache-2.0
"""
Batched engine for continuous batching with multiple concurrent users.

This engine wraps AsyncEngineCore to provide continuous batching
for better throughput when serving multiple concurrent requests.

For MLLM models, all requests (text-only and multimodal) are routed through
the MLLMScheduler, which handles vision encoding and batched generation via
MLLMBatchGenerator. MLLM models only initialise the MLLM scheduler (not the
LLM engine), so text-only requests must also be routed through it.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, extract_multimodal_content, is_mllm_model
from ..message_utils import _normalize_messages
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

_MEDIA_TYPES = frozenset(
    {
        "image_url",
        "video_url",
        "audio_url",
        "image",
        "video",
        "audio",
    }
)


def _has_media_content(messages: list) -> bool:
    """Check if any message contains media content (images, video, audio)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _MEDIA_TYPES:
                    return True
    return False


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _merge_template_kwargs(
    base_kwargs: dict[str, Any],
    chat_template_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge request-level template kwargs with engine defaults."""
    merged = dict(base_kwargs)
    if isinstance(chat_template_kwargs, dict):
        merged.update(chat_template_kwargs)
    return merged


def _has_any_media(
    messages: list[dict[str, Any]],
    images: list[str] | None = None,
    videos: list[str] | None = None,
) -> bool:
    """Check both message content parts and top-level media parameters."""
    return bool(images or videos) or _has_media_content(messages)


def _collect_stop_tokens(tokenizer: Any) -> set[int]:
    """Collect EOS/stop token IDs from tokenizer variants."""
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    stop_tokens: set[int] = set()

    for attr in ("eos_token_id", "eos_token_ids"):
        value = getattr(actual_tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, set, tuple)):
            stop_tokens.update(int(token) for token in value if token is not None)
        else:
            stop_tokens.add(int(value))

    return stop_tokens


def _extract_media_from_messages(messages: list[dict[str, Any]]) -> tuple:
    """
    Extract images and videos from OpenAI-format messages.

    Returns:
        Tuple of (has_media, images_list, videos_list)
    """
    images = []
    videos = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            # Handle Pydantic models
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()

            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")

            if item_type == "image_url":
                img_url = item.get("image_url", {})
                if isinstance(img_url, str):
                    images.append(img_url)
                elif isinstance(img_url, dict):
                    url = img_url.get("url", "")
                    if url:
                        images.append(url)

            elif item_type == "image":
                img = item.get("image") or item.get("url", "")
                if img:
                    images.append(img)

            elif item_type == "video_url":
                vid_url = item.get("video_url", {})
                if isinstance(vid_url, str):
                    videos.append(vid_url)
                elif isinstance(vid_url, dict):
                    url = vid_url.get("url", "")
                    if url:
                        videos.append(url)

            elif item_type == "video":
                vid = item.get("video") or item.get("url", "")
                if vid:
                    videos.append(vid)

    has_media = bool(images or videos)
    return has_media, images, videos


class MLLMModelWrapper:
    """
    Wrapper for MLLM models to make them compatible with BatchGenerator.

    BatchGenerator expects model output to be subscriptable (logits array),
    but MLLM models return LanguageModelOutput objects. This wrapper extracts
    the logits from the output.

    Also handles Gemma 3's required pixel_values argument by injecting None
    for text-only requests.
    """

    def __init__(self, model):
        self._model = model
        # Detect if this is a Gemma 3 model (requires pixel_values as positional arg)
        self._is_gemma3 = (
            hasattr(model, "model_type")
            and "gemma3" in str(getattr(model, "model_type", "")).lower()
        )

    def __call__(self, *args, **kwargs):
        """Call the model and extract logits from LanguageModelOutput."""
        # Gemma 3 requires pixel_values as a positional argument, unlike Qwen
        # which makes it optional. Inject pixel_values=None for text-only requests.
        if self._is_gemma3 and "pixel_values" not in kwargs:
            kwargs["pixel_values"] = None

        output = self._model(*args, **kwargs)
        # If output has logits attribute, return just the logits
        if hasattr(output, "logits"):
            return output.logits
        return output

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model."""
        return getattr(self._model, name)


class BatchedEngine(BaseEngine):
    """
    Batched engine for continuous batching.

    This engine provides better throughput when serving multiple
    concurrent users by batching requests together.

    For MLLM (multimodal) models, this engine uses MLLMScheduler
    which handles images and videos alongside text generation.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        force_mllm: bool = False,
        mtp: bool = False,
        prefill_step_size: int | None = None,
        specprefill_enabled: bool = False,
        specprefill_draft_model_path: str | None = None,
        specprefill_threshold: int = 8192,
        specprefill_keep_pct: float = 0.3,
    ):
        """
        Initialize the batched engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            scheduler_config: Optional scheduler configuration
            stream_interval: Tokens to batch before streaming (1=every token)
            force_mllm: Force loading as MLLM even if not auto-detected
            mtp: Enable MTP per-request routing (text-only → TextModel, media → MLLM)
            prefill_step_size: Chunk size for prompt prefill (default 2048)
            specprefill_enabled: Enable SpecPrefill sparse prefill
            specprefill_draft_model_path: Draft model directory name under ~/ai-models/mlx_models/
            specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill (default 8192)
            specprefill_keep_pct: Fraction of tokens to keep (default 0.3)
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._mtp = mtp
        self._prefill_step_size = prefill_step_size or 2048

        # SpecPrefill configuration
        self._specprefill_enabled = specprefill_enabled
        self._specprefill_draft_model_path = specprefill_draft_model_path
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct
        self._specprefill_lock = asyncio.Lock()
        self._gpu_lock = asyncio.Lock()
        self._draft_model = None

        self._model = None
        self._processor = None  # For MLLM
        self._tokenizer = None  # For LLM
        self._engine = None  # AsyncEngineCore for LLM
        self._mllm_scheduler = None  # MLLMScheduler for MLLM
        self._mllm_instance = None  # MLXMultimodalLM instance
        self._loaded = False

        # Per-request routing state (MLLM+MTP mode)
        self._text_model = None
        self._text_tokenizer = None
        self._text_scheduler = None
        self._text_generation_lock = asyncio.Lock()
        self._text_scheduler_route_enabled = _env_flag(
            "VLLM_MLX_ENABLE_TEXT_BATCH_SCHEDULER_CANARY",
            _env_flag("VLLM_MLX_ENABLE_TEXT_BATCH_SCHEDULER", False),
        )

        # Hybrid cache detection: models that use RotatingKVCache (e.g.
        # Gemma 4 sliding-window) cannot use MLLM continuous batching
        # because BatchRotatingKVCache.merge does not preserve per-token
        # context across decode steps. Tracked upstream as vllm-mlx #159.
        # Set after model load by _detect_hybrid_cache().
        self._has_hybrid_cache = False

        # System prompt KV cache (reduces repeated prefill across requests)
        self._system_kv_snapshot = None  # List of (keys, values) per backbone layer
        self._system_kv_hash = None  # Hash of system prefix text
        self._system_kv_token_count = 0  # Tokens in cached prefix

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        return self._is_mllm

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        if self._is_mllm and self._processor:
            return getattr(self._processor, "tokenizer", self._processor)
        return self._tokenizer

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        if self._is_mllm:
            await self._start_mllm()
        else:
            await self._start_llm()

        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name} (mllm={self._is_mllm})")

    async def _start_mllm(self) -> None:
        """Start the MLLM engine with MLLMScheduler (continuous batching)."""
        from ..mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from ..models.mllm import MLXMultimodalLM

        # Load the MLLM model
        self._mllm_instance = MLXMultimodalLM(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )
        self._mllm_instance.load()

        self._model = self._mllm_instance.model
        self._processor = self._mllm_instance.processor

        # Detect hybrid-cache architectures (e.g. Gemma 4 sliding-window).
        # See _has_hybrid_cache field for upstream issue and rationale.
        self._has_hybrid_cache = self._detect_hybrid_cache()
        if self._has_hybrid_cache:
            logger.warning(
                "Hybrid cache detected (RotatingKVCache layers): text-only "
                "requests will route through the serial mlx_vlm.generate "
                "path via _mllm_instance.chat. Image requests still use "
                "MLLM continuous batching. Tracked upstream: vllm-mlx #159."
            )

        # Create MLLM scheduler config with batch generator support
        if self._scheduler_config and hasattr(self._scheduler_config, "max_num_seqs"):
            max_num_seqs = self._scheduler_config.max_num_seqs
        else:
            max_num_seqs = 16  # Default for continuous batching

        # Get batch sizes from config if available
        prefill_batch_size = getattr(self._scheduler_config, "prefill_batch_size", 4)
        completion_batch_size = getattr(
            self._scheduler_config, "completion_batch_size", 16
        )

        mllm_config = MLLMSchedulerConfig(
            max_num_seqs=max_num_seqs,
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
            enable_vision_cache=True,
            vision_cache_size=100,
        )

        # Create and start MLLM scheduler
        self._mllm_scheduler = MLLMScheduler(
            model=self._model,
            processor=self._processor,
            config=mllm_config,
        )
        self._mllm_scheduler.set_gpu_lock(self._gpu_lock)
        await self._mllm_scheduler.start()

        logger.info(
            f"MLLM Scheduler started with continuous batching: "
            f"max_num_seqs={max_num_seqs}, prefill_batch={prefill_batch_size}, "
            f"completion_batch={completion_batch_size}"
        )

        # Build TextModel for per-request routing (text-only → mlx_lm, media → MLLM).
        # Needed when:
        # - MTP is enabled (text-only gets MTP speedup)
        # - SpecPrefill is enabled (text-only gets sparse prefill)
        # - TextBatchScheduler canary routing is enabled (text-only foundation path)
        if (
            self._mtp
            or self._specprefill_enabled
            or self._text_scheduler_route_enabled
        ):
            try:
                from ..text_model_from_vlm import build_text_model

                self._text_model = build_text_model(
                    self._mllm_instance.model, self._model_name
                )
                if self._text_model is not None:
                    # Get tokenizer from the MLLM instance (same model, shared tokenizer)
                    self._text_tokenizer = self._mllm_instance.get_tokenizer()

                    # Apply Qwen3.5 eos_token fix (matches SimpleEngine pattern)
                    if "qwen3" in self._model_name.lower():
                        self._text_tokenizer.eos_token = "<|im_end|>"
                        self._text_tokenizer.eos_token_id = (
                            self._text_tokenizer.convert_tokens_to_ids("<|im_end|>")
                        )

                    # Check if TextModel actually has MTP
                    has_mtp = (
                        hasattr(self._text_model, "mtp")
                        and self._text_model.mtp is not None
                    )
                    if has_mtp:
                        logger.info(
                            "BatchedEngine MLLM+MTP routing: "
                            "text-only → TextModel (MTP), media → MLLM"
                        )
                    elif self._specprefill_enabled:
                        logger.info(
                            "BatchedEngine text routing: "
                            "text-only → TextModel (SpecPrefill, no MTP), media → MLLM"
                        )
                    elif self._text_scheduler_route_enabled:
                        logger.info(
                            "BatchedEngine text routing: "
                            "text-only → TextModel (TextBatchScheduler canary), media → MLLM"
                        )
                    else:
                        logger.warning(
                            "TextModel built but no MTP head and SpecPrefill disabled — "
                            "text-only won't benefit from routing"
                        )
                        self._text_model = None
                        self._text_tokenizer = None
            except Exception as e:
                logger.error(f"TextModel build failed: {e}")
                self._text_model = None
                self._text_tokenizer = None

        # Load SpecPrefill draft model (for TextModel path — after sparse
        # prefill, decode uses stream_generate with MTP for full throughput)
        if self._specprefill_enabled and self._specprefill_draft_model_path:
            try:
                from pathlib import Path

                from mlx_lm import load as mlx_lm_load

                draft_path = str(
                    Path.home()
                    / "ai-models"
                    / "mlx_models"
                    / self._specprefill_draft_model_path
                )
                self._draft_model, _ = mlx_lm_load(draft_path)
                logger.info(
                    "SpecPrefill draft model loaded: %s (threshold=%d, keep=%.0f%%)",
                    self._specprefill_draft_model_path,
                    self._specprefill_threshold,
                    self._specprefill_keep_pct * 100,
                )
            except Exception as e:
                logger.warning("Failed to load SpecPrefill draft model: %s", e)
                self._specprefill_enabled = False
                self._draft_model = None

        if self._text_model is not None and self._text_tokenizer is not None:
            try:
                from ..text_batch_scheduler import TextBatchScheduler

                self._text_scheduler = TextBatchScheduler(
                    model=self._text_model,
                    tokenizer=self._text_tokenizer,
                    gpu_lock=self._gpu_lock,
                    stop_tokens=_collect_stop_tokens(self._text_tokenizer),
                    draft_model=self._draft_model,
                    enable_mtp=self._mtp
                    and hasattr(self._text_model, "mtp_forward"),
                    prefill_step_size=self._prefill_step_size,
                    specprefill_threshold=self._specprefill_threshold,
                    specprefill_keep_pct=self._specprefill_keep_pct,
                )
                if self._text_scheduler_route_enabled:
                    await self._text_scheduler.start()
                    logger.info(
                        "TextBatchScheduler started for canary routing: mtp=%s",
                        self._mtp,
                    )
                else:
                    logger.info(
                        "TextBatchScheduler prepared with route disabled; "
                        "serial text path remains the production default"
                    )
            except Exception as e:
                logger.error("TextBatchScheduler init failed: %s", e)
                self._text_scheduler = None

    async def _start_llm(self) -> None:
        """Start the LLM engine with AsyncEngineCore."""
        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig
        from ..utils.tokenizer import load_model_with_fallback

        # Build tokenizer config
        tokenizer_config = {"trust_remote_code": self._trust_remote_code}

        # Qwen3 fix
        if "qwen3" in self._model_name.lower() or "Qwen3" in self._model_name:
            tokenizer_config["eos_token"] = "<|im_end|>"

        self._model, self._tokenizer = load_model_with_fallback(
            self._model_name,
            tokenizer_config=tokenizer_config,
        )

        # Validate MTP support if enabled
        if self._scheduler_config and self._scheduler_config.enable_mtp:
            from ..patches.qwen3_next_mtp import validate_mtp_support

            if validate_mtp_support(self._model):
                logger.info("[MTP] Model validated for MTP speculative decoding")
            else:
                logger.warning(
                    "[MTP] MTP validation failed — --enable-mtp will be ignored. "
                    "See warnings above for details."
                )

        # Set Metal memory limits to make allocation failures graceful
        # instead of fatal Metal command buffer errors (SIGABRT)
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                device_info = mx.device_info()
                max_recommended = device_info.get(
                    "max_recommended_working_set_size",
                    device_info.get("memory_size", 0),
                )
                if max_recommended > 0:
                    soft_limit = int(max_recommended * 0.90)
                    mx.set_memory_limit(soft_limit)
                    mx.set_cache_limit(32 * 1024 * 1024 * 1024)  # 32GB
                    logger.info(
                        f"Metal memory limits set: "
                        f"allocation_limit={soft_limit / 1e9:.1f}GB "
                        f"(90% of {max_recommended / 1e9:.1f}GB), "
                        f"cache_limit=32GB"
                    )
        except Exception as e:
            logger.warning(f"Failed to set Metal memory limits: {e}")

        # Create engine config
        scheduler_config = self._scheduler_config or SchedulerConfig()
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
        )

        # Create async engine
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        await self._engine.engine.start()

        # Load SpecPrefill draft model (LLM text-only path)
        if self._specprefill_enabled and self._specprefill_draft_model_path:
            try:
                from pathlib import Path

                from mlx_lm import load as mlx_lm_load

                draft_path = str(
                    Path.home()
                    / "ai-models"
                    / "mlx_models"
                    / self._specprefill_draft_model_path
                )
                self._draft_model, _ = mlx_lm_load(draft_path)
                logger.info(
                    "SpecPrefill draft model loaded: %s (threshold=%d, keep=%.0f%%)",
                    self._specprefill_draft_model_path,
                    self._specprefill_threshold,
                    self._specprefill_keep_pct * 100,
                )
            except Exception as e:
                logger.warning("Failed to load SpecPrefill draft model: %s", e)
                self._specprefill_enabled = False
                self._draft_model = None

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._text_scheduler:
            await self._text_scheduler.stop()
            self._text_scheduler = None

        if self._mllm_scheduler:
            await self._mllm_scheduler.stop()
            self._mllm_scheduler = None

        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
            self._engine = None

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._mllm_instance = None
        self._text_model = None
        self._text_tokenizer = None
        self._draft_model = None
        self._system_kv_snapshot = None
        self._system_kv_hash = None
        self._system_kv_token_count = 0
        self._loaded = False
        logger.info("BatchedEngine stopped")

    def _should_use_text_scheduler(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
    ) -> bool:
        """Gate the foundation scheduler to a safe canary subset."""
        if not self._text_scheduler_route_enabled:
            return False
        if self._text_scheduler is None or self._text_model is None:
            return False
        if _has_any_media(messages, images, videos):
            return False
        return True

    def _detect_hybrid_cache(self) -> bool:
        """Return True if the model has any sliding-window (Rotating) cache layers.

        Originally added as a safety net: when hybrid-cache architectures
        (e.g. Gemma 4 sliding-window) hit the MLLM continuous-batching path
        through BatchRotatingKVCache, the model emitted its first sampled
        token then looped on it until max_tokens. The root cause is in
        mlx_vlm/models/gemma4/language.py: the local `offset` variable was
        a reference to cache.offset (an mx.array on this path) and got
        silently mutated in place by BatchRotatingKVCache._update_in_place
        between the K-rope and Q-rope calls, leaving the query at position
        N+1 while the key was at N. Fix is shipped as a file_overlay on
        mlx_vlm/models/gemma4/language.py: `offset = cache.offset + 0`.

        When True, BatchedEngine.chat / stream_chat routes text-only
        requests through the serial _mllm_instance.chat / stream_chat path
        which uses mlx_vlm.generate directly. This is preserved as a
        defensive fallback for any future hybrid-cache breakage.

        Set VLLM_MLX_DISABLE_HYBRID_CACHE_GATE=1 to force this method to
        return False so hybrid models go through continuous batching. This
        is the desired production state now that the local mlx_vlm fix is
        in place; the env var preserves an explicit on/off knob in case the
        fix needs to be temporarily disabled or audited.
        """
        if os.environ.get("VLLM_MLX_DISABLE_HYBRID_CACHE_GATE") == "1":
            logger.info(
                "VLLM_MLX_DISABLE_HYBRID_CACHE_GATE=1 set: hybrid-cache "
                "fallback gate is bypassed. Hybrid models (e.g. Gemma 4) will "
                "use the BatchedEngine continuous-batching path. This requires "
                "the local mlx_vlm/models/gemma4/language.py `cache.offset + 0` "
                "snapshot fix to be in place (see patches/MANIFEST.yaml)."
            )
            return False
        if self._model is None:
            return False
        lm = getattr(self._model, "language_model", self._model)
        if not hasattr(lm, "make_cache"):
            return False
        try:
            from mlx_lm.models.cache import RotatingKVCache

            sample_cache = lm.make_cache()
        except Exception as e:
            logger.warning("Hybrid cache detection failed: %s", e)
            return False
        return any(isinstance(c, RotatingKVCache) for c in sample_cache)

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        num_images: int = 0,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Apply chat template to messages.

        Uses the processor's (or tokenizer's) apply_chat_template with the
        full message list so that system prompts and conversation history
        are preserved. The previous implementation extracted only the last
        user message text via mlx_vlm.prompt_utils.apply_chat_template,
        which dropped system prompts and all prior turns.
        """
        # Choose the best template applicator.
        # For MLLM models, the processor handles special vision tokens.
        # For text-only models, the tokenizer is sufficient.
        template_applicator = None
        if (
            self._is_mllm
            and self._processor
            and hasattr(self._processor, "apply_chat_template")
        ):
            template_applicator = self._processor
        elif hasattr(self.tokenizer, "apply_chat_template"):
            template_applicator = self.tokenizer

        if template_applicator is not None:
            # Convert OpenAI image_url content parts to HuggingFace format
            # so the processor can insert the correct vision placeholder tokens.
            if self._is_mllm and num_images > 0:
                messages = self._prepare_mllm_messages(messages)

            template_kwargs = _merge_template_kwargs(
                {
                    "tokenize": False,
                    "add_generation_prompt": True,
                },
                chat_template_kwargs,
            )
            if tools:
                template_kwargs["tools"] = tools
            # Pass enable_thinking from env (set by runtime_patches from mode.json)
            import os
            if "enable_thinking" not in template_kwargs:
                if os.environ.get("VLLM_MLX_ENABLE_THINKING", "").lower() in ("1", "true"):
                    template_kwargs["enable_thinking"] = True

            try:
                return template_applicator.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError as e:
                # Some templates don't accept 'tools'; retry without them.
                logger.debug(f"Chat template TypeError, retrying without extras: {e}")
                for key in ["tools"]:
                    if key in template_kwargs:
                        del template_kwargs[key]
                return template_applicator.apply_chat_template(
                    messages, **template_kwargs
                )
        else:
            # Fallback for models without apply_chat_template
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    @staticmethod
    def _prepare_mllm_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style image_url content to HuggingFace format.

        The OpenAI API uses ``{"type": "image_url", "image_url": {"url": ...}}``
        while HuggingFace processors expect ``{"type": "image"}``.

        Args:
            messages: List of chat messages in OpenAI format. Each message is a
                dict with at least ``role`` and ``content`` keys.

        Returns:
            A new list of messages with ``image_url`` parts replaced by
            ``{"type": "image"}`` entries for the HuggingFace processor.
        """
        prepared = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        new_content.append({"type": "image"})
                    elif isinstance(part, (dict, str)):
                        new_content.append(part)
                    # skip non-dict/non-str parts to avoid passing unexpected types
                prepared.append({**msg, "content": new_content})
            else:
                prepared.append(msg)
        return prepared

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            images: Optional image URLs/paths (for MLLM)
            videos: Optional video URLs/paths (for MLLM)
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        raw_output = bool(kwargs.pop("raw_output", False))

        if self._is_mllm and self._mllm_scheduler:
            # Use MLLM scheduler for all requests when model is multimodal.
            # MLLM models only initialise the _mllm_scheduler (not _engine),
            # so text-only requests must also be routed here.
            output = await self._mllm_scheduler.generate(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                **kwargs,
            )

            text = output.output_text if raw_output else clean_output_text(output.output_text)
            return GenerationOutput(
                text=text,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finish_reason=output.finish_reason,
            )

        # Use LLM engine for text-only (non-MLLM models)
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = output.output_text if raw_output else clean_output_text(output.output_text)

        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            images: Optional image URLs/paths (for MLLM)
            videos: Optional video URLs/paths (for MLLM)
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        raw_output = bool(kwargs.pop("raw_output", False))

        if self._is_mllm and self._mllm_scheduler:
            # Use MLLM scheduler for all streaming when model is multimodal
            request_id = await self._mllm_scheduler.add_request_async(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                **kwargs,
            )

            async for output in self._mllm_scheduler.stream_outputs(request_id):
                text = output.output_text if raw_output else clean_output_text(output.output_text)
                new_text = output.new_text if raw_output else clean_output_text(output.new_text)
                yield GenerationOutput(
                    text=text,
                    new_text=new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                )
            return

        # Use LLM engine for text-only
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        prefix_boundary = kwargs.pop("prefix_boundary", 0)
        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            prefix_boundary=prefix_boundary,
        )

        async for output in self._engine.stream_outputs(request_id):
            text = output.output_text if raw_output else clean_output_text(output.output_text)
            new_text = output.new_text if raw_output else clean_output_text(output.new_text)

            yield GenerationOutput(
                text=text,
                new_text=new_text,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finished=output.finished,
                finish_reason=output.finish_reason,
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        For MLLM models, all requests (including text-only) are routed through
        the MLLMScheduler for vision-aware batched generation.
        For non-MLLM models, uses the LLM engine with BatchGenerator.

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        if not self._loaded:
            await self.start()

        # Normalize messages before any path (developer->system, merge consecutive)
        messages = _normalize_messages(messages)

        # Hybrid cache models (e.g. Gemma 4 sliding-window) cannot use the
        # MLLM continuous batching path due to upstream cache merge bug
        # (vllm-mlx #159). Route text-only requests through the serial
        # mlx_vlm.generate path which handles per-step decode correctly.
        if (
            self._has_hybrid_cache
            and self._mllm_instance is not None
            and not _has_any_media(messages, images, videos)
        ):
            logger.info(
                "Hybrid cache route: text-only request → _mllm_instance.chat (serial)"
            )
            # MLLMMultimodalLM.chat is synchronous and CPU/GPU-bound; run in
            # a worker thread under the GPU lock so the event loop stays responsive.
            #
            # Cancellation safety: if this task is cancelled mid-generation
            # (e.g. client disconnect via `_disconnect_guard_nonstream`), a
            # bare ``await asyncio.to_thread(...)`` would raise CancelledError
            # out of the await, exit the ``async with self._gpu_lock:`` block,
            # and release the lock while the MLX worker thread is still
            # running. A subsequent request would then acquire the lock and
            # start a second MLX worker thread in parallel, racing on Metal
            # command buffers and tripping the assertion:
            #
            #   -[_MTLCommandBuffer addCompletedHandler:]:1011:
            #     failed assertion 'Completed handler provided after commit call'
            #
            # which SIGABRTs the server process. To keep the lock held until
            # the worker thread actually finishes, wrap the to_thread call in
            # a shielded task and wait for it to complete inside the except
            # handler before re-raising. This mirrors the pattern already used
            # by MLLMScheduler._process_loop and TextBatchScheduler._step_engine.
            async with self._gpu_lock:
                worker = asyncio.ensure_future(
                    asyncio.to_thread(
                        self._mllm_instance.chat,
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        tools=tools,
                        **kwargs,
                    )
                )
                try:
                    mllm_output = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    try:
                        await worker
                    except Exception:
                        pass
                    raise
            return GenerationOutput(
                text=mllm_output.text,
                prompt_tokens=mllm_output.prompt_tokens,
                completion_tokens=mllm_output.completion_tokens,
                finish_reason=mllm_output.finish_reason,
            )

        # Per-request MTP routing: text-only → TextModel, media → MLLM
        if self._should_use_text_scheduler(
            messages,
            tools=tools,
            images=images,
            videos=videos,
        ):
            logger.info("Text-only request → TextBatchScheduler [non-streaming]")
            last_output = None
            async for output in self._text_scheduler.submit(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs,
            ):
                last_output = output

            if last_output is not None:
                return last_output
            return GenerationOutput(text="", finish_reason="stop")

        if (
            self._text_model is not None
            and not _has_any_media(messages, images, videos)
        ):
            return await self._chat_text_model(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs,
            )

        # Extract images/videos from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos = extract_multimodal_content(messages)
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
            chat_template_kwargs=kwargs.get("chat_template_kwargs"),
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            **kwargs,
        )

    def _compute_prefix_boundary(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None
    ) -> int:
        """Compute token count for the shared prefix across message variations.

        Uses a two-tokenization approach: tokenize the full prompt twice
        (once as-is, once with the last user message replaced by a dummy)
        and find the longest common prefix (LCP).  This gives the exact
        boundary where different user suffixes diverge, avoiding template
        discrepancies (e.g. Qwen3 <think> markers on last assistant).
        """
        # Find index of last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None or last_user_idx == 0:
            return 0
        try:
            template_tools = convert_tools_for_template(tools) if tools else None

            # Tokenize the real prompt
            real_prompt = self._apply_chat_template(messages, template_tools)

            # Build a dummy variant with different last user content
            dummy_messages = list(messages)
            dummy_messages[last_user_idx] = {
                **messages[last_user_idx],
                "content": "XXXXXXXXXX",
            }
            dummy_prompt = self._apply_chat_template(dummy_messages, template_tools)

            tokenizer = self.tokenizer
            if hasattr(tokenizer, "tokenizer"):
                tokenizer = tokenizer.tokenizer

            real_tokens = tokenizer.encode(real_prompt)
            dummy_tokens = tokenizer.encode(dummy_prompt)

            # Find LCP — the point where the two diverge is the boundary
            lcp = 0
            for j in range(min(len(real_tokens), len(dummy_tokens))):
                if real_tokens[j] != dummy_tokens[j]:
                    break
                lcp = j + 1

            return lcp
        except Exception:
            return 0

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        For MLLM models, all requests (including text-only) are streamed through
        the MLLMScheduler for vision-aware batched generation.
        For non-MLLM models, uses the LLM engine with BatchGenerator.

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Normalize messages before any path (developer->system, merge consecutive)
        messages = _normalize_messages(messages)

        # Hybrid cache models (e.g. Gemma 4 sliding-window) cannot use the
        # MLLM continuous batching path due to upstream cache merge bug
        # (vllm-mlx #159). Route text-only requests through the serial
        # mlx_vlm.stream_generate path which handles per-step decode correctly.
        # MLLMMultimodalLM.stream_chat is a synchronous generator; pump it
        # into the asyncio loop via a queue + worker thread under the GPU lock.
        if (
            self._has_hybrid_cache
            and self._mllm_instance is not None
            and not _has_any_media(messages, images, videos)
        ):
            logger.info(
                "Hybrid cache route: text-only request → "
                "_mllm_instance.stream_chat (serial)"
            )

            async with self._gpu_lock:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                _SENTINEL = object()

                def _producer() -> None:
                    try:
                        for chunk in self._mllm_instance.stream_chat(
                            messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            tools=tools,
                            **kwargs,
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception as e:
                        loop.call_soon_threadsafe(queue.put_nowait, e)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

                # Schedule the producer via ``asyncio.ensure_future`` so the
                # surrounding exception handling can cleanly await its
                # completion on cancellation. See the matching cancellation-
                # safety note in ``chat()``: a bare ``loop.run_in_executor``
                # here combined with outer task cancellation would let the
                # ``async with self._gpu_lock:`` block exit while the MLX
                # worker thread is still mid-step, allowing a second request
                # to start a concurrent worker and trip the Metal command
                # buffer assertion.
                producer_task = asyncio.ensure_future(
                    asyncio.to_thread(_producer)
                )
                last_text = ""
                try:
                    while True:
                        chunk = await queue.get()
                        if chunk is _SENTINEL:
                            break
                        if isinstance(chunk, Exception):
                            raise chunk
                        new_text = (chunk.text or "")[len(last_text):]
                        last_text = chunk.text or last_text
                        yield GenerationOutput(
                            text=chunk.text or "",
                            new_text=new_text,
                            prompt_tokens=chunk.prompt_tokens,
                            completion_tokens=chunk.completion_tokens,
                            finished=chunk.finish_reason is not None,
                            finish_reason=chunk.finish_reason,
                        )
                except asyncio.CancelledError:
                    # Outer task cancelled (client disconnect, request
                    # timeout, async generator close). Drain the queue
                    # until the producer's sentinel so the background
                    # thread can exit without blocking, then await the
                    # task to guarantee the MLX worker has fully released
                    # the Metal context before the lock drops.
                    try:
                        while True:
                            chunk = await queue.get()
                            if chunk is _SENTINEL:
                                break
                    except Exception:
                        pass
                    try:
                        await producer_task
                    except Exception:
                        pass
                    raise
                finally:
                    # Normal-exit and exception paths: producer_task is
                    # usually already done here because the consumer loop
                    # only exits once the sentinel arrives. Await it
                    # defensively so the lock release is always ordered
                    # after the worker thread's final MLX call.
                    if not producer_task.done():
                        try:
                            await producer_task
                        except Exception:
                            pass
            return

        # Per-request MTP routing: text-only → TextModel, media → MLLM
        if self._should_use_text_scheduler(
            messages,
            tools=tools,
            images=images,
            videos=videos,
        ):
            logger.info("Text-only request → TextBatchScheduler [streaming]")
            async for output in self._text_scheduler.submit(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs,
            ):
                yield output
            return

        if (
            self._text_model is not None
            and not _has_any_media(messages, images, videos)
        ):
            async for output in self._stream_chat_text_model(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs,
            ):
                yield output
            return

        # Extract images/videos from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos = extract_multimodal_content(messages)
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
            chat_template_kwargs=kwargs.get("chat_template_kwargs"),
        )

        # Compute prefix boundary for cache
        prefix_boundary = self._compute_prefix_boundary(messages, tools)
        if prefix_boundary > 0:
            kwargs["prefix_boundary"] = prefix_boundary

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            **kwargs,
        ):
            yield output

    async def _chat_text_model(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Non-streaming text-only generation via mlx_lm TextModel with MTP.

        Collects all streaming output into a single GenerationOutput.
        Used when MLLM+MTP routing is active and the request has no media.
        """
        logger.info("Text-only request → TextModel (MTP) [non-streaming]")
        accumulated_text = ""
        last_chunk = None
        async for chunk in self._stream_chat_text_model(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            **kwargs,
        ):
            accumulated_text = chunk.text
            last_chunk = chunk
        if last_chunk is not None:
            return GenerationOutput(
                text=accumulated_text,
                prompt_tokens=last_chunk.prompt_tokens,
                completion_tokens=last_chunk.completion_tokens,
                finish_reason=last_chunk.finish_reason,
            )
        return GenerationOutput(text="", finish_reason="stop")

    async def _stream_chat_text_model(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming text-only generation via mlx_lm TextModel with MTP.

        Used when MLLM+MTP routing is active and the request has no media.
        Runs the full generation in a single thread to maintain Metal safety.

        System prompt KV caching: on the first request, prefills system tokens
        and snapshots backbone KV state. Subsequent requests with the same
        system prompt restore the snapshot and only prefill the suffix tokens.

        SpecPrefill: when a draft model is loaded and the prompt exceeds the
        threshold, uses attention-based sparse prefill for faster TTFT.
        Composes with system KV cache (sparse-prefill only the suffix when
        cache hits). Falls back to normal path on any error.
        """
        import hashlib
        import os

        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        # Per-request template/specprefill overrides (from extra_body)
        chat_template_kwargs = kwargs.pop("chat_template_kwargs", None)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct_override = kwargs.pop("specprefill_keep_pct", None)

        # Read enable_thinking from env (set by runtime_patches, consistent with MLLM path)
        enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
        enable_thinking = enable_thinking_env.lower() in ("true", "1", "yes")

        # Deep-convert messages AND tools to pure dicts (Pydantic models in
        # tool_calls/parameters cause Jinja ".items()" errors in chat templates)
        import json

        messages = json.loads(json.dumps(messages, default=str))
        if tools:
            tools = json.loads(json.dumps(tools, default=str))

        # Apply chat template
        template_kwargs = _merge_template_kwargs(
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            },
            chat_template_kwargs,
        )
        if tools:
            template_kwargs["tools"] = tools

        try:
            prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except (TypeError, Exception) as e:
            # Template may reject tools= or enable_thinking=, or tools format
            # may not match template expectations — retry without tools
            logger.debug("Chat template error, retrying without tools: %s", e)
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )

        # Build sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)
        max_tokens = max_tokens or 4096

        # Check MTP support — used by all generation paths below
        _has_mtp = hasattr(self._text_model, "mtp_forward")

        # --- System KV cache: find system prefix boundary ---
        # Detect template format and find first user turn marker.
        # ChatML (Qwen 3.5): <|im_start|>user
        # Gemma 4: <|turn>user
        _USER_MARKERS = ("<|im_start|>user", "<|turn>user")
        marker_pos = -1
        for _um in _USER_MARKERS:
            marker_pos = prompt.find(_um)
            if marker_pos > 0:
                break
        if marker_pos > 0:
            system_prefix = prompt[:marker_pos]
            suffix = prompt[marker_pos:]
            prefix_hash = hashlib.sha256(system_prefix.encode()).hexdigest()[:16]
        else:
            system_prefix = None
            suffix = prompt
            prefix_hash = None

        # Check for cache hit
        cache_hit = (
            prefix_hash is not None
            and prefix_hash == self._system_kv_hash
            and self._system_kv_snapshot is not None
        )

        if cache_hit:
            logger.info(
                "Text-only request → TextModel (MTP) [streaming, system KV cache HIT: "
                "reusing %d cached tokens, hash=%s]",
                self._system_kv_token_count,
                prefix_hash,
            )
        else:
            logger.info("Text-only request → TextModel (MTP) [streaming]")

        prefill_step_size = self._prefill_step_size

        # --- SpecPrefill decision ---
        # Determine whether to use specprefill for this request.
        # Must be decided before entering the generation lock so we can
        # tokenize and check the threshold outside the critical section.
        _SPECPREFILL_MAX_TOKENS = 196608
        use_specprefill = False
        if self._draft_model is not None:
            if specprefill_override is True:
                use_specprefill = True
            elif specprefill_override is None and self._specprefill_enabled:
                use_specprefill = True
            # specprefill_override=False explicitly disables

        # Tokenize to determine token count for specprefill threshold check.
        # We need this for both specprefill and normal paths anyway.
        sp_tokens = None  # tokens to score (suffix or full prompt)
        sp_offset = 0  # position offset for sparse_prefill
        sp_n_total = 0  # total prompt tokens (for logging / threshold)

        if use_specprefill:
            if cache_hit:
                # Score only the suffix — system prefix is already cached
                sp_tokens = self._text_tokenizer.encode(suffix)
                sp_offset = self._system_kv_token_count
                sp_n_total = sp_offset + len(sp_tokens)
            else:
                # Score the full prompt
                sp_tokens = self._text_tokenizer.encode(prompt)
                sp_offset = 0
                sp_n_total = len(sp_tokens)

            n_sp_tokens = len(sp_tokens)

            # Threshold check (skip when force-enabled via per-request override)
            if (
                specprefill_override is not True
                and n_sp_tokens <= self._specprefill_threshold
            ):
                use_specprefill = False

            # Upper bound: cap to avoid draft model OOM
            if use_specprefill and n_sp_tokens > _SPECPREFILL_MAX_TOKENS:
                logger.warning(
                    "SpecPrefill: prompt %d tokens exceeds max %d, "
                    "falling back to normal path",
                    n_sp_tokens,
                    _SPECPREFILL_MAX_TOKENS,
                )
                use_specprefill = False

        # Run under generation lock, all tokens in single thread (Metal safety)
        # CRITICAL: use asyncio.shield to prevent cancellation from releasing
        # the lock while to_thread worker is still executing Metal ops.
        # Without shield, CancelledError propagates through 'async with',
        # releasing the lock while the background thread is mid-eval.
        # Next request acquires the freed lock -> two threads hit Metal
        # concurrently -> SIGABRT (tryCoalescingPreviousComputeCommandEncoder).
        # See: vllm-mlx PR #220, mlx issue #3216.
        async with self._text_generation_lock:

            def _run_with_cache():
                if use_specprefill:
                    try:
                        return _run_specprefill()
                    except Exception as e:
                        logger.error(
                            "SpecPrefill failed, falling back to normal path: %s", e
                        )
                        # Fall through to normal path
                if cache_hit:
                    return _run_cache_hit()
                else:
                    return _run_cache_miss()

            def _run_specprefill():
                """Score tokens, sparse prefill, generate with MTP.

                Composes with system KV cache: when cache_hit, restores the
                system KV snapshot first, then sparse-prefills only the suffix
                tokens with position_offset = system_kv_token_count.

                After Phase 3 (sparse prefill), the cache is fully materialized
                with _OffsetAdjustedRoPE installed. Phase 4 hands off to
                stream_generate for MTP-enabled decode at full throughput.
                """
                import time
                from types import SimpleNamespace

                from ..specprefill import (
                    cleanup_rope,
                    score_tokens,
                    select_chunks,
                    sparse_prefill,
                )

                # Build target cache (optionally restore system KV snapshot)
                target_cache = make_prompt_cache(self._text_model)
                if cache_hit:
                    for layer_idx, snapshot_state in enumerate(
                        self._system_kv_snapshot
                    ):
                        if layer_idx < len(target_cache):
                            target_cache[layer_idx].state = snapshot_state
                    mx.eval([c.state for c in target_cache if hasattr(c, "state")])

                try:
                    # Phase 1: Score with draft model
                    t0 = time.monotonic()
                    importance = score_tokens(
                        self._draft_model,
                        sp_tokens,
                        prefill_step_size=prefill_step_size,
                    )
                    t_score = time.monotonic() - t0

                    # Phase 2: Select important chunks
                    effective_keep = (
                        specprefill_keep_pct_override or self._specprefill_keep_pct
                    )
                    selected = select_chunks(importance, keep_pct=effective_keep)
                    n_selected = selected.shape[0]
                    n_scored = len(sp_tokens)

                    # Phase 3: Sparse prefill on target model
                    t0 = time.monotonic()
                    logits = sparse_prefill(
                        self._text_model,
                        sp_tokens,
                        selected,
                        target_cache,
                        step_size=prefill_step_size,
                        position_offset=sp_offset,
                    )
                    t_prefill = time.monotonic() - t0

                    logger.info(
                        "SpecPrefill: scored %d tokens in %.1fs, "
                        "sparse prefill %d/%d (keep=%.0f%%) in %.1fs "
                        "(offset=%d, effective_keep=%.2f)",
                        n_scored,
                        t_score,
                        n_selected,
                        n_scored,
                        n_selected / n_scored * 100,
                        t_prefill,
                        sp_offset,
                        effective_keep,
                    )

                    # Phase 4: Generate with MTP via stream_generate
                    y0 = sampler(logits[:, -1, :])
                    mx.eval(y0)

                    # Build cache with MTP entries
                    if hasattr(self._text_model, "make_mtp_cache"):
                        gen_cache = list(target_cache) + list(
                            self._text_model.make_mtp_cache()
                        )
                    else:
                        gen_cache = list(target_cache)

                    # y0 was sampled from Phase 3 logits but stream_generate
                    # consumes it as "prompt", so prepend its text
                    y0_text = self._text_tokenizer.decode([y0.item()])
                    eos_id = self._text_tokenizer.eos_token_id
                    results = [
                        SimpleNamespace(
                            text=y0_text,
                            finish_reason=(
                                "stop" if y0.item() == eos_id else None
                            ),
                        )
                    ]

                    if y0.item() != eos_id:
                        for resp in mlx_stream_generate(
                            self._text_model,
                            self._text_tokenizer,
                            prompt=y0.reshape(-1),
                            max_tokens=max_tokens - 1,
                            sampler=sampler,
                            mtp=_has_mtp,
                            prompt_cache=gen_cache,
                            prefill_step_size=prefill_step_size,
                        ):
                            results.append(
                                SimpleNamespace(
                                    text=resp.text,
                                    finish_reason=resp.finish_reason,
                                )
                            )

                    return results, sp_n_total

                finally:
                    cleanup_rope(self._text_model)

            def _run_cache_hit():
                """Restore system KV snapshot, prefill only suffix, generate."""
                # Restore cached KV state into a fresh cache
                restored_cache = make_prompt_cache(self._text_model)
                for layer_idx, snapshot_state in enumerate(self._system_kv_snapshot):
                    if layer_idx < len(restored_cache):
                        restored_cache[layer_idx].state = snapshot_state
                mx.eval([c.state for c in restored_cache if hasattr(c, "state")])

                # Tokenize just the suffix and generate with the primed cache.
                # stream_generate accepts mx.array prompt (skips tokenization)
                # and prompt_cache is forwarded to mtp_generate_step.
                suffix_tokens = self._text_tokenizer.encode(suffix)
                suffix_array = mx.array(suffix_tokens)
                n_suffix = len(suffix_tokens)

                logger.info(
                    "System KV cache HIT: prefilling %d suffix tokens "
                    "(skipped %d cached tokens)",
                    n_suffix,
                    self._system_kv_token_count,
                )

                results = []
                for resp in mlx_stream_generate(
                    self._text_model,
                    self._text_tokenizer,
                    prompt=suffix_array,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    mtp=_has_mtp,
                    num_draft_tokens=1,
                    prompt_cache=restored_cache,
                    prefill_step_size=prefill_step_size,
                ):
                    results.append(resp)
                return results, self._system_kv_token_count + len(suffix_tokens)

            def _run_cache_miss():
                """Full prefill + generation, then snapshot system KV for next time."""
                results = []
                for resp in mlx_stream_generate(
                    self._text_model,
                    self._text_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    mtp=_has_mtp,
                    num_draft_tokens=1,
                    prefill_step_size=prefill_step_size,
                ):
                    results.append(resp)

                # Snapshot system KV for next request (if we found a system prefix)
                if prefix_hash is not None and system_prefix is not None:
                    try:
                        _snapshot_system_kv()
                    except Exception as e:
                        logger.warning("Failed to snapshot system KV cache: %s", e)

                # Get total prompt token count from generation response
                prompt_tokens = 0
                if results and hasattr(results[0], "prompt_tokens"):
                    prompt_tokens = results[0].prompt_tokens
                return results, prompt_tokens

            def _snapshot_system_kv():
                """Prefill just the system prefix on a fresh cache and save snapshot."""
                snapshot_cache = make_prompt_cache(self._text_model)
                prefix_tokens = self._text_tokenizer.encode(system_prefix)
                prefix_ids = mx.array(prefix_tokens)

                # Chunked prefill of system prefix
                for i in range(0, prefix_ids.size, prefill_step_size):
                    chunk = prefix_ids[i : i + prefill_step_size]
                    self._text_model(chunk[None], cache=snapshot_cache)
                    mx.eval([c.state for c in snapshot_cache if hasattr(c, "state")])

                # Save snapshot: deep copy of each cache layer's state
                self._system_kv_snapshot = []
                for c in snapshot_cache:
                    state = c.state
                    if getattr(c, "_use_fused_sdpa", False):
                        # QuantizedSDPACache: state is nested tuples
                        # ((packed, scales, biases), (packed, scales, biases))
                        k_tuple, v_tuple = state
                        self._system_kv_snapshot.append((
                            tuple(mx.array(t) for t in k_tuple),
                            tuple(mx.array(t) for t in v_tuple),
                        ))
                    elif isinstance(state, tuple) and len(state) == 2:
                        # KVCache: (keys, values) — copy to detach from cache
                        keys, values = state
                        self._system_kv_snapshot.append(
                            (mx.array(keys), mx.array(values))
                        )
                    elif isinstance(state, list):
                        # ArraysCache: list of arrays (Mamba/hybrid)
                        self._system_kv_snapshot.append(
                            [mx.array(a) if a is not None else None for a in state]
                        )
                    else:
                        # Unknown cache type — store as-is
                        self._system_kv_snapshot.append(state)

                self._system_kv_token_count = len(prefix_tokens)
                self._system_kv_hash = prefix_hash

                def _entry_bytes(x):
                    if hasattr(x, "nbytes"):
                        return x.nbytes
                    elif isinstance(x, (tuple, list)):
                        return sum(_entry_bytes(i) for i in x if i is not None)
                    return 0

                cache_bytes = sum(_entry_bytes(e) for e in self._system_kv_snapshot)
                logger.info(
                    "System KV cache: stored %d-token snapshot " "(%.1f MB), hash=%s",
                    len(prefix_tokens),
                    cache_bytes / 1e6,
                    prefix_hash,
                )

            _task = asyncio.ensure_future(asyncio.to_thread(_run_with_cache))
            try:
                result = await asyncio.shield(_task)
            except asyncio.CancelledError:
                # Shield was pierced by outer cancellation. Wait for the
                # Metal worker to finish before releasing the lock.
                try:
                    await _task
                except Exception:
                    pass
                raise
            all_resps, prompt_token_count = result

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        for i, resp in enumerate(all_resps):
            token_count += 1
            new_text = resp.text if hasattr(resp, "text") else str(resp)
            accumulated_text += new_text

            is_last = i == len(all_resps) - 1
            finished = is_last or token_count >= max_tokens

            yield GenerationOutput(
                text=accumulated_text,
                new_text=new_text,
                prompt_tokens=prompt_token_count,
                completion_tokens=token_count,
                finished=finished,
                finish_reason="stop" if finished else None,
            )

            if finished:
                break

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=prompt_token_count,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "batched",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
            "running": self._loaded,
            "stream_interval": self._stream_interval,
            "text_scheduler_route_enabled": self._text_scheduler_route_enabled,
            "num_running": 0,
            "num_waiting": 0,
            "num_requests_processed": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        }

        if self._mllm_scheduler:
            try:
                mllm_stats = self._mllm_scheduler.get_stats()
                stats["mllm_scheduler"] = mllm_stats
                # Promote Metal memory stats to top-level for /v1/status
                for key in (
                    "metal_active_memory_gb",
                    "metal_peak_memory_gb",
                    "metal_cache_memory_gb",
                ):
                    if key in mllm_stats:
                        stats[key] = mllm_stats[key]
                stats["num_running"] += int(mllm_stats.get("num_running", 0))
                stats["num_waiting"] += int(mllm_stats.get("num_waiting", 0))
                stats["num_requests_processed"] += int(
                    mllm_stats.get("num_requests_processed", 0)
                )
                stats["total_prompt_tokens"] += int(
                    mllm_stats.get("total_prompt_tokens", 0)
                )
                stats["total_completion_tokens"] += int(
                    mllm_stats.get("total_completion_tokens", 0)
                )
            except Exception as exc:
                logger.warning("Failed to collect MLLM scheduler stats: %s", exc)
                stats["mllm_scheduler"] = {"error": str(exc)}
        elif self._engine:
            stats.update(self._engine.get_stats())

        # SpecPrefill stats
        if self._draft_model is not None:
            stats["specprefill"] = {
                "enabled": self._specprefill_enabled,
                "draft_model": self._specprefill_draft_model_path,
                "threshold": self._specprefill_threshold,
                "keep_pct": self._specprefill_keep_pct,
            }

        if self._text_scheduler:
            try:
                text_stats = self._text_scheduler.get_stats()
                stats["text_scheduler"] = text_stats
                stats["num_running"] += int(text_stats.get("active_requests", 0))
                stats["num_waiting"] += int(
                    text_stats.get("pending_requests", 0)
                ) + int(text_stats.get("deferred_requests", 0))
                stats["num_requests_processed"] += int(
                    text_stats.get("num_requests_processed", 0)
                )
                stats["total_prompt_tokens"] += int(
                    text_stats.get("total_prompt_tokens", 0)
                )
                stats["total_completion_tokens"] += int(
                    text_stats.get("total_completion_tokens", 0)
                )
            except Exception as exc:
                logger.warning("Failed to collect text scheduler stats: %s", exc)
                stats["text_scheduler"] = {"error": str(exc)}

        # System KV cache stats
        if self._system_kv_snapshot is not None:
            cache_bytes = 0
            for entry in self._system_kv_snapshot:
                if isinstance(entry, tuple) and len(entry) == 2:
                    cache_bytes += entry[0].nbytes + entry[1].nbytes
                elif isinstance(entry, list):
                    cache_bytes += sum(a.nbytes for a in entry if a is not None)
            stats["system_kv_cache"] = {
                "tokens": self._system_kv_token_count,
                "hash": self._system_kv_hash,
                "memory_mb": round(cache_bytes / 1e6, 1),
            }

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._mllm_scheduler and self._mllm_scheduler.vision_cache:
            return self._mllm_scheduler.vision_cache.get_stats()
        elif self._engine:
            return self._engine.get_cache_stats()
        return None

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk for persistence across restarts."""
        if self._engine:
            return self._engine.save_cache_to_disk(cache_dir)
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self._engine:
            return self._engine.load_cache_from_disk(cache_dir)
        return 0
