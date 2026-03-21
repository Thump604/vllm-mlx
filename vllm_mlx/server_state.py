# SPDX-License-Identifier: Apache-2.0
"""
Server state container for vllm-mlx.

Replaces module-level globals in server.py with a single ServerState
instance attached to the FastAPI app (app.state.server). All mutable
server state lives here; endpoint handlers access it via request.app.state.server.
"""

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .engine import BaseEngine, BatchedEngine, SimpleEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


# ---------------------------------------------------------------------------
# RateLimiter (moved from server.py — pure utility, no global state)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0  # 1 minute window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed for client.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, 0

        current_time = time.time()
        window_start = current_time - self.window_size

        with self._lock:
            # Clean old requests outside window
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > window_start
            ]

            # Check rate limit
            if len(self._requests[client_id]) >= self.requests_per_minute:
                # Calculate retry-after
                oldest = min(self._requests[client_id])
                retry_after = int(oldest + self.window_size - current_time) + 1
                return False, max(1, retry_after)

            # Record this request
            self._requests[client_id].append(current_time)
            return True, 0


# ---------------------------------------------------------------------------
# ServerState
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """All mutable server state, attached to app.state.server."""

    # Engine
    engine: BaseEngine | None = None
    model_name: str | None = None
    default_max_tokens: int = 32768
    default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
    default_temperature: float | None = None  # Set via --default-temperature
    default_top_p: float | None = None  # Set via --default-top-p

    # MCP
    mcp_manager: object | None = None
    mcp_executor: object | None = None

    # Embedding engine (lazy loaded)
    embedding_engine: object | None = None
    embedding_model_locked: str | None = None  # Set when --embedding-model is used

    # API key authentication
    api_key: str | None = None
    auth_warning_logged: bool = False

    # Reasoning parser (for models like Qwen3, DeepSeek-R1)
    reasoning_parser: object | None = None  # ReasoningParser instance when enabled

    # Tool calling configuration
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = (
        None  # Parser name: auto, mistral, qwen, llama, hermes
    )
    tool_parser_instance: object | None = None  # Instantiated parser

    # Rate limiter
    rate_limiter: RateLimiter = field(
        default_factory=lambda: RateLimiter(requests_per_minute=60, enabled=False)
    )

    # Audio engines (lazy loaded)
    stt_engine: object | None = None
    tts_engine: object | None = None


# ---------------------------------------------------------------------------
# State-aware functions (formerly module-level functions using globals)
# ---------------------------------------------------------------------------


def get_engine(state: ServerState) -> BaseEngine:
    """Get the loaded engine, raising error if not loaded."""
    if state.engine is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.engine


def resolve_temperature(state: ServerState, request_value: float | None) -> float:
    """Resolve temperature: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if state.default_temperature is not None:
        return state.default_temperature
    return _FALLBACK_TEMPERATURE


def resolve_top_p(state: ServerState, request_value: float | None) -> float:
    """Resolve top_p: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if state.default_top_p is not None:
        return state.default_top_p
    return _FALLBACK_TOP_P


def get_cache_dir(state: ServerState) -> str:
    """Get cache persistence directory based on model name."""
    # Use state.model_name which is always a string, set during load_model()
    model_name = state.model_name if state.model_name else "default"
    logger.info(
        f"[_get_cache_dir] model_name={state.model_name!r} type={type(state.model_name)}"
    )
    # Sanitize model name for filesystem
    safe_name = str(model_name).replace("/", "--").replace("\\", "--")
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache", safe_name
    )
    logger.info(f"[_get_cache_dir] cache_dir={cache_dir!r}")
    return cache_dir


def load_prefix_cache_from_disk(state: ServerState) -> None:
    """Load prefix cache from disk during startup."""
    try:
        d = get_cache_dir(state)
        logger.info(f"[lifespan] Loading prefix cache from {d}")
        loaded = state.engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info(f"[lifespan] Loaded {loaded} prefix cache entries")
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to load cache from disk: {e}", exc_info=True)


def save_prefix_cache_to_disk(state: ServerState) -> None:
    """Save prefix cache to disk during shutdown."""
    try:
        d = get_cache_dir(state)
        logger.info(f"[lifespan] Saving prefix cache to {d}")
        saved = state.engine.save_cache_to_disk(d)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to save cache to disk: {e}", exc_info=True)


def _detect_native_tool_support(state: ServerState) -> bool:
    """
    Detect if the active tool parser supports native tool format.

    Native format means role="tool" messages and tool_calls fields
    are preserved instead of being converted to text.

    Returns:
        True if native format should be preserved
    """
    from .tool_parsers import ToolParserManager

    if not state.enable_auto_tool_choice or not state.tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(state.tool_call_parser)
        return parser_cls.supports_native_format()
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{state.tool_call_parser}' not found. "
            f"Available parsers: {ToolParserManager.list_registered()}"
        )
        return False
    except Exception as e:
        # Unexpected error during detection
        logger.warning(f"Failed to detect native tool support: {e}")
        return False


def load_embedding_model(
    state: ServerState,
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
) -> None:
    """Load or reuse the embedding model engine when configured."""
    if not model_name:
        return

    if lock:
        state.embedding_model_locked = model_name

    if (
        reuse_existing
        and state.embedding_engine is not None
        and state.embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    state.embedding_engine = EmbeddingEngine(model_name)
    state.embedding_engine.load()


def load_model(
    state: ServerState,
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    force_mllm: bool = False,
    mtp: bool = False,
    specprefill_enabled: bool = False,
    specprefill_draft_model_path: str | None = None,
    specprefill_threshold: int = 8192,
    specprefill_keep_pct: float = 0.3,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        state: ServerState instance
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        force_mllm: Force loading as MLLM even if not auto-detected
        mtp: Enable native MTP speculative decoding (per-request routing in both engines)
        specprefill_enabled: Enable SpecPrefill (attention-based sparse prefill)
        specprefill_draft_model_path: Path to draft model for SpecPrefill scoring
        specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill
        specprefill_keep_pct: Fraction of tokens to keep during sparse prefill
    """
    state.default_max_tokens = max_tokens
    state.model_name = model_name
    # Reset tool parser instance when model is reloaded (tokenizer may change)
    state.tool_parser_instance = None

    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")

    if use_batching:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        state.engine = BatchedEngine(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            mtp=mtp,
            specprefill_enabled=specprefill_enabled,
            specprefill_draft_model_path=specprefill_draft_model_path,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
        )
        # BatchedEngine will be started in lifespan (uvicorn's event loop)
        # Just log for now
        logger.info(f"Model loaded (batched mode): {model_name}")
    else:
        logger.info(f"Loading model with SimpleEngine: {model_name}")
        state.engine = SimpleEngine(
            model_name=model_name,
            force_mllm=force_mllm,
            mtp=mtp,
            specprefill_enabled=specprefill_enabled,
            specprefill_draft_model_path=specprefill_draft_model_path,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
        )
        # Start SimpleEngine synchronously (no background loop)
        # Use new_event_loop() for Python 3.10+ compatibility (get_event_loop() is deprecated)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(state.engine.start())
        model_type = "MLLM" if state.engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")

    # Set native tool format support on the engine (thread-safe via instance property)
    state.engine.preserve_native_tool_format = _detect_native_tool_support(state)
    if state.engine.preserve_native_tool_format:
        logger.info(f"Native tool format enabled for parser: {state.tool_call_parser}")

    logger.info(f"Default max tokens: {state.default_max_tokens}")
