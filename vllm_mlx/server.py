# SPDX-License-Identifier: Apache-2.0
"""
Unified OpenAI-compatible API server for vllm-mlx.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM and MLLM (Multimodal Language Model) inference using MLX on Apple Silicon.

Supports two modes:
- Simple mode (default): Maximum throughput for single-user scenarios
- Batched mode: Continuous batching for multiple concurrent users

Features:
- Text-only LLM inference (mlx-lm)
- Multimodal MLLM inference with images and video (mlx-vlm)
- OpenAI-compatible chat/completions API
- Streaming responses
- MCP (Model Context Protocol) tool integration
- Tool calling (Qwen/Llama formats)

Usage:
    # Simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Batched mode (for multiple concurrent users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions (with multimodal support)
    - GET /v1/models - List available models
    - GET /health - Health check
    - GET /v1/mcp/tools - List MCP tools
    - GET /v1/mcp/servers - MCP server status
    - POST /v1/mcp/execute - Execute MCP tool
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import Match
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# Import from new modular API
# Re-export for backwards compatibility with tests
from .api.anthropic_adapter import anthropic_to_openai, openai_to_anthropic
from .api.anthropic_models import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicUsage,
)
from .metrics import metrics as _metrics
from .api.models import (
    AssistantMessage,  # noqa: F401
    ChatCompletionChoice,  # noqa: F401
    ChatCompletionChunk,  # noqa: F401
    ChatCompletionChunkChoice,  # noqa: F401
    ChatCompletionChunkDelta,  # noqa: F401
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,  # noqa: F401
    CompletionRequest,
    CompletionResponse,
    ContentPart,  # noqa: F401
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageUrl,  # noqa: F401
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,  # noqa: F401
    MCPServersResponse,
    MCPToolInfo,  # noqa: F401
    MCPToolsResponse,
    Message,  # noqa: F401
    ModelInfo,  # noqa: F401
    ModelsResponse,
    ToolCall,
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.responses_models import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseFunctionTool,
    ResponseIncompleteDetails,
    ResponseInProgressEvent,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseReasoningTextPart,
    ResponseTextContentPart,
    ResponsesRequest,
    ResponsesUsage,
)
from .api.tool_calling import (
    StreamingJsonFenceStripper,
    build_json_system_prompt,
    convert_tools_for_template,
    parse_json_output,
    parse_tool_calls,
)
from .api.utils import (
    SPECIAL_TOKENS_PATTERN,
    clean_output_text,
    extract_multimodal_content,
    is_mllm_model,  # noqa: F401
)
from .engine import BaseEngine, BatchedEngine, GenerationOutput, SimpleEngine
from .api.tool_calling import build_json_logits_processor
from .message_utils import _normalize_messages
from .model_registry import (
    ModelLease,
    ModelManager,
    RegistryServeDefaults,
    ServingProfile,
    load_registry_config,
)
from .reasoning import get_parser as get_reasoning_parser
from .tool_parsers import ToolParserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: BaseEngine | None = None
_model_manager: ModelManager | None = None
_model_name: str | None = None
_model_path: str | None = (
    None  # Actual model path (for cache dir, not affected by --served-model-name)
)
_default_max_tokens: int = 32768
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p
_default_top_k: int | None = None  # Set via --default-top-k
_default_min_p: float | None = None  # Set via --default-min-p
_default_presence_penalty: float | None = None  # Set via --default-presence-penalty
_default_repetition_penalty: float | None = None  # Set via --default-repetition-penalty
_default_thinking_token_budget: int | None = (
    None  # Set via --default-thinking-token-budget
)

_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


def _resolve_temperature(request_value: float | None) -> float:
    """Resolve temperature: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_temperature is not None:
        return _default_temperature
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None) -> float:
    """Resolve top_p: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_top_p is not None:
        return _default_top_p
    return _FALLBACK_TOP_P


def _resolve_top_k(request_value: int | None) -> int:
    """Resolve top_k: request > CLI default > fallback (0 = disabled)."""
    if request_value is not None:
        return request_value
    if _default_top_k is not None:
        return _default_top_k
    return 0


def _resolve_min_p(request_value: float | None) -> float:
    """Resolve min_p: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_min_p is not None:
        return _default_min_p
    return 0.0


def _resolve_presence_penalty(request_value: float | None) -> float:
    """Resolve presence_penalty: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_presence_penalty is not None:
        return _default_presence_penalty
    return 0.0


def _resolve_repetition_penalty(request_value: float | None) -> float:
    """Resolve repetition_penalty: request > CLI default > fallback (1.0 = disabled)."""
    if request_value is not None:
        return request_value
    if _default_repetition_penalty is not None:
        return _default_repetition_penalty
    return 1.0


def _resolve_thinking_token_budget(request_value: int | None) -> int | None:
    """Resolve thinking token budget: request > CLI default > None."""
    if request_value is not None:
        return request_value
    return _default_thinking_token_budget


def _resolve_request_field(
    request: "ChatCompletionRequest | CompletionRequest",
    field_name: str,
    default,
):
    """Resolve request fields with OpenAI-compatible extra_body fallback."""
    value = getattr(request, field_name, None)
    if value is not None:
        return value
    extra_body = getattr(request, "extra_body", None) or {}
    extra_value = extra_body.get(field_name)
    if extra_value is not None:
        return extra_value
    return default


def _resolve_chat_template_kwargs(request: ChatCompletionRequest) -> dict[str, object]:
    """Merge top-level and extra_body chat template kwargs."""
    merged: dict[str, object] = {}
    extra_body = request.extra_body or {}
    extra_kwargs = extra_body.get("chat_template_kwargs")
    if isinstance(extra_kwargs, dict):
        merged.update(extra_kwargs)
    if request.chat_template_kwargs:
        merged.update(request.chat_template_kwargs)
    # Top-level enable_thinking takes precedence (client SDK compat)
    if request.enable_thinking is not None and "enable_thinking" not in merged:
        merged["enable_thinking"] = request.enable_thinking
    return merged


def _apply_serving_profile_defaults(
    request: ChatCompletionRequest,
    profile: ServingProfile,
) -> dict[str, object]:
    """Merge request template kwargs with per-model serving defaults."""
    merged = _resolve_chat_template_kwargs(request)
    if profile.enable_thinking_default is not None and "enable_thinking" not in merged:
        merged["enable_thinking"] = profile.enable_thinking_default
    if (
        profile.force_nonempty_content is not None
        and "force_nonempty_content" not in merged
    ):
        merged["force_nonempty_content"] = profile.force_nonempty_content
    return merged


# Minimum max_tokens when thinking is enabled. Thinking models need
# headroom for reasoning tokens before producing content. Per Unsloth
# docs (https://unsloth.ai/docs/models/qwen3.6): "Adequate Output
# Length: 32,768 tokens for most queries."
_THINKING_MIN_MAX_TOKENS = 16384


def _apply_thinking_max_tokens_floor(
    max_tokens: int,
    request,
    profile: ServingProfile,
) -> int:
    """Enforce a minimum max_tokens when thinking is enabled.

    When thinking mode is active (either from the request or the model's
    serving profile default), a low max_tokens causes the reasoning to
    consume the entire budget, leaving content as null.  This silently
    raises max_tokens to the thinking floor so clients don't have to
    know about thinking token budgets.
    """
    if max_tokens >= _THINKING_MIN_MAX_TOKENS:
        return max_tokens

    kwargs = _apply_serving_profile_defaults(request, profile)
    thinking_on = kwargs.get("enable_thinking")

    # If thinking is explicitly disabled by the client, respect their budget
    if thinking_on is False:
        return max_tokens

    # If thinking is on (default or explicit), enforce the floor
    if thinking_on is True or thinking_on is None:
        logger.debug(
            "Raising max_tokens from %d to %d (thinking mode active)",
            max_tokens,
            _THINKING_MIN_MAX_TOKENS,
        )
        return _THINKING_MIN_MAX_TOKENS

    return max_tokens


def _global_serving_profile() -> ServingProfile:
    """Serving profile for single-model mode."""
    return ServingProfile(
        enable_auto_tool_choice=_enable_auto_tool_choice,
        tool_call_parser=_tool_call_parser,
        reasoning_parser=_reasoning_parser_name,
    )


def _should_force_content_response(
    request: ChatCompletionRequest,
    profile: ServingProfile,
    model_source: str | None,
) -> bool:
    """Whether parser output should be coerced back into assistant content."""
    kwargs = _apply_serving_profile_defaults(request, profile)
    if kwargs.get("force_nonempty_content") is True:
        return True
    if kwargs.get("enable_thinking") is False:
        return True
    # Default-on for Qwen models: thinking output often traps the answer
    # in reasoning_content with empty visible content.
    if model_source and "qwen" in model_source.lower():
        return kwargs.get("force_nonempty_content") is not False
    return False


# Global MCP manager
_mcp_manager = None
_mcp_executor = None

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False

# Reasoning parser (for models like Qwen3, DeepSeek-R1)
_reasoning_parser = None  # ReasoningParser instance when enabled
_reasoning_parser_name: str | None = None

# Tool calling configuration
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None  # Parser name: auto, mistral, qwen, llama, hermes
_tool_parser_instance = None  # Test override hook; production builds fresh parsers
_responses_store: OrderedDict[str, dict] = OrderedDict()
_RESPONSES_STORE_MAX_SIZE: int = 1000


@dataclass
class RequestModelContext:
    """Request-scoped engine/lease context."""

    model_name: str
    engine: BaseEngine
    model_source: str | None = None
    serving_profile: ServingProfile = ServingProfile()
    lease: ModelLease | None = None

    async def release(self) -> None:
        if self.lease is not None:
            lease = self.lease
            self.lease = None
            await lease.release()


def _load_prefix_cache_from_disk() -> None:
    """Load prefix cache from disk during startup."""
    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Loading prefix cache from {d}")
        loaded = _engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info(f"[lifespan] Loaded {loaded} prefix cache entries")
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to load cache from disk: {e}", exc_info=True)


def _save_prefix_cache_to_disk() -> None:
    """Save prefix cache to disk during shutdown."""
    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Saving prefix cache to {d}")
        saved = _engine.save_cache_to_disk(d)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to save cache to disk: {e}", exc_info=True)


def _get_cache_dir() -> str:
    """Get cache persistence directory based on actual model path."""
    # Use _model_path (actual model path) not _model_name (which may be overridden
    # by --served-model-name). This ensures cache is shared regardless of served name.
    model_name = (
        _model_path if _model_path else (_model_name if _model_name else "default")
    )
    logger.info(
        f"[_get_cache_dir] _model_path={_model_path!r} type={type(_model_path)}"
    )
    # Sanitize model name for filesystem
    safe_name = str(model_name).replace("/", "--").replace("\\", "--")
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache", safe_name
    )
    logger.info(f"[_get_cache_dir] cache_dir={cache_dir!r}")
    return cache_dir


def _get_engine_tokenizer(engine) -> object | None:
    """Return the tokenizer backing ``engine``, if exposed.

    Different engine classes store the tokenizer under different attributes.
    We try the common ones and return ``None`` if nothing matches, so that
    optional features like constrained decoding can degrade gracefully.
    """
    for attr in ("_tokenizer", "tokenizer", "_processor", "processor"):
        tok = getattr(engine, attr, None)
        if tok is not None:
            return tok
    return None


async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    global _engine, _mcp_manager, _model_manager

    # Startup: Start engine if loaded (needed for BatchedEngine in uvicorn's event loop)
    if _engine is not None and hasattr(_engine, "_loaded") and not _engine._loaded:
        await _engine.start()
    if _model_manager is not None:
        await _model_manager.preload()

    # Load persisted cache from disk (AFTER engine start — AsyncEngineCore must exist)
    if _engine is not None and hasattr(_engine, "load_cache_from_disk"):
        _load_prefix_cache_from_disk()

    # Initialize MCP if config provided
    mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
    if mcp_config:
        await init_mcp(mcp_config)

    yield

    # Shutdown: Save cache to disk BEFORE stopping engine
    if _engine is not None and hasattr(_engine, "save_cache_to_disk"):
        _save_prefix_cache_to_disk()

    # Shutdown: Close MCP connections and stop engine
    if _mcp_manager is not None:
        await _mcp_manager.stop()
        logger.info("MCP manager stopped")
    if _engine is not None:
        await _engine.stop()
        logger.info("Engine stopped")
    if _model_manager is not None:
        await _model_manager.shutdown()
        logger.info("Model manager stopped")


app = FastAPI(
    title="vllm-mlx API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.2.1",
    lifespan=lifespan,
)

security = HTTPBearer(auto_error=False)


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


# Global rate limiter (disabled by default)
_rate_limiter = RateLimiter(requests_per_minute=60, enabled=False)


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    # Use API key as client ID if available, otherwise use IP
    client_id = request.headers.get(
        "Authorization", request.client.host if request.client else "unknown"
    )

    allowed, retry_after = _rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if authentication is enabled."""
    global _auth_warning_logged

    if _api_key is None:
        # Log warning once about running without authentication
        if not _auth_warning_logged:
            logger.warning(
                "SECURITY WARNING: Server running without API key authentication. "
                "Anyone can access the API. Use --api-key to enable authentication."
            )
            _auth_warning_logged = True
        return True  # No auth required

    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def _metrics_result_from_status(status_code: int) -> str:
    """Map HTTP-ish status codes to low-cardinality inference results."""
    if status_code == 499:
        return "client_closed"
    if status_code == 504:
        return "timeout"
    if status_code >= 500:
        return "error"
    return "success"


def _metrics_path_for_request(request: Request) -> str:
    """Prefer route templates over raw URLs to keep metrics cardinality bounded."""
    route = request.scope.get("route")
    if route is not None:
        path = getattr(route, "path", None)
        if path:
            return str(path)
    for candidate in app.router.routes:
        match, _ = candidate.matches(request.scope)
        if match in (Match.FULL, Match.PARTIAL):
            path = getattr(candidate, "path", None)
            if path:
                return str(path)
    return "__unmatched__"


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Capture generic HTTP request metrics when enabled."""
    if not _metrics.enabled:
        return await call_next(request)

    method = request.method
    path = _metrics_path_for_request(request)
    if path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    _metrics.observe_http_start(method=method, path=path)
    try:
        response = await call_next(request)
    except Exception:
        _metrics.observe_http_finish(
            method=method,
            path=path,
            status_code=500,
            duration=time.perf_counter() - start_time,
        )
        raise

    _metrics.observe_http_finish(
        method=method,
        path=path,
        status_code=response.status_code,
        duration=time.perf_counter() - start_time,
    )
    return response


def get_engine() -> BaseEngine:
    """Get the loaded engine, raising error if not loaded."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _engine


def _list_available_model_names() -> list[str]:
    if _model_manager is not None:
        return sorted(_model_manager._registry.keys())
    return [_model_name] if _model_name else []


def _coerce_tool_arguments(
    arguments_json: str, tool_name: str, tools: list[dict] | None
) -> str:
    """
    Coerce tool call arguments to match the tool schema.

    If a schema field expects "string" but the model produced an object/array,
    JSON-stringify the value. This fixes a common LLM failure mode where models
    output raw JSON objects instead of JSON strings for file content, etc.
    """
    if not tools:
        return arguments_json

    schema = None
    for tool in tools:
        if isinstance(tool, dict) and tool.get("function", {}).get("name") == tool_name:
            schema = tool["function"].get("parameters", {})
            break

    if not schema or "properties" not in schema:
        return arguments_json

    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return arguments_json

    if not isinstance(arguments, dict):
        return arguments_json

    properties = schema.get("properties", {})
    changed = False

    for key, value in arguments.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type == "string" and isinstance(value, (dict, list)):
                arguments[key] = json.dumps(value, ensure_ascii=False, indent=2)
                changed = True

    if changed:
        return json.dumps(arguments, ensure_ascii=False)

    return arguments_json


def _validate_model_name(request_model: str) -> None:
    """Validate that the request model name matches the served model.

    Accepts 'default' as an alias for the loaded model (used by lm_eval's
    local-chat-completions backend which sends model='default').
    """
    if request_model == "default":
        return
    if _model_manager is not None:
        if not _model_manager.has_model(request_model):
            available = ", ".join(f"`{name}`" for name in _list_available_model_names())
            raise HTTPException(
                status_code=404,
                detail=(
                    f"The model `{request_model}` does not exist. "
                    f"Available models: {available}"
                ),
            )
        return
    if _model_name and request_model != _model_name:
        raise HTTPException(
            status_code=404,
            detail=f"The model `{request_model}` does not exist. "
            f"Available model: `{_model_name}`",
        )


async def _acquire_request_model(request_model: str) -> RequestModelContext:
    """Acquire the model/engine that should serve this request."""
    _validate_model_name(request_model)

    if _model_manager is None:
        engine = get_engine()
        profile = _global_serving_profile()
        engine.preserve_native_tool_format = _detect_native_tool_support(profile)
        return RequestModelContext(
            model_name=_model_name or request_model,
            model_source=_model_path or _model_name or request_model,
            engine=engine,
            serving_profile=profile,
        )

    try:
        lease = await _model_manager.acquire(request_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    lease.engine.preserve_native_tool_format = _detect_native_tool_support(
        lease.config.serving_profile
    )
    return RequestModelContext(
        model_name=request_model,
        model_source=lease.config.resolved_source,
        engine=lease.engine,
        serving_profile=lease.config.serving_profile,
        lease=lease,
    )


async def _stream_with_model_context(
    context: RequestModelContext,
    stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Ensure model leases survive for the full streaming response."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await context.release()


def _build_tool_parser(engine: BaseEngine | None, profile: ServingProfile):
    """Create a fresh tool parser instance for a single request/stream."""
    if not profile.enable_auto_tool_choice or not profile.tool_call_parser:
        return None

    if _tool_parser_instance is not None:
        if hasattr(_tool_parser_instance, "reset"):
            _tool_parser_instance.reset()
        return _tool_parser_instance

    parser_cls = ToolParserManager.get_tool_parser(profile.tool_call_parser)
    tokenizer = getattr(engine, "tokenizer", None) if engine is not None else None
    return parser_cls(tokenizer)


def _build_reasoning_parser(
    engine: BaseEngine | None,
    profile: ServingProfile,
):
    """Create a fresh reasoning parser instance for a single request/stream."""
    tokenizer = getattr(engine, "tokenizer", None) if engine is not None else None
    if profile.reasoning_parser is not None:
        parser_cls = get_reasoning_parser(profile.reasoning_parser)
        try:
            return parser_cls(tokenizer)
        except TypeError:
            return parser_cls()
    if _reasoning_parser is None:
        return None
    try:
        return type(_reasoning_parser)(tokenizer)
    except TypeError:
        return type(_reasoning_parser)()


def _parse_tool_calls_with_parser(
    output_text: str,
    request: ChatCompletionRequest | None = None,
    *,
    engine: BaseEngine | None = None,
    profile: ServingProfile | None = None,
) -> tuple[str, list | None]:
    """
    Parse tool calls from model output using the configured parser.

    If --enable-auto-tool-choice is set with --tool-call-parser, uses the
    selected parser. Otherwise falls back to the generic parse_tool_calls.

    Args:
        output_text: The model output text
        request: The original request (for context)

    Returns:
        Tuple of (cleaned_text, tool_calls)
    """
    request_dict = request.model_dump() if request else None

    # tool_choice="none" means never return tool calls — skip all parsing
    if request is not None:
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is None and request_dict:
            tool_choice = request_dict.get("tool_choice")
        if tool_choice == "none":
            return output_text, None

    # If auto tool choice is not enabled, use the generic parser
    if profile is None:
        profile = _global_serving_profile()

    if not profile.enable_auto_tool_choice or not profile.tool_call_parser:
        return parse_tool_calls(output_text, request_dict)

    try:
        parser = _build_tool_parser(engine, profile)
        if parser is None:
            return parse_tool_calls(output_text, request_dict)
        result = parser.extract_tool_calls(output_text, request_dict)
        if result.tools_called:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in result.tool_calls
            ]
            return result.content or "", tool_calls
        else:
            # Fallback: specific parser didn't find tool calls,
            # try generic parser which handles more formats (e.g. Nemotron XML)
            return parse_tool_calls(output_text, request_dict)
    except Exception as e:
        logger.warning(f"Tool parser error: {e}")
        return parse_tool_calls(output_text, request_dict)


def _new_response_item_id(prefix: str) -> str:
    """Generate stable OpenAI-style item ids."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _response_content_to_text(content) -> str:
    """Normalize Responses API content items into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    text_parts = []
    for part in content:
        if isinstance(part, dict):
            part_type = part.get("type")
            text = part.get("text", "")
        else:
            part_type = getattr(part, "type", None)
            text = getattr(part, "text", "")
        if part_type in {"text", "input_text", "output_text"}:
            text_parts.append(text)
    return "\n".join(part for part in text_parts if part)


def _responses_tools_to_chat_tools(
    tools: list[ResponseFunctionTool | dict],
) -> tuple[list[dict] | None, list[str]]:
    """Convert supported Responses tools and report unsupported tool types."""
    if not tools:
        return None, []

    supported: list[dict] = []
    unsupported: list[str] = []

    for tool in tools:
        if isinstance(tool, ResponseFunctionTool):
            tool_type = tool.type
            tool_name = tool.name
            tool_description = tool.description or ""
            tool_parameters = tool.parameters
        elif isinstance(tool, dict):
            tool_type = tool.get("type", "unknown")
            tool_name = tool.get("name", "")
            tool_description = tool.get("description", "")
            tool_parameters = tool.get("parameters", {})
        else:
            unsupported.append(type(tool).__name__)
            continue

        if tool_type == "function":
            supported.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": tool_parameters
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        else:
            unsupported.append(tool_type)

    return supported or None, unsupported


def _responses_input_to_chat_messages(request: ResponsesRequest) -> list[dict]:
    """Convert Responses API input items into chat-completions-style messages."""
    messages: list[dict] = []

    if request.previous_response_id:
        previous = _responses_store.get(request.previous_response_id)
        if previous is None:
            raise HTTPException(
                status_code=404,
                detail=f"Previous response `{request.previous_response_id}` not found",
            )
        messages.extend(copy.deepcopy(previous["messages"]))

    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
        return messages

    for item in request.input:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "message":
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                messages.append(
                    {
                        "role": role,
                        "content": _response_content_to_text(item.get("content")),
                    }
                )
            elif item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get(
                                    "call_id", _new_response_item_id("call")
                                ),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", ""),
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )
            elif item_type == "reasoning":
                parts = item.get("content", [])
                reasoning_text = "\n".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                )
                if reasoning_text:
                    messages.append({"role": "assistant", "content": reasoning_text})
            else:
                logger.info(
                    "Skipping unsupported Responses input item type %r", item_type
                )
            continue

        if isinstance(item, ResponseMessageItem):
            role = item.role
            if role == "developer":
                role = "system"
            messages.append(
                {
                    "role": role,
                    "content": _response_content_to_text(item.content),
                }
            )
        elif isinstance(item, ResponseFunctionCallItem):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        }
                    ],
                }
            )
        elif isinstance(item, ResponseFunctionCallOutputItem):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output,
                }
            )
        elif isinstance(item, ResponseReasoningItem):
            reasoning_text = "\n".join(part.text for part in (item.content or []))
            if reasoning_text:
                messages.append({"role": "assistant", "content": reasoning_text})
        else:
            logger.info(
                "Skipping unsupported Responses input item type %r",
                getattr(item, "type", type(item).__name__),
            )

    return messages


def _responses_request_to_new_persisted_messages(
    request: ResponsesRequest,
) -> list[dict]:
    """Persist only the current request's replayable input items."""
    request_without_history = request.model_copy(
        update={"previous_response_id": None, "instructions": None},
        deep=True,
    )
    return _responses_input_to_chat_messages(request_without_history)


def _responses_request_to_persisted_messages(request: ResponsesRequest) -> list[dict]:
    """Persist replayable history for chained previous_response_id requests.

    Responses `instructions` are intentionally not replayed across
    `previous_response_id`, but replayable message items are.
    """
    messages: list[dict] = []
    if request.previous_response_id:
        previous = _responses_store.get(request.previous_response_id)
        if previous is None:
            raise HTTPException(
                status_code=404,
                detail=f"Previous response `{request.previous_response_id}` not found",
            )
        messages.extend(copy.deepcopy(previous["messages"]))
    messages.extend(_responses_request_to_new_persisted_messages(request))
    return messages


def _responses_request_to_chat_request(
    request: ResponsesRequest,
) -> ChatCompletionRequest:
    """Build a ChatCompletionRequest from a ResponsesRequest."""
    if request.text.format.type == "json_object":
        raise HTTPException(
            status_code=400,
            detail="Responses text.format.type='json_object' is not supported on this backend",
        )
    if request.reasoning is not None:
        logger.debug("Ignoring reasoning configuration (not supported on this backend)")

    tools, unsupported_tools = _responses_tools_to_chat_tools(request.tools)
    messages = _responses_input_to_chat_messages(request)
    if unsupported_tools:
        tool_list = ", ".join(sorted(set(unsupported_tools)))
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "The following requested tool types are not available on this "
                    f"backend: {tool_list}. Do not call them."
                ),
            },
        )

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
    merged_system_content = "\n\n".join(
        str(msg.get("content", "")).strip()
        for msg in system_messages
        if str(msg.get("content", "")).strip()
    )
    messages = (
        [{"role": "system", "content": merged_system_content}]
        if merged_system_content
        else []
    ) + non_system_messages

    return ChatCompletionRequest(
        model=request.model,
        messages=[Message(**msg) for msg in messages],
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=False,
        tools=tools,
        tool_choice=request.tool_choice,
    )


def _build_responses_output_items(
    text: str | None,
    reasoning: str | None,
    tool_calls: list[ToolCall] | None,
) -> list[ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem]:
    """Convert parsed assistant output into Responses API output items."""
    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ] = []

    if reasoning:
        output_items.append(
            ResponseReasoningItem(
                id=_new_response_item_id("rs"),
                content=[ResponseReasoningTextPart(text=reasoning)],
            )
        )

    if text:
        output_items.append(
            ResponseMessageItem(
                id=_new_response_item_id("msg"),
                role="assistant",
                content=[ResponseTextContentPart(type="output_text", text=text)],
            )
        )

    for tool_call in tool_calls or []:
        output_items.append(
            ResponseFunctionCallItem(
                id=_new_response_item_id("fc"),
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
        )

    return output_items


def _response_output_items_to_chat_messages(output_items: list) -> list[dict]:
    """Persist assistant output in chat-completions form for previous_response_id."""
    assistant_text_parts: list[str] = []
    assistant_tool_calls: list[dict] = []

    for item in output_items:
        if isinstance(item, ResponseMessageItem):
            assistant_text_parts.append(_response_content_to_text(item.content))
        elif isinstance(item, ResponseFunctionCallItem):
            assistant_tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )

    if not assistant_text_parts and not assistant_tool_calls:
        return []

    return [
        {
            "role": "assistant",
            "content": "".join(assistant_text_parts),
            "tool_calls": assistant_tool_calls or None,
        }
    ]


def _build_response_object(
    request: ResponsesRequest,
    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ],
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
    response_id: str | None = None,
) -> ResponseObject:
    """Build a full Responses API object."""
    response = ResponseObject(
        id=response_id or _new_response_item_id("resp"),
        model=_model_name or request.model,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        metadata=request.metadata,
        output=output_items,
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_p=_resolve_top_p(request.top_p),
        temperature=_resolve_temperature(request.temperature),
        truncation=request.truncation,
        user=request.user,
        store=request.store,
        usage=ResponsesUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    if finish_reason == "length":
        response.status = "incomplete"
        response.incomplete_details = ResponseIncompleteDetails(
            reason="max_output_tokens"
        )
    return response


def _prepare_responses_request(
    request: ResponsesRequest,
) -> tuple[BaseEngine, ChatCompletionRequest, list[dict], dict]:
    """Prepare a Responses request for execution on the chat engine."""
    _validate_model_name(request.model)
    engine = get_engine()
    chat_request = _responses_request_to_chat_request(request)

    if chat_request.messages:
        logger.info(
            f"[REQUEST] POST /v1/responses stream={request.stream} "
            f"model={request.model!r} items="
            f"{len(request.input) if isinstance(request.input, list) else 1} "
            f"tools={len(request.tools)}"
        )

    messages, images, videos = extract_multimodal_content(
        chat_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": chat_request.max_tokens or _default_max_tokens,
        "temperature": _resolve_temperature(chat_request.temperature),
        "top_p": _resolve_top_p(chat_request.top_p),
        "top_k": _resolve_top_k(None),
        "min_p": _resolve_min_p(None),
        "presence_penalty": _resolve_presence_penalty(None),
        "repetition_penalty": _resolve_repetition_penalty(None),
    }
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(chat_request.tools)
    if images:
        chat_kwargs["images"] = images
    if videos:
        chat_kwargs["videos"] = videos

    return engine, chat_request, messages, chat_kwargs


async def _run_responses_request(
    request: ResponsesRequest,
    raw_request: Request,
) -> tuple[ResponseObject | None, list[dict]]:
    """Execute a Responses API request against the backend chat engine."""
    engine, chat_request, messages, chat_kwargs = _prepare_responses_request(request)

    # Request raw output so reasoning/tool parsers see unstripped text.
    # The server applies clean_output_text() itself after parsing.
    chat_kwargs["raw_output"] = True

    timeout = _default_timeout
    output = await _wait_with_disconnect(
        engine.chat(messages=messages, **chat_kwargs),
        raw_request,
        timeout=timeout,
    )
    if output is None:
        return None, []

    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, chat_request)
    reasoning_text = None
    if _reasoning_parser and not tool_calls:
        reasoning_text, cleaned_text = _reasoning_parser.extract_reasoning(
            cleaned_text or output.text
        )

    output_items = _build_responses_output_items(
        clean_output_text(cleaned_text) if cleaned_text else None,
        reasoning_text,
        tool_calls,
    )
    response_object = _build_response_object(
        request=request,
        output_items=output_items,
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        finish_reason=output.finish_reason,
    )

    persisted_messages = _responses_request_to_persisted_messages(request)
    persisted_messages.extend(_response_output_items_to_chat_messages(output_items))
    if request.store:
        _responses_store[response_object.id] = {
            "messages": copy.deepcopy(persisted_messages),
            "response": response_object.model_copy(deep=True),
        }
        while len(_responses_store) > _RESPONSES_STORE_MAX_SIZE:
            _responses_store.popitem(last=False)

    return response_object, persisted_messages


async def _stream_responses_request(request: ResponsesRequest) -> AsyncIterator[str]:
    """Execute a Responses API request and stream SSE events incrementally."""
    engine, chat_request, messages, chat_kwargs = _prepare_responses_request(request)

    response_id = _new_response_item_id("resp")
    sequence = 1
    base_response = _build_response_object(
        request=request,
        output_items=[],
        prompt_tokens=0,
        completion_tokens=0,
        finish_reason=None,
        response_id=response_id,
    )
    base_response.status = "in_progress"
    base_response.usage = None

    yield _responses_sse_event(
        "response.created",
        ResponseCreatedEvent(sequence_number=sequence, response=base_response),
    )
    sequence += 1
    yield _responses_sse_event(
        "response.in_progress",
        ResponseInProgressEvent(sequence_number=sequence, response=base_response),
    )
    sequence += 1

    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = None
    last_output = None
    raw_accumulated_text = ""
    accumulated_text = ""
    accumulated_reasoning = ""

    text_item_id: str | None = None
    text_output_index: int | None = None
    reasoning_item_id: str | None = None
    reasoning_output_index: int | None = None
    next_output_index = 0

    def _start_text_item() -> list[str]:
        nonlocal text_item_id, text_output_index, next_output_index, sequence
        events: list[str] = []
        if text_item_id is None:
            text_item_id = _new_response_item_id("msg")
            text_output_index = next_output_index
            next_output_index += 1
            events.append(
                _responses_sse_event(
                    "response.output_item.added",
                    ResponseOutputItemAddedEvent(
                        sequence_number=sequence,
                        output_index=text_output_index,
                        item=ResponseMessageItem(
                            id=text_item_id,
                            role="assistant",
                            status="in_progress",
                            content=[],
                        ),
                    ),
                )
            )
            sequence += 1
            events.append(
                _responses_sse_event(
                    "response.content_part.added",
                    ResponseContentPartAddedEvent(
                        sequence_number=sequence,
                        item_id=text_item_id,
                        output_index=text_output_index,
                        content_index=0,
                        part=ResponseTextContentPart(type="output_text", text=""),
                    ),
                )
            )
            sequence += 1
        return events

    def _start_reasoning_item() -> list[str]:
        nonlocal reasoning_item_id, reasoning_output_index, next_output_index, sequence
        events: list[str] = []
        if reasoning_item_id is None:
            reasoning_item_id = _new_response_item_id("rs")
            reasoning_output_index = next_output_index
            next_output_index += 1
            events.append(
                _responses_sse_event(
                    "response.output_item.added",
                    ResponseOutputItemAddedEvent(
                        sequence_number=sequence,
                        output_index=reasoning_output_index,
                        item=ResponseReasoningItem(
                            id=reasoning_item_id,
                            status="in_progress",
                            content=[],
                        ),
                    ),
                )
            )
            sequence += 1
            events.append(
                _responses_sse_event(
                    "response.content_part.added",
                    ResponseContentPartAddedEvent(
                        sequence_number=sequence,
                        item_id=reasoning_item_id,
                        output_index=reasoning_output_index,
                        content_index=0,
                        part=ResponseReasoningTextPart(text=""),
                    ),
                )
            )
            sequence += 1
        return events

    if _reasoning_parser:
        _reasoning_parser.reset_state()

    global _tool_parser_instance
    tool_parser = None
    tool_accumulated_text = ""
    tool_markup_possible = False
    if _enable_auto_tool_choice and _tool_call_parser:
        if _tool_parser_instance is None:
            try:
                parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
                tokenizer = None
                if _engine is not None and hasattr(_engine, "_tokenizer"):
                    tokenizer = _engine._tokenizer
                _tool_parser_instance = parser_cls(tokenizer)
                logger.info(
                    "Initialized tool call parser for responses streaming: %s",
                    _tool_call_parser,
                )
            except Exception as e:
                logger.warning(
                    "Failed to init tool parser for responses streaming: %s", e
                )
        if _tool_parser_instance is not None:
            tool_parser = _tool_parser_instance
            tool_parser.reset()

    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        last_output = output
        finish_reason = output.finish_reason
        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        delta_text = output.new_text or ""
        if not delta_text:
            continue

        previous_text = raw_accumulated_text
        raw_accumulated_text += delta_text

        if _reasoning_parser:
            delta_msg = _reasoning_parser.extract_reasoning_streaming(
                previous_text, raw_accumulated_text, delta_text
            )
            if delta_msg is None:
                continue

            if delta_msg.reasoning:
                for event in _start_reasoning_item():
                    yield event
                accumulated_reasoning += delta_msg.reasoning
                yield _responses_sse_event(
                    "response.reasoning_text.delta",
                    ResponseReasoningTextDeltaEvent(
                        sequence_number=sequence,
                        item_id=reasoning_item_id,
                        output_index=reasoning_output_index,
                        content_index=0,
                        delta=delta_msg.reasoning,
                    ),
                )
                sequence += 1

            if delta_msg.content:
                for event in _start_text_item():
                    yield event
                accumulated_text += delta_msg.content
                yield _responses_sse_event(
                    "response.output_text.delta",
                    ResponseOutputTextDeltaEvent(
                        sequence_number=sequence,
                        item_id=text_item_id,
                        output_index=text_output_index,
                        content_index=0,
                        delta=delta_msg.content,
                    ),
                )
                sequence += 1
            continue

        content = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
        if tool_parser and delta_text:
            if not tool_markup_possible and "<" not in delta_text:
                tool_accumulated_text += delta_text
            else:
                if not tool_markup_possible:
                    tool_markup_possible = True
                tool_result = tool_parser.extract_tool_calls_streaming(
                    tool_accumulated_text,
                    tool_accumulated_text + delta_text,
                    delta_text,
                )
                tool_accumulated_text += delta_text
                if tool_result is None:
                    continue
                if "tool_calls" in tool_result:
                    continue
                content = tool_result.get("content", "")

        if not content:
            continue

        for event in _start_text_item():
            yield event
        accumulated_text += content
        yield _responses_sse_event(
            "response.output_text.delta",
            ResponseOutputTextDeltaEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                delta=content,
            ),
        )
        sequence += 1

    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        raw_accumulated_text, chat_request
    )
    final_text = accumulated_text
    if cleaned_text is not None and not final_text and not tool_calls:
        final_text = clean_output_text(cleaned_text)

    reasoning_item = None
    if reasoning_item_id is not None:
        reasoning_item = ResponseReasoningItem(
            id=reasoning_item_id,
            status="completed",
            content=[ResponseReasoningTextPart(text=accumulated_reasoning)],
        )
        yield _responses_sse_event(
            "response.reasoning_text.done",
            ResponseReasoningTextDoneEvent(
                sequence_number=sequence,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                content_index=0,
                text=accumulated_reasoning,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.content_part.done",
            ResponseContentPartDoneEvent(
                sequence_number=sequence,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                content_index=0,
                part=reasoning_item.content[0],
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=reasoning_output_index,
                item=reasoning_item,
            ),
        )
        sequence += 1

    text_item = None
    if text_item_id is not None or final_text:
        if text_item_id is None:
            for event in _start_text_item():
                yield event
        text_item = ResponseMessageItem(
            id=text_item_id,
            role="assistant",
            status="completed",
            content=[ResponseTextContentPart(type="output_text", text=final_text)],
        )
        yield _responses_sse_event(
            "response.output_text.done",
            ResponseOutputTextDoneEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                text=final_text,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.content_part.done",
            ResponseContentPartDoneEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                part=text_item.content[0],
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=text_output_index,
                item=text_item,
            ),
        )
        sequence += 1

    function_call_items: list[ResponseFunctionCallItem] = []
    for tool_call in tool_calls or []:
        output_index = next_output_index
        next_output_index += 1
        item = ResponseFunctionCallItem(
            id=_new_response_item_id("fc"),
            call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        function_call_items.append(item)
        yield _responses_sse_event(
            "response.output_item.added",
            ResponseOutputItemAddedEvent(
                sequence_number=sequence,
                output_index=output_index,
                item=item.model_copy(update={"status": "in_progress"}),
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.function_call_arguments.delta",
            ResponseFunctionCallArgumentsDeltaEvent(
                sequence_number=sequence,
                item_id=item.id,
                output_index=output_index,
                delta=item.arguments,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=output_index,
                item=item,
            ),
        )
        sequence += 1

    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ] = []
    if reasoning_item is not None:
        output_items.append(reasoning_item)
    if text_item is not None:
        output_items.append(text_item)
    output_items.extend(function_call_items)

    response_object = _build_response_object(
        request=request,
        output_items=output_items,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        response_id=response_id,
    )

    if request.store and last_output is not None:
        persisted_messages = _responses_request_to_persisted_messages(request)
        persisted_messages.extend(_response_output_items_to_chat_messages(output_items))
        _responses_store[response_object.id] = {
            "messages": copy.deepcopy(persisted_messages),
            "response": response_object.model_copy(deep=True),
        }
        while len(_responses_store) > _RESPONSES_STORE_MAX_SIZE:
            _responses_store.popitem(last=False)

    yield _responses_sse_event(
        "response.completed",
        ResponseCompletedEvent(sequence_number=sequence, response=response_object),
    )


def _responses_sse_event(event_type: str, payload: BaseModel | dict) -> str:
    """Encode a Responses API SSE event."""
    data = (
        payload.model_dump_json()
        if isinstance(payload, BaseModel)
        else json.dumps(payload)
    )
    return f"event: {event_type}\ndata: {data}\n\n"


def _detect_native_tool_support(profile: ServingProfile) -> bool:
    """
    Detect if the active tool parser supports native tool format.

    Native format means role="tool" messages and tool_calls fields
    are preserved instead of being converted to text.

    Returns:
        True if native format should be preserved
    """
    if not profile.enable_auto_tool_choice or not profile.tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(profile.tool_call_parser)
        return parser_cls.supports_native_format()
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{profile.tool_call_parser}' not found. "
            f"Available parsers: {ToolParserManager.list_registered()}"
        )
        return False
    except Exception as e:
        # Unexpected error during detection
        logger.warning(f"Failed to detect native tool support: {e}")
        return False


def load_embedding_model(
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
) -> None:
    """Load or reuse the embedding model engine when configured."""
    global _embedding_engine, _embedding_model_locked

    if not model_name:
        return

    if lock:
        _embedding_model_locked = model_name

    if (
        reuse_existing
        and _embedding_engine is not None
        and _embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    _embedding_engine = EmbeddingEngine(model_name)
    _embedding_engine.load()


def load_model(
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    force_mllm: bool = False,
    served_model_name: str | None = None,
    mtp: bool = False,
    prefill_step_size: int = 2048,
    specprefill_enabled: bool = False,
    specprefill_threshold: int = 8192,
    specprefill_keep_pct: float = 0.3,
    specprefill_draft_model: str = None,
    trust_remote_code: bool = False,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        force_mllm: Force loading as MLLM even if not auto-detected
        mtp: Enable native MTP speculative decoding (SimpleEngine only)
        prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
        specprefill_enabled: Enable SpecPrefill (SimpleEngine only)
        specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill (default: 8192)
        specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
        specprefill_draft_model: Path to small draft model for SpecPrefill scoring
    """
    global _engine, _model_manager, _model_name, _model_path, _default_max_tokens

    _default_max_tokens = max_tokens
    _model_manager = None
    _model_path = model_name
    _model_name = served_model_name or model_name

    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")

    if use_batching:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        _engine = BatchedEngine(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            mtp=mtp,
            prefill_step_size=prefill_step_size,
            specprefill_enabled=specprefill_enabled,
            specprefill_draft_model_path=specprefill_draft_model,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            trust_remote_code=trust_remote_code,
        )
        # BatchedEngine will be started in lifespan (uvicorn's event loop)
        # Just log for now
        logger.info(f"Model loaded (batched mode): {model_name}")
    else:
        logger.info(f"Loading model with SimpleEngine: {model_name}")
        _engine = SimpleEngine(
            model_name=model_name,
            force_mllm=force_mllm,
            mtp=mtp,
            prefill_step_size=prefill_step_size,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_draft_model=specprefill_draft_model,
            trust_remote_code=trust_remote_code,
        )
        # Start SimpleEngine synchronously (no background loop)
        # Use new_event_loop() for Python 3.10+ compatibility (get_event_loop() is deprecated)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_engine.start())
        model_type = "MLLM" if _engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")

    # Set native tool format support on the engine (thread-safe via instance property)
    _engine.preserve_native_tool_format = _detect_native_tool_support(
        _global_serving_profile()
    )
    if _engine.preserve_native_tool_format:
        logger.info(
            "Native tool format enabled for parser: %s",
            _global_serving_profile().tool_call_parser,
        )

    logger.info(f"Default max tokens: {_default_max_tokens}")


def load_model_registry(
    config_path: str,
    *,
    defaults: RegistryServeDefaults,
) -> None:
    """Load a registry-backed model manager from YAML configuration."""
    global _engine, _model_manager, _model_name, _model_path, _default_max_tokens

    manager_config, registry = load_registry_config(config_path, defaults)
    _engine = None
    _model_path = None
    _model_name = None
    _default_max_tokens = defaults.max_tokens
    _model_manager = ModelManager(manager_config, registry, defaults)

    logger.info(
        "Loaded models config: %s (%d models, %.1f GB budget)",
        config_path,
        len(registry),
        manager_config.memory_budget_bytes / (1024**3),
    )


def get_usage(output: GenerationOutput) -> Usage:
    """Extract usage metrics from GenerationOutput."""
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )


def _json_safe(value, depth: int = 0):
    if depth >= 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(), depth + 1)
        except Exception:
            return repr(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return repr(value)
    return repr(value)


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint (disabled by default)."""
    if not _metrics.enabled:
        raise HTTPException(status_code=404, detail="Metrics endpoint is disabled")

    payload, content_type = _metrics.render_metrics(
        engine=_engine,
        mcp_manager=_mcp_manager,
    )
    return Response(content=payload, headers={"Content-Type": content_type})


@app.get("/health")
async def health():
    """Health check endpoint."""
    mcp_info = None
    if _mcp_manager is not None:
        connected = sum(
            1 for s in _mcp_manager.get_server_status() if s.state.value == "connected"
        )
        total = len(_mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(_mcp_manager.get_all_tools()),
        }

    try:
        engine_stats = _json_safe(_engine.get_stats()) if _engine else {}
    except Exception as exc:
        logger.exception("Failed to collect health engine stats")
        engine_stats = {"engine_type": "unknown", "error": str(exc)}

    return {
        "status": "healthy",
        "model_loaded": _engine is not None or _model_manager is not None,
        "model_name": _model_name,
        "available_models": _list_available_model_names(),
        "model_type": "mllm" if (_engine and _engine.is_mllm) else "llm",
        "engine_type": engine_stats.get("engine_type", "unknown"),
        "mcp": mcp_info,
    }


@app.get("/v1/status")
async def status():
    """Real-time status with per-request details for debugging and monitoring."""
    if _model_manager is not None:
        return {
            "status": "running",
            "model_manager": {
                "memory_budget_gb": round(
                    _model_manager.memory_budget_bytes / (1024**3), 2
                ),
                "models": _model_manager.list_models(),
            },
        }
    if _engine is None:
        return {"status": "not_loaded", "model": None, "requests": []}

    try:
        stats = _json_safe(_engine.get_stats())
    except Exception as exc:
        logger.exception("Failed to collect engine status")
        return {
            "status": "degraded",
            "model": _model_name,
            "engine_type": None,
            "model_type": "mllm" if (_engine and _engine.is_mllm) else "llm",
            "error": str(exc),
            "requests": [],
        }

    return {
        "status": "running" if stats.get("running") else "stopped",
        "model": _model_name,
        "engine_type": stats.get("engine_type"),
        "model_type": "mllm" if (_engine and _engine.is_mllm) else "llm",
        "uptime_s": round(stats.get("uptime_seconds", 0), 1),
        "steps_executed": stats.get("steps_executed", 0),
        "num_running": stats.get("num_running", 0),
        "num_waiting": stats.get("num_waiting", 0),
        "total_requests_processed": stats.get("num_requests_processed", 0),
        "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
        "total_completion_tokens": stats.get("total_completion_tokens", 0),
        "metal": {
            "active_memory_gb": stats.get("metal_active_memory_gb"),
            "peak_memory_gb": stats.get("metal_peak_memory_gb"),
            "cache_memory_gb": stats.get("metal_cache_memory_gb"),
        },
        "cache": stats.get("memory_aware_cache")
        or stats.get("paged_cache")
        or stats.get("prefix_cache"),
        "engine": stats,
        "requests": stats.get("requests", []),
    }


@app.get("/v1/cache/stats")
async def cache_stats():
    """Get cache statistics for debugging and monitoring."""
    try:
        from mlx_vlm.utils import (
            get_multimodal_kv_cache_stats,
            get_pil_cache_stats,
            get_pixel_values_cache_stats,
        )

        return {
            "multimodal_kv_cache": get_multimodal_kv_cache_stats(),
            "pixel_values_cache": get_pixel_values_cache_stats(),
            "pil_image_cache": get_pil_cache_stats(),
        }
    except ImportError:
        return {"error": "Cache stats not available (mlx_vlm not loaded)"}


@app.delete("/v1/cache")
async def clear_cache():
    """Clear all caches."""
    try:
        from mlx_vlm.utils import (
            clear_multimodal_kv_cache,
            clear_pixel_values_cache,
        )

        clear_multimodal_kv_cache()
        clear_pixel_values_cache()
        return {
            "status": "cleared",
            "caches": ["multimodal_kv", "pixel_values", "pil_image"],
        }
    except ImportError:
        return {"error": "Cache clear not available (mlx_vlm not loaded)"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models."""
    models = []
    if _model_manager is not None:
        models.extend(ModelInfo(id=item["id"]) for item in _model_manager.list_models())
    elif _model_name:
        models.append(ModelInfo(id=_model_name))
    return ModelsResponse(data=models)


# =============================================================================
# Embeddings Endpoint
# =============================================================================


@app.post(
    "/v1/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).

    OpenAI-compatible embeddings API supporting single or batch inputs.

    Single text:
    ```json
    {
      "model": "mlx-community/all-MiniLM-L6-v2-4bit",
      "input": "The quick brown fox jumps over the lazy dog"
    }
    ```

    Batch of texts:
    ```json
    {
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "input": [
        "I love machine learning",
        "Deep learning is fascinating",
        "Neural networks are powerful"
      ]
    }
    ```

    Response:
    ```json
    {
      "object": "list",
      "data": [
        {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
        {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]},
        {"object": "embedding", "index": 2, "embedding": [0.876, 0.221, ...]}
      ],
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "usage": {"prompt_tokens": 24, "total_tokens": 24}
    }
    ```

    Supported models:
    - mlx-community/all-MiniLM-L6-v2-4bit (fast, compact)
    - mlx-community/embeddinggemma-300m-6bit (high quality)
    - mlx-community/bge-large-en-v1.5-4bit (best for English)
    - Any BERT/XLM-RoBERTa/ModernBERT model from HuggingFace
    """
    global _embedding_engine

    try:
        # Resolve model name
        model_name = request.model

        # If an embedding model was pre-configured at startup, only allow that model
        if (
            _embedding_model_locked is not None
            and model_name != _embedding_model_locked
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding model '{model_name}' is not available. "
                    f"This server was started with --embedding-model {_embedding_model_locked}. "
                    f"Only '{_embedding_model_locked}' can be used for embeddings. "
                    f"Restart the server with a different --embedding-model to use '{model_name}'."
                ),
            )

        # Lazy-load or swap embedding engine
        load_embedding_model(model_name, lock=False, reuse_existing=True)

        # Normalise input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        start_time = time.perf_counter()

        # Count tokens for usage reporting
        prompt_tokens = _embedding_engine.count_tokens(texts)

        # Generate embeddings (batch)
        embeddings = _embedding_engine.embed(texts)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens in {elapsed:.2f}s"
        )

        # Build OpenAI-compatible response with ordered indices
        data = [
            EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "mlx-embeddings not installed. Install with: pip install mlx-embeddings"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MCP Endpoints
# =============================================================================


@app.get("/v1/mcp/tools", dependencies=[Depends(verify_api_key)])
async def list_mcp_tools() -> MCPToolsResponse:
    """List all available MCP tools."""
    if _mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in _mcp_manager.get_all_tools():
        tools.append(
            MCPToolInfo(
                name=tool.full_name,
                description=tool.description,
                server=tool.server_name,
                parameters=tool.input_schema,
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


@app.get("/v1/mcp/servers", dependencies=[Depends(verify_api_key)])
async def list_mcp_servers() -> MCPServersResponse:
    """Get status of all MCP servers."""
    if _mcp_manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in _mcp_manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status.name,
                state=status.state.value,
                transport=status.transport.value,
                tools_count=status.tools_count,
                error=status.error,
            )
        )

    return MCPServersResponse(servers=servers)


@app.post("/v1/mcp/execute", dependencies=[Depends(verify_api_key)])
async def execute_mcp_tool(request: MCPExecuteRequest) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    if _mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    result = await _mcp_manager.execute_tool(
        request.tool_name,
        request.arguments,
    )

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )


# =============================================================================
# Audio Endpoints
# =============================================================================

# Global audio engines (lazy loaded)
_stt_engine = None
_tts_engine = None


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """
    Transcribe audio to text (OpenAI Whisper API compatible).

    Supported models:
    - whisper-large-v3 (multilingual, best quality)
    - whisper-large-v3-turbo (faster)
    - whisper-medium, whisper-small (lighter)
    - parakeet-tdt-0.6b-v2 (English, fastest)
    """
    global _stt_engine

    try:
        from .audio.stt import STTEngine  # Lazy import - optional feature

        # Map model aliases to full names
        model_map = {
            "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
            "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "whisper-medium": "mlx-community/whisper-medium-mlx",
            "whisper-small": "mlx-community/whisper-small-mlx",
            "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
            "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
        }
        model_name = model_map.get(model, model)

        # Load engine if needed
        if _stt_engine is None or _stt_engine.model_name != model_name:
            _stt_engine = STTEngine(model_name)
            _stt_engine.load()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = _stt_engine.transcribe(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            return result.text

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = "kokoro",
    input: str = "",
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
):
    """
    Generate speech from text (OpenAI TTS API compatible).

    Supported models:
    - kokoro (fast, lightweight)
    - chatterbox (multilingual, expressive)
    - vibevoice (realtime)
    - voxcpm (Chinese/English)
    """
    global _tts_engine

    try:
        from .audio.tts import TTSEngine  # Lazy import - optional feature

        # Map model aliases to full names
        model_map = {
            "kokoro": "mlx-community/Kokoro-82M-bf16",
            "kokoro-4bit": "mlx-community/Kokoro-82M-4bit",
            "chatterbox": "mlx-community/chatterbox-turbo-fp16",
            "chatterbox-4bit": "mlx-community/chatterbox-turbo-4bit",
            "vibevoice": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            "voxcpm": "mlx-community/VoxCPM1.5",
        }
        model_name = model_map.get(model, model)

        # Load engine if needed
        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=response_format)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model."""
    from .audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}


# =============================================================================
# Streaming disconnect detection
# =============================================================================


async def _disconnect_guard(
    generator: AsyncIterator[str],
    raw_request: Request,
    poll_interval: float = 0.5,
) -> AsyncIterator[str]:
    """Wrap streaming generator to abort on client disconnect.

    Uses asyncio racing: each __anext__() on the inner generator is
    raced against a disconnect poller.  This catches disconnects even
    during prefill when no chunks are being yielded for tens of seconds.

    On disconnect, aclose() propagates down the generator chain to
    engine_core.stream_outputs() finally-block → abort_request().
    """
    import time as _time

    _t0 = _time.monotonic()

    def _elapsed():
        return f"{_time.monotonic() - _t0:.1f}s"

    logger.info(f"[disconnect_guard] START poll_interval={poll_interval}s")

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_elapsed()}"
                )
            if is_disc:
                return

    chunk_count = 0
    disconnect_task: asyncio.Task | None = None
    anext_task: asyncio.Task | None = None
    try:
        aiter = generator.__aiter__()
        disconnect_task = asyncio.create_task(_wait_disconnect())
        while True:
            anext_task = asyncio.ensure_future(aiter.__anext__())
            done, _ = await asyncio.wait(
                [anext_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                logger.info(
                    f"[disconnect_guard] CLIENT DISCONNECTED after "
                    f"{chunk_count} chunks, elapsed={_elapsed()}"
                )
                anext_task.cancel()
                try:
                    await anext_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                break
            try:
                chunk = anext_task.result()
            except StopAsyncIteration:
                logger.info(
                    f"[disconnect_guard] generator exhausted normally, "
                    f"{chunk_count} chunks, elapsed={_elapsed()}"
                )
                break
            chunk_count += 1
            if chunk_count == 1:
                logger.info(
                    f"[disconnect_guard] first chunk arrived, elapsed={_elapsed()}"
                )
            yield chunk
    except GeneratorExit:
        logger.info(
            f"[disconnect_guard] GeneratorExit after {chunk_count} chunks, elapsed={_elapsed()}"
        )
    finally:
        if disconnect_task and not disconnect_task.done():
            disconnect_task.cancel()
        if anext_task and not anext_task.done():
            anext_task.cancel()
        # NOTE: Do NOT call generator.aclose() here.  With run_in_executor,
        # scheduler.step() runs in a background thread.  aclose() would throw
        # GeneratorExit into the async-generator chain, which can trigger
        # mlx::core::eval on the main thread while the executor thread is also
        # mid-eval → Metal assertion failure → SIGABRT.
        #
        # Instead, rely on the task cancellation propagation:
        #   anext_task.cancel() → CancelledError in stream_outputs()
        #   → finally block → abort_request() → request removed from scheduler
        logger.info(
            f"[disconnect_guard] CLEANUP done, {chunk_count} chunks total, elapsed={_elapsed()}"
        )


async def _wait_with_disconnect(
    coro,
    raw_request: Request,
    timeout: float,
    poll_interval: float = 0.5,
):
    """Run a coroutine with both timeout and client disconnect detection.

    For non-streaming requests where _disconnect_guard() can't be used.
    Races the coroutine against a disconnect poller, same pattern as
    _disconnect_guard but for awaitable (non-generator) coroutines.
    """
    import time as _time

    _t0 = _time.monotonic()

    task = asyncio.ensure_future(coro)

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_time.monotonic() - _t0:.1f}s"
                )
            if is_disc:
                return

    disconnect_task = asyncio.create_task(_wait_disconnect())

    try:
        done, _ = await asyncio.wait(
            [task, disconnect_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout:.1f} seconds",
            )

        if disconnect_task in done:
            # Client disconnected
            logger.info(
                f"[disconnect_guard] CLIENT DISCONNECTED (non-stream) "
                f"elapsed={_time.monotonic() - _t0:.1f}s"
            )
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return None  # Signal to caller that client disconnected

        # Task completed
        return task.result()

    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        if not task.done():
            task.cancel()


# =============================================================================
# Completion Endpoints
# =============================================================================


@app.post(
    "/v1/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Create a text completion."""
    tracker = _metrics.track_inference("completions", stream=request.stream)
    try:
        return await _create_completion_inner(request, raw_request, tracker)
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception:
        tracker.finish(result="error")
        raise


async def _create_completion_inner(request, raw_request, tracker):
    model_ctx = await _acquire_request_model(request.model)
    engine = model_ctx.engine

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # --- Detailed request logging ---
    prompt_preview = prompts[0][:200] if prompts else "(empty)"
    prompt_len = sum(len(p) for p in prompts)
    logger.info(
        f"[REQUEST] POST /v1/completions stream={request.stream} "
        f"max_tokens={request.max_tokens} temp={request.temperature} "
        f"top_p={request.top_p} top_k={request.top_k} min_p={request.min_p} "
        f"presence_penalty={request.presence_penalty} "
        f"repetition_penalty={request.repetition_penalty} "
        f"prompt_chars={prompt_len} prompt_preview={prompt_preview!r}"
    )

    if request.stream:
        return StreamingResponse(
            _stream_with_model_context(
                model_ctx,
                _disconnect_guard(
                    stream_completion(engine, prompts[0], request),
                    raw_request,
                ),
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or _default_timeout
    choices = []
    total_completion_tokens = 0
    total_prompt_tokens = 0

    try:
        for i, prompt in enumerate(prompts):
            generate_kwargs = {
                "prompt": prompt,
                "max_tokens": request.max_tokens or _default_max_tokens,
                "temperature": _resolve_temperature(request.temperature),
                "top_p": _resolve_top_p(request.top_p),
                "top_k": _resolve_top_k(request.top_k),
                "min_p": _resolve_min_p(request.min_p),
                "presence_penalty": _resolve_presence_penalty(request.presence_penalty),
                "repetition_penalty": _resolve_repetition_penalty(
                    request.repetition_penalty
                ),
                "stop": request.stop,
            }
            if request.specprefill is not None:
                generate_kwargs["specprefill"] = request.specprefill
            if request.specprefill_keep_pct is not None:
                generate_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct

            output = await _wait_with_disconnect(
                engine.generate(**generate_kwargs),
                raw_request,
                timeout=timeout,
            )
            if output is None:
                return Response(status_code=499)  # Client closed request

            choices.append(
                CompletionChoice(
                    index=i,
                    text=output.text,
                    finish_reason=output.finish_reason,
                )
            )
            total_completion_tokens += output.completion_tokens
            total_prompt_tokens += (
                output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
            )
    finally:
        await model_ctx.release()

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Completion: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    tracker.finish(
        result="success",
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )
    return CompletionResponse(
        model=model_ctx.model_name,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    tracker = _metrics.track_inference("chat_completions", stream=request.stream)
    try:
        return await _create_chat_completion_inner(request, raw_request, tracker)
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception:
        tracker.finish(result="error")
        raise


async def _create_chat_completion_inner(
    request: ChatCompletionRequest, raw_request: Request, tracker
):
    model_ctx = await _acquire_request_model(request.model)
    engine = model_ctx.engine

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    has_tools = bool(request.tools)
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} top_p={request.top_p} "
        f"top_k={request.top_k} min_p={request.min_p} "
        f"presence_penalty={request.presence_penalty} "
        f"repetition_penalty={request.repetition_penalty} "
        f"msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.info(f"[REQUEST] last user message preview: {last_user_preview!r}")

    # For MLLM models, keep original messages with embedded images
    # (MLLM.chat() extracts images from message content internally)
    if engine.is_mllm:
        # Convert Pydantic messages to dicts, excluding None fields
        # to prevent chat templates from misinterpreting key presence
        # (e.g. image_url: null on text parts triggers Qwen3-VL crash)
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []  # MLLM extracts these from messages
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        # For LLM, extract text, images, and videos separately
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)
    if engine.is_mllm and not has_media:
        # MLLM extracts media from messages directly, so images/videos are
        # always empty. Check message content for video/image types instead.
        for msg in request.messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in ("image_url", "image", "video", "video_url"):
                        has_media = True
                        break
            if has_media:
                break

    # Handle response_format — inject system prompt and optionally build a
    # grammar-guided logits processor for token-level constrained decoding.
    response_format = request.response_format
    json_logits_processor = None
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

        # Build constrained-decoding logits processor when tools are not
        # active (tool markup cannot comply with a JSON grammar).
        if not (request.tools and request.tool_choice != "none"):
            tokenizer_obj = _get_engine_tokenizer(engine)
            if tokenizer_obj is not None:
                try:
                    json_logits_processor = build_json_logits_processor(
                        response_format, tokenizer_obj
                    )
                except Exception as exc:
                    logger.warning("Failed to build JSON logits processor: %s", exc)
                    json_logits_processor = None
                if json_logits_processor is not None:
                    logger.info(
                        "Constrained decoding enabled for response_format.type=%s",
                        (
                            getattr(response_format, "type", None)
                            if not isinstance(response_format, dict)
                            else response_format.get("type")
                        ),
                    )

    # Prepare kwargs
    resolved_max_tokens = _apply_thinking_max_tokens_floor(
        _resolve_request_field(request, "max_tokens", _default_max_tokens),
        request,
        model_ctx.serving_profile,
    )
    chat_kwargs = {
        "max_tokens": resolved_max_tokens,
        "temperature": _resolve_temperature(
            _resolve_request_field(request, "temperature", None)
        ),
        "top_p": _resolve_top_p(_resolve_request_field(request, "top_p", None)),
        "top_k": _resolve_top_k(_resolve_request_field(request, "top_k", None)),
        "min_p": _resolve_min_p(_resolve_request_field(request, "min_p", None)),
        "presence_penalty": _resolve_presence_penalty(
            _resolve_request_field(request, "presence_penalty", None)
        ),
        "repetition_penalty": _resolve_repetition_penalty(
            _resolve_request_field(request, "repetition_penalty", None)
        ),
        "stop": request.stop,
    }
    if engine.is_mllm:
        chat_kwargs["preserve_native_tool_format"] = engine.preserve_native_tool_format

    chat_template_kwargs = _apply_serving_profile_defaults(
        request,
        model_ctx.serving_profile,
    )
    if chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = chat_template_kwargs
    chat_kwargs["raw_output"] = True

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # SpecPrefill: per-request overrides (our _resolve_request_field
    # handles both top-level fields and extra_body fallback, which is
    # a superset of the direct-access pattern PR #218 used. chat_template_kwargs
    # forwarding is handled above via _resolve_chat_template_kwargs.)
    specprefill = _resolve_request_field(request, "specprefill", None)
    if specprefill is not None:
        chat_kwargs["specprefill"] = specprefill
    specprefill_keep_pct = _resolve_request_field(request, "specprefill_keep_pct", None)
    if specprefill_keep_pct is not None:
        chat_kwargs["specprefill_keep_pct"] = specprefill_keep_pct

    # Add tools if provided
    if request.tools and request.tool_choice != "none":
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    # --- Thinking-aware processor decision tree ---
    _budget = _resolve_thinking_token_budget(
        getattr(request, "thinking_token_budget", None)
    )
    _budget_effective = _budget is not None

    # Check if the active reasoning parser supports tag-based thinking.
    _parser_has_tags = (
        _reasoning_parser is not None
        and hasattr(_reasoning_parser, "start_token")
        and hasattr(_reasoning_parser, "end_token")
    )

    _thinking_on = chat_kwargs.get("enable_thinking") is not False
    _ctk = chat_kwargs.get("chat_template_kwargs")
    if _ctk and _ctk.get("enable_thinking") is False:
        _thinking_on = False

    _thinking_proc = None

    if _thinking_on and _budget_effective and _parser_has_tags:
        # Build unified thinking-aware processor.
        from vllm_mlx.constrained import ThinkingAwareLogitsProcessor

        _tokenizer = _get_engine_tokenizer(engine)
        _tap = ThinkingAwareLogitsProcessor(
            start_token_ids=_tokenizer.encode(
                _reasoning_parser.start_token, add_special_tokens=False
            ),
            end_token_ids=_tokenizer.encode(
                _reasoning_parser.end_token, add_special_tokens=False
            ),
            thinking_token_budget=_budget,
            inner=json_logits_processor,
            vocab_size=_tokenizer.vocab_size,
        )
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [_tap]
        _thinking_proc = _tap
    elif _thinking_on and not _budget_effective and json_logits_processor is not None:
        # Fallback: thinking + JSON but no budget -> force thinking off
        # (Session 105 compatibility behavior).
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]
        request.enable_thinking = False
        chat_kwargs["enable_thinking"] = False
        ctk = chat_kwargs.get("chat_template_kwargs")
        if ctk is not None:
            ctk["enable_thinking"] = False
    elif json_logits_processor is not None:
        # No thinking or thinking without budget -> JSON processor directly.
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]
    # else: no processor needed.

    # Pass _thinking_proc through kwargs so stream_chat_completion can report
    # reasoning_tokens in the usage chunk.
    chat_kwargs["_thinking_proc"] = _thinking_proc

    if request.stream:

        async def _tracked_stream():
            try:
                async for chunk in _stream_with_model_context(
                    model_ctx,
                    _disconnect_guard(
                        stream_chat_completion(
                            engine,
                            messages,
                            request,
                            model_ctx=model_ctx,
                            **chat_kwargs,
                        ),
                        raw_request,
                    ),
                ):
                    if not tracker._ttft_observed:
                        tracker.observe_ttft()
                    yield chunk
                tracker.finish(result="success")
            except Exception:
                tracker.finish(result="error")
                raise

        return StreamingResponse(
            _tracked_stream(),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    # Request raw output so reasoning/tool parsers see unstripped text.
    chat_kwargs["raw_output"] = True

    start_time = time.perf_counter()
    timeout = request.timeout or _default_timeout

    try:
        output = await _wait_with_disconnect(
            engine.chat(messages=messages, **chat_kwargs),
            raw_request,
            timeout=timeout,
        )
        if output is None:
            return Response(status_code=499)  # Client closed request
    finally:
        await model_ctx.release()

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    reasoning_parser = _build_reasoning_parser(engine, model_ctx.serving_profile)

    # Parse tool calls from output using configured parser
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        output.text,
        request,
        engine=engine,
        profile=model_ctx.serving_profile,
    )

    # Extract reasoning content FIRST (strips channel tokens before JSON extraction)
    reasoning_text = None
    enable_thinking = _apply_serving_profile_defaults(
        request,
        model_ctx.serving_profile,
    ).get("enable_thinking")
    text_to_parse = cleaned_text or output.text
    if reasoning_parser and not tool_calls and enable_thinking is not False:
        try:
            reasoning_text, cleaned_text = reasoning_parser.extract_reasoning(
                text_to_parse,
                request=request,
            )
        except TypeError:
            reasoning_text, cleaned_text = reasoning_parser.extract_reasoning(
                text_to_parse
            )

        if (
            _should_force_content_response(
                request,
                model_ctx.serving_profile,
                model_ctx.model_source,
            )
            and not cleaned_text
        ):
            if reasoning_text:
                cleaned_text, reasoning_text = reasoning_text, None
            elif text_to_parse and text_to_parse.strip():
                cleaned_text = text_to_parse

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        _, parsed_json, is_valid, error = parse_json_output(json_input, response_format)
        if parsed_json is not None:
            # Return JSON as string
            cleaned_text = json.dumps(parsed_json)
        if not is_valid:
            logger.warning(f"JSON validation failed: {error}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    tracker.finish(
        result="success",
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
    )
    return ChatCompletionResponse(
        model=model_ctx.model_name,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=clean_output_text(cleaned_text) if cleaned_text else None,
                    reasoning_content=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
            reasoning_tokens=(
                _thinking_proc.thinking_tokens if _thinking_proc is not None else None
            ),
        ),
    )


# =============================================================================
# Responses API (/v1/responses) — translation layer
# =============================================================================


def _responses_input_to_messages(request):
    """Translate Responses API input + instructions to chat messages."""
    messages = []

    # instructions → system message
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    inp = request.input
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            item_dict = (
                item if isinstance(item, dict) else item.model_dump(exclude_none=True)
            )
            item_type = item_dict.get("type", "message")

            if item_type == "message":
                role = item_dict.get("role", "user")
                content = item_dict.get("content", "")
                # Content can be string or array of parts
                if isinstance(content, list):
                    # Translate input_text → text content parts
                    parts = []
                    for part in content:
                        p = (
                            part
                            if isinstance(part, dict)
                            else (
                                part.model_dump(exclude_none=True)
                                if hasattr(part, "model_dump")
                                else {"type": "text", "text": str(part)}
                            )
                        )
                        ptype = p.get("type", "")
                        if ptype == "input_text":
                            parts.append({"type": "text", "text": p.get("text", "")})
                        elif ptype == "input_image":
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": p.get("image_url", ""),
                                        "detail": p.get("detail", "auto"),
                                    },
                                }
                            )
                        else:
                            parts.append(p)
                    messages.append({"role": role, "content": parts})
                else:
                    messages.append({"role": role, "content": content})

            elif item_type == "function_call_output":
                # Tool result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item_dict.get("call_id", ""),
                        "content": item_dict.get("output", ""),
                    }
                )

            elif item_type == "function_call":
                # Previous assistant tool call (for context)
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item_dict.get("call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": item_dict.get("name", ""),
                                    "arguments": item_dict.get("arguments", "{}"),
                                },
                            }
                        ],
                    }
                )
    return messages


def _responses_tools_to_chat_STALE(tools):
    """STALE — replaced by _responses_tools_to_chat_tools. Keeping temporarily to find cleanup boundary."""
    if not tools:
        return None
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
            )
        # Skip non-function tools (web_search, etc.) — not supported
    return result or None


# REMOVED: stale create_response that was superseded by the proper
# implementation below. The old version used wrong type names and
# a manual chat translation instead of _run_responses_request.
async def _create_response_STALE(raw_request: Request):
    """STALE — replaced by the proper create_response below."""
    from .api.responses_models import (
        ResponseFunctionCallItem as ResponseFunctionCall,
        ResponseObject,
        ResponseMessageItem as ResponseOutputMessage,
        ResponseTextContentPart as ResponseOutputText,
        ResponseReasoningItem,
        ResponsesRequest,
        ResponsesUsage as ResponseUsage,
    )

    body = await raw_request.json()
    request = ResponsesRequest(**body)
    model_ctx = await _acquire_request_model(request.model)
    engine = model_ctx.engine

    logger.info(
        f"[REQUEST] POST /v1/responses stream={request.stream} "
        f"model={request.model!r} max_output_tokens={request.max_output_tokens}"
    )

    # Translate to chat completions format
    messages = _responses_input_to_messages(request)
    tools = _responses_tools_to_chat(request.tools)

    # Build a ChatCompletionRequest-compatible call
    from .api.models import ChatCompletionRequest, Message

    chat_messages = []
    for m in messages:
        if isinstance(m, dict):
            chat_messages.append(
                Message(**{k: v for k, v in m.items() if k in Message.model_fields})
            )
        else:
            chat_messages.append(m)

    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=chat_messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=False,
        tools=tools,
        tool_choice=request.tool_choice,
    )

    if request.stream:
        # Streaming responses
        chat_request.stream = True

        async def _stream_responses():
            resp_id = f"resp_{uuid.uuid4().hex[:16]}"
            msg_id = f"msg_{uuid.uuid4().hex[:16]}"
            seq = 0

            # response.created
            created_resp = ResponseObject(
                id=resp_id,
                model=model_ctx.model_name,
                status="in_progress",
                temperature=request.temperature,
                top_p=request.top_p,
                max_output_tokens=request.max_output_tokens,
                instructions=request.instructions,
            )
            yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': created_resp.model_dump(), 'sequence_number': seq})}\n\n"
            seq += 1

            # output_item.added (message)
            msg_item = ResponseOutputMessage(
                id=msg_id,
                status="in_progress",
                content=[
                    ResponseOutputText(text=""),
                ],
            )
            yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'item': msg_item.model_dump(), 'output_index': 0, 'sequence_number': seq})}\n\n"
            seq += 1

            # content_part.added
            yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}, 'sequence_number': seq})}\n\n"
            seq += 1

            # Stream text deltas from chat completions engine
            chat_kwargs = {
                "max_tokens": request.max_output_tokens or _default_max_tokens,
                "temperature": _resolve_temperature(request.temperature),
                "top_p": _resolve_top_p(request.top_p),
                "stop": getattr(request, "stop", None),
            }
            if engine.is_mllm:
                chat_kwargs["preserve_native_tool_format"] = (
                    engine.preserve_native_tool_format
                )
            chat_template_kwargs = _apply_serving_profile_defaults(
                chat_request,
                model_ctx.serving_profile,
            )
            if chat_template_kwargs:
                chat_kwargs["chat_template_kwargs"] = chat_template_kwargs
            if tools and request.tool_choice != "none":
                chat_kwargs["tools"] = convert_tools_for_template(tools)

            full_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Convert messages to dicts for engine
            msg_dicts = []
            for m in messages:
                if isinstance(m, dict):
                    msg_dicts.append(m)
                elif hasattr(m, "model_dump"):
                    msg_dicts.append(m.model_dump(exclude_none=True))
                else:
                    msg_dicts.append(dict(m))

            async for chunk in engine.stream_chat(messages=msg_dicts, **chat_kwargs):
                new_text = chunk.new_text if hasattr(chunk, "new_text") else ""
                if new_text:
                    full_text += new_text
                    yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'delta': new_text, 'sequence_number': seq})}\n\n"
                    seq += 1
                if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens:
                    prompt_tokens = chunk.prompt_tokens
                if hasattr(chunk, "completion_tokens") and chunk.completion_tokens:
                    completion_tokens = chunk.completion_tokens

            # output_text.done
            yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'text': full_text, 'sequence_number': seq})}\n\n"
            seq += 1

            # content_part.done
            yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': full_text, 'annotations': []}, 'sequence_number': seq})}\n\n"
            seq += 1

            # output_item.done
            msg_item.status = "completed"
            msg_item.content = [ResponseOutputText(text=full_text)]
            yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'item': msg_item.model_dump(), 'output_index': 0, 'sequence_number': seq})}\n\n"
            seq += 1

            # response.completed
            total_tokens = prompt_tokens + completion_tokens
            completed_resp = ResponseObject(
                id=resp_id,
                model=model_ctx.model_name,
                status="completed",
                output=[msg_item.model_dump()],
                output_text=full_text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_output_tokens=request.max_output_tokens,
                instructions=request.instructions,
                usage=ResponseUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )
            yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': completed_resp.model_dump(), 'sequence_number': seq})}\n\n"

        return StreamingResponse(
            _stream_with_model_context(model_ctx, _stream_responses()),
            media_type="text/event-stream",
        )

    # Non-streaming: call chat completion and translate response
    chat_kwargs = {
        "max_tokens": request.max_output_tokens or _default_max_tokens,
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": getattr(request, "stop", None),
        "raw_output": True,
    }
    if engine.is_mllm:
        chat_kwargs["preserve_native_tool_format"] = engine.preserve_native_tool_format
    chat_template_kwargs = _apply_serving_profile_defaults(
        chat_request,
        model_ctx.serving_profile,
    )
    if chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = chat_template_kwargs
    if tools and request.tool_choice != "none":
        chat_kwargs["tools"] = convert_tools_for_template(tools)

    msg_dicts = []
    for m in messages:
        if isinstance(m, dict):
            msg_dicts.append(m)
        elif hasattr(m, "model_dump"):
            msg_dicts.append(m.model_dump(exclude_none=True))
        else:
            msg_dicts.append(dict(m))

    timeout = _default_timeout
    try:
        output = await _wait_with_disconnect(
            engine.chat(messages=msg_dicts, **chat_kwargs),
            raw_request,
            timeout=timeout,
        )
        if output is None:
            return Response(status_code=499)
    finally:
        await model_ctx.release()

    # Parse tool calls and reasoning
    # Skip tool parsing when no tools requested (avoids NoneType warnings)
    if tools:
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(
            output.text,
            chat_request,
            engine=engine,
            profile=model_ctx.serving_profile,
        )
    else:
        cleaned_text, tool_calls = output.text, None
    reasoning_text = None
    reasoning_parser = _build_reasoning_parser(engine, model_ctx.serving_profile)
    if reasoning_parser and not tool_calls:
        text_to_parse = cleaned_text or output.text
        try:
            reasoning_text, cleaned_text = reasoning_parser.extract_reasoning(
                text_to_parse,
                request=chat_request,
            )
        except TypeError:
            reasoning_text, cleaned_text = reasoning_parser.extract_reasoning(
                text_to_parse
            )

    # Build output items
    output_items = []

    # Reasoning item (if present)
    if reasoning_text:
        output_items.append(
            ResponseReasoningItem(
                summary=[{"type": "summary_text", "text": reasoning_text}],
            ).model_dump()
        )

    # Tool calls
    if tool_calls:
        for tc in tool_calls:
            output_items.append(
                ResponseFunctionCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ).model_dump()
            )
    else:
        # Text message
        final_text = (
            clean_output_text(cleaned_text) if cleaned_text else (output.text or "")
        )
        output_items.append(
            ResponseOutputMessage(
                content=[ResponseOutputText(text=final_text)],
            ).model_dump()
        )

    output_text = ""
    for item in output_items:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    output_text += part.get("text", "")

    # Determine status
    status = "completed"
    incomplete_details = None
    if output.finish_reason == "length":
        status = "incomplete"
        incomplete_details = {"reason": "max_output_tokens"}

    return ResponseObject(
        model=_model_name,
        status=status,
        output=output_items,
        output_text=output_text,
        incomplete_details=incomplete_details,
        instructions=request.instructions,
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_output_tokens,
        tools=request.tools or [],
        tool_choice=request.tool_choice,
        usage=ResponseUsage(
            input_tokens=output.prompt_tokens,
            output_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        ),
    )


@app.post(
    "/v1/responses",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_response(request: ResponsesRequest, raw_request: Request):
    """Create a Responses API response."""
    if request.stream:
        return StreamingResponse(
            _disconnect_guard(_stream_responses_request(request), raw_request),
            media_type="text/event-stream",
        )

    response_object, _persisted_messages = await _run_responses_request(
        request, raw_request
    )
    if response_object is None:
        return Response(status_code=499)

    return response_object


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Find existing system message
    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

    if system_idx is not None:
        # Append to existing system message
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{existing}\n\n{instruction}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{existing}\n\n{instruction}"
    else:
        # Prepend new system message
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


# =============================================================================
# Anthropic Messages API Endpoints
# =============================================================================


def _convert_anthropic_stop_reason(openai_reason: str | None) -> str:
    """Convert OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason or "", "end_turn")


@app.post("/v1/messages")
async def create_anthropic_message(
    request: Request,
):
    """
    Anthropic Messages API endpoint.

    Translates Anthropic-format requests to OpenAI format, runs inference
    through the existing engine, and converts the response back.

    Supports both streaming and non-streaming modes.
    """
    tracker = _metrics.track_inference("anthropic_messages", stream=False)
    try:
        return await _create_anthropic_message_inner(request, tracker)
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception:
        tracker.finish(result="error")
        raise


async def _create_anthropic_message_inner(request: Request, tracker):
    # Parse the raw body to handle Anthropic request format.
    # Some clients (e.g. Claude Code) may send JSON with invalid escape
    # sequences like \s, \d in regex patterns within tool definitions.
    # Python's json.loads is strict per RFC 8259 and rejects these.
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        if "Invalid \\escape" in str(e):
            raw = await request.body()
            # Replace lone backslashes (not valid JSON escapes) with \\
            body = json.loads(re.sub(rb'\\(?!["\\/bfnrtu])', rb"\\\\", raw))
        else:
            raise
    anthropic_request = AnthropicRequest(**body)

    model_ctx = await _acquire_request_model(anthropic_request.model)
    engine = model_ctx.engine

    # --- Detailed request logging ---
    n_msgs = len(anthropic_request.messages)
    total_chars = 0
    last_user_preview = ""
    for m in anthropic_request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    sys_chars = len(anthropic_request.system) if anthropic_request.system else 0
    n_tools = len(anthropic_request.tools) if anthropic_request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/messages (anthropic) stream={anthropic_request.stream} "
        f"model={anthropic_request.model!r} max_tokens={anthropic_request.max_tokens} "
        f"msgs={n_msgs} total_chars={total_chars} system_chars={sys_chars} "
        f"tools={n_tools}"
    )
    logger.info(f"[REQUEST] last user message preview: {last_user_preview!r}")

    # Convert Anthropic request -> OpenAI request
    openai_request = anthropic_to_openai(anthropic_request)

    if anthropic_request.stream:
        return StreamingResponse(
            _stream_with_model_context(
                model_ctx,
                _disconnect_guard(
                    _stream_anthropic_messages(
                        engine, openai_request, anthropic_request, model_ctx
                    ),
                    request,
                ),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: run inference through existing engine
    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )
    messages = _normalize_messages(messages)

    anthropic_max = _apply_thinking_max_tokens_floor(
        openai_request.max_tokens or _default_max_tokens,
        openai_request,
        model_ctx.serving_profile,
    )
    chat_kwargs = {
        "max_tokens": anthropic_max,
        "temperature": _resolve_temperature(openai_request.temperature),
        "top_p": _resolve_top_p(openai_request.top_p),
        "top_k": _resolve_top_k(openai_request.top_k),
        "min_p": _resolve_min_p(openai_request.min_p),
        "presence_penalty": _resolve_presence_penalty(openai_request.presence_penalty),
        "repetition_penalty": _resolve_repetition_penalty(
            openai_request.repetition_penalty
        ),
        "stop": openai_request.stop,
        "raw_output": True,
    }
    if engine.is_mllm:
        chat_kwargs["preserve_native_tool_format"] = engine.preserve_native_tool_format
    chat_template_kwargs = _apply_serving_profile_defaults(
        openai_request,
        model_ctx.serving_profile,
    )
    if chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = chat_template_kwargs

    if openai_request.tools and openai_request.tool_choice != "none":
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    # Request raw output so reasoning/tool parsers see unstripped text.
    chat_kwargs["raw_output"] = True

    # Build constrained decoding for response_format (if present on the
    # Anthropic request and forwarded through the adapter).
    response_format = openai_request.response_format
    json_logits_processor = None
    if response_format:
        if not (openai_request.tools and openai_request.tool_choice != "none"):
            tokenizer_obj = _get_engine_tokenizer(engine)
            if tokenizer_obj is not None:
                try:
                    json_logits_processor = build_json_logits_processor(
                        response_format, tokenizer_obj
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to build JSON logits processor (Anthropic): %s", exc
                    )
                    json_logits_processor = None

    # --- Thinking-aware processor decision tree (Anthropic path) ---
    _budget = _resolve_thinking_token_budget(
        getattr(request, "thinking_token_budget", None)
    )
    _budget_effective = _budget is not None

    _parser_has_tags = (
        _reasoning_parser is not None
        and hasattr(_reasoning_parser, "start_token")
        and hasattr(_reasoning_parser, "end_token")
    )

    _thinking_on = chat_kwargs.get("enable_thinking") is not False
    _ctk = chat_kwargs.get("chat_template_kwargs")
    if _ctk and _ctk.get("enable_thinking") is False:
        _thinking_on = False

    _thinking_proc = None

    if _thinking_on and _budget_effective and _parser_has_tags:
        from vllm_mlx.constrained import ThinkingAwareLogitsProcessor

        _tokenizer = _get_engine_tokenizer(engine)
        _tap = ThinkingAwareLogitsProcessor(
            start_token_ids=_tokenizer.encode(
                _reasoning_parser.start_token, add_special_tokens=False
            ),
            end_token_ids=_tokenizer.encode(
                _reasoning_parser.end_token, add_special_tokens=False
            ),
            thinking_token_budget=_budget,
            inner=json_logits_processor,
            vocab_size=_tokenizer.vocab_size,
        )
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [_tap]
        _thinking_proc = _tap
    elif _thinking_on and not _budget_effective and json_logits_processor is not None:
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]
        chat_kwargs["enable_thinking"] = False
        ctk = chat_kwargs.get("chat_template_kwargs")
        if ctk is not None:
            ctk["enable_thinking"] = False
    elif json_logits_processor is not None:
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]

    start_time = time.perf_counter()
    timeout = _default_timeout

    try:
        output = await _wait_with_disconnect(
            engine.chat(messages=messages, **chat_kwargs),
            request,
            timeout=timeout,
        )
        if output is None:
            return Response(status_code=499)  # Client closed request
    finally:
        await model_ctx.release()

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    reasoning_parser = _build_reasoning_parser(engine, model_ctx.serving_profile)

    # Parse tool calls
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        output.text,
        openai_request,
        engine=engine,
        profile=model_ctx.serving_profile,
    )

    # Extract reasoning if parser is configured
    reasoning_text = None
    if reasoning_parser and not tool_calls:
        text_to_parse = cleaned_text or output.text
        try:
            reasoning_text, cleaned_text = reasoning_parser.extract_reasoning(
                text_to_parse,
                request=openai_request,
            )
        except TypeError:
            reasoning_text, cleaned_text = reasoning_parser.extract_reasoning(
                text_to_parse
            )
    # Clean output text
    final_content = None
    if cleaned_text:
        final_content = clean_output_text(cleaned_text)

    # Build Anthropic content blocks directly (with thinking support)
    content_blocks = []

    if reasoning_text:
        content_blocks.append(
            AnthropicResponseContentBlock(type="thinking", thinking=reasoning_text)
        )

    if final_content:
        content_blocks.append(
            AnthropicResponseContentBlock(type="text", text=final_content)
        )

    if tool_calls:
        for tc in tool_calls:
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                tool_input = {}
            content_blocks.append(
                AnthropicResponseContentBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=tool_input,
                )
            )

    if not content_blocks:
        content_blocks.append(AnthropicResponseContentBlock(type="text", text=""))

    stop_reason = _convert_anthropic_stop_reason(
        "tool_calls" if tool_calls else output.finish_reason
    )

    tracker.finish(
        result="success",
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
    )
    anthropic_response = AnthropicResponse(
        model=model_ctx.model_name,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=output.prompt_tokens,
            output_tokens=output.completion_tokens,
            reasoning_tokens=(
                _thinking_proc.thinking_tokens if _thinking_proc is not None else None
            ),
        ),
    )
    return Response(
        content=anthropic_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


@app.post("/v1/messages/count_tokens")
async def count_anthropic_tokens(request: Request):
    """
    Count tokens for an Anthropic Messages API request.

    Uses the model's tokenizer for accurate counting.
    Claude Code calls this endpoint for token budgeting.
    Note: Don't parse via AnthropicRequest — count_tokens requests
    from Claude Code don't include max_tokens.
    """
    body = await request.json()

    requested_model = body.get("model")
    if not requested_model:
        available = _list_available_model_names()
        requested_model = available[0] if len(available) == 1 else ""
    model_ctx = await _acquire_request_model(requested_model)
    tokenizer = model_ctx.engine.tokenizer

    total_tokens = 0

    try:
        # System message
        system = body.get("system", "")
        if isinstance(system, str) and system:
            total_tokens += len(tokenizer.encode(system))
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total_tokens += len(tokenizer.encode(text))

        # Messages
        for msg in body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                if content:
                    total_tokens += len(tokenizer.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text:
                            total_tokens += len(tokenizer.encode(text))
                        # tool_use input
                        if block.get("input"):
                            total_tokens += len(
                                tokenizer.encode(json.dumps(block["input"]))
                            )
                        # tool_result content
                        sub_content = block.get("content", "")
                        if isinstance(sub_content, str) and sub_content:
                            total_tokens += len(tokenizer.encode(sub_content))
                        elif isinstance(sub_content, list):
                            for item in sub_content:
                                if isinstance(item, dict):
                                    item_text = item.get("text", "")
                                    if item_text:
                                        total_tokens += len(tokenizer.encode(item_text))

        # Tools
        for tool in body.get("tools", []):
            name = tool.get("name", "")
            if name:
                total_tokens += len(tokenizer.encode(name))
            desc = tool.get("description", "")
            if desc:
                total_tokens += len(tokenizer.encode(desc))
            if tool.get("input_schema"):
                total_tokens += len(tokenizer.encode(json.dumps(tool["input_schema"])))
    finally:
        await model_ctx.release()

    return {"input_tokens": total_tokens}


async def _stream_anthropic_messages(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    anthropic_request: AnthropicRequest,
    model_ctx: RequestModelContext,
    _thinking_proc=None,
) -> AsyncIterator[str]:
    """
    Stream Anthropic Messages API SSE events.

    Converts OpenAI streaming chunks to Anthropic event format:
    message_start -> content_block_start -> content_block_delta* ->
    content_block_stop -> message_delta -> message_stop
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.perf_counter()

    # Extract messages for engine
    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    stream_anthropic_max = _apply_thinking_max_tokens_floor(
        openai_request.max_tokens or _default_max_tokens,
        openai_request,
        model_ctx.serving_profile,
    )
    chat_kwargs = {
        "max_tokens": stream_anthropic_max,
        "temperature": _resolve_temperature(openai_request.temperature),
        "top_p": _resolve_top_p(openai_request.top_p),
        "top_k": _resolve_top_k(openai_request.top_k),
        "min_p": _resolve_min_p(openai_request.min_p),
        "presence_penalty": _resolve_presence_penalty(openai_request.presence_penalty),
        "repetition_penalty": _resolve_repetition_penalty(
            openai_request.repetition_penalty
        ),
        "stop": openai_request.stop,
    }
    if engine.is_mllm:
        chat_kwargs["preserve_native_tool_format"] = engine.preserve_native_tool_format
    chat_template_kwargs = _apply_serving_profile_defaults(
        openai_request,
        model_ctx.serving_profile,
    )
    if chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = chat_template_kwargs

    if openai_request.tools and openai_request.tool_choice != "none":
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    # Emit message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": anthropic_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    reasoning_parser = _build_reasoning_parser(engine, model_ctx.serving_profile)
    use_reasoning = reasoning_parser is not None

    if reasoning_parser:
        reasoning_parser.reset_state()

    # Block index tracking: with reasoning parser we use index 0 for
    # thinking and index 1 for text; without parser, index 0 for text.
    thinking_block_started = False
    text_block_started = False
    thinking_index = 0
    text_index = 1 if use_reasoning else 0

    if not use_reasoning:
        # No reasoning parser — start text block immediately
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        text_block_started = True

    # Stream content deltas
    accumulated_text = ""
    completion_tokens = 0

    # Tool call streaming suppression — prevents raw tool markup from leaking
    # as text_delta events. Mirrors the OpenAI streaming path logic.
    tool_parser = None
    tool_accumulated_text = ""
    tool_markup_possible = False
    tool_choice = getattr(openai_request, "tool_choice", None)
    if (
        model_ctx.serving_profile.enable_auto_tool_choice
        and model_ctx.serving_profile.tool_call_parser
        and tool_choice != "none"
    ):
        try:
            tool_parser = _build_tool_parser(engine, model_ctx.serving_profile)
        except Exception:
            tool_parser = None
    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        delta_text = output.new_text

        # Track token counts
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        if not delta_text:
            continue

        filtered = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
        if not filtered:
            continue

        if not use_reasoning:
            accumulated_text += filtered
            content_to_emit = filtered

            if tool_parser and content_to_emit:
                if not tool_markup_possible and "<" not in content_to_emit:
                    tool_accumulated_text += content_to_emit
                else:
                    if not tool_markup_possible:
                        tool_markup_possible = True
                    tool_previous = tool_accumulated_text
                    tool_accumulated_text += content_to_emit
                    tool_result = tool_parser.extract_tool_calls_streaming(
                        tool_previous, tool_accumulated_text, content_to_emit
                    )
                    if tool_result is None or "tool_calls" in tool_result:
                        continue
                    content_to_emit = tool_result.get("content", "")
                    if content_to_emit:
                        content_to_emit = _TOOL_MARKUP_PATTERN.sub("", content_to_emit)
                    if not content_to_emit:
                        continue

            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content_to_emit}})}\n\n"
            continue

        previous_text = accumulated_text
        accumulated_text += filtered
        try:
            delta_msg = reasoning_parser.extract_reasoning_streaming(
                previous_text,
                accumulated_text,
                filtered,
                request=openai_request,
            )
        except TypeError:
            delta_msg = reasoning_parser.extract_reasoning_streaming(
                previous_text, accumulated_text, filtered
            )

        if delta_msg is None:
            continue

        if delta_msg.reasoning:
            if not thinking_block_started:
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': thinking_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                thinking_block_started = True
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': thinking_index, 'delta': {'type': 'thinking_delta', 'thinking': delta_msg.reasoning}})}\n\n"

        if delta_msg.content:
            content_to_emit = delta_msg.content

            if tool_parser and content_to_emit:
                if not tool_markup_possible and "<" not in content_to_emit:
                    tool_accumulated_text += content_to_emit
                else:
                    if not tool_markup_possible:
                        tool_markup_possible = True
                    tool_previous = tool_accumulated_text
                    tool_accumulated_text += content_to_emit
                    tool_result = tool_parser.extract_tool_calls_streaming(
                        tool_previous, tool_accumulated_text, content_to_emit
                    )
                    if tool_result is None or "tool_calls" in tool_result:
                        continue
                    content_to_emit = tool_result.get("content", "")
                    if content_to_emit:
                        content_to_emit = _TOOL_MARKUP_PATTERN.sub("", content_to_emit)
                    if not content_to_emit:
                        continue

            if thinking_block_started and not text_block_started:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_block_started = True
            elif not text_block_started:
                text_index = 0
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_block_started = True
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': content_to_emit}})}\n\n"

    if thinking_block_started and not text_block_started:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
        text_index = thinking_index + 1
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        text_block_started = True

    # Check for tool calls in accumulated text
    _, tool_calls = _parse_tool_calls_with_parser(
        accumulated_text,
        openai_request,
        engine=engine,
        profile=model_ctx.serving_profile,
    )

    if text_block_started:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_index})}\n\n"

    # If there are tool calls, emit tool_use blocks
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tool_index = i + 1
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                tool_input = {}

            # content_block_start for tool_use
            tool_block_start = {
                "type": "content_block_start",
                "index": tool_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": {},
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(tool_block_start)}\n\n"

            # Send input as a single delta
            input_json = json.dumps(tool_input)
            input_delta = {
                "type": "content_block_delta",
                "index": tool_index,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(input_delta)}\n\n"

            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"

    # Determine stop reason
    stop_reason = "tool_use" if tool_calls else "end_turn"

    # Emit message_delta with stop_reason and usage
    _usage_dict = {"output_tokens": completion_tokens}
    if _thinking_proc is not None:
        _usage_dict["reasoning_tokens"] = _thinking_proc.thinking_tokens
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": _usage_dict,
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    # Log throughput
    elapsed = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Emit message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# =============================================================================
# Streaming Helpers
# =============================================================================


async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion response."""
    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=request.max_tokens or _default_max_tokens,
        temperature=_resolve_temperature(request.temperature),
        top_p=_resolve_top_p(request.top_p),
        top_k=_resolve_top_k(request.top_k),
        min_p=_resolve_min_p(request.min_p),
        presence_penalty=_resolve_presence_penalty(request.presence_penalty),
        repetition_penalty=_resolve_repetition_penalty(request.repetition_penalty),
        stop=request.stop,
    ):
        data = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": output.new_text,
                    "finish_reason": output.finish_reason if output.finished else None,
                }
            ],
        }
        if output.finished:
            data["usage"] = get_usage(output).model_dump()
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    model_ctx: RequestModelContext,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    _thinking_proc = kwargs.pop("_thinking_proc", None)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    kwargs["raw_output"] = True

    # Check if we should include usage in the final chunk
    include_usage = request.stream_options and request.stream_options.include_usage

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Track if we need to add <think> prefix for thinking models (when no reasoning parser)
    # The template adds <think> to the prompt, so the model output starts inside the think block
    reasoning_parser = _build_reasoning_parser(engine, model_ctx.serving_profile)
    is_thinking_model = (
        "nemotron" in (engine.model_name or "").lower() and not reasoning_parser
    )
    think_prefix_sent = False

    # Reset reasoning parser state for this stream
    if reasoning_parser:
        reasoning_parser.reset_state()

    # Track accumulated text for reasoning parser
    accumulated_text = ""

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    last_output = None
    enable_thinking = _apply_serving_profile_defaults(
        request,
        model_ctx.serving_profile,
    ).get("enable_thinking")

    # Response-format streaming filter -- strip markdown code fences from
    # content when client asked for JSON. Non-streaming path strips fences
    # via ``parse_json_output``; without this, streaming clients see
    # ``"```json{...}```"`` instead of ``"{...}"`` for models that wrap
    # their structured output in markdown (e.g. Gemma 4).
    fence_stripper: StreamingJsonFenceStripper | None = None
    _rf = getattr(request, "response_format", None)
    _rf_type = None
    if _rf is not None:
        _rf_type = getattr(_rf, "type", None)
        if _rf_type is None and isinstance(_rf, dict):
            _rf_type = _rf.get("type")
    if _rf_type in ("json_object", "json_schema"):
        fence_stripper = StreamingJsonFenceStripper()

    # Tool call streaming state
    tool_parser = None
    tool_accumulated_text = ""
    tool_calls_detected = False
    tool_markup_possible = False  # Fast path: skip parsing until tool syntax appears
    tool_choice = getattr(request, "tool_choice", None)
    if (
        model_ctx.serving_profile.enable_auto_tool_choice
        and model_ctx.serving_profile.tool_call_parser
        and tool_choice != "none"
    ):
        try:
            tool_parser = _build_tool_parser(engine, model_ctx.serving_profile)
        except Exception as e:
            logger.warning(f"Failed to init tool parser for streaming: {e}")

    # Stream content
    async for output in engine.stream_chat(messages=messages, **kwargs):
        delta_text = output.new_text
        last_output = output

        # Track token counts from output (updated each chunk)
        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        # Use reasoning parser if enabled (skip when enable_thinking=False)
        if reasoning_parser and delta_text and enable_thinking is not False:
            previous_text = accumulated_text
            accumulated_text += delta_text
            try:
                delta_msg = reasoning_parser.extract_reasoning_streaming(
                    previous_text,
                    accumulated_text,
                    delta_text,
                    request=request,
                )
            except TypeError:
                delta_msg = reasoning_parser.extract_reasoning_streaming(
                    previous_text, accumulated_text, delta_text
                )

            if delta_msg is None:
                # Skip this chunk (e.g., <think> token itself)
                continue

            reasoning = delta_msg.reasoning
            content_delta = delta_msg.content

            # Pipe content through tool parser (reasoning + tool coexistence)
            if tool_parser and content_delta:
                probe_text = tool_accumulated_text[-4:] + content_delta
                if (
                    not tool_markup_possible
                    and "<" not in probe_text
                    and "call:" not in probe_text
                ):
                    tool_accumulated_text += content_delta
                    # No tool markup yet, emit content + reasoning normally
                else:
                    if not tool_markup_possible:
                        tool_markup_possible = True
                    tool_previous = tool_accumulated_text
                    tool_accumulated_text += content_delta
                    request_dict = request.model_dump() if request else None
                    tool_result = tool_parser.extract_tool_calls_streaming(
                        tool_previous,
                        tool_accumulated_text,
                        content_delta,
                        request=request_dict,
                    )

                    if tool_result is None:
                        # Inside tool markup - suppress content, emit reasoning only
                        if reasoning:
                            chunk = ChatCompletionChunk(
                                id=response_id,
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        delta=ChatCompletionChunkDelta(
                                            reasoning_content=reasoning,
                                        ),
                                        finish_reason=(
                                            output.finish_reason
                                            if output.finished
                                            else None
                                        ),
                                    )
                                ],
                                usage=get_usage(output) if output.finished else None,
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        continue

                    if "tool_calls" in tool_result:
                        tool_calls_detected = True
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        tool_calls=tool_result["tool_calls"],
                                        reasoning_content=reasoning,
                                    ),
                                    finish_reason=(
                                        "tool_calls" if output.finished else None
                                    ),
                                )
                            ],
                            usage=get_usage(output) if output.finished else None,
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        continue

                    # Tool parser returned content (not tool calls)
                    content_delta = tool_result.get("content", "")

            # Strip markdown code fences when response_format is set.
            if fence_stripper is not None and not tool_calls_detected:
                content_delta = (
                    fence_stripper.feed(content_delta) if content_delta else ""
                )
                if output.finished:
                    flush = fence_stripper.finalize()
                    if flush:
                        content_delta = content_delta + flush

            chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            content=content_delta if content_delta else None,
                            reasoning_content=reasoning,
                        ),
                        finish_reason=(
                            "tool_calls"
                            if (output.finished and tool_calls_detected)
                            else (output.finish_reason if output.finished else None)
                        ),
                    )
                ],
                usage=get_usage(output) if output.finished else None,
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        else:
            # Standard path without reasoning parsing
            content = delta_text

            # Filter special tokens that may leak into streaming output
            if content:
                content = SPECIAL_TOKENS_PATTERN.sub("", content)

            # Add <think> prefix on first content chunk for thinking models
            if is_thinking_model and not think_prefix_sent and content:
                content = "<think>" + content
                think_prefix_sent = True

            # Tool call streaming parsing
            if tool_parser and delta_text:
                # Fast path: skip full parsing until '<' is seen in the stream,
                # which could start tool markup (e.g. <tool_call>). This avoids
                # per-token string scanning on the growing accumulated text.
                probe_text = tool_accumulated_text[-4:] + delta_text
                if (
                    not tool_markup_possible
                    and "<" not in probe_text
                    and "call:" not in probe_text
                ):
                    tool_accumulated_text += delta_text
                    # No tool markup yet, fall through to normal chunk emission
                else:
                    if not tool_markup_possible:
                        tool_markup_possible = True
                    tool_previous = tool_accumulated_text
                    tool_accumulated_text += delta_text
                    request_dict = request.model_dump() if request else None
                    tool_result = tool_parser.extract_tool_calls_streaming(
                        tool_previous,
                        tool_accumulated_text,
                        delta_text,
                        request=request_dict,
                    )

                    if tool_result is None:
                        # Inside tool markup - suppress output
                        continue

                    if "tool_calls" in tool_result:
                        # Emit structured tool calls
                        tool_calls_detected = True
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        tool_calls=tool_result["tool_calls"]
                                    ),
                                    finish_reason=(
                                        "tool_calls" if output.finished else None
                                    ),
                                )
                            ],
                            usage=get_usage(output) if output.finished else None,
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        continue

                    # Normal content from tool parser
                    content = tool_result.get("content", "")

            # Strip markdown code fences when response_format is set.
            if fence_stripper is not None and not tool_calls_detected:
                content = fence_stripper.feed(content) if content else ""
                if output.finished:
                    flush = fence_stripper.finalize()
                    if flush:
                        content = content + flush

            chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            content=content if content else None
                        ),
                        finish_reason=(
                            "tool_calls"
                            if (output.finished and tool_calls_detected)
                            else (output.finish_reason if output.finished else None)
                        ),
                    )
                ],
                usage=get_usage(output) if output.finished else None,
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Fallback: if tool parser accumulated text but never emitted tool_calls
    # (e.g., </tool_call> never arrived - incomplete tool call)
    if (
        tool_parser
        and tool_accumulated_text
        and not tool_calls_detected
        and ("<tool_call>" in tool_accumulated_text or "call:" in tool_accumulated_text)
    ):
        result = tool_parser.extract_tool_calls(tool_accumulated_text)
        if result.tools_called:
            tool_chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            tool_calls=[
                                {
                                    "index": i,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"],
                                    },
                                }
                                for i, tc in enumerate(result.tool_calls)
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ],
            )
            yield f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Log throughput
    elapsed = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Send final chunk with usage if requested
    if include_usage:
        usage_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[],  # Empty choices for usage-only chunk
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                reasoning_tokens=(
                    _thinking_proc.thinking_tokens
                    if _thinking_proc is not None
                    else None
                ),
            ),
        )
        yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"


# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    global _mcp_manager, _mcp_executor

    try:
        from vllm_mlx.mcp import MCPClientManager, ToolExecutor, load_mcp_config

        config = load_mcp_config(config_path)
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

        _mcp_executor = ToolExecutor(_mcp_manager)

        logger.info(f"MCP initialized with {len(_mcp_manager.get_all_tools())} tools")

    except ImportError:
        logger.error("MCP SDK not installed. Install with: pip install mcp")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP: {e}")
        raise


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="vllm-mlx OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with continuous batching (for multiple users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching for multiple concurrent users",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/all-MiniLM-L6-v2-4bit)",
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Default temperature for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Default top_p for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="Default top_k for generation when not specified in request (0 = disabled)",
    )
    parser.add_argument(
        "--default-min-p",
        type=float,
        default=None,
        help="Default min_p for generation when not specified in request",
    )
    parser.add_argument(
        "--default-presence-penalty",
        type=float,
        default=None,
        help="Default presence_penalty for generation when not specified in request",
    )
    parser.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="Default repetition_penalty for generation when not specified in request (1.0 = disabled)",
    )

    args = parser.parse_args()

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter
    global _default_temperature, _default_top_p
    global _default_top_k, _default_min_p
    global _default_presence_penalty, _default_repetition_penalty
    _api_key = args.api_key
    _default_timeout = args.timeout
    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p
    if args.default_top_k is not None:
        _default_top_k = args.default_top_k
    if args.default_min_p is not None:
        _default_min_p = args.default_min_p
    if args.default_presence_penalty is not None:
        _default_presence_penalty = args.default_presence_penalty
    if args.default_repetition_penalty is not None:
        _default_repetition_penalty = args.default_repetition_penalty

    # Configure rate limiter
    if args.rate_limit > 0:
        _rate_limiter = RateLimiter(requests_per_minute=args.rate_limit, enabled=True)
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if _api_key:
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    _sampling_parts = []
    if _default_temperature is not None:
        _sampling_parts.append(f"temp={_default_temperature}")
    if _default_top_p is not None:
        _sampling_parts.append(f"top_p={_default_top_p}")
    if _default_top_k is not None:
        _sampling_parts.append(f"top_k={_default_top_k}")
    if _default_min_p is not None:
        _sampling_parts.append(f"min_p={_default_min_p}")
    if _default_presence_penalty is not None:
        _sampling_parts.append(f"presence_penalty={_default_presence_penalty}")
    if _default_repetition_penalty is not None:
        _sampling_parts.append(f"repetition_penalty={_default_repetition_penalty}")
    if _sampling_parts:
        logger.info(f"  Sampling defaults: {', '.join(_sampling_parts)}")
    else:
        logger.info("  Sampling defaults: none (using fallbacks)")
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Initialize reasoning parser if specified
    if args.reasoning_parser:
        global _reasoning_parser, _reasoning_parser_name
        from .reasoning import get_parser

        parser_cls = get_parser(args.reasoning_parser)
        _reasoning_parser = parser_cls()
        _reasoning_parser_name = args.reasoning_parser
        logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")
    else:
        _reasoning_parser = None
        _reasoning_parser_name = None

    # Pre-load embedding model if specified
    load_embedding_model(args.embedding_model, lock=True)

    # Load model before starting server
    load_model(
        args.model,
        use_batching=args.continuous_batching,
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
