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
import logging
import os
import secrets

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Import from new modular API
# Re-export for backwards compatibility with tests
from .api.anthropic_adapter import (
    anthropic_to_openai,
    openai_to_anthropic,
)  # noqa: F401
from .api.anthropic_models import AnthropicRequest  # noqa: F401
from .api.models import (
    AssistantMessage,  # noqa: F401
    ChatCompletionChoice,  # noqa: F401
    ChatCompletionChunk,  # noqa: F401
    ChatCompletionChunkChoice,  # noqa: F401
    ChatCompletionChunkDelta,  # noqa: F401
    ChatCompletionRequest,  # noqa: F401
    ChatCompletionResponse,  # noqa: F401
    CompletionChoice,  # noqa: F401
    CompletionRequest,  # noqa: F401
    CompletionResponse,  # noqa: F401
    ContentPart,  # noqa: F401
    EmbeddingData,  # noqa: F401
    EmbeddingRequest,  # noqa: F401
    EmbeddingResponse,  # noqa: F401
    EmbeddingUsage,  # noqa: F401
    FunctionCall,  # noqa: F401
    ImageUrl,  # noqa: F401
    MCPExecuteRequest,  # noqa: F401
    MCPExecuteResponse,  # noqa: F401
    MCPServerInfo,  # noqa: F401
    MCPServersResponse,  # noqa: F401
    MCPToolInfo,  # noqa: F401
    MCPToolsResponse,  # noqa: F401
    Message,  # noqa: F401
    ModelInfo,  # noqa: F401
    ModelsResponse,
    ToolCall,  # noqa: F401
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.tool_calling import (
    convert_tools_for_template,  # noqa: F401
)
from .api.utils import (
    SPECIAL_TOKENS_PATTERN,  # noqa: F401
    clean_output_text,  # noqa: F401
    extract_multimodal_content,  # noqa: F401
    is_mllm_model,  # noqa: F401
)
from .endpoints.anthropic import (  # noqa: F401 — re-export for backwards compat
    _stream_anthropic_messages,
    count_anthropic_tokens,
    create_anthropic_message,
)
from .endpoints.anthropic import router as anthropic_router
from .endpoints.chat import (  # noqa: F401 — re-export for backwards compat
    _disconnect_guard,
    _wait_with_disconnect,
    create_chat_completion,
    create_completion,
    stream_chat_completion,
    stream_completion,
)
from .endpoints.chat import router as chat_router
from .endpoints.media import (  # noqa: F401 — re-export for backwards compat
    create_embeddings,
    create_speech,
    create_transcription,
    execute_mcp_tool,
    list_mcp_servers,
    list_mcp_tools,
    list_voices,
)
from .endpoints.media import router as media_router
from .engine import (
    BaseEngine,
    BatchedEngine,
    GenerationOutput,
    SimpleEngine,
)  # noqa: F401
from .response_processing import (
    inject_json_instruction as _inject_json_instruction,  # noqa: F401 — re-export for test compat
    parse_tool_calls_with_parser,  # noqa: F401 — re-export for backwards compat
)
from .server_state import (
    RateLimiter,
    ServerState,
    get_engine as _get_engine,
    load_embedding_model,
    load_model as load_model_fn,
    load_prefix_cache_from_disk,
    save_prefix_cache_to_disk,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_state(request_or_app) -> ServerState:
    """Extract ServerState from a Request or FastAPI app."""
    if hasattr(request_or_app, "app"):
        return request_or_app.app.state.server
    return request_or_app.state.server


async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    state: ServerState = app.state.server

    # Startup: Start engine if loaded (needed for BatchedEngine in uvicorn's event loop)
    if (
        state.engine is not None
        and hasattr(state.engine, "_loaded")
        and not state.engine._loaded
    ):
        await state.engine.start()

    # Load persisted cache from disk (AFTER engine start — AsyncEngineCore must exist)
    if state.engine is not None and hasattr(state.engine, "load_cache_from_disk"):
        load_prefix_cache_from_disk(state)

    # Initialize MCP if config provided
    mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
    if mcp_config:
        await init_mcp(app, mcp_config)

    yield

    # Shutdown: Save cache to disk BEFORE stopping engine
    if state.engine is not None and hasattr(state.engine, "save_cache_to_disk"):
        save_prefix_cache_to_disk(state)

    # Shutdown: Close MCP connections and stop engine
    if state.mcp_manager is not None:
        await state.mcp_manager.stop()
        logger.info("MCP manager stopped")
    if state.engine is not None:
        await state.engine.stop()
        logger.info("Engine stopped")


app = FastAPI(
    title="vllm-mlx API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.2.1",
    lifespan=lifespan,
)

security = HTTPBearer(auto_error=False)

# Register endpoint routers (dependencies added after verify_api_key is defined)
# -- see below after dependency definitions


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    state = _get_state(request)
    # Use API key as client ID if available, otherwise use IP
    client_id = request.headers.get(
        "Authorization", request.client.host if request.client else "unknown"
    )

    allowed, retry_after = state.rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Verify API key if authentication is enabled."""
    state = _get_state(request)

    if state.api_key is None:
        # Log warning once about running without authentication
        if not state.auth_warning_logged:
            logger.warning(
                "SECURITY WARNING: Server running without API key authentication. "
                "Anyone can access the API. Use --api-key to enable authentication."
            )
            state.auth_warning_logged = True
        return True  # No auth required

    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, state.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_engine(request: Request) -> BaseEngine:
    """Get the loaded engine, raising error if not loaded."""
    state = _get_state(request)
    return _get_engine(state)


# Register endpoint routers with auth + rate-limit dependencies
app.include_router(
    chat_router,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
app.include_router(
    anthropic_router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    media_router,
    dependencies=[Depends(verify_api_key)],
)


@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    state = _get_state(request)
    mcp_info = None
    if state.mcp_manager is not None:
        connected = sum(
            1
            for s in state.mcp_manager.get_server_status()
            if s.state.value == "connected"
        )
        total = len(state.mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(state.mcp_manager.get_all_tools()),
        }

    engine_stats = state.engine.get_stats() if state.engine else {}

    return {
        "status": "healthy",
        "model_loaded": state.engine is not None,
        "model_name": state.model_name,
        "model_type": "mllm" if (state.engine and state.engine.is_mllm) else "llm",
        "engine_type": engine_stats.get("engine_type", "unknown"),
        "mcp": mcp_info,
    }


@app.get("/v1/status")
async def status(request: Request):
    """Real-time status with per-request details for debugging and monitoring."""
    state = _get_state(request)
    if state.engine is None:
        return {"status": "not_loaded", "model": None, "requests": []}

    stats = state.engine.get_stats()

    return {
        "status": "running" if stats.get("running") else "stopped",
        "model": state.model_name,
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
async def list_models(request: Request) -> ModelsResponse:
    """List available models."""
    state = _get_state(request)
    models = []
    if state.model_name:
        models.append(ModelInfo(id=state.model_name))
    return ModelsResponse(data=models)


# =============================================================================
# Embeddings, audio, MCP — moved to endpoints/media.py
# Streaming disconnect detection — moved to endpoints/chat.py
# Anthropic endpoints — moved to endpoints/anthropic.py
# All imported at top for backwards compatibility.
# =============================================================================


# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(app_instance: FastAPI, config_path: str):
    """Initialize MCP manager from config file."""
    state: ServerState = app_instance.state.server

    try:
        from vllm_mlx.mcp import MCPClientManager, ToolExecutor, load_mcp_config

        config = load_mcp_config(config_path)
        state.mcp_manager = MCPClientManager(config)
        await state.mcp_manager.start()

        state.mcp_executor = ToolExecutor(state.mcp_manager)

        logger.info(
            f"MCP initialized with {len(state.mcp_manager.get_all_tools())} tools"
        )

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

    args = parser.parse_args()

    # Create server state
    state = ServerState()
    state.api_key = args.api_key
    state.default_timeout = args.timeout
    if args.default_temperature is not None:
        state.default_temperature = args.default_temperature
    if args.default_top_p is not None:
        state.default_top_p = args.default_top_p

    # Configure rate limiter
    if args.rate_limit > 0:
        state.rate_limiter = RateLimiter(
            requests_per_minute=args.rate_limit, enabled=True
        )
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if state.api_key:
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Initialize reasoning parser if specified
    if args.reasoning_parser:
        from .reasoning import get_parser

        parser_cls = get_parser(args.reasoning_parser)
        state.reasoning_parser = parser_cls()
        logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")

    # Attach state to app before loading models
    app.state.server = state

    # Pre-load embedding model if specified
    load_embedding_model(state, args.embedding_model, lock=True)

    # Load model before starting server
    load_model_fn(
        state,
        args.model,
        use_batching=args.continuous_batching,
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
