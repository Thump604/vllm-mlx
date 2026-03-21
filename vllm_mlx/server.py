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
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
from collections.abc import AsyncIterator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Import from new modular API
# Re-export for backwards compatibility with tests
from .api.anthropic_adapter import anthropic_to_openai, openai_to_anthropic
from .api.anthropic_models import AnthropicRequest
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
    FunctionCall,  # noqa: F401
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
    ToolCall,  # noqa: F401
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    parse_json_output,
)
from .api.utils import (
    SPECIAL_TOKENS_PATTERN,
    clean_output_text,
    extract_multimodal_content,
    is_mllm_model,  # noqa: F401
)
from .engine import BaseEngine, BatchedEngine, GenerationOutput, SimpleEngine  # noqa: F401
from .response_processing import (
    get_usage,
    inject_json_instruction,
    parse_tool_calls_with_parser,
)
from .server_state import (
    RateLimiter,
    ServerState,
    get_engine as _get_engine,
    load_embedding_model,
    load_model as load_model_fn,
    load_prefix_cache_from_disk,
    resolve_temperature,
    resolve_top_p,
    save_prefix_cache_to_disk,
)
from .tool_parsers import ToolParserManager

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
    if state.engine is not None and hasattr(state.engine, "_loaded") and not state.engine._loaded:
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


@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    state = _get_state(request)
    mcp_info = None
    if state.mcp_manager is not None:
        connected = sum(
            1 for s in state.mcp_manager.get_server_status() if s.state.value == "connected"
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
# Embeddings Endpoint
# =============================================================================


@app.post(
    "/v1/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(request: EmbeddingRequest, raw_request: Request) -> EmbeddingResponse:
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
    state = _get_state(raw_request)

    try:
        # Resolve model name
        model_name = request.model

        # If an embedding model was pre-configured at startup, only allow that model
        if (
            state.embedding_model_locked is not None
            and model_name != state.embedding_model_locked
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding model '{model_name}' is not available. "
                    f"This server was started with --embedding-model {state.embedding_model_locked}. "
                    f"Only '{state.embedding_model_locked}' can be used for embeddings. "
                    f"Restart the server with a different --embedding-model to use '{model_name}'."
                ),
            )

        # Lazy-load or swap embedding engine
        load_embedding_model(state, model_name, lock=False, reuse_existing=True)

        # Normalise input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        start_time = time.perf_counter()

        # Count tokens for usage reporting
        prompt_tokens = state.embedding_engine.count_tokens(texts)

        # Generate embeddings (batch)
        embeddings = state.embedding_engine.embed(texts)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens "
            f"in {elapsed:.2f}s"
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
                "mlx-embeddings not installed. "
                "Install with: pip install mlx-embeddings"
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
async def list_mcp_tools(request: Request) -> MCPToolsResponse:
    """List all available MCP tools."""
    state = _get_state(request)
    if state.mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in state.mcp_manager.get_all_tools():
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
async def list_mcp_servers(request: Request) -> MCPServersResponse:
    """Get status of all MCP servers."""
    state = _get_state(request)
    if state.mcp_manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status_info in state.mcp_manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status_info.name,
                state=status_info.state.value,
                transport=status_info.transport.value,
                tools_count=status_info.tools_count,
                error=status_info.error,
            )
        )

    return MCPServersResponse(servers=servers)


@app.post("/v1/mcp/execute", dependencies=[Depends(verify_api_key)])
async def execute_mcp_tool(request: MCPExecuteRequest, raw_request: Request) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    state = _get_state(raw_request)
    if state.mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    result = await state.mcp_manager.execute_tool(
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

@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    request: Request,
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
    state = _get_state(request)

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
        if state.stt_engine is None or state.stt_engine.model_name != model_name:
            state.stt_engine = STTEngine(model_name)
            state.stt_engine.load()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = state.stt_engine.transcribe(tmp_path, language=language)
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
    request: Request,
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
    state = _get_state(request)

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
        if state.tts_engine is None or state.tts_engine.model_name != model_name:
            state.tts_engine = TTSEngine(model_name)
            state.tts_engine.load()

        audio = state.tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = state.tts_engine.to_bytes(audio, format=response_format)

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
    state = _get_state(raw_request)
    engine = _get_engine(state)

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # --- Detailed request logging ---
    prompt_preview = prompts[0][:200] if prompts else "(empty)"
    prompt_len = sum(len(p) for p in prompts)
    logger.info(
        f"[REQUEST] POST /v1/completions stream={request.stream} "
        f"max_tokens={request.max_tokens} temp={request.temperature} "
        f"prompt_chars={prompt_len} prompt_preview={prompt_preview!r}"
    )

    if request.stream:
        return StreamingResponse(
            _disconnect_guard(
                stream_completion(state, engine, prompts[0], request),
                raw_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or state.default_timeout
    choices = []
    total_completion_tokens = 0
    total_prompt_tokens = 0

    for i, prompt in enumerate(prompts):
        output = await _wait_with_disconnect(
            engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens or state.default_max_tokens,
                temperature=resolve_temperature(state, request.temperature),
                top_p=resolve_top_p(state, request.top_p),
                stop=request.stop,
            ),
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

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Completion: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    return CompletionResponse(
        model=request.model,
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
    state = _get_state(raw_request)
    engine = _get_engine(state)

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
        f"temp={request.temperature} msgs={n_msgs} roles={msg_roles} "
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

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            # Inject JSON instruction into messages
            messages = inject_json_instruction(messages, json_instruction)

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": request.max_tokens or state.default_max_tokens,
        "temperature": resolve_temperature(state, request.temperature),
        "top_p": resolve_top_p(state, request.top_p),
    }

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # SpecPrefill: per-request overrides
    if request.specprefill is not None:
        chat_kwargs["specprefill"] = request.specprefill
    if request.specprefill_keep_pct is not None:
        chat_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    if request.stream:
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(state, engine, messages, request, **chat_kwargs),
                raw_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or state.default_timeout

    output = await _wait_with_disconnect(
        engine.chat(messages=messages, **chat_kwargs),
        raw_request,
        timeout=timeout,
    )
    if output is None:
        return Response(status_code=499)  # Client closed request

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls from output using configured parser
    cleaned_text, tool_calls = parse_tool_calls_with_parser(state, output.text, request)

    # Extract reasoning content FIRST (strips channel tokens before JSON extraction)
    reasoning_text = None
    if state.reasoning_parser and not tool_calls:
        text_to_parse = cleaned_text or output.text
        reasoning_text, cleaned_text = state.reasoning_parser.extract_reasoning(
            text_to_parse
        )

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

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=clean_output_text(cleaned_text) if cleaned_text else None,
                    reasoning=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        ),
    )


# =============================================================================
# Anthropic Messages API Endpoints
# =============================================================================


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
    state = _get_state(request)
    engine = _get_engine(state)

    # Parse the raw body to handle Anthropic request format
    body = await request.json()
    anthropic_request = AnthropicRequest(**body)

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
            _disconnect_guard(
                _stream_anthropic_messages(state, engine, openai_request, anthropic_request),
                request,
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

    chat_kwargs = {
        "max_tokens": openai_request.max_tokens or state.default_max_tokens,
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
    }

    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    start_time = time.perf_counter()
    timeout = state.default_timeout

    output = await _wait_with_disconnect(
        engine.chat(messages=messages, **chat_kwargs),
        request,
        timeout=timeout,
    )
    if output is None:
        return Response(status_code=499)  # Client closed request

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls
    cleaned_text, tool_calls = parse_tool_calls_with_parser(
        state, output.text, openai_request
    )

    # Clean output text
    final_content = None
    if cleaned_text:
        final_content = clean_output_text(cleaned_text)

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # Build OpenAI response to convert
    openai_response = ChatCompletionResponse(
        model=openai_request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        ),
    )

    # Convert to Anthropic response
    anthropic_response = openai_to_anthropic(openai_response, anthropic_request.model)
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

    state = _get_state(request)
    engine = _get_engine(state)
    tokenizer = engine.tokenizer

    total_tokens = 0

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

    return {"input_tokens": total_tokens}


async def _stream_anthropic_messages(
    state: ServerState,
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    anthropic_request: AnthropicRequest,
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

    chat_kwargs = {
        "max_tokens": openai_request.max_tokens or state.default_max_tokens,
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
    }

    if openai_request.tools:
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

    # Emit content_block_start for text
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

    # Stream content deltas
    accumulated_text = ""
    completion_tokens = 0

    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        delta_text = output.new_text

        # Track token counts
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        if delta_text:
            # Filter special tokens
            content = SPECIAL_TOKENS_PATTERN.sub("", delta_text)

            if content:
                accumulated_text += content
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

    # Check for tool calls in accumulated text
    _, tool_calls = parse_tool_calls_with_parser(state, accumulated_text, openai_request)

    # Emit content_block_stop for text block
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

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
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": completion_tokens},
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
    state: ServerState,
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion response."""
    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=request.max_tokens or state.default_max_tokens,
        temperature=resolve_temperature(state, request.temperature),
        top_p=resolve_top_p(state, request.top_p),
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
    state: ServerState,
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

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
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Track if we need to add <think> prefix for thinking models (when no reasoning parser)
    # The template adds <think> to the prompt, so the model output starts inside the think block
    is_thinking_model = "nemotron" in request.model.lower() and not state.reasoning_parser
    think_prefix_sent = False

    # Reset reasoning parser state for this stream
    if state.reasoning_parser:
        state.reasoning_parser.reset_state()

    # Track accumulated text for reasoning parser
    accumulated_text = ""

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    last_output = None

    # Tool call streaming state
    tool_parser = None
    tool_accumulated_text = ""
    tool_calls_detected = False
    tool_markup_possible = False  # Fast path: skip parsing until '<' seen
    if state.enable_auto_tool_choice and state.tool_call_parser:
        # Initialize parser if needed (same as parse_tool_calls_with_parser)
        if state.tool_parser_instance is None:
            try:
                parser_cls = ToolParserManager.get_tool_parser(state.tool_call_parser)
                tokenizer = None
                if state.engine is not None and hasattr(state.engine, "_tokenizer"):
                    tokenizer = state.engine._tokenizer
                state.tool_parser_instance = parser_cls(tokenizer)
                logger.info(f"Initialized tool call parser: {state.tool_call_parser}")
            except Exception as e:
                logger.warning(f"Failed to init tool parser for streaming: {e}")
        if state.tool_parser_instance is not None:
            tool_parser = state.tool_parser_instance
            tool_parser.reset()

    # Stream content
    async for output in engine.stream_chat(messages=messages, **kwargs):
        delta_text = output.new_text
        last_output = output

        # Track token counts from output (updated each chunk)
        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        # Use reasoning parser if enabled
        if state.reasoning_parser and delta_text:
            previous_text = accumulated_text
            accumulated_text += delta_text
            delta_msg = state.reasoning_parser.extract_reasoning_streaming(
                previous_text, accumulated_text, delta_text
            )

            if delta_msg is None:
                # Skip this chunk (e.g., <think> token itself)
                continue

            chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            content=delta_msg.content,
                            reasoning=delta_msg.reasoning,
                        ),
                        finish_reason=output.finish_reason if output.finished else None,
                    )
                ],
                usage=get_usage(output) if output.finished else None,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
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
                if not tool_markup_possible and "<" not in delta_text:
                    tool_accumulated_text += delta_text
                    # No tool markup yet, fall through to normal chunk emission
                else:
                    if not tool_markup_possible:
                        tool_markup_possible = True
                    tool_previous = tool_accumulated_text
                    tool_accumulated_text += delta_text
                    tool_result = tool_parser.extract_tool_calls_streaming(
                        tool_previous, tool_accumulated_text, delta_text
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
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        continue

                    # Normal content from tool parser
                    content = tool_result.get("content", "")

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
            yield f"data: {chunk.model_dump_json()}\n\n"

    # Fallback: if tool parser accumulated text but never emitted tool_calls
    # (e.g., </tool_call> never arrived - incomplete tool call)
    if (
        tool_parser
        and tool_accumulated_text
        and not tool_calls_detected
        and "<tool_call>" in tool_accumulated_text
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
            yield f"data: {tool_chunk.model_dump_json()}\n\n"

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
            ),
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


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

        logger.info(f"MCP initialized with {len(state.mcp_manager.get_all_tools())} tools")

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
        state.rate_limiter = RateLimiter(requests_per_minute=args.rate_limit, enabled=True)
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
