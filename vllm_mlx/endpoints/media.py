# SPDX-License-Identifier: Apache-2.0
"""Embeddings, audio, and MCP endpoints.

Provides routes for embedding generation (``/v1/embeddings``), audio
transcription and speech (``/v1/audio/*``), and MCP tool management
(``/v1/mcp/*``) via a FastAPI ``APIRouter``.

Audio and embedding engines use lazy imports so that optional
dependencies (mlx-audio, mlx-embeddings) are only loaded when the
corresponding endpoints are actually called.
"""

import logging
import os
import tempfile
import time

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ..api.models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    MCPToolInfo,
    MCPToolsResponse,
)
from ..server_state import (
    ServerState,
    load_embedding_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_state(request: Request) -> ServerState:
    """Extract ServerState from a Request."""
    return request.app.state.server


router = APIRouter()


# =============================================================================
# Embeddings
# =============================================================================


@router.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest, raw_request: Request
) -> EmbeddingResponse:
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
# Audio: Speech-to-Text
# =============================================================================


@router.post("/v1/audio/transcriptions")
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
        from ..audio.stt import STTEngine  # Lazy import - optional feature

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


# =============================================================================
# Audio: Text-to-Speech
# =============================================================================


@router.post("/v1/audio/speech")
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
        from ..audio.tts import TTSEngine  # Lazy import - optional feature

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


@router.get("/v1/audio/voices")
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model."""
    from ..audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================


@router.get("/v1/mcp/tools")
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


@router.get("/v1/mcp/servers")
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


@router.post("/v1/mcp/execute")
async def execute_mcp_tool(
    request: MCPExecuteRequest, raw_request: Request
) -> MCPExecuteResponse:
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
