# SPDX-License-Identifier: Apache-2.0
"""Post-generation response processing.

Tool call parsing, JSON response format handling, usage extraction.
The constrained JSON fix (suppress tool parsing when response_format
is json_schema/json_object) is bundled here with tool parsing.
"""

import logging
import uuid

from .api.models import (
    ChatCompletionRequest,
    FunctionCall,
    ToolCall,
    Usage,
)
from .api.tool_calling import parse_tool_calls
from .engine import GenerationOutput
from .server_state import ServerState
from .tool_parsers import ToolParserManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


def parse_tool_calls_with_parser(
    state: ServerState,
    output_text: str,
    request: ChatCompletionRequest | None = None,
) -> tuple[str, list | None]:
    """
    Parse tool calls from model output using the configured parser.

    If --enable-auto-tool-choice is set with --tool-call-parser, uses the
    selected parser. Otherwise falls back to the generic parse_tool_calls.

    Args:
        state: ServerState instance
        output_text: The model output text
        request: The original request (for context)

    Returns:
        Tuple of (cleaned_text, tool_calls)
    """
    request_dict = request.model_dump() if request else None

    # If auto tool choice is not enabled, use the generic parser
    if not state.enable_auto_tool_choice or not state.tool_call_parser:
        return parse_tool_calls(output_text, request_dict)

    # Initialize parser if needed
    if state.tool_parser_instance is None:
        try:
            parser_cls = ToolParserManager.get_tool_parser(state.tool_call_parser)
            # Get tokenizer from engine if available
            tokenizer = None
            if state.engine is not None and hasattr(state.engine, "_tokenizer"):
                tokenizer = state.engine._tokenizer
            state.tool_parser_instance = parser_cls(tokenizer)
            logger.info(f"Initialized tool call parser: {state.tool_call_parser}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize tool parser '{state.tool_call_parser}': {e}"
            )
            logger.warning("Falling back to generic parser")
            return parse_tool_calls(output_text, request_dict)

    # Use the configured parser
    try:
        # Reset parser state between requests
        state.tool_parser_instance.reset()
        result = state.tool_parser_instance.extract_tool_calls(output_text, request_dict)
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


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# JSON response format handling
# ---------------------------------------------------------------------------


def inject_json_instruction(messages: list, instruction: str) -> list:
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
