# SPDX-License-Identifier: Apache-2.0
"""
Guided decoding helpers for structured output.

This module keeps the public ``response_format`` contract stable while using
Outlines-backed MLX logits processors underneath.
"""

from __future__ import annotations

import json
from typing import Any


def normalize_response_format(
    response_format: Any | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize ``ResponseFormat`` models into a plain dict."""
    if response_format is None:
        return None

    if hasattr(response_format, "type") and hasattr(response_format, "json_schema"):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
        return rf_dict

    return dict(response_format)


def response_format_to_schema(
    response_format: Any | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Convert a response-format request into the JSON schema used for guidance.

    ``json_object`` is treated as an unconstrained object schema, while
    ``json_schema`` uses the user-provided schema verbatim.
    """
    rf_dict = normalize_response_format(response_format)
    if rf_dict is None:
        return None

    format_type = rf_dict.get("type", "text")
    if format_type == "text":
        return None
    if format_type == "json_object":
        return {
            "type": "object",
            "additionalProperties": True,
        }
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema") or {}
        schema = json_schema_spec.get("schema")
        if not isinstance(schema, dict) or not schema:
            raise ValueError(
                "response_format.type='json_schema' requires a non-empty schema"
            )
        return schema

    raise ValueError(f"Unsupported response_format type: {format_type}")


def uses_guided_decoding(
    response_format: Any | dict[str, Any] | None,
) -> bool:
    """Return True when a response format should engage guided decoding."""
    return response_format_to_schema(response_format) is not None


class GuidedDecodingFactory:
    """Create fresh per-request logits processors for structured output."""

    def __init__(self, model: Any, tokenizer: Any):
        try:
            from outlines.models import from_mlxlm
        except ImportError as exc:  # pragma: no cover - dependency failure
            raise ImportError(
                "Structured output guided decoding requires the 'outlines' package"
            ) from exc

        # Outlines TransformerTokenizer reads eos_token_id, eos_token, and
        # all_special_tokens from tokenizer._tokenizer (the raw backend).
        # Modern transformers keeps these on the wrapper only. Patch them
        # onto the raw backend so Outlines can find them.
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None:
            for attr in (
                "eos_token_id",
                "eos_token",
                "pad_token_id",
                "pad_token",
                "all_special_tokens",
            ):
                if not hasattr(inner, attr):
                    val = getattr(tokenizer, attr, None)
                    if val is not None:
                        setattr(inner, attr, val)

        self._outlines_model = from_mlxlm(model, tokenizer)

    def build_processors(
        self,
        response_format: Any | dict[str, Any] | None,
    ) -> list[Any] | None:
        """Build fresh Outlines logits processors for a request."""
        schema = response_format_to_schema(response_format)
        if schema is None:
            return None

        from outlines.backends import get_json_schema_logits_processor

        processor = get_json_schema_logits_processor(
            None,
            self._outlines_model,
            json.dumps(schema, separators=(",", ":"), ensure_ascii=False),
        )
        # Wrap: BatchGenerator passes (tokens: list, logits: mx.array) but
        # Outlines expects (input_ids: mx.array, logits: mx.array).
        return [_wrap_outlines_processor(processor)]

    def build_serial_processors(
        self,
        response_format: Any | dict[str, Any] | None,
    ) -> list[Any] | None:
        """Build processors for serial stream_generate path.

        The serial path accumulates tokens starting from the last prefill
        chunk (not just generated tokens). The wrapper strips the prompt
        prefix so the FSM only sees generated tokens.
        """
        schema = response_format_to_schema(response_format)
        if schema is None:
            return None

        from outlines.backends import get_json_schema_logits_processor
        from outlines_core import Index, Vocabulary
        import outlines_core as oc

        processor = get_json_schema_logits_processor(
            None,
            self._outlines_model,
            json.dumps(schema, separators=(",", ":"), ensure_ascii=False),
        )

        # Get initial allowed tokens from the FSM index for the manual
        # first-call mask (processor can't handle empty sequence).
        tok = self._outlines_model.mlx_tokenizer
        vocab_dict = {}
        for tid in range(tok.vocab_size):
            ts = tok.convert_ids_to_tokens(tid)
            if ts is not None:
                vocab_dict.setdefault(ts, []).append(tid)
        v = Vocabulary(tok.eos_token_id, vocab_dict)
        idx = Index(
            oc.json_schema.build_regex_from_schema(
                json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
            ),
            v,
        )
        initial_allowed = list(idx.get_allowed_tokens(idx.get_initial_state()))

        return [
            _wrap_outlines_serial(processor, allowed_initial_tokens=initial_allowed)
        ]


def _wrap_outlines_processor(processor):
    """Adapt Outlines processor to BatchGenerator's (list, mx.array) interface.

    BatchGenerator passes the full token history (prompt + generated) but
    Outlines' FSM expects only generated tokens. Track the prompt length
    on the first call and slice subsequent calls to generated-only.
    """
    import mlx.core as mx

    prompt_len = None

    def wrapped(token_ids, logits):
        nonlocal prompt_len
        if isinstance(token_ids, list):
            if prompt_len is None:
                prompt_len = len(token_ids)
            generated = token_ids[prompt_len:]
            if not generated:
                # No generated tokens yet — first call does processor._setup
                # via (1, 0) empty batch. Subsequent empty calls skip.
                if processor.is_first_token:
                    token_ids = mx.zeros((1, 0), dtype=mx.int32)
                else:
                    return logits
            else:
                # Only pass the LAST generated token (processor is stateful)
                token_ids = mx.array([generated[-1]], dtype=mx.int32)[None, :]
        result = processor(token_ids, logits)
        # Debug: check if masking is effective
        if _log.isEnabledFor(logging.DEBUG):
            import numpy as np

            r = np.array(
                result.tolist()[0] if len(result.shape) == 2 else result.tolist()
            )
            finite = np.isfinite(r)
            _log.debug("Guided: %d/%d tokens allowed (non-inf)", finite.sum(), len(r))
        return result

    return wrapped


def _wrap_outlines_serial(processor, allowed_initial_tokens: list[int]):
    """Wrapper for serial stream_generate path.

    mlx-lm's serial generate accumulates tokens as mx.array starting from
    the last prefill chunk. The FSM must only see generated tokens, not the
    prompt suffix. Track the initial token count and slice.

    On the first call (no generated tokens), manually apply the initial-state
    mask instead of calling the processor (which requires at least one token).
    """
    import mlx.core as mx

    prompt_len = None

    def wrapped(token_ids, logits):
        nonlocal prompt_len
        if len(token_ids.shape) == 1:
            token_ids = token_ids[None, :]

        cur_len = token_ids.shape[1]

        if prompt_len is None:
            prompt_len = cur_len
            empty = mx.zeros((1, 0), dtype=mx.int32)
            return processor(empty, logits)

        n_generated = cur_len - prompt_len
        if n_generated <= 0:
            return logits

        generated = token_ids[:, prompt_len:]
        return processor(generated, logits)

    return wrapped
