"""Gemma 4 pre-RoPE K/V capture hook for Thump replay."""

from __future__ import annotations

import contextvars
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

_CAPTURE_SINK: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "vllm_mlx_thump_capture_sink",
    default=None,
)
_PATCHED = False


def _normalize_offset(offset: Any) -> int:
    if hasattr(offset, "item"):
        return int(offset.item())
    return int(offset)


@dataclass
class LayerCapture:
    keys: np.ndarray
    values: np.ndarray


class CaptureCollector:
    """Collect per-layer token K/V slices from Gemma 4 attention."""

    def __init__(self) -> None:
        self._keys: dict[int, list[np.ndarray]] = defaultdict(list)
        self._values: dict[int, list[np.ndarray]] = defaultdict(list)
        self._offsets: dict[int, list[int]] = defaultdict(list)

    def capture(
        self,
        *,
        layer_idx: int,
        offset: int,
        keys: mx.array,
        values: mx.array,
        attention: Any,
    ) -> None:
        keys16 = keys.astype(mx.float16)
        values16 = values.astype(mx.float16)
        mx.eval(keys16, values16)
        self._keys[layer_idx].append(np.array(keys16, copy=True))
        self._values[layer_idx].append(np.array(values16, copy=True))
        self._offsets[layer_idx].append(offset)

    def joined(self) -> dict[int, LayerCapture]:
        out: dict[int, LayerCapture] = {}
        for layer_idx, keys_parts in self._keys.items():
            offsets = self._offsets[layer_idx]
            if offsets and offsets != sorted(offsets):
                raise RuntimeError(
                    f"Captured layer {layer_idx} out of order: {offsets}"
                )
            out[layer_idx] = LayerCapture(
                keys=np.concatenate(keys_parts, axis=0),
                values=np.concatenate(self._values[layer_idx], axis=0),
            )
        return out


@contextmanager
def capture_into(collector: CaptureCollector):
    token = _CAPTURE_SINK.set(collector)
    try:
        yield collector
    finally:
        _CAPTURE_SINK.reset(token)


def install_gemma4_capture_patch() -> None:
    """Patch mlx_lm Gemma 4 attention once to expose pre-RoPE K/V."""
    global _PATCHED
    if _PATCHED:
        return

    from mlx_lm.models import gemma4_text
    from mlx_lm.models.base import scaled_dot_product_attention

    original_call = gemma4_text.Attention.__call__

    def patched(self, x, mask=None, cache=None, shared_kv=None):
        batch, seq_len, _ = x.shape

        queries = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if self.is_kv_shared_layer and shared_kv is not None:
            keys, values = shared_kv
            offset = cache.offset if cache is not None else 0
        else:
            offset = cache.offset if cache is not None else 0
            keys = self.k_proj(x).reshape(
                batch, seq_len, self.n_kv_heads, self.head_dim
            )
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(
                    batch, seq_len, self.n_kv_heads, self.head_dim
                )
            keys = self.k_norm(keys)
            values = self.v_norm(values)

            sink = _CAPTURE_SINK.get()
            if sink is not None and batch == 1:
                sink.capture(
                    layer_idx=self.layer_idx,
                    offset=_normalize_offset(offset),
                    keys=keys[0],
                    values=values[0],
                    attention=self,
                )

            values = values.transpose(0, 2, 1, 3)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        if self.store_full_length_kv:
            self._last_kv = (keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2] :]

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(output)

    gemma4_text.Attention.__call__ = patched
    gemma4_text.Attention.__thump_original_call__ = original_call
    _PATCHED = True
