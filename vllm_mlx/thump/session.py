"""Runtime-side SessionSubstrate for the first Thump replay slice."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import RotatingKVCache

from .adapter import BlockGeometry, RopeConfig, RuntimeHandle
from .capture import LayerCapture


@dataclass(frozen=True)
class LayerSpec:
    layer_index: int
    layer_type: str
    geometry: BlockGeometry
    window_size: int | None = None


@dataclass
class _LayerBank:
    spec: LayerSpec
    path: Path
    handle: RuntimeHandle


class SessionSubstrate:
    """One logical editable session mirrored across per-layer Thump banks."""

    def __init__(
        self,
        layer_specs: list[LayerSpec],
        *,
        block_capacity: int,
        root_dir: str | Path | None = None,
        lib_path: str | Path | None = None,
    ) -> None:
        if not layer_specs:
            raise ValueError("layer_specs must not be empty")
        self.layer_specs = list(layer_specs)
        self.block_size_tokens = layer_specs[0].geometry.block_size_tokens
        self.root_dir = Path(root_dir or tempfile.mkdtemp(prefix="thump-session-"))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._banks: dict[int, _LayerBank] = {}
        for spec in self.layer_specs:
            path = self.root_dir / f"layer-{spec.layer_index:02d}.thump"
            handle = RuntimeHandle.create(
                path,
                block_capacity,
                spec.geometry,
                lib_path=lib_path,
            )
            self._banks[spec.layer_index] = _LayerBank(
                spec=spec, path=path, handle=handle
            )
        self.total_tokens = 0

    @classmethod
    def from_gemma4_model(
        cls,
        model: Any,
        *,
        block_capacity: int,
        root_dir: str | Path | None = None,
        lib_path: str | Path | None = None,
    ) -> "SessionSubstrate":
        args = model.args
        layer_specs: list[LayerSpec] = []
        for layer_idx, layer in enumerate(model.layers):
            attn = layer.self_attn
            layer_key = (
                "sliding_attention"
                if layer.layer_type == "sliding_attention"
                else "full_attention"
            )
            rope_params = dict(args.rope_parameters.get(layer_key, {}))
            partial = float(rope_params.get("partial_rotary_factor", 1.0))
            rope_type = rope_params.get("rope_type", "default")
            if rope_type == "proportional":
                variant = 3
            elif partial < 1.0:
                variant = 2
            else:
                variant = 1
            geometry = BlockGeometry(
                block_size_tokens=16,
                num_kv_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                group_size=attn.n_heads // attn.n_kv_heads,
                rope=RopeConfig(
                    variant=variant,
                    theta=float(rope_params.get("rope_theta", 10000.0)),
                    partial_rotary_factor=partial,
                ),
            )
            window_size = (
                args.sliding_window if layer.layer_type == "sliding_attention" else None
            )
            layer_specs.append(
                LayerSpec(
                    layer_index=layer_idx,
                    layer_type=layer.layer_type,
                    geometry=geometry,
                    window_size=window_size,
                )
            )
        return cls(
            layer_specs,
            block_capacity=block_capacity,
            root_dir=root_dir,
            lib_path=lib_path,
        )

    def close(self) -> None:
        for bank in self._banks.values():
            bank.handle.close()

    def _pack_capture(
        self, capture: LayerCapture, spec: LayerSpec
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = capture.keys.shape[0]
        padded_tokens = (
            math.ceil(tokens / spec.geometry.block_size_tokens)
            * spec.geometry.block_size_tokens
        )
        k = np.zeros(
            (padded_tokens, spec.geometry.num_kv_heads, spec.geometry.head_dim),
            dtype=np.float16,
        )
        v = np.zeros_like(k)
        k[:tokens] = capture.keys.astype(np.float16, copy=False)
        v[:tokens] = capture.values.astype(np.float16, copy=False)
        block_count = padded_tokens // spec.geometry.block_size_tokens
        return (
            k.reshape(block_count * spec.geometry.block_elements),
            v.reshape(block_count * spec.geometry.block_elements),
        )

    def initialize_from_capture(
        self,
        captures: dict[int, LayerCapture],
        *,
        total_tokens: int,
    ) -> None:
        total_blocks = math.ceil(total_tokens / self.block_size_tokens)
        for layer_idx, bank in self._banks.items():
            packed_k, packed_v = self._pack_capture(captures[layer_idx], bank.spec)
            ids = bank.handle.alloc(total_blocks)
            bank.handle.write_blocks(ids, packed_k, packed_v)
            bank.handle.set_sequence(ids)
        self.total_tokens = total_tokens

    def splice_insert_from_capture(
        self,
        insert_token_index: int,
        captures: dict[int, LayerCapture],
        *,
        insert_token_count: int,
    ) -> None:
        if insert_token_index % self.block_size_tokens != 0:
            raise ValueError("insert_token_index must be block aligned for first slice")
        insert_blocks = math.ceil(insert_token_count / self.block_size_tokens)
        insert_at = insert_token_index // self.block_size_tokens
        for layer_idx, bank in self._banks.items():
            packed_k, packed_v = self._pack_capture(captures[layer_idx], bank.spec)
            new_ids = bank.handle.splice_insert(insert_at, insert_blocks)
            bank.handle.write_blocks(new_ids, packed_k, packed_v)
        self.total_tokens += insert_token_count

    def materialize_prompt_cache(self, model: Any, upto_tokens: int) -> list[Any]:
        caches = model.make_cache()
        for bank in self._banks.values():
            spec = bank.spec
            if spec.layer_type == "sliding_attention":
                materialize_tokens = min(upto_tokens, spec.window_size or upto_tokens)
                start_token = max(0, upto_tokens - materialize_tokens)
            else:
                materialize_tokens = upto_tokens
                start_token = 0

            start_block = start_token // self.block_size_tokens
            end_block = math.ceil(upto_tokens / self.block_size_tokens)
            block_count = max(0, end_block - start_block)
            if block_count == 0:
                continue

            flat_k, flat_v = bank.handle.materialize_range(start_block, block_count)
            token_offset = start_token - start_block * self.block_size_tokens
            full_k = flat_k.reshape(
                block_count * self.block_size_tokens,
                spec.geometry.num_kv_heads,
                spec.geometry.head_dim,
            )
            full_v = flat_v.reshape(
                block_count * self.block_size_tokens,
                spec.geometry.num_kv_heads,
                spec.geometry.head_dim,
            )
            token_k = full_k[token_offset : token_offset + materialize_tokens]
            token_v = full_v[token_offset : token_offset + materialize_tokens]
            keys = mx.array(
                np.transpose(token_k[None, ...], (0, 2, 1, 3)),
                dtype=mx.float16,
            )
            values = mx.array(
                np.transpose(token_v[None, ...], (0, 2, 1, 3)),
                dtype=mx.float16,
            )
            cache = caches[bank.spec.layer_index]
            cache.state = (keys, values)
            if isinstance(cache, RotatingKVCache):
                cache.meta_state = (
                    str(cache.keep),
                    str(cache.max_size),
                    str(upto_tokens),
                    str(materialize_tokens),
                )
        return caches
