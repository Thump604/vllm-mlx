"""Runtime-side SessionSubstrate for the first Thump replay slice."""

from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import RotatingKVCache

from .adapter import (
    BlockGeometry,
    RopeConfig,
    RuntimeHandle,
    SessionBankEntry,
    SessionManifest,
    SessionMetadata,
    THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR,
    THUMP_RT_BANK_MODE_EXACT_FP16_SIDECAR,
    THUMP_RT_BANK_MODE_FP8,
    validate_session_manifest,
    write_session_manifest,
)
from .capture import LayerCapture


def _interleave_split_rotary_pairs(keys: np.ndarray, rotated_dims: int) -> np.ndarray:
    """Map MLX split-pair rotary layout to adjacent-pair layout."""
    if rotated_dims <= 0 or rotated_dims > keys.shape[-1]:
        return keys
    half = rotated_dims // 2
    if half == 0:
        return keys
    out = np.array(keys, copy=True)
    rotated = out[..., :rotated_dims]
    interleaved = np.empty_like(rotated)
    interleaved[..., 0::2] = rotated[..., :half]
    interleaved[..., 1::2] = rotated[..., half:rotated_dims]
    out[..., :rotated_dims] = interleaved
    return out


def _deinterleave_split_rotary_pairs(keys: np.ndarray, rotated_dims: int) -> np.ndarray:
    """Restore MLX split-pair rotary layout from adjacent-pair layout."""
    if rotated_dims <= 0 or rotated_dims > keys.shape[-1]:
        return keys
    half = rotated_dims // 2
    if half == 0:
        return keys
    out = np.array(keys, copy=True)
    rotated = out[..., :rotated_dims]
    split = np.empty_like(rotated)
    split[..., :half] = rotated[..., 0::2]
    split[..., half:rotated_dims] = rotated[..., 1::2]
    out[..., :rotated_dims] = split
    return out


def _rotating_cache_next_index(offset: int, max_size: int, keep: int) -> int:
    """Reconstruct the internal write pointer for a full RotatingKVCache."""
    if offset <= max_size:
        return offset
    ring = max_size - keep
    if ring <= 0:
        return max_size
    overflow = offset - max_size
    rem = overflow % ring
    return max_size if rem == 0 else keep + rem


def _restore_rotating_cache_layout(
    temporal: np.ndarray,
    *,
    offset: int,
    max_size: int,
    keep: int,
) -> tuple[np.ndarray, int]:
    """Invert RotatingKVCache._temporal_order for an already-full cache."""
    idx = _rotating_cache_next_index(offset, max_size, keep)
    if offset <= max_size or temporal.shape[2] < max_size:
        return temporal, idx

    out = np.array(temporal, copy=True)
    if keep > 0:
        tail = out[..., keep:, :]
    else:
        tail = out
    ring = tail.shape[2]
    if ring == 0:
        return out, idx

    rotate = idx - keep
    if rotate <= 0 or rotate >= ring:
        return out, idx

    restored_tail = np.concatenate(
        [tail[..., ring - rotate :, :], tail[..., : ring - rotate, :]],
        axis=2,
    )
    if keep > 0:
        out[..., keep:, :] = restored_tail
    else:
        out = restored_tail
    return out, idx


def _cache_temporal_tokens(cache: Any) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Extract token-major K/V tensors from a live MLX cache."""
    keys = getattr(cache, "keys", None)
    values = getattr(cache, "values", None)
    if keys is not None and values is not None:
        if hasattr(cache, "_temporal_order"):
            keys = cache._temporal_order(keys)
            values = cache._temporal_order(values)
    else:
        state = getattr(cache, "state", None)
        if state is None or len(state) != 2:
            raise ValueError(f"cache {type(cache).__name__} has no material state")
        keys, values = state

    if keys.dtype == mx.bfloat16 and values.dtype == mx.bfloat16:
        keys_bits = keys.view(mx.uint16)
        values_bits = values.view(mx.uint16)
        mx.eval(keys_bits, values_bits)
        keys_np = np.array(keys_bits, copy=True)
        values_np = np.array(values_bits, copy=True)
        exact_dtype = "bf16"
        bank_mode = THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR
    else:
        keys16 = keys.astype(mx.float16)
        values16 = values.astype(mx.float16)
        mx.eval(keys16, values16)
        keys_np = np.array(keys16, copy=True)
        values_np = np.array(values16, copy=True)
        exact_dtype = "fp16"
        bank_mode = THUMP_RT_BANK_MODE_EXACT_FP16_SIDECAR
    if (
        keys_np.ndim != 4
        or values_np.ndim != 4
        or keys_np.shape[0] != 1
        or values_np.shape[0] != 1
    ):
        raise ValueError(
            f"cache {type(cache).__name__} state must have shape [1, H, T, D]"
        )
    token_keys = np.transpose(keys_np[0], (1, 0, 2)).copy()
    token_values = np.transpose(values_np[0], (1, 0, 2)).copy()
    return token_keys, token_values, exact_dtype, bank_mode


def _pack_token_window(
    keys: np.ndarray,
    values: np.ndarray,
    spec: LayerSpec,
    *,
    start_token: int,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Pack token-major K/V tensors into a block-aligned bank slice."""
    if keys.shape != values.shape:
        raise ValueError("keys and values must have matching shapes")
    if keys.ndim != 3:
        raise ValueError("keys and values must have shape [T, H, D]")
    if not spec.rope_traditional and spec.rotary_dims > 0:
        keys = _interleave_split_rotary_pairs(keys, spec.rotary_dims)

    block_size = spec.geometry.block_size_tokens
    start_block = start_token // block_size
    token_offset = start_token - start_block * block_size
    token_count = keys.shape[0]
    block_count = math.ceil((token_offset + token_count) / block_size)
    padded_tokens = block_count * block_size

    packed_k = np.zeros(
        (padded_tokens, spec.geometry.num_kv_heads, spec.geometry.head_dim),
        dtype=keys.dtype,
    )
    packed_v = np.zeros_like(packed_k)
    packed_k[token_offset : token_offset + token_count] = keys
    packed_v[token_offset : token_offset + token_count] = values
    return (
        start_block,
        block_count,
        packed_k.reshape(block_count * spec.geometry.block_elements),
        packed_v.reshape(block_count * spec.geometry.block_elements),
    )


@dataclass(frozen=True)
class LayerSpec:
    layer_index: int
    layer_type: str
    geometry: BlockGeometry
    window_size: int | None = None
    rotary_dims: int = 0
    rope_traditional: bool = True


@dataclass
class _LayerBank:
    spec: LayerSpec
    path: Path
    handle: RuntimeHandle
    bank_mode: int = THUMP_RT_BANK_MODE_FP8


@dataclass(frozen=True)
class SessionCheckpoint:
    manifest_path: Path
    root_dir: Path
    model_id_hash: int
    session_id: int
    sequence_id: int
    prompt_tokens: int
    generated_tokens: int
    artifact_bytes: int

    @property
    def context_tokens(self) -> int:
        return self.prompt_tokens + max(0, self.generated_tokens - 1)


def _stable_u64(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode(), digest_size=8).digest(), "little"
    )


def _bundle_artifact_bytes(root_dir: Path) -> int:
    return sum(path.stat().st_size for path in root_dir.iterdir() if path.is_file())


class SessionSubstrate:
    """One logical editable session mirrored across per-layer Thump banks."""

    def __init__(
        self,
        layer_specs: list[LayerSpec],
        *,
        block_capacity: int,
        root_dir: str | Path | None = None,
        lib_path: str | Path | None = None,
        exact_hot_restart: bool = False,
    ) -> None:
        if not layer_specs:
            raise ValueError("layer_specs must not be empty")
        self.layer_specs = list(layer_specs)
        self.block_size_tokens = layer_specs[0].geometry.block_size_tokens
        self.root_dir = Path(root_dir or tempfile.mkdtemp(prefix="thump-session-"))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.exact_hot_restart = exact_hot_restart
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
        self.last_checkpoint: SessionCheckpoint | None = None

    @staticmethod
    def gemma4_layer_specs(
        model: Any,
        *,
        block_size_tokens: int = 16,
    ) -> list[LayerSpec]:
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
            rope = attn.rope
            rope_dims = int(getattr(rope, "dims", attn.head_dim))
            partial = float(rope_dims) / float(attn.head_dim)
            rope_type = rope_params.get("rope_type", "default")
            if rope_type == "proportional":
                variant = 3
            elif partial < 1.0:
                variant = 2
            else:
                variant = 1
            geometry = BlockGeometry(
                block_size_tokens=block_size_tokens,
                num_kv_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                group_size=attn.n_heads // attn.n_kv_heads,
                rope=RopeConfig(
                    variant=variant,
                    theta=float(
                        getattr(rope, "base", rope_params.get("rope_theta", 10000.0))
                    ),
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
                    rotary_dims=rope_dims,
                    rope_traditional=bool(getattr(rope, "traditional", True)),
                )
            )
        return layer_specs

    @classmethod
    def from_gemma4_model(
        cls,
        model: Any,
        *,
        block_size_tokens: int = 16,
        block_capacity: int,
        root_dir: str | Path | None = None,
        lib_path: str | Path | None = None,
        exact_hot_restart: bool = False,
    ) -> "SessionSubstrate":
        return cls(
            cls.gemma4_layer_specs(model, block_size_tokens=block_size_tokens),
            block_capacity=block_capacity,
            root_dir=root_dir,
            lib_path=lib_path,
            exact_hot_restart=exact_hot_restart,
        )

    @classmethod
    def attach_from_manifest(
        cls,
        layer_specs: list[LayerSpec],
        manifest_path: str | Path,
        *,
        lib_path: str | Path | None = None,
        expected_model_id_hash: int | None = None,
        require_exact_hot_restart: bool = False,
    ) -> tuple["SessionSubstrate", SessionCheckpoint]:
        if not layer_specs:
            raise ValueError("layer_specs must not be empty")
        manifest, bank_entries = validate_session_manifest(
            manifest_path, lib_path=lib_path
        )
        if expected_model_id_hash is not None and (
            manifest.model_id_hash != expected_model_id_hash
        ):
            raise ValueError("checkpoint model_id_hash does not match current model")
        spec_by_layer = {spec.layer_index: spec for spec in layer_specs}
        root_dir = Path(manifest_path).parent
        self = cls.__new__(cls)
        self.layer_specs = list(layer_specs)
        self.block_size_tokens = layer_specs[0].geometry.block_size_tokens
        self.root_dir = root_dir
        expected_layers = set(spec_by_layer)
        exact_layers = {
            entry.layer_index for entry in bank_entries if entry.has_exact_sidecar
        }
        if require_exact_hot_restart and exact_layers != expected_layers:
            missing = sorted(expected_layers - exact_layers)
            raise ValueError(
                f"checkpoint exact hot restart sidecar missing for layers {missing}"
            )
        self.exact_hot_restart = exact_layers == expected_layers
        self._banks = {}
        for bank_entry in bank_entries:
            if bank_entry.layer_index not in spec_by_layer:
                raise ValueError(
                    f"checkpoint layer {bank_entry.layer_index} is not supported by the runtime"
                )
            spec = spec_by_layer[bank_entry.layer_index]
            path = root_dir / bank_entry.bank_relpath
            handle = RuntimeHandle.attach(path, lib_path=lib_path)
            handle.validate_session_snapshot()
            self._banks[spec.layer_index] = _LayerBank(
                spec=spec,
                path=path,
                handle=handle,
                bank_mode=bank_entry.bank_mode or THUMP_RT_BANK_MODE_FP8,
            )
        self.total_tokens = int(manifest.prompt_tokens) + max(
            0, int(manifest.generated_tokens) - 1
        )
        self.last_checkpoint = SessionCheckpoint(
            manifest_path=Path(manifest_path),
            root_dir=root_dir,
            model_id_hash=manifest.model_id_hash,
            session_id=manifest.session_id,
            sequence_id=manifest.sequence_id,
            prompt_tokens=manifest.prompt_tokens,
            generated_tokens=manifest.generated_tokens,
            artifact_bytes=(
                _bundle_artifact_bytes(root_dir)
            ),
        )
        return self, self.last_checkpoint

    @classmethod
    def attach_gemma4_checkpoint(
        cls,
        model: Any,
        manifest_path: str | Path,
        *,
        block_size_tokens: int = 16,
        lib_path: str | Path | None = None,
        expected_model_id_hash: int | None = None,
        require_exact_hot_restart: bool = False,
    ) -> tuple["SessionSubstrate", SessionCheckpoint]:
        return cls.attach_from_manifest(
            cls.gemma4_layer_specs(model, block_size_tokens=block_size_tokens),
            manifest_path,
            lib_path=lib_path,
            expected_model_id_hash=expected_model_id_hash,
            require_exact_hot_restart=require_exact_hot_restart,
        )

    def close(self) -> None:
        for bank in self._banks.values():
            bank.handle.close()

    def _pack_capture(
        self, capture: LayerCapture, spec: LayerSpec
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = capture.keys.shape[0]
        keys = capture.keys
        if not spec.rope_traditional and spec.rotary_dims > 0:
            keys = _interleave_split_rotary_pairs(keys, spec.rotary_dims)
        padded_tokens = (
            math.ceil(tokens / spec.geometry.block_size_tokens)
            * spec.geometry.block_size_tokens
        )
        k = np.zeros(
            (padded_tokens, spec.geometry.num_kv_heads, spec.geometry.head_dim),
            dtype=np.float16,
        )
        v = np.zeros_like(k)
        k[:tokens] = keys.astype(np.float16, copy=False)
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
            bank.handle.write_blocks(
                ids,
                packed_k,
                packed_v,
                exact=self.exact_hot_restart,
                exact_dtype="fp16",
            )
            bank.handle.set_sequence(ids)
            bank.bank_mode = (
                THUMP_RT_BANK_MODE_EXACT_FP16_SIDECAR
                if self.exact_hot_restart
                else THUMP_RT_BANK_MODE_FP8
            )
        self.total_tokens = total_tokens

    def initialize_from_live_cache(
        self,
        caches: list[Any],
        *,
        total_tokens: int,
    ) -> None:
        """Initialize an exact hot-restart session from live cache state."""
        if not self.exact_hot_restart:
            raise ValueError("initialize_from_live_cache requires exact_hot_restart")
        total_blocks = math.ceil(total_tokens / self.block_size_tokens)
        for layer_idx, bank in self._banks.items():
            ids = bank.handle.alloc(total_blocks)
            bank.handle.set_sequence(ids)

            token_keys, token_values, exact_dtype, bank_mode = _cache_temporal_tokens(
                caches[layer_idx]
            )
            start_token = max(0, total_tokens - token_keys.shape[0])
            start_block, block_count, packed_k, packed_v = _pack_token_window(
                token_keys,
                token_values,
                bank.spec,
                start_token=start_token,
            )
            if start_block != 0 or block_count != total_blocks:
                zeros = np.zeros(
                    total_blocks * bank.spec.geometry.block_elements,
                    dtype=np.float16,
                )
                bank.handle.write_blocks(ids, zeros, zeros, exact=False)
            bank.handle.write_blocks(
                ids[start_block : start_block + block_count],
                packed_k,
                packed_v,
                exact=True,
                exact_dtype=exact_dtype,
            )
            bank.bank_mode = bank_mode
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
            bank.handle.write_blocks(
                new_ids,
                packed_k,
                packed_v,
                exact=self.exact_hot_restart,
                exact_dtype="fp16",
            )
        self.total_tokens += insert_token_count

    def replace_equal_length_from_capture(
        self,
        replace_token_index: int,
        captures: dict[int, LayerCapture],
        *,
        replace_token_count: int,
    ) -> None:
        if replace_token_index % self.block_size_tokens != 0:
            raise ValueError("replace_token_index must be block aligned")
        if replace_token_count % self.block_size_tokens != 0:
            raise ValueError("replace_token_count must be block aligned")
        replace_blocks = replace_token_count // self.block_size_tokens
        replace_at = replace_token_index // self.block_size_tokens
        for layer_idx, bank in self._banks.items():
            packed_k, packed_v = self._pack_capture(captures[layer_idx], bank.spec)
            new_ids = bank.handle.splice_replace_equal_length(
                replace_at, replace_blocks
            )
            bank.handle.write_blocks(
                new_ids,
                packed_k,
                packed_v,
                exact=self.exact_hot_restart,
                exact_dtype="fp16",
            )

    def checkpoint(
        self,
        manifest_path: str | Path,
        *,
        model_id_hash: int,
        session_id: int,
        sequence_id: int,
        prompt_tokens: int,
        generated_tokens: int,
        flags: int = 1,
    ) -> SessionCheckpoint:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        bank_entries: list[SessionBankEntry] = []
        for layer_idx in sorted(self._banks):
            bank = self._banks[layer_idx]
            bank.handle.sequence_id = sequence_id
            bank.handle.set_session_metadata(
                SessionMetadata(
                    flags=flags,
                    model_id_hash=model_id_hash,
                    session_id=session_id,
                    layer_index=layer_idx,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                )
            )
            bank.handle.validate_session_snapshot()
            bank_entries.append(
                SessionBankEntry(
                    layer_index=layer_idx,
                    bank_relpath=bank.path.relative_to(self.root_dir).as_posix(),
                )
            )

        write_session_manifest(
            manifest_path,
            SessionManifest(
                flags=flags,
                model_id_hash=model_id_hash,
                session_id=session_id,
                sequence_id=sequence_id,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                bank_count=len(bank_entries),
            ),
            bank_entries,
        )
        checkpoint = SessionCheckpoint(
            manifest_path=manifest_path,
            root_dir=self.root_dir,
            model_id_hash=model_id_hash,
            session_id=session_id,
            sequence_id=sequence_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            artifact_bytes=_bundle_artifact_bytes(self.root_dir),
        )
        self.last_checkpoint = checkpoint
        return checkpoint

    def materialize_prompt_cache(self, model: Any, upto_tokens: int) -> list[Any]:
        caches = model.make_cache()
        for bank in self._banks.values():
            spec = bank.spec
            attn = None
            if hasattr(model, "layers"):
                layer = model.layers[spec.layer_index]
                attn = getattr(layer, "self_attn", None)
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

            exact_dtype = (
                "bf16"
                if bank.bank_mode == THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR
                else "fp16"
            )
            flat_k, flat_v = bank.handle.materialize_range(
                start_block,
                block_count,
                exact=self.exact_hot_restart,
                exact_dtype=exact_dtype,
            )
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
            if not spec.rope_traditional and spec.rotary_dims > 0:
                token_k = _deinterleave_split_rotary_pairs(token_k, spec.rotary_dims)
            token_k_hthd = np.transpose(token_k[None, ...], (0, 2, 1, 3))
            token_v_hthd = np.transpose(token_v[None, ...], (0, 2, 1, 3))
            if self.exact_hot_restart and bank.bank_mode == THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR:
                keys = mx.array(token_k_hthd, dtype=mx.uint16).view(mx.bfloat16)
                values = mx.array(token_v_hthd, dtype=mx.uint16).view(mx.bfloat16)
            else:
                keys = mx.array(token_k_hthd, dtype=mx.float16)
                values = mx.array(token_v_hthd, dtype=mx.float16)
            # Exact hot restart sidecars store live post-RoPE cache state.
            # Capture-backed recovery still stores pre-RoPE keys and must
            # rebuild the live cache layout by reapplying rope here.
            if (
                not self.exact_hot_restart
                and attn is not None
                and hasattr(attn, "rope")
            ):
                keys = attn.rope(keys, offset=start_token)
            cache = caches[bank.spec.layer_index]
            alloc_tokens = materialize_tokens
            cache_step = getattr(cache, "step", None)
            cache_max = getattr(cache, "max_size", None)
            if cache_step:
                alloc_tokens = max(
                    cache_step,
                    math.ceil(upto_tokens / cache_step) * cache_step,
                )
            if cache_max is not None:
                alloc_tokens = min(alloc_tokens, cache_max)

            restore_idx = min(upto_tokens, alloc_tokens)
            if isinstance(cache, RotatingKVCache):
                keep = int(getattr(cache, "keep", 0))
                max_size = int(getattr(cache, "max_size", alloc_tokens))
                if (
                    self.exact_hot_restart
                    and bank.bank_mode == THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR
                ):
                    restored_k_bits, restore_idx = _restore_rotating_cache_layout(
                        np.array(keys.view(mx.uint16), copy=True),
                        offset=upto_tokens,
                        max_size=max_size,
                        keep=keep,
                    )
                    restored_v_bits, _ = _restore_rotating_cache_layout(
                        np.array(values.view(mx.uint16), copy=True),
                        offset=upto_tokens,
                        max_size=max_size,
                        keep=keep,
                    )
                    keys = mx.array(restored_k_bits, dtype=mx.uint16).view(mx.bfloat16)
                    values = mx.array(restored_v_bits, dtype=mx.uint16).view(
                        mx.bfloat16
                    )
                else:
                    restored_k, restore_idx = _restore_rotating_cache_layout(
                        np.asarray(keys),
                        offset=upto_tokens,
                        max_size=max_size,
                        keep=keep,
                    )
                    restored_v, _ = _restore_rotating_cache_layout(
                        np.asarray(values),
                        offset=upto_tokens,
                        max_size=max_size,
                        keep=keep,
                    )
                    keys = mx.array(restored_k, dtype=keys.dtype)
                    values = mx.array(restored_v, dtype=values.dtype)

            if hasattr(cache, "keys") and hasattr(cache, "values"):
                padded_k = mx.zeros(
                    (keys.shape[0], keys.shape[1], alloc_tokens, keys.shape[3]),
                    dtype=keys.dtype,
                )
                padded_v = mx.zeros(
                    (values.shape[0], values.shape[1], alloc_tokens, values.shape[3]),
                    dtype=values.dtype,
                )
                padded_k[..., :materialize_tokens, :] = keys
                padded_v[..., :materialize_tokens, :] = values
                cache.keys = padded_k
                cache.values = padded_v
                if hasattr(cache, "offset"):
                    cache.offset = upto_tokens
                if hasattr(cache, "_idx"):
                    cache._idx = restore_idx
            else:
                cache.state = (keys, values)
            if isinstance(cache, RotatingKVCache):
                cache.meta_state = (
                    str(cache.keep),
                    str(cache.max_size),
                    str(upto_tokens),
                    str(restore_idx),
                )
        return caches


def model_id_hash_for_path(model_path: str | Path) -> int:
    return _stable_u64(str(Path(model_path)))
