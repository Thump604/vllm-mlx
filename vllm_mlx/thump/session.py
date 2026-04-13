"""Runtime-side SessionSubstrate for Thump continuity."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import ArraysCache, RotatingKVCache

from .adapter import (
    BlockGeometry,
    LinearStateSpec as AdapterLinearStateSpec,
    RopeConfig,
    RuntimeHandle,
    SessionBankEntry,
    SessionManifest,
    SessionMetadata,
    THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR,
    THUMP_RT_BANK_MODE_EXACT_FP16_SIDECAR,
    THUMP_RT_BANK_MODE_FP8,
    THUMP_RT_BANK_MODE_LINEAR_SSM_F32,
    THUMP_RT_LAYER_STATE_KIND_KV_ATTENTION,
    THUMP_RT_LAYER_STATE_KIND_LINEAR_SSM,
    THUMP_RT_LAYER_STATE_KIND_ROTATING_KV,
    materialize_linear_state_f32,
    validate_session_manifest,
    write_linear_state_f32,
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

    if keys.dtype != values.dtype:
        raise ValueError(
            f"cache {type(cache).__name__} keys/values must use matching dtypes"
        )

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
    geometry: BlockGeometry | None = None
    layer_state_kind: int = THUMP_RT_LAYER_STATE_KIND_KV_ATTENTION
    linear_state_spec: AdapterLinearStateSpec | None = None
    window_size: int | None = None
    rotary_dims: int = 0
    rope_traditional: bool = True


@dataclass
class _LayerBank:
    spec: LayerSpec
    path: Path
    handle: RuntimeHandle | None = None
    bank_mode: int = THUMP_RT_BANK_MODE_FP8
    linear_state: tuple[AdapterLinearStateSpec, np.ndarray, np.ndarray] | None = None


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


def _block_size_tokens_from_specs(layer_specs: list[LayerSpec]) -> int:
    for spec in layer_specs:
        if spec.geometry is not None:
            return spec.geometry.block_size_tokens
    raise ValueError("layer_specs must include at least one block-backed layer")


_THUMP_GEMMA4_MODEL_TYPES = frozenset({"gemma4", "gemma4_text"})
_THUMP_QWEN35_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def _load_model_config_from_model_path(model_path: str | Path | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_text_config_from_model_path(model_path: str | Path | None) -> dict[str, Any]:
    payload = _load_model_config_from_model_path(model_path)
    text_config = payload.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return payload if isinstance(payload, dict) else {}


def _normalized_model_type(value: Any) -> str:
    return str(value or "").strip().lower()


def _model_type_candidates(
    model: Any,
    *,
    model_path: str | Path | None = None,
) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        normalized = _normalized_model_type(value)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    config = getattr(model, "config", None)
    add(getattr(config, "model_type", None))
    payload = _load_model_config_from_model_path(model_path)
    add(payload.get("model_type"))
    text_config = payload.get("text_config")
    if isinstance(text_config, dict):
        add(text_config.get("model_type"))
    return candidates


def _linear_state_from_cache(
    cache: Any,
) -> tuple[AdapterLinearStateSpec, np.ndarray, np.ndarray]:
    if not isinstance(cache, ArraysCache):
        raise ValueError(f"cache {type(cache).__name__} is not an ArraysCache")
    if (
        getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
    ):
        raise NotImplementedError(
            "ArraysCache exact hot restart only supports unbatched linear state today"
        )
    state = getattr(cache, "state", None)
    if state is None or len(state) != 2:
        raise ValueError("ArraysCache linear state must contain conv and ssm tensors")
    conv_state, ssm_state = state
    if conv_state is None or ssm_state is None:
        raise ValueError("ArraysCache linear state is incomplete")
    if conv_state.dtype != mx.bfloat16:
        raise ValueError(
            f"ArraysCache conv_state must use bfloat16, got {conv_state.dtype}"
        )
    if ssm_state.dtype != mx.float32:
        raise ValueError(
            f"ArraysCache ssm_state must use float32, got {ssm_state.dtype}"
        )
    conv_bits = conv_state.view(mx.uint16)
    mx.eval(conv_bits, ssm_state)
    conv_np = np.array(conv_bits, copy=True)
    ssm_np = np.array(ssm_state, copy=True)
    if conv_np.ndim != 3 or ssm_np.ndim != 4:
        raise ValueError(
            "ArraysCache linear state must have conv shape [B, H, C] and "
            "ssm shape [B, heads, value_dim, key_dim]"
        )
    spec = AdapterLinearStateSpec(
        conv_batch=int(conv_np.shape[0]),
        conv_history=int(conv_np.shape[1]),
        conv_channels=int(conv_np.shape[2]),
        ssm_batch=int(ssm_np.shape[0]),
        ssm_heads=int(ssm_np.shape[1]),
        ssm_value_dim=int(ssm_np.shape[2]),
        ssm_key_dim=int(ssm_np.shape[3]),
    )
    return spec, conv_np.reshape(-1), ssm_np.reshape(-1)


def _restore_arrays_cache_linear_state(
    cache: Any,
    spec: AdapterLinearStateSpec,
    conv_bits: np.ndarray,
    ssm_state: np.ndarray,
) -> None:
    if not isinstance(cache, ArraysCache):
        raise ValueError(f"cache {type(cache).__name__} is not an ArraysCache")
    conv = mx.array(
        conv_bits.reshape(spec.conv_batch, spec.conv_history, spec.conv_channels),
        dtype=mx.uint16,
    ).view(mx.bfloat16)
    ssm = mx.array(
        ssm_state.reshape(
            spec.ssm_batch,
            spec.ssm_heads,
            spec.ssm_value_dim,
            spec.ssm_key_dim,
        ),
        dtype=mx.float32,
    )
    cache.state = [conv, ssm]
    cache.lengths = None
    cache.left_padding = None
    cache.rollback_state = None


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
        self.block_size_tokens = _block_size_tokens_from_specs(layer_specs)
        self.root_dir = Path(root_dir or tempfile.mkdtemp(prefix="thump-session-"))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.exact_hot_restart = exact_hot_restart
        self.lib_path = lib_path
        self._banks: dict[int, _LayerBank] = {}
        for spec in self.layer_specs:
            if spec.linear_state_spec is not None:
                path = self.root_dir / f"layer-{spec.layer_index:02d}.linearf32"
                self._banks[spec.layer_index] = _LayerBank(
                    spec=spec,
                    path=path,
                    handle=None,
                    bank_mode=THUMP_RT_BANK_MODE_LINEAR_SSM_F32,
                )
                continue
            if spec.geometry is None:
                raise ValueError(f"layer {spec.layer_index} is missing block geometry")
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
    def model_family(
        model: Any,
        *,
        model_path: str | Path | None = None,
    ) -> str:
        candidates = _model_type_candidates(model, model_path=model_path)
        for model_type in candidates:
            if model_type in _THUMP_GEMMA4_MODEL_TYPES:
                return "gemma4"
            if model_type in _THUMP_QWEN35_MODEL_TYPES or model_type.startswith(
                "qwen3_5"
            ):
                return "qwen3_5"
        detected = candidates[0] if candidates else "<unknown>"
        raise NotImplementedError(
            "Thump session substrate does not support "
            f"model_type={detected!r}; only gemma4 and qwen3_5 families are "
            "recognized at this boundary"
        )

    @staticmethod
    def gemma4_layer_specs(
        model: Any,
        *,
        block_size_tokens: int = 16,
        model_path: str | Path | None = None,
    ) -> list[LayerSpec]:
        args = getattr(model, "args", None)
        config = getattr(model, "config", None)
        rope_parameters = getattr(args, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = getattr(config, "rope_parameters", {}) or {}
        text_config = _load_text_config_from_model_path(model_path)
        if not rope_parameters:
            rope_parameters = text_config.get("rope_parameters", {}) or {}
        sliding_window = getattr(args, "sliding_window", None)
        if sliding_window is None:
            sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            sliding_window = text_config.get("sliding_window")
        configured_layer_types = getattr(args, "layer_types", None)
        if configured_layer_types is None:
            configured_layer_types = getattr(config, "layer_types", None)
        if configured_layer_types is not None:
            configured_layer_types = list(configured_layer_types)
            if len(configured_layer_types) != len(model.layers):
                configured_layer_types = None
        if configured_layer_types is None:
            file_layer_types = text_config.get("layer_types")
            if file_layer_types is not None:
                file_layer_types = list(file_layer_types)
                if len(file_layer_types) == len(model.layers):
                    configured_layer_types = file_layer_types
        layer_specs: list[LayerSpec] = []
        for layer_idx, layer in enumerate(model.layers):
            attn = layer.self_attn
            configured_layer_type = None
            if configured_layer_types is not None:
                configured_layer_type = str(configured_layer_types[layer_idx])
            runtime_layer_type = str(
                configured_layer_type or getattr(layer, "layer_type", "") or ""
            )
            layer_type = (
                runtime_layer_type
                if runtime_layer_type in {"sliding_attention", "full_attention"}
                else "full_attention"
            )
            layer_key = layer_type
            rope_params = dict(rope_parameters.get(layer_key, {}))
            rope = attn.rope
            rope_partial = rope_params.get("partial_rotary_factor")
            if rope_partial is not None:
                partial = float(rope_partial)
                rope_dims = max(1, int(round(float(attn.head_dim) * partial)))
            else:
                rope_dims = int(getattr(rope, "dims", attn.head_dim))
                partial = float(rope_dims) / float(attn.head_dim)
            rope_type = rope_params.get("rope_type", "default")
            if partial < 1.0:
                variant = 2
            elif rope_type == "proportional":
                variant = 3
            else:
                variant = 1
            theta = rope_params.get("rope_theta")
            if theta is None:
                theta = getattr(rope, "base", None)
            if theta is None:
                theta = 10000.0
            geometry = BlockGeometry(
                block_size_tokens=block_size_tokens,
                num_kv_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                group_size=attn.n_heads // attn.n_kv_heads,
                rope=RopeConfig(
                    variant=variant,
                    theta=float(theta),
                    partial_rotary_factor=partial,
                ),
            )
            window_size = sliding_window if layer_type == "sliding_attention" else None
            layer_specs.append(
                LayerSpec(
                    layer_index=layer_idx,
                    layer_type=layer_type,
                    geometry=geometry,
                    layer_state_kind=(
                        THUMP_RT_LAYER_STATE_KIND_ROTATING_KV
                        if layer_type == "sliding_attention"
                        else THUMP_RT_LAYER_STATE_KIND_KV_ATTENTION
                    ),
                    window_size=window_size,
                    rotary_dims=rope_dims,
                    rope_traditional=bool(getattr(rope, "traditional", True)),
                )
            )
        return layer_specs

    @staticmethod
    def qwen3_5_layer_specs(
        model: Any,
        *,
        block_size_tokens: int = 16,
        model_path: str | Path | None = None,
    ) -> list[LayerSpec]:
        args = getattr(model, "args", None)
        config = getattr(model, "config", None)
        text_config = _load_text_config_from_model_path(model_path)
        configured_layer_types = getattr(args, "layer_types", None)
        if configured_layer_types is None:
            configured_layer_types = getattr(config, "layer_types", None)
        if configured_layer_types is not None:
            configured_layer_types = list(configured_layer_types)
            if len(configured_layer_types) != len(model.layers):
                configured_layer_types = None
        if configured_layer_types is None:
            file_layer_types = text_config.get("layer_types")
            if file_layer_types is not None:
                file_layer_types = list(file_layer_types)
                if len(file_layer_types) == len(model.layers):
                    configured_layer_types = file_layer_types
        rope_theta = getattr(args, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = text_config.get("rope_theta", 100000.0)
        partial = getattr(args, "partial_rotary_factor", None)
        if partial is None:
            partial = getattr(config, "partial_rotary_factor", None)
        if partial is None:
            partial = text_config.get("partial_rotary_factor", 0.25)
        layer_specs: list[LayerSpec] = []
        for layer_idx, layer in enumerate(model.layers):
            configured_layer_type = None
            if configured_layer_types is not None:
                configured_layer_type = str(configured_layer_types[layer_idx])
            inferred_layer_type = (
                "linear_attention"
                if bool(getattr(layer, "is_linear", False))
                else "full_attention"
            )
            layer_type = str(configured_layer_type or inferred_layer_type or "")
            if layer_type == "linear_attention":
                linear_attn = getattr(layer, "linear_attn", None)
                if linear_attn is None:
                    raise ValueError(
                        f"Qwen layer {layer_idx} is marked linear but has no linear_attn"
                    )
                layer_specs.append(
                    LayerSpec(
                        layer_index=layer_idx,
                        layer_type=layer_type,
                        layer_state_kind=THUMP_RT_LAYER_STATE_KIND_LINEAR_SSM,
                        linear_state_spec=AdapterLinearStateSpec(
                            conv_batch=1,
                            conv_history=int(getattr(linear_attn, "conv_kernel_size"))
                            - 1,
                            conv_channels=int(getattr(linear_attn, "conv_dim")),
                            ssm_batch=1,
                            ssm_heads=int(getattr(linear_attn, "num_v_heads")),
                            ssm_value_dim=int(getattr(linear_attn, "head_v_dim")),
                            ssm_key_dim=int(getattr(linear_attn, "head_k_dim")),
                        ),
                    )
                )
                continue

            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise ValueError(
                    f"Qwen layer {layer_idx} is marked full_attention but has no self_attn"
                )
            rope = attn.rope
            rope_dims = int(round(float(getattr(attn, "head_dim")) * float(partial)))
            geometry = BlockGeometry(
                block_size_tokens=block_size_tokens,
                num_kv_heads=int(getattr(attn, "num_key_value_heads")),
                head_dim=int(getattr(attn, "head_dim")),
                group_size=int(getattr(attn, "num_attention_heads"))
                // int(getattr(attn, "num_key_value_heads")),
                rope=RopeConfig(
                    variant=2 if float(partial) < 1.0 else 1,
                    theta=float(getattr(rope, "base", rope_theta)),
                    partial_rotary_factor=float(partial),
                ),
            )
            layer_specs.append(
                LayerSpec(
                    layer_index=layer_idx,
                    layer_type="full_attention",
                    geometry=geometry,
                    layer_state_kind=THUMP_RT_LAYER_STATE_KIND_KV_ATTENTION,
                    rotary_dims=rope_dims,
                    rope_traditional=bool(getattr(rope, "traditional", False)),
                )
            )
        return layer_specs

    @classmethod
    def layer_specs_for_model(
        cls,
        model: Any,
        *,
        block_size_tokens: int = 16,
        model_path: str | Path | None = None,
    ) -> list[LayerSpec]:
        family = cls.model_family(model, model_path=model_path)
        if family == "gemma4":
            return cls.gemma4_layer_specs(
                model,
                block_size_tokens=block_size_tokens,
                model_path=model_path,
            )
        if family == "qwen3_5":
            return cls.qwen3_5_layer_specs(
                model,
                block_size_tokens=block_size_tokens,
                model_path=model_path,
            )
        raise NotImplementedError(
            "Thump session layer-spec extraction is not implemented for "
            f"{family} family"
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        block_size_tokens: int = 16,
        block_capacity: int,
        root_dir: str | Path | None = None,
        lib_path: str | Path | None = None,
        exact_hot_restart: bool = False,
        model_path: str | Path | None = None,
    ) -> "SessionSubstrate":
        family = cls.model_family(model, model_path=model_path)
        if family == "qwen3_5" and not exact_hot_restart:
            raise NotImplementedError(
                "Qwen session substrate is only implemented for exact_hot_restart "
                "today"
            )
        return cls(
            cls.layer_specs_for_model(
                model,
                block_size_tokens=block_size_tokens,
                model_path=model_path,
            ),
            block_capacity=block_capacity,
            root_dir=root_dir,
            lib_path=lib_path,
            exact_hot_restart=exact_hot_restart,
        )

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
        model_path: str | Path | None = None,
    ) -> "SessionSubstrate":
        return cls.from_model(
            model,
            block_size_tokens=block_size_tokens,
            block_capacity=block_capacity,
            root_dir=root_dir,
            lib_path=lib_path,
            exact_hot_restart=exact_hot_restart,
            model_path=model_path,
        )

    @classmethod
    def attach_checkpoint(
        cls,
        model: Any,
        manifest_path: str | Path,
        *,
        block_size_tokens: int = 16,
        lib_path: str | Path | None = None,
        expected_model_id_hash: int | None = None,
        require_exact_hot_restart: bool = False,
        model_path: str | Path | None = None,
    ) -> tuple["SessionSubstrate", SessionCheckpoint]:
        return cls.attach_from_manifest(
            cls.layer_specs_for_model(
                model,
                block_size_tokens=block_size_tokens,
                model_path=model_path,
            ),
            manifest_path,
            lib_path=lib_path,
            expected_model_id_hash=expected_model_id_hash,
            require_exact_hot_restart=require_exact_hot_restart,
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
        self.block_size_tokens = _block_size_tokens_from_specs(layer_specs)
        self.root_dir = root_dir
        self.lib_path = lib_path
        expected_layers = set(spec_by_layer)
        exact_layers = {
            entry.layer_index
            for entry in bank_entries
            if (
                entry.has_exact_sidecar
                or entry.bank_mode == THUMP_RT_BANK_MODE_LINEAR_SSM_F32
                or entry.layer_state_kind == THUMP_RT_LAYER_STATE_KIND_LINEAR_SSM
            )
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
            self._banks[spec.layer_index] = _LayerBank(
                spec=spec,
                path=path,
                handle=None,
                bank_mode=bank_entry.bank_mode or THUMP_RT_BANK_MODE_FP8,
            )
            if spec.linear_state_spec is None:
                handle = RuntimeHandle.attach(path, lib_path=lib_path)
                handle.validate_session_snapshot()
                self._banks[spec.layer_index].handle = handle
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
            artifact_bytes=(_bundle_artifact_bytes(root_dir)),
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
        model_path: str | Path | None = None,
    ) -> tuple["SessionSubstrate", SessionCheckpoint]:
        return cls.attach_checkpoint(
            model,
            manifest_path,
            block_size_tokens=block_size_tokens,
            lib_path=lib_path,
            expected_model_id_hash=expected_model_id_hash,
            require_exact_hot_restart=require_exact_hot_restart,
            model_path=model_path,
        )

    def close(self) -> None:
        for bank in self._banks.values():
            if bank.handle is not None:
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
            if bank.spec.linear_state_spec is not None:
                raise NotImplementedError(
                    "capture-backed mixed-state session init is not implemented"
                )
            packed_k, packed_v = self._pack_capture(captures[layer_idx], bank.spec)
            if bank.handle is None:
                raise ValueError(f"layer {layer_idx} has no runtime handle")
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
            if bank.spec.linear_state_spec is not None:
                spec, conv_state, ssm_state = _linear_state_from_cache(
                    caches[layer_idx]
                )
                if spec != bank.spec.linear_state_spec:
                    raise ValueError(
                        f"layer {layer_idx} linear state spec does not match model contract"
                    )
                bank.linear_state = (spec, conv_state, ssm_state)
                bank.bank_mode = THUMP_RT_BANK_MODE_LINEAR_SSM_F32
                continue
            if bank.handle is None:
                raise ValueError(f"layer {layer_idx} has no runtime handle")
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
            if bank.spec.linear_state_spec is not None:
                raise NotImplementedError(
                    "mixed-state splice_insert is not implemented"
                )
            packed_k, packed_v = self._pack_capture(captures[layer_idx], bank.spec)
            if bank.handle is None:
                raise ValueError(f"layer {layer_idx} has no runtime handle")
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
            if bank.spec.linear_state_spec is not None:
                raise NotImplementedError(
                    "mixed-state replace_equal_length is not implemented"
                )
            packed_k, packed_v = self._pack_capture(captures[layer_idx], bank.spec)
            if bank.handle is None:
                raise ValueError(f"layer {layer_idx} has no runtime handle")
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
            metadata = SessionMetadata(
                flags=flags,
                model_id_hash=model_id_hash,
                session_id=session_id,
                layer_index=layer_idx,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
            )
            if bank.spec.linear_state_spec is not None:
                if bank.linear_state is None:
                    raise ValueError(
                        f"layer {layer_idx} has no captured linear state to checkpoint"
                    )
                spec, conv_state, ssm_state = bank.linear_state
                write_linear_state_f32(
                    bank.path,
                    sequence_id,
                    metadata,
                    spec,
                    conv_state,
                    ssm_state,
                    lib_path=self.lib_path,
                )
            else:
                if bank.handle is None:
                    raise ValueError(f"layer {layer_idx} has no runtime handle")
                bank.handle.sequence_id = sequence_id
                bank.handle.set_session_metadata(metadata)
                bank.handle.validate_session_snapshot()
            manifest_bank_mode = bank.bank_mode
            if bank.spec.linear_state_spec is None and manifest_bank_mode in (
                THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR,
                THUMP_RT_BANK_MODE_EXACT_FP16_SIDECAR,
            ):
                # The native manifest writer probes exact sidecars from the bank path
                # during normalization. Raw entries must arrive as FP8/non-exact.
                manifest_bank_mode = THUMP_RT_BANK_MODE_FP8
            bank_entries.append(
                SessionBankEntry(
                    layer_index=layer_idx,
                    bank_relpath=bank.path.relative_to(self.root_dir).as_posix(),
                    bank_mode=manifest_bank_mode,
                    layer_state_kind=bank.spec.layer_state_kind,
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
            lib_path=self.lib_path,
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
            if spec.linear_state_spec is not None:
                _meta, conv_bits, ssm_state = materialize_linear_state_f32(
                    bank.path,
                    spec.linear_state_spec,
                    lib_path=self.lib_path,
                )
                _restore_arrays_cache_linear_state(
                    caches[bank.spec.layer_index],
                    spec.linear_state_spec,
                    conv_bits,
                    ssm_state,
                )
                continue
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
            if bank.handle is None:
                raise ValueError(f"layer {spec.layer_index} has no runtime handle")

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
            if (
                self.exact_hot_restart
                and bank.bank_mode == THUMP_RT_BANK_MODE_EXACT_BF16_SIDECAR
            ):
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
