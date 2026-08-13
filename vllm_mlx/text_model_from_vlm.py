# SPDX-License-Identifier: Apache-2.0
"""Construct an mlx_lm TextModel from mlx_vlm-loaded model weights.

When mlx_vlm loads a model, it strips native Qwen MTP weights in sanitize().
This module builds a parallel mlx_lm text model that:
1. Shares backbone + lm_head weights with the vlm model (zero-copy)
2. Loads the exact MTP subtree from safetensors on disk
3. Uses mlx-lm's native sanitize/load capability handshake
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

logger = logging.getLogger(__name__)


def _import_gemma_text_model_classes():
    from mlx_lm.models.gemma4_text import Model, ModelArgs

    return Model, ModelArgs


def _import_qwen_model_classes(model_type: str):
    """Import the outer wrapper that owns Qwen's sanitize/load handshake."""

    if "moe" in model_type:
        from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
    else:
        from mlx_lm.models.qwen3_5 import Model, ModelArgs

    return Model, ModelArgs


def _import_text_model_classes(model_type: str):
    """Compatibility helper retained for the separate Gemma extraction path."""

    if _is_gemma_text_model_type(model_type):
        return _import_gemma_text_model_classes()
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    return TextModel, TextModelArgs


def _safetensors_shapes(shard_path: Path, keys: set[str]) -> dict[str, tuple[int, ...]]:
    """Read selected shapes from a safetensors header without loading arrays."""

    with shard_path.open("rb") as shard:
        header_size_bytes = shard.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {shard_path.name}")
        (header_size,) = struct.unpack("<Q", header_size_bytes)
        # Headers are normally a few MiB. Bound corrupt lengths before reading.
        if header_size <= 0 or header_size > 256 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header size: {shard_path.name}")
        header = json.loads(shard.read(header_size))

    shapes: dict[str, tuple[int, ...]] = {}
    for key in keys:
        entry = header.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
            raise ValueError(
                f"Safetensors index entry missing from shard header: {key}"
            )
        shapes[key] = tuple(int(dim) for dim in entry["shape"])
    return shapes


def _qwen_checkpoint_requires_norm_shift(model_path: Path) -> bool:
    """Recover the official Qwen sanitizer convention from canonical headers.

    mlx-vlm has already normalized the live backbone and removed the evidence
    used by mlx-lm's sanitizer.  Inspecting raw checkpoint shapes restores that
    evidence without loading or duplicating any backbone tensors.
    """

    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        raise ValueError("Native Qwen extraction requires an indexed checkpoint")
    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid native Qwen safetensors index")

    conv_entries = {
        key: shard
        for key, shard in weight_map.items()
        if "conv1d.weight" in key and ".mtp." not in key
    }
    if not conv_entries:
        raise ValueError(
            "Native Qwen checkpoint sanitation convention cannot be proven"
        )

    by_shard: dict[str, set[str]] = {}
    for key, shard in conv_entries.items():
        if not isinstance(shard, str):
            raise ValueError(f"Invalid safetensors shard for {key}")
        by_shard.setdefault(shard, set()).add(key)

    conventions: set[bool] = set()
    for shard_name, keys in by_shard.items():
        shard_path = model_path / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Native Qwen shard not found: {shard_name}")
        for shape in _safetensors_shapes(shard_path, keys).values():
            if len(shape) < 3:
                raise ValueError("Invalid native Qwen conv1d checkpoint shape")
            conventions.add(shape[-1] != 1)
    if len(conventions) != 1:
        raise ValueError("Mixed native Qwen conv1d checkpoint conventions")
    return conventions.pop()


def _sanitize_raw_qwen_mtp(
    wrapper: Any,
    raw_mtp_weights: dict[str, mx.array],
    *,
    shift_norm_weights: bool,
) -> dict[str, mx.array]:
    """Normalize raw MTP through the exact official outer-model sanitizer.

    The live VLM backbone has already been sanitized.  A shape-only sentinel
    recreates the raw backbone signal for this isolated MTP normalization pass;
    it is discarded before the final one-shot identity handshake and strict
    load.  This also applies the official dense/MoE layout normalization.
    """

    probe = dict(raw_mtp_weights)
    sentinel_key = "model.language_model.layers.0.linear_attn.conv1d.weight"
    if shift_norm_weights:
        probe[sentinel_key] = mx.zeros((1, 1, 2), dtype=mx.float32)
    sanitized = wrapper.sanitize(probe)
    normalized = {
        key: value
        for key, value in sanitized.items()
        if key.startswith("language_model.mtp.")
    }
    if not normalized and raw_mtp_weights:
        raise ValueError("Native Qwen MTP sanitizer removed the configured subtree")
    return normalized


def _is_native_qwen_model_type(model_type: str) -> bool:
    return model_type in {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }


def _is_gemma_text_model_type(model_type: str) -> bool:
    """Match the Gemma 4 text-model family, including unified variants."""

    return model_type.startswith("gemma4")


def _quantize_extracted_model(
    model: Any,
    *,
    config: dict[str, Any],
    text_config: dict[str, Any],
    weight_names: set[str],
) -> None:
    """Mirror mlx-lm's per-leaf quantization decision before strict loading."""

    quantization = text_config.get("quantization", config.get("quantization"))
    if quantization is None:
        return

    per_layer_overrides = {
        key: value for key, value in quantization.items() if isinstance(value, dict)
    }

    def _class_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if f"{path}.scales" not in weight_names:
            return False
        if path in quantization:
            return quantization[path]
        for key, override in per_layer_overrides.items():
            if key.endswith("." + path) or path.endswith("." + key):
                return override
        return True

    nn.quantize(
        model,
        group_size=quantization.get("group_size", 64),
        bits=quantization.get("bits", 8),
        mode=quantization.get("mode", "affine"),
        class_predicate=_class_predicate,
    )


def _realize_model_arrays(model: Any) -> None:
    """Realize parameters and private arrays before crossing MLX threads."""

    if hasattr(model, "modules"):
        mx.eval(
            [
                value
                for module in model.modules()
                for value in module.values()
                if isinstance(value, mx.array)
            ]
        )


def _build_qwen_text_model(
    vlm_language_model: Any,
    model_path: Path,
    config: dict[str, Any],
    text_config: dict[str, Any],
) -> Any:
    """Build Qwen through the official outer sanitize/load handshake.

    The wrapper is temporary but essential: its sanitizer maps VLM checkpoint
    namespaces to ``language_model.*`` and delegates the one-shot MTP key/array
    identity handshake to the native TextModel.  Loading the backbone and head
    separately, or loading a bare TextModel, can never activate the capability.
    """

    model_type = text_config.get("model_type", "")
    Model, ModelArgs = _import_qwen_model_classes(model_type)
    wrapper = Model(ModelArgs.from_dict(config))

    raw_weights = {
        f"language_model.{name}": value
        for name, value in mlx.utils.tree_flatten(vlm_language_model.parameters())
    }
    raw_mtp_weights = _load_mtp_weights(model_path)
    configured_layers = int(text_config.get("mtp_num_hidden_layers", 0) or 0)
    if configured_layers and not raw_mtp_weights:
        raise ValueError(
            "Native Qwen MTP is configured but its indexed weight subtree is missing"
        )
    mtp_weights = raw_mtp_weights
    if configured_layers:
        mtp_weights = _sanitize_raw_qwen_mtp(
            wrapper,
            raw_mtp_weights,
            shift_norm_weights=_qwen_checkpoint_requires_norm_shift(model_path),
        )
    duplicate_keys = raw_weights.keys() & mtp_weights.keys()
    if duplicate_keys:
        raise ValueError(
            "Native Qwen extraction contains duplicate checkpoint keys: "
            + ", ".join(sorted(duplicate_keys)[:3])
        )
    raw_weights.update(mtp_weights)

    sanitized = wrapper.sanitize(raw_weights)
    sanitized_items = list(sanitized.items())
    _quantize_extracted_model(
        wrapper,
        config=config,
        text_config=text_config,
        weight_names=set(sanitized),
    )
    # One strict load is part of the capability contract.  The exact arrays
    # returned by sanitize must reach this call; copying or separate partial
    # loads invalidate the native one-shot handshake.
    wrapper.load_weights(sanitized_items, strict=True)

    text_model = wrapper.language_model
    capability = getattr(text_model, "mtp_capability", None)
    if configured_layers and not (
        capability is not None and capability.supported is True
    ):
        reason = (
            "native_mtp_capability_missing" if capability is None else capability.reason
        )
        raise ValueError(f"Native Qwen MTP load did not activate capability: {reason}")
    return text_model


def _build_gemma_text_model(
    vlm_language_model: Any,
    config: dict[str, Any],
    text_config: dict[str, Any],
) -> Any:
    """Preserve the independent Gemma extraction behavior."""

    TextModel, TextModelArgs = _import_gemma_text_model_classes()
    text_model = TextModel(TextModelArgs.from_dict(text_config))
    vlm_weights = mlx.utils.tree_flatten(vlm_language_model.parameters())
    _quantize_extracted_model(
        text_model,
        config=config,
        text_config=text_config,
        weight_names={name for name, _ in vlm_weights},
    )
    text_model.load_weights(vlm_weights, strict=False)
    return text_model


def build_text_model(vlm_model: Any, model_path: str | Path) -> Any | None:
    """Build an mlx_lm TextModel from a vlm-loaded model's weights.

    Args:
        vlm_model: The mlx_vlm-loaded model (has .language_model attribute)
        model_path: Path to the model directory (contains config.json + safetensors)

    Returns:
        mlx_lm TextModel with MTP support, or None on failure.
    """
    if vlm_model is None:
        return None

    model_path = Path(model_path) if model_path else None
    if model_path is None or not (model_path / "config.json").exists():
        return None

    model_type = ""
    text_model_cls = None
    try:
        config = json.loads((model_path / "config.json").read_text())
        text_config = config.get("text_config", config)
        model_type = text_config.get("model_type") or config.get("model_type", "")
        vlm_lm = vlm_model.language_model
        if _is_native_qwen_model_type(model_type):
            text_model = _build_qwen_text_model(vlm_lm, model_path, config, text_config)
            capability = text_model.mtp_capability
            logger.info(
                "Built native Qwen TextModel (MTP=%s, reason=%s)",
                capability.supported,
                capability.reason,
            )
        elif _is_gemma_text_model_type(model_type):
            Model, _ = _import_gemma_text_model_classes()
            text_model_cls = f"{Model.__module__}.{Model.__qualname__}"
            text_model = _build_gemma_text_model(vlm_lm, config, text_config)
        else:
            raise ValueError(
                f"Unsupported VLM text extraction model_type={model_type!r}"
            )

        # Put the derived TextModel in eval mode. mlx_lm.load / mlx_vlm.load both
        # eval() their models (this is also what LM Studio's mlx-engine does), but
        # this freshly-constructed TextModel defaults to training=True. Hybrid
        # layers (Qwen3.5/3.6 gated-delta) select their compute path with
        # `use_kernel = not self.training`, so in training mode prefill/decode fall
        # to the slow Python recurrence instead of the Metal kernel.
        text_model.train(False)

        # Realize every array the model holds before it leaves the build
        # thread — including underscore-private module attributes such as
        # RoPE._freqs, which parameters() excludes. MLX lazy graphs are tagged
        # to the stream of the thread that recorded them; a lazy array
        # surviving into generation dies with "There is no Stream(gpu, N) in
        # current thread" the moment a worker on another thread evaluates it
        # (Gemma 4: the scaled-RoPE _freqs of the first full_attention layer).
        _realize_model_arrays(text_model)

        return text_model

    except ImportError as e:
        logger.error("Cannot import mlx_lm native text model support: %s", e)
        return None
    except Exception as e:
        # Name the model_type and the class that was picked. Without them this
        # is a bare TypeError from inside someone else's constructor, and the
        # engine carries on with _text_model=None — a route quietly losing its
        # backend, which reads like a warning rather than the failure it is.
        logger.error(
            "Failed to build TextModel from vlm (model_type=%r, class=%s): %s",
            model_type,
            text_model_cls or "<not selected>",
            e,
            exc_info=True,
        )
        return None


def _load_mtp_weights(model_path: Path) -> dict[str, mx.array]:
    """Load exact indexed MTP entries without rewriting their namespaces."""
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return {}

    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map", {})

    # Find MTP keys and their shard files
    mtp_keys: dict[str, str] = {}
    for key, shard in weight_map.items():
        if ".mtp." in key:
            mtp_keys[key] = shard

    if not mtp_keys:
        return {}

    # Group by shard to minimize I/O
    shards: dict[str, list[str]] = {}
    for key, shard in mtp_keys.items():
        shards.setdefault(shard, []).append(key)

    weights = {}
    for shard_file, keys in shards.items():
        shard_path = model_path / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Native Qwen MTP shard not found: {shard_file}")
        shard_data = mx.load(str(shard_path))
        for key in keys:
            if key not in shard_data:
                raise ValueError(
                    f"Native Qwen MTP index entry missing from shard: {key}"
                )
            weights[key] = shard_data[key]

    return weights
