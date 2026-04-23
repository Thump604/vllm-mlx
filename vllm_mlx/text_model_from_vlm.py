# SPDX-License-Identifier: Apache-2.0
"""Construct an mlx_lm TextModel from mlx_vlm-loaded model weights.

When mlx_vlm loads a model, it strips MTP weights in sanitize().
This module builds a parallel mlx_lm TextModel that:
1. Shares backbone + lm_head weights with the vlm model (zero-copy)
2. Loads MTP weights from safetensors on disk
3. Provides full mlx_lm API: return_hidden, n_confirmed, mtp_forward, make_mtp_cache
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

logger = logging.getLogger(__name__)


def _resolve_quantization_recipe(
    path: str,
    quantization: dict[str, Any],
    all_weight_names: set[str],
) -> dict[str, Any] | bool:
    """Return per-module quantization params for an extracted text layer.

    Handles MoE naming mismatch: model tree uses `mlp.shared_expert.up_proj`
    but some quantized models store weights as `mlp.up_proj` (flat naming).
    Check both the exact path and the flattened alternative.
    """
    if f"{path}.scales" in all_weight_names:
        pass
    else:
        alt_path = path.replace(".mlp.shared_expert.", ".mlp.")
        if alt_path != path and f"{alt_path}.scales" in all_weight_names:
            pass
        else:
            return False

    recipe = {
        "group_size": quantization.get("group_size", 64),
        "bits": quantization.get("bits", 8),
        "mode": quantization.get("mode", "affine"),
    }

    for key in (path, f"language_model.{path}"):
        override = quantization.get(key)
        if isinstance(override, dict):
            recipe.update(
                {
                    k: override[k]
                    for k in ("group_size", "bits", "mode")
                    if k in override
                }
            )
            break

    return recipe


def _remap_moe_mtp_weights(
    mtp_weights: list[tuple[str, Any]],
    text_model: Any,
    logger,
) -> list[tuple[str, Any]]:
    """Remap MTP weight keys to match the model tree for MoE models."""
    model_param_names = {
        f"mtp.{n}" for n, _ in mlx.utils.tree_flatten(text_model.mtp.parameters())
    }
    disk_names = {n for n, _ in mtp_weights}

    disk_base = {
        n
        for n in disk_names
        if n.endswith(".weight") and not n.endswith(".weight.weight")
    }
    if disk_base.issubset(
        model_param_names | {n for n in disk_names if ".scales" in n or ".biases" in n}
    ):
        disk_weight_keys = {
            n
            for n in disk_names
            if not n.endswith(".scales") and not n.endswith(".biases")
        }
        if disk_weight_keys.issubset(model_param_names):
            logger.info(
                "_remap_moe_mtp_weights: keys match model tree, no remap needed"
            )
            return mtp_weights

    logger.info(
        "_remap_moe_mtp_weights: disk format differs from model tree, applying remap"
    )
    remapped = []
    remap_count = 0

    for name, val in mtp_weights:
        new_name = name

        if ".weight.weight" in new_name:
            new_name = new_name.replace(".weight.weight", ".weight")
        elif ".weight.scales" in new_name:
            new_name = new_name.replace(".weight.scales", ".scales")
        elif ".weight.biases" in new_name:
            new_name = new_name.replace(".weight.biases", ".biases")

        if ".mlp.experts.down_proj" in new_name:
            new_name = new_name.replace(
                ".mlp.experts.down_proj", ".mlp.switch_mlp.down_proj"
            )

        if ".mlp.experts.gate_up_proj" in new_name:
            suffix = new_name.split(".mlp.experts.gate_up_proj")[-1]
            prefix = new_name.split(".mlp.experts.gate_up_proj")[0]
            split_dim = 1 if val.ndim == 3 else 0
            half = val.shape[split_dim] // 2
            gate_val = mx.take(val, mx.arange(half), axis=split_dim)
            up_val = mx.take(val, mx.arange(half, val.shape[split_dim]), axis=split_dim)
            remapped.append((f"{prefix}.mlp.switch_mlp.gate_proj{suffix}", gate_val))
            remapped.append((f"{prefix}.mlp.switch_mlp.up_proj{suffix}", up_val))
            remap_count += 1
            continue

        if new_name != name:
            remap_count += 1
        remapped.append((new_name, val))

    logger.info("_remap_moe_mtp_weights: remapped %d keys", remap_count)
    return remapped


def build_text_model(vlm_model: Any, model_path: str | Path) -> Any | None:
    """Build an mlx_lm TextModel from a vlm-loaded model's weights.

    Args:
        vlm_model: The mlx_vlm-loaded model (has .language_model attribute)
        model_path: Path to the model directory (contains config.json + safetensors)

    Returns:
        mlx_lm TextModel with MTP support, or None on failure or unsupported family.

    Notes:
        Only supports the Qwen 3.5 family (qwen3_5, qwen3_5_moe and their *_text
        variants). Other MLLM families (Gemma 4, Nemotron H) return None without
        attempting to import qwen3_5 — feeding their config.json into the Qwen
        TextModelArgs schema would crash inside Qwen3NextMLP.__init__ (e.g.
        ZeroDivisionError on Gemma 4 because shared_expert_intermediate_size is
        absent). The matching call sites in engine/simple.py and engine/batched.py
        already treat None as "no text routing"; the MLLM path remains usable.
    """
    if vlm_model is None:
        return None

    model_path = Path(model_path) if model_path else None
    if model_path is None or not (model_path / "config.json").exists():
        return None

    try:
        config = json.loads((model_path / "config.json").read_text())
        text_config = config.get("text_config", config)

        text_model_type = (text_config.get("model_type") or "").lower()
        top_model_type = (config.get("model_type") or "").lower()
        if not (
            text_model_type.startswith("qwen3_5")
            or top_model_type.startswith("qwen3_5")
        ):
            logger.info(
                "build_text_model: skipping model_type=%r "
                "(only Qwen 3.5 family is supported)",
                text_model_type or top_model_type or "<unknown>",
            )
            return None

        num_experts = text_config.get("num_experts", 0) or 0

        # Always import from qwen3_5 — TextModel and TextModelArgs handle both
        # dense and MoE natively (MTPDecoderLayer auto-selects SparseMoeBlock
        # when args.num_experts > 0). qwen3_5_moe.py does NOT export these.
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        # Build args with proper __post_init__ (handles partial_rotary_factor,
        # rope_scaling, head_dim derivation)
        args = TextModelArgs.from_dict(text_config)
        text_model = TextModel(args)

        # Collect all weights first: backbone from vlm + MTP from safetensors
        vlm_lm = vlm_model.language_model
        vlm_weights = mlx.utils.tree_flatten(vlm_lm.parameters())
        mtp_weights = _load_mtp_weights(model_path)

        if num_experts > 0 and mtp_weights:
            mtp_weights = _remap_moe_mtp_weights(mtp_weights, text_model, logger)

        all_weight_names = set(name for name, _ in vlm_weights)
        all_weight_names.update(name for name, _ in mtp_weights)

        # Quantize the TextModel skeleton to match source weights.
        # Use config.json quantization metadata if available; otherwise infer
        # from the presence of .scales keys in the weight names.
        quantization = text_config.get("quantization", config.get("quantization", None))
        mtp_scale_names = sorted(
            name for name, _ in mtp_weights if name.endswith(".scales")
        )
        logger.info(
            "build_text_model: quantization config present=%s, mtp_scales=%s",
            quantization is not None,
            mtp_scale_names[:5],
        )
        if quantization is None:
            if any(name.endswith(".scales") for name, _ in vlm_weights) or any(
                name.endswith(".scales") for name, _ in mtp_weights
            ):
                quantization = {"group_size": 64, "bits": 8, "mode": "affine"}
                logger.info(
                    "Inferred quantization from weight names "
                    "(config.json missing 'quantization' key): %s",
                    quantization,
                )

        if quantization is not None:

            def _class_predicate(path, module):
                if not hasattr(module, "to_quantized"):
                    return False
                return _resolve_quantization_recipe(
                    path, quantization, all_weight_names
                )

            nn.quantize(
                text_model,
                class_predicate=_class_predicate,
            )

        # Transfer backbone + lm_head weights from vlm language_model (zero-copy).
        # strict=False because TextModel has MTP params that vlm doesn't have yet.
        text_model.load_weights(vlm_weights, strict=False)

        logger.info(
            "Transferred %d weight arrays from vlm language_model", len(vlm_weights)
        )

        # Load MTP weights from safetensors.
        if mtp_weights:
            _has_moe_mtp = (
                hasattr(text_model, "mtp")
                and text_model.mtp is not None
                and any(
                    hasattr(layer, "mlp") and hasattr(layer.mlp, "shared_expert")
                    for layer in getattr(text_model.mtp, "layers", [])
                )
            )
            _weights_use_flat = any(
                "mtp.layers." in name
                and ".mlp." in name
                and ".shared_expert." not in name
                and ".gate." not in name
                and name.split(".")[-1] in ("weight", "scales", "biases")
                and any(sub in name for sub in ("gate_proj", "up_proj", "down_proj"))
                for name, _ in mtp_weights
            )
            if _has_moe_mtp and _weights_use_flat:
                _remapped = []
                for name, val in mtp_weights:
                    if (
                        "mtp.layers." in name
                        and ".mlp." in name
                        and ".shared_expert." not in name
                    ):
                        new_name = name.replace(".mlp.", ".mlp.shared_expert.", 1)
                        _remapped.append((new_name, val))
                    else:
                        _remapped.append((name, val))
                logger.info(
                    "Remapped %d MTP weights from flat to MoE naming",
                    sum(1 for a, b in zip(mtp_weights, _remapped) if a[0] != b[0]),
                )
                mtp_weights = _remapped
                all_weight_names = set(name for name, _ in vlm_weights)
                all_weight_names.update(name for name, _ in mtp_weights)

            text_model.load_weights(mtp_weights, strict=False)
            logger.info("Loaded %d MTP weights from safetensors", len(mtp_weights))

            if hasattr(text_model, "mtp") and text_model.mtp is not None:
                _broken_paths = []
                for path, module in text_model.mtp.named_modules():
                    if not isinstance(module, nn.Linear):
                        continue
                    if isinstance(module, nn.QuantizedLinear):
                        continue
                    w = getattr(module, "weight", None)
                    if isinstance(w, dict):
                        full_path = f"mtp.{path}" if path else "mtp"
                        _broken_paths.append(full_path)

                if _broken_paths:
                    logger.warning(
                        "MTP quantization mismatch in %d modules: %s. "
                        "Returning None so MLLM scheduler handles all generation for this model.",
                        len(_broken_paths),
                        _broken_paths[:5],
                    )
                    return None
        else:
            logger.warning("No MTP weights found in %s", model_path.name)

        # Inject MTP if TextModel doesn't have native MTP support.
        # mlx_lm's qwen3_5.TextModel strips MTP weights in sanitize(),
        # so we inject MTP module + methods at runtime.
        if not hasattr(text_model, "mtp") or text_model.mtp is None:
            num_mtp = text_config.get("mtp_num_hidden_layers", 0)
            if num_mtp == 0:
                num_mtp = text_config.get("num_nextn_predict_layers", 0)
            if num_mtp > 0:
                from .patches.qwen3_5_mtp import inject_mtp_support

                inject_mtp_support(text_model, model_path, config)

        if hasattr(text_model, "mtp") and text_model.mtp is not None:
            mx.eval(text_model.mtp.parameters())
            logger.info(
                "TextModel built with MTP support (%d layers)",
                args.mtp_num_hidden_layers,
            )
        else:
            logger.info("TextModel built without MTP (mtp_num_hidden_layers=0)")

        return text_model

    except ImportError as e:
        logger.error("Cannot import mlx_lm TextModel (need PR #990): %s", e)
        return None
    except Exception as e:
        logger.error("Failed to build TextModel from vlm: %s", e)
        return None


def _load_mtp_weights(model_path: Path) -> list[tuple[str, mx.array]]:
    """Load MTP weights from safetensors, stripping the language_model. prefix.

    mlx_vlm's sanitize() strips mtp.* keys during model loading,
    but the weights are still on disk in the safetensors files.
    """
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return []

    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map", {})

    # Find MTP keys and their shard files
    mtp_keys: dict[str, tuple[str, str]] = {}
    for key, shard in weight_map.items():
        if ".mtp." in key:
            # Strip "language_model." prefix to match mlx_lm namespace
            clean = (
                key.replace("language_model.", "", 1)
                if key.startswith("language_model.")
                else key
            )
            mtp_keys[key] = (clean, shard)

    if not mtp_keys:
        for mtp_file in (
            model_path / "mtp" / "weights.safetensors",
            model_path / "model-mtp.safetensors",
        ):
            if not mtp_file.exists():
                continue
            raw = mx.load(str(mtp_file))
            weights = []
            for key, value in raw.items():
                clean = (
                    key.replace("language_model.", "", 1)
                    if key.startswith("language_model.")
                    else key
                )
                if clean.startswith("mtp."):
                    weights.append((clean, value))
            if weights:
                logger.info(
                    "_load_mtp_weights: loaded %d fallback tensors from %s",
                    len(weights),
                    mtp_file.relative_to(model_path),
                )
                return weights
        return []

    # Group by shard to minimize I/O
    shards: dict[str, list[tuple[str, str]]] = {}
    for orig, (clean, shard) in mtp_keys.items():
        shards.setdefault(shard, []).append((orig, clean))

    weights = []
    for shard_file, key_pairs in shards.items():
        shard_path = model_path / shard_file
        if not shard_path.exists():
            logger.warning("MTP shard not found: %s", shard_file)
            continue
        shard_data = mx.load(str(shard_path))
        for orig, clean in key_pairs:
            if orig in shard_data:
                weights.append((clean, shard_data[orig]))

    return weights
