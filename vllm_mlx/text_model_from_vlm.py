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
    """Return per-module quantization params for an extracted text layer."""
    if f"{path}.scales" not in all_weight_names:
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

        # Architecture gate: only Qwen 3.5 family configs map cleanly into
        # mlx_lm.models.qwen3_5.TextModelArgs. Bail out before the import for
        # everything else so we don't rely on the except-Exception backstop
        # below to swallow a ZeroDivisionError on every restart.
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

        all_weight_names = set(name for name, _ in vlm_weights)
        all_weight_names.update(name for name, _ in mtp_weights)

        # Quantize the TextModel skeleton to match source weights.
        # Use config.json quantization metadata if available; otherwise infer
        # from the presence of .scales keys in the weight names.
        quantization = text_config.get("quantization", config.get("quantization", None))
        logger.info(
            "build_text_model: quantization config=%s, "
            "mtp_scales_present=%s, vlm_scales_present=%s",
            quantization,
            any(name.endswith(".scales") for name, _ in mtp_weights),
            any(name.endswith(".scales") for name, _ in vlm_weights),
        )
        if quantization is None:
            # Infer: if any weight has .scales, the model is quantized.
            # Use safe defaults matching the common 8-bit affine recipe.
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

        # Load MTP weights from safetensors
        if mtp_weights:
            text_model.load_weights(mtp_weights, strict=False)
            logger.info("Loaded %d MTP weights from safetensors", len(mtp_weights))

            # Fix quantization mismatch: some model quants (e.g. Qwen 3.6
            # 8-bit) quantize MTP Linear layers on disk but nn.quantize()
            # may miss them during skeleton setup. Detect and fix: if any
            # Linear module got dict-valued weights, rebuild with quantization.
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
                        "Returning None so MLLM scheduler handles all "
                        "generation for this model.",
                        len(_broken_paths),
                        _broken_paths[:5],
                    )
                    # Can't use TextModel+MTP when quantized weights don't
                    # match the skeleton. Return None so engine falls back
                    # to MLLM scheduler (which handles thinking natively).
                    return None
        else:
            logger.warning("No MTP weights found in %s", model_path.name)

        # Verify MTP is functional
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
