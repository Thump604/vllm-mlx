# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gemma 4 LiteRT MTP converter."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from vllm_mlx.patches.gemma4_litert_mtp_convert import convert_extracted_tensors


def _write_tensor(
    root: Path,
    slug: str,
    array: np.ndarray,
    *,
    scales: np.ndarray | None = None,
    zero_points: np.ndarray | None = None,
) -> str:
    array_path = root / f"{slug}.npy"
    np.save(array_path, array, allow_pickle=False)
    if scales is not None:
        np.save(root / f"{slug}.scales.npy", scales, allow_pickle=False)
    if zero_points is not None:
        np.save(root / f"{slug}.zero_points.npy", zero_points, allow_pickle=False)
    return str(array_path)


def test_convert_extracted_tensors(tmp_path: Path):
    extract_dir = tmp_path / "extract"
    output_dir = tmp_path / "output"
    extract_dir.mkdir()

    pre_project_path = _write_tensor(
        extract_dir,
        "mtp_pre_project",
        np.full((4, 64), 3, dtype=np.int8),
        scales=np.full((4,), 0.25, dtype=np.float32),
        zero_points=np.zeros((4,), dtype=np.int64),
    )
    div_path = _write_tensor(
        extract_dir,
        "decode_div",
        np.array(30.0, dtype=np.float32),
    )
    q_proj_path = _write_tensor(
        extract_dir,
        "layer0_q_proj",
        np.full((8, 64), -2, dtype=np.int8),
        scales=np.full((8,), 0.5, dtype=np.float32),
        zero_points=np.zeros((8,), dtype=np.int64),
    )

    extract_summary = {
        "tensors": [
            {
                "name": "MtpDrafterModel.mtp_pre_project/mtp_pre_proj/btm,md->btd/dot_general",
                "shape": [4, 64],
                "type": "INT8",
                "array_path": pre_project_path,
                "quantization": {
                    "quantized_dimension": 0,
                    "scale_count": 4,
                    "zero_point_count": 4,
                },
            },
            {
                "name": "MtpDrafterModel.decode_softmax/transformer.decode_softmax/transformer._post_process_decoding/div",
                "shape": [],
                "type": "FLOAT32",
                "array_path": div_path,
                "quantization": None,
            },
            {
                "name": "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/q_einsum/reshape;layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/q_einsum/btd,dH->btH/dot_general",
                "shape": [8, 64],
                "type": "INT8",
                "array_path": q_proj_path,
                "quantization": {
                    "quantized_dimension": 0,
                    "scale_count": 8,
                    "zero_point_count": 8,
                },
            },
        ]
    }
    extract_summary_path = tmp_path / "extract-summary.json"
    extract_summary_path.write_text(json.dumps(extract_summary) + "\n")

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"model_type": "gemma4_text"}) + "\n")

    payload = convert_extracted_tensors(
        extract_summary_path,
        contract_path=contract_path,
        output_dir=output_dir,
        target_bits=5,
        group_size=64,
    )

    assert payload["converted_tensor_count"] == 3
    assert payload["ignored_tensor_count"] == 0

    shard = mx.load(str(output_dir / "model-mtp-litert.safetensors"))
    assert "mtp_drafter.mtp_pre_project.weight" in shard
    assert "mtp_drafter.mtp_pre_project.scales" in shard
    assert "mtp_drafter.mtp_pre_project.biases" in shard
    assert "mtp_drafter.decode_softmax.div" in shard
    assert "mtp_drafter.layers.0.attn.q_proj.weight" in shard
    assert shard["mtp_drafter.decode_softmax.div"].dtype == mx.bfloat16

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["adapter_payload_format"] == "gemma4_litert_mtp_v1"
    assert summary["saved_key_count"] == len(shard)
