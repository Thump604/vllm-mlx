# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.laguna.config import ModelConfig
from mlx_vlm.models.laguna.language import LanguageModel

FIXTURE = Path(__file__).parent / "fixtures/model_configs/laguna-s-2-1.json"
ARTIFACT_FIXTURE = (
    Path(__file__).parent / "fixtures/model_fit/laguna-s-2-1-artifact.json"
)


def _config():
    return json.loads(FIXTURE.read_text())["config"]


def test_installed_mlx_vlm_accepts_exact_laguna_structure():
    config = ModelConfig.from_dict(_config())

    assert config.model_type == "laguna"
    assert config.num_hidden_layers == 48
    assert config.num_experts == 256
    assert config.num_experts_per_tok == 10
    assert config.sliding_window == 512
    assert config.layer_types.count("sliding_attention") == 36
    assert config.layer_types.count("full_attention") == 12


def test_laguna_mixed_attention_builds_matching_cache_types():
    config = ModelConfig.from_dict(_config())
    layers = [SimpleNamespace(attention_type=kind) for kind in config.layer_types]
    cache = LanguageModel.make_cache(SimpleNamespace(args=config, layers=layers))

    assert len(cache) == 48
    assert sum(isinstance(item, RotatingKVCache) for item in cache) == 36
    assert sum(isinstance(item, KVCache) for item in cache) == 12
    assert all(
        item.max_size == 512 for item in cache if isinstance(item, RotatingKVCache)
    )


def test_laguna_quantization_contract_keeps_router_at_eight_bits():
    quantization = json.loads(ARTIFACT_FIXTURE.read_text())["quantization"]

    assert quantization["default"] == {
        "mode": "affine",
        "bits": 4,
        "group_size": 64,
    }
    assert quantization["router"] == {
        "mode": "affine",
        "bits": 8,
        "group_size": 64,
        "overrides": 47,
    }
