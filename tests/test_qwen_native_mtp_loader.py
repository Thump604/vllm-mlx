# SPDX-License-Identifier: Apache-2.0
"""Synthetic contract tests for native Qwen3.5/3.6 loader integration."""

from __future__ import annotations

import json
import importlib
import sys
import struct
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import mlx.core as mx
from mlx.utils import tree_flatten

import vllm_mlx.text_model_from_vlm as extraction
import vllm_mlx.utils.tokenizer as tokenizer_utils


@dataclass(frozen=True)
class _Capability:
    supported: bool
    reason: str
    num_layers: int


class _FakeTextModel:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._loaded = False
        self.training = True

    @property
    def mtp_capability(self):
        if not self.num_layers:
            return _Capability(False, "native_mtp_head_not_configured", 0)
        if not self._loaded:
            return _Capability(False, "native_mtp_weights_not_loaded", self.num_layers)
        return _Capability(True, "supported", self.num_layers)

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, tokens):
        return ("ordinary-forward", tokens)


class _FakeQwenWrapper:
    instances = []
    sanitize_error = None

    def __init__(self, args):
        self.args = args
        self.language_model = _FakeTextModel(args["mtp_num_hidden_layers"])
        self.sanitize_calls = []
        self.load_calls = []
        self.events = []
        self.__class__.instances.append(self)

    def sanitize(self, weights):
        self.events.append("sanitize")
        self.sanitize_calls.append(weights)
        if self.__class__.sanitize_error is not None:
            raise self.__class__.sanitize_error
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.language_model.mtp."):
                key = key.replace("model.language_model.", "language_model.", 1)
            sanitized[key] = value
        self._handshake = tuple(
            sorted((key, id(value)) for key, value in sanitized.items())
        )
        return sanitized

    def load_weights(self, weights, strict=True):
        self.events.append("load")
        assert isinstance(weights, list)
        assert strict is True
        assert tuple(sorted((key, id(value)) for key, value in weights)) == (
            self._handshake
        )
        self.load_calls.append((weights, strict))
        self.language_model._loaded = bool(self.language_model.num_layers)


class _FakeQwenArgs:
    seen = []

    @classmethod
    def from_dict(cls, config):
        cls.seen.append(config)
        text_config = config["text_config"]
        return {
            "model_type": text_config["model_type"],
            "mtp_num_hidden_layers": text_config.get("mtp_num_hidden_layers", 0),
            "num_experts": text_config.get("num_experts", 0),
        }


class _FakeVLMLanguageModel:
    def __init__(self, weights):
        self._weights = weights

    def parameters(self):
        return self._weights


def _write_config(tmp_path, *, model_type, mtp_layers, num_experts=0, **extra):
    model_path = tmp_path / model_type
    model_path.mkdir()
    text_config = {
        "model_type": model_type,
        "mtp_num_hidden_layers": mtp_layers,
        "num_experts": num_experts,
        **extra,
    }
    config = {"model_type": "qwen3_5", "text_config": text_config}
    (model_path / "config.json").write_text(json.dumps(config))
    return model_path, config


@pytest.fixture
def fake_qwen_loader(monkeypatch):
    _FakeQwenWrapper.instances.clear()
    _FakeQwenWrapper.sanitize_error = None
    _FakeQwenArgs.seen.clear()
    monkeypatch.setattr(
        extraction,
        "_import_qwen_model_classes",
        lambda _model_type: (_FakeQwenWrapper, _FakeQwenArgs),
    )
    monkeypatch.setattr(
        extraction, "_qwen_checkpoint_requires_norm_shift", lambda _path: False
    )
    monkeypatch.setattr(
        extraction.mlx.utils,
        "tree_flatten",
        lambda weights: list(weights.items()),
    )
    monkeypatch.setattr(extraction, "_realize_model_arrays", lambda _model: None)
    monkeypatch.setattr(extraction.nn, "quantize", lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    ("model_type", "num_experts"),
    (("qwen3_5_text", 0), ("qwen3_5_moe_text", 128)),
)
def test_qwen_vlm_extraction_uses_outer_one_shot_handshake(
    tmp_path, monkeypatch, fake_qwen_loader, model_type, num_experts
):
    model_path, config = _write_config(
        tmp_path,
        model_type=model_type,
        mtp_layers=1,
        num_experts=num_experts,
    )
    backbone = object()
    head = object()
    mtp = object()
    vlm = SimpleNamespace(
        language_model=_FakeVLMLanguageModel(
            {"model.embed_tokens.weight": backbone, "lm_head.weight": head}
        )
    )
    monkeypatch.setattr(
        extraction,
        "_load_mtp_weights",
        lambda _path: {"model.language_model.mtp.layers.0.weight": mtp},
    )

    text_model = extraction.build_text_model(vlm, model_path)

    wrapper = _FakeQwenWrapper.instances[-1]
    assert _FakeQwenArgs.seen == [config]
    assert text_model is wrapper.language_model
    assert text_model.mtp_capability.supported is True
    assert text_model.training is False
    assert len(wrapper.sanitize_calls) == 2
    assert len(wrapper.load_calls) == 1
    assert wrapper.events == ["sanitize", "sanitize", "load"]
    loaded = dict(wrapper.load_calls[0][0])
    assert loaded["language_model.model.embed_tokens.weight"] is backbone
    assert loaded["language_model.lm_head.weight"] is head
    assert loaded["language_model.mtp.layers.0.weight"] is mtp
    assert text_model([1, 2]) == ("ordinary-forward", [1, 2])


def test_qwen_configured_mtp_missing_subtree_fails_closed(
    tmp_path, monkeypatch, fake_qwen_loader
):
    model_path, _ = _write_config(tmp_path, model_type="qwen3_5_text", mtp_layers=1)
    vlm = SimpleNamespace(
        language_model=_FakeVLMLanguageModel({"model.weight": object()})
    )
    monkeypatch.setattr(extraction, "_load_mtp_weights", lambda _path: {})

    assert extraction.build_text_model(vlm, model_path) is None
    assert _FakeQwenWrapper.instances[-1].load_calls == []


@pytest.mark.parametrize(
    "error",
    (
        ValueError("missing language_model.mtp.layers.0.weight"),
        ValueError("unexpected language_model.mtp.layers.1.weight"),
        ValueError("incomplete quantized triplet language_model.mtp.fc"),
    ),
)
def test_qwen_malformed_mtp_sanitize_never_partially_loads(
    tmp_path, monkeypatch, fake_qwen_loader, error
):
    model_path, _ = _write_config(
        tmp_path, model_type="qwen3_5_moe_text", mtp_layers=1, num_experts=64
    )
    vlm = SimpleNamespace(
        language_model=_FakeVLMLanguageModel({"model.weight": object()})
    )
    monkeypatch.setattr(
        extraction,
        "_load_mtp_weights",
        lambda _path: {"language_model.mtp.weight": object()},
    )
    _FakeQwenWrapper.sanitize_error = error

    assert extraction.build_text_model(vlm, model_path) is None
    assert _FakeQwenWrapper.instances[-1].load_calls == []


def test_qwen_base_checkpoint_remains_ordinary_without_injected_head(
    tmp_path, monkeypatch, fake_qwen_loader
):
    model_path, _ = _write_config(tmp_path, model_type="qwen3_5_text", mtp_layers=0)
    vlm = SimpleNamespace(
        language_model=_FakeVLMLanguageModel({"model.weight": object()})
    )
    monkeypatch.setattr(extraction, "_load_mtp_weights", lambda _path: {})

    text_model = extraction.build_text_model(vlm, model_path)

    assert text_model.mtp_capability == _Capability(
        False, "native_mtp_head_not_configured", 0
    )
    assert not hasattr(text_model, "mtp")
    assert text_model([7]) == ("ordinary-forward", [7])
    assert len(_FakeQwenWrapper.instances[-1].load_calls) == 1


def test_qwen_quantization_precedes_single_strict_load(
    tmp_path, monkeypatch, fake_qwen_loader
):
    model_path, _ = _write_config(
        tmp_path,
        model_type="qwen3_5_text",
        mtp_layers=1,
        quantization={"group_size": 64, "bits": 4},
    )
    vlm = SimpleNamespace(
        language_model=_FakeVLMLanguageModel({"model.weight": object()})
    )
    monkeypatch.setattr(
        extraction,
        "_load_mtp_weights",
        lambda _path: {
            "language_model.mtp.fc.weight": object(),
            "language_model.mtp.fc.scales": object(),
            "language_model.mtp.fc.biases": object(),
        },
    )

    def fake_quantize(model, **kwargs):
        model.events.append("quantize")
        quantizable = SimpleNamespace(to_quantized=lambda: None)
        assert kwargs["group_size"] == 64
        assert kwargs["bits"] == 4
        assert kwargs["class_predicate"]("language_model.mtp.fc", quantizable) is True

    monkeypatch.setattr(extraction.nn, "quantize", fake_quantize)

    assert extraction.build_text_model(vlm, model_path) is not None
    wrapper = _FakeQwenWrapper.instances[-1]
    assert wrapper.events == ["sanitize", "sanitize", "quantize", "load"]
    assert len(wrapper.load_calls) == 1


def test_qwen_quantization_failure_never_strict_loads(
    tmp_path, monkeypatch, fake_qwen_loader
):
    model_path, _ = _write_config(
        tmp_path,
        model_type="qwen3_5_text",
        mtp_layers=1,
        quantization={"group_size": 64, "bits": 4},
    )
    vlm = SimpleNamespace(
        language_model=_FakeVLMLanguageModel({"model.weight": object()})
    )
    monkeypatch.setattr(
        extraction,
        "_load_mtp_weights",
        lambda _path: {"language_model.mtp.weight": object()},
    )
    monkeypatch.setattr(
        extraction.nn,
        "quantize",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad quant tree")),
    )

    assert extraction.build_text_model(vlm, model_path) is None
    assert _FakeQwenWrapper.instances[-1].load_calls == []


def test_indexed_mtp_loader_preserves_exact_keys(tmp_path, monkeypatch):
    index = {
        "weight_map": {
            "model.language_model.mtp.layers.0.weight": "model-mtp.safetensors",
            "language_model.model.embed_tokens.weight": "model-base.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "model-mtp.safetensors").touch()
    value = object()
    monkeypatch.setattr(
        extraction.mx,
        "load",
        lambda _path: {"model.language_model.mtp.layers.0.weight": value},
    )

    weights = extraction._load_mtp_weights(tmp_path)

    assert weights == {"model.language_model.mtp.layers.0.weight": value}


def test_indexed_mtp_loader_rejects_missing_shard_entry(tmp_path, monkeypatch):
    key = "language_model.mtp.layers.0.weight"
    index = {"weight_map": {key: "model-mtp.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "model-mtp.safetensors").touch()
    monkeypatch.setattr(extraction.mx, "load", lambda _path: {})

    with pytest.raises(ValueError, match="index entry missing"):
        extraction._load_mtp_weights(tmp_path)


def _write_header_only_safetensors(path, entries):
    header = json.dumps(entries, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_qwen_checkpoint_sanitation_convention_comes_from_raw_headers(tmp_path):
    raw_key = "model.language_model.layers.0.linear_attn.conv1d.weight"
    index = {"weight_map": {raw_key: "model-1.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    _write_header_only_safetensors(
        tmp_path / "model-1.safetensors",
        {raw_key: {"dtype": "F32", "shape": [16, 1, 3], "data_offsets": [0, 0]}},
    )

    assert extraction._qwen_checkpoint_requires_norm_shift(tmp_path) is True


def test_qwen_checkpoint_rejects_mixed_conv_conventions(tmp_path):
    raw = "model.language_model.layers.0.linear_attn.conv1d.weight"
    native = "model.language_model.layers.1.linear_attn.conv1d.weight"
    index = {"weight_map": {raw: "model-1.safetensors", native: "model-2.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    _write_header_only_safetensors(
        tmp_path / "model-1.safetensors",
        {raw: {"dtype": "F32", "shape": [16, 1, 3], "data_offsets": [0, 0]}},
    )
    _write_header_only_safetensors(
        tmp_path / "model-2.safetensors",
        {native: {"dtype": "F32", "shape": [16, 3, 1], "data_offsets": [0, 0]}},
    )

    with pytest.raises(ValueError, match="Mixed native Qwen conv1d"):
        extraction._qwen_checkpoint_requires_norm_shift(tmp_path)


def test_legacy_injection_is_explicitly_qwen3_next_only(monkeypatch):
    calls = []
    module = types.ModuleType("vllm_mlx.patches.qwen3_next_mtp")
    module.inject_mtp_support = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    tokenizer_utils._try_inject_mtp(
        object(), "/model", {"model_type": "qwen3_5", "num_nextn_predict_layers": 1}
    )
    tokenizer_utils._try_inject_mtp(
        object(), "/model", {"model_type": "other", "num_nextn_predict_layers": 1}
    )
    tokenizer_utils._try_inject_mtp(
        object(),
        "/model",
        {"model_type": "qwen3_next", "num_nextn_predict_layers": 1},
    )

    assert len(calls) == 1


def test_legacy_post_load_injection_is_explicitly_qwen3_next_only(
    tmp_path, monkeypatch
):
    calls = []
    module = types.ModuleType("vllm_mlx.patches.qwen3_next_mtp")
    module.inject_mtp_support = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        tokenizer_utils, "_try_inject_mtp", lambda *args: calls.append(args)
    )
    monkeypatch.setattr("mlx_lm.utils._download", lambda _name: tmp_path)

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "mtp_num_hidden_layers": 1})
    )
    tokenizer_utils._try_inject_mtp_post_load(object(), "qwen")
    assert calls == []

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_next", "num_nextn_predict_layers": 1})
    )
    (tmp_path / "model-mtp.safetensors").touch()
    tokenizer_utils._try_inject_mtp_post_load(object(), "qwen-next")
    assert len(calls) == 1


_HIDDEN = 32
_HEAD_DIM = 8


def _real_config(*, moe: bool) -> dict:
    text = {
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "hidden_size": _HIDDEN,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 3,
        "full_attention_interval": 2,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-5,
        "head_dim": _HEAD_DIM,
        "rope_theta": 1000.0,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 128,
        "mtp_num_hidden_layers": 1,
    }
    if moe:
        text.update(
            {
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64,
                "moe_intermediate_size": 32,
            }
        )
    return {"model_type": text["model_type"], "text_config": text}


_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
)


def _raw_checkpoint_from_model(model) -> dict[str, mx.array]:
    """Reverse the official raw-conv/RMSNorm transforms for a tiny model."""

    raw = {}
    for key, value in tree_flatten(model.parameters()):
        if "conv1d.weight" in key:
            assert value.shape[-1] == 1
            value = value.moveaxis(1, 2)
        if value.ndim == 1 and any(key.endswith(sfx) for sfx in _NORM_SUFFIXES):
            value = value - 1.0
        raw[key] = value
    return raw


@pytest.mark.parametrize("moe", (False, True))
def test_extracted_mtp_matches_official_full_raw_sanitize_and_load(moe):
    """Numerically prove mixed-state extraction equals canonical sanitation."""

    module = importlib.import_module(
        "mlx_lm.models.qwen3_5_moe" if moe else "mlx_lm.models.qwen3_5"
    )
    config = _real_config(moe=moe)
    args = module.ModelArgs.from_dict(config)
    source = module.Model(args)
    if not hasattr(type(source.language_model), "mtp_capability"):
        pytest.skip("requires mlx-lm native Qwen MTP capability branch")
    source.set_dtype(mx.float32)
    mx.eval(source.parameters())
    raw = _raw_checkpoint_from_model(source)
    assert any(
        "conv1d.weight" in key and value.shape[-1] != 1 for key, value in raw.items()
    )

    reference = module.Model(args)
    canonical = reference.sanitize(dict(raw))
    reference.load_weights(list(canonical.items()), strict=True)
    assert reference.language_model.mtp_capability.supported is True

    extracted = module.Model(args)
    raw_mtp = {key: value for key, value in raw.items() if ".mtp." in key}
    live_backbone = {
        key: value for key, value in canonical.items() if ".mtp." not in key
    }
    normalized_mtp = extraction._sanitize_raw_qwen_mtp(
        extracted, raw_mtp, shift_norm_weights=True
    )
    final = extracted.sanitize({**live_backbone, **normalized_mtp})
    extracted.load_weights(list(final.items()), strict=True)
    assert extracted.language_model.mtp_capability.supported is True

    assert set(final) == set(canonical)
    for key in canonical:
        assert mx.allclose(final[key], canonical[key], rtol=0, atol=0).item()
    reference_params = dict(tree_flatten(reference.parameters()))
    extracted_params = dict(tree_flatten(extracted.parameters()))
    assert set(extracted_params) == set(reference_params)
    for key in reference_params:
        assert mx.allclose(
            extracted_params[key], reference_params[key], rtol=0, atol=0
        ).item()

    mtp_norm = "language_model.mtp.norm.weight"
    assert mx.allclose(canonical[mtp_norm], raw[mtp_norm] + 1.0).item()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "missing"),
        ("unexpected", "unexpected"),
        ("partial_quantized", "incomplete quantized triplet"),
    ),
)
def test_official_qwen_sanitizer_rejects_malformed_mtp(mutation, match):
    module = importlib.import_module("mlx_lm.models.qwen3_5")
    if not hasattr(module.TextModel, "mtp_capability"):
        pytest.skip("requires mlx-lm native Qwen MTP capability branch")
    config = _real_config(moe=False)
    wrapper = module.Model(module.ModelArgs.from_dict(config))
    wrapper.set_dtype(mx.float32)
    mx.eval(wrapper.parameters())
    raw = _raw_checkpoint_from_model(wrapper)
    raw_mtp = {key: value for key, value in raw.items() if ".mtp." in key}
    if mutation == "missing":
        raw_mtp.pop("language_model.mtp.norm.weight")
    elif mutation == "unexpected":
        raw_mtp["language_model.mtp.unexpected.weight"] = mx.zeros((1,))
    else:
        raw_mtp["language_model.mtp.fc.scales"] = mx.zeros((1,))

    with pytest.raises(ValueError, match=match):
        extraction._sanitize_raw_qwen_mtp(wrapper, raw_mtp, shift_norm_weights=True)
    assert wrapper.language_model.mtp_capability.supported is False
