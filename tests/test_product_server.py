# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as server
from vllm_mlx.lifecycle import ModelSpec, ResidencyManager, bind_model_spec_to_profile
from vllm_mlx.model_profile import compute_subject_digest


class FakeEngine:
    async def start(self):
        return None

    async def stop(self):
        return None


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog_and_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for name, value in {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "chat_template.jinja": "{{ messages }}",
        "generation_config.json": "{}",
        "model-00001-of-00001.safetensors": "weights",
        "model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00001.safetensors"
                }
            }
        ),
    }.items():
        (artifact / name).write_text(value)
    root = Path(__file__).parents[1]
    profile = json.loads(
        (root / "schemas/examples/model-profile-v1.example.json").read_text()
    )
    profile["profile_id"] = "managed-model"
    profile["profile_revision"] = 1
    profile["identity"]["artifact_id"] = "managed-model"
    profile["identity"]["served_model_name"] = "managed-model"
    profile["artifact"]["quantization"]["source"] = "conversion_manifest"
    profile["artifact"]["hashes"] = {
        "config_sha256": _sha(artifact / "config.json"),
        "tokenizer_sha256": _sha(artifact / "tokenizer.json"),
        "chat_template_sha256": _sha(artifact / "chat_template.jinja"),
        "generation_config_sha256": _sha(artifact / "generation_config.json"),
        "weights_manifest_sha256": hashlib.sha256(
            (
                f"{_sha(artifact / 'model-00001-of-00001.safetensors')}"
                "  model-00001-of-00001.safetensors\n"
            ).encode()
        ).hexdigest(),
    }
    profile["subject_digest"] = compute_subject_digest(profile)
    profile["qualification"] = {
        "status": "qualified",
        "reason": "test fixture",
        "evidence": [
            {
                "evidence_id": "product-server-fixture",
                "kind": "test",
                "location": "tests/test_product_server.py",
                "artifact_sha256": "a" * 64,
                "result": "pass",
                "hardware_fingerprint": "test-host",
                "workload_id": "product-server",
                "subject_digest": profile["subject_digest"],
                "created_at": "2026-07-21T00:00:00Z",
            }
        ],
    }
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "managed.json").write_text(json.dumps(profile))
    return catalog, artifact, profile


def _args(tmp_path, catalog, artifact):
    return SimpleNamespace(
        product_state_dir=str(tmp_path / "state"),
        product_catalog=str(catalog),
        product_artifact_binding=[f"managed-model={artifact}"],
        product_model_root=str(tmp_path / "managed"),
        auto_unload_idle_seconds=0.0,
        lazy_load_model=True,
        host="127.0.0.1",
        port=8080,
    )


@pytest.mark.anyio
async def test_product_server_configures_routes_without_loading_a_model(
    tmp_path, monkeypatch
):
    catalog, artifact, _profile = _catalog_and_artifact(tmp_path)
    monkeypatch.delenv("VLLM_MLX_LIFECYCLE_STATE_PATH", raising=False)
    server._configure_managed_product(_args(tmp_path, catalog, artifact))
    try:
        assert server._engine is None
        assert server._model_name is None
        assert server._default_model_key is None
        assert server._product_control_service is not None
        with TestClient(server.app) as client:
            response = client.get("/api/v1/control/catalog")
            assert response.status_code == 200
            assert response.json()["data"][0]["profile_id"] == "managed-model"
            status = client.get("/api/v1/control/status")
            assert status.status_code == 200
            assert status.json()["data"]["state"] == "unloaded"
    finally:
        server._residency_manager = None
        server._product_control_service = None
        server._default_model_key = None
        os.environ.pop("VLLM_MLX_LIFECYCLE_STATE_PATH", None)


def test_product_server_rejects_persisted_profile_downgraded_after_activation(
    tmp_path, monkeypatch
):
    catalog, artifact, profile = _catalog_and_artifact(tmp_path)
    state_path = tmp_path / "state" / "lifecycle.json"
    state_path.parent.mkdir()
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))
    first = ResidencyManager(lambda spec: FakeEngine())
    first.register_model(
        bind_model_spec_to_profile(ModelSpec("default", str(artifact)), profile)
    )
    first._control.close()

    profile["qualification"] = {
        "status": "not_qualified",
        "reason": "evidence withdrawn",
        "evidence": [],
    }
    (catalog / "managed.json").write_text(json.dumps(profile))

    with pytest.raises(RuntimeError, match="no longer qualified"):
        server._configure_managed_product(_args(tmp_path, catalog, artifact))

    # Restore validation happens before the persistent lifecycle lock is acquired.
    second = ResidencyManager(lambda spec: FakeEngine())
    second._control.close()
    os.environ.pop("VLLM_MLX_LIFECYCLE_STATE_PATH", None)


def test_clear_managed_product_profile_restores_pre_profile_defaults(monkeypatch):
    baseline = {
        "max_tokens": 101,
        "max_request_tokens": 202,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 17,
        "min_p": 0.1,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_token_budget": 303,
    }
    monkeypatch.setattr(server, "_managed_product_base_defaults", baseline)
    monkeypatch.setattr(server, "_model_name", "stale-model")
    monkeypatch.setattr(server, "_model_path", "/stale/model")
    monkeypatch.setattr(server, "_default_model_key", "default")
    monkeypatch.setattr(server, "_default_max_tokens", 999)
    monkeypatch.setattr(server, "_max_request_tokens", 999)
    monkeypatch.setattr(server, "_default_temperature", 1.0)
    monkeypatch.setattr(server, "_default_top_p", 1.0)
    monkeypatch.setattr(server, "_default_top_k", 99)
    monkeypatch.setattr(server, "_default_min_p", 0.9)
    monkeypatch.setattr(server, "_default_presence_penalty", 0.9)
    monkeypatch.setattr(server, "_default_repetition_penalty", 1.9)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {"stale": True})
    monkeypatch.setattr(server, "_default_thinking_token_budget", 999)

    server._clear_managed_product_profile()

    assert server._model_name is None
    assert server._model_path is None
    assert server._default_model_key is None
    assert server._default_max_tokens == 101
    assert server._max_request_tokens == 202
    assert server._default_temperature == 0.3
    assert server._default_top_p == 0.8
    assert server._default_top_k == 17
    assert server._default_min_p == 0.1
    assert server._default_presence_penalty == 0.2
    assert server._default_repetition_penalty == 1.1
    assert server._default_chat_template_kwargs == {"enable_thinking": False}
    assert server._default_thinking_token_budget == 303


def test_qwen_managed_profile_applies_exact_qualified_request_defaults(monkeypatch):
    root = Path(__file__).parents[1]
    profile = json.loads(
        (root / "catalog/profiles/qwen3.6-35b-a3b-8bit-v1.json").read_text()
    )
    spec = ModelSpec(
        "default",
        "/models/qwen",
        max_tokens=profile["serving"]["limits"]["max_output_tokens"],
        force_mllm=True,
    )
    monkeypatch.setattr(server, "_tool_parser_instance", None)

    server._apply_managed_product_profile(profile, spec)

    assert server._resolve_temperature(None) == 0.7
    assert server._resolve_top_p(None) == 0.8
    assert server._resolve_top_k(None) == 20
    assert server._resolve_min_p(None) == 0.0
    assert server._resolve_presence_penalty(None) == 1.5
    assert server._resolve_repetition_penalty(None) == 1.0
    assert server._resolve_chat_template_kwargs(None) == {"enable_thinking": False}


@pytest.mark.anyio
async def test_product_server_restores_exact_persisted_profile_configuration(
    tmp_path, monkeypatch
):
    catalog, artifact, profile = _catalog_and_artifact(tmp_path)
    state_path = tmp_path / "state" / "lifecycle.json"
    state_path.parent.mkdir()
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))
    first = ResidencyManager(lambda spec: FakeEngine())
    spec = bind_model_spec_to_profile(ModelSpec("default", str(artifact)), profile)
    first.register_model(spec)
    first._control.close()

    server._configure_managed_product(_args(tmp_path, catalog, artifact))
    try:
        assert server._model_name == "managed-model"
        assert server._model_path == str(artifact)
        assert server._default_model_key == "default"
        assert server._product_control_service.status()["active_profile"] == {
            "profile_id": "managed-model",
            "profile_revision": 1,
            "subject_digest": profile["subject_digest"],
        }
        assert server._product_control_service.status()["state"] == "unloaded"
    finally:
        await server._residency_manager.shutdown()
        server._residency_manager = None
        server._product_control_service = None
        os.environ.pop("VLLM_MLX_LIFECYCLE_STATE_PATH", None)
